"""
trading_crew.py
----------------
MCP CLIENT — portfolio-level CrewAI trading crew.

Rebuilt from the class starter trading_crew.py / trading_crew_1.py for a
WHOLE-PORTFOLIO run (15 stocks at once, not one ticker per invocation),
with a hedging agent and a daily self-improvement loop layered on top.

Prerequisites
-------------
    pip install -r ../requirements.txt
    export GROQ_API_KEY="your_groq_key"        # free tier: console.groq.com
    export ALPACA_API_KEY="your_OWN_paper_key"  # do NOT reuse the key leaked
    export ALPACA_SECRET_KEY="your_OWN_paper_secret"  # in the class file comment

Usage
-----
    python trading_crew.py --config ../config/aman.yaml

One call to run() = one full cycle (fetch -> analyze -> hedge -> decide ->
execute -> log). scripts/run_session.py wraps this in the 9:30am-4:00pm ET
loop; this file itself runs exactly once per invocation, which also makes
it easy to test on a single cycle without sitting through the whole window.
"""

# ============================================================
# 0.0 Imports
# ============================================================
import argparse
import json
import os
import sys
from pathlib import Path

import yaml
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process, LLM
from crewai_tools import MCPServerAdapter
from mcp import StdioServerParameters

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from logging_utils import save_agent_log, load_previous_decision_report

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

SERVERS_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..",
    "mcp_servers", "stock_trading", "servers",
)


# ============================================================
# 1.0 Config loading
# ============================================================
def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


# ============================================================
# 2.0 Build the crew for one cycle
# ============================================================
def build_crew(cfg: dict, mcp_tools: list, yesterday_report: str | None):
    symbols = [s["symbol"] for s in cfg["portfolio"]["stocks"]]
    capital = cfg["portfolio"]["starting_capital"]
    risk = cfg["risk"]
    hedging = cfg["hedging"]
    data_cfg = cfg["data"]

    llm = LLM(
        model=cfg["llm"]["model"],
        api_key=os.environ.get("GROQ_API_KEY", ""),
        temperature=cfg["llm"].get("temperature", 0.2),
    )

    # ------------------------------------------------------------
    # 2.1 Agents
    # ------------------------------------------------------------
    data_fetcher_agent = Agent(
        role="Portfolio Data Fetcher",
        goal=(
            f"Fetch {data_cfg['lookback_days']} days of OHLCV data for the "
            f"ENTIRE portfolio ({', '.join(symbols)}) plus the SPY benchmark "
            "in a single batched call, so no downstream agent ever makes a "
            "per-symbol data request."
        ),
        backstory=(
            "You are a market-data specialist. You know repeated per-ticker "
            "API calls are slow and risk rate limits, so you always batch."
        ),
        tools=mcp_tools,
        llm=llm,
        verbose=True,
    )

    technical_analyst_agent = Agent(
        role="Portfolio Technical Analyst",
        goal=(
            f"For EVERY symbol in {symbols}, call calculate_indicators and "
            "calculate_beta. Produce one consolidated cross-sectional table "
            "(not 15 separate reports) covering RSI, MACD, Bollinger %b, "
            "ATR, trend, relative volume, and realized beta for each stock, "
            "plus the portfolio's capital-weighted average beta assuming an "
            "equal-weight starting allocation."
        ),
        backstory=(
            "You are a quantitative analyst. You think in cross-sections, "
            "not single-stock stories — your job is to rank the whole book "
            "against itself, not analyze one ticker in isolation."
        ),
        tools=mcp_tools,
        llm=llm,
        verbose=True,
    )

    hedging_agent = Agent(
        role="Hedging and Risk Overlay Agent",
        goal=(
            f"Using the portfolio's weighted average beta from the technical "
            f"analyst, and {hedging['instrument']} option chain data, size a "
            f"tail-hedge ({hedging['option_type']} options) that covers "
            f"roughly {hedging['target_hedge_ratio']*100:.0f}% of the "
            f"portfolio's beta-weighted dollar exposure. Cap total premium "
            f"spend at {hedging['max_hedge_cost_pct_of_portfolio']}% of "
            f"portfolio value. State: underlying, strike, expiry, contract "
            f"count, estimated premium cost, and what this hedge does and "
            f"does NOT protect against (it will not offset single-stock "
            f"idiosyncratic risk, only broad market moves)."
        ),
        backstory=(
            "You are a derivatives risk manager. Your job is capital "
            "preservation during tail events, not return-chasing — you are "
            "deliberately the most conservative voice on this team, same "
            "role a risk desk plays at a real trading firm."
        ),
        tools=mcp_tools,
        llm=llm,
        verbose=True,
    )

    decision_maker_agent = Agent(
        role="Portfolio Decision-Maker",
        goal=(
            f"Given the technical analyst's cross-sectional table and the "
            f"hedging agent's overlay recommendation{' and YESTERDAY-S RESULTS below' if yesterday_report else ''}, "
            f"decide buy/sell/hold AND a dollar amount for EVERY symbol in "
            f"{symbols} simultaneously, allocating against a ${capital:,.0f} "
            f"starting capital base. Hard rules you MUST follow: "
            f"(1) no single position may exceed {risk['max_position_weight_pct']}% "
            f"of portfolio value, "
            f"(2) keep at least {risk['min_cash_buffer_pct']}% of capital as cash, "
            f"(3) state a confidence score 0-1 per decision; if confidence < "
            f"{risk['min_confidence_to_trade']}, the decision MUST be hold, "
            f"(4) do not recommend more than {risk['max_trades_per_symbol_per_day']} "
            f"trade(s) per symbol in this cycle — no flip-flopping a position "
            f"within the same session, "
            f"(5) factor in the hedging agent's overlay cost when sizing "
            f"equity positions so total capital usage (equities + hedge "
            f"premium) never exceeds available capital. "
            f"Your final output MUST end with a fenced JSON code block of "
            f"the exact shape: "
            f'{{"decisions": [{{"symbol": "...", "action": "buy|sell|hold", '
            f'"notional": 0.0, "confidence": 0.0, "reason": "..."}}], '
            f'"hedge": {{"underlying": "...", "option_symbol": "...", '
            f'"contracts": 0, "reason": "..."}}}} '
            f"— this block is machine-parsed to trigger execution, so the "
            f"JSON must be syntactically valid and nothing else may follow it."
        ),
        backstory=(
            "You are the senior portfolio manager who makes the final call. "
            "You weigh the quant signals against the risk desk's hedge "
            "recommendation and, when available, your own team's results "
            "from the prior session — you adjust sizing and conviction based "
            "on what actually worked yesterday, not just today's raw signals."
        ),
        tools=[],
        llm=llm,
        verbose=True,
    )

    execution_agent = Agent(
        role="Portfolio Execution Agent",
        goal=(
            "Parse the decision-maker's final JSON block. Call "
            "place_portfolio_orders ONCE with the full `decisions` list "
            "(skip entries where action is 'hold' — the tool handles this, "
            "but never call it per-symbol). If `hedge.contracts` > 0 and "
            "hedging is enabled, separately call buy_option_hedge with the "
            "hedge's option_symbol and contracts. Report a clean execution "
            "summary: how many buys/sells/holds, and the hedge order result."
        ),
        backstory=(
            "You are a disciplined execution specialist. You execute "
            "exactly what the decision-maker approved — no more, no less — "
            "and you always submit the equity batch as a single call."
        ),
        tools=mcp_tools,
        llm=llm,
        verbose=True,
    )

    # ------------------------------------------------------------
    # 2.2 Tasks
    # ------------------------------------------------------------
    task_fetch = Task(
        description=(
            f"Call fetch_ohlcv_batch ONCE with symbols={symbols}, "
            f"lookback_days={data_cfg['lookback_days']}, "
            f"timeframe='{data_cfg['timeframe']}', include_benchmark=True. "
            "Confirm every symbol was cached successfully; flag any that "
            "came back missing."
        ),
        expected_output="Per-symbol bar counts and date range; list of any missing symbols.",
        agent=data_fetcher_agent,
    )

    task_analyze = Task(
        description=(
            f"For every symbol in {symbols}, call calculate_indicators and "
            f"calculate_beta (window_days={data_cfg['beta_window_days']}). "
            "Build one markdown table with columns: Symbol, Close, RSI, "
            "MACD Crossover, Bollinger %b, Trend, Relative Volume, Beta, "
            "Beta Class. Add a closing line stating the portfolio's "
            "capital-weighted average beta (equal-weight assumption)."
        ),
        expected_output="One consolidated cross-sectional markdown table plus the weighted average beta.",
        agent=technical_analyst_agent,
        context=[task_fetch],
    )

    task_hedge = Task(
        description=(
            f"Using the technical analyst's weighted average beta, call "
            f"get_option_chain for underlying='{hedging['instrument']}', "
            f"option_type='{hedging['option_type']}', "
            f"max_expiration_days={hedging['max_expiration_days']}. "
            "Pick a near-the-money strike and a short-dated expiry inside "
            "that window, size the contract count to cover roughly "
            f"{hedging['target_hedge_ratio']*100:.0f}% of the portfolio's "
            f"beta-weighted dollar exposure, and keep total estimated "
            f"premium under {hedging['max_hedge_cost_pct_of_portfolio']}% "
            f"of ${capital:,.0f}. If hedging.enabled is False or no "
            "suitable contract is found, explicitly recommend hedge "
            "contracts=0 and say why."
        ),
        expected_output=(
            "A hedge recommendation stating underlying, option_symbol, "
            "strike, expiry, contracts, estimated premium, and limitations."
        ),
        agent=hedging_agent,
        context=[task_analyze],
    )

    decision_context = [task_analyze, task_hedge]
    yesterday_note = ""
    if yesterday_report:
        yesterday_note = (
            "\n\nYESTERDAY'S RESULTS (your own team's prior decision log — "
            "use this to adjust conviction/sizing, e.g. cut size on symbols "
            "that underperformed their signal, raise conviction where the "
            "signal played out as expected):\n\n" + yesterday_report
        )

    task_decide = Task(
        description=(
            "Synthesize the technical analyst's table and the hedging "
            "agent's recommendation into final per-symbol decisions plus "
            "the hedge order, following every hard rule in your goal. "
            + yesterday_note
        ),
        expected_output=(
            "A short rationale section, followed by the required fenced "
            "JSON block (decisions + hedge) as the very last thing in your output."
        ),
        agent=decision_maker_agent,
        context=decision_context,
    )

    task_execute = Task(
        description=(
            "Extract the JSON block from the decision-maker's output. Call "
            "place_portfolio_orders with its `decisions` list. If hedging "
            f"is enabled ({hedging['enabled']}) and hedge.contracts > 0, "
            "also call buy_option_hedge. Summarize what was executed."
        ),
        expected_output="Execution summary: buy/sell/hold counts, order results, hedge order result.",
        agent=execution_agent,
        context=[task_decide],
    )

    crew = Crew(
        agents=[
            data_fetcher_agent, technical_analyst_agent, hedging_agent,
            decision_maker_agent, execution_agent,
        ],
        tasks=[task_fetch, task_analyze, task_hedge, task_decide, task_execute],
        process=Process.sequential,
        verbose=True,
    )

    agent_task_pairs = [
        ("PortfolioDataFetcher", task_fetch),
        ("PortfolioTechnicalAnalyst", task_analyze),
        ("HedgingRiskAgent", task_hedge),
        ("PortfolioDecisionMaker", task_decide),
        ("PortfolioExecutionAgent", task_execute),
    ]

    return crew, agent_task_pairs


# ============================================================
# 3.0 Run one cycle
# ============================================================
def run_one_cycle(config_path: str):
    cfg = load_config(config_path)
    log_dir = cfg["logs"]["run_dir"]

    indicators_server = StdioServerParameters(
        command="python",
        args=[os.path.join(SERVERS_DIR, "technical_indicators_mcp_server.py")],
        env={**os.environ},
    )
    orders_server = StdioServerParameters(
        command="python",
        args=[os.path.join(SERVERS_DIR, "alpaca_orders_mcp_server.py")],
        env={**os.environ},
    )

    # Connects to BOTH local MCP servers at once; mcp_tools ends up as the
    # union of both servers' tools (fetch/indicators/beta/options-chain
    # from server 1, order placement from server 2).
    with MCPServerAdapter([indicators_server, orders_server]) as mcp_tools:
        yesterday_report = load_previous_decision_report(log_dir)

        crew, agent_task_pairs = build_crew(cfg, mcp_tools, yesterday_report)

        print(f"\n{'='*70}")
        print(f"  {cfg['student_name']}'s Portfolio Crew — "
              f"{len(cfg['portfolio']['stocks'])} stocks  |  "
              f"${cfg['portfolio']['starting_capital']:,.0f} capital")
        print(f"  Hedging: {'ON' if cfg['hedging']['enabled'] else 'OFF'}  |  "
              f"LLM: {cfg['llm']['model']}")
        print(f"{'='*70}\n")

        result = crew.kickoff()

        # Save each task's raw output to its own md file, per the brief's
        # agentNameDDMMYYYY-HH:MM.md naming.
        for agent_name, task in agent_task_pairs:
            output_text = str(task.output) if task.output else "(no output captured)"
            path = save_agent_log(log_dir, agent_name, output_text)
            print(f"  logged: {path}")

        print(f"\n{'='*70}\n  CYCLE COMPLETE\n{'='*70}\n")
        print(result)
        return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="../config/config.yaml")
    args = parser.parse_args()
    run_one_cycle(args.config)
