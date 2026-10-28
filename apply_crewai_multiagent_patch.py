#!/usr/bin/env python3
"""Safely upgrade the compact trader to a 3-agent CrewAI pipeline.

Run from the project root. Makes a timestamped backup before writing.
Does not change Alpaca paper mode, order-server checks, or risk limits.
"""
from __future__ import annotations
import re
import shutil
from datetime import datetime
from pathlib import Path

TARGET = Path("crew/trading_crew.py")


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"Expected exactly one match for {label}; found {count}. No file was changed.")
    return text.replace(old, new, 1)


def main() -> None:
    if not TARGET.is_file():
        raise SystemExit("Run this script from your project root; crew/trading_crew.py was not found.")
    source = TARGET.read_text(encoding="utf-8")
    if "CrewAI pipeline: Technical Signal Analyst -> Portfolio Strategist -> Risk & Execution Supervisor" in source:
        raise SystemExit("Patch already appears to be applied; no changes made.")

    updated = source
    updated = replace_once(
        updated,
        '"""\nLow-token, single-inference portfolio trader for Groq + local MCP servers.\n\nData fetching and indicator calculations run deterministically in Python/MCP.\nGroq receives one compact portfolio snapshot and returns one JSON decision set.\nPython validates that output and submits one batch to Alpaca PAPER trading.\nThe independent order server remains the final authority for risk checks.\n"""',
        '"""\nCrewAI multi-agent portfolio trader for Groq + local MCP servers.\n\nPython/MCP fetches and computes market evidence. Three CrewAI specialists analyze\nsignals, build a portfolio plan, and produce a risk-constrained JSON decision set.\nPython validates decisions and the independent MCP order server remains the final\nauthority for account, position, cash, frequency, halt, and paper-only controls.\n"""',
        "module description",
    )

    old_score = '''    if 40.0 <= rsi <= 65.0:\n        buy_score += 1.0\n    elif 35.0 <= rsi < 40.0 or 65.0 < rsi <= 68.0:\n        buy_score += 0.5\n    elif rsi >= 70.0 or (0.0 < rsi < 30.0):\n        buy_score -= 2.0\n'''
    new_score = '''    if 40.0 <= rsi <= 65.0:\n        buy_score += 1.0\n    elif 35.0 <= rsi < 40.0 or 65.0 < rsi <= 68.0:\n        buy_score += 0.5\n    elif 68.0 < rsi < 75.0:\n        # Allow a modest RSI extension only when trend, MACD, and price location agree.\n        buy_score -= 0.5\n    elif rsi >= 75.0 or (0.0 < rsi < 30.0):\n        buy_score -= 2.0\n'''
    updated = replace_once(updated, old_score, new_score, "RSI score window")

    old_gate = '''    buy_rsi_ok = 35.0 <= rsi <= 68.0\n    # Confirmation is an actual bullish crossover OR constructive position\n    # inside the Bollinger range. No full-day volume comparison intraday.\n    macd_bullish_momentum = (macd == "bullish") or (has_macd_histogram and macd_histogram > 0)\n    buy_confirmation = macd_bullish_momentum or (0.15 <= bb <= 0.85)\n    buy_eligible = (\n        not trading_halted\n        and trend == "up"\n        and macd != "bearish"\n        and buy_rsi_ok\n        and buy_confirmation\n        and buy_score >= 3.0\n    )\n'''
    new_gate = '''    # RSI up to 75 is allowed for trend-following only if price is not\n    # already overextended relative to the Bollinger bands.\n    buy_rsi_ok = 35.0 <= rsi <= 75.0\n    macd_bullish_momentum = (macd == "bullish") or (has_macd_histogram and macd_histogram > 0)\n    buy_price_position_ok = 0.15 <= bb <= 0.85\n    buy_confirmation = macd_bullish_momentum or buy_price_position_ok\n    buy_eligible = (\n        not trading_halted\n        and trend == "up"\n        and macd != "bearish"\n        and buy_rsi_ok\n        and buy_price_position_ok\n        and buy_confirmation\n        and buy_score >= 3.0\n    )\n'''
    updated = replace_once(updated, old_gate, new_gate, "entry gate")

    old_diag = '''    if not buy_rsi_ok:\n        why.append(f"RSI={rsi if rsi >= 0 else 'missing'} outside 35-68")\n    if macd == "bearish":\n        why.append("MACD crossover bearish")\n    if not buy_confirmation:\n        why.append("no positive MACD histogram/crossover or constructive Bollinger position")\n'''
    new_diag = '''    if not buy_rsi_ok:\n        why.append(f"RSI={rsi if rsi >= 0 else 'missing'} outside 35-75")\n    if macd == "bearish":\n        why.append("MACD crossover bearish")\n    if not buy_price_position_ok:\n        why.append(f"Bollinger %B={bb if -10 < bb < 10 else 'missing'} outside 0.15-0.85 entry range")\n    if not buy_confirmation:\n        why.append("no positive MACD histogram/crossover or constructive Bollinger position")\n'''
    updated = replace_once(updated, old_diag, new_diag, "entry diagnostics")

    old_prompt = '''        "ENTRY RULE: consider BUY if trend is 'Up', RSI is 35-68, MACD is not 'Bearish', "\n        "and either MACD='Bullish' or bb_percent_b is between 0.15 and 0.85. Favor HOLD if the "\n        "indicators conflict, the trend is not Up, RSI is outside the range, or required data is "\n        "missing. A fresh MACD crossover is not required when the Bollinger position is constructive. "\n'''
    new_prompt = '''        "ENTRY RULE: actively evaluate a small starter BUY when trend is 'Up', MACD histogram "\n        "is positive (or crossover is 'Bullish'), RSI is 35-75, and bb_percent_b is 0.15-0.85. "\n        "RSI 69-75 is allowed only when trend and momentum are positive and price remains below "\n        "the upper Bollinger zone; do not chase when %B > 0.85. Favor HOLD if histogram is "\n        "non-positive without a bullish crossover, price location is stretched, the trend is not Up, "\n        "required data is missing, or trading is halted. A fresh crossover is not required when "\n        "the histogram is already positive. "\n'''
    updated = replace_once(updated, old_prompt, new_prompt, "decision-agent entry prompt")

    pattern = re.compile(
        r'            agent = Agent\(\n'
        r'                role="Compact Portfolio Decision-Maker",.*?'
        r'            raw = str\(crew\.kickoff\(\)\)',
        re.DOTALL,
    )
    matches = list(pattern.finditer(updated))
    if len(matches) != 1:
        raise RuntimeError(f"Expected exactly one single-agent CrewAI block; found {len(matches)}. No file was changed.")

    new_crew_block = r'''            # Three real CrewAI specialists run sequentially on every successful cycle.
            # All agents see evidence only; Python/order-server safeguards remain authoritative.
            analyst = Agent(
                role="Technical Signal Analyst",
                goal="Identify evidence-backed trend and momentum candidates for every symbol.",
                backstory=(
                    "You are a technical analyst. Read numeric RSI, MACD histogram/lines, "
                    "crossover, trend, and Bollinger %B carefully. None crossover means no "
                    "fresh crossover, not missing momentum. Never invent market data."
                ),
                llm=llm, verbose=False, max_iter=1, allow_delegation=False,
            )
            strategist = Agent(
                role="Portfolio Strategist",
                goal="Turn technical evidence into balanced BUY, SELL, and HOLD proposals.",
                backstory=(
                    "You manage a diversified long-only paper portfolio. You seek valid "
                    "opportunities without forcing trades, chasing extended prices, shorting, "
                    "using leverage, or proposing options."
                ),
                llm=llm, verbose=False, max_iter=1, allow_delegation=False,
            )
            risk_supervisor = Agent(
                role="Risk & Execution Supervisor",
                goal="Produce the final schema-valid decision set within every configured limit.",
                backstory=(
                    "You are the final pre-trade reviewer. Never short or use leverage. "
                    "Use exact positions/cash and leave the reserve intact. Halt means no buys. "
                    "A SELL requires an existing long position."
                ),
                llm=llm, verbose=False, max_iter=1, allow_delegation=False,
            )

            analysis_task = Task(
                description=(
                    "Analyze all symbols in the supplied original market/account context. "
                    "For every symbol, report trend, RSI, MACD crossover, MACD histogram, "
                    "Bollinger %B, and classify the setup as bullish, mixed, or bearish. "
                    "Explicitly identify whether positive histogram momentum persists without "
                    "a fresh crossover. Identify possible entries only when trend is Up, "
                    "RSI is 35-75, histogram is positive or crossover is Bullish, and %B is "
                    "0.15-0.85. Keep it concise and numeric; do not size orders or invent data.\n\n"
                    + task_description
                ),
                expected_output="Concise per-symbol technical assessment using supplied numbers only.",
                agent=analyst,
            )
            strategy_task = Task(
                description=(
                    "Use the technical analyst's findings to assess every symbol. Recommend "
                    "BUY only for an Up trend, RSI 35-75, positive MACD histogram or Bullish "
                    "crossover, and Bollinger %B 0.15-0.85. RSI 69-75 is not an automatic veto "
                    "when momentum is positive and price remains in that band. Avoid chasing %B "
                    "above 0.85. SELL only an existing long when trend is Down or MACD is Bearish "
                    "and RSI <= 40 or %B <= 0.20 confirms weakness. Otherwise HOLD. Consider all "
                    "eligible candidates; do not default the entire portfolio to HOLD without "
                    "evaluating each row. Return a concise symbol/action/reason proposal, not JSON."
                ),
                expected_output="Concise proposed portfolio actions with an evidence-based reason per candidate.",
                agent=strategist,
                context=[analysis_task],
            )
            decision_task = Task(
                description=(
                    "Finalize the entire portfolio from the original evidence and preceding "
                    "specialist outputs. Apply the stated rules; do not force a trade where no "
                    "candidate qualifies, but do not leave eligible candidates at HOLD merely "
                    "because RSI is between 69 and 75 when positive momentum and %B 0.15-0.85 "
                    "agree. Return only valid JSON with exactly one entry per supplied symbol, "
                    "keys symbol/action/notional/confidence/reason, and actions buy/sell/hold. "
                    "HOLD notional must be 0. Buy notional must be positive and within the "
                    "max_position_dollars and cash reserve; use a starter size near 25% of the "
                    "per-position cap where account capacity permits. Confidence must be 0-1 and "
                    "meet the configured minimum for BUY. Never buy during a halt. Never short, "
                    "use leverage, invent data, or propose options. Limit each reason to 8 words.\n\n"
                    "ORIGINAL EVIDENCE AND LIMITS:\n" + task_description
                ),
                expected_output="Only valid JSON with one risk-reviewed decision per portfolio symbol.",
                agent=risk_supervisor,
                context=[analysis_task, strategy_task],
            )
            crew = Crew(
                agents=[analyst, strategist, risk_supervisor],
                tasks=[analysis_task, strategy_task, decision_task],
                process=Process.sequential,
                verbose=False,
            )
            print("  CrewAI pipeline: Technical Signal Analyst -> Portfolio Strategist -> Risk & Execution Supervisor")
            raw = str(crew.kickoff())
            # Keep each specialist's actual CrewAI output for the assignment audit trail.
            for stage_name, stage_task in (
                ("TechnicalSignalAnalyst", analysis_task),
                ("PortfolioStrategist", strategy_task),
                ("RiskExecutionSupervisor", decision_task),
            ):
                stage_output = getattr(stage_task, "output", None)
                if stage_output is not None:
                    stage_text = getattr(stage_output, "raw", None) or str(stage_output)
                    try:
                        save_agent_log(cfg.get("logs", {}).get("run_dir", "logs"), stage_name, str(stage_text))
                    except Exception as log_exc:
                        print(f"  [CrewAI log warning] {stage_name}: {type(log_exc).__name__}: {log_exc}")'''
    updated = pattern.sub(lambda _m: new_crew_block, updated, count=1)

    backup = TARGET.with_name(TARGET.name + "." + datetime.now().strftime("%Y%m%d-%H%M%S") + ".bak")
    shutil.copy2(TARGET, backup)
    TARGET.write_text(updated, encoding="utf-8", newline="\n")
    print(f"Backup created: {backup}")
    print("Updated: crew/trading_crew.py")
    print("CrewAI now runs 3 sequential specialist agents/tasks per successful model attempt.")
    print("Signal policy allows a measured RSI 69-75 trend continuation only with positive momentum and %B 0.15-0.85.")
    print("Existing risk limits and Alpaca paper-only configuration were not modified.")

if __name__ == "__main__":
    main()
