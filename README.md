# Portfolio Trading Agent Crew — Group Project

CrewAI + MCP autonomous paper-trading system for the "Trade Algorithmically
using Trading Agents" brief. Trades a whole 15-stock portfolio at once
(not one ticker per run), with a compact technical-analysis workflow designed for Groq's free-tier
token-per-minute limits. Optional volatility/fear regression and options
tools remain in the MCP server but are not used in the default cycle.

Built on top of the class starter files (`trading_crew.py`,
`technical_indicators_mcp_server.py`, `alpaca_trading_mcpServer.py`) —
extended for batched multi-symbol data, a portfolio-level decision maker,
reusability across all three group members; optional hedging remains disabled in the example config.

## ⚠️ Security note

The class file comment for `trading_crew.py` had a real Alpaca paper key
and secret hardcoded in it. Those are now compromised — **do not use
them.** Generate your own fresh paper keys at
https://app.alpaca.markets/paper/dashboard/overview, and never commit
`.env` or a filled-in `config/<name>.yaml` to a shared repo (`.gitignore`
here already excludes both — keep it that way).

## Project layout

```
mcp_servers/stock_trading/servers/
    technical_indicators_mcp_server.py   MCP server 1: batched OHLCV fetch,
                                          indicators (RSI/MACD/Bollinger/
                                          ATR/trend), realized beta, compact
                                          whole-portfolio snapshot, plus optional tools
    alpaca_orders_mcp_server.py          MCP server 2: whole-portfolio order
                                          batch, options hedge orders,
                                          account/positions
crew/
    trading_crew.py                      MCP client — one-inference decision
                                          pass with deterministic pre/post-processing
    logging_utils.py                     agentNameDDMMYYYY-HH:MM.md logging
                                          + "load yesterday's report" helper
scripts/
    run_session.py                       live-window loop (run THIS during
                                          the actual trading session)
config/
    config.example.yaml                  copy per-teammate, this is what
                                          makes the system reusable
logs/                                    created at runtime, one dated
                                          subfolder per trading day
webapp/                                  placeholder — not built yet
```

## Setup (each group member does this individually)

```bash
cd trading-agent-group
python -m venv .venv && source .venv/bin/activate   # or use uv
pip install -r requirements.txt

cp .env.example .env              # fill in YOUR OWN keys
cp config/config.example.yaml config/<your_name>.yaml   # edit your stock list
```

The example config has options hedging disabled to keep the default run compact and reliable. If you deliberately enable `hedging.enabled` for a later experiment, confirm options trading is available on your Alpaca paper account before relying on the hedge leg.

## Test a single cycle (no need to sit through the whole window)

```bash
cd scripts
python run_session.py --config ../config/<your_name>.yaml --once
```

Check `logs/<DDMMYYYY>/` for four stage logs: data fetch, technical snapshot, decisions, and execution.

## Run the real session

```bash
cd scripts
python run_session.py --config ../config/<your_name>.yaml
```

Start this around 7:00pm IST (9:30am ET). It loops every
`session.cycle_interval_minutes` (default 30) until 1:30am IST (4:00pm
ET), re-running the compact cycle each interval and logging its four stages. Earlier verbose reports are not re-fed to Groq. A daily-loss circuit breaker
(`risk.daily_loss_stop_pct`) blocks new positions (not the hedge) if the
account drops too far intraday.

## Compact Groq workflow

The default cycle makes one LLM inference. Python calls the batched OHLCV fetch,
computes technical indicators and realized beta for the whole portfolio through one
compact local tool, retrieves current paper-account positions/cash, then sends a
small JSON snapshot to Groq. The returned decisions are validated locally and submitted
in one batch to the paper-order server. It does not ask an LLM to call MCP tools.

The default example config uses `groq/openai/gpt-oss-20b`, a 1,200-token
completion ceiling, 120 calendar days of OHLCV data, a 60-trading-day beta
window, and options hedging disabled. These values reduce prompt size; the actual
Groq limits depend on the account plan/model and can change.

The server-side paper-trading and risk guards remain authoritative. The order
server is pinned to Alpaca paper mode and independently checks position limits,
cash buffer, confidence, trade frequency, halt flag, and hedge limits if hedging
is explicitly enabled in a personal config. Do not switch `paper=True` to live
trading. Optional regression and options tools remain in the MCP server for
future experimentation but are not part of the compact default run.

## Safety model — read before any live run

The decision prompt includes the configured risk limits. Python then validates model output
against current Alpaca paper-account cash, positions, confidence, sizing and the daily halt state.
The `alpaca_orders_mcp_server.py` independently applies its own configured checks immediately
before orders are submitted. No LLM output is trusted as a final order instruction. Concretely,
before an order reaches Alpaca:

- `action` is matched against an explicit `buy`/`sell`/`hold` allow-list —
  anything else is rejected outright, never silently treated as a sell.
- Position size and cash buffer are checked against the account's real
  `portfolio_value`/`cash` fetched at call time.
- `confidence` is required on every buy and checked against
  `min_confidence_to_trade` server-side.
- Per-symbol trade frequency is tracked in an on-disk ledger
  (`logs/<date>/trade_ledger.json`) that persists across cycles and
  restarts — not just within one batch call.
- The options hedge independently re-parses the OCC symbol (underlying,
  expiry, type, strike) and fetches a live quote to cap premium spend —
  it never trusts a cost figure the agent stated.
- A tripped daily-loss stop writes `logs/<date>/.trading_halted`, a file
  both `run_session.py` and the orders server check — once written, every
  new buy is rejected for the rest of that calendar day, across restarts,
  regardless of which script is driving. `--once` is gated by this exactly
  the same way the live loop is; there's no bypass path anymore.

The compact cycle writes a safety snapshot into its execution-stage log. You can also call
`get_safety_status` via an MCP inspector to see the current halt state, limits applied and
today's per-symbol ledger when investigating a run with fewer trades than expected.

## What's not built yet

- **Web interface** (brief section 9) — next.
- Individually-verified beta for the 10 non-high-beta stocks in the
  example config — the `calculate_beta` tool computes this from real
  Alpaca data on first run; check the Portfolio Technical Analyst's log
  to confirm your actual mix still clears the 30% high-beta rule once
  real numbers are in, and swap a stock out if it doesn't.
- Options API surface (`get_option_chain`, `buy_option_hedge`) is written
  against alpaca-py's documented endpoints but **not yet run against a
  live account** — test this early, options trading support across
  alpaca-py versions has moved fast; if a call shape is off, the error
  messages from Alpaca are usually specific enough to patch quickly.
