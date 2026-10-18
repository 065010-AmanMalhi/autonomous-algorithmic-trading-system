# Portfolio Trading Agent Crew — Group Project

CrewAI + MCP autonomous paper-trading system for the "Trade Algorithmically
using Trading Agents" brief. Trades a whole 15-stock portfolio at once
(not one ticker per run), with a volatility/implied-volatility "fear"
regression agent, an options-based tail hedge, and a day-over-day
self-improvement loop.

Built on top of the class starter files (`trading_crew.py`,
`technical_indicators_mcp_server.py`, `alpaca_trading_mcpServer.py`) —
extended for batched multi-symbol data, a portfolio-level decision maker,
hedging, and reusability across all three group members.

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
                                          ATR/trend), realized beta, realized
                                          vs implied volatility, market-wide
                                          fear level (VIXY proxy), a linear-
                                          regression fear/vol combiner, and
                                          option chain lookup
    alpaca_orders_mcp_server.py          MCP server 2: whole-portfolio order
                                          batch, options hedge orders,
                                          account/positions
crew/
    trading_crew.py                      MCP client — 6-agent crew, one
                                          run_one_cycle() = one full pass
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

In the Alpaca dashboard, confirm **options trading is enabled** on your
paper account (needed for the hedge leg) — Settings → Options Trading →
request Level 1 or 2 approval. This is instant on paper accounts.

## Test a single cycle (no need to sit through the whole window)

```bash
cd scripts
python run_session.py --config ../config/<your_name>.yaml --once
```

Check `logs/<DDMMYYYY>/` for the five agent output files afterward.

## Run the real session

```bash
cd scripts
python run_session.py --config ../config/<your_name>.yaml
```

Start this around 7:00pm IST (9:30am ET). It loops every
`session.cycle_interval_minutes` (default 30) until 1:30am IST (4:00pm
ET), re-running the whole crew each cycle, logging every agent's output,
and feeding the prior cycle's decision-maker report back in as context —
that's the improve-as-we-go loop. A daily-loss circuit breaker
(`risk.daily_loss_stop_pct`) blocks new positions (not the hedge) if the
account drops too far intraday.

## Volatility & fear regression agent

Sits between the Technical Analyst and the Hedging Agent in the crew. For
every symbol it computes:

- **Realized volatility (RV)** — annualized stdev of daily log returns
  (`calculate_realized_volatility`).
- **Implied volatility (IV)** — read directly from Alpaca's option
  snapshot endpoint for the nearest at-the-money contract
  (`get_iv_metrics`; Alpaca computes IV/greeks itself via Black-Scholes,
  this tool just reads those fields rather than re-deriving them).
- **IV-RV spread** — a wide positive spread (IV well above RV) means the
  options market is pricing in more future movement than has actually
  occurred: a classic sign the market senses an event or elevated risk
  ahead.
- **Market-wide fear level** — via VIXY (a VIX-futures ETF), since Alpaca
  only serves equities/ETPs and doesn't carry the real CBOE VIX index
  (`get_market_volatility`).
- **`fear_adjusted_signal`** — a single combined number per symbol from
  `calculate_fear_regression`: an in-sample OLS linear regression (via
  scikit-learn) of next-day return on realized vol, RSI, relative volume
  and market-fear-change, with the IV-RV spread applied as a post-fit
  adjustment. **Read the docstring on that function before trusting it**
  — with ~60 data points and a handful of features this is a transparent
  way to combine signals with sensible weights, not a validated
  forecasting model. The tool reports `fit_r_squared` for exactly this
  reason; below ~0.15 the decision-maker is instructed to treat the
  reading as noise.

This agent's output feeds two places: the Hedging Agent leans toward
more hedge contracts (within the premium cap) when market fear is
`Elevated`, and the Decision-Maker treats a Fear-leaning signal as a
reason to size down or wait on an otherwise-bullish name — one input
among several, never a standalone trigger.

## Safety model — read before any live run

Every `risk.*` and `hedging.*` limit in your config is enforced **twice**:
once as an instruction in the decision-maker agent's prompt, and
independently again, server-side, in `alpaca_orders_mcp_server.py` —
against LIVE Alpaca account data, not whatever the agent's JSON claims.
The second layer is the one that actually matters; the first is just
there so the agent doesn't waste cycles proposing trades it knows will be
rejected. Concretely, before any order reaches Alpaca:

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

Call the `get_safety_status` tool (via the MCP inspector, or ask the crew
to call it) anytime to see current halt state, the limits actually being
applied, and today's trade counts — useful when a run produces fewer
trades than you expected and you want to know whether that's the agents
being cautious or a limit kicking in.

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
