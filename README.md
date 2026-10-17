# Autonomous Algorithmic Trading System

A CrewAI + MCP multi-agent system for **Alpaca paper trading** across a configurable stock portfolio. It combines market-data collection, technical analysis, portfolio-level decisions, optional options hedging, order execution, and per-agent logs.

> **Important:** This project is for coursework and paper trading. It is not financial advice. Use a dedicated Alpaca paper account only. Automated trading can place orders without asking again once a session is running.

## Features
- CrewAI agents coordinated through MCP tools.
- Batched market data and technical indicators.
- Portfolio-level trade decisions with configurable risk limits.
- Optional SPY/QQQ put-hedge workflow.
- Alpaca paper-account orders and position/account checks.
- Session loop with a daily-loss gate and agent logs.

## Setup
See [docs/SETUP_GUIDE.md](docs/SETUP_GUIDE.md) for complete Windows/macOS/Linux setup instructions.

Clone, create a Python 3.11 or 3.12 virtual environment, and install dependencies:

    git clone https://github.com/065010-AmanMalhi/autonomous-algorithmic-trading-system.git
    cd autonomous-algorithmic-trading-system
    python -m pip install -r requirements.txt

Copy .env.example to .env, add your own Alpaca **paper** credentials and Groq key, then copy config/config.example.yaml to a personal YAML file and edit it. Run one cycle with:

    python scripts/run_session.py --config config/myname.yaml --once

## Repository structure
- crew/: CrewAI orchestration and logging.
- mcp_servers/stock_trading/servers/: market-indicator and Alpaca order MCP servers.
- scripts/run_session.py: single-cycle test or session loop.
- config/config.example.yaml: sample configuration.
- docs/SETUP_GUIDE.md: team installation guide.

## Security and trading risk
- Never commit .env, API keys, personal YAML config files, or runtime logs.
- Use Alpaca paper-trading credentials only.
- Review all orders and positions in the Alpaca dashboard.
- The daily-loss gate does not liquidate positions or cancel open orders.
- Options-related calls require paper-account permission checks and smoke testing.

The project is a coursework prototype, not production trading software or financial advice.
