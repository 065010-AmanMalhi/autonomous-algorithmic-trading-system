# Portfolio Trading Agent Crew

A CrewAI + MCP multi-agent system for **Alpaca paper trading** across a configurable stock portfolio. It combines market-data collection, technical analysis, portfolio-level decisions, optional options hedging, order execution, and per-agent logs.

> **Important:** This project is for coursework and paper trading. It is not financial advice. Use a dedicated Alpaca paper account only. Do not switch to live-trading credentials. Automated trading can place orders without asking again once a session is running.

## Features

- CrewAI agents coordinated through MCP tools.
- Batched market data and technical indicators.
- Portfolio-level trade decisions with configurable risk limits.
- Optional SPY/QQQ put-hedge workflow.
- Alpaca paper-account orders and position/account checks.
- Session loop with a daily-loss gate and agent logs.

## Repository structure

```text
config/
  config.example.yaml          Safe template; copy and customize locally
crew/
  trading_crew.py              Builds and runs the agent crew
  logging_utils.py              Saves agent reports
mcp_servers/stock_trading/servers/
  technical_indicators_mcp_server.py
  alpaca_orders_mcp_server.py
scripts/
  run_session.py                Single-cycle test or session loop
.env.example                    Environment-variable template
requirements.txt
```

## Setup guide

A standalone team setup checklist is also available at [`docs/SETUP_GUIDE.md`](docs/SETUP_GUIDE.md).

### 1. Install prerequisites

- Python 3.11 or 3.12 recommended.
- Git.
- An Alpaca account with **paper trading** enabled.
- A Groq API key for the configured LLM provider.

Check Python and Git:

```bash
python --version
git --version
```

### 2. Clone the repository

```bash
git clone https://github.com/065010-AmanMalhi/autonomous-algorithmic-trading-system.git
cd autonomous-algorithmic-trading-system
```

### 3. Create and activate a virtual environment

**Windows PowerShell:**

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

If PowerShell blocks activation, run this for the current terminal only, then activate again:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

**macOS / Linux:**

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

### 4. Install dependencies

From the project root:

```bash
pip install -r requirements.txt
```

If installation fails, confirm the Python version and read the first package-specific error. Some packages may not yet support the newest Python release; use Python 3.11 or 3.12 rather than Python 3.14.

### 5. Add API credentials locally

Copy the sample environment file:

**Windows PowerShell**
```powershell
Copy-Item .env.example .env
notepad .env
```

**macOS / Linux**
```bash
cp .env.example .env
nano .env
```

Fill in the values:

```dotenv
ALPACA_API_KEY=your_paper_key
ALPACA_SECRET_KEY=your_paper_secret
GROQ_API_KEY=your_groq_key
```

Create fresh **paper-trading** keys in the Alpaca paper dashboard and a key at the Groq console. Never share credentials in chat, commit them to Git, or use live Alpaca keys.

The application loads `.env` automatically. Keep it local; `.gitignore` excludes it.

### 6. Create your personal portfolio configuration

Copy the example config. Each team member should use their own local filename and values.

**Windows PowerShell**
```powershell
Copy-Item config/config.example.yaml config/myname.yaml
notepad config/myname.yaml
```

**macOS / Linux**
```bash
cp config/config.example.yaml config/myname.yaml
nano config/myname.yaml
```

Update at least:
- `student_name`
- `portfolio.starting_capital`
- `portfolio.stocks` (symbols and `high_beta_hint`)
- risk limits and hedge settings as agreed by the group

Do not commit your personal `config/myname.yaml` file. The repository ignores `config/*.yaml` except `config/config.example.yaml`. Keep the example file free of credentials.

### 7. Run a one-cycle smoke test

Run from the **project root**:

```bash
python scripts/run_session.py --config config/myname.yaml --once
```

This runs one cycle and exits, without waiting for the regular market session. Order tools are **dry-run by default** (`TRADING_EXECUTION_ENABLED=false`): intended orders should be reported as `DRY_RUN` and must not reach Alpaca. Check `logs/` for generated agent reports and inspect the console for API or market-data errors. Keep options disabled until contract and premium risk checks are implemented and reviewed.

Before enabling the hedge, confirm that options trading and the required permissions are available on your Alpaca **paper** account. The options-related calls have not been verified against every version of `alpaca-py`; test this feature early and do not assume it is working solely because the code starts.

### 8. Run the scheduled session loop

```bash
python scripts/run_session.py --config config/myname.yaml
```

The loop checks the configured US Eastern clock window and runs a full agent cycle at the configured interval. Keep the terminal and machine awake while it runs. Stop with `Ctrl+C`. Order execution remains disabled by default. The runner does not yet check exchange holidays or market-calendar status.

The daily-loss check blocks new cycles when the threshold is breached, but it does **not** liquidate existing positions or cancel open orders. Review positions and orders in Alpaca separately.

## Common troubleshooting

- **Missing API key / authentication errors:** check `.env` spelling and values, then restart the process.
- **Groq model not found or rate limited:** check the current model availability in the Groq console and update `llm.model` in your local YAML if needed.
- **No market data or invalid ticker:** verify the symbol and Alpaca data permissions.
- **Options or hedge errors:** verify paper-account options permissions and test the API calls against your installed `alpaca-py` version.
- **PowerShell cannot activate `.venv`:** use the process-scoped execution-policy command above.
- **Dependency install fails:** use Python 3.11/3.12 in a fresh virtual environment and retry.

## Security checklist before pushing

- Never commit `.env`, API keys, secrets, or personal config files.
- Rotate any credentials that were ever hardcoded or shared.
- Use only Alpaca paper credentials for this coursework project.
- Review `git status --ignored` and `git diff --cached` before committing.

## Known limitations

- The web interface is not implemented; `webapp/` is only a placeholder.
- The options API calls need account-level smoke testing.
- Validate the portfolio's actual beta mix from the runtime reports; `high_beta_hint` is only a local label.
- The daily-loss guard skips future agent cycles but does not close positions or cancel orders.

## Team workflow

1. Pull the latest `main` before starting work.
2. Create a branch for changes: `git switch -c feature/short-description`.
3. Do not commit `.env` or personalized YAML files.
4. Open a pull request and have another member review it.
