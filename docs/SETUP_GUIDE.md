# Team Setup Guide

This guide helps each group member install and run the Portfolio Trading Agent Crew locally. Each person must use their own API credentials and local configuration. The system is intended for coursework and **Alpaca paper trading only**.

## What you need

- Git
- Python 3.11 or 3.12 (recommended; avoid Python 3.14 for now)
- An Alpaca account with paper trading enabled
- A Groq API key for the configured language model
- A terminal such as PowerShell, Terminal, or Bash

## 1. Clone the repository

```bash
git clone https://github.com/065010-AmanMalhi/autonomous-algorithmic-trading-system.git
cd autonomous-algorithmic-trading-system
```

## 2. Create a virtual environment

### Windows PowerShell

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

If activation is blocked, allow it for this PowerShell process only and try again:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

### macOS / Linux

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

## 3. Install dependencies

Run from the repository root with the virtual environment active:

```bash
python -m pip install -r requirements.txt
```

If dependency resolution fails, confirm that the active interpreter is Python 3.11 or 3.12 and that the virtual environment is activated.

## 4. Configure API keys

Copy the template:

- Windows PowerShell: `Copy-Item .env.example .env`
- macOS / Linux: `cp .env.example .env`

Open `.env` and fill in your own credentials:

```dotenv
ALPACA_API_KEY=your_paper_key
ALPACA_SECRET_KEY=your_paper_secret
GROQ_API_KEY=your_groq_key
```

Create or retrieve keys from the **Alpaca paper-trading** dashboard and create a Groq key in the Groq console. Do not use Alpaca live-trading keys. Never paste secrets into source files, screenshots, chat, or GitHub. `.env` is excluded by `.gitignore`.

## 5. Create your local portfolio config

Keep the shared `config/config.example.yaml` unchanged. Copy it to a personal file:

- Windows PowerShell: `Copy-Item config/config.example.yaml config/myname.yaml`
- macOS / Linux: `cp config/config.example.yaml config/myname.yaml`

Edit `config/myname.yaml`. At minimum, check `student_name`, `portfolio.starting_capital`, `portfolio.stocks`, risk limits, hedge settings, LLM model, and session times. Coordinate portfolio choices and risk settings with the group. `high_beta_hint` is a label; the system computes beta separately at runtime.

Personal config files under `config/*.yaml` are ignored by Git, except the tracked `config/config.example.yaml`. Do not force-add your personal config.

## 6. Run one test cycle

From the repository root, with `.venv` active and `.env` configured:

```bash
python scripts/run_session.py --config config/myname.yaml --once
```

This bypasses the normal session-time window, but not the daily-loss circuit breaker. Review terminal output and generated reports in `logs/`. Order tools are dry-run by default (`TRADING_EXECUTION_ENABLED=false`), so intended orders should be marked `DRY_RUN` and must not be sent to Alpaca. Verify no orders were created. Keep `TRADING_OPTIONS_ENABLED=false` while options remain unvalidated.

## 7. Run the session loop

```bash
python scripts/run_session.py --config config/myname.yaml
```

The runner checks the configured US Eastern clock window and repeats at the configured interval. Keep the terminal and machine awake. Stop the process with `Ctrl+C`. Stopping the script does not cancel open orders. The daily-loss gate fails closed if account equity cannot be verified and prevents subsequent cycles after its threshold is reached; it does not liquidate positions or cancel orders. The current session loop does not yet check exchange holidays or market-calendar status.

## Troubleshooting

- **Authentication errors:** check variable names and values in `.env`, then restart the process.
- **Groq model error or rate limit:** check current model availability and update `llm.model` in your personal YAML if required.
- **Missing market data / symbol errors:** check the ticker and your Alpaca data permissions.
- **Options/hedge errors:** options API behavior and account permissions need to be tested on your own paper account; do not assume the hedge flow works without a successful test.
- **PowerShell activation error:** use the process-scoped execution-policy command above.
- **Install errors:** use Python 3.11 or 3.12 in a clean virtual environment.

## Before every run

- Verify `.env` contains your own paper-trading credentials.
- Keep `TRADING_EXECUTION_ENABLED=false` during development and testing.
- Keep `TRADING_OPTIONS_ENABLED=false`; hedging is disabled in the example config.
- Verify the configuration path and portfolio/risk settings.
- Confirm the Alpaca account is paper trading.
- Review existing positions and open orders.
- In default dry-run mode, order tools must not submit orders. Do not enable execution until all independent risk checks have been implemented and reviewed.

## Team contribution workflow

```bash
git pull origin main
git switch -c feature/short-description
# make and test your changes
git add <specific-files>
git commit -m "Describe the change"
git push -u origin feature/short-description
```

Open a pull request and ask another group member to review it. Never commit `.env`, personal YAML configs, logs, or API keys.

## Known limitations

- The web interface is not implemented; `webapp/` is only a placeholder.
- Options API calls require paper-account smoke testing against the installed `alpaca-py` version.
- `high_beta_hint` is not treated as verified beta data.
- The daily-loss guard skips future cycles but does not close positions or cancel open orders.
