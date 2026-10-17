# Team Setup Guide

Each group member should install the project locally, use their own API credentials, and maintain a personal config file. This system is intended for coursework and **Alpaca paper trading only**.

## Prerequisites
- Git
- Python 3.11 or 3.12 (recommended; avoid Python 3.14 for now)
- An Alpaca account with paper trading enabled
- A Groq API key for the configured language model

## 1. Clone

    git clone https://github.com/065010-AmanMalhi/autonomous-algorithmic-trading-system.git
    cd autonomous-algorithmic-trading-system

## 2. Create a virtual environment

### Windows PowerShell

    py -3.12 -m venv .venv
    .\.venv\Scripts\Activate.ps1
    python -m pip install --upgrade pip

If activation is blocked, run this in the current PowerShell process, then activate again:

    Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
    .\.venv\Scripts\Activate.ps1

### macOS / Linux

    python3.12 -m venv .venv
    source .venv/bin/activate
    python -m pip install --upgrade pip

## 3. Install dependencies
From the repository root, with the environment activated:

    python -m pip install -r requirements.txt

If dependency resolution fails, verify the active interpreter is Python 3.11 or 3.12 and that the virtual environment is activated.

## 4. Configure credentials
Copy the template:
- PowerShell: Copy-Item .env.example .env
- macOS/Linux: cp .env.example .env

Edit .env:

    ALPACA_API_KEY=your_paper_key
    ALPACA_SECRET_KEY=your_paper_secret
    GROQ_API_KEY=your_groq_key

Use credentials from the **Alpaca paper-trading** dashboard, not live trading. Create a Groq key in the Groq console. Never paste secrets into source files, screenshots, chat, or GitHub. The .env file is excluded by .gitignore.

## 5. Create a personal portfolio config
Keep config/config.example.yaml unchanged. Copy it to a personal file:
- PowerShell: Copy-Item config/config.example.yaml config/myname.yaml
- macOS/Linux: cp config/config.example.yaml config/myname.yaml

Edit config/myname.yaml: check student_name, portfolio.starting_capital, portfolio.stocks, risk limits, hedge settings, LLM model, and session times. Coordinate choices with the group. high_beta_hint is a local label; the system computes beta separately at runtime.

Personal YAML files under config/*.yaml are ignored by Git except for the tracked example config. Do not force-add personal configs.

## 6. Run one cycle

    python scripts/run_session.py --config config/myname.yaml --once

This bypasses the session-time window and runs one cycle. Review console output and reports in logs/. A one-cycle run can submit paper orders, so confirm the account is paper trading and inspect orders/positions afterward.

## 7. Run the session loop

    python scripts/run_session.py --config config/myname.yaml

The runner checks the configured US Eastern session window and repeats at the configured interval. Keep the terminal and machine awake; stop with Ctrl+C. Stopping the script does not cancel open orders. The daily-loss gate prevents future cycles after its threshold is reached, but does not liquidate positions or cancel orders.

## Troubleshooting
- **Authentication errors:** check variable names and values in .env, then restart.
- **Groq model/rate-limit errors:** check current model availability and update llm.model in your personal YAML if needed.
- **Missing market data / invalid ticker:** verify the symbol and Alpaca data permissions.
- **Options/hedge errors:** verify paper-account options permissions and test against the installed alpaca-py version.
- **PowerShell activation error:** use the process-scoped execution-policy command above.
- **Install errors:** use Python 3.11 or 3.12 in a clean virtual environment.

## Before every run
- Confirm your own paper-trading credentials are loaded.
- Confirm config path, portfolio, and risk settings.
- Review existing positions and open orders.
- Remember autonomous execution may submit orders without another confirmation.

## Team workflow

    git pull origin main
    git switch -c feature/short-description
    # make and test changes
    git add <specific-files>
    git commit -m "Describe the change"
    git push -u origin feature/short-description

Open a pull request and request a peer review. Never commit .env, personal YAML configs, logs, or API keys.

## Known limitations
- The web interface is not implemented; webapp/ is only a placeholder.
- Options API calls require paper-account smoke testing.
- high_beta_hint is not verified beta data.
- The daily-loss guard skips future cycles but does not close positions or cancel open orders.
