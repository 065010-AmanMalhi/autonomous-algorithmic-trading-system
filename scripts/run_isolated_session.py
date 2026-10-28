"""
run_isolated_session.py

Launch one student's paper-trading session with explicit credentials and
an account-ID check. Run from the repository root:
    python scripts/run_isolated_session.py --student aman
    python scripts/run_isolated_session.py --student mayank
    python scripts/run_isolated_session.py --student aditi

Add --once to run one gated cycle:
    python scripts/run_isolated_session.py --student aman --once

Do not launch the original run_session.py directly for student sessions.
"""
import argparse
import os
import runpy
import sys
from pathlib import Path

from dotenv import load_dotenv
from alpaca.trading.client import TradingClient

ROOT = Path(__file__).resolve().parent.parent

STUDENTS = {
    "aman": {
        "env": ROOT / ".env",
        "config": ROOT / "config" / "aman.yaml",
        "account_id": "5ec8611a-ef81-4079-ae4b-40a120e425bb",
    },
    "mayank": {
        "env": ROOT / ".env.mayank",
        "config": ROOT / "config" / "mayank.yaml",
        "account_id": "0061b552-be41-4825-89ba-b6dcc69b80ab",
    },
    "aditi": {
        "env": ROOT / ".env.aditi",
        "config": ROOT / "config" / "aditi.yaml",
        "account_id": "15a2908f-9468-40f0-9cb6-cd4664d28ff8",
    },
}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Launch a student paper-trading session with account isolation."
    )
    parser.add_argument("--student", required=True, choices=sorted(STUDENTS))
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run one gated cycle, then exit (orders may still be submitted).",
    )
    args = parser.parse_args()

    settings = STUDENTS[args.student]
    env_file = settings["env"]
    config_file = settings["config"]
    runner_file = ROOT / "scripts" / "run_session.py"

    for required_path in (env_file, config_file, runner_file):
        if not required_path.is_file():
            raise SystemExit(f"BLOCKED: required file not found: {required_path}")

    # Override inherited terminal variables so a stale key cannot silently
    # select another student's Alpaca account.
    load_dotenv(env_file, override=True)

    key = os.environ.get("ALPACA_API_KEY", "").strip()
    secret = os.environ.get("ALPACA_SECRET_KEY", "").strip()
    if not key or not secret:
        raise SystemExit(f"BLOCKED: missing Alpaca credentials in {env_file.name}")

    try:
        client = TradingClient(key, secret, paper=True)
        account = client.get_account()
    except Exception as exc:
        raise SystemExit(
            f"BLOCKED: Alpaca account verification failed ({type(exc).__name__})."
        ) from exc

    actual_id = str(account.id)
    expected_id = settings["account_id"]
    if actual_id != expected_id:
        raise SystemExit(
            f"BLOCKED: {args.student} credentials resolved to an unexpected "
            f"Alpaca account ID ({actual_id}); expected account does not match. "
            "No session was started."
        )

    if str(account.status).split(".")[-1].upper() != "ACTIVE":
        raise SystemExit(f"BLOCKED: Alpaca account status is {account.status}.")
    if getattr(account, "trading_blocked", False) or getattr(account, "account_blocked", False):
        raise SystemExit("BLOCKED: Alpaca reports this account is blocked.")

    print(
        f"Verified student={args.student}; account_id={actual_id}; "
        f"config={config_file.relative_to(ROOT)}"
    )

    # Bind all order-server processes to this verified paper account.
    os.environ["EXPECTED_ALPACA_ACCOUNT_ID"] = expected_id

    # Ensure the runner receives an explicit absolute config path.
    sys.argv = [str(runner_file), "--config", str(config_file)]
    if args.once:
        sys.argv.append("--once")

    # The selected credentials are already in this process environment;
    # trading_crew.py passes them to its MCP subprocesses.
    runpy.run_path(str(runner_file), run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
