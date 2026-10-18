"""
run_session.py
---------------
Entry point for the LIVE trading window. Run this on YOUR OWN machine in
VS Code (per the brief) during 9:30am-4:00pm ET = 7:00pm-1:30am IST — this
cannot run inside the Claude chat session, which doesn't stay alive for
6.5 hours and doesn't hold brokerage keys for unattended live sessions.

What it does each cycle:
    1. Checks the current time is within the configured session window.
    2. Checks the daily-loss circuit breaker (risk.daily_loss_stop_pct)
       against Alpaca's actual account equity before allowing new trades.
    3. Calls trading_crew.run_one_cycle(), which fetches data, analyzes,
       hedges, decides, executes, and logs — all for the whole portfolio
       at once.
    4. Sleeps for risk.cycle_interval_minutes and repeats.

Usage
-----
    python run_session.py --config ../config/aman.yaml

Stop anytime with Ctrl+C — in-flight day orders remain exactly as Alpaca
left them; this script does not cancel open orders on exit.
"""

import argparse
import os
import sys
import time
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "crew"))
import yaml
from trading_crew import run_one_cycle, load_config

ET = ZoneInfo("America/New_York")


def within_session_window(cfg: dict) -> bool:
    now_et = datetime.now(ET)
    start_h, start_m = map(int, cfg["session"]["start_time_et"].split(":"))
    end_h, end_m = map(int, cfg["session"]["end_time_et"].split(":"))
    start = now_et.replace(hour=start_h, minute=start_m, second=0, microsecond=0)
    end = now_et.replace(hour=end_h, minute=end_m, second=0, microsecond=0)
    return start <= now_et <= end


def check_daily_loss_stop(cfg: dict) -> bool:
    """
    Returns True if trading should continue, False if the daily-loss
    circuit breaker has tripped (existing positions/hedge are left alone —
    this only blocks NEW position-opening decisions by short-circuiting
    before the crew runs).

    NOTE: this does a lightweight direct Alpaca account check rather than
    going through the MCP server, since we need it to gate whether the
    crew runs at all.
    """
    import requests

    api_key = os.environ.get("ALPACA_API_KEY", "")
    secret = os.environ.get("ALPACA_SECRET_KEY", "")
    headers = {"APCA-API-KEY-ID": api_key, "APCA-API-SECRET-KEY": secret}

    try:
        resp = requests.get("https://paper-api.alpaca.markets/v2/account", headers=headers, timeout=10)
        resp.raise_for_status()
        acct = resp.json()
        equity = float(acct["equity"])
        last_equity = float(acct["last_equity"])  # prior trading day's close
        if last_equity <= 0 or equity <= 0:
            print("  !! DAILY LOSS STOP: invalid account equity; failing closed and skipping this cycle.")
            return False
        change_pct = (equity - last_equity) / last_equity * 100
        stop_pct = cfg["risk"]["daily_loss_stop_pct"]
        if change_pct <= -abs(stop_pct):
            print(f"  !! DAILY LOSS STOP TRIPPED: {change_pct:.2f}% (limit -{stop_pct}%). "
                  "Skipping new trades this cycle.")
            return False
        return True
    except Exception as exc:
        print(f"  !! DAILY LOSS STOP: could not verify account equity ({exc}); failing closed and skipping this cycle.")
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="../config/config.yaml")
    parser.add_argument("--once", action="store_true", help="Run a single cycle and exit, ignoring the time window (useful for testing).")
    args = parser.parse_args()

    cfg = load_config(args.config)
    interval_min = cfg["session"]["cycle_interval_minutes"]

    print(f"Session runner started for {cfg['student_name']}. "
          f"Window: {cfg['session']['start_time_et']}-{cfg['session']['end_time_et']} ET, "
          f"cycle every {interval_min} min.")

    if args.once:
        # --once bypasses the clock window for testing, never the loss circuit breaker.
        if not check_daily_loss_stop(cfg):
            print("One-cycle run blocked by the daily-loss safety check.")
            return
        run_one_cycle(args.config)
        return

    while True:
        now_et = datetime.now(ET)

        if not within_session_window(cfg):
            print(f"[{now_et.strftime('%H:%M:%S %Z')}] Outside trading window. Sleeping 5 min.")
            time.sleep(5 * 60)
            continue

        if not check_daily_loss_stop(cfg):
            time.sleep(interval_min * 60)
            continue

        print(f"[{now_et.strftime('%H:%M:%S %Z')}] Running cycle...")
        try:
            run_one_cycle(args.config)
        except Exception as exc:
            print(f"  !! Cycle failed: {exc}. Will retry next interval.")

        time.sleep(interval_min * 60)


if __name__ == "__main__":
    main()
