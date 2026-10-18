"""
run_session.py
---------------
Entry point for the LIVE trading window. Run this on YOUR OWN machine in
VS Code (per the brief) during 9:30am-4:00pm ET = 7:00pm-1:30am IST — this
cannot run inside the Claude chat session, which doesn't stay alive for
6.5 hours and doesn't hold brokerage keys for unattended live sessions.

What it does each cycle:
    1. Checks the current time is within the configured session window.
    2. Checks the on-disk trading-halted flag FIRST (cheap, no API call).
       If a previous cycle already tripped the daily-loss circuit breaker
       today, every subsequent cycle skips immediately — this is a
       persistent, file-based latch (see logs/<date>/.trading_halted),
       not an in-memory flag that forgets itself on restart.
    3. If not already halted, checks the daily-loss circuit breaker
       (risk.daily_loss_stop_pct) against Alpaca's live account equity.
       Tripping it WRITES the halt file — it does not just skip one
       cycle and quietly retry next interval.
    4. Calls trading_crew.run_one_cycle(), which fetches data, analyzes,
       hedges, decides, executes, and logs — all for the whole portfolio
       at once. The execution layer (alpaca_orders_mcp_server.py) ALSO
       checks this same halt file independently before submitting any
       buy order, so a bug here doesn't leave the account unprotected.
    5. Sleeps for session.cycle_interval_minutes and repeats.

IMPORTANT: --once no longer skips any of this. A single test cycle is
still gated by the halt file and the live daily-loss check, exactly like
the real loop — the only thing --once skips is the time-window check and
the sleep/repeat, so you can test outside market hours.

Usage
-----
    python run_session.py --config ../config/aman.yaml           # full loop
    python run_session.py --config ../config/aman.yaml --once    # one cycle, for testing

Stop anytime with Ctrl+C — in-flight day orders remain exactly as Alpaca
left them; this script does not cancel open orders on exit.
"""

import argparse
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "crew"))
from trading_crew import run_one_cycle, load_config

ET = ZoneInfo("America/New_York")


def within_session_window(cfg: dict) -> bool:
    now_et = datetime.now(ET)
    start_h, start_m = map(int, cfg["session"]["start_time_et"].split(":"))
    end_h, end_m = map(int, cfg["session"]["end_time_et"].split(":"))
    start = now_et.replace(hour=start_h, minute=start_m, second=0, microsecond=0)
    end = now_et.replace(hour=end_h, minute=end_m, second=0, microsecond=0)
    return start <= now_et <= end


# ---------------------------------------------------------------
# These two helpers MUST compute the exact same path as
# alpaca_orders_mcp_server.py's _halt_path() (same log dir, same
# DDMMYYYY folder name) — that shared file is the whole point: it's how
# a trip here and a check inside the execution server agree on state
# without the two processes talking to each other.
# ---------------------------------------------------------------
def _halt_path(cfg: dict) -> Path:
    log_dir = Path(cfg["logs"]["run_dir"]).resolve()
    day_dir = log_dir / datetime.now().strftime("%d%m%Y")
    day_dir.mkdir(parents=True, exist_ok=True)
    return day_dir / ".trading_halted"


def already_halted(cfg: dict) -> tuple[bool, str]:
    path = _halt_path(cfg)
    if path.exists():
        return True, path.read_text(encoding="utf-8").strip()
    return False, ""


def trip_halt(cfg: dict, reason: str) -> None:
    path = _halt_path(cfg)
    if not path.exists():
        path.write_text(f"{datetime.now().isoformat()} :: {reason}", encoding="utf-8")


def check_daily_loss_stop(cfg: dict) -> bool:
    """
    Returns True if trading should continue. Checks the on-disk halt flag
    first (fast path, survives restarts); only if not already halted does
    it hit Alpaca's live account endpoint. If this check trips, it WRITES
    the halt file — every cycle for the rest of today (in this process AND
    the execution server) will see it as halted from then on.
    """
    halted, reason = already_halted(cfg)
    if halted:
        print(f"  Trading halted ({reason}). Skipping this cycle.")
        return False

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
        if last_equity <= 0:
            return True
        change_pct = (equity - last_equity) / last_equity * 100
        stop_pct = cfg["risk"]["daily_loss_stop_pct"]
        if change_pct <= -abs(stop_pct):
            reason = f"daily loss stop tripped at {change_pct:.2f}% (limit -{stop_pct}%)"
            print(f"  !! {reason.upper()}. Halting new positions for the rest of today.")
            trip_halt(cfg, reason)
            return False
        return True
    except Exception as exc:
        # Fail SAFE: if we can't verify the account is fine, don't trade.
        print(f"  WARNING: could not check daily loss stop ({exc}). Treating as NOT safe to trade this cycle.")
        return False


def run_gated_cycle(config_path: str, cfg: dict) -> None:
    """One cycle, always behind the halt check — this is what both --once
    and the main loop call, so there's exactly one code path, not two."""
    if not check_daily_loss_stop(cfg):
        return
    run_one_cycle(config_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="../config/config.yaml")
    parser.add_argument("--once", action="store_true", help="Run a single gated cycle and exit, ignoring the time window (useful for testing).")
    args = parser.parse_args()

    cfg = load_config(args.config)
    interval_min = cfg["session"]["cycle_interval_minutes"]

    print(f"Session runner started for {cfg['student_name']}. "
          f"Window: {cfg['session']['start_time_et']}-{cfg['session']['end_time_et']} ET, "
          f"cycle every {interval_min} min.")

    if args.once:
        run_gated_cycle(args.config, cfg)
        return

    while True:
        now_et = datetime.now(ET)

        if not within_session_window(cfg):
            print(f"[{now_et.strftime('%H:%M:%S %Z')}] Outside trading window. Sleeping 5 min.")
            time.sleep(5 * 60)
            continue

        print(f"[{now_et.strftime('%H:%M:%S %Z')}] Running cycle...")
        try:
            run_gated_cycle(args.config, cfg)
        except Exception as exc:
            print(f"  !! Cycle failed: {exc}. Will retry next interval.")

        time.sleep(interval_min * 60)


if __name__ == "__main__":
    main()
