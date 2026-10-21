import argparse
import asyncio
import os
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import requests
import yaml
from dotenv import load_dotenv
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetCalendarRequest

ET = ZoneInfo("America/New_York")
ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env", override=False)


def load_session_config(path):
    config_path = Path(path)
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    config_path = config_path.resolve()

    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if not isinstance(cfg, dict):
        raise ValueError("Config must contain a YAML mapping.")

    for section in ("portfolio", "risk", "session", "logs"):
        if not isinstance(cfg.get(section), dict):
            raise ValueError(f"Missing or invalid config section: {section}")

    stocks = cfg["portfolio"].get("stocks", [])
    if not stocks:
        raise ValueError("No portfolio stocks configured.")

    symbols = [
        item.get("symbol", "").strip().upper()
        for item in stocks
        if isinstance(item, dict) and item.get("symbol", "").strip()
    ]
    if len(symbols) != len(stocks) or len(set(symbols)) != len(symbols):
        raise ValueError("Portfolio contains invalid or duplicate symbols.")

    start = datetime.strptime(cfg["session"]["start_time_et"], "%H:%M").time()
    end = datetime.strptime(cfg["session"]["end_time_et"], "%H:%M").time()
    interval = int(cfg["session"]["cycle_interval_minutes"])
    if start >= end or interval < 1:
        raise ValueError("Invalid session window or cycle interval.")

    cfg["_config_path"] = str(config_path)
    cfg["_symbols"] = symbols
    return cfg


def make_client():
    key = os.environ.get("ALPACA_API_KEY", "").strip()
    secret = os.environ.get("ALPACA_SECRET_KEY", "").strip()
    if not key or not secret:
        raise RuntimeError("Alpaca paper API credentials are missing from this terminal.")
    return TradingClient(key, secret, paper=True)


def market_status(cfg):
    client = make_client()
    now = datetime.now(ET)
    clock = client.get_clock()

    # Verify the broker's exchange calendar for today's session.
    sessions = client.get_calendar(GetCalendarRequest(start=now.date(), end=now.date()))
    if not sessions:
        return False, "Alpaca calendar reports no session today."

    start = datetime.strptime(cfg["session"]["start_time_et"], "%H:%M").time()
    end = datetime.strptime(cfg["session"]["end_time_et"], "%H:%M").time()
    in_configured_window = start <= now.time().replace(tzinfo=None) < end

    if not clock.is_open:
        return False, "Alpaca reports the market is closed."
    if not in_configured_window:
        return False, "Outside the configured trading window."

    return True, f"Market open; configured window active ({now:%Y-%m-%d %H:%M:%S %Z})."


def check_daily_loss(cfg):
    # Keep the same local-date convention as the existing order server's
    # halt-file implementation so both processes see the same latch.
    log_dir = Path(cfg["logs"]["run_dir"])
    if not log_dir.is_absolute():
        log_dir = ROOT / log_dir
    halt_path = log_dir / datetime.now().strftime("%d%m%Y") / ".trading_halted"

    if halt_path.exists():
        print(f"HALTED: {halt_path.read_text(encoding='utf-8').strip()}")
        return False

    key = os.environ.get("ALPACA_API_KEY", "").strip()
    secret = os.environ.get("ALPACA_SECRET_KEY", "").strip()
    if not key or not secret:
        print("Risk check failed: Alpaca credentials are missing.")
        return False

    try:
        response = requests.get(
            "https://paper-api.alpaca.markets/v2/account",
            headers={
                "APCA-API-KEY-ID": key,
                "APCA-API-SECRET-KEY": secret,
            },
            timeout=10,
        )
        response.raise_for_status()
        account = response.json()
        equity = float(account["equity"])
        prior_equity = float(account["last_equity"])
        stop_pct = float(cfg["risk"]["daily_loss_stop_pct"])

        if not all(map(lambda x: x == x and abs(x) != float("inf"),
                       (equity, prior_equity, stop_pct))):
            raise ValueError("Non-finite equity or risk limit.")
        if prior_equity <= 0 or equity <= 0 or stop_pct <= 0:
            raise ValueError("Invalid equity baseline or daily-loss limit.")

        change_pct = (equity - prior_equity) / prior_equity * 100
        if change_pct <= -abs(stop_pct):
            halt_path.parent.mkdir(parents=True, exist_ok=True)
            halt_path.write_text(
                f"{datetime.now().isoformat()} :: daily loss stop tripped "
                f"at {change_pct:.2f}% (limit -{stop_pct}%)",
                encoding="utf-8",
            )
            print(f"Daily loss stop tripped: {change_pct:.2f}%.")
            return False

        return True
    except Exception as exc:
        print(f"Risk check failed; skipping cycle: {exc}")
        return False


def run_cycle(config_path):
    # Import the existing CrewAI workflow only when a trading cycle is
    # explicitly enabled. This runner does not replace the crew.
    import sys
    sys.path.insert(0, str(ROOT / "crew"))
    from trading_crew import run_one_cycle

    run_one_cycle(config_path)


async def main():
    parser = argparse.ArgumentParser(
        description="Async scheduler for the existing Alpaca paper-trading crew."
    )
    parser.add_argument("--config", default="config/aman.yaml")
    parser.add_argument(
        "--trade",
        action="store_true",
        help="Enable order-capable CrewAI cycles. Without this flag, only checks status.",
    )
    args = parser.parse_args()

    cfg = load_session_config(args.config)
    config_path = cfg["_config_path"]
    print(f"Portfolio: {len(cfg['_symbols'])} stocks: {', '.join(cfg['_symbols'])}")
    print(
        f"Configured window: {cfg['session']['start_time_et']}-"
        f"{cfg['session']['end_time_et']} ET; "
        f"interval: {cfg['session']['cycle_interval_minutes']} minutes"
    )

    try:
        ok, reason = await asyncio.to_thread(market_status, cfg)
        print(f"Market check: {reason}")
    except Exception as exc:
        ok, reason = False, f"Could not verify Alpaca market status: {exc}"
        print(reason)

    if not args.trade:
        print("CHECK-ONLY mode. No trading cycle was started.")
        return

    print("Paper-trading loop enabled. Press Ctrl+C to stop scheduling new cycles.")
    interval = cfg["session"]["cycle_interval_minutes"] * 60

    while True:
        try:
            ok, reason = await asyncio.to_thread(market_status, cfg)
            if not ok:
                print(f"[{datetime.now(ET):%H:%M:%S %Z}] Skipping: {reason}")
                await asyncio.sleep(60)
                continue

            if not await asyncio.to_thread(check_daily_loss, cfg):
                print("Risk gate blocked this cycle; checking again in 60 seconds.")
                await asyncio.sleep(60)
                continue

            print(f"[{datetime.now(ET):%H:%M:%S %Z}] Starting portfolio cycle.")
            # One worker at a time; this await prevents overlapping cycles.
            await asyncio.to_thread(run_cycle, config_path)
            print("Portfolio cycle finished.")
            await asyncio.sleep(interval)

        except asyncio.CancelledError:
            raise
        except KeyboardInterrupt:
            print("Stop requested.")
            break
        except Exception as exc:
            print(f"Cycle/scheduler error; no overlapping retry: {exc}")
            await asyncio.sleep(60)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Scheduler stopped.")



