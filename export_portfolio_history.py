from pathlib import Path
from datetime import datetime, timedelta, timezone
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetPortfolioHistoryRequest
import csv

ENV_FILES = {
    "aman": ".env",
    "mayank": ".env.mayank",
    "aditi": ".env.aditi",
}

def load_env(filename):
    env = {}
    for line in Path(filename).read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            env[k.strip()] = v.strip().strip('"').strip("'")
    return env

def pick(env, names):
    return next((env[n] for n in names if env.get(n)), None)

out = Path("metrics_data")
out.mkdir(exist_ok=True)

end = datetime.now(timezone.utc)
start = end - timedelta(days=365)

for owner, env_file in ENV_FILES.items():
    try:
        env = load_env(env_file)
        key = pick(env, ["APCA_API_KEY_ID", "ALPACA_API_KEY", "ALPACA_API_KEY_ID"])
        secret = pick(env, ["APCA_API_SECRET_KEY", "ALPACA_SECRET_KEY", "ALPACA_API_SECRET_KEY"])

        if not key or not secret:
            print(f"{owner}: credentials missing")
            continue

        client = TradingClient(key, secret, paper=True)
        request = GetPortfolioHistoryRequest(
            start=start,
            end=end,
            timeframe="1D",
            extended_hours=False,
        )
        history = client.get_portfolio_history(request)

        timestamps = history.timestamp or []
        equities = history.equity or []
        pnls = history.profit_loss or []
        pnl_pcts = history.profit_loss_pct or []

        raw_file = out / f"{owner}_portfolio_history_raw.csv"
        valid_file = out / f"{owner}_portfolio_history_valid.csv"

        rows = []
        for i, ts in enumerate(timestamps):
            equity = equities[i] if i < len(equities) else None
            pnl = pnls[i] if i < len(pnls) else None
            pnl_pct = pnl_pcts[i] if i < len(pnl_pcts) else None
            rows.append([ts, equity, pnl, pnl_pct])

        for filename, selected in (
            (raw_file, rows),
            (valid_file, [r for r in rows if r[1] is not None and float(r[1]) > 0]),
        ):
            with filename.open("w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["timestamp", "equity", "profit_loss", "profit_loss_pct"])
                writer.writerows(selected)

        positive = [r for r in rows if r[1] is not None and float(r[1]) > 0]
        print(
            f"{owner}: raw={len(rows)}, positive={len(positive)}, "
            f"first_valid={positive[0][0] if positive else 'none'}, "
            f"last_valid={positive[-1][0] if positive else 'none'}"
        )

    except Exception as exc:
        print(f"{owner}: {type(exc).__name__}: {exc}")

print(f"Files saved in: {out.resolve()}")
