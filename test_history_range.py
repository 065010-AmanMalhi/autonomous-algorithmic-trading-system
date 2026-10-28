from pathlib import Path
from datetime import datetime, timedelta, timezone
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetPortfolioHistoryRequest

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

env = load_env(".env")
key = pick(env, ["APCA_API_KEY_ID", "ALPACA_API_KEY", "ALPACA_API_KEY_ID"])
secret = pick(env, ["APCA_API_SECRET_KEY", "ALPACA_SECRET_KEY", "ALPACA_API_SECRET_KEY"])

if not key or not secret:
    raise SystemExit("Could not find Alpaca credentials in .env")

client = TradingClient(key, secret, paper=True)
end = datetime.now(timezone.utc)
start = end - timedelta(days=365)

request = GetPortfolioHistoryRequest(
    start=start,
    end=end,
    timeframe="1D",
    extended_hours=False,
)

history = client.get_portfolio_history(request)
timestamps = history.timestamp or []
equities = history.equity or []

print("Points returned:", len(timestamps))
print("First 10 observations:")
for ts, equity in list(zip(timestamps, equities))[:10]:
    print(ts, equity)

print("Last 10 observations:")
for ts, equity in list(zip(timestamps, equities))[-10:]:
    print(ts, equity)

positive = [
    (ts, equity) for ts, equity in zip(timestamps, equities)
    if equity is not None and equity > 0
]
print("Positive-equity observations:", len(positive))
if positive:
    print("First positive:", positive[0])
    print("Last positive:", positive[-1])
