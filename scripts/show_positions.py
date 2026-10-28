import os
from pathlib import Path
from dotenv import load_dotenv
from alpaca.trading.client import TradingClient

# Load local environment variables without printing secrets.
load_dotenv()
key = os.getenv("ALPACA_API_KEY")
secret = os.getenv("ALPACA_SECRET_KEY")

if not key or not secret:
    raise SystemExit("Alpaca credentials missing from environment/.env.")

client = TradingClient(key, secret, paper=True)
account = client.get_account()
positions = client.get_all_positions()

print("ACCOUNT STATUS:", account.status)
print("PAPER ACCOUNT: True")
print("POSITIONS:", len(positions))
print("-" * 90)
print(f"{'SYMBOL':<12}{'QTY':>12}{'SIDE':>10}{'MARKET VALUE':>20}{'UNREALIZED P/L':>20}")

for p in positions:
    print(
        f"{p.symbol:<12}{str(p.qty):>12}{str(p.side):>10}"
        f"{float(p.market_value):>20,.2f}"
        f"{float(p.unrealized_pl):>20,.2f}"
    )

print("-" * 90)
