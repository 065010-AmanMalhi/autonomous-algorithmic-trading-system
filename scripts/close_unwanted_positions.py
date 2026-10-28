import os
from dotenv import load_dotenv
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetOrdersRequest
from alpaca.trading.enums import QueryOrderStatus

load_dotenv()

client = TradingClient(
    os.environ["ALPACA_API_KEY"],
    os.environ["ALPACA_SECRET_KEY"],
    paper=True,
)

EXPECTED_ACCOUNT_ID = "5ec8611a-ef81-4079-ae4b-40a120e425bb"
targets = ["CVX", "MA", "META"]

account = client.get_account()
if str(account.id) != EXPECTED_ACCOUNT_ID:
    raise SystemExit("ABORTED: account ID does not match the expected paper account.")
if str(account.status) != "AccountStatus.ACTIVE":
    raise SystemExit(f"ABORTED: account status is {account.status}.")
if account.trading_blocked:
    raise SystemExit("ABORTED: trading is blocked.")

positions = {p.symbol.upper(): p for p in client.get_all_positions()}
orders = client.get_orders(
    filter=GetOrdersRequest(status=QueryOrderStatus.OPEN)
)
open_targets = [o for o in orders if o.symbol.upper() in targets]

if open_targets:
    print("ABORTED: existing open orders found for target symbols:")
    for o in open_targets:
        print(f"{o.symbol}: side={o.side}, qty={o.qty}, status={o.status}, id={o.id}")
    raise SystemExit("Resolve these orders first; no close orders were submitted.")

print("Verified Alpaca PAPER account:", account.id)
print("\nPositions to close:")
for symbol in targets:
    p = positions.get(symbol)
    if p:
        print(f"{symbol}: qty={p.qty}, market_value=${float(p.market_value):,.2f}")
    else:
        print(f"{symbol}: no position found")

present = [s for s in targets if s in positions]
if not present:
    raise SystemExit("No target positions found; nothing to close.")

confirmation = input(
    "\nType CLOSE CVX MA META to submit paper-market close orders: "
).strip()
if confirmation != "CLOSE CVX MA META":
    raise SystemExit("Cancelled. No orders submitted.")

for symbol in present:
    try:
        result = client.close_position(symbol)
        print(f"SUBMITTED {symbol}: order_id={result.id}, status={result.status}")
    except Exception as exc:
        print(f"FAILED {symbol}: {type(exc).__name__}: {exc}")
