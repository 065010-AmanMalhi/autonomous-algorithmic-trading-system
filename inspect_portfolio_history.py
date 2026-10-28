from pathlib import Path
from datetime import datetime, timedelta, timezone
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetOrdersRequest
from alpaca.trading.enums import QueryOrderStatus

ENV_FILES = {
    "Aman": ".env",
    "Mayank": ".env.mayank",
    "Aditi": ".env.aditi",
}

def load_env(path):
    values = {}
    for line in Path(path).read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values

def first_value(env, names):
    return next((env[n] for n in names if env.get(n)), None)

key_names = ("APCA_API_KEY_ID", "ALPACA_API_KEY", "ALPACA_API_KEY_ID")
secret_names = ("APCA_API_SECRET_KEY", "ALPACA_SECRET_KEY", "ALPACA_API_SECRET_KEY")

for owner, filename in ENV_FILES.items():
    try:
        env = load_env(filename)
        key = first_value(env, key_names)
        secret = first_value(env, secret_names)
        if not key or not secret:
            print(f"\n{owner}: credentials missing in {filename}")
            continue

        client = TradingClient(key, secret, paper=True)
        account = client.get_account()
        request = GetOrdersRequest(
            status=QueryOrderStatus.ALL,
            after=datetime.now(timezone.utc) - timedelta(days=10),
            limit=500,
            nested=True,
        )
        orders = client.get_orders(filter=request)

        print(f"\n{owner}: account_status={account.status}, equity={account.equity}")
        print(f"Orders returned: {len(orders)}")
        for order in orders:
            print(
                f"{order.submitted_at} | {order.symbol} | {order.side} | "
                f"status={order.status} | qty={order.qty} | "
                f"filled_qty={order.filled_qty} | "
                f"filled_avg_price={order.filled_avg_price}"
            )
    except Exception as exc:
        print(f"\n{owner}: {type(exc).__name__}: {exc}")
