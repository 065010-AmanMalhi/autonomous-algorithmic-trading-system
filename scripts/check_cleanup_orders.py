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

targets = {"CVX", "MA", "META"}
orders = client.get_orders(
    filter=GetOrdersRequest(status=QueryOrderStatus.OPEN)
)

for order in orders:
    if order.symbol in targets:
        print(
            f"symbol={order.symbol} side={order.side} "
            f"qty={order.qty} status={order.status} id={order.id}"
        )
