"""
alpaca_orders_mcp_server.py
-----------------------------
Local MCP server #2: Alpaca PAPER trading execution — equities + options.

Extends the class starter file (alpaca_trading_mcpServer.py) with:
  1. place_portfolio_orders — submit a whole batch of buy/sell/hold
     decisions for the portfolio in one tool call (the execution agent
     passes a list, not one call per symbol).
  2. buy_option_hedge / close_option_position — single-leg options orders
     for the tail-hedge overlay (SPY/QQQ puts).
  3. get_portfolio_beta_exposure helper stub left for the crew to call via
     the OTHER server's calculate_beta per symbol and sum weighted.

Requires: pip install mcp alpaca-py
Env vars: ALPACA_API_KEY, ALPACA_SECRET_KEY  (your PAPER keys — never the
ones hardcoded in the class file comment; generate your own.)
"""

import os

from mcp.server.fastmcp import FastMCP
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, GetOptionContractsRequest
from alpaca.trading.enums import OrderSide, TimeInForce, AssetClass

mcp = FastMCP("AlpacaOrders")

ALPACA_API_KEY = os.environ.get("ALPACA_API_KEY", "")
ALPACA_SECRET_KEY = os.environ.get("ALPACA_SECRET_KEY", "")

# paper=True pins every call in this file to Alpaca's paper-trading
# endpoint — fake money, real order-routing behavior. Do not flip this to
# False without a very deliberate conversation with yourself first.
_trading_client = TradingClient(ALPACA_API_KEY, ALPACA_SECRET_KEY, paper=True)


def _order_to_dict(order) -> dict:
    return {
        "order_id": str(order.id),
        "symbol": order.symbol,
        "qty": str(order.qty) if order.qty else None,
        "notional": str(order.notional) if getattr(order, "notional", None) else None,
        "side": str(order.side),
        "status": str(order.status),
        "submitted_at": str(order.submitted_at),
    }


# ============================================================
# Equities — single order (kept from the class starter, useful for
# manual testing / the human-in-the-loop CLI clients you were given)
# ============================================================
@mcp.tool()
def buy_stock(symbol: str, qty: float = None, notional: float = None) -> dict:
    """
    Place a PAPER-trading market BUY order. Pass exactly one of qty or
    notional (notional = dollar amount, supports fractional shares).
    """
    if (qty is None) == (notional is None):
        return {"error": "Pass exactly one of qty or notional, not both/neither."}
    order_request = MarketOrderRequest(
        symbol=symbol.upper(),
        qty=qty,
        notional=notional,
        side=OrderSide.BUY,
        time_in_force=TimeInForce.DAY,
    )
    order = _trading_client.submit_order(order_request)
    return _order_to_dict(order)


@mcp.tool()
def sell_stock(symbol: str, qty: float = None, notional: float = None) -> dict:
    """
    Place a PAPER-trading market SELL order. Pass exactly one of qty or
    notional.
    """
    if (qty is None) == (notional is None):
        return {"error": "Pass exactly one of qty or notional, not both/neither."}
    order_request = MarketOrderRequest(
        symbol=symbol.upper(),
        qty=qty,
        notional=notional,
        side=OrderSide.SELL,
        time_in_force=TimeInForce.DAY,
    )
    order = _trading_client.submit_order(order_request)
    return _order_to_dict(order)


# ============================================================
# Equities — WHOLE-PORTFOLIO batch order (the brief/your requirement:
# trade the whole portfolio at once, not one call at a time per symbol)
# ============================================================
@mcp.tool()
def place_portfolio_orders(decisions: list[dict]) -> dict:
    """
    Submit a batch of buy/sell/hold decisions for the portfolio in one
    tool call. This is the execution agent's main tool — it should pass
    the FULL set of per-symbol decisions from the decision-maker agent at
    once, not call buy_stock/sell_stock symbol-by-symbol.

    Parameters
    ----------
    decisions : list of dicts, each:
        {
          "symbol": "NVDA",
          "action": "buy" | "sell" | "hold",
          "notional": 4500.0        # dollar amount (omit/ignore if action is "hold")
        }

    Returns
    -------
    dict with one result entry per decision — order confirmation for
    buy/sell, or a "skipped (hold)" note.

    Orders are submitted one-by-one under the hood (Alpaca's REST API has
    no native batch-order endpoint), but from the agent's side this is a
    single call, which is what matters for keeping the crew's tool-call
    count and rate-limit exposure down versus 15 separate agent calls.
    """
    results = []
    for d in decisions:
        symbol = d.get("symbol", "").upper()
        action = d.get("action", "hold").lower()

        if action == "hold" or not symbol:
            results.append({"symbol": symbol, "action": "hold", "result": "skipped (hold)"})
            continue

        notional = d.get("notional")
        if not notional or notional <= 0:
            results.append({"symbol": symbol, "action": action, "result": "error: missing/invalid notional"})
            continue

        try:
            order_request = MarketOrderRequest(
                symbol=symbol,
                notional=notional,
                side=OrderSide.BUY if action == "buy" else OrderSide.SELL,
                time_in_force=TimeInForce.DAY,
            )
            order = _trading_client.submit_order(order_request)
            results.append({"symbol": symbol, "action": action, "result": _order_to_dict(order)})
        except Exception as exc:
            results.append({"symbol": symbol, "action": action, "result": f"error: {exc}"})

    return {"orders_submitted": len(results), "results": results}


# ============================================================
# Options — tail hedge (requires options trading enabled on the account)
# ============================================================
@mcp.tool()
def buy_option_hedge(option_symbol: str, contracts: int) -> dict:
    """
    Buy `contracts` of a single-leg option (long put for a tail hedge, or
    long call) by its OCC symbol — get this symbol from get_option_chain
    in technical_indicators_mcp_server.py first.

    Each contract = 100 shares of underlying exposure. Long options are
    risk-defined (max loss = premium paid), which is why this stays a
    single buy_order_request rather than anything requiring margin
    approval beyond basic options trading.

    Parameters
    ----------
    option_symbol : OCC-format contract symbol, e.g. "SPY251121P00580000".
    contracts     : Number of contracts to buy (whole number).
    """
    try:
        order_request = MarketOrderRequest(
            symbol=option_symbol,
            qty=contracts,
            side=OrderSide.BUY,
            time_in_force=TimeInForce.DAY,
        )
        order = _trading_client.submit_order(order_request)
        return _order_to_dict(order)
    except Exception as exc:
        return {
            "error": str(exc),
            "note": (
                "If this fails, confirm options trading is enabled on your "
                "paper account (Alpaca dashboard > paper account > options "
                "approval level) and that the OCC symbol is correct."
            ),
        }


@mcp.tool()
def close_option_position(option_symbol: str) -> dict:
    """Close an existing open option position (sell-to-close) by OCC symbol."""
    try:
        result = _trading_client.close_position(option_symbol)
        return {"symbol": option_symbol, "status": "close order submitted", "detail": str(result)}
    except Exception as exc:
        return {"error": str(exc)}


# ============================================================
# Account / positions (kept from class starter, extended to list ALL
# positions at once instead of one symbol at a time)
# ============================================================
@mcp.tool()
def get_all_positions() -> dict:
    """List every open position (equities + options) in the paper account."""
    positions = _trading_client.get_all_positions()
    return {
        "count": len(positions),
        "positions": [
            {
                "symbol": p.symbol,
                "asset_class": str(p.asset_class),
                "qty": str(p.qty),
                "market_value": str(p.market_value),
                "unrealized_pl": str(p.unrealized_pl),
                "current_price": str(p.current_price),
            }
            for p in positions
        ],
    }


@mcp.tool()
def get_open_orders(symbol: str = "") -> dict:
    """List currently open (unfilled) orders, optionally filtered by symbol."""
    orders = _trading_client.get_orders()
    if symbol:
        orders = [o for o in orders if o.symbol.upper() == symbol.upper()]
    return {
        "count": len(orders),
        "orders": [
            {
                "order_id": str(o.id),
                "symbol": o.symbol,
                "qty": str(o.qty),
                "side": str(o.side),
                "status": str(o.status),
                "submitted_at": str(o.submitted_at),
            }
            for o in orders
        ],
    }


@mcp.tool()
def cancel_all_orders(symbol: str = "") -> dict:
    """Cancel all open orders, optionally restricted to one symbol."""
    if symbol:
        orders = _trading_client.get_orders()
        matching = [o for o in orders if o.symbol.upper() == symbol.upper()]
        for o in matching:
            _trading_client.cancel_order_by_id(o.id)
        return {"cancelled_count": len(matching), "symbol": symbol.upper()}
    _trading_client.cancel_orders()
    return {"cancelled": "all open orders"}


@mcp.tool()
def get_account_info() -> dict:
    """Get current paper-trading account cash, buying power, and portfolio value."""
    acct = _trading_client.get_account()
    return {
        "cash": str(acct.cash),
        "buying_power": str(acct.buying_power),
        "portfolio_value": str(acct.portfolio_value),
        "trading_blocked": acct.trading_blocked,
        "account_blocked": acct.account_blocked,
    }


if __name__ == "__main__":
    mcp.run(transport="stdio")
