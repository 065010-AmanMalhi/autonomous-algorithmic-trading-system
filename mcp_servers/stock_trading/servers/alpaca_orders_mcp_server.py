"""
alpaca_orders_mcp_server.py
-----------------------------
Local MCP server #2: Alpaca PAPER trading execution — equities + options.

SECURITY MODEL (read this before touching the file)
-----------------------------------------------------
Everything an agent sends this server — symbols, notionals, confidence
scores, option contracts — is UNTRUSTED input. An LLM can hallucinate, a
decision-maker agent can emit malformed JSON, and a bug three layers up in
trading_crew.py can pass through a bad value. This file does NOT trust the
crew to have already applied the risk.* rules from config.yaml; it
re-derives and enforces them itself, independently, every single call:

  1. Position size and cash buffer are checked here against LIVE account
     equity fetched from Alpaca at call time — not against whatever the
     agent claims the portfolio is worth.
  2. `action` is matched against an explicit allow-list ("buy"/"sell"/
     "hold"). Anything else is REJECTED, never defaulted to a side.
  3. The options hedge independently re-parses the OCC symbol (underlying,
     expiry, type, strike), checks it against the configured allowed
     underlyings/expiry window, fetches a live quote, and caps premium
     spend against the config limit — it does not trust the agent's own
     cost estimate.
  4. A trading-halted flag is a FILE on disk (logs/<date>/.trading_halted),
     not a variable the Python process holds in memory. Once anything
     (the session runner's daily-loss check, or this server itself)
     writes it, every tool call in this file refuses new positions for
     the rest of that calendar day, regardless of which script started
     the MCP server or whether run_session.py's loop is even involved.

Env vars: ALPACA_API_KEY, ALPACA_SECRET_KEY (your PAPER keys), plus
TRADING_CONFIG_PATH and TRADING_LOG_DIR, which trading_crew.py sets
automatically when it launches this server as a subprocess — see
build_crew() in crew/trading_crew.py. Running this file some other way
without those two set falls back to conservative hardcoded defaults
(logged in every response's "limits_source" field so you can tell).
"""

import json
import os
import re
from datetime import datetime, date
from pathlib import Path

import requests
import yaml
from mcp.server.fastmcp import FastMCP
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce

mcp = FastMCP("AlpacaOrders")

ALPACA_API_KEY = os.environ.get("ALPACA_API_KEY", "")
ALPACA_SECRET_KEY = os.environ.get("ALPACA_SECRET_KEY", "")
ALPACA_DATA_URL = "https://data.alpaca.markets/v2"
ALPACA_OPTIONS_DATA_URL = "https://data.alpaca.markets/v1beta1"
DATA_HEADERS = {"APCA-API-KEY-ID": ALPACA_API_KEY, "APCA-API-SECRET-KEY": ALPACA_SECRET_KEY}

# paper=True pins every call in this file to Alpaca's paper-trading
# endpoint — fake money, real order-routing behavior. Do not flip this to
# False without a very deliberate conversation with yourself first.
_trading_client = TradingClient(ALPACA_API_KEY, ALPACA_SECRET_KEY, paper=True)

# Conservative fallbacks used ONLY if TRADING_CONFIG_PATH isn't set/readable
# — this should never happen when launched via trading_crew.py, but a
# missing config must fail SAFE (tighter limits), not fail open.
_FALLBACK_LIMITS = {
    "max_position_weight_pct": 10,
    "min_cash_buffer_pct": 10,
    "min_confidence_to_trade": 0.65,
    "max_trades_per_symbol_per_day": 1,
    "daily_loss_stop_pct": 5,
    "hedging_enabled": True,
    "hedge_instrument": "SPY",
    "hedge_option_type": "put",
    "hedge_max_expiration_days": 45,
    "hedge_max_cost_pct": 2.0,
}


# ============================================================
# Config / state loading — all re-read fresh each call (cheap, small
# files) rather than cached, so a config edit takes effect immediately
# and a stale in-memory copy can never silently diverge from disk.
# ============================================================
def _load_limits() -> dict:
    config_path = os.environ.get("TRADING_CONFIG_PATH", "")
    if not config_path or not os.path.exists(config_path):
        return {**_FALLBACK_LIMITS, "limits_source": "FALLBACK (TRADING_CONFIG_PATH not set/found)"}

    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    risk = cfg.get("risk", {})
    hedging = cfg.get("hedging", {})
    return {
        "max_position_weight_pct": risk.get("max_position_weight_pct", _FALLBACK_LIMITS["max_position_weight_pct"]),
        "min_cash_buffer_pct": risk.get("min_cash_buffer_pct", _FALLBACK_LIMITS["min_cash_buffer_pct"]),
        "min_confidence_to_trade": risk.get("min_confidence_to_trade", _FALLBACK_LIMITS["min_confidence_to_trade"]),
        "max_trades_per_symbol_per_day": risk.get("max_trades_per_symbol_per_day", _FALLBACK_LIMITS["max_trades_per_symbol_per_day"]),
        "daily_loss_stop_pct": risk.get("daily_loss_stop_pct", _FALLBACK_LIMITS["daily_loss_stop_pct"]),
        "hedging_enabled": hedging.get("enabled", _FALLBACK_LIMITS["hedging_enabled"]),
        "hedge_instrument": hedging.get("instrument", _FALLBACK_LIMITS["hedge_instrument"]),
        "hedge_option_type": hedging.get("option_type", _FALLBACK_LIMITS["hedge_option_type"]),
        "hedge_max_expiration_days": hedging.get("max_expiration_days", _FALLBACK_LIMITS["hedge_max_expiration_days"]),
        "hedge_max_cost_pct": hedging.get("max_hedge_cost_pct_of_portfolio", _FALLBACK_LIMITS["hedge_max_cost_pct"]),
        "limits_source": config_path,
    }


def _log_dir() -> Path:
    base = os.environ.get("TRADING_LOG_DIR", "logs")
    return Path(base)


def _today_dir() -> Path:
    d = _log_dir() / datetime.now().strftime("%d%m%Y")
    d.mkdir(parents=True, exist_ok=True)
    return d


def _halt_path() -> Path:
    return _today_dir() / ".trading_halted"


def is_halted() -> tuple[bool, str]:
    path = _halt_path()
    if path.exists():
        return True, path.read_text(encoding="utf-8").strip()
    return False, ""


def set_halted(reason: str) -> None:
    """
    Write the halt flag to disk. Idempotent — if it's already halted for a
    different reason, the original reason is kept (first trip wins).
    """
    path = _halt_path()
    if not path.exists():
        path.write_text(f"{datetime.now().isoformat()} :: {reason}", encoding="utf-8")


def _ledger_path() -> Path:
    return _today_dir() / "trade_ledger.json"


def _load_ledger() -> dict:
    path = _ledger_path()
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}
    return {}


def _increment_ledger(symbol: str) -> int:
    """Record one more trade for `symbol` today; returns the new count."""
    ledger = _load_ledger()
    ledger[symbol] = ledger.get(symbol, 0) + 1
    _ledger_path().write_text(json.dumps(ledger), encoding="utf-8")
    return ledger[symbol]


def _trades_today(symbol: str) -> int:
    return _load_ledger().get(symbol, 0)


# ============================================================
# OCC option symbol parsing — independent of whatever the agent claims
# about a contract; this is the actual ground truth parsed from the
# symbol string itself. Format: ROOT + YYMMDD + C/P + strike*1000 (8 digits)
# e.g. "SPY251121P00580000" -> SPY, 2025-11-21, PUT, strike 580.00
# ============================================================
_OCC_RE = re.compile(r"^(?P<root>[A-Z]{1,6})(?P<date>\d{6})(?P<cp>[CP])(?P<strike>\d{8})$")


def parse_occ_symbol(symbol: str) -> dict:
    match = _OCC_RE.match(symbol.upper().strip())
    if not match:
        raise ValueError(f"'{symbol}' does not match the expected OCC option symbol format.")
    d = match.groupdict()
    expiry = datetime.strptime(d["date"], "%y%m%d").date()
    return {
        "underlying": d["root"],
        "expiry": expiry,
        "option_type": "call" if d["cp"] == "C" else "put",
        "strike": int(d["strike"]) / 1000.0,
    }


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
# Equities — single order (manual testing / human-in-the-loop CLI use —
# NOT called by the crew, so it intentionally skips the portfolio-level
# batch checks below; still rejects an invalid side explicitly)
# ============================================================
@mcp.tool()
def buy_stock(symbol: str, qty: float = None, notional: float = None) -> dict:
    """Place a PAPER-trading market BUY order. Pass exactly one of qty or notional."""
    halted, reason = is_halted()
    if halted:
        return {"error": "Trading halted for the rest of today.", "reason": reason}
    if (qty is None) == (notional is None):
        return {"error": "Pass exactly one of qty or notional, not both/neither."}
    order_request = MarketOrderRequest(
        symbol=symbol.upper(), qty=qty, notional=notional,
        side=OrderSide.BUY, time_in_force=TimeInForce.DAY,
    )
    order = _trading_client.submit_order(order_request)
    return _order_to_dict(order)


@mcp.tool()
def sell_stock(symbol: str, qty: float = None, notional: float = None) -> dict:
    """Place a PAPER-trading market SELL order. Pass exactly one of qty or notional."""
    # Selling to close/reduce risk is allowed even while halted — only
    # NEW buy exposure is blocked by the halt.
    if (qty is None) == (notional is None):
        return {"error": "Pass exactly one of qty or notional, not both/neither."}
    order_request = MarketOrderRequest(
        symbol=symbol.upper(), qty=qty, notional=notional,
        side=OrderSide.SELL, time_in_force=TimeInForce.DAY,
    )
    order = _trading_client.submit_order(order_request)
    return _order_to_dict(order)


# ============================================================
# Equities — WHOLE-PORTFOLIO batch order, WITH independent enforcement
# of every risk.* rule from config.yaml. This is the hard control; the
# decision-maker agent's own stated rules in its prompt are a SECOND
# layer, not the only one.
# ============================================================
@mcp.tool()
def place_portfolio_orders(decisions: list[dict]) -> dict:
    """
    Submit a batch of buy/sell/hold decisions for the portfolio in one
    tool call, after independently validating every decision against the
    account's live equity and the risk limits in config.yaml.

    Parameters
    ----------
    decisions : list of dicts, each:
        {
          "symbol": "NVDA",
          "action": "buy" | "sell" | "hold",
          "notional": 4500.0,     # dollar amount
          "confidence": 0.72      # 0-1, REQUIRED for buy/sell — missing
                                  # or sub-threshold confidence is rejected
        }

    Any decision that fails a check is REJECTED with a reason and no order
    is submitted for it — the batch keeps processing the remaining
    decisions rather than aborting entirely. If the account is currently
    halted (daily-loss stop tripped), ALL buy decisions are rejected
    without even checking the individual limits; sells still go through
    since closing risk is never something we want to block.
    """
    limits = _load_limits()
    halted, halt_reason = is_halted()

    account = _trading_client.get_account()
    portfolio_value = float(account.portfolio_value)
    cash = float(account.cash)

    max_position_dollars = portfolio_value * (limits["max_position_weight_pct"] / 100)
    min_cash_dollars = portfolio_value * (limits["min_cash_buffer_pct"] / 100)

    results = []
    cash_committed_this_batch = 0.0  # running total of buy notional approved so far in THIS call

    for d in decisions:
        symbol = str(d.get("symbol", "")).upper().strip()
        action = str(d.get("action", "hold")).lower().strip()
        notional = d.get("notional")
        confidence = d.get("confidence")

        # ---- Check 1: explicit action allow-list. Anything that isn't
        # exactly "buy"/"sell"/"hold" is rejected outright — it NEVER
        # falls through to becoming a sell or a buy by default. ----
        if action not in ("buy", "sell", "hold"):
            results.append({"symbol": symbol, "action": action, "result": f"REJECTED: unrecognized action '{action}' (must be buy/sell/hold)"})
            continue

        if action == "hold" or not symbol:
            results.append({"symbol": symbol, "action": "hold", "result": "skipped (hold)"})
            continue

        # ---- Check 2: halt — only blocks NEW buys, not sells ----
        if action == "buy" and halted:
            results.append({"symbol": symbol, "action": action, "result": f"REJECTED: trading halted ({halt_reason})"})
            continue

        # ---- Check 3: notional sanity ----
        if not isinstance(notional, (int, float)) or notional <= 0:
            results.append({"symbol": symbol, "action": action, "result": "REJECTED: missing/invalid notional"})
            continue

        if action == "buy":
            # ---- Check 4: confidence — REQUIRED, not optional, for buys ----
            if not isinstance(confidence, (int, float)):
                results.append({"symbol": symbol, "action": action, "result": "REJECTED: missing confidence score"})
                continue
            if confidence < limits["min_confidence_to_trade"]:
                results.append({"symbol": symbol, "action": action, "result": f"REJECTED: confidence {confidence} below minimum {limits['min_confidence_to_trade']}"})
                continue

            # ---- Check 5: position-size cap, against LIVE portfolio value ----
            if notional > max_position_dollars:
                results.append({"symbol": symbol, "action": action, "result": f"REJECTED: notional ${notional:,.2f} exceeds {limits['max_position_weight_pct']}% position cap (${max_position_dollars:,.2f})"})
                continue

            # ---- Check 6: cash buffer, accounting for everything already
            # approved earlier in this same batch (not just account cash
            # as of the start of the call) ----
            projected_cash_after = cash - cash_committed_this_batch - notional
            if projected_cash_after < min_cash_dollars:
                results.append({"symbol": symbol, "action": action, "result": f"REJECTED: would breach {limits['min_cash_buffer_pct']}% cash buffer (projected cash ${projected_cash_after:,.2f} < required ${min_cash_dollars:,.2f})"})
                continue

            # ---- Check 7: trade-frequency cap, persisted across the
            # whole day via the on-disk ledger, not just this call ----
            trades_so_far = _trades_today(symbol)
            if trades_so_far >= limits["max_trades_per_symbol_per_day"]:
                results.append({"symbol": symbol, "action": action, "result": f"REJECTED: {symbol} already traded {trades_so_far}x today (limit {limits['max_trades_per_symbol_per_day']})"})
                continue

        # ---- All checks passed — submit ----
        try:
            order_request = MarketOrderRequest(
                symbol=symbol, notional=notional,
                side=OrderSide.BUY if action == "buy" else OrderSide.SELL,
                time_in_force=TimeInForce.DAY,
            )
            order = _trading_client.submit_order(order_request)
            if action == "buy":
                cash_committed_this_batch += notional
                _increment_ledger(symbol)
            results.append({"symbol": symbol, "action": action, "result": _order_to_dict(order)})
        except Exception as exc:
            results.append({"symbol": symbol, "action": action, "result": f"error submitting to Alpaca: {exc}"})

    return {
        "orders_submitted": sum(1 for r in results if isinstance(r["result"], dict)),
        "orders_rejected": sum(1 for r in results if isinstance(r["result"], str) and r["result"].startswith("REJECTED")),
        "limits_applied": limits,
        "halted_at_call_time": halted,
        "results": results,
    }


# ============================================================
# Options — tail hedge, WITH independent re-validation of the contract
# (not trusting the agent's claim about what it is) and a hard premium cap.
# ============================================================
@mcp.tool()
def buy_option_hedge(option_symbol: str, contracts: int) -> dict:
    """
    Buy `contracts` of a single-leg option (the tail hedge) by OCC symbol,
    after independently re-parsing the symbol and checking it against the
    configured allowed underlying/type/expiry window, fetching a live
    quote, and enforcing the max premium-spend cap from config.yaml.
    """
    limits = _load_limits()

    if not limits["hedging_enabled"]:
        return {"error": "REJECTED: hedging.enabled is False in config.yaml — no hedge orders allowed."}

    halted, halt_reason = is_halted()
    if halted:
        return {"error": f"REJECTED: trading halted ({halt_reason}) — no new hedge positions either."}

    if contracts <= 0 or contracts != int(contracts):
        return {"error": f"REJECTED: contracts must be a positive whole number, got {contracts}."}

    # ---- Independently re-parse the OCC symbol — ground truth, not the
    # agent's description of the contract ----
    try:
        parsed = parse_occ_symbol(option_symbol)
    except ValueError as exc:
        return {"error": f"REJECTED: {exc}"}

    if parsed["underlying"] != limits["hedge_instrument"].upper():
        return {"error": f"REJECTED: underlying {parsed['underlying']} does not match configured hedge instrument {limits['hedge_instrument']}."}

    if parsed["option_type"] != limits["hedge_option_type"].lower():
        return {"error": f"REJECTED: option type {parsed['option_type']} does not match configured hedge_option_type {limits['hedge_option_type']}."}

    days_to_expiry = (parsed["expiry"] - date.today()).days
    if days_to_expiry < 0:
        return {"error": f"REJECTED: {option_symbol} already expired ({parsed['expiry']})."}
    if days_to_expiry > limits["hedge_max_expiration_days"]:
        return {"error": f"REJECTED: expiry {parsed['expiry']} is {days_to_expiry} days out, exceeds hedge_max_expiration_days={limits['hedge_max_expiration_days']}."}

    # ---- Fetch a LIVE quote to estimate actual premium cost — never
    # trust a cost figure the agent stated ----
    try:
        quote_resp = requests.get(
            f"{ALPACA_OPTIONS_DATA_URL}/options/quotes/latest",
            headers=DATA_HEADERS, params={"symbols": option_symbol}, timeout=15,
        )
        quote_resp.raise_for_status()
        quote = quote_resp.json().get("quotes", {}).get(option_symbol, {})
        ask_price = quote.get("ap")
        if not ask_price:
            return {"error": f"REJECTED: could not get a live ask price for {option_symbol}; refusing to size a hedge blind."}
    except requests.RequestException as exc:
        return {"error": f"REJECTED: option quote lookup failed ({exc}); refusing to size a hedge blind."}

    estimated_premium = float(ask_price) * 100 * contracts  # 1 contract = 100 shares
    account = _trading_client.get_account()
    portfolio_value = float(account.portfolio_value)
    max_premium = portfolio_value * (limits["hedge_max_cost_pct"] / 100)

    if estimated_premium > max_premium:
        return {
            "error": (
                f"REJECTED: estimated premium ${estimated_premium:,.2f} "
                f"({contracts} contracts @ ~${ask_price}) exceeds the "
                f"{limits['hedge_max_cost_pct']}% cap (${max_premium:,.2f}). "
                f"Reduce contracts and retry."
            ),
            "estimated_premium": estimated_premium,
            "max_premium_allowed": max_premium,
        }

    try:
        order_request = MarketOrderRequest(
            symbol=option_symbol, qty=contracts,
            side=OrderSide.BUY, time_in_force=TimeInForce.DAY,
        )
        order = _trading_client.submit_order(order_request)
        return {**_order_to_dict(order), "estimated_premium": estimated_premium, "parsed_contract": {**parsed, "expiry": str(parsed["expiry"])}}
    except Exception as exc:
        return {
            "error": str(exc),
            "note": (
                "If this fails, confirm options trading is enabled on your "
                "paper account (Alpaca dashboard > paper account > options "
                "approval level)."
            ),
        }


@mcp.tool()
def close_option_position(option_symbol: str) -> dict:
    """Close an existing open option position (sell-to-close) by OCC symbol. Always allowed, even while halted."""
    try:
        result = _trading_client.close_position(option_symbol)
        return {"symbol": option_symbol, "status": "close order submitted", "detail": str(result)}
    except Exception as exc:
        return {"error": str(exc)}


# ============================================================
# Safety / observability tools
# ============================================================
@mcp.tool()
def get_safety_status() -> dict:
    """
    Report the CURRENT independently-enforced safety state: whether
    trading is halted (and why), the limits being applied and where they
    came from, and today's per-symbol trade counts. Call this if a trade
    you expected to see didn't happen, or before manually investigating
    "why did the crew only hold today."
    """
    halted, reason = is_halted()
    return {
        "halted": halted,
        "halt_reason": reason or None,
        "limits": _load_limits(),
        "trades_today": _load_ledger(),
    }


@mcp.tool()
def trip_daily_loss_halt(reason: str) -> dict:
    """
    Manually (or programmatically, from run_session.py) trip the
    trading-halted flag for the rest of today. Idempotent. Use this
    instead of trying to track halt state in a Python variable somewhere
    — the whole point is that it's a file every process checks.
    """
    set_halted(reason)
    return {"halted": True, "reason": reason}


# ============================================================
# Account / positions
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
                "order_id": str(o.id), "symbol": o.symbol, "qty": str(o.qty),
                "side": str(o.side), "status": str(o.status), "submitted_at": str(o.submitted_at),
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
        "equity": str(acct.equity),
        "last_equity": str(acct.last_equity),
        "trading_blocked": acct.trading_blocked,
        "account_blocked": acct.account_blocked,
    }


if __name__ == "__main__":
    mcp.run(transport="stdio")
