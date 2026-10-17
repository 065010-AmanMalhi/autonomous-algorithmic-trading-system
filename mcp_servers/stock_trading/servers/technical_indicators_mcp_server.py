"""
technical_indicators_mcp_server.py
-----------------------------------
Local MCP server #1: market data + technical indicators.

Extends the class starter file (technical_indicators_mcp_server.py) in two
required directions from the project brief:

  1. Batched multi-symbol OHLCV fetch — ONE Alpaca call for the whole
     portfolio instead of one call per ticker (brief, section 8).
  2. A wider indicator set (pandas_ta) plus a realized-beta tool, so the
     30%-high-beta portfolio rule can be verified from real data rather
     than a third-party snapshot.

Keep this file under mcp_servers/stock_trading/servers/, same as the class
layout. Started automatically by the MCP client (trading_crew.py) over
stdio — you don't need to run it standalone, though you can for the MCP
inspector (see bottom of file).

Requires: pip install mcp pandas pandas_ta requests
Env vars: ALPACA_API_KEY, ALPACA_SECRET_KEY  (your PAPER keys)
"""

# ============================================================
# 0.0 Imports
# ============================================================
import os
from datetime import datetime, timedelta, timezone

import pandas as pd
import pandas_ta as ta  # 180+ indicators, built on top of pandas — pip install pandas_ta
import requests
from mcp.server.fastmcp import FastMCP

# ============================================================
# 1.0 MCP server name
# ============================================================
mcp = FastMCP("TechnicalIndicators")

# ============================================================
# 2.0 Alpaca configuration
# ============================================================
ALPACA_API_KEY = os.environ.get("ALPACA_API_KEY", "")
ALPACA_SECRET_KEY = os.environ.get("ALPACA_SECRET_KEY", "")
ALPACA_DATA_URL = "https://data.alpaca.markets/v2"
ALPACA_OPTIONS_URL = "https://data.alpaca.markets/v1beta1"

HEADERS = {
    "APCA-API-KEY-ID": ALPACA_API_KEY,
    "APCA-API-SECRET-KEY": ALPACA_SECRET_KEY,
}

# Benchmark used for realized-beta calculations (S&P 500 proxy).
BENCHMARK_SYMBOL = "SPY"

# ============================================================
# 3.0 In-memory cache
# Keyed by symbol now (was a single global df in the class starter) so the
# whole 15-stock portfolio can be held at once and indicator tools can
# pull any symbol's frame without re-fetching.
#   _cache[symbol] = pd.DataFrame of OHLCV bars, indexed by timestamp
# ============================================================
_cache: dict[str, pd.DataFrame] = {}


# ============================================================
# 4.0 Tools
# ============================================================

# ------------------------------------------------------------
# 4.1 fetch_ohlcv_batch — ONE Alpaca call for the whole portfolio
# ------------------------------------------------------------
@mcp.tool()
def fetch_ohlcv_batch(
    symbols: list[str],
    lookback_days: int = 180,
    timeframe: str = "1Day",
    include_benchmark: bool = True,
) -> dict:
    """
    Fetch OHLCV bars for MULTIPLE symbols in a single Alpaca request and
    cache each symbol's DataFrame for the indicator tools below.

    Call this ONCE per portfolio cycle before calling any other tool in
    this server — do not loop fetch calls per symbol, that defeats the
    point of batching and burns rate limit.

    Parameters
    ----------
    symbols           : List of tickers, e.g. ["NVDA", "AMD", "COIN", ...].
    lookback_days      : Calendar days of history — default 180
                         (enough for MACD's 26-period slow EMA and a
                         ~1yr-ish realized beta if you raise it to 365).
    timeframe          : Alpaca bar size — "1Day" (default) | "1Hour" | "15Min".
    include_benchmark  : Also fetch SPY and cache it for calculate_beta,
                         even if SPY isn't one of your portfolio symbols.

    Returns
    -------
    dict: per-symbol bar counts + date range, plus which symbols (if any)
    returned no data (bad ticker, market closed the whole window, etc).
    """
    fetch_list = [s.upper() for s in symbols]
    if include_benchmark and BENCHMARK_SYMBOL not in fetch_list:
        fetch_list.append(BENCHMARK_SYMBOL)

    end_dt = datetime.now(timezone.utc)
    start_dt = end_dt - timedelta(days=lookback_days)

    params = {
        # Comma-separated symbols = ONE call for the whole list. This is
        # the batching the brief requires instead of a per-ticker loop.
        "symbols": ",".join(fetch_list),
        "timeframe": timeframe,
        "start": start_dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "end": end_dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "limit": 10_000,
        "feed": "iex",  # free feed; switch to "sip" if your account has a paid data plan
    }

    url = f"{ALPACA_DATA_URL}/stocks/bars"
    response = requests.get(url, headers=HEADERS, params=params, timeout=30)
    response.raise_for_status()

    bars_by_symbol = response.json().get("bars", {})

    summary = {}
    missing = []
    for symbol in fetch_list:
        bars = bars_by_symbol.get(symbol, [])
        if not bars:
            missing.append(symbol)
            continue

        df = pd.DataFrame(bars)
        df["t"] = pd.to_datetime(df["t"])
        df = df.set_index("t").sort_index()
        df = df.rename(columns={"o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"})
        df = df[["open", "high", "low", "close", "volume"]]

        _cache[symbol] = df
        summary[symbol] = {
            "bars_fetched": len(df),
            "start": str(df.index[0]),
            "end": str(df.index[-1]),
        }

    if missing:
        summary["_missing"] = missing

    return {
        "symbols_requested": fetch_list,
        "symbols_cached": [s for s in fetch_list if s in _cache],
        "summary": summary,
        "status": "ok" if not missing else "partial",
    }


def _get_df(symbol: str) -> pd.DataFrame:
    symbol = symbol.upper()
    if symbol not in _cache:
        raise RuntimeError(
            f"No cached data for {symbol}. Call fetch_ohlcv_batch with this "
            "symbol included before calling an indicator tool."
        )
    return _cache[symbol]


# ------------------------------------------------------------
# 4.2 calculate_indicators — composite indicator bundle for ONE symbol
# ------------------------------------------------------------
@mcp.tool()
def calculate_indicators(symbol: str) -> dict:
    """
    Calculate a bundle of technical indicators for one cached symbol:
    RSI(14), MACD(12/26/9), Bollinger Bands(20,2), ATR(14), SMA(20/50)
    trend, and relative volume. Uses pandas_ta.

    You MUST call fetch_ohlcv_batch (with this symbol included) first.

    A lightweight smoothing pass (3-bar rolling mean of closes) is applied
    before indicator math to reduce single-bar noise, per the "denoise
    before deciding" idea from reinforcement-learning trading literature —
    this is NOT full signal decomposition, just cheap noise reduction that
    costs nothing and doesn't lag the signal much.

    Returns
    -------
    dict with rsi, macd (line/signal/histogram/crossover), bollinger
    (upper/mid/lower/%b), atr, sma20, sma50, trend ("Up"/"Down"/"Flat"),
    relative_volume, latest_close, as_of, bars_used.
    """
    symbol = symbol.upper()
    df = _get_df(symbol)

    if len(df) < 55:
        raise ValueError(
            f"Only {len(df)} bars cached for {symbol}; need >= 55 for a "
            "reliable SMA50/MACD read. Re-run fetch_ohlcv_batch with a "
            "larger lookback_days."
        )

    work = df.copy()
    # Cheap noise reduction: 3-bar rolling mean, NOT used for latest_close
    # (we still report the true latest close for execution decisions).
    work["close_smooth"] = work["close"].rolling(3, min_periods=1).mean()

    rsi = ta.rsi(work["close_smooth"], length=14)
    macd = ta.macd(work["close_smooth"], fast=12, slow=26, signal=9)
    bbands = ta.bbands(work["close_smooth"], length=20, std=2)
    atr = ta.atr(work["high"], work["low"], work["close"], length=14)
    sma20 = ta.sma(work["close"], length=20)
    sma50 = ta.sma(work["close"], length=50)

    latest_close = round(float(work["close"].iloc[-1]), 4)
    latest_rsi = round(float(rsi.iloc[-1]), 2)
    macd_line = round(float(macd.iloc[-1, 0]), 4)
    macd_hist = round(float(macd.iloc[-1, 1]), 4)
    macd_signal = round(float(macd.iloc[-1, 2]), 4)
    prev_hist = float(macd.iloc[-2, 1]) if len(macd) >= 2 else 0.0

    if prev_hist <= 0 and macd_hist > 0:
        crossover = "Bullish"
    elif prev_hist >= 0 and macd_hist < 0:
        crossover = "Bearish"
    else:
        crossover = "None"

    bb_lower = float(bbands.iloc[-1, 0])
    bb_mid = float(bbands.iloc[-1, 1])
    bb_upper = float(bbands.iloc[-1, 2])
    pct_b = (latest_close - bb_lower) / (bb_upper - bb_lower) if bb_upper != bb_lower else 0.5

    sma20_last = float(sma20.iloc[-1])
    sma50_last = float(sma50.iloc[-1])
    if sma20_last > sma50_last * 1.005:
        trend = "Up"
    elif sma20_last < sma50_last * 0.995:
        trend = "Down"
    else:
        trend = "Flat"

    avg_vol_20 = float(work["volume"].tail(20).mean())
    latest_vol = float(work["volume"].iloc[-1])
    relative_volume = round(latest_vol / avg_vol_20, 2) if avg_vol_20 > 0 else None

    rsi_signal = "Overbought" if latest_rsi >= 70 else ("Oversold" if latest_rsi <= 30 else "Neutral")

    return {
        "symbol": symbol,
        "latest_close": latest_close,
        "as_of": str(df.index[-1]),
        "bars_used": len(df),
        "rsi": {"value": latest_rsi, "signal": rsi_signal},
        "macd": {
            "line": macd_line,
            "signal": macd_signal,
            "histogram": macd_hist,
            "crossover": crossover,
        },
        "bollinger": {
            "upper": round(bb_upper, 4),
            "mid": round(bb_mid, 4),
            "lower": round(bb_lower, 4),
            "percent_b": round(pct_b, 3),  # 0 = at lower band, 1 = at upper band
        },
        "atr_14": round(float(atr.iloc[-1]), 4),
        "sma20": round(sma20_last, 4),
        "sma50": round(sma50_last, 4),
        "trend": trend,
        "relative_volume": relative_volume,
    }


# ------------------------------------------------------------
# 4.3 calculate_beta — realized beta vs SPY, for the 30%-high-beta rule
# ------------------------------------------------------------
@mcp.tool()
def calculate_beta(symbol: str, window_days: int = 120) -> dict:
    """
    Compute realized beta of `symbol` vs SPY from cached daily bars
    (covariance of daily returns / variance of SPY daily returns).

    Requires fetch_ohlcv_batch to have been called with include_benchmark
    left at its default True (or SPY included explicitly) AND this symbol.

    Use this to verify the brief's 30%-stocks-must-be-high-beta rule from
    real data instead of a third-party site snapshot.

    Parameters
    ----------
    symbol      : Ticker to compute beta for.
    window_days : Trading-day window for the covariance calc — default 120
                  (~6 months). Needs at least window_days+1 cached bars.
    """
    symbol = symbol.upper()
    df = _get_df(symbol)
    bench = _get_df(BENCHMARK_SYMBOL)

    merged = pd.DataFrame({
        "asset": df["close"],
        "bench": bench["close"],
    }).dropna()

    if len(merged) < window_days + 1:
        raise ValueError(
            f"Only {len(merged)} overlapping bars between {symbol} and "
            f"{BENCHMARK_SYMBOL}; need >= {window_days + 1}. Increase "
            "lookback_days on fetch_ohlcv_batch."
        )

    merged = merged.tail(window_days + 1)
    returns = merged.pct_change().dropna()

    covariance = returns["asset"].cov(returns["bench"])
    variance = returns["bench"].var()
    beta = covariance / variance if variance != 0 else None

    return {
        "symbol": symbol,
        "benchmark": BENCHMARK_SYMBOL,
        "window_days": window_days,
        "beta": round(float(beta), 3) if beta is not None else None,
        "classification": (
            "High-beta" if beta is not None and beta >= 1.3 else
            "Market-like" if beta is not None and beta >= 0.8 else
            "Low-beta" if beta is not None else "Unknown"
        ),
    }


# ------------------------------------------------------------
# 4.4 get_option_chain — for the hedging agent (puts on SPY/QQQ etc.)
# ------------------------------------------------------------
@mcp.tool()
def get_option_chain(
    underlying: str,
    option_type: str = "put",
    max_expiration_days: int = 45,
    strikes_around_money: int = 6,
) -> dict:
    """
    Fetch a slice of the live option chain for `underlying` (e.g. "SPY" or
    "QQQ") from Alpaca's options data API — near-the-money contracts
    expiring within max_expiration_days, for the hedging agent to pick a
    tail-hedge strike/expiry from.

    NOTE: options trading must be enabled on your Alpaca paper account
    (Level 1 or 2) for this and buy_option_hedge (in the orders server) to
    work. Verify this in the Alpaca dashboard before relying on it.

    Parameters
    ----------
    underlying            : Underlying ticker, e.g. "SPY".
    option_type            : "put" (default, for downside hedges) or "call".
    max_expiration_days    : Only return contracts expiring within this
                              many days — default 45 (short-dated tail hedge,
                              matches the 15-20 day trading window).
    strikes_around_money    : How many strikes above/below the current
                              underlying price to return (rough ATM window).
    """
    underlying = underlying.upper()

    # Current underlying price, needed to center the strike window.
    if underlying in _cache:
        spot = float(_cache[underlying]["close"].iloc[-1])
    else:
        trade_resp = requests.get(
            f"{ALPACA_DATA_URL}/stocks/{underlying}/trades/latest",
            headers=HEADERS, timeout=15,
        )
        trade_resp.raise_for_status()
        spot = float(trade_resp.json()["trade"]["p"])

    end_date = (datetime.now(timezone.utc) + timedelta(days=max_expiration_days)).strftime("%Y-%m-%d")

    params = {
        "underlying_symbols": underlying,
        "expiration_date_lte": end_date,
        "type": option_type.lower(),
        "strike_price_gte": round(spot * 0.85, 2),
        "strike_price_lte": round(spot * 1.05, 2),
        "limit": 100,
    }
    # Contracts endpoint lives on the trading (not data) host in alpaca-py;
    # the raw REST path below is the paper-trading host's /v2/options/contracts.
    url = "https://paper-api.alpaca.markets/v2/options/contracts"
    response = requests.get(url, headers=HEADERS, params=params, timeout=20)
    response.raise_for_status()
    contracts = response.json().get("option_contracts", [])

    return {
        "underlying": underlying,
        "spot_price": round(spot, 2),
        "option_type": option_type,
        "contracts_found": len(contracts),
        "contracts": [
            {
                "symbol": c.get("symbol"),
                "strike_price": c.get("strike_price"),
                "expiration_date": c.get("expiration_date"),
                "open_interest": c.get("open_interest"),
            }
            for c in contracts[:strikes_around_money * 2]
        ],
    }


# ============================================================
# 5.0 Entry point
# ============================================================
if __name__ == "__main__":
    mcp.run(transport="stdio")

"""
Verify this server standalone (same as the class file):

    uv run mcp dev technical_indicators_mcp_server.py

Opens the MCP inspector UI — connect, open the Tools tab, and try
fetch_ohlcv_batch with symbols=["AAPL","NVDA","SPY"] before testing the
other tools (they all read from the cache fetch_ohlcv_batch fills).
"""
