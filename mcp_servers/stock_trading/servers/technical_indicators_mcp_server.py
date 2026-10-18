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
import math
import os
from datetime import datetime, timedelta, timezone

import numpy as np                      # pip install numpy — array math for the regression combiner
import pandas as pd
import pandas_ta as ta  # 180+ indicators, built on top of pandas — pip install pandas_ta
import requests
from mcp.server.fastmcp import FastMCP
from sklearn.linear_model import LinearRegression  # pip install scikit-learn — the "single output" combiner model

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

# Market-wide fear/volatility proxy. Alpaca only serves US equities/ETPs —
# the actual CBOE VIX index (^VIX) is NOT a tradable equity symbol and
# Alpaca does not carry it (confirmed on Alpaca's own community forum:
# https://forum.alpaca.markets/t/how-to-get-volatility-data-vix-vxtlt-etc/2907).
# VIXY (ProShares VIX Short-Term Futures ETF) tracks near-term VIX futures
# and IS a normal equity ticker Alpaca serves through the same batched
# fetch_ohlcv_batch call as everything else — so it's used here as the
# market-fear proxy instead of standing up a second, non-Alpaca data feed.
MARKET_FEAR_PROXY_SYMBOL = "VIXY"

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
    include_market_fear_proxy: bool = True,
) -> dict:
    """
    Fetch OHLCV bars for MULTIPLE symbols in a single Alpaca request and
    cache each symbol's DataFrame for the indicator tools below.

    Call this ONCE per portfolio cycle before calling any other tool in
    this server — do not loop fetch calls per symbol, that defeats the
    point of batching and burns rate limit.

    Parameters
    ----------
    symbols                    : List of tickers, e.g. ["NVDA", "AMD", "COIN", ...].
    lookback_days               : Calendar days of history — default 180
                                  (enough for MACD's 26-period slow EMA and a
                                  ~1yr-ish realized beta if you raise it to 365).
    timeframe                   : Alpaca bar size — "1Day" (default) | "1Hour" | "15Min".
    include_benchmark           : Also fetch SPY and cache it for calculate_beta,
                                  even if SPY isn't one of your portfolio symbols.
    include_market_fear_proxy   : Also fetch VIXY (market-wide fear/volatility
                                  proxy — see MARKET_FEAR_PROXY_SYMBOL above)
                                  and cache it for get_market_volatility and
                                  calculate_fear_regression.

    Returns
    -------
    dict: per-symbol bar counts + date range, plus which symbols (if any)
    returned no data (bad ticker, market closed the whole window, etc).
    """
    fetch_list = [s.upper() for s in symbols]
    if include_benchmark and BENCHMARK_SYMBOL not in fetch_list:
        fetch_list.append(BENCHMARK_SYMBOL)
    if include_market_fear_proxy and MARKET_FEAR_PROXY_SYMBOL not in fetch_list:
        fetch_list.append(MARKET_FEAR_PROXY_SYMBOL)

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
# 4.4 calculate_realized_volatility — annualized historical volatility,
# the "RV" half of the IV-vs-RV comparison the fear agent needs.
# ------------------------------------------------------------
@mcp.tool()
def calculate_realized_volatility(symbol: str, window_days: int = 20) -> dict:
    """
    Compute annualized REALIZED volatility (RV) for `symbol` from cached
    daily bars: the standard deviation of daily log returns over the last
    `window_days`, scaled to a yearly figure by sqrt(252) (252 = average
    US trading days per year — standard annualization convention).

    This is what option traders compare implied volatility AGAINST: when
    implied volatility (what the options market is pricing in, see
    get_iv_metrics) sits well above realized volatility, the market is
    pricing in more future movement than has actually been happening —
    one classic sign of "the market senses an event coming."

    Requires fetch_ohlcv_batch to have been called with this symbol first.

    Parameters
    ----------
    symbol      : Ticker to compute realized volatility for.
    window_days : Trading-day window for the return-std calc — default 20
                  (~1 trading month), matching common short-dated IV tenors.
    """
    symbol = symbol.upper()
    df = _get_df(symbol)

    if len(df) < window_days + 1:
        raise ValueError(
            f"Only {len(df)} bars cached for {symbol}; need >= {window_days + 1} "
            "for a realized-volatility read. Re-run fetch_ohlcv_batch with a "
            "larger lookback_days."
        )

    # Log returns, not simple % returns — log returns are additive over
    # time and are the standard input for volatility math (this matches
    # how Black-Scholes-style implied volatility is itself defined, so RV
    # and IV end up on a genuinely comparable footing).
    closes = df["close"].tail(window_days + 1)
    log_returns = np.log(closes / closes.shift(1)).dropna()

    daily_std = float(log_returns.std())
    annualized_rv = daily_std * math.sqrt(252)  # 252 trading days/year, standard convention

    return {
        "symbol": symbol,
        "window_days": window_days,
        "daily_return_std": round(daily_std, 6),
        "annualized_realized_volatility": round(annualized_rv, 4),  # e.g. 0.35 = 35% annualized
        "as_of": str(df.index[-1]),
    }


# ------------------------------------------------------------
# 4.5 get_iv_metrics — ATM implied volatility + greeks from Alpaca's
# live option snapshot endpoint, the "IV" half of the comparison.
# ------------------------------------------------------------
@mcp.tool()
def get_iv_metrics(symbol: str, max_expiration_days: int = 45) -> dict:
    """
    Fetch the at-the-money (ATM) option's implied volatility (IV) and
    greeks for `symbol`, using Alpaca's option snapshot endpoint, which
    computes IV and greeks itself via Black-Scholes
    (https://docs.alpaca.markets/reference/optionchain — fields
    `impliedVolatility` and `greeks.{delta,gamma,theta,vega,rho}`). This
    server does NOT compute Black-Scholes itself — it reads Alpaca's own
    calculation, which is the more reliable source of truth.

    "ATM" here means: among live contracts expiring within
    max_expiration_days, pick the single CALL contract whose strike is
    closest to the current spot price (calls and puts have near-identical
    ATM IV by put-call parity, so one side is enough for a single IV read).

    NOTE: requires options trading enabled on your Alpaca paper account
    (Level 1 or 2) — same prerequisite as get_option_chain.

    Parameters
    ----------
    symbol               : Underlying ticker, e.g. "NVDA".
    max_expiration_days  : Only consider contracts expiring within this
                           many days — default 45, matching the
                           short-dated tail-hedge window used elsewhere.

    Returns
    -------
    dict with: symbol, atm_contract_symbol, strike, expiration_date,
    days_to_expiration, implied_volatility, delta, gamma, theta, vega, rho,
    spot_price. If no contracts are found (e.g. options not enabled, or
    the underlying has no listed options), returns an "error" key instead
    of raising — this lets calculate_fear_regression degrade gracefully
    for a symbol with no options market rather than aborting the whole run.
    """
    symbol = symbol.upper()

    # Spot price to find the closest-to-the-money strike.
    if symbol in _cache:
        spot = float(_cache[symbol]["close"].iloc[-1])
    else:
        trade_resp = requests.get(
            f"{ALPACA_DATA_URL}/stocks/{symbol}/trades/latest",
            headers=HEADERS, timeout=15,
        )
        trade_resp.raise_for_status()
        spot = float(trade_resp.json()["trade"]["p"])

    end_date = (datetime.now(timezone.utc) + timedelta(days=max_expiration_days)).strftime("%Y-%m-%d")

    # Step 1: find candidate contracts near the money (same contracts
    # endpoint get_option_chain uses) — narrowed to CALLs only, since we
    # just need one ATM IV read, not a full chain.
    contracts_resp = requests.get(
        "https://paper-api.alpaca.markets/v2/options/contracts",
        headers=HEADERS,
        params={
            "underlying_symbols": symbol,
            "expiration_date_lte": end_date,
            "type": "call",
            "strike_price_gte": round(spot * 0.92, 2),
            "strike_price_lte": round(spot * 1.08, 2),
            "limit": 50,
        },
        timeout=20,
    )
    contracts_resp.raise_for_status()
    contracts = contracts_resp.json().get("option_contracts", [])

    if not contracts:
        return {
            "symbol": symbol,
            "error": f"No listed call contracts found for {symbol} within {max_expiration_days} days of expiry.",
            "spot_price": round(spot, 2),
        }

    # Pick the contract with strike closest to spot = the ATM contract.
    atm_contract = min(contracts, key=lambda c: abs(float(c["strike_price"]) - spot))
    contract_symbol = atm_contract["symbol"]

    # Step 2: pull its live snapshot (implied volatility + greeks) —
    # Alpaca computes both server-side; we just read the fields.
    snapshot_resp = requests.get(
        f"{ALPACA_OPTIONS_URL}/options/snapshots/{symbol}",
        headers=HEADERS,
        params={"symbols": contract_symbol},
        timeout=20,
    )
    snapshot_resp.raise_for_status()
    snapshot = snapshot_resp.json().get("snapshots", {}).get(contract_symbol, {})

    implied_vol = snapshot.get("impliedVolatility")
    greeks = snapshot.get("greeks", {})

    if implied_vol is None:
        return {
            "symbol": symbol,
            "atm_contract_symbol": contract_symbol,
            "error": "Snapshot returned no impliedVolatility for this contract (thin/no quote activity).",
            "spot_price": round(spot, 2),
        }

    expiry = atm_contract.get("expiration_date")
    days_to_expiry = (datetime.strptime(expiry, "%Y-%m-%d").date() - datetime.now(timezone.utc).date()).days if expiry else None

    return {
        "symbol": symbol,
        "atm_contract_symbol": contract_symbol,
        "strike": float(atm_contract["strike_price"]),
        "expiration_date": expiry,
        "days_to_expiration": days_to_expiry,
        "implied_volatility": round(float(implied_vol), 4),  # same scale as annualized_realized_volatility (e.g. 0.42 = 42%)
        "delta": greeks.get("delta"),
        "gamma": greeks.get("gamma"),
        "theta": greeks.get("theta"),
        "vega": greeks.get("vega"),
        "rho": greeks.get("rho"),
        "spot_price": round(spot, 2),
    }


# ------------------------------------------------------------
# 4.6 get_market_volatility — market-wide fear level via VIXY proxy
# ------------------------------------------------------------
@mcp.tool()
def get_market_volatility(window_days: int = 20) -> dict:
    """
    Read the market-wide volatility/fear level from VIXY (the market-fear
    proxy — see MARKET_FEAR_PROXY_SYMBOL comment near the top of this
    file for why VIXY instead of the real CBOE VIX index). Reports both
    VIXY's current price LEVEL (a rough stand-in for "the VIX number")
    and its own recent % change (a rising VIXY = rising market fear,
    falling VIXY = complacency returning).

    Requires fetch_ohlcv_batch to have been called with
    include_market_fear_proxy left at its default True.

    Parameters
    ----------
    window_days : Lookback window for computing VIXY's own recent return
                  and its percentile rank within that window — default 20.
    """
    df = _get_df(MARKET_FEAR_PROXY_SYMBOL)

    if len(df) < window_days + 1:
        raise ValueError(
            f"Only {len(df)} bars cached for {MARKET_FEAR_PROXY_SYMBOL}; "
            f"need >= {window_days + 1}. Re-run fetch_ohlcv_batch with a "
            "larger lookback_days."
        )

    closes = df["close"].tail(window_days + 1)
    latest = float(closes.iloc[-1])
    prior = float(closes.iloc[-2])
    pct_change_1d = (latest - prior) / prior if prior != 0 else 0.0

    # Percentile rank of today's level within the lookback window — e.g.
    # 0.90 means market fear today is higher than 90% of the last
    # `window_days` readings, a cheap way to say "fear is elevated
    # relative to its OWN recent range," not just "fear went up today."
    percentile_rank = float((closes <= latest).mean())

    if percentile_rank >= 0.80:
        fear_level = "Elevated"
    elif percentile_rank <= 0.20:
        fear_level = "Complacent"
    else:
        fear_level = "Normal"

    return {
        "proxy_symbol": MARKET_FEAR_PROXY_SYMBOL,
        "latest_level": round(latest, 4),
        "pct_change_1d": round(pct_change_1d, 4),
        "percentile_rank_in_window": round(percentile_rank, 3),
        "fear_level": fear_level,
        "window_days": window_days,
        "as_of": str(df.index[-1]),
        "note": "VIXY is a VIX-futures-tracking ETF, not the CBOE VIX index itself — treat this as a directional proxy for market fear, not the literal VIX print.",
    }


# ------------------------------------------------------------
# 4.7 calculate_fear_regression — combines RV, IV, the IV-RV spread,
# market-wide fear, relative volume and RSI into ONE linear-regression
# output per symbol, via scikit-learn's LinearRegression.
# ------------------------------------------------------------
@mcp.tool()
def calculate_fear_regression(symbol: str, window_days: int = 60) -> dict:
    """
    Fit a simple linear regression that combines several volatility/fear
    features into a SINGLE predicted-return signal for `symbol`:

        next-day return  ~  IV-RV spread + realized volatility
                           + relative volume + RSI + market fear change

    HOW IT WORKS (read this before trusting the output):
    This is an IN-SAMPLE ordinary-least-squares fit over the last
    `window_days` trading days — it fits the straight line that best
    explains what ALREADY happened, then reads off what that same line
    says about TODAY's feature values. It is a transparent way to combine
    several signals into one number with sensible weights (the
    regression coefficients), not a validated forecasting model: with
    only ~60 data points and 5 features it will not generalize reliably,
    and a high R² here mostly means the features correlate with recent
    returns in-sample, not that tomorrow is predictable. Treat the output
    as a DESCRIPTIVE "how much do current fear/vol conditions resemble
    conditions that preceded up-moves vs down-moves in this symbol's
    recent history," not a probability.

    Use the fit_r_squared field to judge how much to trust a given
    reading — below ~0.15, the regression is barely distinguishing
    anything and the signal should be treated as noise.

    Requires fetch_ohlcv_batch (with this symbol AND include_market_fear_proxy
    left True) to have been called first. Also calls get_iv_metrics
    internally for the current IV reading — if that returns no options
    data (e.g. options not enabled, or the ticker has no listed options),
    this tool still returns a result but flags iv_available: false and
    drops IV-dependent terms from the fit.

    Parameters
    ----------
    symbol      : Ticker to build the combined fear/vol signal for.
    window_days : Trading-day window for both the feature history and the
                  regression fit — default 60 (needs RSI/MACD-style
                  indicators to already have enough bars behind them, so
                  this implicitly requires a longer fetch_ohlcv_batch
                  lookback_days, e.g. >= window_days + 60).
    """
    symbol = symbol.upper()
    df = _get_df(symbol)
    market_df = _get_df(MARKET_FEAR_PROXY_SYMBOL)

    min_bars_needed = window_days + 30  # extra buffer for RSI's own warm-up period
    if len(df) < min_bars_needed:
        raise ValueError(
            f"Only {len(df)} bars cached for {symbol}; need >= {min_bars_needed} "
            "for a fear-regression fit. Re-run fetch_ohlcv_batch with a "
            "larger lookback_days."
        )

    # ---- Build the daily feature history over the window ----
    work = df.copy()
    work["log_return"] = np.log(work["close"] / work["close"].shift(1))
    work["forward_return"] = work["log_return"].shift(-1)  # TARGET: next day's return
    work["realized_vol_20"] = work["log_return"].rolling(20).std() * math.sqrt(252)
    work["rsi_14"] = ta.rsi(work["close"], length=14)
    avg_vol_20 = work["volume"].rolling(20).mean()
    work["relative_volume"] = work["volume"] / avg_vol_20

    market_returns = market_df["close"].pct_change().rename("market_fear_change")
    work = work.join(market_returns, how="left")

    history = work.tail(window_days).dropna(
        subset=["forward_return", "realized_vol_20", "rsi_14", "relative_volume", "market_fear_change"]
    )

    if len(history) < 20:
        raise ValueError(
            f"Only {len(history)} complete feature rows available for {symbol} "
            "after dropping warm-up/NaN rows; need >= 20 for a usable regression "
            "fit. Increase window_days or fetch_ohlcv_batch's lookback_days."
        )

    # ---- Try to add the IV-RV spread as a feature (needs a live options
    # read, which can fail independently of the price-history fit above) ----
    iv_result = get_iv_metrics(symbol)
    iv_available = "implied_volatility" in iv_result and "error" not in iv_result
    current_rv = float(work["realized_vol_20"].iloc[-1])
    current_iv_rv_spread = (iv_result["implied_volatility"] - current_rv) if iv_available else None

    # Feature matrix X and target y for the OLS fit. Each row = one
    # historical trading day; each column = one feature. We do NOT
    # include IV history here (Alpaca's option snapshot endpoint only
    # gives TODAY's IV, not a historical IV time series on a free/paper
    # setup) — IV-RV spread is instead applied as a one-time adjustment
    # to today's prediction after the fit, not as a regression feature.
    feature_cols = ["realized_vol_20", "rsi_14", "relative_volume", "market_fear_change"]
    X = history[feature_cols].to_numpy()
    y = history["forward_return"].to_numpy()

    model = LinearRegression()
    model.fit(X, y)
    r_squared = float(model.score(X, y))

    # Score TODAY's feature values through the fitted line.
    today_features = np.array([[
        current_rv,
        float(work["rsi_14"].iloc[-1]),
        float(work["relative_volume"].iloc[-1]),
        float(work["market_fear_change"].iloc[-1]) if not pd.isna(work["market_fear_change"].iloc[-1]) else 0.0,
    ]])
    predicted_forward_return = float(model.predict(today_features)[0])

    # IV-RV spread adjustment: a wide POSITIVE spread (IV >> RV) means
    # the options market is pricing in more movement than has actually
    # occurred — classic "market senses an event" / elevated-fear signal
    # — which should pull the combined signal toward caution (negative
    # adjustment), scaled modestly so it nudges rather than dominates.
    iv_adjustment = -0.15 * current_iv_rv_spread if current_iv_rv_spread is not None else 0.0
    combined_fear_adjusted_signal = predicted_forward_return + iv_adjustment

    if combined_fear_adjusted_signal > 0.002:
        interpretation = "Greed-leaning (conditions historically preceded up-moves)"
    elif combined_fear_adjusted_signal < -0.002:
        interpretation = "Fear-leaning (conditions historically preceded down-moves / elevated risk)"
    else:
        interpretation = "Neutral"

    confidence = "Low (R² < 0.15 — treat as noise)" if r_squared < 0.15 else (
        "Moderate" if r_squared < 0.35 else "Higher (still in-sample, not a forecast guarantee)"
    )

    return {
        "symbol": symbol,
        "window_days": window_days,
        "features_used": feature_cols + (["iv_rv_spread (post-fit adjustment)"] if iv_available else []),
        "current_realized_volatility": round(current_rv, 4),
        "current_implied_volatility": round(iv_result["implied_volatility"], 4) if iv_available else None,
        "iv_available": iv_available,
        "iv_rv_spread": round(current_iv_rv_spread, 4) if current_iv_rv_spread is not None else None,
        "regression_coefficients": dict(zip(feature_cols, [round(c, 6) for c in model.coef_])),
        "fit_r_squared": round(r_squared, 3),
        "fit_confidence": confidence,
        "predicted_forward_return_raw": round(predicted_forward_return, 5),
        "fear_adjusted_signal": round(combined_fear_adjusted_signal, 5),
        "interpretation": interpretation,
        "as_of": str(df.index[-1]),
    }


# ------------------------------------------------------------
# 4.8 get_option_chain — for the hedging agent (puts on SPY/QQQ etc.)
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
