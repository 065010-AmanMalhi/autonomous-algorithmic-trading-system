"""
Combined Alpaca paper-account performance metrics
Read-only: retrieves account history and SPY bars. Does not place trades.

Run from the project directory:
    python .\calculate_portfolio_metrics.py

Assumptions:
- Three separate Alpaca paper accounts, each loaded from its own .env file.
- Daily returns are reconstructed from daily portfolio P&L / prior combined equity.
- Risk-free rate and minimum acceptable return are set to zero.
- Jensen's alpha is estimated by OLS against SPY daily returns.
"""

from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import dotenv_values
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetPortfolioHistoryRequest
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

ACCOUNTS = [
    ("Aman", ".env"),
    ("Mayank", ".env.mayank"),
    ("Aditi", ".env.aditi"),
]
MIN_OBSERVATIONS = 30
TRADING_DAYS_PER_YEAR = 252


def get_credentials(env_file):
    env = dotenv_values(env_file)
    key = (
        env.get("APCA_API_KEY_ID")
        or env.get("ALPACA_API_KEY")
        or env.get("ALPACA_API_KEY_ID")
    )
    secret = (
        env.get("APCA_API_SECRET_KEY")
        or env.get("ALPACA_SECRET_KEY")
        or env.get("ALPACA_API_SECRET_KEY")
    )
    if not key or not secret:
        raise ValueError(f"Missing Alpaca key/secret variable in {env_file}")
    return key, secret


def fetch_account_history(name, env_file):
    key, secret = get_credentials(env_file)
    client = TradingClient(key, secret, paper=True)
    account = client.get_account()
    history = client.get_portfolio_history(
        GetPortfolioHistoryRequest(period="1A", timeframe="1D")
    )

    timestamps = list(history.timestamp or [])
    if not timestamps:
        raise ValueError(f"No portfolio history returned for {name}")

    df = pd.DataFrame({
        "date": pd.to_datetime(timestamps, unit="s", utc=True).date,
        "equity": list(history.equity or []),
        "pnl": list(history.profit_loss or []),
        "pnl_pct": list(history.profit_loss_pct or []),
    })
    df = (
        df.drop_duplicates("date")
        .set_index("date")
        .sort_index()
        .apply(pd.to_numeric, errors="coerce")
    )

    print(
        f"{name}: status={account.status}, current_equity=${float(account.equity):,.2f}, "
        f"history_points={len(df)}, range={df.index.min()} to {df.index.max()}"
    )
    return df, key, secret


def fetch_spy_returns(key, secret, start_date, end_date):
    data_client = StockHistoricalDataClient(key, secret)
    start = datetime.combine(start_date, datetime.min.time(), tzinfo=timezone.utc)
    end = datetime.combine(
        end_date + timedelta(days=1),
        datetime.min.time(),
        tzinfo=timezone.utc,
    )
    request = StockBarsRequest(
        symbol_or_symbols=["SPY"],
        timeframe=TimeFrame.Day,
        start=start,
        end=end,
        adjustment="all",
    )
    bars = data_client.get_stock_bars(request).df
    if bars.empty:
        raise ValueError("No SPY daily bars returned. Check Alpaca market-data access.")

    spy = bars.reset_index()
    spy["date"] = pd.to_datetime(spy["timestamp"], utc=True).dt.date
    spy = spy.sort_values("timestamp").drop_duplicates("date").set_index("date")
    return spy["close"].astype(float).pct_change().rename("spy_return")


def main():
    histories = {}
    first_key = first_secret = None

    for name, env_file in ACCOUNTS:
        if not Path(env_file).exists():
            raise FileNotFoundError(f"Required environment file not found: {env_file}")
        df, key, secret = fetch_account_history(name, env_file)
        histories[name] = df
        if first_key is None:
            first_key, first_secret = key, secret

    # Align all account data on dates that exist in all three histories.
    equity = pd.concat(
        {name: df["equity"] for name, df in histories.items()},
        axis=1,
        join="inner",
    ).dropna()

    pnl = pd.concat(
        {name: df["pnl"] for name, df in histories.items()},
        axis=1,
        join="inner",
    ).dropna()

    common_dates = equity.index.intersection(pnl.index)
    equity = equity.loc[common_dates]
    pnl = pnl.loc[common_dates]

    if len(equity) < 2:
        raise ValueError(
            "Fewer than two common daily equity points across the three accounts."
        )

    combined_equity = equity.sum(axis=1)
    combined_pnl = pnl.sum(axis=1)

    # Use the daily P&L divided by previous combined equity, which is more
    # robust to simple account-size differences than averaging account returns.
    portfolio_returns = (
        combined_pnl / combined_equity.shift(1)
    ).replace([np.inf, -np.inf], np.nan).dropna()

    start_date = min(df.index.min() for df in histories.values())
    end_date = max(df.index.max() for df in histories.values())
    spy_returns = fetch_spy_returns(first_key, first_secret, start_date, end_date)

    data = pd.concat(
        [
            portfolio_returns.rename("portfolio_return"),
            spy_returns,
        ],
        axis=1,
        join="inner",
    ).replace([np.inf, -np.inf], np.nan).dropna()

    # Diagnostics always print before deciding whether to calculate metrics.
    print("\n========== DATA COVERAGE ==========")
    for name, df in histories.items():
        print(f"{name} observations: {len(df)}")
    print(f"Common dates across all accounts: {len(equity)}")
    print(f"Combined portfolio daily returns: {len(portfolio_returns)}")
    print(f"SPY daily returns available: {len(spy_returns.dropna())}")
    print(f"Matched portfolio/SPY returns: {len(data)}")
    if len(data):
        print(f"Matched range: {data.index.min()} to {data.index.max()}")

    print("\n========== CURRENT EQUITY ==========")
    for name, df in histories.items():
        print(f"{name}: ${float(df['equity'].iloc[-1]):,.2f}")
    print(f"Combined latest history equity: ${float(combined_equity.iloc[-1]):,.2f}")

    if len(data) < MIN_OBSERVATIONS:
        print(
            f"\nNOT CALCULATED: only {len(data)} matched daily observations; "
            f"at least {MIN_OBSERVATIONS} are required by this script."
        )
        print(
            "The diagnostic counts above identify whether account-date alignment "
            "or benchmark coverage is limiting the sample."
        )
        return

    r = data["portfolio_return"].astype(float)
    market = data["spy_return"].astype(float)

    # Simplified assumption: zero daily risk-free rate and zero MAR.
    rf_daily = 0.0
    excess = r - rf_daily
    volatility = excess.std(ddof=1)
    sharpe = (
        excess.mean() / volatility * np.sqrt(TRADING_DAYS_PER_YEAR)
        if volatility > 0 else np.nan
    )

    downside = np.minimum(excess.to_numpy(), 0.0)
    downside_deviation = np.sqrt(np.mean(downside ** 2))
    sortino = (
        excess.mean() / downside_deviation * np.sqrt(TRADING_DAYS_PER_YEAR)
        if downside_deviation > 0 else np.nan
    )

    # Jensen's alpha: intercept from daily OLS regression against SPY.
    beta, alpha_daily = np.polyfit(market.to_numpy() - rf_daily, excess.to_numpy(), 1)
    alpha_annual = alpha_daily * TRADING_DAYS_PER_YEAR
    cumulative_return = (1.0 + r).prod() - 1.0

    print("\n========== COMBINED PORTFOLIO METRICS ==========")
    print(f"Observations used: {len(data)}")
    print(f"Period: {data.index.min()} to {data.index.max()}")
    print(f"Sharpe ratio (annualized, risk-free rate = 0): {sharpe:.4f}")
    print(f"Sortino ratio (annualized, MAR = 0): {sortino:.4f}")
    print(f"Jensen's alpha (annualized arithmetic): {alpha_annual:.4%}")
    print(f"SPY beta: {beta:.4f}")
    print(f"Cumulative return from daily P&L series: {cumulative_return:.4%}")
    print("\nCaveats:")
    print("- Verify account deposits/withdrawals and portfolio-history semantics.")
    print("- History may be shorter than one full year despite period='1A'.")
    print("- Zero risk-free rate is a simplifying assumption.")
    print("- These are estimates, not investment advice.")


if __name__ == "__main__":
    main()
