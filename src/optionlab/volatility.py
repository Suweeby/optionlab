"""Volatility (and drift) estimators from a series of historical prices.

All estimators work on LOG returns r_t = ln(P_t / P_{t-1}) of adjusted close prices,
which matches the GBM model: ln(S_T/S_0) is a sum of independent daily log returns.
Daily numbers are annualized with the usual 252-trading-day convention.
"""
from __future__ import annotations

import math
from collections.abc import Iterable

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def _check_prices(prices) -> None:
    if not isinstance(prices, pd.Series):
        raise ValueError("prices must be a pandas Series.")
    if len(prices) < 3:
        raise ValueError("Need at least 3 prices (2 returns) to estimate volatility.")
    values = prices.to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError("prices must be finite and positive.")


def _check_trading_days(trading_days: float) -> None:
    if not trading_days > 0:
        raise ValueError("trading_days must be positive.")


def log_returns(prices: pd.Series) -> pd.Series:
    """Daily log returns ln(P_t / P_{t-1}); one element shorter than the price series."""
    _check_prices(prices)
    return np.log(prices.astype(float)).diff().dropna()


def historical_volatility(prices: pd.Series, window: int | None = None,
                          trading_days: float = TRADING_DAYS) -> float:
    """Annualized close-to-close volatility: stdev(log returns, ddof=1) * sqrt(trading_days).

    window: use only the most recent `window` returns (default: all of them).
    """
    _check_trading_days(trading_days)
    r = log_returns(prices)
    if window is not None:
        if window < 2 or window > len(r):
            raise ValueError(f"window must be between 2 and the number of returns ({len(r)}).")
        r = r.iloc[-int(window):]
    return float(r.std(ddof=1) * math.sqrt(trading_days))


def rolling_volatility(prices: pd.Series, window: int = 21,
                       trading_days: float = TRADING_DAYS) -> pd.Series:
    """Annualized volatility over a moving window of `window` returns."""
    _check_trading_days(trading_days)
    r = log_returns(prices)
    if window < 2 or window > len(r):
        raise ValueError(f"window must be between 2 and the number of returns ({len(r)}).")
    return (r.rolling(int(window)).std(ddof=1) * math.sqrt(trading_days)).dropna()


def ewma_volatility(prices: pd.Series, lam: float = 0.94,
                    trading_days: float = TRADING_DAYS) -> pd.Series:
    """Exponentially weighted volatility (RiskMetrics style), annualized.

    Variance recursion:  s2_t = lam * s2_{t-1} + (1 - lam) * r_t^2, started at s2_0 = r_0^2.
    Recent returns weigh more, so it reacts faster to shocks than a plain window.
    It treats the mean return as zero, as is standard for daily data.
    """
    _check_trading_days(trading_days)
    if not 0.0 < lam < 1.0:
        raise ValueError("lam must be strictly between 0 and 1.")
    r = log_returns(prices)
    variance = (r**2).ewm(alpha=1.0 - lam, adjust=False).mean()
    return np.sqrt(variance * trading_days)


def volatility_by_lookback(prices: pd.Series, windows: Iterable[int] = (21, 63, 126, 252),
                           trading_days: float = TRADING_DAYS) -> pd.DataFrame:
    """Volatility estimate for several lookback windows, plus the full sample.

    Shows the honest answer to "which volatility is right?": it depends on the window.
    Windows longer than the available history are left out.
    """
    n = len(log_returns(prices))
    sizes = sorted({int(w) for w in windows if 2 <= w <= n} | {n})
    rows = [
        {"window_days": w, "observations": w,
         "volatility": historical_volatility(prices, window=w, trading_days=trading_days)}
        for w in sizes
    ]
    return pd.DataFrame(rows)


def annualized_drift(prices: pd.Series, trading_days: float = TRADING_DAYS) -> float:
    """Estimate of the real-world drift mu: mean log return * trading_days + sigma^2 / 2.

    Under GBM, E[ln(S_t/S_0)] = (mu - sigma^2/2) t, so we add sigma^2/2 back.
    WARNING: this estimate is extremely noisy (see drift_standard_error).
    """
    _check_trading_days(trading_days)
    r = log_returns(prices)
    sigma = float(r.std(ddof=1) * math.sqrt(trading_days))
    return float(r.mean() * trading_days + 0.5 * sigma**2)


def drift_standard_error(prices: pd.Series, trading_days: float = TRADING_DAYS) -> float:
    """Standard error of the annualized drift estimate: sigma / sqrt(years of data).

    It depends only on the LENGTH of the history, not on how often you sample it.
    """
    _check_trading_days(trading_days)
    r = log_returns(prices)
    sigma = float(r.std(ddof=1) * math.sqrt(trading_days))
    years = len(r) / trading_days
    return sigma / math.sqrt(years)
