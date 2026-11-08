"""Real-market price data: download, clean, cache locally, and fall back gracefully.

load_prices() order of preference:
    1. a fresh local cache file that covers the requested window
    2. a new download (yfinance), which then refreshes the cache
    3. if the download fails: an older cache file, else clearly labelled SYNTHETIC sample data
An unknown ticker never falls back to sample data: it raises TickerNotFoundError.
"""
from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Callable

import pandas as pd

from optionlab.simulation.gbm import simulate_gbm

DEFAULT_CACHE_DIR = Path("data") / "cache"       # relative to where the app is run from
_TICKER_RE = re.compile(r"^\^?[A-Z0-9][A-Z0-9.\-=^]{0,14}$")   # optional leading ^ for indices


class DataUnavailableError(RuntimeError):
    """Prices could not be obtained and no fallback was allowed or available."""


class TickerNotFoundError(DataUnavailableError):
    """The data source returned no usable prices for this ticker."""


@dataclass(frozen=True)
class PriceData:
    ticker: str
    prices: pd.Series        # adjusted close, ascending DatetimeIndex
    source: str              # "network", "cache", "stale-cache" or "sample"
    note: str = ""

    @property
    def is_real(self) -> bool:
        return self.source != "sample"


def normalize_ticker(ticker: str) -> str:
    """Upper-case and validate a ticker. The ticker becomes a file name, so be strict."""
    cleaned = str(ticker).strip().upper()
    if not _TICKER_RE.match(cleaned):
        raise ValueError(f"'{ticker}' is not a valid ticker symbol.")
    return cleaned


def clean_prices(raw) -> pd.Series:
    """Turn a raw download (DataFrame or Series) into a clean adjusted-close Series.

    Picks 'Adj Close' or 'Close', drops missing and non-positive prices, removes
    duplicate dates, sorts by date and strips time zones. Needs >= 2 valid prices.
    """
    if isinstance(raw, pd.DataFrame):
        for column in ("Adj Close", "Close", "adj_close"):
            if column in raw.columns:
                series = raw[column]
                break
        else:
            raise ValueError("No 'Adj Close' or 'Close' column in the data.")
    elif isinstance(raw, pd.Series):
        series = raw
    else:
        raise ValueError("Expected a pandas DataFrame or Series of prices.")

    series = pd.to_numeric(series, errors="coerce")
    index = pd.DatetimeIndex(pd.to_datetime(series.index))
    if index.tz is not None:
        index = index.tz_localize(None)
    series = pd.Series(series.to_numpy(dtype=float), index=index.normalize(), name="adj_close")

    series = series[series.notna() & (series > 0)]
    series = series[~series.index.duplicated(keep="last")].sort_index()
    if len(series) < 2:
        raise ValueError("Fewer than 2 valid prices in the data.")
    return series


def sample_prices(n_days: int = 504, seed: int = 0, end: date | None = None,
                  S0: float = 100.0, mu: float = 0.08, sigma: float = 0.25) -> pd.Series:
    """SYNTHETIC prices from our own GBM generator, for offline use. Not real market data."""
    end = end or date.today()
    index = pd.bdate_range(end=pd.Timestamp(end), periods=n_days + 1)
    path = simulate_gbm(S0, mu, sigma, T=n_days / 252, n_steps=n_days, n_paths=1, seed=seed)[0]
    return pd.Series(path, index=index, name="adj_close")


def _yfinance_fetcher(ticker: str, start: date, end: date) -> pd.DataFrame:
    """Default downloader. yfinance is imported here so the rest works without it."""
    import yfinance as yf

    return yf.Ticker(ticker).history(start=start.isoformat(), end=end.isoformat(),
                                     auto_adjust=True)


def _read_cache(path: Path) -> pd.Series | None:
    try:
        frame = pd.read_csv(path, index_col=0, parse_dates=True)
        return clean_prices(frame["adj_close"])
    except Exception:           # missing, unreadable or corrupt: treat as no cache
        return None


def _write_cache(path: Path, prices: pd.Series) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    prices.rename("adj_close").to_csv(temp, index_label="date")
    os.replace(temp, path)      # atomic: a crash never leaves a half-written cache


def _is_fresh(path: Path, max_age_hours: float) -> bool:
    return time.time() - path.stat().st_mtime <= max_age_hours * 3600


def _covers(prices: pd.Series, start: date) -> bool:
    """Does the cached history reach back to (about) the requested start date?"""
    return prices.index.min() <= pd.Timestamp(start) + pd.Timedelta(days=7)


def _window(prices: pd.Series, start: date) -> pd.Series:
    return prices.loc[pd.Timestamp(start):]


def load_prices(
    ticker: str,
    lookback_days: int = 730,
    cache_dir: Path | str = DEFAULT_CACHE_DIR,
    max_age_hours: float = 24.0,
    fetcher: Callable[[str, date, date], pd.DataFrame] | None = None,
    today: date | None = None,
    allow_sample: bool = True,
) -> PriceData:
    """Adjusted close prices for roughly the last `lookback_days` calendar days."""
    ticker = normalize_ticker(ticker)
    if lookback_days < 30:
        raise ValueError("lookback_days must be at least 30.")
    today = today or date.today()
    start = today - timedelta(days=int(lookback_days))
    fetch = fetcher or _yfinance_fetcher
    path = Path(cache_dir) / f"{ticker}.csv"
    cached = _read_cache(path) if path.exists() else None

    # 1) fresh cache that covers the window
    if cached is not None and _is_fresh(path, max_age_hours) and _covers(cached, start):
        return PriceData(ticker, _window(cached, start), "cache", "Loaded from the local cache.")

    # 2) download
    try:
        raw = fetch(ticker, start, today + timedelta(days=1))      # end date is exclusive
    except Exception as exc:    # network down, rate limit, yfinance missing, ...
        return _fallback(ticker, cached, start, today, allow_sample,
                         f"Download failed ({type(exc).__name__}: {exc}).")
    try:
        prices = clean_prices(raw)
    except ValueError:
        if cached is not None:
            return _stale(ticker, cached, start, "The data source returned no usable prices.")
        raise TickerNotFoundError(
            f"No price data returned for '{ticker}'. Check the ticker symbol.") from None

    _write_cache(path, prices)
    return PriceData(ticker, _window(prices, start), "network", "Downloaded just now.")


def _stale(ticker: str, cached: pd.Series, start: date, reason: str) -> PriceData:
    note = f"{reason} Showing cached data last updated {cached.index[-1].date()}."
    return PriceData(ticker, _window(cached, start), "stale-cache", note)


def _fallback(ticker: str, cached, start: date, today: date, allow_sample: bool,
              reason: str) -> PriceData:
    if cached is not None:
        return _stale(ticker, cached, start, reason)
    if allow_sample:
        prices = sample_prices(end=today)
        note = (f"{reason} No cached data either: showing SYNTHETIC sample prices "
                "(not real market data).")
        return PriceData(ticker, prices, "sample", note)
    raise DataUnavailableError(f"{reason} No cached data available.")