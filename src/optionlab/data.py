"""Real-market price data: download, clean, cache locally, and fall back gracefully.

load_prices() order of preference:
    1. a fresh local cache file that covers the requested window
    2. a new download (yfinance), which then refreshes the cache
    3. if the download fails: an older cache file, else clearly labelled SYNTHETIC sample data
An unknown ticker never falls back to sample data: it raises TickerNotFoundError.
refresh_prices() always downloads and never falls back, so a failed refresh is visible.

Cache layout (inside the cache folder, data/cache by default):
    <TICKER>.csv   adjusted close prices, one row per trading day
    index.json     per-ticker metadata (first/last date, rows, last refresh, currency, ...)
    options/       option chains (see market.py)

A download covers the span the cache already holds plus the requested window, and replaces
the file. So a short request never shrinks the history, and every row in the file shares one
adjustment basis (the data source recomputes adjusted closes after each dividend or split).

Uploaded CSV files (load_prices_from_csv) are labelled "upload" and never written to the cache.
"""
from __future__ import annotations

import io
import json
import os
import re
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import IO, Callable

import numpy as np
import pandas as pd

from optionlab.config import DEFAULT_LOOKBACK_DAYS, MIN_VOL_RETURNS
from optionlab.simulation.gbm import simulate_gbm
from optionlab.volatility import require_min_history

__all__ = [
    "DEFAULT_CACHE_DIR", "INDEX_FILE", "LIST_COLUMNS", "REAL_SOURCES",
    "DataUnavailableError", "TickerNotFoundError", "PriceData",
    "resolve_cache_dir", "normalize_ticker", "clean_prices", "clean_prices_with_report",
    "sample_prices", "load_prices", "refresh_prices", "list_cached", "require_min_history",
    "load_prices_from_csv", "prices_to_csv",
]

DEFAULT_CACHE_DIR = Path("data") / "cache"       # relative to where the app is run from
INDEX_FILE = "index.json"
REAL_SOURCES = frozenset({"network", "cache", "stale-cache"})
LIST_COLUMNS = ["ticker", "first_date", "last_date", "rows", "last_refresh_utc", "source",
                "currency", "history_start"]
_TICKER_RE = re.compile(r"^\^?[A-Z0-9][A-Z0-9.\-=^]{0,14}$")   # optional leading ^ for indices
_SLACK = timedelta(days=7)      # weekends and holidays between a calendar date and a trading day


class DataUnavailableError(RuntimeError):
    """Prices could not be obtained and no fallback was allowed or available."""


class TickerNotFoundError(DataUnavailableError):
    """The data source returned no usable prices for this ticker."""


@dataclass(frozen=True)
class PriceData:
    """Adjusted close prices for one ticker, plus where they came from."""

    ticker: str
    prices: pd.Series        # adjusted close, ascending DatetimeIndex
    source: str              # "network", "cache", "stale-cache", "upload" or "sample"
    note: str = ""
    currency: str | None = None                            # e.g. "USD", when the source says
    cleaning: dict[str, int] = field(default_factory=dict)  # rows dropped while cleaning, by reason

    @property
    def is_real(self) -> bool:
        """True only for market data we downloaded (now or earlier).

        False for SYNTHETIC sample data and for user uploads, whose origin cannot be checked.
        """
        return self.source in REAL_SOURCES

    @property
    def is_user_supplied(self) -> bool:
        """True for prices read from a file the user uploaded."""
        return self.source == "upload"


# ---------------------------------------------------------------- small helpers
def resolve_cache_dir(cache_dir: Path | str | None = None) -> Path:
    """Explicit argument first, then the OPTIONLAB_CACHE_DIR environment variable, then data/cache."""
    if cache_dir is not None:
        return Path(cache_dir)
    env = os.environ.get("OPTIONLAB_CACHE_DIR")
    return Path(env) if env else DEFAULT_CACHE_DIR


def normalize_ticker(ticker: str) -> str:
    """Upper-case and validate a ticker. The ticker becomes a file name, so be strict."""
    cleaned = str(ticker).strip().upper()
    if not _TICKER_RE.match(cleaned):
        raise ValueError(f"'{ticker}' is not a valid ticker symbol.")
    return cleaned


def _as_date(value: date | str) -> date:
    """A date, datetime, pandas Timestamp or 'YYYY-MM-DD' string as a plain date."""
    try:
        return pd.Timestamp(value).date()
    except (ValueError, TypeError):
        raise ValueError(f"'{value}' is not a valid date (expected YYYY-MM-DD).") from None


def _resolve_window(lookback_days: int, today: date, start: date | str | None,
                    end: date | str | None) -> tuple[date, date]:
    """Turn (lookback_days) or (start, end) into a validated (start, end) pair of dates."""
    end_date = today if end is None else min(_as_date(end), today)
    if start is None:
        if lookback_days < 30:
            raise ValueError("lookback_days must be at least 30.")
        start_date = end_date - timedelta(days=int(lookback_days))
    else:
        start_date = _as_date(start)
    if (end_date - start_date).days < 30:
        raise ValueError(f"The date range {start_date} to {end_date} must span at least 30 "
                         "calendar days, with the start before the end.")
    return start_date, end_date


# ---------------------------------------------------------------- cleaning
def clean_prices_with_report(raw: pd.DataFrame | pd.Series) -> tuple[pd.Series, dict[str, int]]:
    """Clean a raw price table and report how many rows were dropped and why.

    Picks 'Adj Close' or 'Close', drops missing, non-finite and non-positive prices, removes
    duplicate dates (keeping the last), sorts by date and strips time zones.
    Report keys: rows_in, missing_or_non_finite, non_positive, duplicate_dates, rows_kept.
    Raises ValueError if fewer than 2 valid prices remain.
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

    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    index = pd.DatetimeIndex(pd.to_datetime(series.index))
    if index.tz is not None:
        index = index.tz_localize(None)
    series = pd.Series(values, index=index.normalize(), name="adj_close")

    report = {"rows_in": int(len(series))}
    finite = np.isfinite(series.to_numpy())
    report["missing_or_non_finite"] = int((~finite).sum())
    series = series[finite]
    positive = series > 0
    report["non_positive"] = int((~positive).sum())
    series = series[positive]
    duplicated = series.index.duplicated(keep="last")
    report["duplicate_dates"] = int(duplicated.sum())
    series = series[~duplicated].sort_index()
    report["rows_kept"] = int(len(series))
    if len(series) < 2:
        raise ValueError("Fewer than 2 valid prices in the data.")
    return series, report


def clean_prices(raw: pd.DataFrame | pd.Series) -> pd.Series:
    """Turn a raw download (DataFrame or Series) into a clean adjusted-close Series.

    Same cleaning as clean_prices_with_report, without the report. Needs >= 2 valid prices.
    """
    return clean_prices_with_report(raw)[0]


def sample_prices(n_days: int = 504, seed: int = 0, end: date | None = None,
                  S0: float = 100.0, mu: float = 0.08, sigma: float = 0.25) -> pd.Series:
    """SYNTHETIC prices from our own GBM generator, for offline use. Not real market data."""
    end = end or date.today()
    index = pd.bdate_range(end=pd.Timestamp(end), periods=n_days + 1)
    path = simulate_gbm(S0, mu, sigma, T=n_days / 252, n_steps=n_days, n_paths=1, seed=seed)[0]
    return pd.Series(path, index=index, name="adj_close")


# ---------------------------------------------------------------- downloader
def _yfinance_fetcher(ticker: str, start: date, end: date) -> pd.DataFrame:
    """Default downloader. yfinance is imported here so the rest works without it.

    The currency, when Yahoo reports it, is attached as frame.attrs["currency"].
    """
    import yfinance as yf

    handle = yf.Ticker(ticker)
    frame = handle.history(start=start.isoformat(), end=end.isoformat(), auto_adjust=True)
    try:                                   # best effort: an extra lookup that may fail offline
        frame.attrs["currency"] = handle.fast_info["currency"]
    except Exception:
        pass
    return frame


# ---------------------------------------------------------------- cache files
def _price_path(folder: Path, ticker: str) -> Path:
    return folder / f"{ticker}.csv"


def _read_cache(path: Path) -> pd.Series | None:
    try:
        frame = pd.read_csv(path, index_col=0, parse_dates=True)
        return clean_prices(frame["adj_close"])
    except Exception:           # missing, unreadable or corrupt: treat as no cache
        return None


def _atomic_write_text(path: Path, text: str) -> None:
    """Write via a temporary file, so a crash never leaves a half-written file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(text, encoding="utf-8", newline="")     # keep "\n" line endings on Windows
    os.replace(temp, path)


def _write_cache(path: Path, prices: pd.Series) -> None:
    _atomic_write_text(path, prices.rename("adj_close").to_csv(index_label="date"))


def _is_fresh(path: Path, max_age_hours: float) -> bool:
    return time.time() - path.stat().st_mtime <= max_age_hours * 3600


def _read_index(folder: Path) -> dict[str, dict]:
    """The metadata index as {ticker: entry}. Missing or corrupt: an empty index."""
    try:
        index = json.loads((folder / INDEX_FILE).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return index if isinstance(index, dict) else {}


def _update_index(folder: Path, ticker: str, entry: dict) -> None:
    index = _read_index(folder)
    index[ticker] = entry
    _atomic_write_text(folder / INDEX_FILE, json.dumps(index, indent=2, sort_keys=True))


def _entry_date(entry: dict, key: str) -> date | None:
    """A date stored in an index entry, or None if missing or malformed."""
    try:
        return date.fromisoformat(entry[key])
    except (KeyError, TypeError, ValueError):
        return None


def _make_entry(ticker: str, prices: pd.Series, refreshed: datetime, source: str,
                currency: str | None, history_start: date | None) -> dict:
    return {
        "ticker": ticker,
        "first_date": prices.index[0].date().isoformat(),
        "last_date": prices.index[-1].date().isoformat(),
        "rows": int(len(prices)),
        "last_refresh_utc": refreshed.astimezone(timezone.utc).isoformat(timespec="seconds"),
        "source": source,
        "currency": currency,
        "history_start": history_start.isoformat() if history_start else None,
    }


# ---------------------------------------------------------------- coverage and merging
def _covers(cached: pd.Series, start: date, entry: dict) -> bool:
    """Does the cached history reach back to (about) the requested start date?

    If we know the stock's first trading day (history_start), nothing earlier exists, so a
    cache that starts there covers any earlier request too. This stops young stocks from being
    re-downloaded on every call.
    """
    needed = start
    history_start = _entry_date(entry, "history_start")
    if history_start is not None:
        needed = max(start, history_start)
    return cached.index[0] <= pd.Timestamp(needed + _SLACK)


def _merge_histories(old: pd.Series | None, new: pd.Series) -> pd.Series:
    """Combine a fresh download with the old cache without losing history.

    Normally the download already reaches back as far as the cache (we ask for that span),
    so the new data simply replaces the old. If the source returned less history, the older
    rows are kept, rescaled by the median new/old price ratio on the overlapping dates so both
    parts share one adjustment basis. Without any overlap the two cannot be aligned safely,
    and only the new data is kept.
    """
    if old is None or old.index[0] >= new.index[0] - pd.Timedelta(_SLACK):
        return new
    overlap = old.index.intersection(new.index)
    if len(overlap) == 0:
        return new
    ratio = float(np.median(new.loc[overlap].to_numpy() / old.loc[overlap].to_numpy()))
    older = old[old.index < new.index[0]] * ratio
    return pd.concat([older, new]).rename("adj_close")


def _history_start(first: date, fetch_start: date, previous: date | None) -> date | None:
    """The stock's first trading day, if this download proves it, else None.

    We asked for data from fetch_start. If the data starts well after that, the source has
    nothing earlier, so `first` is the stock's first trading day. A date learned earlier is
    kept as long as the data still starts there.
    """
    if first > fetch_start + _SLACK:
        return first
    if previous is not None and abs((first - previous).days) <= _SLACK.days:
        return first
    return None


# ---------------------------------------------------------------- results
def _window(prices: pd.Series, start: date, end: date) -> pd.Series:
    return prices.loc[pd.Timestamp(start):pd.Timestamp(end)]


def _result(ticker: str, prices: pd.Series, start: date, end: date, source: str, note: str,
            entry: dict, cleaning: dict[str, int] | None = None) -> PriceData:
    window = _window(prices, start, end)
    if len(window) < 2:
        raise ValueError(
            f"Fewer than 2 prices for {ticker} between {start} and {end}. Available data runs "
            f"from {prices.index[0].date()} to {prices.index[-1].date()}.")
    return PriceData(ticker, window, source, note, entry.get("currency"), dict(cleaning or {}))


def _download(ticker: str, start: date, end: date, today: date, folder: Path,
              cached: pd.Series | None, entry: dict,
              fetch: Callable[[str, date, date], pd.DataFrame]) -> PriceData:
    """Download, merge with the cache, save, and return the requested window.

    Raises DataUnavailableError if the download fails and TickerNotFoundError if it returns
    no usable prices.
    """
    fetch_start = start if cached is None else min(start, cached.index[0].date())
    try:
        raw = fetch(ticker, fetch_start, today + timedelta(days=1))      # end date is exclusive
    except Exception as exc:    # network down, rate limit, yfinance missing, ...
        raise DataUnavailableError(f"Download failed ({type(exc).__name__}: {exc}).") from exc
    try:
        new, report = clean_prices_with_report(raw)
    except ValueError:
        raise TickerNotFoundError(
            f"No price data returned for '{ticker}'. Check the ticker symbol.") from None

    currency = getattr(raw, "attrs", {}).get("currency")
    currency = currency if isinstance(currency, str) and currency else entry.get("currency")
    merged = _merge_histories(cached, new)
    history_start = _history_start(merged.index[0].date(), fetch_start,
                                   _entry_date(entry, "history_start"))
    new_entry = _make_entry(ticker, merged, datetime.now(timezone.utc), "network", currency,
                            history_start)
    note = "Downloaded just now."
    try:
        _write_cache(_price_path(folder, ticker), merged)
        _update_index(folder, ticker, new_entry)
    except OSError as exc:      # read-only folder, disk full, file locked: still show the data
        note += f" It could not be saved to the local cache ({exc})."
    return _result(ticker, merged, start, end, "network", note, new_entry, report)


def _fallback(ticker: str, cached: pd.Series | None, entry: dict, start: date, end: date,
              today: date, allow_sample: bool, reason: str) -> PriceData:
    if cached is not None:
        return _stale(ticker, cached, entry, start, end, reason)
    if allow_sample:
        prices = sample_prices(end=today)
        note = (f"{reason} No cached data either: showing SYNTHETIC sample prices "
                "(not real market data).")
        return PriceData(ticker, prices, "sample", note)
    raise DataUnavailableError(f"{reason} No cached data available.")


def _stale(ticker: str, cached: pd.Series, entry: dict, start: date, end: date,
           reason: str) -> PriceData:
    note = f"{reason} Showing cached data last updated {cached.index[-1].date()}."
    return _result(ticker, cached, start, end, "stale-cache", note, entry)


# ---------------------------------------------------------------- public loaders
def load_prices(
    ticker: str,
    lookback_days: int = DEFAULT_LOOKBACK_DAYS,
    cache_dir: Path | str | None = None,
    max_age_hours: float = 24.0,
    fetcher: Callable[[str, date, date], pd.DataFrame] | None = None,
    today: date | None = None,
    allow_sample: bool = True,
    *,
    start: date | str | None = None,
    end: date | str | None = None,
) -> PriceData:
    """Adjusted close prices for the last `lookback_days` calendar days, or from `start` to `end`.

    `start` (if given) replaces `lookback_days`; `end` defaults to today and is capped at today.
    The range must span at least 30 calendar days.
    """
    ticker = normalize_ticker(ticker)
    today = today or date.today()
    start_date, end_date = _resolve_window(lookback_days, today, start, end)
    fetch = fetcher or _yfinance_fetcher
    folder = resolve_cache_dir(cache_dir)
    path = _price_path(folder, ticker)
    cached = _read_cache(path) if path.exists() else None
    entry = _read_index(folder).get(ticker, {}) if cached is not None else {}

    # 1) fresh cache that covers the window
    if cached is not None and _is_fresh(path, max_age_hours) and _covers(cached, start_date, entry):
        return _result(ticker, cached, start_date, end_date, "cache",
                       "Loaded from the local cache.", entry)

    # 2) download, 3) fall back
    try:
        return _download(ticker, start_date, end_date, today, folder, cached, entry, fetch)
    except TickerNotFoundError:
        if cached is None:
            raise
        return _stale(ticker, cached, entry, start_date, end_date,
                      "The data source returned no usable prices.")
    except DataUnavailableError as exc:
        return _fallback(ticker, cached, entry, start_date, end_date, today, allow_sample, str(exc))


def refresh_prices(
    ticker: str,
    lookback_days: int = DEFAULT_LOOKBACK_DAYS,
    cache_dir: Path | str | None = None,
    fetcher: Callable[[str, date, date], pd.DataFrame] | None = None,
    today: date | None = None,
    *,
    start: date | str | None = None,
    end: date | str | None = None,
) -> PriceData:
    """Download now, ignoring how fresh the cache is, and update the cache.

    Unlike load_prices it never falls back: a failed download raises DataUnavailableError and
    an unknown ticker raises TickerNotFoundError, so the user can see the refresh did not happen.
    """
    ticker = normalize_ticker(ticker)
    today = today or date.today()
    start_date, end_date = _resolve_window(lookback_days, today, start, end)
    folder = resolve_cache_dir(cache_dir)
    path = _price_path(folder, ticker)
    cached = _read_cache(path) if path.exists() else None
    entry = _read_index(folder).get(ticker, {}) if cached is not None else {}
    return _download(ticker, start_date, end_date, today, folder, cached, entry,
                     fetcher or _yfinance_fetcher)


def list_cached(cache_dir: Path | str | None = None) -> pd.DataFrame:
    """One row per cached ticker, with the columns in LIST_COLUMNS, sorted by ticker.

    Metadata comes from index.json. A cache file without metadata (e.g. written by an older
    version) is read directly; an unreadable one is listed with source "unreadable" and 0 rows,
    so problems are visible rather than hidden.
    """
    folder = resolve_cache_dir(cache_dir)
    if not folder.is_dir():
        return pd.DataFrame(columns=LIST_COLUMNS)
    index = _read_index(folder)
    rows = []
    for path in sorted(folder.glob("*.csv")):
        try:
            ticker = normalize_ticker(path.stem)
        except ValueError:
            continue                                  # not a file this module wrote
        if ticker != path.stem:
            continue
        entry = index.get(ticker)
        if isinstance(entry, dict):
            rows.append({column: entry.get(column) for column in LIST_COLUMNS})
            continue
        prices = _read_cache(path)
        if prices is None:
            rows.append({"ticker": ticker, "rows": 0, "source": "unreadable"})
            continue
        refreshed = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
        rows.append(_make_entry(ticker, prices, refreshed, "cache (no metadata)", None, None))
    return pd.DataFrame(rows, columns=LIST_COLUMNS)


# ---------------------------------------------------------------- upload and export
def _bad_rows(mask: pd.Series, column: pd.Series) -> str:
    """'rows 3, 7 (e.g. 'abc')' for the first few flagged rows, numbered as in the file."""
    lines = [int(i) + 2 for i in mask[mask].index[:5]]      # +2: header line, 1-based numbering
    example = column[mask].iloc[0]
    more = "" if mask.sum() <= 5 else f" and {int(mask.sum()) - 5} more"
    return f"line {', '.join(map(str, lines))}{more} (e.g. '{example}')"


def load_prices_from_csv(file_or_buffer: str | Path | IO, ticker_label: str) -> PriceData:
    """Read a user's price file: a 'Date' column plus an 'Adj Close' or 'Close' column.

    Dates must be written YYYY-MM-DD (anything after the first 10 characters, such as a time,
    is ignored). Blank prices are dropped; text in the price column is an error. The result is
    cleaned like a download, needs at least MIN_VOL_RETURNS daily returns, is labelled
    "upload", and is never written to the cache.
    """
    label = normalize_ticker(ticker_label)
    try:
        frame = pd.read_csv(file_or_buffer, dtype=str, keep_default_na=False)
    except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError) as exc:
        raise ValueError(f"Could not read the file as CSV ({exc}).") from None

    columns = {str(c).strip().lower(): c for c in frame.columns}
    date_col = columns.get("date")
    price_col = next((columns[c] for c in ("adj close", "adj_close", "close") if c in columns), None)
    if date_col is None or price_col is None:
        found = ", ".join(map(str, frame.columns)) or "none"
        raise ValueError("The file needs a 'Date' column and an 'Adj Close' or 'Close' column "
                         f"(found: {found}).")

    raw_dates = frame[date_col].astype(str).str.strip()
    dates = pd.to_datetime(raw_dates.str[:10], format="%Y-%m-%d", errors="coerce")
    bad_dates = dates.isna()
    if bad_dates.any():
        raise ValueError(f"Dates must look like YYYY-MM-DD; check {_bad_rows(bad_dates, raw_dates)}.")

    raw_prices = frame[price_col].astype(str).str.strip()
    numbers = pd.to_numeric(raw_prices, errors="coerce")
    non_numeric = numbers.isna() & (raw_prices != "")
    if non_numeric.any():
        raise ValueError(f"Non-numeric prices in the '{price_col}' column; check "
                         f"{_bad_rows(non_numeric, raw_prices)}.")

    try:
        prices, report = clean_prices_with_report(
            pd.Series(numbers.to_numpy(dtype=float), index=pd.DatetimeIndex(dates)))
    except ValueError as exc:
        raise ValueError(f"The file has too few usable prices: {exc}") from None
    require_min_history(prices, MIN_VOL_RETURNS, "an uploaded price file")
    dropped = report["rows_in"] - report["rows_kept"]
    note = (f"User-supplied file: {len(prices)} prices from {prices.index[0].date()} to "
            f"{prices.index[-1].date()}" + (f", {dropped} rows dropped while cleaning" if dropped else "")
            + ". Not checked against any market source.")
    return PriceData(label, prices, "upload", note, None, report)


def prices_to_csv(data: PriceData | pd.Series) -> bytes:
    """A clean 'date,adj_close' CSV (UTF-8 bytes) for any loaded dataset, ready to download.

    The file can be read back with load_prices_from_csv.
    """
    prices = data.prices if isinstance(data, PriceData) else data
    if not isinstance(prices, pd.Series) or prices.empty:
        raise ValueError("Nothing to export: expected a non-empty price series.")
    buffer = io.StringIO()
    out = prices.rename("adj_close")
    out.index = pd.DatetimeIndex(out.index).strftime("%Y-%m-%d")
    out.to_csv(buffer, index_label="date")
    return buffer.getvalue().encode("utf-8")
