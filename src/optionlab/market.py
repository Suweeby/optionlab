"""Real option quotes: load (with cache), clean, and compare with model prices.

Quotes are cleaned as the brief says: use the mid-price (bid + ask) / 2 and discard
zero-bid, crossed, very wide, or stale quotes. Black-Scholes assumes European exercise
and no dividends, so differences from real quotes are expected (see the page's notes).
"""
from __future__ import annotations

import math
import os
import re
import time
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Protocol

import numpy as np
import pandas as pd

from optionlab.data import DataUnavailableError, normalize_ticker, resolve_cache_dir
from optionlab.implied_vol import implied_vol
from optionlab.pricing.black_scholes import bs_price

_EXPIRY_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
CHAIN_COLUMNS = ["option_type", "contractSymbol", "strike", "bid", "ask", "lastPrice",
                 "volume", "openInterest", "impliedVolatility", "lastTradeDate"]


# ---------------------------------------------------------------- dates
def parse_expiry(expiry: date | str) -> date:
    """Accept a date or a 'YYYY-MM-DD' string (strict, because it becomes part of a file name)."""
    if isinstance(expiry, datetime):
        return expiry.date()
    if isinstance(expiry, date):
        return expiry
    if not isinstance(expiry, str) or not _EXPIRY_RE.match(expiry):
        raise ValueError(f"'{expiry}' is not a valid expiry date (expected YYYY-MM-DD).")
    return date.fromisoformat(expiry)


def time_to_expiry(expiry: date | str, today: date | None = None) -> float:
    """Years until expiry, counted in calendar days / 365."""
    today = today or date.today()
    days = (parse_expiry(expiry) - today).days
    if days <= 0:
        raise ValueError("The expiry date has passed (or is today).")
    return days / 365.0


# ---------------------------------------------------------------- data source
class OptionsSource(Protocol):
    """Anything that can list expiries and return an option chain (the real source or a test double)."""

    def expiries(self, ticker: str) -> list[str]:
        """Expiry dates (YYYY-MM-DD strings) with listed options."""
        ...

    def chain(self, ticker: str, expiry: str) -> tuple[pd.DataFrame, pd.DataFrame]:
        """(calls, puts) tables for one expiry."""
        ...


class YFinanceOptionsSource:
    """Default source. yfinance is imported lazily so the rest works without it."""

    def expiries(self, ticker: str) -> list[str]:
        """Expiry dates (YYYY-MM-DD strings) with listed options."""
        import yfinance as yf

        return list(yf.Ticker(ticker).options)

    def chain(self, ticker: str, expiry: str) -> tuple[pd.DataFrame, pd.DataFrame]:
        """(calls, puts) tables for one expiry."""
        import yfinance as yf

        chain = yf.Ticker(ticker).option_chain(expiry)
        return chain.calls, chain.puts


def standardize_chain(calls: pd.DataFrame, puts: pd.DataFrame) -> pd.DataFrame:
    """Stack calls and puts into one table with the columns in CHAIN_COLUMNS."""
    parts = []
    for option_type, frame in (("call", calls), ("put", puts)):
        frame = pd.DataFrame(frame).copy()
        frame["option_type"] = option_type
        for column in CHAIN_COLUMNS:
            if column not in frame.columns:
                frame[column] = np.nan
        parts.append(frame[CHAIN_COLUMNS])
    chain = pd.concat(parts, ignore_index=True)
    stamp = pd.to_datetime(chain["lastTradeDate"], errors="coerce", utc=True)
    chain["lastTradeDate"] = stamp.dt.tz_localize(None)
    return chain


# ---------------------------------------------------------------- loading with cache
@dataclass(frozen=True)
class ChainData:
    """Calls and puts for one expiry, plus where the quotes came from."""

    ticker: str
    expiry: str
    chain: pd.DataFrame
    source: str          # "network", "cache" or "stale-cache"
    note: str = ""


def _chain_path(cache_dir: Path, ticker: str, expiry: str) -> Path:
    return cache_dir / "options" / f"{ticker}_{expiry}.csv"


def _read_chain(path: Path) -> pd.DataFrame | None:
    try:
        frame = pd.read_csv(path, parse_dates=["lastTradeDate"])
        return frame[CHAIN_COLUMNS] if len(frame) else None
    except Exception:
        return None


def _write_chain(path: Path, chain: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    chain.to_csv(temp, index=False)
    os.replace(temp, path)


def list_expiries(ticker: str, cache_dir: Path | str | None = None,
                  source: OptionsSource | None = None, today: date | None = None) -> list[str]:
    """Future expiry dates with listed options. Offline: the expiries found in the local cache."""
    ticker = normalize_ticker(ticker)
    today = today or date.today()
    source = source or YFinanceOptionsSource()
    try:
        found = [str(e) for e in source.expiries(ticker)]
    except Exception as exc:
        folder = resolve_cache_dir(cache_dir) / "options"
        found = [p.stem[len(ticker) + 1:] for p in folder.glob(f"{ticker}_*.csv")] if folder.exists() else []
        if not found:
            raise DataUnavailableError(
                f"Could not load option expiries ({type(exc).__name__}: {exc}) and nothing is cached.") from None
    valid = sorted(e for e in found if _EXPIRY_RE.match(e) and parse_expiry(e) > today)
    if not valid:
        raise DataUnavailableError(f"No listed options with a future expiry were found for '{ticker}'.")
    return valid


def load_option_chain(ticker: str, expiry: date | str, cache_dir: Path | str | None = None,
                      max_age_minutes: float = 15.0,
                      source: OptionsSource | None = None) -> ChainData:
    """Calls and puts for one expiry: fresh cache, else download, else older cache, else an error."""
    ticker = normalize_ticker(ticker)
    expiry = parse_expiry(expiry).isoformat()
    source = source or YFinanceOptionsSource()
    path = _chain_path(resolve_cache_dir(cache_dir), ticker, expiry)
    cached = _read_chain(path) if path.exists() else None

    if cached is not None and time.time() - path.stat().st_mtime <= max_age_minutes * 60:
        return ChainData(ticker, expiry, cached, "cache", "Loaded from the local cache.")

    def stale(reason: str) -> ChainData:
        if cached is None:
            raise DataUnavailableError(f"{reason} No cached quotes available.")
        age = (time.time() - path.stat().st_mtime) / 3600
        return ChainData(ticker, expiry, cached, "stale-cache",
                         f"{reason} Showing cached quotes from {age:.1f} hours ago.")

    try:
        calls, puts = source.chain(ticker, expiry)
    except Exception as exc:
        return stale(f"Download failed ({type(exc).__name__}: {exc}).")
    chain = standardize_chain(calls, puts)
    if chain.empty:
        return stale("The data source returned no quotes.")
    _write_chain(path, chain)
    return ChainData(ticker, expiry, chain, "network", "Downloaded just now.")


# ---------------------------------------------------------------- cleaning
def clean_quotes(chain: pd.DataFrame, option_type: str | None = None, now: datetime | None = None,
                 max_rel_spread: float = 0.5, max_age_days: float = 7.0,
                 min_open_interest: float = 0.0) -> tuple[pd.DataFrame, dict]:
    """Keep usable quotes and add the mid-price. Returns (clean table, report of what was dropped).

    Dropped, in this order: missing/zero bid or ask; crossed quote (ask < bid);
    wide spread ((ask-bid)/mid > max_rel_spread); last trade older than max_age_days;
    open interest below min_open_interest.
    """
    df = chain.copy()
    if option_type is not None:
        if option_type not in ("call", "put"):
            raise ValueError("option_type must be 'call' or 'put'.")
        df = df[df["option_type"] == option_type]
    for column in ("strike", "bid", "ask", "openInterest", "volume"):
        df[column] = pd.to_numeric(df[column], errors="coerce")
    df["mid"] = (df["bid"] + df["ask"]) / 2.0
    now = now or pd.Timestamp.now(tz="UTC").tz_localize(None)
    age_days = (pd.Timestamp(now) - pd.to_datetime(df["lastTradeDate"])).dt.total_seconds() / 86400.0

    report = {"total": int(len(df))}
    steps = [
        ("missing_or_zero_bid", ~((df["bid"] > 0) & (df["ask"] > 0) & (df["strike"] > 0))),
        ("crossed_quote", df["ask"] < df["bid"]),
        ("wide_spread", (df["ask"] - df["bid"]) / df["mid"] > max_rel_spread),
        ("stale_last_trade", age_days > max_age_days),
        ("low_open_interest", df["openInterest"].fillna(0) < min_open_interest),
    ]
    for name, mask in steps:
        mask = mask.reindex(df.index).fillna(False)
        report[name] = int(mask.sum())
        df = df[~mask]
        age_days = age_days.reindex(df.index)
    report["kept"] = int(len(df))
    return df.sort_values(["option_type", "strike"]).reset_index(drop=True), report


def only_otm(df: pd.DataFrame, spot: float) -> pd.DataFrame:
    """Out-of-the-money (and at-the-money) options only: calls with K >= S, puts with K <= S.
    Their prices are mostly time value, so implied volatility is best determined there."""
    keep = ((df["option_type"] == "call") & (df["strike"] >= spot)) | (
        (df["option_type"] == "put") & (df["strike"] <= spot))
    return df[keep].reset_index(drop=True)


# ---------------------------------------------------------------- comparison with the model
def compare_quotes(quotes: pd.DataFrame, spot: float, r: float, T: float, hist_vol: float) -> pd.DataFrame:
    """Model price (Black-Scholes at the historical volatility) and implied volatility for each quote."""
    if not (math.isfinite(spot) and spot > 0):
        raise ValueError("spot must be positive.")
    if T <= 0:
        raise ValueError("T must be positive.")
    if not (math.isfinite(hist_vol) and hist_vol >= 0):
        raise ValueError("hist_vol must be a non-negative number.")
    rows = []
    for q in quotes.itertuples(index=False):
        model = bs_price(spot, q.strike, r, hist_vol, T, q.option_type)
        try:
            iv = implied_vol(q.mid, spot, q.strike, r, T, q.option_type)
        except ValueError:                  # price outside no-arbitrage bounds or search range
            iv = float("nan")
        rows.append({
            "option_type": q.option_type, "strike": q.strike, "moneyness": q.strike / spot,
            "bid": q.bid, "ask": q.ask, "mid": q.mid, "model_price": model,
            "model_minus_mid": model - q.mid, "model_in_spread": bool(q.bid <= model <= q.ask),
            "implied_vol": iv, "vendor_iv": q.impliedVolatility,
            "volume": q.volume, "open_interest": q.openInterest,
        })
    return pd.DataFrame(rows)


def atm_implied_vol(compared: pd.DataFrame, spot: float) -> tuple[float, float] | None:
    """(strike, implied vol) at the strike closest to the spot price, or None if there is none."""
    usable = compared[np.isfinite(compared["implied_vol"])]
    if usable.empty:
        return None
    row = usable.iloc[(usable["strike"] - spot).abs().argmin()]
    return float(row["strike"]), float(row["implied_vol"])
