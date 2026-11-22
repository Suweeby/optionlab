"""Data-quality audit for one price series: a table of checks marked PASS, WARN or FAIL.

FAIL = do not use the data as it is. WARN = usable, but look at it before trusting it.
Each check is simple enough to verify by hand on a tiny series (see tests/test_data_audit.py).
The audit checks the data's internal consistency; it cannot tell whether the source's prices
are correct.
"""
from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from optionlab.config import MIN_HISTORY_DAYS, MIN_VOL_RETURNS
from optionlab.data import PriceData

AUDIT_COLUMNS = ["check", "status", "observed", "criterion"]
STATUSES = ("PASS", "WARN", "FAIL")

# source -> (status, plain-language meaning)
_SOURCE_STATUS = {
    "network": ("PASS", "downloaded just now"),
    "cache": ("PASS", "local copy of a recent download"),
    "stale-cache": ("WARN", "old local copy; the latest download failed or is overdue"),
    "upload": ("WARN", "user-supplied file, not checked against any market source"),
    "sample": ("FAIL", "SYNTHETIC sample data, not market data"),
}


def _row(check: str, status: str, observed: str, criterion: str) -> dict:
    return {"check": check, "status": status, "observed": observed, "criterion": criterion}


def _missing_business_days(dates: pd.DatetimeIndex) -> np.ndarray:
    """For each pair of consecutive dates, the Mon-Fri days strictly between them.

    Exchange holidays are not known here, so a holiday counts as one missing day.
    """
    days = dates.values.astype("datetime64[D]")
    return np.busday_count(days[:-1] + np.timedelta64(1, "D"), days[1:])


def audit_prices(
    prices: pd.Series,
    source: str,
    today: date | None = None,
    cleaning: dict[str, int] | None = None,
    *,
    min_history: int = MIN_HISTORY_DAYS,
    min_vol_returns: int = MIN_VOL_RETURNS,
    max_gap_bdays: int = 7,
    big_move: float = 0.25,
    stale_bdays: int = 3,
) -> pd.DataFrame:
    """Run the quality checks on one price series; returns a table with AUDIT_COLUMNS.

    prices:   the series to check (raw or cleaned) with a DatetimeIndex.
    source:   where it came from ("network", "cache", "stale-cache", "upload", "sample").
    cleaning: the report from clean_prices_with_report, if the series was cleaned; rows the
              cleaning removed turn a PASS into a WARN, so problems are not hidden.
    """
    if not isinstance(prices, pd.Series) or not isinstance(prices.index, pd.DatetimeIndex):
        raise ValueError("prices must be a pandas Series with a DatetimeIndex.")
    if prices.empty:
        raise ValueError("The price series is empty: nothing to audit.")
    today = today or date.today()
    cleaning = cleaning or {}
    rows = []

    # 1) dates strictly increasing and unique
    n_dup = int(prices.index.duplicated().sum())
    in_order = bool(prices.index.is_monotonic_increasing)
    if n_dup or not in_order:
        observed = f"{n_dup} duplicate dates" + ("" if in_order else "; dates out of order")
        rows.append(_row("Dates", "FAIL", observed, "strictly increasing, no duplicates"))
    elif cleaning.get("duplicate_dates"):
        rows.append(_row("Dates", "WARN", f"{cleaning['duplicate_dates']} duplicate dates removed "
                         "while cleaning", "strictly increasing, no duplicates"))
    else:
        rows.append(_row("Dates", "PASS", "unique and in order", "strictly increasing, no duplicates"))

    # 2) prices finite and positive
    values = prices.to_numpy(dtype=float)
    valid = np.isfinite(values) & (values > 0)
    n_bad = int((~valid).sum())
    dropped = cleaning.get("missing_or_non_finite", 0) + cleaning.get("non_positive", 0)
    if n_bad:
        rows.append(_row("Prices", "FAIL", f"{n_bad} missing, non-finite or non-positive prices",
                         "finite and > 0"))
    elif dropped:
        rows.append(_row("Prices", "WARN", f"{dropped} bad prices removed while cleaning",
                         "finite and > 0"))
    else:
        rows.append(_row("Prices", "PASS", "all finite and positive", "finite and > 0"))

    # the remaining checks use the usable part of the series
    clean = prices[valid]
    clean = clean[~clean.index.duplicated(keep="last")].sort_index()

    # 3) enough history
    n_returns = max(len(clean) - 1, 0)
    criterion = f">= {min_history} returns for drift, >= {min_vol_returns} for volatility"
    if n_returns >= min_history:
        rows.append(_row("History", "PASS", f"{n_returns} daily returns", criterion))
    elif n_returns >= min_vol_returns:
        rows.append(_row("History", "WARN", f"{n_returns} daily returns: enough for volatility, "
                         "not for a drift estimate", criterion))
    else:
        rows.append(_row("History", "FAIL", f"{n_returns} daily returns: too few even for "
                         "volatility", criterion))

    # 4) largest calendar gap and 5) suspicious single-day moves
    gap_criterion = f"<= {max_gap_bdays} missing business days in a row"
    move_criterion = f"|log return| <= {big_move:g} (larger: unadjusted split or bad price?)"
    if len(clean) >= 2:
        missing = _missing_business_days(clean.index)
        worst = int(missing.max())
        before = clean.index[int(missing.argmax()) + 1].date()
        status = "WARN" if worst > max_gap_bdays else "PASS"
        rows.append(_row("Largest gap", status, f"{worst} missing business days (before {before})",
                         gap_criterion))

        log_returns = np.log(clean).diff().dropna()
        big = log_returns[log_returns.abs() > big_move]
        if len(big):
            listed = ", ".join(f"{d.date()} ({np.expm1(r):+.0%})" for d, r in big.iloc[:3].items())
            more = "" if len(big) <= 3 else f" and {len(big) - 3} more"
            rows.append(_row("Big moves", "WARN", f"{len(big)} day(s): {listed}{more}", move_criterion))
        else:
            largest = float(log_returns.abs().max())
            rows.append(_row("Big moves", "PASS", f"largest |log return| {largest:.3f}", move_criterion))
    else:
        rows.append(_row("Largest gap", "WARN", "fewer than 2 usable prices", gap_criterion))
        rows.append(_row("Big moves", "WARN", "fewer than 2 usable prices", move_criterion))

    # 6) staleness: business days between the last price and today (today itself excluded)
    last = clean.index[-1].date() if len(clean) else prices.index.max().date()
    lag = 0
    if last < today:
        lag = int(np.busday_count(np.datetime64(last) + np.timedelta64(1, "D"), np.datetime64(today)))
    rows.append(_row("Staleness", "PASS" if lag <= stale_bdays else "WARN",
                     f"last price {last}, {lag} business days before {today}",
                     f"<= {stale_bdays} business days old"))

    # 7) source
    status, meaning = _SOURCE_STATUS.get(source, ("WARN", "unknown source"))
    rows.append(_row("Source", status, f"{source}: {meaning}", "downloaded market data"))
    return pd.DataFrame(rows, columns=AUDIT_COLUMNS)


def audit_price_data(data: PriceData, today: date | None = None) -> pd.DataFrame:
    """audit_prices for a loaded dataset, using its source and cleaning report."""
    return audit_prices(data.prices, data.source, today, data.cleaning)


def count_statuses(table: pd.DataFrame) -> dict[str, int]:
    """How many checks are PASS, WARN and FAIL, e.g. {"PASS": 5, "WARN": 2, "FAIL": 0}."""
    return {status: int((table["status"] == status).sum()) for status in STATUSES}
