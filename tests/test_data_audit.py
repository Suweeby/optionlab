"""Data-quality audit, checked on tiny hand-made series whose answers are easy to verify."""
from datetime import date

import numpy as np
import pandas as pd
import pytest

from optionlab.audit.data_audit import (
    AUDIT_COLUMNS, STATUSES, audit_price_data, audit_prices, count_statuses,
)
from optionlab.data import PriceData

MONDAY = pd.Timestamp("2026-01-05")


def _series(values, dates=None):
    dates = pd.DatetimeIndex(dates) if dates is not None else pd.bdate_range(MONDAY, periods=len(values))
    return pd.Series(values, index=dates, dtype=float)


def _smooth(n):
    """n prices growing 0.05% a day: no gaps, no big moves."""
    return _series(100 * np.exp(0.0005 * np.arange(n)))


def _status(table, check):
    return table.set_index("check").loc[check, "status"]


def _observed(table, check):
    return table.set_index("check").loc[check, "observed"]


# ---------------------------------------------------------------- shape and the clean case
def test_clean_long_series_passes_everything():
    prices = _smooth(300)
    table = audit_prices(prices, "network", today=prices.index[-1].date())
    assert list(table.columns) == AUDIT_COLUMNS
    assert list(table["check"]) == ["Dates", "Prices", "History", "Largest gap", "Big moves",
                                    "Staleness", "Source"]
    assert set(table["status"]) == {"PASS"}
    assert count_statuses(table) == {"PASS": 7, "WARN": 0, "FAIL": 0}


# ---------------------------------------------------------------- dates
def test_duplicate_dates_fail():
    dates = list(pd.bdate_range(MONDAY, periods=30))
    dates[5] = dates[4]
    table = audit_prices(_series(np.linspace(100, 110, 30), dates), "network")
    assert _status(table, "Dates") == "FAIL" and "1 duplicate" in _observed(table, "Dates")


def test_dates_out_of_order_fail():
    dates = list(pd.bdate_range(MONDAY, periods=30))
    dates[3], dates[4] = dates[4], dates[3]
    table = audit_prices(_series(np.linspace(100, 110, 30), dates), "network")
    assert _status(table, "Dates") == "FAIL" and "out of order" in _observed(table, "Dates")


def test_duplicates_removed_by_cleaning_are_a_warning():
    table = audit_prices(_smooth(30), "upload", cleaning={"duplicate_dates": 2})
    assert _status(table, "Dates") == "WARN"


# ---------------------------------------------------------------- prices
@pytest.mark.parametrize("bad", [0.0, -5.0, np.nan, np.inf])
def test_non_positive_or_non_finite_prices_fail(bad):
    values = np.linspace(100, 110, 30)
    values[10] = bad
    table = audit_prices(_series(values), "network")
    assert _status(table, "Prices") == "FAIL"


def test_prices_removed_by_cleaning_are_a_warning():
    table = audit_prices(_smooth(30), "upload", cleaning={"missing_or_non_finite": 2, "non_positive": 1})
    assert _status(table, "Prices") == "WARN" and "3 bad prices" in _observed(table, "Prices")


# ---------------------------------------------------------------- history
@pytest.mark.parametrize("n_prices, expected", [(253, "PASS"),     # 252 returns: drift OK
                                                (252, "WARN"),     # 251 returns
                                                (22, "WARN"),      # 21 returns: volatility OK
                                                (21, "FAIL")])     # 20 returns
def test_history_thresholds(n_prices, expected):
    assert _status(audit_prices(_smooth(n_prices), "network"), "History") == expected


# ---------------------------------------------------------------- gaps
def test_gap_of_seven_missing_business_days_passes_and_eight_warns():
    # Fri 9 Jan -> Wed 21 Jan skips 12-16 and 19-20 Jan: 7 business days
    base = list(pd.bdate_range(MONDAY, "2026-01-09"))
    seven = base + list(pd.bdate_range("2026-01-21", periods=25))
    eight = base + list(pd.bdate_range("2026-01-22", periods=25))
    t7 = audit_prices(_series(np.full(len(seven), 100.0), seven), "network")
    t8 = audit_prices(_series(np.full(len(eight), 100.0), eight), "network")
    assert _status(t7, "Largest gap") == "PASS" and _observed(t7, "Largest gap").startswith("7 ")
    assert _status(t8, "Largest gap") == "WARN" and "before 2026-01-22" in _observed(t8, "Largest gap")


def test_weekends_are_not_gaps():
    assert _observed(audit_prices(_smooth(30), "network"), "Largest gap").startswith("0 ")


# ---------------------------------------------------------------- big moves
def test_a_fifty_percent_jump_is_flagged_with_its_date():
    values = np.full(30, 100.0)
    values[10:] = 150.0                                  # log(1.5) = 0.405 > 0.25
    table = audit_prices(_series(values), "network")
    assert _status(table, "Big moves") == "WARN"
    assert f"{_series(values).index[10].date()} (+50%)" in _observed(table, "Big moves")


def test_a_twenty_percent_move_is_not_flagged():
    values = np.full(30, 100.0)
    values[10:] = 120.0                                  # log(1.2) = 0.18
    assert _status(audit_prices(_series(values), "network"), "Big moves") == "PASS"


# ---------------------------------------------------------------- staleness
def test_staleness_counts_business_days_only():
    prices = _series(np.linspace(100, 110, 30), pd.bdate_range(end="2026-11-20", periods=30))  # Fri
    monday = audit_prices(prices, "network", today=date(2026, 11, 23))
    later = audit_prices(prices, "network", today=date(2026, 12, 4))
    assert _status(monday, "Staleness") == "PASS" and ", 0 business days" in _observed(monday, "Staleness")
    assert _status(later, "Staleness") == "WARN" and ", 9 business days" in _observed(later, "Staleness")


# ---------------------------------------------------------------- source
@pytest.mark.parametrize("source, expected", [("network", "PASS"), ("cache", "PASS"),
                                              ("stale-cache", "WARN"), ("upload", "WARN"),
                                              ("sample", "FAIL"), ("mystery", "WARN")])
def test_source_status(source, expected):
    assert _status(audit_prices(_smooth(30), source), "Source") == expected


# ---------------------------------------------------------------- inputs and helpers
def test_bad_inputs_raise():
    with pytest.raises(ValueError):
        audit_prices([100, 101], "network")
    with pytest.raises(ValueError):
        audit_prices(pd.Series([100.0, 101.0]), "network")           # no DatetimeIndex
    with pytest.raises(ValueError):
        audit_prices(pd.Series([], dtype=float, index=pd.DatetimeIndex([])), "network")


def test_audit_price_data_uses_the_datasets_source_and_cleaning_report():
    data = PriceData("X", _smooth(300), "sample", cleaning={"non_positive": 1})
    table = audit_price_data(data, today=data.prices.index[-1].date())
    assert _status(table, "Source") == "FAIL" and _status(table, "Prices") == "WARN"
    assert set(table["status"]) <= set(STATUSES)
