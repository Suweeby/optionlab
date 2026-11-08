import os
import sys
import time
import types
from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from optionlab.data import (
    DataUnavailableError, PriceData, TickerNotFoundError, clean_prices, load_prices,
    normalize_ticker, sample_prices,
)
from optionlab.volatility import historical_volatility

TODAY = date(2026, 11, 6)


class FakeFetcher:
    """Stands in for yfinance: returns a yfinance-shaped DataFrame and counts its calls."""

    def __init__(self, fail=None, empty=False):
        self.calls = []
        self.fail = fail
        self.empty = empty

    def __call__(self, ticker, start, end):
        self.calls.append((ticker, start, end))
        if self.fail:
            raise self.fail
        if self.empty:
            return pd.DataFrame()
        index = pd.bdate_range(start, end - timedelta(days=1)).tz_localize("America/New_York")
        close = 100 + 0.1 * np.arange(len(index))
        return pd.DataFrame({"Open": close, "Close": close, "Volume": 1000}, index=index)


def _age_file(path, hours):
    old = time.time() - hours * 3600
    os.utime(path, (old, old))


# ---------------------------------------------------------------- tickers
def test_ticker_normalization():
    assert normalize_ticker(" aapl ") == "AAPL"
    for good in ("BRK-B", "^GSPC", "EURUSD=X", "RELIANCE.NS"):
        assert normalize_ticker(good) == good


@pytest.mark.parametrize("bad", ["", "  ", "../evil", "a b", "A" * 20, "AAPL;rm", "C:\\x", "/etc/passwd"])
def test_invalid_tickers_raise(bad):
    with pytest.raises(ValueError):
        normalize_ticker(bad)


# ---------------------------------------------------------------- cleaning
def test_clean_prices_handles_messy_data():
    index = pd.to_datetime(["2026-01-05", "2026-01-02", "2026-01-02", "2026-01-06", "2026-01-07", "2026-01-08"])
    index = index.tz_localize("America/New_York")
    raw = pd.DataFrame({"Close": [11.0, 10.0, 10.5, np.nan, 0.0, 12.0]}, index=index)
    out = clean_prices(raw)
    assert out.name == "adj_close"
    assert out.index.tz is None
    assert out.index.is_monotonic_increasing and out.index.is_unique
    assert list(out.index.strftime("%Y-%m-%d")) == ["2026-01-02", "2026-01-05", "2026-01-08"]
    assert list(out.values) == [10.5, 11.0, 12.0]       # duplicate date keeps the last value


def test_clean_prices_prefers_adj_close_and_accepts_series():
    index = pd.bdate_range("2026-01-01", periods=3)
    raw = pd.DataFrame({"Close": [1.0, 2.0, 3.0], "Adj Close": [10.0, 20.0, 30.0]}, index=index)
    assert list(clean_prices(raw).values) == [10.0, 20.0, 30.0]
    assert list(clean_prices(pd.Series([5.0, 6.0], index=index[:2])).values) == [5.0, 6.0]


def test_clean_prices_errors():
    index = pd.bdate_range("2026-01-01", periods=3)
    with pytest.raises(ValueError):
        clean_prices(pd.DataFrame({"Open": [1, 2, 3]}, index=index))     # no price column
    with pytest.raises(ValueError):
        clean_prices(pd.DataFrame({"Close": [1.0, np.nan, np.nan]}, index=index))   # < 2 valid
    with pytest.raises(ValueError):
        clean_prices([1, 2, 3])


# ---------------------------------------------------------------- load_prices
def test_first_load_downloads_and_writes_cache(tmp_path):
    fetch = FakeFetcher()
    data = load_prices("aapl", cache_dir=tmp_path, fetcher=fetch, today=TODAY)
    assert isinstance(data, PriceData)
    assert data.ticker == "AAPL" and data.source == "network" and data.is_real
    assert (tmp_path / "AAPL.csv").exists()
    assert len(fetch.calls) == 1
    ticker, start, end = fetch.calls[0]
    assert ticker == "AAPL"
    assert start == TODAY - timedelta(days=730)
    assert end == TODAY + timedelta(days=1)                   # yfinance end date is exclusive
    assert data.prices.index.min() >= pd.Timestamp(TODAY - timedelta(days=730))


def test_second_load_uses_the_fresh_cache(tmp_path):
    fetch = FakeFetcher()
    first = load_prices("AAPL", cache_dir=tmp_path, fetcher=fetch, today=TODAY)
    second = load_prices("AAPL", cache_dir=tmp_path, fetcher=fetch, today=TODAY)
    assert len(fetch.calls) == 1
    assert second.source == "cache"
    assert np.allclose(first.prices.to_numpy(), second.prices.to_numpy())


def test_stale_cache_triggers_a_new_download(tmp_path):
    fetch = FakeFetcher()
    load_prices("AAPL", cache_dir=tmp_path, fetcher=fetch, today=TODAY)
    _age_file(tmp_path / "AAPL.csv", hours=48)
    data = load_prices("AAPL", cache_dir=tmp_path, fetcher=fetch, today=TODAY)
    assert len(fetch.calls) == 2 and data.source == "network"


def test_cache_that_does_not_reach_back_far_enough_is_refetched(tmp_path):
    fetch = FakeFetcher()
    load_prices("AAPL", lookback_days=100, cache_dir=tmp_path, fetcher=fetch, today=TODAY)
    data = load_prices("AAPL", lookback_days=700, cache_dir=tmp_path, fetcher=fetch, today=TODAY)
    assert len(fetch.calls) == 2 and data.source == "network"


def test_corrupt_cache_is_ignored(tmp_path):
    (tmp_path / "AAPL.csv").write_text("not,a,valid\ncache file")
    fetch = FakeFetcher()
    data = load_prices("AAPL", cache_dir=tmp_path, fetcher=fetch, today=TODAY)
    assert data.source == "network" and len(fetch.calls) == 1


def test_download_failure_falls_back_to_stale_cache(tmp_path):
    load_prices("AAPL", cache_dir=tmp_path, fetcher=FakeFetcher(), today=TODAY)
    _age_file(tmp_path / "AAPL.csv", hours=72)
    broken = FakeFetcher(fail=ConnectionError("no internet"))
    data = load_prices("AAPL", cache_dir=tmp_path, fetcher=broken, today=TODAY)
    assert data.source == "stale-cache" and data.is_real
    assert "Download failed" in data.note and "ConnectionError" in data.note


def test_download_failure_without_cache_gives_labelled_synthetic_data(tmp_path):
    broken = FakeFetcher(fail=ConnectionError("no internet"))
    data = load_prices("AAPL", cache_dir=tmp_path, fetcher=broken, today=TODAY)
    assert data.source == "sample" and not data.is_real
    assert "SYNTHETIC" in data.note
    assert len(data.prices) > 100
    assert not (tmp_path / "AAPL.csv").exists()           # sample data is never cached as real


def test_sample_fallback_can_be_disabled(tmp_path):
    broken = FakeFetcher(fail=ConnectionError("no internet"))
    with pytest.raises(DataUnavailableError):
        load_prices("AAPL", cache_dir=tmp_path, fetcher=broken, today=TODAY, allow_sample=False)


def test_unknown_ticker_never_returns_sample_data(tmp_path):
    with pytest.raises(TickerNotFoundError):
        load_prices("ZZZZZ", cache_dir=tmp_path, fetcher=FakeFetcher(empty=True), today=TODAY)


def test_empty_download_with_a_cache_uses_the_cache(tmp_path):
    load_prices("AAPL", cache_dir=tmp_path, fetcher=FakeFetcher(), today=TODAY)
    _age_file(tmp_path / "AAPL.csv", hours=72)
    data = load_prices("AAPL", cache_dir=tmp_path, fetcher=FakeFetcher(empty=True), today=TODAY)
    assert data.source == "stale-cache"


def test_bad_inputs_raise_before_any_file_access(tmp_path):
    fetch = FakeFetcher()
    with pytest.raises(ValueError):
        load_prices("../evil", cache_dir=tmp_path, fetcher=fetch, today=TODAY)
    with pytest.raises(ValueError):
        load_prices("AAPL", lookback_days=5, cache_dir=tmp_path, fetcher=fetch, today=TODAY)
    assert fetch.calls == [] and list(tmp_path.iterdir()) == []


def test_default_downloader_calls_yfinance_correctly(tmp_path, monkeypatch):
    seen = {}

    class FakeTicker:
        def __init__(self, symbol):
            seen["symbol"] = symbol

        def history(self, start, end, auto_adjust):
            seen.update(start=start, end=end, auto_adjust=auto_adjust)
            index = pd.bdate_range("2026-01-01", periods=5)
            return pd.DataFrame({"Close": [1.0, 2.0, 3.0, 4.0, 5.0]}, index=index)

    monkeypatch.setitem(sys.modules, "yfinance", types.SimpleNamespace(Ticker=FakeTicker))
    data = load_prices("msft", cache_dir=tmp_path, today=TODAY)           # no fetcher: real adapter
    assert seen["symbol"] == "MSFT"
    assert seen["auto_adjust"] is True
    assert seen["start"] == (TODAY - timedelta(days=730)).isoformat()
    assert seen["end"] == (TODAY + timedelta(days=1)).isoformat()
    assert data.source == "network"


def test_missing_yfinance_is_treated_as_a_download_failure(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "yfinance", None)                    # makes `import yfinance` fail
    data = load_prices("AAPL", cache_dir=tmp_path, today=TODAY)
    assert data.source == "sample"
    assert "Download failed" in data.note


# ---------------------------------------------------------------- sample data
def test_sample_prices_are_deterministic_positive_and_have_the_stated_volatility():
    a = sample_prices(seed=3, end=TODAY)
    b = sample_prices(seed=3, end=TODAY)
    assert a.equals(b)
    assert (a > 0).all() and a.index.is_monotonic_increasing
    assert not a.equals(sample_prices(seed=4, end=TODAY))
    assert historical_volatility(a) == pytest.approx(0.25, abs=0.04)

# ---------------------------------------------------------------- cache directory
def test_resolve_cache_dir_order(tmp_path, monkeypatch):
    from optionlab.data import DEFAULT_CACHE_DIR, resolve_cache_dir
    monkeypatch.delenv("OPTIONLAB_CACHE_DIR", raising=False)
    assert resolve_cache_dir() == DEFAULT_CACHE_DIR
    monkeypatch.setenv("OPTIONLAB_CACHE_DIR", str(tmp_path))
    assert resolve_cache_dir() == tmp_path
    assert resolve_cache_dir(tmp_path / "x") == tmp_path / "x"          # explicit argument wins