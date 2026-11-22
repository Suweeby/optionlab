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


# ================================================================ Phase 1: data layer upgrade
import io  # noqa: E402
import json  # noqa: E402

from optionlab import data as data_module  # noqa: E402
from optionlab.data import (  # noqa: E402
    LIST_COLUMNS, clean_prices_with_report, list_cached, load_prices_from_csv, prices_to_csv,
    refresh_prices, require_min_history,
)


class YoungFetcher(FakeFetcher):
    """A stock that only started trading on `first`: nothing earlier is ever returned."""

    def __init__(self, first, **kwargs):
        super().__init__(**kwargs)
        self.first = first

    def __call__(self, ticker, start, end):
        frame = super().__call__(ticker, max(start, self.first), end)
        self.calls[-1] = (ticker, start, end)              # record what was asked for
        return frame


class TruncatingFetcher(FakeFetcher):
    """Returns at most the last `days` calendar days, on a different adjustment basis."""

    def __init__(self, days, scale, **kwargs):
        super().__init__(**kwargs)
        self.days, self.scale = days, scale

    def __call__(self, ticker, start, end):
        frame = super().__call__(ticker, start, end)
        cutoff = pd.Timestamp(end - timedelta(days=self.days)).tz_localize("America/New_York")
        frame = frame[frame.index >= cutoff].copy()
        frame["Close"] = frame["Close"] * self.scale
        return frame


class CurrencyFetcher(FakeFetcher):
    """Like the real adapter, reports the currency in frame.attrs."""

    def __call__(self, ticker, start, end):
        frame = super().__call__(ticker, start, end)
        frame.attrs["currency"] = "USD"
        return frame


def _rows(path):
    return len(pd.read_csv(path))


# ---------------------------------------------------------------- defect 1: young stocks
def test_young_stock_is_downloaded_once_not_on_every_call(tmp_path):
    fetch = YoungFetcher(first=date(2026, 6, 1))
    for _ in range(3):
        data = load_prices("NEWCO", cache_dir=tmp_path, fetcher=fetch, today=TODAY)
    assert len(fetch.calls) == 1
    assert data.source == "cache"
    assert list_cached(tmp_path).loc[0, "history_start"] == "2026-06-01"


def test_young_stock_first_date_survives_a_short_refresh(tmp_path):
    fetch = YoungFetcher(first=date(2026, 6, 1))
    load_prices("NEWCO", cache_dir=tmp_path, fetcher=fetch, today=TODAY)
    _age_file(tmp_path / "NEWCO.csv", hours=48)
    load_prices("NEWCO", lookback_days=60, cache_dir=tmp_path, fetcher=fetch, today=TODAY)
    load_prices("NEWCO", lookback_days=730, cache_dir=tmp_path, fetcher=fetch, today=TODAY)
    assert len(fetch.calls) == 2                       # the long request after it hits the cache


# ---------------------------------------------------------------- defect 2: no lost history
def test_short_refresh_of_a_stale_cache_keeps_the_full_history(tmp_path):
    fetch = FakeFetcher()
    load_prices("AAPL", lookback_days=730, cache_dir=tmp_path, fetcher=fetch, today=TODAY)
    rows_before = _rows(tmp_path / "AAPL.csv")
    first_cached = pd.read_csv(tmp_path / "AAPL.csv")["date"].iloc[0]
    _age_file(tmp_path / "AAPL.csv", hours=48)
    short = load_prices("AAPL", lookback_days=60, cache_dir=tmp_path, fetcher=fetch, today=TODAY)
    assert short.source == "network"
    assert short.prices.index[0] >= pd.Timestamp(TODAY - timedelta(days=60))     # window only
    assert _rows(tmp_path / "AAPL.csv") == rows_before                           # file kept it all
    assert fetch.calls[1][1] == date.fromisoformat(first_cached)                 # asked for the span


def test_shorter_download_is_spliced_on_one_adjustment_basis(tmp_path):
    load_prices("AAPL", lookback_days=730, cache_dir=tmp_path, fetcher=FakeFetcher(), today=TODAY)
    rows_before = _rows(tmp_path / "AAPL.csv")
    _age_file(tmp_path / "AAPL.csv", hours=48)
    # the source now returns only 100 days, re-adjusted (all prices halved)
    fetch = TruncatingFetcher(days=100, scale=0.5)
    data = load_prices("AAPL", lookback_days=730, cache_dir=tmp_path, fetcher=fetch, today=TODAY)
    assert _rows(tmp_path / "AAPL.csv") == rows_before
    returns = np.log(data.prices).diff().dropna()
    assert returns.abs().max() < 0.01                  # no fake jump at the seam
    assert data.prices.iloc[0] == pytest.approx(50.0)  # old part rescaled: 100 * 0.5


def test_merge_without_overlap_keeps_only_the_new_data():
    old = pd.Series([1.0, 2.0], index=pd.to_datetime(["2020-01-01", "2020-01-02"]))
    new = pd.Series([5.0, 6.0], index=pd.to_datetime(["2026-01-01", "2026-01-02"]))
    assert data_module._merge_histories(old, new).equals(new)
    assert data_module._merge_histories(None, new).equals(new)


# ---------------------------------------------------------------- date ranges
def test_explicit_date_range(tmp_path):
    fetch = FakeFetcher()
    data = load_prices("AAPL", cache_dir=tmp_path, fetcher=fetch, today=TODAY,
                       start="2026-01-05", end=date(2026, 6, 30))
    assert fetch.calls[0][1] == date(2026, 1, 5)
    assert fetch.calls[0][2] == TODAY + timedelta(days=1)       # cache always runs to today
    assert data.prices.index[0] == pd.Timestamp("2026-01-05")
    assert data.prices.index[-1] <= pd.Timestamp("2026-06-30")


def test_end_after_today_is_capped(tmp_path):
    data = load_prices("AAPL", cache_dir=tmp_path, fetcher=FakeFetcher(), today=TODAY,
                       start="2026-01-05", end="2030-01-01")
    assert data.prices.index[-1] <= pd.Timestamp(TODAY)


@pytest.mark.parametrize("start, end", [("2026-06-30", "2026-01-05"),     # reversed
                                        ("2026-06-01", "2026-06-20"),     # under 30 days
                                        ("not-a-date", None)])
def test_bad_date_ranges_raise(tmp_path, start, end):
    with pytest.raises(ValueError):
        load_prices("AAPL", cache_dir=tmp_path, fetcher=FakeFetcher(), today=TODAY,
                    start=start, end=end)


def test_range_before_a_stock_existed_gives_a_readable_error(tmp_path):
    fetch = YoungFetcher(first=date(2026, 6, 1))
    with pytest.raises(ValueError, match="Available data runs from 2026-06-01"):
        load_prices("NEWCO", cache_dir=tmp_path, fetcher=fetch, today=TODAY,
                    start="2025-01-01", end="2025-06-01")


# ---------------------------------------------------------------- metadata index and list_cached
def test_index_records_metadata(tmp_path):
    load_prices("AAPL", cache_dir=tmp_path, fetcher=CurrencyFetcher(), today=TODAY)
    entry = json.loads((tmp_path / "index.json").read_text())["AAPL"]
    assert set(entry) == set(LIST_COLUMNS)
    assert entry["rows"] == _rows(tmp_path / "AAPL.csv")
    assert entry["source"] == "network" and entry["currency"] == "USD"
    assert entry["history_start"] is None                       # old stock: start not reached
    again = load_prices("AAPL", cache_dir=tmp_path, fetcher=FakeFetcher(), today=TODAY)
    assert again.source == "cache" and again.currency == "USD"  # currency kept on a cache hit


def test_list_cached_shows_every_ticker(tmp_path):
    for ticker in ("MSFT", "AAPL"):
        load_prices(ticker, cache_dir=tmp_path, fetcher=FakeFetcher(), today=TODAY)
    table = list_cached(tmp_path)
    assert list(table.columns) == LIST_COLUMNS
    assert list(table["ticker"]) == ["AAPL", "MSFT"]
    assert (table["rows"] > 400).all()


def test_list_cached_handles_missing_old_corrupt_and_foreign_files(tmp_path):
    assert list_cached(tmp_path / "nothing_here").empty
    load_prices("AAPL", cache_dir=tmp_path, fetcher=FakeFetcher(), today=TODAY)
    (tmp_path / "index.json").write_text("{ broken json")              # corrupt index
    (tmp_path / "BAD.csv").write_text("garbage")                       # corrupt cache file
    (tmp_path / "not a ticker.csv").write_text("x")                    # not ours: ignored
    table = list_cached(tmp_path).set_index("ticker")
    assert list(table.index) == ["AAPL", "BAD"]
    assert table.loc["AAPL", "source"] == "cache (no metadata)" and table.loc["AAPL", "rows"] > 400
    assert table.loc["BAD", "source"] == "unreadable" and table.loc["BAD", "rows"] == 0


# ---------------------------------------------------------------- refresh_prices
def test_refresh_downloads_even_when_the_cache_is_fresh(tmp_path):
    fetch = FakeFetcher()
    load_prices("AAPL", cache_dir=tmp_path, fetcher=fetch, today=TODAY)
    data = refresh_prices("AAPL", cache_dir=tmp_path, fetcher=fetch, today=TODAY)
    assert len(fetch.calls) == 2 and data.source == "network"


def test_failed_refresh_raises_and_keeps_the_cache(tmp_path):
    load_prices("AAPL", cache_dir=tmp_path, fetcher=FakeFetcher(), today=TODAY)
    before = (tmp_path / "AAPL.csv").read_text()
    with pytest.raises(DataUnavailableError, match="Download failed"):
        refresh_prices("AAPL", cache_dir=tmp_path, fetcher=FakeFetcher(fail=ConnectionError("x")),
                       today=TODAY)
    with pytest.raises(TickerNotFoundError):
        refresh_prices("AAPL", cache_dir=tmp_path, fetcher=FakeFetcher(empty=True), today=TODAY)
    assert (tmp_path / "AAPL.csv").read_text() == before


def test_unwritable_cache_still_returns_the_download(tmp_path, monkeypatch):
    def refuse(*args):
        raise PermissionError("read-only folder")
    monkeypatch.setattr(data_module, "_write_cache", refuse)
    data = load_prices("AAPL", cache_dir=tmp_path, fetcher=FakeFetcher(), today=TODAY)
    assert data.source == "network" and "could not be saved" in data.note


# ---------------------------------------------------------------- cleaning report and minimum history
def test_cleaning_report_counts_each_reason():
    index = pd.to_datetime(["2026-01-02", "2026-01-02", "2026-01-05", "2026-01-06",
                            "2026-01-07", "2026-01-08"])
    raw = pd.Series([10.0, 10.5, np.nan, np.inf, -1.0, 12.0], index=index)
    prices, report = clean_prices_with_report(raw)
    assert list(prices.values) == [10.5, 12.0]
    assert report == {"rows_in": 6, "missing_or_non_finite": 2, "non_positive": 1,
                      "duplicate_dates": 1, "rows_kept": 2}


def test_require_min_history_is_available_from_the_data_layer():
    prices = pd.Series(np.linspace(100, 110, 22), index=pd.bdate_range("2026-01-01", periods=22))
    require_min_history(prices, 21)                                    # exactly enough: no error
    with pytest.raises(ValueError, match="need at least 22 daily returns"):
        require_min_history(prices, 22, "a test")


# ---------------------------------------------------------------- CSV upload
def _csv(n=30, header="Date,Adj Close", price=lambda i: 100 + i, extra=None):
    days = pd.bdate_range("2026-01-02", periods=n)
    lines = [header] + [f"{d:%Y-%m-%d},{price(i)}" for i, d in enumerate(days)]
    if extra:
        lines += extra
    return io.StringIO("\n".join(lines) + "\n")


def test_upload_good_file(tmp_path, monkeypatch):
    monkeypatch.setenv("OPTIONLAB_CACHE_DIR", str(tmp_path))
    data = load_prices_from_csv(_csv(30), "mydata")
    assert data.ticker == "MYDATA" and data.source == "upload"
    assert data.is_user_supplied and not data.is_real
    assert len(data.prices) == 30 and data.prices.iloc[0] == 100.0
    assert "User-supplied" in data.note
    assert list(tmp_path.iterdir()) == []                              # never written to the cache


def test_upload_accepts_close_any_case_and_time_suffixes():
    days = pd.bdate_range("2026-01-02", periods=25)
    text = "date,CLOSE\n" + "\n".join(f"{d:%Y-%m-%d} 00:00:00-05:00,{50 + i}"
                                      for i, d in enumerate(days))
    data = load_prices_from_csv(io.StringIO(text), "X")
    assert data.prices.index[0] == pd.Timestamp("2026-01-02") and len(data.prices) == 25


def test_upload_prefers_adj_close_over_close():
    days = pd.bdate_range("2026-01-02", periods=25)
    text = "Date,Close,Adj Close\n" + "\n".join(f"{d:%Y-%m-%d},1,{10 + i}" for i, d in enumerate(days))
    assert load_prices_from_csv(io.StringIO(text), "X").prices.iloc[0] == 10.0


def test_upload_missing_column_names_what_was_found():
    with pytest.raises(ValueError, match=r"found: Date, Open"):
        load_prices_from_csv(_csv(30, header="Date,Open"), "X")


def test_upload_non_numeric_price_points_to_the_line():
    rows = _csv(30, price=lambda i: "abc" if i == 3 else 100 + i)
    with pytest.raises(ValueError, match=r"line 5 \(e\.g\. 'abc'\)"):      # header + 4th data row
        load_prices_from_csv(rows, "X")


def test_upload_bad_dates_and_empty_files():
    with pytest.raises(ValueError, match="YYYY-MM-DD"):
        load_prices_from_csv(io.StringIO("Date,Close\n05/01/2026,1\n06/01/2026,2\n"), "X")
    with pytest.raises(ValueError, match="Could not read"):
        load_prices_from_csv(io.StringIO(""), "X")


def test_upload_with_too_few_rows_is_refused():
    with pytest.raises(ValueError, match="Not enough history for an uploaded price file"):
        load_prices_from_csv(_csv(10), "X")


def test_upload_blank_prices_are_dropped_and_counted():
    data = load_prices_from_csv(_csv(30, extra=["2026-03-02,", "2026-03-03,"]), "X")
    assert len(data.prices) == 30
    assert data.cleaning["missing_or_non_finite"] == 2 and "2 rows dropped" in data.note


def test_upload_rejects_an_unsafe_label():
    with pytest.raises(ValueError, match="not a valid ticker"):
        load_prices_from_csv(_csv(30), "../evil")


# ---------------------------------------------------------------- export
def test_export_round_trips_through_upload(tmp_path):
    original = load_prices("AAPL", cache_dir=tmp_path, fetcher=FakeFetcher(), today=TODAY)
    blob = prices_to_csv(original)
    assert blob.decode().splitlines()[0] == "date,adj_close"
    back = load_prices_from_csv(io.BytesIO(blob), "AAPL")
    assert np.allclose(back.prices.to_numpy(), original.prices.to_numpy())
    assert (back.prices.index == original.prices.index).all()


def test_export_refuses_empty_input():
    with pytest.raises(ValueError, match="Nothing to export"):
        prices_to_csv(pd.Series(dtype=float))
