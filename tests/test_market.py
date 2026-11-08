import math
import os
import sys
import time
import types
from datetime import date, datetime

import numpy as np
import pandas as pd
import pytest

from optionlab.data import DataUnavailableError
from optionlab.market import (
    CHAIN_COLUMNS, atm_implied_vol, clean_quotes, compare_quotes, list_expiries,
    load_option_chain, only_otm, parse_expiry, standardize_chain, time_to_expiry,
)
from optionlab.pricing.black_scholes import bs_price

TODAY = date(2026, 11, 6)
NOW = datetime(2026, 11, 6, 12, 0)
SPOT, R, SIGMA = 100.0, 0.04, 0.30
EXPIRY = "2026-12-06"
T = 30 / 365


def raw_side(option_type, strikes, rel=0.01, stamp=None):
    prices = [bs_price(SPOT, k, R, SIGMA, T, option_type) for k in strikes]
    stamp = stamp or pd.Timestamp("2026-11-06 10:00", tz="America/New_York")
    return pd.DataFrame({
        "contractSymbol": [f"X{option_type[0]}{int(k)}" for k in strikes], "strike": strikes,
        "bid": [p * (1 - rel) for p in prices], "ask": [p * (1 + rel) for p in prices],
        "lastPrice": prices, "volume": 10, "openInterest": 100,
        "impliedVolatility": SIGMA, "lastTradeDate": stamp,
    })


STRIKES = [90.0, 95.0, 100.0, 105.0, 110.0]


class FakeSource:
    def __init__(self, fail=False):
        self.fail, self.chain_calls, self.expiry_calls = fail, 0, 0

    def expiries(self, ticker):
        self.expiry_calls += 1
        if self.fail:
            raise ConnectionError("offline")
        return [EXPIRY, "2026-11-01", "2027-01-15"]            # one is already in the past

    def chain(self, ticker, expiry):
        self.chain_calls += 1
        if self.fail:
            raise ConnectionError("offline")
        return raw_side("call", STRIKES), raw_side("put", STRIKES)


def age(path, minutes):
    old = time.time() - minutes * 60
    os.utime(path, (old, old))


# ---------------------------------------------------------------- dates
def test_time_to_expiry_and_parsing():
    assert time_to_expiry("2026-12-06", today=TODAY) == pytest.approx(30 / 365)
    assert parse_expiry(date(2027, 1, 1)) == date(2027, 1, 1)
    with pytest.raises(ValueError):
        time_to_expiry("2026-11-06", today=TODAY)              # today
    with pytest.raises(ValueError):
        time_to_expiry("2026-10-01", today=TODAY)              # past
    for bad in ("../x", "2026-13-40", "tomorrow", "20261206", 5):
        with pytest.raises(ValueError):
            parse_expiry(bad)


# ---------------------------------------------------------------- standardizing and cleaning
def test_standardize_chain_has_all_columns_and_naive_timestamps():
    chain = standardize_chain(raw_side("call", STRIKES), raw_side("put", STRIKES).drop(columns=["volume"]))
    assert list(chain.columns) == CHAIN_COLUMNS
    assert set(chain["option_type"]) == {"call", "put"} and len(chain) == 10
    assert chain["lastTradeDate"].dt.tz is None
    assert chain.loc[chain["option_type"] == "put", "volume"].isna().all()    # missing column filled


def test_clean_quotes_filters_and_reports():
    calls = raw_side("call", [80.0, 90.0, 95.0, 100.0, 105.0, 110.0, 120.0])
    calls.loc[0, "bid"] = 0.0                                   # zero bid
    calls.loc[1, ["bid", "ask"]] = [5.0, 4.0]                   # crossed
    calls.loc[2, ["bid", "ask"]] = [0.1, 5.0]                   # very wide spread
    calls.loc[3, "lastTradeDate"] = pd.Timestamp("2026-10-01", tz="UTC")   # stale
    calls.loc[4, "bid"] = np.nan                                # missing bid
    chain = standardize_chain(calls, raw_side("put", STRIKES))
    clean, report = clean_quotes(chain, option_type="call", now=NOW)
    assert list(clean["strike"]) == [110.0, 120.0]
    assert report == {"total": 7, "missing_or_zero_bid": 2, "crossed_quote": 1,
                      "wide_spread": 1, "stale_last_trade": 1, "low_open_interest": 0, "kept": 2}
    assert np.allclose(clean["mid"], (clean["bid"] + clean["ask"]) / 2)


def test_clean_quotes_open_interest_filter_and_option_type_check():
    chain = standardize_chain(raw_side("call", STRIKES), raw_side("put", STRIKES))
    chain.loc[0, "openInterest"] = 2
    clean, report = clean_quotes(chain, "call", now=NOW, min_open_interest=10)
    assert report["low_open_interest"] == 1 and len(clean) == 4
    with pytest.raises(ValueError):
        clean_quotes(chain, "straddle", now=NOW)


def test_only_otm():
    chain = standardize_chain(raw_side("call", STRIKES), raw_side("put", STRIKES))
    clean, _ = clean_quotes(chain, now=NOW)
    otm = only_otm(clean, SPOT)
    assert set(otm.loc[otm["option_type"] == "call", "strike"]) == {100.0, 105.0, 110.0}
    assert set(otm.loc[otm["option_type"] == "put", "strike"]) == {90.0, 95.0, 100.0}


# ---------------------------------------------------------------- comparison
def _compared(hist_vol):
    chain = standardize_chain(raw_side("call", STRIKES), raw_side("put", STRIKES))
    clean, _ = clean_quotes(chain, option_type="call", now=NOW)
    return compare_quotes(clean, SPOT, R, T, hist_vol)


def test_implied_vol_is_recovered_from_mid_prices():
    out = _compared(SIGMA)
    assert np.allclose(out["implied_vol"], SIGMA, atol=1e-6)
    assert out["model_in_spread"].all()                         # model at the true vol sits inside bid-ask
    assert np.allclose(out["vendor_iv"], SIGMA)


def test_model_below_market_when_historical_vol_is_lower():
    out = _compared(0.20)                                       # market was priced at 30%
    assert (out["model_minus_mid"] < 0).all()
    assert not out["model_in_spread"].any()
    assert np.allclose(out["model_minus_mid"], out["model_price"] - out["mid"])
    assert np.allclose(out["moneyness"], out["strike"] / SPOT)


def test_unsolvable_quote_gives_nan_not_an_error():
    chain = standardize_chain(raw_side("call", [80.0]), raw_side("put", [80.0]))
    chain.loc[0, ["bid", "ask"]] = [10.0, 10.4]                 # call worth less than its intrinsic ~20
    clean, _ = clean_quotes(chain, option_type="call", now=NOW)
    out = compare_quotes(clean, SPOT, R, T, 0.3)
    assert math.isnan(out.loc[0, "implied_vol"])


def test_atm_implied_vol_picks_the_nearest_strike():
    out = _compared(SIGMA)
    assert atm_implied_vol(out, 103.0)[0] == 105.0
    assert atm_implied_vol(out, 100.0) == (100.0, pytest.approx(SIGMA, abs=1e-6))
    assert atm_implied_vol(out.assign(implied_vol=np.nan), 100.0) is None


def test_compare_quotes_input_checks():
    chain = standardize_chain(raw_side("call", STRIKES), raw_side("put", STRIKES))
    clean, _ = clean_quotes(chain, "call", now=NOW)
    for bad in (dict(spot=-1), dict(T=0), dict(hist_vol=-0.1)):
        args = dict(spot=SPOT, r=R, T=T, hist_vol=0.3) | bad
        with pytest.raises(ValueError):
            compare_quotes(clean, **args)


# ---------------------------------------------------------------- loading and cache
def test_expiries_future_only_and_sorted(tmp_path):
    assert list_expiries("AAPL", cache_dir=tmp_path, source=FakeSource(), today=TODAY) == [EXPIRY, "2027-01-15"]


def test_expiries_fall_back_to_cached_chains_when_offline(tmp_path):
    load_option_chain("AAPL", EXPIRY, cache_dir=tmp_path, source=FakeSource())
    got = list_expiries("AAPL", cache_dir=tmp_path, source=FakeSource(fail=True), today=TODAY)
    assert got == [EXPIRY]
    with pytest.raises(DataUnavailableError):
        list_expiries("MSFT", cache_dir=tmp_path, source=FakeSource(fail=True), today=TODAY)


def test_chain_downloads_then_uses_fresh_cache(tmp_path):
    src = FakeSource()
    first = load_option_chain("aapl", EXPIRY, cache_dir=tmp_path, source=src)
    second = load_option_chain("AAPL", EXPIRY, cache_dir=tmp_path, source=src)
    assert first.source == "network" and second.source == "cache" and src.chain_calls == 1
    assert (tmp_path / "options" / f"AAPL_{EXPIRY}.csv").exists()
    assert len(first.chain) == len(second.chain) == 10
    assert list(second.chain.columns) == CHAIN_COLUMNS


def test_old_cache_is_refreshed_and_used_when_download_fails(tmp_path):
    load_option_chain("AAPL", EXPIRY, cache_dir=tmp_path, source=FakeSource())
    path = tmp_path / "options" / f"AAPL_{EXPIRY}.csv"
    age(path, minutes=120)
    src = FakeSource()
    assert load_option_chain("AAPL", EXPIRY, cache_dir=tmp_path, source=src).source == "network"
    age(path, minutes=120)
    stale = load_option_chain("AAPL", EXPIRY, cache_dir=tmp_path, source=FakeSource(fail=True))
    assert stale.source == "stale-cache" and "Download failed" in stale.note


def test_failure_without_cache_raises(tmp_path):
    with pytest.raises(DataUnavailableError):
        load_option_chain("AAPL", EXPIRY, cache_dir=tmp_path, source=FakeSource(fail=True))


def test_bad_ticker_or_expiry_raise_before_any_io(tmp_path):
    src = FakeSource()
    with pytest.raises(ValueError):
        load_option_chain("../evil", EXPIRY, cache_dir=tmp_path, source=src)
    with pytest.raises(ValueError):
        load_option_chain("AAPL", "../../etc", cache_dir=tmp_path, source=src)
    assert src.chain_calls == 0 and list(tmp_path.iterdir()) == []


def test_default_source_calls_yfinance(tmp_path, monkeypatch):
    seen = {}

    class FakeTicker:
        def __init__(self, symbol):
            seen["symbol"] = symbol
            self.options = (EXPIRY,)

        def option_chain(self, expiry):
            seen["expiry"] = expiry
            return types.SimpleNamespace(calls=raw_side("call", STRIKES), puts=raw_side("put", STRIKES))

    monkeypatch.setitem(sys.modules, "yfinance", types.SimpleNamespace(Ticker=FakeTicker))
    assert list_expiries("msft", cache_dir=tmp_path, today=TODAY) == [EXPIRY]
    data = load_option_chain("msft", EXPIRY, cache_dir=tmp_path)
    assert seen == {"symbol": "MSFT", "expiry": EXPIRY} and data.source == "network"
