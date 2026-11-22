"""Smoke tests: the Streamlit pages run without crashing and show sensible numbers."""
import sys
from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

APP = Path(__file__).resolve().parents[1] / "app"


def _pricer() -> AppTest:
    return AppTest.from_file(str(APP / "pages" / "1_Pricer.py"), default_timeout=60).run()


def test_home_page_runs():
    at = AppTest.from_file(str(APP / "streamlit_app.py"), default_timeout=60).run()
    assert not at.exception


def test_pricer_default_inputs_show_three_prices():
    at = _pricer()
    assert not at.exception
    assert len(at.metric) == 8                      # 3 prices + 5 Greeks
    assert at.metric[0].value == "10.4506"          # Black-Scholes anchor call


def test_pricer_put():
    at = _pricer()
    at.radio(key="option_type").set_value("put").run()
    assert not at.exception
    assert at.metric[0].value == "5.5735"           # Black-Scholes anchor put


def test_pricer_percent_inputs_are_converted():
    at = _pricer()
    at.number_input(key="sigma_pct").set_value(40.0).run()
    assert not at.exception
    assert float(at.metric[0].value) > 10.4506      # higher volatility -> dearer call


def test_odd_paths_with_antithetic_shows_error_not_crash():
    at = _pricer()
    at.number_input(key="n_paths").set_value(100_001).run()
    assert not at.exception
    assert len(at.error) == 1


def test_impossible_tree_probability_shows_error_not_crash():
    at = _pricer()
    at.number_input(key="n_steps").set_value(1)
    at.number_input(key="r_pct").set_value(200.0)
    at.number_input(key="sigma_pct").set_value(10.0).run()
    assert not at.exception
    assert len(at.error) == 1

def test_pricer_shows_greeks_with_correct_signs():
    at = _pricer()
    assert float(at.metric[3].value) == pytest.approx(0.6368, abs=1e-4)    # call delta
    assert float(at.metric[6].value) < 0                                   # call theta per day
    at.radio(key="option_type").set_value("put").run()
    assert float(at.metric[3].value) < 0                                   # put delta

# ---- Simulator and Convergence pages ----------------------------------------------

def _page(name: str) -> AppTest:
    return AppTest.from_file(str(APP / "pages" / name), default_timeout=120).run()


def test_simulator_real_world_default():
    at = _page("2_Simulator.py")
    assert not at.exception
    assert [m.label for m in at.metric][-1] == "95% value at risk"
    p_above = float(at.metric[0].value)
    assert p_above == pytest.approx(0.618, abs=0.02)          # N(0.3) for mu = 8%, sigma = 20%


def test_simulator_risk_neutral_mode():
    at = _page("2_Simulator.py")
    at.radio(key="sim_mode").set_value("Risk-neutral (drift = r)").run()
    assert not at.exception
    assert len(at.warning) == 1                               # the "not a prediction" warning
    labels = [m.label for m in at.metric]
    assert "95% value at risk" not in labels
    assert float(at.metric[0].value) == pytest.approx(0.5596, abs=0.02)    # N(d2) for r = 5%


def test_simulator_handles_few_paths_and_bad_inputs():
    at = _page("2_Simulator.py")
    at.number_input(key="sim_paths").set_value(10)
    at.number_input(key="sim_draw").set_value(100).run()      # more drawn than simulated: capped
    assert not at.exception and len(at.error) == 0


def test_convergence_default_page():
    at = _page("3_Convergence.py")
    assert not at.exception
    assert len(at.tabs) == 2
    assert at.metric[0].value == "10.4506"
    labels = [m.label for m in at.metric]
    assert any(label.startswith("Tree price at n = 1,000") for label in labels)


def test_convergence_put():
    at = _page("3_Convergence.py")
    at.radio(key="conv_type").set_value("put").run()
    assert not at.exception
    assert at.metric[0].value == "5.5735"

# ---- Market Data page (with a stand-in for yfinance, so no network is needed) ------

import types  # noqa: E402
from datetime import date, timedelta  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from optionlab.pricing.black_scholes import bs_price  # noqa: E402

FAKE_SPOT, FAKE_SIGMA, FAKE_R = 150.0, 0.30, 0.04
FAKE_EXPIRY = (date.today() + timedelta(days=30)).isoformat()


def _fake_yfinance(fail=False):
    class FakeTicker:
        def __init__(self, symbol):
            if fail:
                raise ConnectionError("offline")
            self.options = (FAKE_EXPIRY,)

        def history(self, start, end, auto_adjust):
            index = pd.bdate_range(start, pd.Timestamp(end) - pd.Timedelta(days=1))
            walk = np.cumsum(np.random.default_rng(0).normal(0, 0.012, len(index)))
            return pd.DataFrame({"Close": FAKE_SPOT * np.exp(walk - walk[-1])}, index=index)

        def option_chain(self, expiry):
            strikes = np.arange(120.0, 181.0, 5.0)
            T = 30 / 365
            stamp = pd.Timestamp.now(tz="UTC") - pd.Timedelta(hours=1)

            def side(kind):
                price = np.array([bs_price(FAKE_SPOT, k, FAKE_R, FAKE_SIGMA, T, kind) for k in strikes])
                return pd.DataFrame({"contractSymbol": "X", "strike": strikes, "bid": price * 0.99,
                                     "ask": price * 1.01, "lastPrice": price, "volume": 5,
                                     "openInterest": 50, "impliedVolatility": FAKE_SIGMA,
                                     "lastTradeDate": stamp})
            return types.SimpleNamespace(calls=side("call"), puts=side("put"))

    return types.SimpleNamespace(Ticker=FakeTicker)


@pytest.fixture
def market_env(tmp_path, monkeypatch):
    import streamlit as st
    st.cache_data.clear()
    monkeypatch.setenv("OPTIONLAB_CACHE_DIR", str(tmp_path))
    yield monkeypatch
    st.cache_data.clear()


def _market(**kwargs) -> AppTest:
    return AppTest.from_file(str(APP / "pages" / "4_Market_Data.py"), default_timeout=120).run()


def _percent(metric) -> float:
    return float(metric.value.strip("%")) / 100


def test_market_page_recovers_implied_volatility(market_env):
    market_env.setitem(sys.modules, "yfinance", _fake_yfinance())
    at = _market()
    assert not at.exception and len(at.error) == 0
    labels = [m.label for m in at.metric]
    assert labels[:2] == ["Spot price (last close)", "Historical volatility"]
    assert at.metric[0].value == "150.00"
    atm = next(m for m in at.metric if m.label.startswith("Implied volatility near the money"))
    assert _percent(atm) == pytest.approx(FAKE_SIGMA, abs=0.005)      # quotes were priced at 30%


def test_market_page_put_side(market_env):
    market_env.setitem(sys.modules, "yfinance", _fake_yfinance())
    at = _market()
    at.radio(key="mkt_type").set_value("put").run()
    assert not at.exception and len(at.error) == 0
    atm = next(m for m in at.metric if m.label.startswith("Implied volatility near the money"))
    assert _percent(atm) == pytest.approx(FAKE_SIGMA, abs=0.005)


def test_market_page_offline_shows_error_not_crash(market_env):
    market_env.setitem(sys.modules, "yfinance", _fake_yfinance(fail=True))
    at = _market()
    assert not at.exception
    assert len(at.error) == 1                                          # synthetic data is refused


def test_market_page_rejects_a_bad_ticker(market_env):
    market_env.setitem(sys.modules, "yfinance", _fake_yfinance())
    at = AppTest.from_file(str(APP / "pages" / "4_Market_Data.py"), default_timeout=120).run()
    at.text_input(key="mkt_ticker").set_value("../evil").run()
    assert not at.exception and len(at.error) == 1

# ---- Data page (fake yfinance, temporary cache) ------------------------------------


def _fake_yf_unknown(*unknown):
    """Like _fake_yfinance, but the given symbols return no price history (unknown tickers)."""
    base = _fake_yfinance().Ticker

    class Ticker(base):
        def __init__(self, symbol):
            super().__init__(symbol)
            self.symbol = symbol

        def history(self, start, end, auto_adjust):
            if self.symbol in unknown:
                return pd.DataFrame()
            return super().history(start, end, auto_adjust)

    return types.SimpleNamespace(Ticker=Ticker)


GOOD_CSV = ("Date,Adj Close\n" + "\n".join(
    f"{d:%Y-%m-%d},{100 + i}" for i, d in enumerate(pd.bdate_range("2026-01-02", periods=40)))).encode()


def _data_page() -> AppTest:
    return AppTest.from_file(str(APP / "pages" / "0_Data.py"), default_timeout=120).run()


def _download(at: AppTest, extra: str = "") -> AppTest:
    at.text_input(key="data_extra").set_value(extra)
    return at.button(key="data_download").click().run()


def test_data_page_with_an_empty_cache(market_env):
    market_env.setitem(sys.modules, "yfinance", _fake_yfinance())
    at = _data_page()
    assert not at.exception and len(at.error) == 0
    assert any("Nothing is cached yet" in i.value for i in at.info)


def test_data_page_downloads_and_shows_table_audit_and_export(market_env):
    market_env.setitem(sys.modules, "yfinance", _fake_yfinance())
    at = _download(_data_page(), "msft")
    assert not at.exception and len(at.error) == 0
    assert len(at.success) == 2                                         # AAPL (preset) and MSFT
    assert at.selectbox(key="data_inspect").options == ["AAPL", "MSFT"]
    assert len(at.dataframe) == 2                                       # cache table + audit table
    assert any("Data quality audit" in m.value for m in at.markdown)
    assert len(at.get("download_button")) == 1
    at.button(key="data_use").click().run()
    assert at.session_state["selected_ticker"] == "AAPL"
    assert at.session_state["selected_is_upload"] is False


def test_data_page_unknown_ticker_does_not_block_the_others(market_env):
    market_env.setitem(sys.modules, "yfinance", _fake_yf_unknown("ZZZZ"))
    at = _download(_data_page(), "ZZZZ")
    assert not at.exception
    assert len(at.success) == 1 and len(at.error) == 1
    assert "ZZZZ" in at.error[0].value and "Check the ticker symbol" in at.error[0].value


def test_data_page_offline_shows_errors_not_a_crash(market_env):
    market_env.setitem(sys.modules, "yfinance", _fake_yfinance(fail=True))
    at = _download(_data_page())
    assert not at.exception
    assert len(at.error) == 1 and "Download failed" in at.error[0].value
    assert any("Nothing is cached yet" in i.value for i in at.info)


def test_data_page_rejects_an_invalid_ticker_but_downloads_the_rest(market_env):
    market_env.setitem(sys.modules, "yfinance", _fake_yfinance())
    at = _download(_data_page(), "../evil")
    assert not at.exception
    assert len(at.error) == 1 and "not a valid ticker" in at.error[0].value
    assert len(at.success) == 1


def test_data_page_accepts_a_good_upload(market_env):
    market_env.setitem(sys.modules, "yfinance", _fake_yfinance())
    at = _data_page()
    at.file_uploader(key="data_upload").set_value(("prices.csv", GOOD_CSV, "text/csv")).run()
    assert not at.exception and len(at.error) == 0
    assert "MYDATA (upload)" in at.selectbox(key="data_inspect").options


def test_data_page_explains_a_bad_upload(market_env):
    market_env.setitem(sys.modules, "yfinance", _fake_yfinance())
    at = _data_page()
    at.file_uploader(key="data_upload").set_value(("bad.csv", b"Date,Open\n2026-01-02,1\n", "text/csv")).run()
    assert not at.exception
    assert len(at.error) == 1 and "Could not use this file" in at.error[0].value

# ---- Pages wired to the selected dataset (Phase 3) -------------------------------------

import io  # noqa: E402

from optionlab.data import load_prices_from_csv  # noqa: E402


def _with_upload(page: str) -> AppTest:
    """A page whose session already holds an uploaded dataset (40 prices, last one 139) as the selection."""
    at = AppTest.from_file(str(APP / "pages" / page), default_timeout=120)
    at.session_state["uploads"] = {"MYDATA": load_prices_from_csv(io.BytesIO(GOOD_CSV), "MYDATA")}
    at.session_state["selected_ticker"] = "MYDATA"
    at.session_state["selected_is_upload"] = True
    return at.run()


def _fake_yf_no_options():
    """Prices exist, but no options are listed (like most non-US stocks)."""
    base = _fake_yfinance().Ticker

    class Ticker(base):
        def __init__(self, symbol):
            super().__init__(symbol)
            self.options = ()

    return types.SimpleNamespace(Ticker=Ticker)


def test_pricer_market_toggle_fills_spot_and_volatility(market_env):
    market_env.setitem(sys.modules, "yfinance", _fake_yfinance())
    at = _pricer()
    assert at.toggle(key="pricer_use_market").label == "Use market data for AAPL"
    at.toggle(key="pricer_use_market").set_value(True).run()
    assert not at.exception and len(at.error) == 0
    assert any("S = 150" in m.value for m in at.markdown)            # fake market spot price
    assert any("not a forecast" in i.value for i in at.info)
    assert len(at.metric) == 8 and at.metric[0].value != "10.4506"


def test_pricer_market_toggle_refuses_synthetic_data(market_env):
    market_env.setitem(sys.modules, "yfinance", _fake_yfinance(fail=True))
    at = _pricer()
    at.toggle(key="pricer_use_market").set_value(True).run()
    assert not at.exception
    assert len(at.error) == 1 and "SYNTHETIC" in at.error[0].value
    assert at.metric[0].value == "10.4506"                          # typed inputs still used


def test_pricer_uses_an_uploaded_dataset(market_env):
    at = _with_upload("1_Pricer.py")
    assert at.toggle(key="pricer_use_market").label == "Use market data for MYDATA (upload)"
    at.toggle(key="pricer_use_market").set_value(True).run()
    assert not at.exception and len(at.error) == 0
    assert any("S = 139" in m.value for m in at.markdown)


def test_simulator_historical_drift_comes_with_its_standard_error(market_env):
    market_env.setitem(sys.modules, "yfinance", _fake_yfinance())
    at = _page("2_Simulator.py")
    at.toggle(key="sim_use_market").set_value(True).run()
    at.radio(key="sim_mu_source").set_value("Historical estimate (noisy)").run()
    assert not at.exception and len(at.error) == 0
    assert any("±" in w.value and "not a forecast" in w.value for w in at.warning)


def test_simulator_historical_drift_needs_a_year_of_data(market_env):
    at = _with_upload("2_Simulator.py")                              # only 39 returns
    at.toggle(key="sim_use_market").set_value(True).run()
    at.radio(key="sim_mu_source").set_value("Historical estimate (noisy)").run()
    assert not at.exception
    assert len(at.error) == 1 and "Not enough history for a drift estimate" in at.error[0].value
    assert len(at.metric) > 0                                        # still simulates (assumed mu)


def test_market_page_without_options_still_shows_spot_and_volatility(market_env):
    market_env.setitem(sys.modules, "yfinance", _fake_yf_no_options())
    at = _market()
    assert not at.exception and len(at.error) == 0
    assert [m.label for m in at.metric][:2] == ["Spot price (last close)", "Historical volatility"]
    assert any("No option quotes" in w.value for w in at.warning)


def test_market_page_starts_at_the_selected_ticker(market_env):
    market_env.setitem(sys.modules, "yfinance", _fake_yfinance())
    at = AppTest.from_file(str(APP / "pages" / "4_Market_Data.py"), default_timeout=120)
    at.session_state["selected_ticker"] = "MSFT"
    at.run()
    assert not at.exception and len(at.error) == 0
    assert at.text_input(key="mkt_ticker").value == "MSFT"


def _audit_page() -> AppTest:
    return AppTest.from_file(str(APP / "pages" / "5_Audit.py"), default_timeout=180).run()


def test_audit_page_before_running():
    at = _audit_page()
    assert not at.exception and len(at.error) == 0
    assert any("Run the quick audit" in i.value for i in at.info)
    assert len(at.metric) == 0


def test_audit_page_runs_the_real_quick_audit():
    at = _audit_page()
    at.button(key="audit_run").click().run()
    assert not at.exception and len(at.error) == 0
    assert [m.label for m in at.metric] == ["PASS", "WARN", "FAIL"]
    assert at.metric[2].value == "0"
    assert len(at.success) == 1 and "passed" in at.success[0].value
    assert len(at.get("download_button")) == 2


def test_audit_page_shows_a_crash_instead_of_crashing(monkeypatch):
    from optionlab.audit import report as report_module

    def boom(quick):
        raise RuntimeError("disk on fire")
    monkeypatch.setattr(report_module, "run_all", boom)
    at = _audit_page()
    at.button(key="audit_run").click().run()
    assert not at.exception
    assert len(at.error) == 1 and "RuntimeError: disk on fire" in at.error[0].value


def test_audit_page_reports_failures(monkeypatch):
    from optionlab.audit import report as report_module
    from optionlab.audit.claims import CLAIM_COLUMNS
    from optionlab.validation import COLUMNS

    table = pd.DataFrame([{"area": "Claims", "check": "a broken claim", "expected": "", "observed": "",
                           "criterion": "", "status": "FAIL"}], columns=COLUMNS)
    failing = report_module.AuditReport(table, pd.DataFrame(columns=CLAIM_COLUMNS),
                                        {"git_commit": "abc", "mode": "quick"})
    monkeypatch.setattr(report_module, "run_all", lambda quick: failing)
    at = _audit_page()
    at.button(key="audit_run").click().run()
    assert not at.exception
    assert len(at.error) == 1 and "1 check(s) failed" in at.error[0].value


def test_market_page_explains_why_an_upload_is_not_used(market_env):
    market_env.setitem(sys.modules, "yfinance", _fake_yfinance())
    at = _with_upload("4_Market_Data.py")
    assert not at.exception and len(at.error) == 0
    assert at.text_input(key="mkt_ticker").value == "AAPL"
    assert any("cannot be compared" in i.value for i in at.info)
