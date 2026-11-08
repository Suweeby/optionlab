"""Smoke tests: the Streamlit pages run without crashing and show sensible numbers."""
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