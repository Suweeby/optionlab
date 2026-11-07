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
    assert len(at.metric) == 3
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