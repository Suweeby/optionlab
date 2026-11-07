import numpy as np
import pytest

from optionlab.simulation.stats import (
    gbm_log_return_theory, gbm_terminal_mean, log_return_stats,
    prob_above, terminal_summary, value_at_risk,
)

ST = np.array([80.0, 90.0, 100.0, 110.0, 120.0])


def test_prob_above_is_strict():
    assert prob_above(ST, 100) == 0.4      # only 110 and 120
    assert prob_above(ST, 50) == 1.0
    assert prob_above(ST, 500) == 0.0


def test_summary_values():
    s = terminal_summary(ST, K=100)
    assert s["mean"] == 100.0
    assert s["min"] == 80.0 and s["max"] == 120.0
    assert s["p50"] == 100.0
    assert s["prob_above_K"] == 0.4
    assert s["std"] == pytest.approx(np.std(ST, ddof=1))


def test_summary_without_strike_has_no_prob_key():
    assert "prob_above_K" not in terminal_summary(ST)


def test_value_at_risk_known_case():
    # losses vs S0=100 are [20, 10, 0, -10, -20]; the 75% quantile of loss is exactly 10
    assert value_at_risk(ST, 100, confidence=0.75) == pytest.approx(10.0)


def test_value_at_risk_is_zero_when_quantile_is_a_gain():
    assert value_at_risk(np.array([110.0, 120.0, 130.0]), 100, 0.95) == 0.0


def test_log_return_stats_and_theory():
    s = log_return_stats(np.array([100.0, 100.0]), 100.0)
    assert s["mean"] == 0.0 and s["variance"] == 0.0
    t = gbm_log_return_theory(0.08, 0.2, 1.0)
    assert t["mean"] == pytest.approx(0.06)
    assert t["variance"] == pytest.approx(0.04)


def test_terminal_mean_theory():
    assert gbm_terminal_mean(100, 0.0, 1.0) == 100.0
    assert gbm_terminal_mean(100, 0.05, 2.0) == pytest.approx(100 * np.exp(0.1))


@pytest.mark.parametrize("bad", [np.array([]), np.array([[1.0, 2.0]]), np.array([1.0, np.nan])])
def test_bad_price_arrays_raise(bad):
    with pytest.raises(ValueError):
        terminal_summary(bad)


def test_invalid_parameters_raise():
    with pytest.raises(ValueError):
        prob_above(ST, 0)
    with pytest.raises(ValueError):
        value_at_risk(ST, 100, confidence=1.0)
    with pytest.raises(ValueError):
        value_at_risk(ST, 0)
    with pytest.raises(ValueError):
        log_return_stats(ST, -1)