import math

import numpy as np
import pandas as pd
import pytest

from optionlab.simulation.gbm import simulate_gbm
from optionlab.volatility import (
    annualized_drift, drift_standard_error, ewma_volatility, historical_volatility,
    log_returns, rolling_volatility, volatility_by_lookback,
)


def _series(values):
    index = pd.bdate_range("2024-01-01", periods=len(values))
    return pd.Series(values, index=index, dtype=float)


def _gbm_series(mu, sigma, years, seed):
    n = int(years * 252)
    path = simulate_gbm(100.0, mu, sigma, T=years, n_steps=n, n_paths=1, seed=seed)[0]
    return _series(path)


def test_log_returns_values():
    r = log_returns(_series([100, 110, 99]))
    assert np.allclose(r.to_numpy(), [math.log(1.1), math.log(0.9)])
    assert len(r) == 2


def test_historical_volatility_matches_the_formula():
    prices = _gbm_series(0.05, 0.3, 2, seed=1)
    r = np.log(prices).diff().dropna().to_numpy()
    assert historical_volatility(prices) == pytest.approx(np.std(r, ddof=1) * math.sqrt(252))


def test_annualization_factor_is_configurable():
    prices = _gbm_series(0.05, 0.3, 1, seed=2)
    v252 = historical_volatility(prices)
    v365 = historical_volatility(prices, trading_days=365)
    assert v365 == pytest.approx(v252 * math.sqrt(365 / 252))


def test_recovers_the_simulated_volatility():
    prices = _gbm_series(0.05, 0.30, 40, seed=3)         # 40 years of daily data
    assert historical_volatility(prices) == pytest.approx(0.30, abs=0.01)


def test_window_uses_only_the_most_recent_returns():
    prices = _gbm_series(0.05, 0.3, 2, seed=4)
    r = np.log(prices).diff().dropna().to_numpy()
    expected = np.std(r[-21:], ddof=1) * math.sqrt(252)
    assert historical_volatility(prices, window=21) == pytest.approx(expected)


def test_rolling_volatility_shape_and_last_value():
    prices = _gbm_series(0.05, 0.3, 1, seed=5)
    n_returns = len(prices) - 1
    rolling = rolling_volatility(prices, window=21)
    assert len(rolling) == n_returns - 21 + 1
    assert rolling.iloc[-1] == pytest.approx(historical_volatility(prices, window=21))


def test_ewma_matches_a_manual_recursion():
    prices = _gbm_series(0.05, 0.3, 1, seed=6)
    r = np.log(prices).diff().dropna().to_numpy()
    lam = 0.94
    s2 = r[0] ** 2
    manual = [math.sqrt(s2 * 252)]
    for x in r[1:]:
        s2 = lam * s2 + (1 - lam) * x**2
        manual.append(math.sqrt(s2 * 252))
    assert np.allclose(ewma_volatility(prices, lam=lam).to_numpy(), manual)


def test_ewma_stays_near_true_volatility_when_volatility_is_constant():
    prices = _gbm_series(0.05, 0.25, 10, seed=7)
    ewma = ewma_volatility(prices).iloc[50:]
    assert ewma.mean() == pytest.approx(0.25, rel=0.1)


def test_ewma_reacts_faster_than_a_long_window_to_a_volatility_jump():
    rng = np.random.default_rng(0)
    calm = rng.normal(0, 0.10 / math.sqrt(252), 300)
    wild = rng.normal(0, 0.60 / math.sqrt(252), 20)
    prices = _series(100 * np.exp(np.cumsum(np.concatenate([[0.0], calm, wild]))))
    assert ewma_volatility(prices).iloc[-1] > 1.5 * historical_volatility(prices)


def test_lookback_table():
    prices = _gbm_series(0.05, 0.3, 2, seed=8)            # 504 returns
    table = volatility_by_lookback(prices, windows=(21, 63, 252, 1000))
    assert list(table["window_days"]) == [21, 63, 252, 504]       # 1000 is too long, full sample added
    assert (table["volatility"] > 0).all()
    assert table["volatility"].iloc[-1] == pytest.approx(historical_volatility(prices))


def test_drift_is_exact_for_constant_growth_and_has_zero_error():
    t = np.arange(253)
    prices = _series(100 * np.exp(0.10 * t / 252))
    assert annualized_drift(prices) == pytest.approx(0.10, abs=1e-9)
    assert drift_standard_error(prices) == pytest.approx(0.0, abs=1e-9)


def test_drift_standard_error_depends_on_years_not_sampling_frequency():
    prices = _gbm_series(0.05, 0.30, 4, seed=9)
    sigma = historical_volatility(prices)
    assert drift_standard_error(prices) == pytest.approx(sigma / math.sqrt(4), rel=0.01)


def test_drift_estimate_is_noisy():
    # with 2 years of data at 25% volatility the standard error is ~18 percentage points
    prices = _gbm_series(0.08, 0.25, 2, seed=10)
    assert drift_standard_error(prices) > 0.15


@pytest.mark.parametrize("bad_prices", [
    _series([100, 101]),                       # too short
    _series([100, -1, 102]),                   # negative price
    _series([100, 0, 102]),                    # zero price
    _series([100, float("nan"), 102]),
])
def test_bad_prices_raise(bad_prices):
    with pytest.raises(ValueError):
        historical_volatility(bad_prices)


def test_non_series_input_raises():
    with pytest.raises(ValueError):
        historical_volatility([100, 101, 102, 103])


def test_invalid_parameters_raise():
    prices = _gbm_series(0.05, 0.3, 1, seed=11)
    with pytest.raises(ValueError):
        historical_volatility(prices, window=1)
    with pytest.raises(ValueError):
        historical_volatility(prices, window=10_000)
    with pytest.raises(ValueError):
        historical_volatility(prices, trading_days=0)
    with pytest.raises(ValueError):
        rolling_volatility(prices, window=1)
    with pytest.raises(ValueError):
        ewma_volatility(prices, lam=1.0)
    with pytest.raises(ValueError):
        ewma_volatility(prices, lam=0.0)