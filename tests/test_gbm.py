import math

import numpy as np
import pytest

from optionlab.pricing.black_scholes import d2
from optionlab.simulation.gbm import simulate_gbm, simulate_terminal_prices, time_grid
from optionlab.simulation.stats import (
    gbm_log_return_theory, gbm_terminal_mean, log_return_stats, prob_above,
)

S0, MU, SIGMA, T = 100.0, 0.08, 0.2, 1.0
N = 200_000


def test_shape_and_initial_column():
    paths = simulate_gbm(S0, MU, SIGMA, T, n_steps=10, n_paths=50, seed=1)
    assert paths.shape == (50, 11)
    assert (paths[:, 0] == S0).all()


def test_prices_are_positive():
    paths = simulate_gbm(S0, MU, 0.8, 5.0, n_steps=50, n_paths=1000, seed=1)
    assert (paths > 0).all()


def test_same_seed_reproduces_and_different_seed_differs():
    a = simulate_gbm(S0, MU, SIGMA, T, 5, 100, seed=7)
    b = simulate_gbm(S0, MU, SIGMA, T, 5, 100, seed=7)
    c = simulate_gbm(S0, MU, SIGMA, T, 5, 100, seed=8)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


def test_generator_object_is_accepted_and_advances():
    rng = np.random.default_rng(3)
    a = simulate_gbm(S0, MU, SIGMA, T, 5, 100, seed=rng)
    b = simulate_gbm(S0, MU, SIGMA, T, 5, 100, seed=rng)
    assert not np.array_equal(a, b)


def test_zero_volatility_is_deterministic_growth():
    paths = simulate_gbm(S0, MU, 0.0, 2.0, n_steps=8, n_paths=3, seed=1)
    expected = S0 * np.exp(MU * time_grid(2.0, 8))
    assert np.allclose(paths, expected)


def test_log_return_mean_and_variance_match_theory():
    ST = simulate_terminal_prices(S0, MU, SIGMA, T, N, seed=42)
    sample = log_return_stats(ST, S0)
    theory = gbm_log_return_theory(MU, SIGMA, T)
    se_mean = math.sqrt(theory["variance"] / N)
    assert abs(sample["mean"] - theory["mean"]) < 4 * se_mean
    assert sample["variance"] == pytest.approx(theory["variance"], rel=0.02)


def test_terminal_distribution_does_not_depend_on_number_of_steps():
    # exact discretization: 1 step and 100 steps give the same law for S_T
    theory = gbm_log_return_theory(MU, SIGMA, T)
    for n_steps in (1, 100):
        ST = simulate_gbm(S0, MU, SIGMA, T, n_steps, 50_000, seed=5)[:, -1]
        s = log_return_stats(ST, S0)
        assert s["mean"] == pytest.approx(theory["mean"], abs=4 * math.sqrt(theory["variance"] / 50_000))
        assert s["variance"] == pytest.approx(theory["variance"], rel=0.04)


@pytest.mark.parametrize("drift", [0.0, 0.08, -0.05])
def test_expected_terminal_price_is_s0_times_exp_drift_t(drift):
    ST = simulate_terminal_prices(S0, drift, SIGMA, T, N, seed=11)
    se = ST.std(ddof=1) / math.sqrt(N)
    assert abs(ST.mean() - gbm_terminal_mean(S0, drift, T)) < 4 * se


def test_risk_neutral_discounted_price_is_a_martingale():
    r = 0.05
    ST = simulate_terminal_prices(S0, r, SIGMA, T, N, seed=21)
    disc = math.exp(-r * T) * ST
    assert abs(disc.mean() - S0) < 4 * disc.std(ddof=1) / math.sqrt(N)


def test_risk_neutral_prob_above_strike_matches_N_d2():
    from scipy.stats import norm
    r, K = 0.05, 105.0
    ST = simulate_terminal_prices(S0, r, SIGMA, T, N, seed=31)
    p = float(norm.cdf(d2(S0, K, r, SIGMA, T)))
    se = math.sqrt(p * (1 - p) / N)
    assert abs(prob_above(ST, K) - p) < 4 * se


def test_antithetic_pairs_have_opposite_shocks():
    paths = simulate_gbm(S0, MU, SIGMA, T, n_steps=1, n_paths=1000, seed=2, antithetic=True)
    ST = paths[:, -1]
    pair_sum = np.log(ST[:500] / S0) + np.log(ST[500:] / S0)
    assert np.allclose(pair_sum, 2 * (MU - 0.5 * SIGMA**2) * T)


def test_antithetic_needs_even_number_of_paths():
    with pytest.raises(ValueError):
        simulate_gbm(S0, MU, SIGMA, T, 1, 101, antithetic=True)


def test_terminal_helper_matches_one_step_paths():
    a = simulate_terminal_prices(S0, MU, SIGMA, T, 100, seed=9)
    b = simulate_gbm(S0, MU, SIGMA, T, 1, 100, seed=9)[:, -1]
    assert np.array_equal(a, b)


def test_time_grid():
    g = time_grid(2.0, 4)
    assert np.allclose(g, [0.0, 0.5, 1.0, 1.5, 2.0])


@pytest.mark.parametrize("bad", [
    dict(S0=0), dict(S0=-1), dict(sigma=-0.1), dict(T=0), dict(T=-1),
    dict(n_steps=0), dict(n_steps=2.5), dict(n_paths=0), dict(drift=float("nan")),
])
def test_invalid_inputs_raise(bad):
    args = dict(S0=S0, drift=MU, sigma=SIGMA, T=T, n_steps=5, n_paths=10)
    with pytest.raises(ValueError):
        simulate_gbm(**{**args, **bad})
