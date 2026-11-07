import math

import numpy as np
import pytest

from optionlab.payoffs import payoff
from optionlab.pricing.black_scholes import bs_price, call_price, put_price
from optionlab.pricing.monte_carlo import mc_price
from optionlab.simulation.gbm import simulate_terminal_prices

ANCHOR = dict(S=100, K=100, r=0.05, sigma=0.2, T=1.0)

SETS = [
    dict(S=100, K=100, r=0.05, sigma=0.2, T=1.0),
    dict(S=120, K=100, r=0.03, sigma=0.25, T=0.5),   # in the money
    dict(S=80, K=100, r=0.03, sigma=0.25, T=1.0),    # out of the money
    dict(S=100, K=110, r=0.01, sigma=0.6, T=2.0),    # high volatility
]


def test_is_exactly_the_shared_generator_plus_a_payoff():
    """The merge point: mc_price must equal discounted mean payoff of the shared GBM draws."""
    P = SETS[1]
    ST = simulate_terminal_prices(P["S"], P["r"], P["sigma"], P["T"], 5000, seed=3)
    expected = math.exp(-P["r"] * P["T"]) * float(np.mean(payoff(ST, P["K"], "call")))
    assert mc_price(**P, n_paths=5000, seed=3).price == pytest.approx(expected, rel=1e-12)


@pytest.mark.parametrize("params", SETS)
@pytest.mark.parametrize("option_type", ["call", "put"])
@pytest.mark.parametrize("antithetic", [False, True])
def test_price_within_four_standard_errors_of_black_scholes(params, option_type, antithetic):
    res = mc_price(**params, n_paths=200_000, option_type=option_type, seed=7, antithetic=antithetic)
    bs = bs_price(**params, option_type=option_type)
    assert abs(res.price - bs) < 4 * res.std_error


def test_confidence_interval_structure():
    res = mc_price(**ANCHOR, n_paths=10_000, seed=1)
    z = 1.959963984540054
    assert res.ci_low == pytest.approx(res.price - z * res.std_error)
    assert res.ci_high == pytest.approx(res.price + z * res.std_error)
    assert res.ci_low < res.price < res.ci_high
    wider = mc_price(**ANCHOR, n_paths=10_000, seed=1, confidence=0.99)
    assert wider.ci_high - wider.ci_low > res.ci_high - res.ci_low


@pytest.mark.parametrize("antithetic", [False, True])
def test_standard_error_is_calibrated(antithetic):
    """Over many independent runs: ~95% of CIs contain the true price, and the
    reported SE matches the real spread of the estimates."""
    bs = call_price(**ANCHOR)
    rng = np.random.default_rng(123)
    results = [mc_price(**ANCHOR, n_paths=2000, seed=rng, antithetic=antithetic) for _ in range(400)]
    coverage = np.mean([r.ci_low <= bs <= r.ci_high for r in results])
    assert 0.92 <= coverage <= 0.98
    empirical_sd = np.std([r.price for r in results], ddof=1)
    mean_reported_se = np.mean([r.std_error for r in results])
    assert mean_reported_se == pytest.approx(empirical_sd, rel=0.15)


def test_antithetic_reduces_standard_error_for_a_call():
    plain = mc_price(**ANCHOR, n_paths=100_000, seed=1)
    anti = mc_price(**ANCHOR, n_paths=100_000, seed=1, antithetic=True)
    assert anti.std_error < 0.9 * plain.std_error


def test_error_shrinks_like_one_over_sqrt_n():
    se_small = mc_price(**ANCHOR, n_paths=10_000, seed=2).std_error
    se_large = mc_price(**ANCHOR, n_paths=1_000_000, seed=2).std_error
    assert se_small / se_large == pytest.approx(10.0, rel=0.1)    # 100x paths -> 10x smaller


def test_put_call_parity_holds_within_noise():
    c = mc_price(**ANCHOR, n_paths=200_000, option_type="call", seed=11)
    p = mc_price(**ANCHOR, n_paths=200_000, option_type="put", seed=12)
    forward = ANCHOR["S"] - ANCHOR["K"] * math.exp(-ANCHOR["r"] * ANCHOR["T"])
    assert abs((c.price - p.price) - forward) < 4 * math.hypot(c.std_error, p.std_error)


def test_seed_reproducibility():
    a = mc_price(**ANCHOR, n_paths=1000, seed=5)
    b = mc_price(**ANCHOR, n_paths=1000, seed=5)
    c = mc_price(**ANCHOR, n_paths=1000, seed=6)
    assert a == b
    assert a.price != c.price


def test_zero_volatility_is_deterministic():
    res = mc_price(100, 90, 0.05, 0.0, 1.0, n_paths=100, seed=1)
    assert res.price == pytest.approx(100 - 90 * math.exp(-0.05))
    assert res.std_error == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize("bad", [
    dict(S=-1), dict(K=0), dict(sigma=-0.1), dict(T=0), dict(T=-1),
    dict(n_paths=1), dict(option_type="straddle"), dict(confidence=1.0), dict(confidence=0.0),
])
def test_invalid_inputs_raise(bad):
    with pytest.raises(ValueError):
        mc_price(**{**ANCHOR, "n_paths": 1000, **bad})


def test_antithetic_input_rules():
    with pytest.raises(ValueError):
        mc_price(**ANCHOR, n_paths=1001, antithetic=True)    # odd
    with pytest.raises(ValueError):
        mc_price(**ANCHOR, n_paths=2, antithetic=True)       # too few pairs