import math

import pytest

from optionlab.payoffs import payoff
from optionlab.pricing.binomial import binomial_price
from optionlab.pricing.black_scholes import bs_price

ARGS = dict(S=30, K=30, r=0.08, sigma=0.3, T=0.25)


def test_matches_original_script():
    assert binomial_price(**ARGS, n=100) == pytest.approx(2.0841, abs=1e-3)


def test_put_call_parity_holds_in_the_tree():
    c = binomial_price(**ARGS, n=200, option_type="call")
    p = binomial_price(**ARGS, n=200, option_type="put")
    assert c - p == pytest.approx(30 - 30 * math.exp(-0.08 * 0.25), abs=1e-9)


def test_call_never_below_intrinsic_lower_bound():
    c = binomial_price(S=50, K=30, r=0.05, sigma=0.2, T=1, n=100)
    assert c >= 50 - 30 * math.exp(-0.05)


@pytest.mark.parametrize("bad", [dict(S=-1), dict(K=0), dict(sigma=0), dict(T=-1), dict(n=0)])
def test_invalid_inputs_raise(bad):
    with pytest.raises(ValueError):
        binomial_price(**{**ARGS, "n": 100, **bad})


def test_invalid_option_type_raises():
    with pytest.raises(ValueError):
        binomial_price(**ARGS, n=10, option_type="straddle")


def test_probability_outside_unit_interval_raises():
    # huge rate, tiny volatility, one step: p > 1 would allow arbitrage
    with pytest.raises(ValueError):
        binomial_price(S=100, K=100, r=2.0, sigma=0.1, T=1.0, n=1)


@pytest.mark.parametrize("n", [1, 3, 7])
@pytest.mark.parametrize("option_type", ["call", "put"])
def test_matches_explicit_binomial_sum(n, option_type):
    """Independent check: price = e^(-rT) * sum_j C(n,j) p^j (1-p)^(n-j) * payoff_j."""
    S, K, r, sigma, T = 100, 95, 0.05, 0.3, 1.0
    dt = T / n
    u = math.exp(sigma * math.sqrt(dt))
    d = 1 / u
    p = (math.exp(r * dt) - d) / (u - d)
    total = 0.0
    for j in range(n + 1):
        ST = S * u**j * d ** (n - j)
        total += math.comb(n, j) * p**j * (1 - p) ** (n - j) * float(payoff(ST, K, option_type))
    expected = math.exp(-r * T) * total
    assert binomial_price(S, K, r, sigma, T, n, option_type) == pytest.approx(expected, rel=1e-12)


# ---- convergence to Black-Scholes -------------------------------------------------

SETS = [
    dict(S=100, K=100, r=0.05, sigma=0.2, T=1.0),
    dict(S=120, K=100, r=0.03, sigma=0.25, T=0.5),   # in the money
    dict(S=80, K=100, r=0.03, sigma=0.25, T=1.0),    # out of the money
    dict(S=100, K=110, r=0.01, sigma=0.6, T=2.0),    # high volatility
]


def _worst_error(params, option_type, n_range):
    """Largest |binomial - BS| over several consecutive n (the tree zigzags with n)."""
    bs = bs_price(**params, option_type=option_type)
    return max(abs(binomial_price(**params, n=n, option_type=option_type) - bs) for n in n_range)


@pytest.mark.parametrize("params", SETS)
@pytest.mark.parametrize("option_type", ["call", "put"])
def test_binomial_converges_to_black_scholes(params, option_type):
    early = _worst_error(params, option_type, range(50, 56))
    late = _worst_error(params, option_type, range(1000, 1006))
    assert late < 0.01
    assert late < 0.2 * early        # ~20x more steps -> error at least 5x smaller


def test_error_scales_like_one_over_n():
    p = SETS[0]
    e1 = _worst_error(p, "call", range(100, 106))
    e2 = _worst_error(p, "call", range(1000, 1006))
    assert e1 / e2 == pytest.approx(10.0, rel=0.25)   # 10x more steps -> ~10x smaller error
