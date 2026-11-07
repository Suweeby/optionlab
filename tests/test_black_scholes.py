import math

import pytest

from optionlab.pricing.black_scholes import (
    bs_price, call_price, d1, d2, put_price,
)

ANCHOR = dict(S=100, K=100, r=0.05, sigma=0.2, T=1.0)

CASES = [
    dict(S=100, K=100, r=0.05, sigma=0.2, T=1.0),
    dict(S=30, K=30, r=0.08, sigma=0.3, T=0.25),
    dict(S=50, K=80, r=0.01, sigma=0.5, T=2.0),
    dict(S=120, K=90, r=0.03, sigma=0.15, T=0.5),
    dict(S=100, K=100, r=-0.01, sigma=0.2, T=1.0),   # negative rate is allowed
]


def test_reference_values():
    assert call_price(**ANCHOR) == pytest.approx(10.4506, abs=1e-4)
    assert put_price(**ANCHOR) == pytest.approx(5.5735, abs=1e-4)


@pytest.mark.parametrize("p", CASES)
def test_put_call_parity(p):
    lhs = call_price(**p) - put_price(**p)
    rhs = p["S"] - p["K"] * math.exp(-p["r"] * p["T"])
    assert lhs == pytest.approx(rhs, abs=1e-10)


@pytest.mark.parametrize("p", CASES)
def test_no_arbitrage_bounds(p):
    S, K, r, T = p["S"], p["K"], p["r"], p["T"]
    pv_K = K * math.exp(-r * T)
    c, pu = call_price(**p), put_price(**p)
    assert max(S - pv_K, 0) - 1e-12 <= c <= S
    assert max(pv_K - S, 0) - 1e-12 <= pu <= pv_K


def test_d2_is_d1_minus_sigma_root_t():
    a = d1(**ANCHOR)
    assert d2(**ANCHOR) == pytest.approx(a - 0.2 * math.sqrt(1.0))


def test_ross_omega_equals_d1():
    S, K, r, sigma, T = 100, 95, 0.04, 0.3, 0.75
    omega = (r * T + sigma**2 * T / 2 - math.log(K / S)) / (sigma * math.sqrt(T))
    assert d1(S, K, r, sigma, T) == pytest.approx(omega)


def test_expiry_returns_intrinsic_value():
    assert call_price(110, 100, 0.05, 0.2, 0) == 10.0
    assert put_price(90, 100, 0.05, 0.2, 0) == 10.0
    assert call_price(90, 100, 0.05, 0.2, 0) == 0.0


def test_price_continuous_as_T_goes_to_zero():
    assert call_price(110, 100, 0.05, 0.2, 1e-9) == pytest.approx(10.0, abs=1e-4)
    assert put_price(110, 100, 0.05, 0.2, 1e-9) == pytest.approx(0.0, abs=1e-6)


def test_zero_volatility_is_discounted_forward_payoff():
    S, K, r, T = 100, 90, 0.05, 1.0
    expected = max(S - K * math.exp(-r * T), 0)
    assert call_price(S, K, r, 0.0, T) == pytest.approx(expected)
    assert call_price(S, K, r, 1e-8, T) == pytest.approx(expected, abs=1e-6)


def test_deep_in_and_out_of_the_money():
    # deep ITM call ~ S - K e^(-rT); deep OTM call ~ 0; mirror for puts
    assert call_price(300, 100, 0.05, 0.2, 1) == pytest.approx(300 - 100 * math.exp(-0.05), abs=1e-6)
    assert call_price(20, 100, 0.05, 0.2, 1) == pytest.approx(0.0, abs=1e-6)
    assert put_price(300, 100, 0.05, 0.2, 1) == pytest.approx(0.0, abs=1e-6)


def test_call_increases_with_volatility():
    assert call_price(100, 100, 0.05, 0.4, 1) > call_price(100, 100, 0.05, 0.2, 1)


@pytest.mark.parametrize("bad", [
    dict(S=-1), dict(S=0), dict(K=-5), dict(K=0),
    dict(sigma=-0.1), dict(T=-1), dict(S=float("nan")), dict(r=float("inf")),
])
def test_invalid_inputs_raise(bad):
    with pytest.raises(ValueError):
        call_price(**{**ANCHOR, **bad})
    with pytest.raises(ValueError):
        put_price(**{**ANCHOR, **bad})


def test_d1_undefined_at_zero_T_or_sigma():
    with pytest.raises(ValueError):
        d1(100, 100, 0.05, 0.2, 0)
    with pytest.raises(ValueError):
        d1(100, 100, 0.05, 0.0, 1)


def test_bs_price_dispatch():
    assert bs_price(**ANCHOR, option_type="put") == put_price(**ANCHOR)
    with pytest.raises(ValueError):
        bs_price(**ANCHOR, option_type="straddle")