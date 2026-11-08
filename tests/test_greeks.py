import math

import pytest

from optionlab.greeks import fd_greeks, greeks, mc_delta, to_trader_units
from optionlab.pricing.black_scholes import bs_price

ANCHOR = dict(S=100, K=100, r=0.05, sigma=0.2, T=1.0)

CASES = [
    dict(S=100, K=100, r=0.05, sigma=0.2, T=1.0),
    dict(S=30, K=30, r=0.08, sigma=0.3, T=0.25),
    dict(S=50, K=80, r=0.01, sigma=0.5, T=2.0),
    dict(S=120, K=90, r=0.03, sigma=0.15, T=0.5),
    dict(S=100, K=100, r=-0.01, sigma=0.2, T=1.0),    # negative rate
]


def test_reference_values_for_the_anchor_call():
    g = greeks(**ANCHOR, option_type="call")
    assert g.delta == pytest.approx(0.6368, abs=1e-4)
    assert g.gamma == pytest.approx(0.018762, abs=1e-6)
    assert g.vega == pytest.approx(37.5240, abs=1e-3)
    assert g.theta == pytest.approx(-6.4140, abs=1e-3)
    assert g.rho == pytest.approx(53.2325, abs=1e-3)


def test_reference_values_for_the_anchor_put():
    g = greeks(**ANCHOR, option_type="put")
    assert g.delta == pytest.approx(-0.3632, abs=1e-4)
    assert g.theta == pytest.approx(-1.6579, abs=1e-3)
    assert g.rho == pytest.approx(-41.8905, abs=1e-3)


@pytest.mark.parametrize("params", CASES)
@pytest.mark.parametrize("option_type", ["call", "put"])
def test_analytic_greeks_match_finite_differences(params, option_type):
    a = greeks(**params, option_type=option_type)
    f = fd_greeks(**params, option_type=option_type)
    for name in ("delta", "gamma", "vega", "theta", "rho"):
        assert getattr(f, name) == pytest.approx(getattr(a, name), rel=1e-4, abs=1e-6), name


@pytest.mark.parametrize("params", CASES)
def test_put_call_relationships(params):
    c, p = greeks(**params, option_type="call"), greeks(**params, option_type="put")
    K, r, T = params["K"], params["r"], params["T"]
    pv_K = K * math.exp(-r * T)
    assert c.delta - p.delta == pytest.approx(1.0)
    assert c.gamma == pytest.approx(p.gamma)
    assert c.vega == pytest.approx(p.vega)
    assert c.rho - p.rho == pytest.approx(T * pv_K)
    assert c.theta - p.theta == pytest.approx(-r * pv_K)


@pytest.mark.parametrize("params", CASES)
def test_signs_and_ranges(params):
    c, p = greeks(**params, option_type="call"), greeks(**params, option_type="put")
    assert 0.0 < c.delta < 1.0
    assert -1.0 < p.delta < 0.0
    assert c.gamma > 0 and c.vega > 0
    assert c.rho > 0 and p.rho < 0


def test_deep_in_and_out_of_the_money_delta():
    assert greeks(300, 100, 0.05, 0.2, 1.0).delta == pytest.approx(1.0, abs=1e-6)
    assert greeks(20, 100, 0.05, 0.2, 1.0).delta == pytest.approx(0.0, abs=1e-6)
    assert greeks(300, 100, 0.05, 0.2, 1.0, "put").delta == pytest.approx(0.0, abs=1e-6)


def test_gamma_is_largest_near_the_money():
    atm = greeks(100, 100, 0.05, 0.2, 1.0).gamma
    assert atm > greeks(150, 100, 0.05, 0.2, 1.0).gamma
    assert atm > greeks(60, 100, 0.05, 0.2, 1.0).gamma


def test_delta_is_the_slope_of_price_against_spot():
    h = 0.5
    slope = (bs_price(100 + h, 100, 0.05, 0.2, 1.0) - bs_price(100 - h, 100, 0.05, 0.2, 1.0)) / (2 * h)
    assert slope == pytest.approx(greeks(**ANCHOR).delta, abs=1e-3)


def test_trader_units():
    raw = greeks(**ANCHOR)
    t = to_trader_units(raw)
    assert t.delta == raw.delta and t.gamma == raw.gamma
    assert t.vega == pytest.approx(raw.vega / 100)
    assert t.theta == pytest.approx(raw.theta / 365)
    assert t.rho == pytest.approx(raw.rho / 100)
    # one vol point really is worth about vega/100
    moved = bs_price(100, 100, 0.05, 0.21, 1.0) - bs_price(100, 100, 0.05, 0.20, 1.0)
    assert moved == pytest.approx(t.vega, abs=2e-3)


def test_near_the_limits_values_stay_finite():
    for kwargs in (dict(T=1e-6), dict(sigma=1e-4)):
        g = greeks(**{**ANCHOR, **kwargs})
        assert all(math.isfinite(x) for x in (g.delta, g.gamma, g.vega, g.theta, g.rho))


@pytest.mark.parametrize("option_type", ["call", "put"])
def test_mc_delta_matches_analytic_delta(option_type):
    est, se = mc_delta(**ANCHOR, n_paths=200_000, option_type=option_type, seed=1)
    assert abs(est - greeks(**ANCHOR, option_type=option_type).delta) < 4 * se


def test_mc_delta_is_reproducible():
    assert mc_delta(**ANCHOR, n_paths=1000, seed=3) == mc_delta(**ANCHOR, n_paths=1000, seed=3)


@pytest.mark.parametrize("bad", [
    dict(S=-1), dict(K=0), dict(sigma=-0.1), dict(T=-1),
    dict(sigma=0.0), dict(T=0.0), dict(option_type="straddle"),
])
def test_invalid_inputs_raise(bad):
    with pytest.raises(ValueError):
        greeks(**{**ANCHOR, **bad})
    with pytest.raises(ValueError):
        fd_greeks(**{**ANCHOR, **bad})


def test_mc_delta_needs_enough_paths():
    with pytest.raises(ValueError):
        mc_delta(**ANCHOR, n_paths=1)
