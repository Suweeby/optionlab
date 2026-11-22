"""Property-based tests: hypothesis searches for inputs that break a no-arbitrage rule.

Each test states a rule that must hold for EVERY valid option, and hypothesis tries hundreds of
inputs (and shrinks any failure to the simplest example). Inputs cover a realistic range:
S in [50, 150], K/S in [0.6, 1.6], r in [0, 10%], sigma in [5%, 80%], T in [0.05, 3] years.
"""
import math

from hypothesis import assume, given, settings
from hypothesis import strategies as hst

from optionlab.audit.model_audit import quad_price
from optionlab.greeks import greeks
from optionlab.implied_vol import implied_vol
from optionlab.pricing.black_scholes import bs_price, d1

TOL = 1e-9
options = hst.tuples(
    hst.floats(50, 150), hst.floats(0.6, 1.6), hst.floats(0.0, 0.10),
    hst.floats(0.05, 0.80), hst.floats(0.05, 3.0),
).map(lambda t: (t[0], t[0] * t[1], t[2], t[3], t[4]))          # (S, K, r, sigma, T)
many = settings(max_examples=200, deadline=None)


@many
@given(options)
def test_put_call_parity(opt):
    S, K, r, sigma, T = opt
    C, P = bs_price(*opt, "call"), bs_price(*opt, "put")
    assert abs(C - P - (S - K * math.exp(-r * T))) <= TOL * S


@many
@given(options)
def test_price_bounds(opt):
    S, K, r, sigma, T = opt
    disc_k = K * math.exp(-r * T)
    assert max(S - disc_k, 0) - TOL <= bs_price(*opt, "call") <= S + TOL
    assert max(disc_k - S, 0) - TOL <= bs_price(*opt, "put") <= disc_k + TOL


@many
@given(options, hst.floats(0.001, 0.5))
def test_call_increases_with_volatility(opt, extra):
    S, K, r, sigma, T = opt
    assert bs_price(S, K, r, sigma + extra, T) >= bs_price(*opt) - TOL


@many
@given(options, hst.floats(0.01, 2.0))
def test_call_increases_with_time(opt, extra):
    S, K, r, sigma, T = opt
    assert bs_price(S, K, r, sigma, T + extra) >= bs_price(*opt) - TOL


@many
@given(options)
def test_call_decreasing_and_convex_in_strike(opt):
    S, K, r, sigma, T = opt
    h = 0.01 * K
    down, mid, up = (bs_price(S, k, r, sigma, T) for k in (K - h, K, K + h))
    slope = (up - mid) / h
    assert -math.exp(-r * T) - TOL <= slope <= TOL
    assert down - 2 * mid + up >= -TOL


@many
@given(options)
def test_delta_gamma_vega(opt):
    g = greeks(*opt, "call")
    assert 0 <= g.delta <= 1 and g.gamma >= 0 and g.vega >= 0
    if abs(d1(*opt)) < 6:                 # further out the exact values round to 0 or 1
        assert 0 < g.delta < 1 and g.gamma > 0 and g.vega > 0


@many
@given(options, hst.sampled_from(["call", "put"]))
def test_implied_volatility_round_trip(opt, kind):
    S, K, r, sigma, T = opt
    assume(greeks(*opt, kind).vega > 1e-3)          # the price must actually depend on sigma
    price = bs_price(*opt, kind)
    assert abs(implied_vol(price, S, K, r, T, kind) - sigma) < 1e-6


@settings(max_examples=60, deadline=None)
@given(options, hst.sampled_from(["call", "put"]))
def test_quadrature_agrees_with_black_scholes(opt, kind):
    assert abs(quad_price(*opt, kind) - bs_price(*opt, kind)) < 1e-8
