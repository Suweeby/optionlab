import math

import pytest

from optionlab.greeks import greeks
from optionlab.implied_vol import IVResult, implied_vol, implied_vol_detailed
from optionlab.pricing.black_scholes import bs_price

MKT = dict(S=100, K=100, r=0.05, T=1.0)


def test_round_trip_for_the_brief_anchor():
    for option_type in ("call", "put"):
        price = bs_price(**MKT, sigma=0.25, option_type=option_type)
        assert implied_vol(price, **MKT, option_type=option_type) == pytest.approx(0.25, abs=1e-9)


@pytest.mark.parametrize("K", [80, 90, 100, 110, 120])
@pytest.mark.parametrize("T", [0.25, 1.0, 2.0])
@pytest.mark.parametrize("sigma", [0.1, 0.25, 0.6, 1.0])
@pytest.mark.parametrize("option_type", ["call", "put"])
def test_round_trip_recovers_sigma(K, T, sigma, option_type):
    market = dict(S=100, K=K, r=0.03, T=T)
    price = bs_price(**market, sigma=sigma, option_type=option_type)
    assert implied_vol(price, **market, option_type=option_type) == pytest.approx(sigma, abs=1e-7)


def test_newton_converges_in_a_handful_of_iterations_near_the_money():
    price = bs_price(**MKT, sigma=0.25)
    res = implied_vol_detailed(price, **MKT)
    assert isinstance(res, IVResult)
    assert res.method == "newton"
    assert res.iterations <= 6


def test_bisection_fallback_is_used_when_newton_cannot_start():
    # deep out of the money, short expiry: vega at the initial guess is ~0, so Newton gives up
    market = dict(S=100, K=200, r=0.03, T=0.1)
    price = bs_price(**market, sigma=0.8)
    res = implied_vol_detailed(price, **market)
    assert res.method == "bisection"
    assert res.vol == pytest.approx(0.8, abs=1e-7)


@pytest.mark.parametrize("option_type", ["call", "put"])
def test_forced_bisection_gives_the_same_answer_as_newton(option_type):
    price = bs_price(**MKT, sigma=0.37, option_type=option_type)
    newton = implied_vol_detailed(price, **MKT, option_type=option_type)
    bisect = implied_vol_detailed(price, **MKT, option_type=option_type, max_newton_iter=0)
    assert bisect.method == "bisection"
    assert bisect.vol == pytest.approx(newton.vol, abs=1e-8)
    assert bisect.iterations > newton.iterations          # bisection is much slower


def _is_identifiable(K, T, sigma):
    """False when the price is numerically equal to its no-arbitrage bound (no volatility information)."""
    price = bs_price(100, K, 0.03, sigma, T)
    floor = max(100 - K * math.exp(-0.03 * T), 0.0)
    return floor + 1e-9 < price < 100 - 1e-9


REPRICE_CASES = [(K, T, s) for K in (60, 100, 150, 200) for T in (0.02, 0.5, 3.0)
                 for s in (0.05, 0.3, 1.5) if _is_identifiable(K, T, s)]


@pytest.mark.parametrize("K, T, sigma", REPRICE_CASES)
def test_solution_always_reproduces_the_market_price(K, T, sigma):
    """Even where sigma is hard to pin down (tiny vega), the solver's sigma reprices the option."""
    market = dict(S=100, K=K, r=0.03, T=T)
    price = bs_price(**market, sigma=sigma)
    iv = implied_vol(price, **market)
    assert bs_price(**market, sigma=iv) == pytest.approx(price, abs=1e-9)


def test_price_numerically_at_the_bound_has_no_implied_vol():
    # a far out-of-the-money, short-dated call is worth 0.0 in floating point
    price = bs_price(100, 200, 0.03, 0.05, 0.02)
    assert price == 0.0
    with pytest.raises(ValueError, match="minimum"):
        implied_vol(price, S=100, K=200, r=0.03, T=0.02)


def test_low_vega_means_volatility_is_poorly_determined():
    # deep in the money, short expiry: price barely reacts to sigma
    assert greeks(100, 60, 0.03, 0.6, 0.02).vega < 1e-6


def test_higher_price_means_higher_implied_vol():
    vols = [implied_vol(p, **MKT) for p in (6.0, 8.0, 10.0, 12.0, 15.0)]
    assert vols == sorted(vols)


def test_call_and_put_from_the_same_sigma_give_the_same_implied_vol():
    c = bs_price(**MKT, sigma=0.31, option_type="call")
    p = bs_price(**MKT, sigma=0.31, option_type="put")
    assert implied_vol(c, **MKT, option_type="call") == pytest.approx(
        implied_vol(p, **MKT, option_type="put"), abs=1e-8)


def test_price_below_intrinsic_value_raises():
    # S - K e^(-rT) = 100 - 90 e^(-0.05) = 14.4 is the floor for this call
    with pytest.raises(ValueError, match="minimum"):
        implied_vol(10.0, S=100, K=90, r=0.05, T=1.0, option_type="call")


def test_price_at_or_above_upper_bound_raises():
    with pytest.raises(ValueError, match="maximum"):
        implied_vol(100.0, **MKT, option_type="call")           # a call can never be worth S
    with pytest.raises(ValueError, match="maximum"):
        implied_vol(120.0, **MKT, option_type="put")            # a put can never exceed K e^(-rT)


def test_volatility_outside_the_search_range_raises():
    price = bs_price(**MKT, sigma=3.0)
    with pytest.raises(ValueError, match="above the search range"):
        implied_vol(price, **MKT, upper=2.0)
    assert implied_vol(price, **MKT, upper=5.0) == pytest.approx(3.0, abs=1e-7)


@pytest.mark.parametrize("bad", [
    dict(S=-1), dict(K=0), dict(T=0.0), dict(T=-1.0),
    dict(option_type="straddle"), dict(price=float("nan")),
])
def test_invalid_inputs_raise(bad):
    args = dict(price=10.0, S=100, K=100, r=0.05, T=1.0, option_type="call")
    with pytest.raises(ValueError):
        implied_vol(**{**args, **bad})


def test_invalid_search_range_raises():
    with pytest.raises(ValueError):
        implied_vol(10.0, **MKT, lower=1.0, upper=0.5)