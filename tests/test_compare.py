import pytest

from optionlab.pricing.compare import price_all

ANCHOR = dict(S=100, K=100, r=0.05, sigma=0.2, T=1.0)


def test_three_methods_agree_for_the_anchor_call():
    res = price_all(**ANCHOR, n_steps=500, n_paths=200_000, seed=1)
    assert res.black_scholes == pytest.approx(10.4506, abs=1e-4)
    assert abs(res.binomial_minus_bs) < 0.01
    assert abs(res.mc_minus_bs) < 4 * res.monte_carlo.std_error
    assert res.bs_inside_mc_interval == (res.monte_carlo.ci_low <= res.black_scholes <= res.monte_carlo.ci_high)


def test_put_is_priced_as_a_put():
    res = price_all(**ANCHOR, option_type="put", n_steps=500, n_paths=200_000, seed=1)
    assert res.black_scholes == pytest.approx(5.5735, abs=1e-4)
    assert abs(res.binomial_minus_bs) < 0.01


def test_seed_makes_it_reproducible():
    a = price_all(**ANCHOR, n_paths=10_000, seed=4)
    b = price_all(**ANCHOR, n_paths=10_000, seed=4)
    assert a == b


@pytest.mark.parametrize("bad", [
    dict(option_type="straddle"), dict(sigma=0.0), dict(T=0.0), dict(S=-1), dict(K=0),
])
def test_invalid_inputs_raise_value_error(bad):
    with pytest.raises(ValueError):
        price_all(**{**ANCHOR, **bad}, n_paths=1000)


def test_odd_number_of_paths_with_antithetic_raises():
    with pytest.raises(ValueError):
        price_all(**ANCHOR, n_paths=1001, antithetic=True)
