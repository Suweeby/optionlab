import math

import pytest

from optionlab.pricing.binomial import binomial_price

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