import numpy as np
import pytest

from optionlab.payoffs import call_payoff, payoff, put_payoff


def test_call_payoff_values():
    out = call_payoff(np.array([80.0, 100.0, 120.0]), 100)
    assert np.allclose(out, [0.0, 0.0, 20.0])


def test_put_payoff_values():
    out = put_payoff(np.array([80.0, 100.0, 120.0]), 100)
    assert np.allclose(out, [20.0, 0.0, 0.0])


def test_payoffs_are_never_negative():
    ST = np.linspace(1, 300, 500)
    assert (call_payoff(ST, 100) >= 0).all()
    assert (put_payoff(ST, 100) >= 0).all()


def test_call_minus_put_is_forward_payoff():
    # (x)^+ - (-x)^+ = x, so call - put = S_T - K for every S_T.
    ST = np.linspace(1, 300, 500)
    assert np.allclose(call_payoff(ST, 100) - put_payoff(ST, 100), ST - 100)


def test_scalar_input_works():
    assert float(call_payoff(120, 100)) == 20.0


def test_dispatch_and_invalid_type():
    assert float(payoff(90, 100, "put")) == 10.0
    with pytest.raises(ValueError):
        payoff(90, 100, "straddle")


def test_invalid_strike_raises():
    with pytest.raises(ValueError):
        call_payoff(100, 0)