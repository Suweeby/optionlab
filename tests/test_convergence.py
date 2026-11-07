import pytest

from optionlab.convergence import binomial_convergence
from optionlab.pricing.black_scholes import call_price

P = dict(S=100, K=100, r=0.05, sigma=0.2, T=1.0)


def test_table_structure_and_values():
    df = binomial_convergence(**P, n_values=(10, 100, 1000))
    assert list(df.columns) == ["n", "binomial", "black_scholes", "error", "abs_error"]
    assert list(df["n"]) == [10, 100, 1000]
    assert df["black_scholes"].iloc[0] == pytest.approx(call_price(**P))
    assert (df["error"] == df["binomial"] - df["black_scholes"]).all()
    assert (df["abs_error"] == df["error"].abs()).all()


def test_put_option_type_is_used():
    df = binomial_convergence(**P, option_type="put", n_values=(500,))
    assert df["black_scholes"].iloc[0] == pytest.approx(5.5735, abs=1e-4)


def test_error_shrinks_across_default_grid():
    df = binomial_convergence(**P)
    assert df["abs_error"].iloc[-1] < df["abs_error"].iloc[0]
    assert df["abs_error"].iloc[-1] < 0.01