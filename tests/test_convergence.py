import pytest
from optionlab.convergence import binomial_convergence, monte_carlo_convergence, monte_carlo_single_runs
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


# ---- Monte Carlo convergence ------------------------------------------------------

def test_mc_table_structure():
    df = monte_carlo_convergence(**P, path_counts=(100, 1000), replications=5)
    assert list(df.columns) == ["n_paths", "black_scholes", "mean_price", "mean_std_error", "rmse"]
    assert list(df["n_paths"]) == [100, 1000]
    assert df["black_scholes"].iloc[0] == pytest.approx(call_price(**P))


def test_mc_standard_error_and_rmse_fall_like_one_over_sqrt_n():
    df = monte_carlo_convergence(**P, path_counts=(1_000, 100_000), replications=100, seed=3)
    small, large = df.iloc[0], df.iloc[1]
    assert small["mean_std_error"] / large["mean_std_error"] == pytest.approx(10.0, rel=0.05)
    assert small["rmse"] / large["rmse"] == pytest.approx(10.0, rel=0.35)
    # reported SE is honest: it matches the actual RMSE
    assert large["mean_std_error"] == pytest.approx(large["rmse"], rel=0.3)


def test_mc_convergence_is_reproducible_and_needs_replications():
    a = monte_carlo_convergence(**P, path_counts=(500,), replications=4, seed=9)
    b = monte_carlo_convergence(**P, path_counts=(500,), replications=4, seed=9)
    assert a.equals(b)
    with pytest.raises(ValueError):
        monte_carlo_convergence(**P, replications=1)

# ---- single Monte Carlo runs with confidence intervals ----------------------------

def test_single_runs_table_structure():
    df = monte_carlo_single_runs(**P, path_counts=(1_000, 10_000), seed=1)
    assert list(df.columns) == ["n_paths", "black_scholes", "price", "std_error",
                                "ci_low", "ci_high", "error", "covers"]
    assert list(df["n_paths"]) == [1_000, 10_000]
    assert (df["error"] == df["price"] - df["black_scholes"]).all()
    assert (df["covers"] == ((df["ci_low"] <= df["black_scholes"]) & (df["black_scholes"] <= df["ci_high"]))).all()
    assert (df["ci_low"] < df["price"]).all() and (df["price"] < df["ci_high"]).all()


def test_single_runs_standard_error_shrinks_and_is_reproducible():
    a = monte_carlo_single_runs(**P, path_counts=(1_000, 100_000), seed=2)
    b = monte_carlo_single_runs(**P, path_counts=(1_000, 100_000), seed=2)
    assert a.equals(b)
    assert a["std_error"].iloc[0] / a["std_error"].iloc[1] == pytest.approx(10.0, rel=0.15)


def test_single_runs_antithetic_has_smaller_standard_error():
    plain = monte_carlo_single_runs(**P, path_counts=(100_000,), seed=3)
    anti = monte_carlo_single_runs(**P, path_counts=(100_000,), seed=3, antithetic=True)
    assert anti["std_error"].iloc[0] < 0.9 * plain["std_error"].iloc[0]