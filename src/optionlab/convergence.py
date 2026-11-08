"""Convergence studies: how close do the approximations get to Black-Scholes?"""
from __future__ import annotations

import numpy as np
import pandas as pd

from optionlab.pricing.binomial import binomial_price
from optionlab.pricing.black_scholes import bs_price
from optionlab.pricing.monte_carlo import mc_price

DEFAULT_N_VALUES = (5, 10, 20, 50, 100, 200, 500, 1000, 2000)
DEFAULT_PATH_COUNTS = (100, 1_000, 10_000, 100_000, 1_000_000)


def binomial_convergence(
    S: float,
    K: float,
    r: float,
    sigma: float,
    T: float,
    option_type: str = "call",
    n_values=DEFAULT_N_VALUES,
) -> pd.DataFrame:
    """Binomial price for each n, compared with the Black-Scholes price.

    Columns: n, binomial, black_scholes, error (binomial - BS), abs_error.
    """
    bs = bs_price(S, K, r, sigma, T, option_type)
    rows = []
    for n in n_values:
        price = binomial_price(S, K, r, sigma, T, n, option_type)
        rows.append(
            {
                "n": n,
                "binomial": price,
                "black_scholes": bs,
                "error": price - bs,
                "abs_error": abs(price - bs),
            }
        )
    return pd.DataFrame(rows)


def monte_carlo_convergence(
    S: float,
    K: float,
    r: float,
    sigma: float,
    T: float,
    option_type: str = "call",
    path_counts=DEFAULT_PATH_COUNTS,
    replications: int = 20,
    antithetic: bool = False,
    seed: int = 42,
) -> pd.DataFrame:
    """Repeat the Monte Carlo pricer `replications` times (independent random
    numbers) for each path count and compare with Black-Scholes.

    Columns: n_paths, black_scholes, mean_price, mean_std_error (average
    reported SE), rmse (root-mean-square error of the price over the
    replications). Both mean_std_error and rmse should fall like 1/sqrt(n_paths).
    """
    if replications < 2:
        raise ValueError("replications must be at least 2.")
    bs = bs_price(S, K, r, sigma, T, option_type)
    rng = np.random.default_rng(seed)       # one generator -> every run uses fresh numbers
    rows = []
    for n in path_counts:
        results = [
            mc_price(S, K, r, sigma, T, n_paths=n, option_type=option_type,
                     seed=rng, antithetic=antithetic)
            for _ in range(replications)
        ]
        prices = np.array([res.price for res in results])
        rows.append(
            {
                "n_paths": n,
                "black_scholes": bs,
                "mean_price": float(prices.mean()),
                "mean_std_error": float(np.mean([res.std_error for res in results])),
                "rmse": float(np.sqrt(np.mean((prices - bs) ** 2))),
            }
        )
    return pd.DataFrame(rows)

def monte_carlo_single_runs(
    S: float,
    K: float,
    r: float,
    sigma: float,
    T: float,
    option_type: str = "call",
    path_counts=DEFAULT_PATH_COUNTS,
    antithetic: bool = False,
    seed: int = 42,
    confidence: float = 0.95,
) -> pd.DataFrame:
    """One Monte Carlo run per path count (fresh random numbers each), with its confidence interval.

    Columns: n_paths, black_scholes, price, std_error, ci_low, ci_high, error, covers
    (covers = does the confidence interval contain the Black-Scholes price?).
    """
    bs = bs_price(S, K, r, sigma, T, option_type)
    rng = np.random.default_rng(seed)
    rows = []
    for n in path_counts:
        res = mc_price(S, K, r, sigma, T, n_paths=n, option_type=option_type, seed=rng,
                       antithetic=antithetic, confidence=confidence)
        rows.append(
            {
                "n_paths": n,
                "black_scholes": bs,
                "price": res.price,
                "std_error": res.std_error,
                "ci_low": res.ci_low,
                "ci_high": res.ci_high,
                "error": res.price - bs,
                "covers": bool(res.ci_low <= bs <= res.ci_high),
            }
        )
    return pd.DataFrame(rows)