"""Convergence studies: how close do the approximations get to Black-Scholes?"""
from __future__ import annotations

import pandas as pd

from optionlab.pricing.binomial import binomial_price
from optionlab.pricing.black_scholes import bs_price

DEFAULT_N_VALUES = (5, 10, 20, 50, 100, 200, 500, 1000, 2000)


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