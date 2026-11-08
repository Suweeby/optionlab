"""Price one option with all three methods (used by the Pricer page)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from optionlab.pricing.binomial import binomial_price
from optionlab.pricing.black_scholes import bs_price
from optionlab.pricing.monte_carlo import MCResult, mc_price


@dataclass(frozen=True)
class PriceComparison:
    """The same option priced by all three methods."""

    black_scholes: float
    binomial: float
    monte_carlo: MCResult

    @property
    def binomial_minus_bs(self) -> float:
        """Binomial price minus Black-Scholes price."""
        return self.binomial - self.black_scholes

    @property
    def mc_minus_bs(self) -> float:
        """Monte Carlo price minus Black-Scholes price."""
        return self.monte_carlo.price - self.black_scholes

    @property
    def bs_inside_mc_interval(self) -> bool:
        """Does the Monte Carlo confidence interval contain the Black-Scholes price?"""
        mc = self.monte_carlo
        return mc.ci_low <= self.black_scholes <= mc.ci_high


def price_all(
    S: float,
    K: float,
    r: float,
    sigma: float,
    T: float,
    option_type: str = "call",
    n_steps: int = 500,
    n_paths: int = 100_000,
    seed: int | np.random.Generator | None = None,
    antithetic: bool = True,
) -> PriceComparison:
    """Black-Scholes, CRR binomial and Monte Carlo prices for one European option.

    All three methods need sigma > 0 and T > 0 (the tree and the simulation are
    undefined otherwise); invalid inputs raise ValueError with a readable message.
    """
    bs = bs_price(S, K, r, sigma, T, option_type)
    tree = binomial_price(S, K, r, sigma, T, n_steps, option_type)
    mc = mc_price(S, K, r, sigma, T, n_paths=n_paths, option_type=option_type,
                  seed=seed, antithetic=antithetic)
    return PriceComparison(black_scholes=bs, binomial=tree, monte_carlo=mc)
