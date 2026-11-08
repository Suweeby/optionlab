"""Monte Carlo pricer for European options, built on the shared GBM generator.

Price = e^(-rT) * E*[payoff(S_T)], estimated by the average of simulated payoffs.
The simulation runs in pricing mode: drift = r (risk-neutral), via
simulation.gbm.simulate_terminal_prices. This pricer is the simulator plus a payoff.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.stats import norm

from optionlab.payoffs import payoff
from optionlab.simulation.gbm import simulate_terminal_prices


@dataclass(frozen=True)
class MCResult:
    """A Monte Carlo price estimate with its uncertainty."""

    price: float        # Monte Carlo estimate of the option price
    std_error: float    # standard error of that estimate
    ci_low: float       # lower end of the confidence interval
    ci_high: float      # upper end of the confidence interval
    n_paths: int
    antithetic: bool
    confidence: float


def mc_price(
    S: float,
    K: float,
    r: float,
    sigma: float,
    T: float,
    n_paths: int = 100_000,
    option_type: str = "call",
    seed: int | np.random.Generator | None = None,
    antithetic: bool = False,
    confidence: float = 0.95,
) -> MCResult:
    """Price a European call/put by Monte Carlo and report its uncertainty.

    With antithetic=True, paths come in (Z, -Z) pairs; n_paths must be even and
    at least 4. The standard error is then computed from the pair averages,
    because the two paths inside a pair are not independent.
    """
    if option_type not in ("call", "put"):
        raise ValueError("option_type must be 'call' or 'put'.")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be strictly between 0 and 1.")
    if n_paths < 2 or (antithetic and n_paths < 4):
        raise ValueError("Need at least 2 paths (4 with antithetic) to estimate a standard error.")

    # Same generator as the simulator, with the risk-neutral drift r.
    ST = simulate_terminal_prices(S, r, sigma, T, n_paths, seed=seed, antithetic=antithetic)
    discounted = math.exp(-r * T) * payoff(ST, K, option_type)

    if antithetic:
        half = n_paths // 2
        samples = 0.5 * (discounted[:half] + discounted[half:])   # one independent value per pair
    else:
        samples = discounted

    price = float(samples.mean())
    std_error = float(samples.std(ddof=1) / math.sqrt(samples.size))
    z = float(norm.ppf(0.5 + confidence / 2))
    return MCResult(price, std_error, price - z * std_error, price + z * std_error,
                    int(n_paths), antithetic, confidence)
