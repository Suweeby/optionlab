"""Cox-Ross-Rubinstein (CRR) binomial tree pricer for European options."""
from __future__ import annotations

import math

import numpy as np

from optionlab.payoffs import payoff


def binomial_price(
    S: float,
    K: float,
    r: float,
    sigma: float,
    T: float,
    n: int = 100,
    option_type: str = "call",
) -> float:
    """Price a European call or put with a CRR binomial tree.

    S: spot price, K: strike, r: continuously compounded risk-free rate,
    sigma: annual volatility, T: years to expiry, n: number of time steps.

    Unlike Black-Scholes, the tree needs sigma > 0 and T > 0: with sigma = 0
    we would get u = d = 1 and the probability formula divides by zero.
    """
    if S <= 0 or K <= 0:
        raise ValueError("S and K must be positive.")
    if sigma <= 0:
        raise ValueError("sigma must be positive.")
    if T <= 0:
        raise ValueError("T must be positive.")
    if n < 1:
        raise ValueError("n must be at least 1.")

    dt = T / n
    u = math.exp(sigma * math.sqrt(dt))      # up factor
    d = 1.0 / u                              # down factor (tree recombines)
    p = (math.exp(r * dt) - d) / (u - d)     # risk-neutral up probability
    if not 0.0 < p < 1.0:
        raise ValueError("Risk-neutral probability outside (0, 1): increase n or sigma.")
    disc = math.exp(-r * dt)                 # one-step discount factor

    # Terminal prices: after n steps with j up-moves, S * u^j * d^(n-j).
    j = np.arange(n + 1)
    ST = S * u**j * d ** (n - j)
    values = payoff(ST, K, option_type)      # shared payoff function (raises on bad type)

    # Backward induction: each node = discounted expected value of its two children.
    for _ in range(n):
        values = disc * (p * values[1:] + (1.0 - p) * values[:-1])

    return float(values[0])
