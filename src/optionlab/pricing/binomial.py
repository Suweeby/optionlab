"""Cox-Ross-Rubinstein (CRR) binomial tree pricer for European options."""
from __future__ import annotations

import math

import numpy as np


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
    """
    if S <= 0 or K <= 0:
        raise ValueError("S and K must be positive.")
    if sigma <= 0:
        raise ValueError("sigma must be positive.")
    if T <= 0:
        raise ValueError("T must be positive.")
    if n < 1:
        raise ValueError("n must be at least 1.")
    if option_type not in ("call", "put"):
        raise ValueError("option_type must be 'call' or 'put'.")

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
    values = np.maximum(ST - K, 0.0) if option_type == "call" else np.maximum(K - ST, 0.0)

    # Backward induction: each node = discounted expected value of its two children.
    # values[1:] are the up-children, values[:-1] the down-children; the array shrinks by 1 per step.
    for _ in range(n):
        values = disc * (p * values[1:] + (1.0 - p) * values[:-1])

    return float(values[0])