"""Payoff functions for European options (vectorized)."""
from __future__ import annotations

import numpy as np


def call_payoff(ST, K: float):
    """Call payoff (S_T - K)^+ = max(S_T - K, 0). ST may be a number or an array."""
    if K <= 0:
        raise ValueError("K (strike) must be positive.")
    return np.maximum(np.asarray(ST, dtype=float) - K, 0.0)


def put_payoff(ST, K: float):
    """Put payoff (K - S_T)^+ = max(K - S_T, 0). ST may be a number or an array."""
    if K <= 0:
        raise ValueError("K (strike) must be positive.")
    return np.maximum(K - np.asarray(ST, dtype=float), 0.0)


def payoff(ST, K: float, option_type: str):
    """Dispatch to the call or put payoff by name."""
    if option_type == "call":
        return call_payoff(ST, K)
    if option_type == "put":
        return put_payoff(ST, K)
    raise ValueError("option_type must be 'call' or 'put'.")