"""Payoff functions for European options (vectorized)."""
from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike


def call_payoff(ST: ArrayLike, K: float) -> np.ndarray:
    """Call payoff (S_T - K)^+ = max(S_T - K, 0). ST may be a number or an array."""
    if K <= 0:
        raise ValueError("K (strike) must be positive.")
    return np.maximum(np.asarray(ST, dtype=float) - K, 0.0)


def put_payoff(ST: ArrayLike, K: float) -> np.ndarray:
    """Put payoff (K - S_T)^+ = max(K - S_T, 0). ST may be a number or an array."""
    if K <= 0:
        raise ValueError("K (strike) must be positive.")
    return np.maximum(K - np.asarray(ST, dtype=float), 0.0)


def payoff(ST: ArrayLike, K: float, option_type: str) -> np.ndarray:
    """Dispatch to the call or put payoff by name."""
    if option_type == "call":
        return call_payoff(ST, K)
    if option_type == "put":
        return put_payoff(ST, K)
    raise ValueError("option_type must be 'call' or 'put'.")

def moneyness(S: float, K: float, option_type: str, tol: float = 0.01) -> str:
    """Describe an option as 'in the money', 'at the money' or 'out of the money'.

    'At the money' means S is within tol (default 1%) of K.
    A call is in the money when S > K; a put when S < K.
    """
    if S <= 0 or K <= 0:
        raise ValueError("S and K must be positive.")
    if option_type not in ("call", "put"):
        raise ValueError("option_type must be 'call' or 'put'.")
    if abs(S / K - 1.0) <= tol:
        return "at the money"
    in_the_money = S > K if option_type == "call" else S < K
    return "in the money" if in_the_money else "out of the money"
