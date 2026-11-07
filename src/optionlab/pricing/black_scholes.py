"""Black-Scholes closed-form prices for European options (no dividends).

Notation (Ross's symbols in brackets):
    S      spot price                       (x)
    K      strike                           (K)
    r      continuously compounded rate     (r)
    sigma  annual volatility                (sigma)
    T      years to expiry                  (t)
    d1     = omega
    d2     = omega - sigma*sqrt(T)
    N      standard normal CDF              (Phi)
"""
from __future__ import annotations

import math

from scipy.stats import norm


def _validate(S: float, K: float, r: float, sigma: float, T: float) -> None:
    if not all(math.isfinite(x) for x in (S, K, r, sigma, T)):
        raise ValueError("All inputs must be finite numbers.")
    if S <= 0:
        raise ValueError("S (spot price) must be positive.")
    if K <= 0:
        raise ValueError("K (strike) must be positive.")
    if sigma < 0:
        raise ValueError("sigma (volatility) cannot be negative.")
    if T < 0:
        raise ValueError("T (time to expiry) cannot be negative.")


def d1(S: float, K: float, r: float, sigma: float, T: float) -> float:
    """d1 = [ln(S/K) + (r + sigma^2/2) T] / (sigma sqrt(T)). Needs sigma > 0 and T > 0."""
    _validate(S, K, r, sigma, T)
    if sigma == 0 or T == 0:
        raise ValueError("d1 is undefined when sigma = 0 or T = 0.")
    return (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))


def d2(S: float, K: float, r: float, sigma: float, T: float) -> float:
    """d2 = d1 - sigma sqrt(T)."""
    return d1(S, K, r, sigma, T) - sigma * math.sqrt(T)


def call_price(S: float, K: float, r: float, sigma: float, T: float) -> float:
    """European call: S N(d1) - K e^(-rT) N(d2)."""
    _validate(S, K, r, sigma, T)
    pv_K = K * math.exp(-r * T)               # present value of the strike
    if sigma == 0 or T == 0:                  # limits: no randomness left
        return max(S - pv_K, 0.0)
    D1 = d1(S, K, r, sigma, T)
    D2 = D1 - sigma * math.sqrt(T)
    return float(S * norm.cdf(D1) - pv_K * norm.cdf(D2))


def put_price(S: float, K: float, r: float, sigma: float, T: float) -> float:
    """European put: K e^(-rT) N(-d2) - S N(-d1)."""
    _validate(S, K, r, sigma, T)
    pv_K = K * math.exp(-r * T)
    if sigma == 0 or T == 0:
        return max(pv_K - S, 0.0)
    D1 = d1(S, K, r, sigma, T)
    D2 = D1 - sigma * math.sqrt(T)
    return float(pv_K * norm.cdf(-D2) - S * norm.cdf(-D1))


def bs_price(S: float, K: float, r: float, sigma: float, T: float,
             option_type: str = "call") -> float:
    """Price a European call or put by name."""
    if option_type == "call":
        return call_price(S, K, r, sigma, T)
    if option_type == "put":
        return put_price(S, K, r, sigma, T)
    raise ValueError("option_type must be 'call' or 'put'.")