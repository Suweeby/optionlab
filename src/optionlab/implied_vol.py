"""Implied volatility: the sigma that makes the Black-Scholes price equal a market price.

Black-Scholes maps sigma -> price and has no closed-form inverse, so we solve
    f(sigma) = BS(sigma) - market_price = 0
numerically. Vega > 0 makes the price strictly increasing in sigma, so there is at
most one solution, and one exists exactly when the price is inside the no-arbitrage bounds.

Method: Newton-Raphson (derivative = vega), with a bisection fallback when Newton
fails (vega ~ 0, a step outside the allowed range, or no convergence).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from optionlab.greeks import greeks
from optionlab.pricing.black_scholes import _validate, bs_price


@dataclass(frozen=True)
class IVResult:
    """An implied volatility with how it was found."""

    vol: float          # the implied volatility
    iterations: int     # iterations used by the method that produced the answer
    method: str         # "newton" or "bisection"


def _price_bounds(S: float, K: float, r: float, T: float, option_type: str) -> tuple[float, float]:
    """No-arbitrage bounds (lower, upper) for a European option price."""
    pv_K = K * math.exp(-r * T)
    if option_type == "call":
        return max(S - pv_K, 0.0), S
    return max(pv_K - S, 0.0), pv_K


def _newton(price, S, K, r, T, option_type, x0, lo, hi, tol, max_iter):
    """Newton-Raphson on f(sigma) = BS(sigma) - price. Returns (vol, iterations) or None."""
    x = x0
    for i in range(1, max_iter + 1):
        diff = bs_price(S, K, r, x, T, option_type) - price
        if abs(diff) < tol:
            return x, i - 1
        vega = greeks(S, K, r, x, T, option_type).vega
        if vega < 1e-12:                    # tangent is flat: the step would explode
            return None
        x = x - diff / vega                 # Newton step: where the tangent line hits zero
        if not lo < x < hi:                 # left the allowed range
            return None
    return None                             # did not converge in time


def _bisection(price, S, K, r, T, option_type, lo, hi, tol, max_iter):
    """Bisection on [lo, hi]: f(lo) <= 0 <= f(hi), so the root is bracketed. Halve until close."""
    for i in range(1, max_iter + 1):
        mid = 0.5 * (lo + hi)
        diff = bs_price(S, K, r, mid, T, option_type) - price
        if abs(diff) < tol or (hi - lo) < 1e-14:
            return mid, i
        if diff > 0:
            hi = mid                        # model price too high -> volatility too high
        else:
            lo = mid
    return 0.5 * (lo + hi), max_iter


def implied_vol_detailed(
    price: float, S: float, K: float, r: float, T: float,
    option_type: str = "call",
    initial: float = 0.2,
    tol: float = 1e-10,
    max_newton_iter: int = 50,
    max_bisection_iter: int = 200,
    lower: float = 1e-6,
    upper: float = 5.0,
) -> IVResult:
    """Solve for the implied volatility and report how it was found.

    price: observed option price. lower/upper: search range for sigma (0.0001% to 500%).
    Raises ValueError if the price is outside the no-arbitrage bounds or no
    volatility in [lower, upper] reproduces it.
    """
    _validate(S, K, r, 0.0, T)                  # S, K, r finite and positive where needed, T >= 0
    if option_type not in ("call", "put"):
        raise ValueError("option_type must be 'call' or 'put'.")
    if T <= 0:
        raise ValueError("T must be positive to have an implied volatility.")
    if not math.isfinite(price):
        raise ValueError("price must be a finite number.")
    if not 0.0 < lower < upper:
        raise ValueError("Need 0 < lower < upper.")

    low_bound, high_bound = _price_bounds(S, K, r, T, option_type)
    if price <= low_bound:
        raise ValueError(
            f"Price {price:.6g} is at or below the minimum possible value {low_bound:.6g}; "
            "no positive volatility reproduces it.")
    if price >= high_bound:
        raise ValueError(
            f"Price {price:.6g} is at or above the maximum possible value {high_bound:.6g}; "
            "no finite volatility reproduces it.")
    if bs_price(S, K, r, lower, T, option_type) > price:
        raise ValueError(f"Implied volatility is below the search range ({lower:g}).")
    if bs_price(S, K, r, upper, T, option_type) < price:
        raise ValueError(f"Implied volatility is above the search range ({upper:g}).")

    found = _newton(price, S, K, r, T, option_type,
                    min(max(initial, lower * 2), upper / 2), lower, upper, tol, max_newton_iter)
    if found is not None:
        return IVResult(found[0], found[1], "newton")

    vol, iters = _bisection(price, S, K, r, T, option_type, lower, upper, tol, max_bisection_iter)
    return IVResult(vol, iters, "bisection")


def implied_vol(price: float, S: float, K: float, r: float, T: float,
                option_type: str = "call", **kwargs) -> float:
    """Implied volatility as a plain number (see implied_vol_detailed for options)."""
    return implied_vol_detailed(price, S, K, r, T, option_type, **kwargs).vol
