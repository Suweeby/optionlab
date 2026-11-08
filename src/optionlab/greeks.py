"""Option Greeks: analytic Black-Scholes formulas, a finite-difference checker,
and a Monte Carlo (pathwise) delta.

Greeks are partial derivatives of the option value V(S, sigma, T, r):
    delta = dV/dS        gamma = d2V/dS2      vega = dV/dsigma
    theta = dV/dt = -dV/dT (value lost as time passes)      rho = dV/dr

Raw units: vega per 1.00 (=100 vol points), theta per year, rho per 1.00 (=100 rate points).
Use to_trader_units() for per-vol-point, per-day and per-1%-rate numbers.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, replace

import numpy as np
from scipy.stats import norm

from optionlab.pricing.black_scholes import _validate, bs_price, d1, d2
from optionlab.simulation.gbm import simulate_terminal_prices


@dataclass(frozen=True)
class Greeks:
    """The five Greeks of one option."""

    delta: float
    gamma: float
    vega: float
    theta: float
    rho: float


def _check(S, K, r, sigma, T, option_type) -> None:
    _validate(S, K, r, sigma, T)
    if option_type not in ("call", "put"):
        raise ValueError("option_type must be 'call' or 'put'.")
    if sigma == 0 or T == 0:
        raise ValueError("Greeks are undefined when sigma = 0 or T = 0.")


def greeks(S: float, K: float, r: float, sigma: float, T: float,
           option_type: str = "call") -> Greeks:
    """Analytic Black-Scholes Greeks in raw (calculus) units."""
    _check(S, K, r, sigma, T, option_type)
    D1, D2 = d1(S, K, r, sigma, T), d2(S, K, r, sigma, T)
    pdf_d1 = norm.pdf(D1)
    sqrt_T = math.sqrt(T)
    pv_K = K * math.exp(-r * T)

    gamma = pdf_d1 / (S * sigma * sqrt_T)          # same for call and put
    vega = S * pdf_d1 * sqrt_T                     # same for call and put
    decay = -S * pdf_d1 * sigma / (2 * sqrt_T)     # time decay from volatility, same for both

    if option_type == "call":
        delta = norm.cdf(D1)
        theta = decay - r * pv_K * norm.cdf(D2)
        rho = T * pv_K * norm.cdf(D2)
    else:
        delta = norm.cdf(D1) - 1.0
        theta = decay + r * pv_K * norm.cdf(-D2)
        rho = -T * pv_K * norm.cdf(-D2)
    return Greeks(float(delta), float(gamma), float(vega), float(theta), float(rho))


def to_trader_units(g: Greeks) -> Greeks:
    """vega per 1 vol point (1%), theta per calendar day, rho per 1% change in rate."""
    return replace(g, vega=g.vega / 100.0, theta=g.theta / 365.0, rho=g.rho / 100.0)


def fd_greeks(S: float, K: float, r: float, sigma: float, T: float,
              option_type: str = "call", rel_step: float = 1e-4) -> Greeks:
    """Greeks by central finite differences of the Black-Scholes price.

    Used to check the analytic formulas: it needs no calculus, only the pricer.
        f'(x)  ~ [f(x+h) - f(x-h)] / (2h)           error ~ h^2
        f''(x) ~ [f(x+h) - 2 f(x) + f(x-h)] / h^2   error ~ h^2
    h must be small for accuracy but not tiny, or floating-point round-off dominates.
    """
    _check(S, K, r, sigma, T, option_type)

    def price(S_=S, sigma_=sigma, T_=T, r_=r):
        return bs_price(S_, K, r_, sigma_, T_, option_type)

    hS, hv, hT, hr = rel_step * S, rel_step * sigma, rel_step * T, rel_step
    v0 = price()
    delta = (price(S_=S + hS) - price(S_=S - hS)) / (2 * hS)
    gamma = (price(S_=S + hS) - 2 * v0 + price(S_=S - hS)) / hS**2
    vega = (price(sigma_=sigma + hv) - price(sigma_=sigma - hv)) / (2 * hv)
    theta = -(price(T_=T + hT) - price(T_=T - hT)) / (2 * hT)     # time passing = T shrinking
    rho = (price(r_=r + hr) - price(r_=r - hr)) / (2 * hr)
    return Greeks(delta, gamma, vega, theta, rho)


def mc_delta(S: float, K: float, r: float, sigma: float, T: float,
             n_paths: int = 100_000, option_type: str = "call",
             seed: int | np.random.Generator | None = None) -> tuple[float, float]:
    """Monte Carlo delta by the pathwise method. Returns (estimate, standard error).

    S_T = S * exp(...) so dS_T/dS = S_T / S. Differentiating the payoff inside the
    expectation: call -> e^(-rT) * 1{S_T > K} * S_T / S ; put -> -e^(-rT) * 1{S_T < K} * S_T / S.
    """
    _check(S, K, r, sigma, T, option_type)
    if n_paths < 2:
        raise ValueError("n_paths must be at least 2.")
    ST = simulate_terminal_prices(S, r, sigma, T, n_paths, seed=seed)    # risk-neutral drift r
    if option_type == "call":
        samples = math.exp(-r * T) * (ST > K) * ST / S
    else:
        samples = -math.exp(-r * T) * (ST < K) * ST / S
    return float(samples.mean()), float(samples.std(ddof=1) / math.sqrt(n_paths))
