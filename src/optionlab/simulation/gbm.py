"""The shared geometric Brownian motion (GBM) path generator.

One implementation serves both layers of OptionLab:
  * pricing mode    -> drift = r   (risk-neutral), used by the Monte Carlo pricer
  * simulation mode -> drift = mu  (real-world), used by the Simulator page

Model:  dS = drift * S dt + sigma * S dW
Exact step:  S(t+dt) = S(t) * exp((drift - sigma^2/2) dt + sigma sqrt(dt) Z),  Z ~ N(0, 1)
"""
from __future__ import annotations

import math

import numpy as np


def _validate(S0: float, drift: float, sigma: float, T: float,
              n_steps: int, n_paths: int) -> None:
    if not all(math.isfinite(x) for x in (S0, drift, sigma, T)):
        raise ValueError("S0, drift, sigma and T must be finite numbers.")
    if S0 <= 0:
        raise ValueError("S0 (initial price) must be positive.")
    if sigma < 0:
        raise ValueError("sigma (volatility) cannot be negative.")
    if T <= 0:
        raise ValueError("T (horizon in years) must be positive.")
    if int(n_steps) != n_steps or n_steps < 1:
        raise ValueError("n_steps must be a positive integer.")
    if int(n_paths) != n_paths or n_paths < 1:
        raise ValueError("n_paths must be a positive integer.")


def time_grid(T: float, n_steps: int) -> np.ndarray:
    """Times 0, dt, 2dt, ..., T (n_steps + 1 points), matching the columns of the paths."""
    return np.linspace(0.0, T, int(n_steps) + 1)


def simulate_gbm(
    S0: float,
    drift: float,
    sigma: float,
    T: float,
    n_steps: int,
    n_paths: int,
    seed: int | np.random.Generator | None = None,
    antithetic: bool = False,
) -> np.ndarray:
    """Simulate GBM price paths. Returns an array of shape (n_paths, n_steps + 1).

    Column 0 is S0; column k is the price at time k*T/n_steps.
    seed: an int, a numpy Generator, or None (fresh randomness).
    antithetic: if True, the second half of the paths uses -Z where the first
        half uses Z (n_paths must be even). Row i and row i + n_paths/2 are a pair.
    """
    _validate(S0, drift, sigma, T, n_steps, n_paths)
    rng = np.random.default_rng(seed)
    dt = T / n_steps

    if antithetic:
        if n_paths % 2 != 0:
            raise ValueError("n_paths must be even when antithetic=True.")
        half = rng.standard_normal((n_paths // 2, n_steps))
        Z = np.concatenate([half, -half], axis=0)
    else:
        Z = rng.standard_normal((n_paths, n_steps))

    # Log-price increments are independent normals, so a cumulative sum gives
    # the whole path at once (no Python loop over paths or steps).
    log_increments = (drift - 0.5 * sigma**2) * dt + sigma * math.sqrt(dt) * Z
    log_paths = np.cumsum(log_increments, axis=1)

    paths = np.empty((n_paths, n_steps + 1))
    paths[:, 0] = S0
    paths[:, 1:] = S0 * np.exp(log_paths)
    return paths


def simulate_terminal_prices(
    S0: float,
    drift: float,
    sigma: float,
    T: float,
    n_paths: int,
    seed: int | np.random.Generator | None = None,
    antithetic: bool = False,
) -> np.ndarray:
    """Only S_T for each path (1-D array). The exact scheme needs a single step."""
    paths = simulate_gbm(S0, drift, sigma, T, n_steps=1, n_paths=n_paths,
                         seed=seed, antithetic=antithetic)
    return paths[:, -1]
