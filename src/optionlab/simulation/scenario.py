"""A simulation scenario for the Simulator page: a few full paths to draw,
plus many terminal prices for the statistics. Uses the shared GBM generator only."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from optionlab.simulation.gbm import simulate_gbm, simulate_terminal_prices, time_grid


@dataclass(frozen=True)
class Scenario:
    S0: float
    drift: float
    sigma: float
    T: float
    times: np.ndarray              # (n_steps + 1,)
    sample_paths: np.ndarray       # (n_draw, n_steps + 1): the paths that get drawn
    terminal_prices: np.ndarray    # (n_paths,): S_T of ALL simulated paths


def simulate_scenario(
    S0: float,
    drift: float,
    sigma: float,
    T: float,
    n_steps: int = 252,
    n_paths: int = 20_000,
    n_draw: int = 100,
    seed: int | np.random.Generator | None = None,
) -> Scenario:
    """Simulate n_paths terminal prices, keeping full paths for only n_draw of them.

    Storing every path would need n_paths * n_steps numbers. The terminal price does not
    depend on the number of steps (exact discretization), so the other paths are
    simulated in one step each. All paths are independent draws from the same law.
    """
    if int(n_draw) != n_draw or n_draw < 1:
        raise ValueError("n_draw must be a positive integer.")
    n_draw = int(min(n_draw, n_paths))
    rng = np.random.default_rng(seed)

    sample = simulate_gbm(S0, drift, sigma, T, n_steps, n_draw, seed=rng)
    if n_paths > n_draw:
        rest = simulate_terminal_prices(S0, drift, sigma, T, int(n_paths - n_draw), seed=rng)
    else:
        rest = np.empty(0)
    terminal = np.concatenate([sample[:, -1], rest])
    return Scenario(S0, drift, sigma, T, time_grid(T, n_steps), sample, terminal)