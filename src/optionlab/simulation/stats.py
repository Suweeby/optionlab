"""Summary statistics for simulated terminal prices, and GBM theory to check them against."""
from __future__ import annotations

import math

import numpy as np
from scipy.stats import norm


def _as_prices(ST) -> np.ndarray:
    arr = np.asarray(ST, dtype=float)
    if arr.ndim != 1 or arr.size == 0:
        raise ValueError("ST must be a non-empty 1-D array of prices.")
    if not np.isfinite(arr).all():
        raise ValueError("ST contains non-finite values.")
    return arr


def prob_above(ST, K: float) -> float:
    """Fraction of simulated terminal prices strictly above K, i.e. estimate of P(S_T > K)."""
    if K <= 0:
        raise ValueError("K must be positive.")
    return float(np.mean(_as_prices(ST) > K))


def value_at_risk(ST, S0: float, confidence: float = 0.95) -> float:
    """Simple simulation VaR: the loss (S0 - S_T) that is exceeded in only
    (1 - confidence) of the scenarios. Reported as a positive number; 0 if the
    quantile of the loss is a gain."""
    if S0 <= 0:
        raise ValueError("S0 must be positive.")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be strictly between 0 and 1.")
    loss = S0 - _as_prices(ST)
    return max(float(np.quantile(loss, confidence)), 0.0)


def terminal_summary(ST, K: float | None = None,
                     percentiles=(5, 25, 50, 75, 95)) -> dict:
    """Mean, standard deviation, min, max, percentiles and (optionally) P(S_T > K)."""
    arr = _as_prices(ST)
    out = {
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
        "min": float(arr.min()),
        "max": float(arr.max()),
    }
    for p, v in zip(percentiles, np.percentile(arr, percentiles)):
        out[f"p{p:g}"] = float(v)
    if K is not None:
        out["prob_above_K"] = prob_above(arr, K)
    return out


def log_return_stats(ST, S0: float) -> dict:
    """Sample mean and variance of ln(S_T / S0)."""
    if S0 <= 0:
        raise ValueError("S0 must be positive.")
    lr = np.log(_as_prices(ST) / S0)
    return {"mean": float(lr.mean()), "variance": float(lr.var(ddof=1))}


def gbm_log_return_theory(drift: float, sigma: float, T: float) -> dict:
    """Exact law of ln(S_T/S0) under GBM: Normal((drift - sigma^2/2) T, sigma^2 T)."""
    return {"mean": (drift - 0.5 * sigma**2) * T, "variance": sigma**2 * T}


def gbm_terminal_mean(S0: float, drift: float, T: float) -> float:
    """E[S_T] = S0 * exp(drift * T) (the sigma^2/2 terms cancel exactly)."""
    return S0 * math.exp(drift * T)

def gbm_quantile(S0: float, drift: float, sigma: float, t, q: float):
    """The q-quantile of S_t under GBM, for a time t or an array of times.

    S_t = S0 exp((drift - sigma^2/2) t + sigma sqrt(t) Z), so the quantile uses z_q = norm.ppf(q).
    """
    if S0 <= 0:
        raise ValueError("S0 must be positive.")
    if sigma < 0:
        raise ValueError("sigma cannot be negative.")
    if not 0.0 < q < 1.0:
        raise ValueError("q must be strictly between 0 and 1.")
    t = np.asarray(t, dtype=float)
    if (t < 0).any():
        raise ValueError("t cannot be negative.")
    return S0 * np.exp((drift - 0.5 * sigma**2) * t + sigma * np.sqrt(t) * norm.ppf(q))