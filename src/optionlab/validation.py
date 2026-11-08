"""Validation suite: the correctness anchors of the project, run as a readable report.

Each check compares an observed value with an expected one under a stated criterion.
Random checks use fixed seeds so the report is reproducible.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy.stats import norm

from optionlab.greeks import fd_greeks, greeks
from optionlab.implied_vol import implied_vol, implied_vol_detailed
from optionlab.pricing.binomial import binomial_price
from optionlab.pricing.black_scholes import bs_price, call_price, d2, put_price
from optionlab.pricing.monte_carlo import mc_price
from optionlab.simulation.gbm import simulate_gbm, simulate_terminal_prices
from optionlab.simulation.stats import (
    gbm_log_return_theory, gbm_terminal_mean, log_return_stats, prob_above,
)
from optionlab.volatility import historical_volatility

ANCHOR = dict(S=100.0, K=100.0, r=0.05, sigma=0.2, T=1.0)
COLUMNS = ["area", "check", "expected", "observed", "criterion", "status"]


def _row(area, check, expected, observed, criterion, passed):
    return {"area": area, "check": check, "expected": expected, "observed": observed,
            "criterion": criterion, "status": "PASS" if passed else "FAIL"}


def _raises(fn, *args, **kwargs) -> bool:
    try:
        fn(*args, **kwargs)
    except ValueError:
        return True
    return False


def run_validation(seed: int = 42) -> pd.DataFrame:
    """Run every check and return a table with a PASS/FAIL status per check."""
    A = ANCHOR
    rows = []
    rng = np.random.default_rng(seed)

    # ---- Black-Scholes
    c, p = call_price(**A), put_price(**A)
    rows.append(_row("Black-Scholes", "call, S=K=100, r=5%, sigma=20%, T=1", 10.4506, c, "abs error < 1e-4", abs(c - 10.4506) < 1e-4))
    rows.append(_row("Black-Scholes", "put, same inputs", 5.5735, p, "abs error < 1e-4", abs(p - 5.5735) < 1e-4))
    parity = A["S"] - A["K"] * math.exp(-A["r"] * A["T"])
    rows.append(_row("Black-Scholes", "put-call parity C - P = S - K e^(-rT)", parity, c - p, "abs error < 1e-10", abs((c - p) - parity) < 1e-10))
    near_zero_T = call_price(110, 100, 0.05, 0.2, 1e-9)
    rows.append(_row("Black-Scholes", "T -> 0: call tends to intrinsic value", 10.0, near_zero_T, "abs error < 1e-4", abs(near_zero_T - 10.0) < 1e-4))
    fwd = max(100 - 90 * math.exp(-0.05), 0)
    near_zero_sigma = call_price(100, 90, 0.05, 1e-8, 1.0)
    rows.append(_row("Black-Scholes", "sigma -> 0: call tends to discounted forward payoff", fwd, near_zero_sigma, "abs error < 1e-6", abs(near_zero_sigma - fwd) < 1e-6))
    deep_itm = call_price(300, 100, 0.05, 0.2, 1.0)
    rows.append(_row("Black-Scholes", "deep in the money: call ~ S - K e^(-rT)", 300 - 100 * math.exp(-0.05), deep_itm, "abs error < 1e-6", abs(deep_itm - (300 - 100 * math.exp(-0.05))) < 1e-6))
    deep_otm = call_price(20, 100, 0.05, 0.2, 1.0)
    rows.append(_row("Black-Scholes", "deep out of the money: call ~ 0", 0.0, deep_otm, "abs error < 1e-6", abs(deep_otm) < 1e-6))
    bad = [dict(S=-1), dict(K=-1), dict(sigma=-0.1), dict(T=-1)]
    all_raise = all(_raises(call_price, **{**A, **b}) and _raises(put_price, **{**A, **b}) for b in bad)
    rows.append(_row("Black-Scholes", "negative S, K, sigma or T raise ValueError", "all raise", "all raise" if all_raise else "some do not", "every case raises", all_raise))

    # ---- binomial tree
    bc, bp = binomial_price(n=500, **A), binomial_price(n=500, option_type="put", **A)
    rows.append(_row("Binomial", "put-call parity inside the tree (n=500)", parity, bc - bp, "abs error < 1e-9", abs((bc - bp) - parity) < 1e-9))
    err2000 = binomial_price(n=2000, **A) - c
    rows.append(_row("Binomial", "converges to Black-Scholes (n=2000)", 0.0, err2000, "abs error < 3/n", abs(err2000) < 3 / 2000))
    worst = lambda ns: max(abs(binomial_price(n=n, **A) - c) for n in ns)
    ratio = worst(range(100, 106)) / worst(range(1000, 1006))
    rows.append(_row("Binomial", "error shrinks like 1/n (10x steps -> 10x smaller)", 10.0, ratio, "ratio within 25% of 10", abs(ratio - 10) < 2.5))

    # ---- Monte Carlo
    for n_paths in (10_000, 100_000, 1_000_000):
        res = mc_price(**A, n_paths=n_paths, seed=rng)
        z = (res.price - c) / res.std_error
        rows.append(_row("Monte Carlo", f"price within 3 standard errors of Black-Scholes (N={n_paths:,})", c, res.price, f"|z| < 3 (z = {z:+.2f})", abs(z) < 3))
    results = [mc_price(**A, n_paths=2000, seed=rng) for _ in range(300)]
    coverage = float(np.mean([r.ci_low <= c <= r.ci_high for r in results]))
    rows.append(_row("Monte Carlo", "95% intervals contain the true price (300 runs)", 0.95, coverage, "coverage between 0.92 and 0.98", 0.92 <= coverage <= 0.98))

    def rmse(n, reps):
        prices = np.array([mc_price(**A, n_paths=n, seed=rng).price for _ in range(reps)])
        return float(np.sqrt(np.mean((prices - c) ** 2)))

    ratio = rmse(1_000, 100) / rmse(100_000, 100)
    rows.append(_row("Monte Carlo", "error shrinks like 1/sqrt(N) (100x paths -> 10x smaller)", 10.0, ratio, "ratio within 35% of 10", abs(ratio - 10) < 3.5))
    plain = mc_price(**A, n_paths=100_000, seed=1).std_error
    anti = mc_price(**A, n_paths=100_000, seed=1, antithetic=True).std_error
    rows.append(_row("Monte Carlo", "antithetic variates reduce the standard error", "ratio < 1", anti / plain, "ratio < 0.9", anti / plain < 0.9))
    mc_c = mc_price(**A, n_paths=200_000, seed=rng)
    mc_p = mc_price(**A, n_paths=200_000, option_type="put", seed=rng)
    se = math.hypot(mc_c.std_error, mc_p.std_error)
    rows.append(_row("Monte Carlo", "put-call parity holds within noise (N=200,000)", parity, mc_c.price - mc_p.price, "within 4 combined standard errors", abs((mc_c.price - mc_p.price) - parity) < 4 * se))

    # ---- GBM generator
    n = 200_000
    ST = simulate_terminal_prices(100.0, 0.08, 0.2, 1.0, n, seed=rng)
    sample, theory = log_return_stats(ST, 100.0), gbm_log_return_theory(0.08, 0.2, 1.0)
    se_mean = math.sqrt(theory["variance"] / n)
    rows.append(_row("GBM", "mean of ln(S_T/S_0) = (mu - sigma^2/2) T", theory["mean"], sample["mean"], "within 4 standard errors", abs(sample["mean"] - theory["mean"]) < 4 * se_mean))
    rows.append(_row("GBM", "variance of ln(S_T/S_0) = sigma^2 T", theory["variance"], sample["variance"], "relative error < 2%", abs(sample["variance"] / theory["variance"] - 1) < 0.02))
    rows.append(_row("GBM", "E[S_T] = S_0 e^(mu T)", gbm_terminal_mean(100.0, 0.08, 1.0), float(ST.mean()), "within 4 standard errors", abs(ST.mean() - gbm_terminal_mean(100.0, 0.08, 1.0)) < 4 * ST.std(ddof=1) / math.sqrt(n)))
    RN = simulate_terminal_prices(100.0, 0.05, 0.2, 1.0, n, seed=rng)
    disc = math.exp(-0.05) * RN
    rows.append(_row("GBM", "risk-neutral: discounted E[S_T] = S_0", 100.0, float(disc.mean()), "within 4 standard errors", abs(disc.mean() - 100.0) < 4 * disc.std(ddof=1) / math.sqrt(n)))
    n_d2 = float(norm.cdf(d2(100.0, 105.0, 0.05, 0.2, 1.0)))
    est = prob_above(RN, 105.0)
    rows.append(_row("GBM", "risk-neutral P(S_T > K) = N(d2)", n_d2, est, "within 4 standard errors", abs(est - n_d2) < 4 * math.sqrt(n_d2 * (1 - n_d2) / n)))
    path = simulate_gbm(100.0, 0.05, 0.30, T=40, n_steps=40 * 252, n_paths=1, seed=rng)[0]
    est_vol = historical_volatility(pd.Series(path))
    rows.append(_row("Volatility", "estimator recovers sigma = 30% from 40 years of daily data", 0.30, est_vol, "abs error < 0.01", abs(est_vol - 0.30) < 0.01))

    # ---- Greeks
    worst_rel = 0.0
    for option_type in ("call", "put"):
        a, f = greeks(**A, option_type=option_type), fd_greeks(**A, option_type=option_type)
        for name in ("delta", "gamma", "vega", "theta", "rho"):
            x, y = getattr(a, name), getattr(f, name)
            worst_rel = max(worst_rel, abs(x - y) / max(abs(x), 1e-8))
    rows.append(_row("Greeks", "analytic Greeks match finite differences (5 Greeks, call and put)", 0.0, worst_rel, "max relative difference < 1e-4", worst_rel < 1e-4))
    gc, gp = greeks(**A), greeks(**A, option_type="put")
    rows.append(_row("Greeks", "delta(call) - delta(put) = 1", 1.0, gc.delta - gp.delta, "abs error < 1e-12", abs(gc.delta - gp.delta - 1) < 1e-12))

    # ---- implied volatility
    for option_type in ("call", "put"):
        price = bs_price(A["S"], A["K"], A["r"], 0.25, A["T"], option_type)
        iv = implied_vol(price, A["S"], A["K"], A["r"], A["T"], option_type)
        rows.append(_row("Implied volatility", f"round trip: price a {option_type} at sigma=0.25, solve back", 0.25, iv, "abs error < 1e-8", abs(iv - 0.25) < 1e-8))
    detail = implied_vol_detailed(bs_price(100, 100, 0.05, 0.25, 1.0), 100, 100, 0.05, 1.0)
    rows.append(_row("Implied volatility", "Newton-Raphson converges in a few steps at the money", "<= 6", detail.iterations, "iterations <= 6", detail.method == "newton" and detail.iterations <= 6))
    far = bs_price(100, 200, 0.03, 0.8, 0.1)
    fallback = implied_vol_detailed(far, 100, 200, 0.03, 0.1)
    rows.append(_row("Implied volatility", "bisection fallback solves the case where Newton cannot start", 0.8, fallback.vol, f"abs error < 1e-7, method = {fallback.method}", fallback.method == "bisection" and abs(fallback.vol - 0.8) < 1e-7))
    below = _raises(implied_vol, 10.0, 100, 90, 0.05, 1.0)
    rows.append(_row("Implied volatility", "price below the no-arbitrage minimum raises ValueError", "raises", "raises" if below else "does not", "raises", below))

    return pd.DataFrame(rows, columns=COLUMNS)


def _fmt(value) -> str:
    if isinstance(value, (float, np.floating)):
        return f"{value:.6g}"
    return str(value)


def to_markdown(table: pd.DataFrame) -> str:
    """Render a table as GitHub-flavoured markdown (no extra dependency)."""
    header = "| " + " | ".join(table.columns) + " |"
    divider = "|" + "|".join(" --- " for _ in table.columns) + "|"
    body = ["| " + " | ".join(_fmt(v) for v in row) + " |" for row in table.itertuples(index=False)]
    return "\n".join([header, divider, *body])
