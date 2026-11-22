"""Model audit: checks of the pricing code that do not rely on the code checking itself.

- an independent fourth pricing method (numerical integration) compared with Black-Scholes
- an external textbook anchor (Hull)
- no-arbitrage invariants over a seeded random grid of inputs
- seed robustness: the random validation checks rerun with many seeds
- a distribution test: GBM daily log returns against the theoretical normal (Kolmogorov-Smirnov)
Every check returns rows with the same columns as validation.run_validation().
"""
from __future__ import annotations

import math
from collections.abc import Callable, Sequence

import numpy as np
import pandas as pd
from scipy import integrate
from scipy.stats import binom, kstest, norm

from optionlab.greeks import greeks
from optionlab.implied_vol import implied_vol
from optionlab.pricing.black_scholes import bs_price, d1
from optionlab.simulation.gbm import simulate_gbm
from optionlab.validation import COLUMNS, run_validation

_TRUNCATE_SD = 12.0          # beyond 12 standard deviations the normal density is below 1e-31


def _row(area: str, check: str, expected, observed, criterion: str, status: str) -> dict:
    return {"area": area, "check": check, "expected": expected, "observed": observed,
            "criterion": criterion, "status": status}


# ---------------------------------------------------------------- 1. independent pricing method
def quad_price(S: float, K: float, r: float, sigma: float, T: float,
               option_type: str = "call") -> float:
    """European option price by numerical integration, without the Black-Scholes formula.

    Under risk-neutral GBM, y = ln(S_T) is normal with mean m = ln S + (r - sigma^2/2) T and
    standard deviation s = sigma sqrt(T). The price is the discounted expected payoff:
        call = e^(-rT) * integral over y > ln K of (e^y - K) * phi(y)
        put  = e^(-rT) * integral over y < ln K of (K - e^y) * phi(y)
    where phi is that normal density, integrated with scipy.integrate.quad.
    """
    if min(S, K, sigma, T) <= 0:
        raise ValueError("S, K, sigma and T must be positive.")
    if option_type not in ("call", "put"):
        raise ValueError("option_type must be 'call' or 'put'.")
    m = math.log(S) + (r - 0.5 * sigma**2) * T
    s = sigma * math.sqrt(T)
    log_k = math.log(K)

    def density(y: float) -> float:
        return math.exp(-0.5 * ((y - m) / s) ** 2) / (s * math.sqrt(2 * math.pi))

    # e^y * density(y) is a normal density shifted up by s^2, so widen the upper limit by s^2
    lo, hi = m - _TRUNCATE_SD * s, m + s * s + _TRUNCATE_SD * s
    if option_type == "call":
        a, b = max(log_k, lo), hi
        payoff = lambda y: (math.exp(y) - K) * density(y)
    else:
        a, b = lo, min(log_k, hi)
        payoff = lambda y: (K - math.exp(y)) * density(y)
    if a >= b:          # the strike is beyond 12 standard deviations: the option is worthless
        return 0.0
    value, _ = integrate.quad(payoff, a, b, epsabs=1e-12, epsrel=1e-12, limit=200)
    return math.exp(-r * T) * value


QUAD_CASES = [(100, 100, 0.05, 0.20, 1.0), (42, 40, 0.10, 0.20, 0.5), (100, 80, 0.03, 0.35, 2.0),
              (100, 130, 0.01, 0.15, 0.25), (50, 50, 0.0, 0.60, 3.0)]


def check_quadrature() -> list[dict]:
    """Quadrature vs Black-Scholes on QUAD_CASES (calls and puts), plus the anchor value."""
    worst = max(abs(quad_price(*case, kind) - bs_price(*case, kind))
                for case in QUAD_CASES for kind in ("call", "put"))
    anchor_q, anchor_bs = quad_price(100, 100, 0.05, 0.2, 1.0), bs_price(100, 100, 0.05, 0.2, 1.0)
    return [
        _row("Independent method", "quadrature matches Black-Scholes, S=K=100, r=5%, sigma=20%, T=1",
             anchor_bs, anchor_q, "abs error < 1e-8",
             "PASS" if abs(anchor_q - anchor_bs) < 1e-8 else "FAIL"),
        _row("Independent method", f"quadrature matches Black-Scholes ({2 * len(QUAD_CASES)} options)",
             0.0, worst, "max abs error < 1e-8", "PASS" if worst < 1e-8 else "FAIL"),
    ]


# ---------------------------------------------------------------- 2. textbook anchor
HULL = dict(S=42.0, K=40.0, r=0.10, sigma=0.20, T=0.5)       # Hull, Options, Futures and Other Derivatives


def check_textbook_anchor() -> list[dict]:
    """Hull's worked example: call 4.76 and put 0.81, to the 2 decimals the book prints."""
    rows = []
    for kind, book in (("call", 4.76), ("put", 0.81)):
        price = bs_price(**HULL, option_type=kind)
        rows.append(_row("Textbook anchor", f"Hull example {kind}: S=42, K=40, r=10%, sigma=20%, T=0.5",
                         book, price, "equal to 2 decimals",
                         "PASS" if round(price, 2) == book else "FAIL"))
    return rows


# ---------------------------------------------------------------- 3. no-arbitrage invariants
def _random_cases(n: int, seed: int) -> np.ndarray:
    """n rows of (S, K, r, sigma, T) over a realistic range, from a seeded generator."""
    rng = np.random.default_rng(seed)
    S = rng.uniform(50, 150, n)
    return np.column_stack([S, S * rng.uniform(0.6, 1.6, n), rng.uniform(0.0, 0.10, n),
                            rng.uniform(0.05, 0.80, n), rng.uniform(0.05, 3.0, n)])


def check_invariants(n_cases: int = 200, seed: int = 42,
                     pricer: Callable[..., float] = bs_price) -> list[dict]:
    """Count violations of no-arbitrage rules over n_cases random options; 0 is required.

    `pricer` (default bs_price) can be replaced to show that a broken pricer is caught.
    Strict positivity of delta, gamma and vega is only required where |d1| < 6: further out
    the exact values are below 1e-9 and round to 0 or 1 in floating point.
    """
    tol = 1e-9
    names = ["put-call parity", "price bounds", "call increases with sigma", "call increases with T",
             "call decreases in K, slope in [-e^(-rT), 0]", "call is convex in K",
             "call delta in (0, 1)", "gamma > 0 and vega > 0", "implied volatility recovers sigma"]
    violations = dict.fromkeys(names, 0)
    iv_tested = 0
    for S, K, r, sigma, T in _random_cases(n_cases, seed):
        C, P = pricer(S, K, r, sigma, T, "call"), pricer(S, K, r, sigma, T, "put")
        disc_k = K * math.exp(-r * T)
        violations["put-call parity"] += abs(C - P - (S - disc_k)) > tol * S
        violations["price bounds"] += not (max(S - disc_k, 0) - tol <= C <= S + tol
                                           and max(disc_k - S, 0) - tol <= P <= disc_k + tol)
        violations["call increases with sigma"] += pricer(S, K, r, sigma + 0.01, T, "call") < C - tol
        violations["call increases with T"] += pricer(S, K, r, sigma, T + 0.05, "call") < C - tol
        h = 0.01 * K
        up, down = pricer(S, K + h, r, sigma, T, "call"), pricer(S, K - h, r, sigma, T, "call")
        slope = (up - C) / h
        violations["call decreases in K, slope in [-e^(-rT), 0]"] += not (
            -math.exp(-r * T) - tol <= slope <= tol)
        violations["call is convex in K"] += down - 2 * C + up < -tol
        g = greeks(S, K, r, sigma, T, "call")
        central = abs(d1(S, K, r, sigma, T)) < 6
        violations["call delta in (0, 1)"] += not (0 <= g.delta <= 1 and (not central or 0 < g.delta < 1))
        violations["gamma > 0 and vega > 0"] += not (g.gamma >= 0 and g.vega >= 0
                                                     and (not central or (g.gamma > 0 and g.vega > 0)))
        if g.vega > 1e-3:                    # where the price actually depends on sigma
            iv_tested += 1
            try:
                violations["implied volatility recovers sigma"] += abs(
                    implied_vol(C, S, K, r, T, "call") - sigma) > 1e-6
            except ValueError:
                violations["implied volatility recovers sigma"] += 1
    rows = []
    for name in names:
        n = iv_tested if name.startswith("implied") else n_cases
        rows.append(_row("No-arbitrage", f"{name} ({n} random options, seed {seed})", 0,
                         int(violations[name]), "0 violations",
                         "PASS" if violations[name] == 0 else "FAIL"))
    return rows


# ---------------------------------------------------------------- 4. seed robustness
def check_seed_robustness(n_seeds: int = 20,
                          runner: Callable[[int], pd.DataFrame] = run_validation) -> dict:
    """Rerun the validation suite with seeds 0 .. n_seeds-1 and report the pass rates.

    PASS: every check passes on every seed. WARN: some check fails on at most 10% of seeds
    (a statistical check at the 4-standard-error level may occasionally miss). FAIL: worse.
    """
    passes: dict[str, int] = {}
    for seed in range(n_seeds):
        table = runner(seed)
        for check, status in zip(table["check"], table["status"]):
            passes[check] = passes.get(check, 0) + (status == "PASS")
    worst_check = min(passes, key=passes.get)
    worst_rate = passes[worst_check] / n_seeds
    always = sum(count == n_seeds for count in passes.values())
    status = "PASS" if worst_rate == 1 else ("WARN" if worst_rate >= 0.9 else "FAIL")
    observed = f"{always} of {len(passes)} checks pass on every seed"
    if worst_rate < 1:
        observed += f"; lowest: {worst_rate:.0%} for '{worst_check}'"
    return _row("Seed robustness", f"validation suite rerun with {n_seeds} seeds (0..{n_seeds - 1})",
                "every check passes on every seed", observed,
                "PASS: 100%; WARN: each check >= 90%", status)


# ---------------------------------------------------------------- 5. distribution test
def gbm_ks_pvalues(n_tests: int = 20, n_steps: int = 1000, seed: int = 42,
                   mu: float = 0.08, sigma: float = 0.20) -> np.ndarray:
    """KS p-values: each test compares one simulated path's daily log returns with the
    theoretical Normal((mu - sigma^2/2) dt, sigma^2 dt), dt = 1/252."""
    dt = 1 / 252
    paths = simulate_gbm(100.0, mu, sigma, T=n_steps * dt, n_steps=n_steps, n_paths=n_tests, seed=seed)
    returns = np.diff(np.log(paths), axis=1)
    dist = norm(loc=(mu - 0.5 * sigma**2) * dt, scale=sigma * math.sqrt(dt))
    return np.array([kstest(row, dist.cdf).pvalue for row in returns])


def ks_verdict(p_values: Sequence[float], level: float = 0.05,
               alpha: float = 0.001) -> tuple[int, float, str]:
    """(rejections, chance probability, status) for repeated KS tests.

    If the data really are normal, each test rejects with probability `level`, so the number
    of rejections is Binomial(n, level). FAIL only if at least this many rejections would happen
    by chance with probability below `alpha`; occasional single rejections are expected.
    """
    p_values = np.asarray(p_values, dtype=float)
    rejections = int((p_values < level).sum())
    chance = float(binom.sf(rejections - 1, len(p_values), level))      # P(X >= rejections)
    return rejections, chance, "FAIL" if chance < alpha else "PASS"


def check_gbm_ks(n_tests: int = 20, seed: int = 42) -> dict:
    """The KS distribution check as one report row."""
    p_values = gbm_ks_pvalues(n_tests=n_tests, seed=seed)
    rejections, chance, status = ks_verdict(p_values)
    return _row("Distribution", f"GBM daily log returns are normal (KS test, {n_tests} paths x 1000 days)",
                f"about {0.05 * n_tests:.1f} rejections at the 5% level by chance",
                f"{rejections} rejections; median p = {float(np.median(p_values)):.2f}; "
                f"chance of >= {rejections}: {chance:.3f}",
                "FAIL only if that chance < 0.001", status)


# ---------------------------------------------------------------- all model checks
def run_model_audit(quick: bool = True, seed: int = 42) -> pd.DataFrame:
    """All model checks as one table. quick=True uses smaller sizes (for tests and CI)."""
    n_cases, n_seeds, n_tests = (200, 3, 20) if quick else (2000, 20, 100)
    rows = (check_quadrature() + check_textbook_anchor() + check_invariants(n_cases, seed)
            + [check_seed_robustness(n_seeds), check_gbm_ks(n_tests, seed)])
    return pd.DataFrame(rows, columns=COLUMNS)
