"""Claims registry: every key claim the project makes, where it makes it, and the check behind it.

Most checks look up rows of tables that were already computed (the validation suite and the
model and consistency audits), so nothing is checked twice. A claim with no check, or whose
check fails or crashes, is a FAIL: an unchecked claim is not accepted.
"""
from __future__ import annotations

import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from optionlab.data import PriceData, load_prices
from optionlab.market import require_real_prices
from optionlab.validation import COLUMNS

CLAIM_COLUMNS = ["id", "statement", "location", "status", "evidence"]


@dataclass(frozen=True)
class ClaimContext:
    """What claim checks may look at: the computed audit tables and the repository root."""

    validation: pd.DataFrame
    model: pd.DataFrame
    consistency: pd.DataFrame
    root: Path


CheckFn = Callable[[ClaimContext], "tuple[bool, str]"]


@dataclass(frozen=True)
class Claim:
    """One claim: an id, the sentence, where the project states it, and how it is checked."""

    id: str
    statement: str
    location: str
    check: CheckFn | None


# ---------------------------------------------------------------- check builders
def rows_pass(table: str, contains: tuple[str, ...] = (), area: str = "",
              allowed: tuple[str, ...] = ("PASS",)) -> CheckFn:
    """A check that passes if every matching row of a computed table has an allowed status.

    Rows match if their 'check' text contains any of `contains`, or their area equals `area`.
    No matching row is a failure (the claim would be unchecked).
    """
    def check(ctx: ClaimContext) -> tuple[bool, str]:
        frame = getattr(ctx, table)
        mask = pd.Series(False, index=frame.index)
        for text in contains:
            mask |= frame["check"].str.contains(text, regex=False)
        if area:
            mask |= frame["area"] == area
        rows = frame[mask]
        if rows.empty:
            return False, f"no {table} check matches {contains or area!r}"
        good = rows["status"].isin(allowed)
        return bool(good.all()), f"{int(good.sum())} of {len(rows)} {table} checks pass"
    return check


def all_of(*checks: CheckFn) -> CheckFn:
    """A check that passes only if every given check passes."""
    def check(ctx: ClaimContext) -> tuple[bool, str]:
        results = [c(ctx) for c in checks]
        return all(ok for ok, _ in results), "; ".join(text for _, text in results)
    return check


def _raises(fn: Callable, *args) -> bool:
    try:
        fn(*args)
    except ValueError:
        return True
    return False


# ---------------------------------------------------------------- claims that need their own check
def check_antithetic_ratio(ctx: ClaimContext) -> tuple[bool, str]:
    """README: antithetic variates cut the standard error to about 0.71 of the plain estimate."""
    rows = ctx.validation[ctx.validation["check"].str.contains("antithetic variates reduce", regex=False)]
    if rows.empty:
        return False, "validation check for antithetic variates not found"
    ratio = float(rows["observed"].iloc[0])
    return 0.66 <= ratio <= 0.76, f"standard-error ratio {ratio:.3f} (claim: about 0.71)"


def check_synthetic_never_compared(ctx: ClaimContext) -> tuple[bool, str]:
    """The guard refuses SYNTHETIC data and uploads, and the Market Data page really calls it."""
    series = pd.Series([100.0, 101.0], index=pd.bdate_range("2026-01-05", periods=2))
    refused = all(_raises(require_real_prices, PriceData("X", series, s)) for s in ("sample", "upload"))
    allowed = not _raises(require_real_prices, PriceData("X", series, "network"))
    page = ctx.root / "app" / "pages" / "4_Market_Data.py"
    wired = page.is_file() and "require_real_prices(pdata)" in page.read_text(encoding="utf-8")
    return refused and allowed and wired, (
        f"guard refuses sample and upload: {refused}; allows downloads: {allowed}; "
        f"Market Data page calls it: {wired}")


def check_sample_never_cached(ctx: ClaimContext) -> tuple[bool, str]:
    """Offline with no cache: the fallback is labelled SYNTHETIC and nothing is written."""
    def offline(*_args):
        raise ConnectionError("offline (audit check)")

    with tempfile.TemporaryDirectory() as folder:
        data = load_prices("AUDIT", cache_dir=folder, fetcher=offline)
        written = list(Path(folder).iterdir())
    labelled = "SYNTHETIC" in data.note
    ok = data.source == "sample" and not data.is_real and labelled and not written
    return ok, f"source = {data.source}; labelled SYNTHETIC: {labelled}; files written: {len(written)}"


def check_tests_need_no_network(ctx: ClaimContext) -> tuple[bool, str]:
    """Static check: the autouse network guard exists, and a test proves it blocks connections.

    The guard itself only runs inside pytest, so this check confirms it is in place; the test
    suite (tests/test_network_guard.py) confirms it works.
    """
    conftest = ctx.root / "tests" / "conftest.py"
    guard_test = ctx.root / "tests" / "test_network_guard.py"
    if not conftest.is_file():
        return False, "tests/conftest.py not found"
    text = conftest.read_text(encoding="utf-8")
    ok = "autouse=True" in text and "def _no_network" in text and guard_test.is_file()
    return ok, ("autouse network guard in tests/conftest.py: " + str("def _no_network" in text)
                + f"; tests/test_network_guard.py present: {guard_test.is_file()}")


# ---------------------------------------------------------------- the registry
CLAIMS: list[Claim] = [
    Claim("bs-anchor", "Black-Scholes gives 10.4506 for the anchor call (S=K=100, r=5%, sigma=20%, T=1)",
          "README: Using the engine, Results", rows_pass("validation", ("call, S=K=100",))),
    Claim("pricers-agree", "Black-Scholes, the binomial tree and Monte Carlo agree within their errors",
          "README: What it does; Pricer page",
          rows_pass("validation", ("converges to Black-Scholes", "within 3 standard errors"))),
    Claim("independent-method", "A fourth method (numerical integration) reproduces Black-Scholes",
          "Audit page; docs/theory.md", rows_pass("model", ("quadrature matches",))),
    Claim("textbook-anchor", "Prices match Hull's worked example (call 4.76, put 0.81)",
          "Audit page", rows_pass("model", ("Hull example",))),
    Claim("binomial-rate", "The binomial error falls like 1/n", "README: Results; Convergence page",
          rows_pass("validation", ("like 1/n (",))),
    Claim("mc-rate", "The Monte Carlo error falls like 1/sqrt(N)", "README: Results; Convergence page",
          rows_pass("validation", ("1/sqrt(N)",))),
    Claim("mc-coverage", "95% intervals cover the true price about 95% of the time",
          "README: Design notes; Pricer page", rows_pass("validation", ("95% intervals",))),
    Claim("antithetic", "Antithetic variates cut the standard error to about 0.71",
          "README: Results", check_antithetic_ratio),
    Claim("greeks-fd", "Analytic Greeks match finite differences", "README: What it does",
          rows_pass("validation", ("finite differences",))),
    Claim("iv-recovers", "The implied-volatility solver recovers sigma", "README: What it does",
          all_of(rows_pass("validation", ("round trip",)),
                 rows_pass("model", ("implied volatility recovers",)))),
    Claim("no-arbitrage", "Prices respect no-arbitrage rules (parity, bounds, monotonicity, convexity)",
          "docs/theory.md", rows_pass("model", area="No-arbitrage")),
    Claim("gbm-theory", "Simulated GBM matches its theoretical moments and distribution",
          "README: What it does; Simulator page",
          all_of(rows_pass("validation", area="GBM"), rows_pass("model", ("KS test",)))),
    Claim("seed-robust", "Results do not depend on one lucky seed",
          "README: Design notes", rows_pass("model", area="Seed robustness", allowed=("PASS", "WARN"))),
    Claim("docs-match-results", "README numbers match docs/results and the current code",
          "README: Results", rows_pass("consistency", area="README vs results")),
    Claim("honest-language", "The app and docs make no price predictions",
          "README: top; every page", rows_pass("consistency", area="Honesty")),
    Claim("documented", "Public functions have docstrings and type hints",
          "README: Design notes", rows_pass("consistency", area="Documentation")),
    Claim("synthetic-not-compared", "Synthetic (and uploaded) data is never used for the model-vs-market comparison",
          "README: Design notes; Market Data page", check_synthetic_never_compared),
    Claim("synthetic-labelled", "Synthetic fallback data is always labelled and never cached",
          "README: Design notes", check_sample_never_cached),
    Claim("tests-offline", "The tests need no network", "README: Design notes",
          check_tests_need_no_network),
]


def run_claims(ctx: ClaimContext, claims: list[Claim] | None = None) -> pd.DataFrame:
    """Evaluate each claim; a missing, failing or crashing check is a FAIL."""
    rows = []
    for claim in CLAIMS if claims is None else claims:
        if claim.check is None:
            ok, evidence = False, "no check registered for this claim"
        else:
            try:
                ok, evidence = claim.check(ctx)
            except Exception as exc:     # a broken check must not hide the other claims
                ok, evidence = False, f"check raised {type(exc).__name__}: {exc}"
        rows.append({"id": claim.id, "statement": claim.statement, "location": claim.location,
                     "status": "PASS" if ok else "FAIL", "evidence": evidence})
    return pd.DataFrame(rows, columns=CLAIM_COLUMNS)


def claims_as_rows(claims: pd.DataFrame) -> pd.DataFrame:
    """The claims table in the common audit columns, for the combined report table."""
    rows = [{"area": "Claims", "check": f"[{c.id}] {c.statement}", "expected": "its check passes",
             "observed": c.evidence, "criterion": f"stated in: {c.location}", "status": c.status}
            for c in claims.itertuples(index=False)]
    return pd.DataFrame(rows, columns=COLUMNS)
