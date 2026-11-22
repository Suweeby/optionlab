"""Audit report: run every audit, combine the results, and record how to reproduce them.

run_all() runs the validation suite, the model audit, the consistency audit and the claims,
and returns one table (area, check, expected, observed, criterion, status), the claims table,
and a reproducibility manifest (time, git commit, versions, seeds, sizes, counts).
"""
from __future__ import annotations

import json
import platform
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

import pandas as pd

from optionlab.audit.claims import ClaimContext, claims_as_rows, run_claims
from optionlab.audit.consistency_audit import PROJECT_ROOT, run_consistency_audit
from optionlab.audit.model_audit import audit_sizes, run_model_audit
from optionlab.validation import COLUMNS, run_validation, to_markdown as table_markdown

STATUSES = ("PASS", "WARN", "FAIL")
PACKAGES = ("optionlab", "numpy", "scipy", "pandas")


@dataclass(frozen=True)
class AuditReport:
    """The combined result of one audit run."""

    table: pd.DataFrame          # every check, COLUMNS
    claims: pd.DataFrame         # one row per claim, claims.CLAIM_COLUMNS
    manifest: dict

    @property
    def counts(self) -> dict[str, int]:
        """PASS / WARN / FAIL counts over every check (claims included)."""
        return {s: int((self.table["status"] == s).sum()) for s in STATUSES}

    @property
    def passed(self) -> bool:
        """True when nothing failed (warnings are allowed)."""
        return self.counts["FAIL"] == 0


# ---------------------------------------------------------------- manifest helpers
def _git(root: Path, *args: str) -> str | None:
    try:
        done = subprocess.run(["git", *args], cwd=root, capture_output=True, text=True,
                              timeout=10, check=True)
    except (OSError, subprocess.SubprocessError):       # git missing, not a repo, timeout
        return None
    return done.stdout.strip()


def git_commit(root: Path | None = None) -> str:
    """The checked-out commit hash, or 'unknown' if git is unavailable."""
    return _git(Path(root) if root else PROJECT_ROOT, "rev-parse", "HEAD") or "unknown"


def git_has_changes(root: Path | None = None) -> bool | str:
    """True if there are uncommitted changes, False if clean, 'unknown' without git."""
    status = _git(Path(root) if root else PROJECT_ROOT, "status", "--porcelain")
    return "unknown" if status is None else bool(status)


def _version(package: str) -> str:
    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return "not installed"


def build_manifest(quick: bool, seed: int, counts: dict[str, int], n_checks: int,
                   seconds: float, root: Path | None = None) -> dict:
    """Everything needed to reproduce (and date) an audit run."""
    sizes = audit_sizes(quick)
    return {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "mode": "quick" if quick else "full",
        "git_commit": git_commit(root),
        "uncommitted_changes": git_has_changes(root),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": {name: _version(name) for name in PACKAGES},
        "seeds": {"validation": seed, "model_audit": seed,
                  "seed_robustness": f"0..{sizes['robustness_seeds'] - 1}"},
        "sizes": sizes,
        "checks": n_checks,
        "counts": counts,
        "duration_seconds": round(seconds, 1),
    }


# ---------------------------------------------------------------- run everything
def run_all(quick: bool = True, seed: int = 42, root: Path | None = None) -> AuditReport:
    """Run validation, model audit, consistency audit and claims; return the combined report."""
    root = Path(root) if root else PROJECT_ROOT
    started = time.perf_counter()
    validation = run_validation(seed)
    model = run_model_audit(quick, seed)
    consistency = run_consistency_audit(root, validation)
    claims = run_claims(ClaimContext(validation, model, consistency, root))
    table = pd.concat([validation, model, consistency, claims_as_rows(claims)], ignore_index=True)
    table = table[COLUMNS]
    counts = {s: int((table["status"] == s).sum()) for s in STATUSES}
    manifest = build_manifest(quick, seed, counts, len(table), time.perf_counter() - started, root)
    return AuditReport(table, claims, manifest)


# ---------------------------------------------------------------- rendering
def _escape(table: pd.DataFrame) -> pd.DataFrame:
    """Escape '|' so text cannot break a markdown table."""
    return table.map(lambda v: v.replace("|", "\\|") if isinstance(v, str) else v)


def to_markdown(report: AuditReport) -> str:
    """The report as a Markdown document (summary, manifest, results, claims)."""
    m, c = report.manifest, report.counts             # missing manifest fields show as '?'
    commit = str(m.get("git_commit", "unknown"))[:12]
    dirty = " (with uncommitted changes)" if m.get("uncommitted_changes") is True else ""
    manifest_rows = pd.DataFrame(
        [(key, json.dumps(value) if isinstance(value, dict) else value) for key, value in m.items()],
        columns=["field", "value"])
    return "\n".join([
        "# OptionLab audit report",
        "",
        f"Generated {m.get('generated_utc', '?')} · {m.get('mode', '?')} mode · commit `{commit}`{dirty}",
        "",
        f"**{c['PASS']} PASS · {c['WARN']} WARN · {c['FAIL']} FAIL** across {len(report.table)} checks.",
        "",
        "These checks show that the code does what the project states: reference values, "
        "agreement between independent methods, no-arbitrage rules, calibrated random results, "
        "documents that match the results, and honest wording. They do not show that "
        "Black-Scholes describes real markets, that the downloaded data is correct, or that "
        "the list of checks is complete.",
        "",
        "## Reproducibility manifest",
        "",
        table_markdown(_escape(manifest_rows)),
        "",
        "## Claims",
        "",
        table_markdown(_escape(report.claims)),
        "",
        "## All checks",
        "",
        table_markdown(_escape(report.table)),
        "",
    ])


def _plain(value):
    """JSON fallback for numpy numbers and anything else."""
    return value.item() if hasattr(value, "item") else str(value)


def to_json(report: AuditReport) -> str:
    """The report as JSON: {"manifest": ..., "claims": [...], "results": [...]}."""
    return json.dumps({"manifest": report.manifest,
                       "claims": report.claims.to_dict(orient="records"),
                       "results": report.table.to_dict(orient="records")},
                      indent=2, default=_plain)
