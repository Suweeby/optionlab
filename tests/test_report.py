"""Claims registry, combined audit report, and the run_audit script."""
import importlib.util
import json
import subprocess
from pathlib import Path

import pandas as pd
import pytest

from optionlab.audit import report as report_module
from optionlab.audit.claims import (
    CLAIM_COLUMNS, CLAIMS, Claim, ClaimContext, check_sample_never_cached,
    check_synthetic_never_compared, check_tests_need_no_network, claims_as_rows, rows_pass,
    run_claims,
)
from optionlab.audit.consistency_audit import PROJECT_ROOT
from optionlab.audit.report import AuditReport, git_commit, run_all, to_json, to_markdown
from optionlab.validation import COLUMNS

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_audit.py"


@pytest.fixture(scope="module")
def report():
    """One real quick audit, shared by the tests in this file (it takes about 10 s)."""
    return run_all(quick=True)


def _table(rows):
    return pd.DataFrame([dict(area="A", check=c, expected="", observed="", criterion="", status=s)
                         for c, s in rows], columns=COLUMNS)


def _context(validation=None, root=PROJECT_ROOT):
    empty = _table([])
    return ClaimContext(validation if validation is not None else empty, empty, empty, Path(root))


# ---------------------------------------------------------------- the registry
def test_registry_is_complete_and_well_formed():
    ids = [c.id for c in CLAIMS]
    assert len(ids) == len(set(ids)) >= 15
    assert all(c.statement and c.location and callable(c.check) for c in CLAIMS)


def test_every_claim_passes_on_the_real_project(report):
    assert list(report.claims.columns) == CLAIM_COLUMNS
    assert len(report.claims) == len(CLAIMS)
    failing = report.claims[report.claims["status"] != "PASS"]
    assert failing.empty, failing[["id", "evidence"]]


def test_unchecked_failing_or_crashing_claims_fail():
    def crash(ctx):
        raise RuntimeError("boom")

    claims = [Claim("none", "has no check", "nowhere", None),
              Claim("fails", "a failing row", "nowhere", rows_pass("validation", ("x",))),
              Claim("missing", "no matching row", "nowhere", rows_pass("validation", ("zzz",))),
              Claim("crash", "check crashes", "nowhere", crash)]
    table = run_claims(_context(_table([("x check", "FAIL")])), claims).set_index("id")
    assert (table["status"] == "FAIL").all()
    assert "no check registered" in table.loc["none", "evidence"]
    assert "no validation check matches" in table.loc["missing", "evidence"]
    assert "RuntimeError: boom" in table.loc["crash", "evidence"]


def test_rows_pass_allows_warnings_only_when_told():
    ctx = _context(_table([("seed check", "WARN")]))
    assert rows_pass("validation", ("seed",))(ctx)[0] is False
    assert rows_pass("validation", ("seed",), allowed=("PASS", "WARN"))(ctx)[0] is True


def test_market_guard_claim_needs_the_page_to_call_the_guard(tmp_path):
    assert check_synthetic_never_compared(_context())[0] is True
    (tmp_path / "app" / "pages").mkdir(parents=True)
    (tmp_path / "app" / "pages" / "4_Market_Data.py").write_text("# guard removed\n")
    ok, evidence = check_synthetic_never_compared(_context(root=tmp_path))
    assert ok is False and "Market Data page calls it: False" in evidence


def test_synthetic_data_claim_writes_nothing():
    ok, evidence = check_sample_never_cached(_context())
    assert ok is True and "files written: 0" in evidence


def test_no_network_claim_fails_without_the_guard(tmp_path):
    assert check_tests_need_no_network(_context())[0] is True
    assert check_tests_need_no_network(_context(root=tmp_path))[0] is False


def test_claims_become_report_rows(report):
    rows = claims_as_rows(report.claims)
    assert list(rows.columns) == COLUMNS and (rows["area"] == "Claims").all()


# ---------------------------------------------------------------- the report
def test_report_combines_every_audit(report):
    table = report.table
    assert list(table.columns) == COLUMNS
    assert {"Black-Scholes", "No-arbitrage", "Honesty", "Claims"} <= set(table["area"])
    assert len(table) == 31 + 15 + 10 + len(CLAIMS)
    assert sum(report.counts.values()) == len(table)
    assert report.passed and report.counts["FAIL"] == 0


def test_manifest_records_how_to_reproduce(report):
    m = report.manifest
    for key in ("generated_utc", "mode", "git_commit", "uncommitted_changes", "python",
                "packages", "seeds", "sizes", "counts", "duration_seconds"):
        assert key in m
    assert m["mode"] == "quick" and m["seeds"]["validation"] == 42
    assert m["sizes"]["robustness_seeds"] == 3 and m["counts"] == report.counts
    assert m["git_commit"] == "unknown" or len(m["git_commit"]) == 40


def test_git_commit_is_unknown_without_git(monkeypatch):
    def no_git(*args, **kwargs):
        raise FileNotFoundError("git")
    monkeypatch.setattr(subprocess, "run", no_git)
    assert git_commit() == "unknown"


def test_markdown_and_json_renderings(report):
    text = to_markdown(report)
    assert text.startswith("# OptionLab audit report")
    for heading in ("## Reproducibility manifest", "## Claims", "## All checks"):
        assert heading in text
    assert f"**{report.counts['PASS']} PASS" in text
    data = json.loads(to_json(report))
    assert set(data) == {"manifest", "claims", "results"}
    assert len(data["results"]) == len(report.table) and len(data["claims"]) == len(CLAIMS)


def test_failed_report_is_not_passed():
    bad = AuditReport(_table([("a", "PASS"), ("b", "FAIL")]), pd.DataFrame(columns=CLAIM_COLUMNS), {})
    assert bad.counts == {"PASS": 1, "WARN": 0, "FAIL": 1} and not bad.passed


def test_rendering_tolerates_an_incomplete_manifest():
    bare = AuditReport(_table([("a", "PASS")]), pd.DataFrame(columns=CLAIM_COLUMNS), {})
    assert "commit `unknown`" in to_markdown(bare) and json.loads(to_json(bare))["manifest"] == {}


# ---------------------------------------------------------------- the script
def _script():
    spec = importlib.util.spec_from_file_location("run_audit", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_script_writes_both_reports_and_exits_0(report, tmp_path, monkeypatch):
    monkeypatch.setattr(report_module, "run_all", lambda quick, seed: report)
    assert _script().main(["--quick", "--out", str(tmp_path)]) == 0
    assert (tmp_path / "audit_report.md").read_text(encoding="utf-8").startswith("# OptionLab")
    assert json.loads((tmp_path / "audit_report.json").read_text(encoding="utf-8"))["manifest"]


def test_script_exits_1_on_any_failure(report, tmp_path, monkeypatch):
    failing = AuditReport(_table([("broken claim", "FAIL")]), report.claims, report.manifest)
    monkeypatch.setattr(report_module, "run_all", lambda quick, seed: failing)
    assert _script().main(["--out", str(tmp_path)]) == 1


def test_script_passes_full_mode_and_seed(report, tmp_path, monkeypatch):
    seen = {}

    def fake(quick, seed):
        seen.update(quick=quick, seed=seed)
        return report
    monkeypatch.setattr(report_module, "run_all", fake)
    _script().main(["--full", "--seed", "7", "--out", str(tmp_path)])
    assert seen == {"quick": False, "seed": 7}
