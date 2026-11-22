"""Consistency audit: passes on the real repository and catches planted inconsistencies."""
import shutil

import pandas as pd

from optionlab.audit.consistency_audit import (
    PROJECT_ROOT, check_readme_results, find_undocumented, honesty_files, run_consistency_audit,
    scan_honesty,
)
from optionlab.validation import COLUMNS


def _fake_repo(tmp_path, readme_edit=None):
    """A copy of the README and docs/results, optionally with one text replacement."""
    (tmp_path / "docs" / "results").mkdir(parents=True)
    for name in ("pricing_comparison.csv", "validation.csv"):
        shutil.copy(PROJECT_ROOT / "docs" / "results" / name, tmp_path / "docs" / "results" / name)
    readme = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
    if readme_edit:
        old, new = readme_edit
        assert old in readme
        readme = readme.replace(old, new)
    (tmp_path / "README.md").write_text(readme, encoding="utf-8")
    return tmp_path


def _statuses(rows):
    return {r["check"]: r["status"] for r in rows}


# ---------------------------------------------------------------- the real repository
def test_consistency_audit_passes_on_the_repository():
    table = run_consistency_audit()
    assert list(table.columns) == COLUMNS
    assert (table["status"] == "PASS").all(), table.loc[table["status"] != "PASS", ["check", "observed"]]


# ---------------------------------------------------------------- README vs results
def test_copied_readme_passes(tmp_path):
    rows = check_readme_results(_fake_repo(tmp_path))
    assert len(rows) == 8 and set(_statuses(rows).values()) == {"PASS"}


def test_a_changed_readme_number_is_caught(tmp_path):
    rows = check_readme_results(_fake_repo(tmp_path, ("| 10.4306 |", "| 10.4310 |")))
    assert _statuses(rows)["results table: Binomial tree, n = 100"] == "FAIL"


def test_a_wrong_check_count_is_caught(tmp_path):
    rows = check_readme_results(_fake_repo(tmp_path, ("**31 checks", "**32 checks")))
    assert _statuses(rows)["README check count matches validation.csv and the current suite"] == "FAIL"


def test_a_suite_that_grew_without_a_readme_update_is_caught(tmp_path):
    grown = pd.DataFrame({"status": ["PASS"] * 32})
    rows = check_readme_results(_fake_repo(tmp_path), validation_table=grown)
    assert _statuses(rows)["README check count matches validation.csv and the current suite"] == "FAIL"


def test_missing_files_fail_instead_of_crashing(tmp_path):
    rows = check_readme_results(tmp_path)
    assert len(rows) == 1 and rows[0]["status"] == "FAIL"


# ---------------------------------------------------------------- honesty scan
def test_honesty_scan_flags_prediction_language(tmp_path):
    page = tmp_path / "page.py"
    page.write_text("\n".join([
        'st.write("This stock will grow.")',                  # line 1: banned
        'st.write("This is not a forecast.")',                # line 2: allowed negation
        'st.write("It is not a forecaster either.")',         # line 3: allowed negation
        'st.write("Our forecast for next year")',             # line 4: banned
        'st.write("GUARANTEED returns")',                     # line 5: banned, any case
        'st.write("It predicts nothing")',                    # line 6: banned (even negated this way)
    ]), encoding="utf-8")
    found = [(line, phrase) for _, line, phrase in scan_honesty([page])]
    assert found == [(1, "will grow"), (4, "forecast"), (5, "guaranteed"), (6, "predicts")]


def test_honesty_scope_covers_app_readme_docs_but_not_generated_audits():
    files = {p.relative_to(PROJECT_ROOT).as_posix() for p in honesty_files()}
    assert "README.md" in files and "app/pages/1_Pricer.py" in files
    assert "docs/limitations.md" in files
    assert not any(f.startswith("docs/audit/") for f in files)


# ---------------------------------------------------------------- docstrings and type hints
def test_docstring_check_finds_each_kind_of_gap(tmp_path):
    (tmp_path / "bad.py").write_text(
        "def no_doc(x: int) -> int:\n    return x\n\n"
        "def no_hints(x):\n    '''Doc.'''\n    return x\n\n"
        "def _private(x):\n    return x\n\n"
        "class Thing:\n    def method(self, y) -> int:\n        '''Doc.'''\n        return y\n",
        encoding="utf-8")
    (tmp_path / "good.py").write_text('"""Module doc."""\n\ndef f(x: int) -> int:\n    """Doc."""\n'
                                      "    return x\n", encoding="utf-8")
    assert find_undocumented(tmp_path) == [
        "bad.py: module has no docstring",
        "bad.py:no_doc: no docstring",
        "bad.py:no_hints: incomplete type hints",
        "bad.py:Thing: no docstring",
        "bad.py:Thing.method: incomplete type hints",
    ]


def test_the_package_is_fully_documented():
    assert find_undocumented() == []
