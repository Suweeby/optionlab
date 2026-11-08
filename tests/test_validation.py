import importlib.util
from pathlib import Path

import pandas as pd
import pytest

from optionlab.validation import COLUMNS, _row, run_validation, to_markdown


@pytest.fixture(scope="module")
def table():
    return run_validation()


def test_every_check_passes(table):
    failed = table[table["status"] != "PASS"]
    assert failed.empty, failed[["area", "check", "expected", "observed"]].to_string()


def test_structure_and_coverage(table):
    assert list(table.columns) == COLUMNS
    assert set(table["status"]) <= {"PASS", "FAIL"}
    assert {"Black-Scholes", "Binomial", "Monte Carlo", "GBM", "Greeks", "Implied volatility",
            "Volatility"} <= set(table["area"])
    assert len(table) >= 30


def test_a_failing_check_is_reported_as_fail():
    assert _row("x", "y", 1.0, 2.0, "equal", False)["status"] == "FAIL"
    assert _row("x", "y", 1.0, 1.0, "equal", True)["status"] == "PASS"


def test_markdown_rendering():
    text = to_markdown(pd.DataFrame({"a": [1.23456789, "text"], "b": [True, 2]}))
    lines = text.splitlines()
    assert lines[0] == "| a | b |" and lines[1] == "| --- | --- |"
    assert lines[2] == "| 1.23457 | True |" and len(lines) == 4
    assert to_markdown(pd.DataFrame({"n": [1_000_000]})).splitlines()[2] == "| 1000000 |"   # no 1e+06


def test_generate_results_script_writes_all_files(tmp_path):
    path = Path(__file__).resolve().parents[1] / "scripts" / "generate_results.py"
    spec = importlib.util.spec_from_file_location("generate_results", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    tables = module.main(out_dir=tmp_path, quick=True, images=False)
    for name in tables:
        assert (tmp_path / f"{name}.csv").exists()
    for figure in ("fig_binomial_convergence", "fig_monte_carlo_convergence", "fig_gbm_paths", "fig_greeks"):
        assert (tmp_path / f"{figure}.html").exists()
    text = (tmp_path / "results.md").read_text(encoding="utf-8")
    assert "checks pass" in text and "Greeks" in text
    assert (tables["validation"]["status"] == "PASS").all()