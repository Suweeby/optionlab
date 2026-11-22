"""Consistency audit: the documents say what the code does.

- README numbers (results table, exact value, "N checks") match docs/results/*.csv, and the
  deterministic numbers still match what the current code computes
- honesty scan: no prediction language in the app, README or docs (explicit negations allowed)
- every public function and class in src/optionlab has a docstring and type hints
"""
from __future__ import annotations

import ast
import re
from collections.abc import Iterable
from pathlib import Path

import pandas as pd

from optionlab.pricing.binomial import binomial_price
from optionlab.pricing.black_scholes import bs_price
from optionlab.validation import COLUMNS

PROJECT_ROOT = Path(__file__).resolve().parents[3]       # src/optionlab/audit -> repository root
PACKAGE_DIR = Path(__file__).resolve().parents[1]        # src/optionlab

BANNED_PHRASES = ("predicts", "will grow", "guaranteed", "expected to outperform", "forecast")
# Negations already in the text. "not a forecast" also covers "not a forecaster".
ALLOWED_PHRASES = ("not a forecast",)

_TABLE_ROW = re.compile(r"^\| ((?:Binomial tree|Monte Carlo)[^|]*?) \| ([-+0-9.]+) \| ([-+0-9.]+) "
                        r"\| ([^|]+?) \|", re.MULTILINE)


def _row(area: str, check: str, expected, observed, criterion: str, status: str) -> dict:
    return {"area": area, "check": check, "expected": expected, "observed": observed,
            "criterion": criterion, "status": status}


def _same_4dp(printed: float, exact: float) -> bool:
    """Does the README's 4-decimal number equal the exact value rounded to 4 decimals?"""
    return abs(printed - exact) <= 0.5e-4 + 1e-12


# ---------------------------------------------------------------- README vs results
def check_readme_results(root: Path | None = None,
                         validation_table: pd.DataFrame | None = None) -> list[dict]:
    """Compare the README's numbers with docs/results and with the current code."""
    root = Path(root) if root else PROJECT_ROOT
    area = "README vs results"
    try:
        readme = (root / "README.md").read_text(encoding="utf-8")
        pricing = pd.read_csv(root / "docs" / "results" / "pricing_comparison.csv").set_index("method")
        validation = pd.read_csv(root / "docs" / "results" / "validation.csv")
    except (OSError, ValueError) as exc:
        return [_row(area, "README and docs/results are readable", "files present", str(exc),
                     "files exist", "FAIL")]

    rows = []
    table_rows = list(_TABLE_ROW.finditer(readme))
    if not table_rows:
        rows.append(_row(area, "results table found in README", "a table", "none", "found", "FAIL"))
    for match in table_rows:
        method = match.group(1).strip()
        price, error, se_text = float(match.group(2)), float(match.group(3)), match.group(4).strip()
        if method not in pricing.index:
            rows.append(_row(area, f"results table: {method}", "a row in pricing_comparison.csv",
                             "missing", "method present", "FAIL"))
            continue
        exact = pricing.loc[method]
        ok = _same_4dp(price, exact["call"]) and _same_4dp(error, exact["call - BS"])
        if se_text != "n/a":
            ok = ok and _same_4dp(float(se_text), exact["call SE"])
        rows.append(_row(area, f"results table: {method}",
                         f"{exact['call']:.4f} / {exact['call - BS']:+.4f} / {exact['call SE']:.4f}",
                         f"{price:.4f} / {error:+.4f} / {se_text}",
                         "price / error / SE equal to 4 decimals", "PASS" if ok else "FAIL"))

    exact_value = re.search(r"exact value (\d+\.\d+)", readme)
    bs_now = bs_price(100, 100, 0.05, 0.2, 1.0)
    rows.append(_row(area, "README 'exact value' equals Black-Scholes", f"{bs_now:.4f}",
                     exact_value.group(1) if exact_value else "not found", "equal to 4 decimals",
                     "PASS" if exact_value and _same_4dp(float(exact_value.group(1)), bs_now) else "FAIL"))

    count = re.search(r"runs \*\*(\d+) checks, all passing\*\*", readme)
    n_now = len(validation_table) if validation_table is not None else len(validation)
    all_pass = (validation["status"] == "PASS").all() and (
        validation_table is None or (validation_table["status"] == "PASS").all())
    ok = bool(count) and int(count.group(1)) == len(validation) == n_now and all_pass
    rows.append(_row(area, "README check count matches validation.csv and the current suite",
                     f"{len(validation)} in csv, {n_now} now, all PASS",
                     f"README says {count.group(1) if count else 'nothing'}", "all equal, all PASS",
                     "PASS" if ok else "FAIL"))

    # code drift: deterministic numbers in the results must still be what the code computes
    worst = 0.0
    for method in pricing.index:
        n = re.fullmatch(r"Binomial tree, n = ([\d,]+)", method)
        if n:
            now = binomial_price(100, 100, 0.05, 0.2, 1.0, int(n.group(1).replace(",", "")))
            worst = max(worst, abs(now - pricing.loc[method, "call"]))
    rows.append(_row(area, "binomial results still match the current code", 0.0, worst,
                     "max abs difference < 1e-10", "PASS" if worst < 1e-10 else "FAIL"))
    return rows


# ---------------------------------------------------------------- honesty scan
def honesty_files(root: Path | None = None) -> list[Path]:
    """The user-facing text: app/**/*.py, README.md and docs/**/*.md (except generated audits)."""
    root = Path(root) if root else PROJECT_ROOT
    files = sorted((root / "app").rglob("*.py")) + [root / "README.md"]
    files += [p for p in sorted((root / "docs").rglob("*.md")) if "audit" not in p.parts]
    return [p for p in files if p.is_file()]


def scan_honesty(paths: Iterable[Path], banned: Iterable[str] = BANNED_PHRASES,
                 allowed: Iterable[str] = ALLOWED_PHRASES) -> list[tuple[str, int, str]]:
    """(file, line number, phrase) for each banned phrase not inside an allowed negation.

    Case-insensitive. Allowed phrases are blanked out first, so "not a forecast" passes but
    "a forecast" on its own does not.
    """
    findings = []
    for path in paths:
        for number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
            text = line.lower()
            for phrase in allowed:
                text = text.replace(phrase, " " * len(phrase))
            findings += [(str(path), number, phrase) for phrase in banned if phrase in text]
    return findings


# ---------------------------------------------------------------- docstrings and type hints
def find_undocumented(package_dir: Path | None = None) -> list[str]:
    """Public modules, classes, functions and methods missing a docstring or type hints."""
    package_dir = Path(package_dir) if package_dir else PACKAGE_DIR
    problems = []
    for path in sorted(package_dir.rglob("*.py")):
        name = path.relative_to(package_dir).as_posix()
        tree = ast.parse(path.read_text(encoding="utf-8"))
        if ast.get_docstring(tree) is None:
            problems.append(f"{name}: module has no docstring")
        for node in tree.body:
            if isinstance(node, ast.ClassDef) and not node.name.startswith("_"):
                if ast.get_docstring(node) is None:
                    problems.append(f"{name}:{node.name}: no docstring")
                functions = [(f"{node.name}.", f) for f in node.body if isinstance(f, ast.FunctionDef)]
            elif isinstance(node, ast.FunctionDef):
                functions = [("", node)]
            else:
                continue
            for owner, fn in functions:
                if fn.name.startswith("_"):
                    continue
                where = f"{name}:{owner}{fn.name}"
                if ast.get_docstring(fn) is None:
                    problems.append(f"{where}: no docstring")
                args = [a for a in fn.args.args + fn.args.kwonlyargs if a.arg not in ("self", "cls")]
                if fn.returns is None or any(a.annotation is None for a in args):
                    problems.append(f"{where}: incomplete type hints")
    return problems


# ---------------------------------------------------------------- all consistency checks
def _list_status(area: str, check: str, problems: list, criterion: str) -> dict:
    shown = "; ".join(str(p) for p in problems[:5]) + (" ..." if len(problems) > 5 else "")
    return _row(area, check, 0, f"{len(problems)}" + (f": {shown}" if problems else ""), criterion,
                "PASS" if not problems else "FAIL")


def run_consistency_audit(root: Path | None = None,
                          validation_table: pd.DataFrame | None = None) -> pd.DataFrame:
    """All consistency checks as one table with the validation columns."""
    root = Path(root) if root else PROJECT_ROOT
    files = honesty_files(root)
    honesty = [f"{Path(f).relative_to(root).as_posix()}:{n} '{p}'" for f, n, p in scan_honesty(files)]
    rows = check_readme_results(root, validation_table)
    rows.append(_list_status("Honesty", f"no prediction language in {len(files)} app/README/docs files",
                             honesty, "0 banned phrases outside allowed negations"))
    rows.append(_list_status("Documentation", "public functions have docstrings and type hints",
                             find_undocumented(root / "src" / "optionlab"), "0 missing"))
    return pd.DataFrame(rows, columns=COLUMNS)
