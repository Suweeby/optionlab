"""Run every audit and write docs/audit/audit_report.md and audit_report.json.

Usage:
    python scripts/run_audit.py            quick mode (default; what CI runs)
    python scripts/run_audit.py --full     more random cases, seeds and KS tests
    python scripts/run_audit.py --seed 7 --out some/folder

Exit code 1 if any check FAILs (warnings do not fail), so CI stops on a broken claim.
"""
import argparse
import sys
from pathlib import Path

from optionlab.audit import report as audit_report

DEFAULT_OUT = Path(__file__).resolve().parents[1] / "docs" / "audit"


def main(argv: list[str] | None = None) -> int:
    """Parse arguments, run the audit, write both report files; return the exit code."""
    parser = argparse.ArgumentParser(description="Run the OptionLab audits.")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--quick", action="store_true", help="quick mode (default)")
    mode.add_argument("--full", action="store_true", help="full mode: larger random checks")
    parser.add_argument("--seed", type=int, default=42, help="seed for the random checks")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="folder for the reports")
    args = parser.parse_args(argv)

    quick = not args.full
    print(f"Running the {'quick' if quick else 'full'} audit (seed {args.seed})...")
    report = audit_report.run_all(quick=quick, seed=args.seed)

    args.out.mkdir(parents=True, exist_ok=True)
    md_path, json_path = args.out / "audit_report.md", args.out / "audit_report.json"
    md_path.write_text(audit_report.to_markdown(report), encoding="utf-8")
    json_path.write_text(audit_report.to_json(report), encoding="utf-8")

    c = report.counts
    print(f"{c['PASS']} PASS, {c['WARN']} WARN, {c['FAIL']} FAIL "
          f"in {report.manifest['duration_seconds']} s")
    for row in report.table[report.table["status"] != "PASS"].itertuples(index=False):
        print(f"  {row.status}: [{row.area}] {row.check} -> {row.observed}")
    print(f"Wrote {md_path} and {json_path}")
    return 0 if report.passed else 1


if __name__ == "__main__":
    sys.exit(main())
