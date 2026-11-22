"""Audit page: run the project's audits and see which of its claims hold."""
import streamlit as st

from optionlab.audit import report as audit_report

st.set_page_config(page_title="Audit · OptionLab", page_icon="📈", layout="wide")
REPORT_KEY = "audit_report"

st.title("Audit")
st.caption("Automatic checks that the code does what this project says it does. "
           "The same audit runs on every commit in continuous integration.")

with st.expander("What the audits prove, and what they do not"):
    st.markdown(
        """
**They show that the code meets its own stated claims:**
- the formulas reproduce reference values (including a textbook example);
- three pricing methods, plus an independent fourth one (numerical integration), agree;
- no-arbitrage rules hold over hundreds of random inputs;
- random results are calibrated (convergence rates, interval coverage, distribution tests)
  and do not depend on one lucky seed;
- the numbers in the README match the generated results, and the wording stays honest.

**They do not show:**
- that Black-Scholes describes real markets. It assumes constant volatility, European
  exercise and no dividends; the Market Data page shows where real quotes differ;
- that downloaded data is correct (the Data page audits its internal consistency only);
- that the list of checks is complete. A clean audit means no known problem was found.
"""
    )

if st.button("Run the quick audit", type="primary", key="audit_run",
             help="Validation suite, model audit, consistency audit and claims: about 10-20 s."):
    with st.spinner("Running the validation suite, model, consistency and claims checks..."):
        try:
            st.session_state[REPORT_KEY] = audit_report.run_all(quick=True)
        except Exception as exc:          # show the problem on the page instead of crashing it
            st.error(f"The audit could not run: {type(exc).__name__}: {exc}")

report = st.session_state.get(REPORT_KEY)
if report is None:
    st.info("Click **Run the quick audit** to start. The full audit (more random cases, seeds and "
            "distribution tests) runs from the command line: `python scripts/run_audit.py --full`.")
    st.stop()

# ---------------------------------------------------------------- summary
counts, manifest = report.counts, report.manifest
c1, c2, c3 = st.columns(3)
c1.metric("PASS", counts["PASS"])
c2.metric("WARN", counts["WARN"])
c3.metric("FAIL", counts["FAIL"])
if counts["FAIL"]:
    st.error(f"{counts['FAIL']} check(s) failed. The rows marked FAIL below show what was expected "
             "and what was observed.")
elif counts["WARN"]:
    st.warning(f"No failures, {counts['WARN']} warning(s): look at the rows marked WARN.")
else:
    st.success(f"All {sum(counts.values())} checks and claims passed.")
commit = str(manifest.get("git_commit", "unknown"))[:12]
st.caption(f"Run at {manifest.get('generated_utc', '?')} (UTC) · {manifest.get('mode', '?')} mode · "
           f"commit {commit} · {manifest.get('duration_seconds', '?')} s")

# ---------------------------------------------------------------- claims and checks
st.subheader("Claims")
st.caption("Each claim the project makes, where it makes it, and the evidence from its check.")
st.dataframe(report.claims, hide_index=True, width="stretch")

st.subheader("All checks")
areas = ["All areas"] + sorted(report.table["area"].unique())
area = st.selectbox("Area", areas, key="audit_area")
shown = report.table if area == "All areas" else report.table[report.table["area"] == area]
st.dataframe(shown.astype(str), hide_index=True, width="stretch")

with st.expander("Reproducibility manifest"):
    st.json(manifest)

d1, d2 = st.columns(2)
d1.download_button("Download report (Markdown)", data=audit_report.to_markdown(report),
                   file_name="audit_report.md", mime="text/markdown", key="audit_md")
d2.download_button("Download report (JSON)", data=audit_report.to_json(report),
                   file_name="audit_report.json", mime="application/json", key="audit_json")
