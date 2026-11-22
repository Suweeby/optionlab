"""Audits: automatic checks that the data and the project's claims hold up.

data_audit          quality checks for one price series (PASS / WARN / FAIL)
model_audit         independent pricing method, textbook anchor, no-arbitrage invariants,
                    seed robustness, distribution (KS) test
consistency_audit   README vs results, honesty scan, docstrings and type hints
"""
