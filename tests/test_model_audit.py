"""Model audit: each check passes on the real code AND fails when fed something broken."""
import numpy as np
import pandas as pd
import pytest
from scipy.stats import kstest, norm

from optionlab.audit.model_audit import (
    QUAD_CASES, check_invariants, check_seed_robustness, check_textbook_anchor, gbm_ks_pvalues,
    ks_verdict, quad_price, run_model_audit,
)
from optionlab.pricing.black_scholes import bs_price
from optionlab.validation import COLUMNS


# ---------------------------------------------------------------- quadrature (4th method)
@pytest.mark.parametrize("case", QUAD_CASES)
@pytest.mark.parametrize("kind", ["call", "put"])
def test_quadrature_matches_black_scholes(case, kind):
    assert quad_price(*case, kind) == pytest.approx(bs_price(*case, kind), abs=1e-8)


def test_quadrature_anchor_value():
    assert quad_price(100, 100, 0.05, 0.2, 1.0) == pytest.approx(10.450583572185565, abs=1e-10)


def test_quadrature_far_out_of_the_money_is_worthless():
    assert quad_price(100, 1000, 0.05, 0.2, 0.1) == 0.0                 # strike 12+ SDs away
    assert quad_price(1000, 100, 0.05, 0.2, 0.1, "put") == 0.0


@pytest.mark.parametrize("bad", [dict(S=0), dict(K=-1), dict(sigma=0), dict(T=0),
                                 dict(option_type="straddle")])
def test_quadrature_rejects_bad_inputs(bad):
    args = dict(S=100, K=100, r=0.05, sigma=0.2, T=1.0, option_type="call") | bad
    with pytest.raises(ValueError):
        quad_price(**args)


# ---------------------------------------------------------------- textbook anchor
def test_hull_example_matches_the_book():
    rows = check_textbook_anchor()
    assert [r["expected"] for r in rows] == [4.76, 0.81]
    assert all(r["status"] == "PASS" for r in rows)


# ---------------------------------------------------------------- invariants
def test_invariants_hold_for_black_scholes():
    rows = check_invariants(n_cases=100, seed=7)
    assert len(rows) == 9
    assert all(r["status"] == "PASS" and r["observed"] == 0 for r in rows)


def test_invariants_catch_a_broken_pricer():
    def broken(S, K, r, sigma, T, option_type="call"):          # puts 5 cents too expensive
        price = bs_price(S, K, r, sigma, T, option_type)
        return price + 0.05 if option_type == "put" else price

    rows = {r["check"].split(" (")[0]: r for r in check_invariants(n_cases=50, seed=7, pricer=broken)}
    assert rows["put-call parity"]["status"] == "FAIL"
    assert rows["put-call parity"]["observed"] == 50
    assert rows["call is convex in K"]["status"] == "PASS"      # calls were left alone


# ---------------------------------------------------------------- seed robustness
def _fake_runner(failing_seeds):
    def runner(seed):
        status = "FAIL" if seed in failing_seeds else "PASS"
        return pd.DataFrame({"check": ["steady", "statistical"], "status": ["PASS", status]})
    return runner


@pytest.mark.parametrize("failing, expected", [(set(), "PASS"),         # never fails
                                               ({3}, "WARN"),          # 1 in 10: allowed now and then
                                               ({1, 4, 8}, "FAIL")])   # 3 in 10: not chance
def test_seed_robustness_verdicts(failing, expected):
    row = check_seed_robustness(10, runner=_fake_runner(failing))
    assert row["status"] == expected
    if failing:
        assert "'statistical'" in row["observed"]


# ---------------------------------------------------------------- KS distribution test
@pytest.mark.parametrize("p_values, rejections, status", [
    ([0.5] * 20, 0, "PASS"),
    ([0.01] + [0.5] * 19, 1, "PASS"),          # one rejection in 20 is ordinary chance
    ([0.01] * 6 + [0.5] * 14, 6, "FAIL"),      # 6 in 20: chance about 0.0003
])
def test_ks_verdict(p_values, rejections, status):
    found, chance, verdict = ks_verdict(p_values)
    assert found == rejections and verdict == status
    assert 0.0 <= chance <= 1.0


def test_gbm_returns_pass_the_ks_check():
    p_values = gbm_ks_pvalues(n_tests=20, seed=42)
    assert p_values.shape == (20,) and ((0 <= p_values) & (p_values <= 1)).all()
    assert ks_verdict(p_values)[2] == "PASS"


def test_ks_check_detects_fat_tails():
    # Student-t returns (3 degrees of freedom) scaled to the same variance are NOT normal
    rng = np.random.default_rng(0)
    scale = 0.0126
    p_values = [kstest(scale * rng.standard_t(3, 1000) / np.sqrt(3), norm(0, scale).cdf).pvalue
                for _ in range(20)]
    assert ks_verdict(p_values)[2] == "FAIL"


# ---------------------------------------------------------------- the whole model audit
def test_quick_model_audit_passes_on_the_real_code():
    table = run_model_audit(quick=True)
    assert list(table.columns) == COLUMNS
    assert len(table) == 15
    assert (table["status"] == "PASS").all(), table.loc[table["status"] != "PASS", ["check", "observed"]]
