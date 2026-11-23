# OptionLab audit report

Generated 2026-11-22T13:04:43+00:00 · quick mode · commit `5c5859fc0f10` (with uncommitted changes)

**75 PASS · 0 WARN · 0 FAIL** across 75 checks.

These checks show that the code does what the project states: reference values, agreement between independent methods, no-arbitrage rules, calibrated random results, documents that match the results, and honest wording. They do not show that Black-Scholes describes real markets, that the downloaded data is correct, or that the list of checks is complete.

## Reproducibility manifest

| field | value |
| --- | --- |
| generated_utc | 2026-11-22T13:04:43+00:00 |
| mode | quick |
| git_commit | 5c5859fc0f10701b9654ecfaecf0b155a1f79c44 |
| uncommitted_changes | True |
| python | 3.11.1 |
| platform | Windows-10-10.0.26300-SP0 |
| packages | {"optionlab": "0.1.0", "numpy": "2.4.6", "scipy": "1.17.1", "pandas": "3.1.1"} |
| seeds | {"validation": 42, "model_audit": 42, "seed_robustness": "0..2"} |
| sizes | {"invariant_cases": 200, "robustness_seeds": 3, "ks_tests": 20} |
| checks | 75 |
| counts | {"PASS": 75, "WARN": 0, "FAIL": 0} |
| duration_seconds | 9.9 |

## Claims

| id | statement | location | status | evidence |
| --- | --- | --- | --- | --- |
| bs-anchor | Black-Scholes gives 10.4506 for the anchor call (S=K=100, r=5%, sigma=20%, T=1) | README: Using the engine, Results | PASS | 1 of 1 validation checks pass |
| pricers-agree | Black-Scholes, the binomial tree and Monte Carlo agree within their errors | README: What it does; Pricer page | PASS | 4 of 4 validation checks pass |
| independent-method | A fourth method (numerical integration) reproduces Black-Scholes | Audit page; docs/theory.md | PASS | 2 of 2 model checks pass |
| textbook-anchor | Prices match Hull's worked example (call 4.76, put 0.81) | Audit page | PASS | 2 of 2 model checks pass |
| binomial-rate | The binomial error falls like 1/n | README: Results; Convergence page | PASS | 1 of 1 validation checks pass |
| mc-rate | The Monte Carlo error falls like 1/sqrt(N) | README: Results; Convergence page | PASS | 1 of 1 validation checks pass |
| mc-coverage | 95% intervals cover the true price about 95% of the time | README: Design notes; Pricer page | PASS | 1 of 1 validation checks pass |
| antithetic | Antithetic variates cut the standard error to about 0.71 | README: Results | PASS | standard-error ratio 0.707 (claim: about 0.71) |
| greeks-fd | Analytic Greeks match finite differences | README: What it does | PASS | 1 of 1 validation checks pass |
| iv-recovers | The implied-volatility solver recovers sigma | README: What it does | PASS | 2 of 2 validation checks pass; 1 of 1 model checks pass |
| no-arbitrage | Prices respect no-arbitrage rules (parity, bounds, monotonicity, convexity) | docs/theory.md | PASS | 9 of 9 model checks pass |
| gbm-theory | Simulated GBM matches its theoretical moments and distribution | README: What it does; Simulator page | PASS | 5 of 5 validation checks pass; 1 of 1 model checks pass |
| seed-robust | Results do not depend on one lucky seed | README: Design notes | PASS | 1 of 1 model checks pass |
| docs-match-results | README numbers match docs/results and the current code | README: Results | PASS | 8 of 8 consistency checks pass |
| honest-language | The app and docs make no price predictions | README: top; every page | PASS | 1 of 1 consistency checks pass |
| documented | Public functions have docstrings and type hints | README: Design notes | PASS | 1 of 1 consistency checks pass |
| synthetic-not-compared | Synthetic (and uploaded) data is never used for the model-vs-market comparison | README: Design notes; Market Data page | PASS | guard refuses sample and upload: True; allows downloads: True; Market Data page calls it: True |
| synthetic-labelled | Synthetic fallback data is always labelled and never cached | README: Design notes | PASS | source = sample; labelled SYNTHETIC: True; files written: 0 |
| tests-offline | The tests need no network | README: Design notes | PASS | autouse network guard in tests/conftest.py: True; tests/test_network_guard.py present: True |

## All checks

| area | check | expected | observed | criterion | status |
| --- | --- | --- | --- | --- | --- |
| Black-Scholes | call, S=K=100, r=5%, sigma=20%, T=1 | 10.4506 | 10.4506 | abs error < 1e-4 | PASS |
| Black-Scholes | put, same inputs | 5.5735 | 5.57353 | abs error < 1e-4 | PASS |
| Black-Scholes | put-call parity C - P = S - K e^(-rT) | 4.87706 | 4.87706 | abs error < 1e-10 | PASS |
| Black-Scholes | T -> 0: call tends to intrinsic value | 10 | 10 | abs error < 1e-4 | PASS |
| Black-Scholes | sigma -> 0: call tends to discounted forward payoff | 14.3894 | 14.3894 | abs error < 1e-6 | PASS |
| Black-Scholes | deep in the money: call ~ S - K e^(-rT) | 204.877 | 204.877 | abs error < 1e-6 | PASS |
| Black-Scholes | deep out of the money: call ~ 0 | 0 | 3.4165e-15 | abs error < 1e-6 | PASS |
| Black-Scholes | negative S, K, sigma or T raise ValueError | all raise | all raise | every case raises | PASS |
| Binomial | put-call parity inside the tree (n=500) | 4.87706 | 4.87706 | abs error < 1e-9 | PASS |
| Binomial | converges to Black-Scholes (n=2000) | 0 | -0.000999797 | abs error < 3/n | PASS |
| Binomial | error shrinks like 1/n (10x steps -> 10x smaller) | 10 | 9.98861 | ratio within 25% of 10 | PASS |
| Monte Carlo | price within 3 standard errors of Black-Scholes (N=10,000) | 10.4506 | 10.3452 | \|z\| < 3 (z = -0.71) | PASS |
| Monte Carlo | price within 3 standard errors of Black-Scholes (N=100,000) | 10.4506 | 10.4269 | \|z\| < 3 (z = -0.51) | PASS |
| Monte Carlo | price within 3 standard errors of Black-Scholes (N=1,000,000) | 10.4506 | 10.4574 | \|z\| < 3 (z = +0.46) | PASS |
| Monte Carlo | 95% intervals contain the true price (300 runs) | 0.95 | 0.96 | coverage between 0.92 and 0.98 | PASS |
| Monte Carlo | error shrinks like 1/sqrt(N) (100x paths -> 10x smaller) | 10 | 9.98452 | ratio within 35% of 10 | PASS |
| Monte Carlo | antithetic variates reduce the standard error | ratio < 1 | 0.706779 | ratio < 0.9 | PASS |
| Monte Carlo | put-call parity holds within noise (N=200,000) | 4.87706 | 4.88861 | within 4 combined standard errors | PASS |
| GBM | mean of ln(S_T/S_0) = (mu - sigma^2/2) T | 0.06 | 0.0599547 | within 4 standard errors | PASS |
| GBM | variance of ln(S_T/S_0) = sigma^2 T | 0.04 | 0.0398214 | relative error < 2% | PASS |
| GBM | E[S_T] = S_0 e^(mu T) | 108.329 | 108.314 | within 4 standard errors | PASS |
| GBM | risk-neutral: discounted E[S_T] = S_0 | 100 | 100.06 | within 4 standard errors | PASS |
| GBM | risk-neutral P(S_T > K) = N(d2) | 0.462574 | 0.463195 | within 4 standard errors | PASS |
| Volatility | estimator recovers sigma = 30% from 40 years of daily data | 0.3 | 0.297938 | abs error < 0.01 | PASS |
| Greeks | analytic Greeks match finite differences (5 Greeks, call and put) | 0 | 2.36799e-08 | max relative difference < 1e-4 | PASS |
| Greeks | delta(call) - delta(put) = 1 | 1 | 1 | abs error < 1e-12 | PASS |
| Implied volatility | round trip: price a call at sigma=0.25, solve back | 0.25 | 0.25 | abs error < 1e-8 | PASS |
| Implied volatility | round trip: price a put at sigma=0.25, solve back | 0.25 | 0.25 | abs error < 1e-8 | PASS |
| Implied volatility | Newton-Raphson converges in a few steps at the money | <= 6 | 3 | iterations <= 6 | PASS |
| Implied volatility | bisection fallback solves the case where Newton cannot start | 0.8 | 0.8 | abs error < 1e-7, method = bisection | PASS |
| Implied volatility | price below the no-arbitrage minimum raises ValueError | raises | raises | raises | PASS |
| Independent method | quadrature matches Black-Scholes, S=K=100, r=5%, sigma=20%, T=1 | 10.4506 | 10.4506 | abs error < 1e-8 | PASS |
| Independent method | quadrature matches Black-Scholes (10 options) | 0 | 1.03029e-13 | max abs error < 1e-8 | PASS |
| Textbook anchor | Hull example call: S=42, K=40, r=10%, sigma=20%, T=0.5 | 4.76 | 4.75942 | equal to 2 decimals | PASS |
| Textbook anchor | Hull example put: S=42, K=40, r=10%, sigma=20%, T=0.5 | 0.81 | 0.808599 | equal to 2 decimals | PASS |
| No-arbitrage | put-call parity (200 random options, seed 42) | 0 | 0 | 0 violations | PASS |
| No-arbitrage | price bounds (200 random options, seed 42) | 0 | 0 | 0 violations | PASS |
| No-arbitrage | call increases with sigma (200 random options, seed 42) | 0 | 0 | 0 violations | PASS |
| No-arbitrage | call increases with T (200 random options, seed 42) | 0 | 0 | 0 violations | PASS |
| No-arbitrage | call decreases in K, slope in [-e^(-rT), 0] (200 random options, seed 42) | 0 | 0 | 0 violations | PASS |
| No-arbitrage | call is convex in K (200 random options, seed 42) | 0 | 0 | 0 violations | PASS |
| No-arbitrage | call delta in (0, 1) (200 random options, seed 42) | 0 | 0 | 0 violations | PASS |
| No-arbitrage | gamma > 0 and vega > 0 (200 random options, seed 42) | 0 | 0 | 0 violations | PASS |
| No-arbitrage | implied volatility recovers sigma (197 random options, seed 42) | 0 | 0 | 0 violations | PASS |
| Seed robustness | validation suite rerun with 3 seeds (0..2) | every check passes on every seed | 31 of 31 checks pass on every seed | PASS: 100%; WARN: each check >= 90% | PASS |
| Distribution | GBM daily log returns are normal (KS test, 20 paths x 1000 days) | about 1.0 rejections at the 5% level by chance | 2 rejections; median p = 0.22; chance of >= 2: 0.264 | FAIL only if that chance < 0.001 | PASS |
| README vs results | results table: Binomial tree, n = 100 | 10.4306 / -0.0200 / 0.0000 | 10.4306 / -0.0200 / n/a | price / error / SE equal to 4 decimals | PASS |
| README vs results | results table: Binomial tree, n = 2,000 | 10.4496 / -0.0010 / 0.0000 | 10.4496 / -0.0010 / n/a | price / error / SE equal to 4 decimals | PASS |
| README vs results | results table: Monte Carlo, N = 100,000 | 10.4205 / -0.0300 / 0.0468 | 10.4205 / -0.0300 / 0.0468 | price / error / SE equal to 4 decimals | PASS |
| README vs results | results table: Monte Carlo, N = 100,000, antithetic | 10.4673 / +0.0167 / 0.0331 | 10.4673 / +0.0167 / 0.0331 | price / error / SE equal to 4 decimals | PASS |
| README vs results | results table: Monte Carlo, N = 1,000,000, antithetic | 10.4565 / +0.0060 / 0.0104 | 10.4565 / +0.0060 / 0.0104 | price / error / SE equal to 4 decimals | PASS |
| README vs results | README 'exact value' equals Black-Scholes | 10.4506 | 10.4506 | equal to 4 decimals | PASS |
| README vs results | README check count matches validation.csv and the current suite | 31 in csv, 31 now, all PASS | README says 31 | all equal, all PASS | PASS |
| README vs results | binomial results still match the current code | 0 | 0 | max abs difference < 1e-10 | PASS |
| Honesty | no prediction language in 12 app/README/docs files | 0 | 0 | 0 banned phrases outside allowed negations | PASS |
| Documentation | public functions have docstrings and type hints | 0 | 0 | 0 missing | PASS |
| Claims | [bs-anchor] Black-Scholes gives 10.4506 for the anchor call (S=K=100, r=5%, sigma=20%, T=1) | its check passes | 1 of 1 validation checks pass | stated in: README: Using the engine, Results | PASS |
| Claims | [pricers-agree] Black-Scholes, the binomial tree and Monte Carlo agree within their errors | its check passes | 4 of 4 validation checks pass | stated in: README: What it does; Pricer page | PASS |
| Claims | [independent-method] A fourth method (numerical integration) reproduces Black-Scholes | its check passes | 2 of 2 model checks pass | stated in: Audit page; docs/theory.md | PASS |
| Claims | [textbook-anchor] Prices match Hull's worked example (call 4.76, put 0.81) | its check passes | 2 of 2 model checks pass | stated in: Audit page | PASS |
| Claims | [binomial-rate] The binomial error falls like 1/n | its check passes | 1 of 1 validation checks pass | stated in: README: Results; Convergence page | PASS |
| Claims | [mc-rate] The Monte Carlo error falls like 1/sqrt(N) | its check passes | 1 of 1 validation checks pass | stated in: README: Results; Convergence page | PASS |
| Claims | [mc-coverage] 95% intervals cover the true price about 95% of the time | its check passes | 1 of 1 validation checks pass | stated in: README: Design notes; Pricer page | PASS |
| Claims | [antithetic] Antithetic variates cut the standard error to about 0.71 | its check passes | standard-error ratio 0.707 (claim: about 0.71) | stated in: README: Results | PASS |
| Claims | [greeks-fd] Analytic Greeks match finite differences | its check passes | 1 of 1 validation checks pass | stated in: README: What it does | PASS |
| Claims | [iv-recovers] The implied-volatility solver recovers sigma | its check passes | 2 of 2 validation checks pass; 1 of 1 model checks pass | stated in: README: What it does | PASS |
| Claims | [no-arbitrage] Prices respect no-arbitrage rules (parity, bounds, monotonicity, convexity) | its check passes | 9 of 9 model checks pass | stated in: docs/theory.md | PASS |
| Claims | [gbm-theory] Simulated GBM matches its theoretical moments and distribution | its check passes | 5 of 5 validation checks pass; 1 of 1 model checks pass | stated in: README: What it does; Simulator page | PASS |
| Claims | [seed-robust] Results do not depend on one lucky seed | its check passes | 1 of 1 model checks pass | stated in: README: Design notes | PASS |
| Claims | [docs-match-results] README numbers match docs/results and the current code | its check passes | 8 of 8 consistency checks pass | stated in: README: Results | PASS |
| Claims | [honest-language] The app and docs make no price predictions | its check passes | 1 of 1 consistency checks pass | stated in: README: top; every page | PASS |
| Claims | [documented] Public functions have docstrings and type hints | its check passes | 1 of 1 consistency checks pass | stated in: README: Design notes | PASS |
| Claims | [synthetic-not-compared] Synthetic (and uploaded) data is never used for the model-vs-market comparison | its check passes | guard refuses sample and upload: True; allows downloads: True; Market Data page calls it: True | stated in: README: Design notes; Market Data page | PASS |
| Claims | [synthetic-labelled] Synthetic fallback data is always labelled and never cached | its check passes | source = sample; labelled SYNTHETIC: True; files written: 0 | stated in: README: Design notes | PASS |
| Claims | [tests-offline] The tests need no network | its check passes | autouse network guard in tests/conftest.py: True; tests/test_network_guard.py present: True | stated in: README: Design notes | PASS |
