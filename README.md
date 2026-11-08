# OptionLab

An options pricer and stock simulator in Python. It prices European options three independent
ways (Black-Scholes, a CRR binomial tree and Monte Carlo), computes Greeks and implied
volatility, simulates stock price paths under geometric Brownian motion, and compares the model
with real option quotes in a Streamlit app.

It is a **pricing and simulation tool, not a price predictor**, trading strategy or financial advice.

## What it does

- **Three pricers that check each other:** Black-Scholes closed form, binomial tree, Monte Carlo
  (with standard error, confidence interval and antithetic variates).
- **Greeks:** analytic delta, gamma, vega, theta, rho, verified against finite differences, plus a
  Monte Carlo (pathwise) delta.
- **Implied volatility:** Newton-Raphson using vega, with a bisection fallback.
- **One shared GBM engine, two modes:** the Monte Carlo pricer is the simulator plus a payoff.
  Pricing mode drifts at the risk-free rate `r`; simulation mode drifts at an assumed `mu`.
- **Real data:** prices and option chains from yfinance, cleaned (mid-prices, zero-bid and stale
  quotes removed), cached locally, with an offline fallback; historical and EWMA volatility.
- **Streamlit app:** Pricer, Simulator, Convergence and Market Data pages.

## Screenshots

![Pricer](docs/images/pricer.png)
![Simulator](docs/images/simulator.png)
![Convergence](docs/images/convergence.png)
![Market Data](docs/images/market_data.png)

## Quick start

Requires Python 3.11 or newer.

```bash
git clone https://github.com/<your-username>/optionlab.git
cd optionlab
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1      macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
pip install -e .

python -m pytest -q                      # run the tests
streamlit run app/streamlit_app.py       # launch the app
python scripts/generate_results.py       # regenerate docs/results/
```

Using the engine directly:

```python
from optionlab.pricing.black_scholes import call_price
from optionlab.pricing.binomial import binomial_price
from optionlab.pricing.monte_carlo import mc_price
from optionlab.greeks import greeks
from optionlab.implied_vol import implied_vol

S, K, r, sigma, T = 100, 100, 0.05, 0.2, 1.0
call_price(S, K, r, sigma, T)                                            # 10.4506
binomial_price(S, K, r, sigma, T, n=500)                                 # 10.4466
mc_price(S, K, r, sigma, T, n_paths=100_000, seed=42, antithetic=True)   # 10.4673 +- 0.0331
greeks(S, K, r, sigma, T).delta                                          # 0.6368
implied_vol(12.336, S, K, r, T)                                          # 0.25
```

## Results

All results are reproducible (fixed seeds). The validation suite runs **31 checks, all passing**,
covering the reference values, put-call parity, convergence rates, Monte Carlo calibration, GBM
moments, Greeks and implied volatility. Full tables and figures: [`docs/results/results.md`](docs/results/results.md).

European call, S = K = 100, r = 5%, sigma = 20%, T = 1 (exact value 10.4506):

| Method | Price | Error vs Black-Scholes | Standard error |
|---|---|---|---|
| Binomial tree, n = 100 | 10.4306 | -0.0200 | n/a |
| Binomial tree, n = 2,000 | 10.4496 | -0.0010 | n/a |
| Monte Carlo, N = 100,000 | 10.4205 | -0.0300 | 0.0468 |
| Monte Carlo, N = 100,000, antithetic | 10.4673 | +0.0167 | 0.0331 |
| Monte Carlo, N = 1,000,000, antithetic | 10.4565 | +0.0060 | 0.0104 |

- The binomial error shrinks like **1/n** (with a zigzag between odd and even n).
- The Monte Carlo error shrinks like **1/sqrt(N)**; 100x more paths gives 10x less error.
- Antithetic variates cut the standard error to about 0.71 of the plain estimate.

## Theory

Short derivations and a Ross-to-standard notation map are in [`docs/theory.md`](docs/theory.md):
no-arbitrage and the CRR tree, the limit to Black-Scholes, GBM and the `sigma^2/2` term, the
Greeks, implied volatility, Monte Carlo and antithetic variates, and volatility estimation.

## Project structure

```
src/optionlab/
  payoffs.py              call/put payoffs, moneyness
  pricing/                black_scholes.py, binomial.py, monte_carlo.py, compare.py
  simulation/             gbm.py (the shared generator), stats.py, scenario.py
  greeks.py               analytic Greeks, finite-difference checker, MC delta
  implied_vol.py          Newton-Raphson with bisection fallback
  volatility.py           historical, rolling and EWMA volatility; drift estimate
  data.py, market.py      yfinance prices and option chains, cleaning, caching
  convergence.py          convergence studies
  validation.py           the validation suite
app/                      Streamlit app (streamlit_app.py and pages/)
scripts/                  demos and generate_results.py
tests/                    pytest suite (over 450 tests)
docs/                     theory.md, limitations.md, results/
```

The engine never imports Streamlit; the app depends on the engine, not the other way round.

## Design notes

- **One GBM implementation.** The exact discretization means a European price needs one step, so
  the pricer and the simulator share `simulation/gbm.py`.
- **Reproducible randomness.** Every stochastic function takes a `seed` (int or numpy Generator).
- **Tests for random code** compare with theory within standard errors, and check interval
  coverage over many independent runs, rather than relying on lucky seeds.
- **Honest data handling.** Synthetic sample data is always labelled, and the Market Data page
  refuses to use it.

## Limitations

OptionLab assumes European exercise, no dividends and constant volatility, so real quotes (mostly
American options on dividend-paying stocks, with a volatility smile) will differ from its prices.
Expected returns cannot be estimated reliably from history, and yfinance data can be delayed or
incomplete. See [`docs/limitations.md`](docs/limitations.md) for the full list.

## Tech stack

Python 3.11+, numpy, scipy, pandas, yfinance, plotly, streamlit, pytest.

## Disclaimer

Educational software. Nothing here is financial advice or a prediction of future prices.