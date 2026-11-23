# Limitations

OptionLab is an educational pricing and simulation tool. It is not a forecaster, a trading
system or financial advice. Know these limits before reading anything into its numbers.

## Model assumptions
- **European exercise only.** Most listed stock options are American-style and can be exercised
  early, so real quotes include an early-exercise premium that Black-Scholes does not capture.
- **No dividends.** Many stocks pay dividends, which lower call values and raise put values.
- **Constant volatility and interest rate.** Real volatility changes over time and across strikes
  (the "smile" or "skew"), so no single sigma fits a whole option chain.
- **Geometric Brownian motion.** Prices are lognormal, with no jumps and no fat tails beyond what
  the lognormal allows. Real markets have crashes and gaps.

## What the numbers mean
- **Pricing mode (risk-neutral, drift = r)** gives pricing weights. Its probabilities, such as
  N(d2), are not predictions of what will happen.
- **Simulation mode (real-world, drift = mu)** shows possible futures for an *assumed* mu.
  Expected return is extremely hard to estimate from data: its standard error is about
  sigma / sqrt(years of data), roughly 18 percentage points for two years at 25% volatility.
- **Historical volatility has no single correct value.** It depends on the lookback window and
  on the estimator (rolling window versus EWMA).
- **Implied volatility is poorly determined** where vega is near zero (deep in-the-money or very
  short-dated options). The app defaults to out-of-the-money quotes for this reason.

## Numerical limits
- The binomial tree converges slowly (error about 1/n) and costs about n^2 operations.
- Monte Carlo has random error that shrinks only like 1/sqrt(N). Reported intervals miss the true
  value about 1 time in 20 by design.
- Greeks are undefined at exactly sigma = 0 or T = 0, and the tree rejects those inputs.

## Data limits
- **yfinance data can be delayed, incomplete or rate-limited.** The spot price is the last close,
  which may not match the moment the option quotes were taken.
- Quotes are cleaned (zero bids, crossed, wide-spread and stale quotes are dropped) and the
  mid-price is used. A mid-price is not a tradable price.
- Time to expiry uses calendar days / 365, while volatility is annualized with 252 trading days;
  this common convention is an approximation.
- The risk-free rate is a user input; the app does not look it up.
- Offline, the app falls back to cached data. Only the price loader can fall back to clearly
  labelled SYNTHETIC sample data, and the Market Data page refuses to use it.
- **Adjusted closes change after the fact.** The data source re-adjusts past prices after each
  dividend or split, so a refresh downloads the whole cached span again instead of appending.
- **Uploaded files are not verified.** A user CSV is cleaned and checked like a download, but its
  origin cannot be checked, so it is labelled user-supplied and never used for the model-vs-market
  comparison.
- **The data-quality audit checks internal consistency only** (dates, positive prices, history
  length, gaps, large one-day moves, staleness). It cannot tell whether the source's prices are
  correct. Exchange holidays count as missing days, and real crashes trigger the large-move warning
  just as bad prices do.
- **Short histories limit what can be estimated:** at least 21 daily returns for a volatility
  estimate and 252 for a drift estimate. Younger stocks get volatility but no historical drift.

## Audit limits
- The audits show that the code meets the project's own stated claims. They do not show that
  Black-Scholes describes real markets.
- A clean audit means no known problem was found, not that none exists: the checks cover the
  claims that were written down.
- Statistical checks can fail by chance. The audit uses tolerant criteria (about 4 standard
  errors, coverage bands, repeated KS tests judged by their binomial count) so false alarms are
  rare but possible; a single WARN in the seed-robustness check is allowed for this reason.
- The quick mode used by tests and CI runs fewer random cases than `--full`.
- Some checks are static: for example, "tests need no network" is confirmed by finding the guard
  in the test configuration; the test suite itself shows the guard works.

## Not implemented
American options, dividend yield, exotic payoffs, GARCH volatility and out-of-sample comparison of
volatility estimators (see the stretch goals).