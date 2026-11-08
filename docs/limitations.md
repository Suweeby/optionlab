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

## Not implemented
American options, dividend yield, exotic payoffs, GARCH volatility and out-of-sample comparison of
volatility estimators (see the stretch goals).