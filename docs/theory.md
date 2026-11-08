# Theory notes

My own notes for OptionLab: the maths behind each module, and a map between Ross's notation and
the standard one. Written to be explainable line by line.

## 1. Notation map

| Meaning | Standard (used in the code) | Ross (check against the book) |
|---|---|---|
| Spot price today | S | initial stock price symbol of the chapter |
| Strike | K | K |
| Time to expiry (years) | T | t |
| Continuously compounded rate | r | r |
| Volatility | sigma | sigma |
| Standard normal CDF | N(.) | Phi(.) |
| d1 | d1 = [ln(S/K) + (r + sigma^2/2) T] / (sigma sqrt(T)) | omega |
| d2 | d2 = d1 - sigma sqrt(T) | omega - sigma sqrt(t) |

Three traps:
- `(x)^+` means `max(x, 0)`. It is not an exponent.
- Continuous discounting needs the rate in the exponent: `e^(-rT)`, never `e^(-T)`.
- Ross's `K` and the standard `K` are the same strike; some sources write `X` for it.

## 2. Payoffs, moneyness, parity

- Call payoff `(S_T - K)^+`, put payoff `(K - S_T)^+`.
- A call is in the money when `S > K`; a put when `S < K`.
- Parity: `(S_T - K)^+ - (K - S_T)^+ = S_T - K` for every `S_T`. That is a forward payoff, worth
  `S - K e^(-rT)` today, so `C - P = S - K e^(-rT)` whatever the model.

## 3. No-arbitrage pricing and the binomial model

- An option can be replicated by a portfolio of the stock and a risk-free bond, so its price is the
  cost of that portfolio. The stock's real-world expected return never enters.
- CRR tree: `u = e^(sigma sqrt(dt))`, `d = 1/u`, `p = (e^(r dt) - d) / (u - d)`.
  `p` is a pricing weight chosen so that `p u + (1-p) d = e^(r dt)`, i.e. the stock grows at `r` on average.
- `d = 1/u` makes the tree recombine: `n + 1` terminal nodes instead of `2^n`.
- Price = discounted expectation of the payoff under `p`, computed by backward induction:
  `V = e^(-r dt) [p V_up + (1-p) V_down]`.
- The delta of the option is the number of shares in the replicating portfolio:
  `(V_up - V_down) / (S_up - S_down)`.

## 4. From the tree to Black-Scholes

`ln(S_T/S) = (2J - n) sigma sqrt(dt)` with `J ~ Binomial(n, p)`. For small `dt`,
`p ~ 1/2 + (r - sigma^2/2) sqrt(dt) / (2 sigma)`. Then

- mean `= n sigma sqrt(dt) (2p - 1) = (r - sigma^2/2) T`
- variance `= 4 sigma^2 dt n p (1-p) ~ sigma^2 T`

By the CLT, `ln(S_T/S)` becomes `Normal((r - sigma^2/2) T, sigma^2 T)`: the lognormal law of
Black-Scholes. The tree price error shrinks like `1/n` (with a zigzag between odd and even `n`).

## 5. Geometric Brownian motion

- `dS = mu S dt + sigma S dW`, with exact solution `S_t = S_0 exp((mu - sigma^2/2) t + sigma W_t)`.
- Exact step: `S(t+dt) = S(t) exp((mu - sigma^2/2) dt + sigma sqrt(dt) Z)`, `Z ~ N(0,1)`.
  There is no discretization error, so one step is enough for a European option.
- **Why `sigma^2/2`:** by the normal MGF, `E[e^(sigma W_T)] = e^(sigma^2 T/2)`. For
  `E[S_T] = S_0 e^(mu T)` the drift of `ln S` must be `mu - sigma^2/2`.
  Conceptually (Ito): `W` is so jagged that `(dW)^2 = dt`, so `d ln S` has an extra second-order
  term `-1/2 sigma^2 dt` that the ordinary chain rule misses. (No proof needed.)
- `S_T` is lognormal: mean `S_0 e^(mu T)`, median `S_0 e^((mu - sigma^2/2) T)` (lower; right skew).
- **Two measures, one engine.** Pricing mode uses drift `r` (risk-neutral: the discounted price is
  a martingale). Simulation mode uses drift `mu` (real world). Probabilities mean different things:
  with drift `r`, `P(S_T > K) = N(d2)` is a pricing weight, not a forecast.

## 6. Black-Scholes

- `C = S N(d1) - K e^(-rT) N(d2)`, `P = K e^(-rT) N(-d2) - S N(-d1)`.
- `N(d2)` is the risk-neutral probability that `S_T > K`. `K e^(-rT) N(d2)` is the present value
  of the strike expected to be paid. `S N(d1)` is the present value of the stock expected to be
  received; `N(d1)` is the same event weighted by stock value, and it is the call's delta.
- Limits: `T -> 0` gives intrinsic value; `sigma -> 0` gives `max(S - K e^(-rT), 0)`.

## 7. Greeks

Partial derivatives of the price (`phi` is the normal density):

| | Call | Put |
|---|---|---|
| delta | `N(d1)` | `N(d1) - 1` |
| gamma | `phi(d1) / (S sigma sqrt(T))` | same |
| vega | `S phi(d1) sqrt(T)` | same |
| theta | `-S phi(d1) sigma / (2 sqrt(T)) - r K e^(-rT) N(d2)` | `... + r K e^(-rT) N(-d2)` |
| rho | `T K e^(-rT) N(d2)` | `-T K e^(-rT) N(-d2)` |

- Checked by central finite differences: `f'(x) ~ [f(x+h) - f(x-h)] / (2h)`, error `~h^2`.
  Too small an `h` loses digits to round-off, so there is a sweet spot (relative step about `1e-4`).
- Parity relations: `delta_c - delta_p = 1`; equal gamma and vega.
- Raw units are per 1.00 of the input; trader units are per vol point, per day, per 1% rate.

## 8. Implied volatility

- Solve `BS(sigma) - market_price = 0`. Vega > 0 makes the price strictly increasing in sigma, so
  there is at most one solution, and one exists iff the price is inside the no-arbitrage bounds.
- Newton-Raphson: `sigma <- sigma - (BS(sigma) - price) / vega(sigma)`; quadratic convergence
  (about 3 steps at the money). Fails when vega is about 0 (deep out of the money, very short expiry),
  so a bisection fallback on `[1e-6, 5]` takes over (about 35-40 halvings).
- Where vega is tiny, sigma is poorly determined: use out-of-the-money options.

## 9. Monte Carlo

- Price `= e^(-rT) E*[payoff(S_T)]`, estimated by the average over `N` simulated terminal prices
  with drift `r`. Standard error `s / sqrt(N)`; 95% interval `price +- 1.96 SE`.
- The interval is a statement about the method: about 95% of such intervals contain the true price.
- Error `~ 1/sqrt(N)`: 100x more paths gives 10x less error.
- **Antithetic variates:** pair `Z` with `-Z`. For a monotone payoff the pair is negatively
  correlated, so `Var((X + X')/2) = Var(X) (1 + rho)/2` is smaller. The standard error must use
  the `N/2` pair averages (the independent units), not all `N` paths.
- Pathwise delta: differentiate inside the expectation. `dS_T/dS = S_T/S`, so
  `delta_call = e^(-rT) E[1{S_T > K} S_T / S]`, which equals `N(d1)`.

## 10. Volatility and drift from data

- Log returns `r_t = ln(P_t / P_{t-1})` of adjusted closes; they add over time and are normal under GBM.
- Historical volatility `= stdev(r_t, ddof=1) * sqrt(252)`. The `sqrt` appears because the variances
  of independent days add: annual variance `= 252 sigma_daily^2`.
- EWMA: `s2_t = lambda s2_{t-1} + (1 - lambda) r_t^2`, `lambda = 0.94`; effective memory about 17 days.
- There is no single correct volatility: it depends on the window and the estimator.
- Drift: `mu_hat = mean(r_t) * 252 + sigma_hat^2 / 2`, with standard error `sigma / sqrt(years)`.
  It depends only on the length of the history, so it is far noisier than volatility.

## 11. Questions I should be able to answer

1. *Why does the real-world expected return not appear in the option price?* Replication / no-arbitrage.
2. *Where does `sigma^2/2` come from?* Normal MGF (or the Ito correction).
3. *What are `N(d1)` and `N(d2)`?* Section 6.
4. *Why does Monte Carlo converge like `1/sqrt(N)` and the tree like `1/n`?* CLT versus discretization error; the tree costs `n^2`, Monte Carlo is linear in `N` and dimension-independent.
5. *Why Newton with a bisection fallback?* Fast near the root, but the tangent is flat where vega is about 0.
6. *Why is implied volatility not flat across strikes, and why does it matter?* The smile contradicts constant sigma.
7. *Why can't I trust the historical drift?* Standard error about 18 points for two years at 25% volatility.
8. *What are the model's main limitations?* See `docs/limitations.md`.