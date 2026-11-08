"""Monte Carlo convergence: price with confidence intervals vs N, and error vs N (log-log)."""
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from optionlab.convergence import monte_carlo_convergence
from optionlab.pricing.black_scholes import call_price
from optionlab.pricing.monte_carlo import mc_price

S, K, r, sigma, T = 100, 100, 0.05, 0.2, 1.0
bs = call_price(S, K, r, sigma, T)
counts = (100, 1_000, 10_000, 100_000, 1_000_000)

# 1) Replicated study: does the error really fall like 1/sqrt(N)?
plain = monte_carlo_convergence(S, K, r, sigma, T, path_counts=counts, replications=20)
anti = monte_carlo_convergence(S, K, r, sigma, T, path_counts=counts, replications=20, antithetic=True)
print(f"Black-Scholes: {bs:.4f}\n")
print("Plain Monte Carlo:\n", plain.to_string(index=False, float_format=lambda x: f"{x:.5f}"))
print("\nAntithetic Monte Carlo:\n", anti.to_string(index=False, float_format=lambda x: f"{x:.5f}"))

# 2) One run per N, shown with its 95% confidence interval
rng = np.random.default_rng(42)
runs = [mc_price(S, K, r, sigma, T, n_paths=n, seed=rng) for n in counts]

fig = make_subplots(rows=1, cols=2, subplot_titles=("MC price (95% CI) vs N", "Error vs N (log-log)"))
fig.add_trace(go.Scatter(
    x=list(counts), y=[x.price for x in runs], mode="markers", name="MC price",
    error_y=dict(type="data", symmetric=False,
                 array=[x.ci_high - x.price for x in runs],
                 arrayminus=[x.price - x.ci_low for x in runs])), row=1, col=1)
fig.add_hline(y=bs, line_dash="dash", annotation_text="Black-Scholes", row=1, col=1)

n = np.array(counts, dtype=float)
fig.add_trace(go.Scatter(x=plain["n_paths"], y=plain["rmse"], mode="lines+markers",
                         name="RMSE (plain)"), row=1, col=2)
fig.add_trace(go.Scatter(x=anti["n_paths"], y=anti["rmse"], mode="lines+markers",
                         name="RMSE (antithetic)"), row=1, col=2)
fig.add_trace(go.Scatter(x=n, y=plain["rmse"].iloc[0] * np.sqrt(n[0] / n), mode="lines",
                         name="reference c/sqrt(N)", line=dict(dash="dash")), row=1, col=2)
fig.update_xaxes(title_text="paths N", type="log", row=1, col=1)
fig.update_xaxes(title_text="paths N", type="log", row=1, col=2)
fig.update_yaxes(title_text="price", row=1, col=1)
fig.update_yaxes(title_text="root-mean-square error", type="log", row=1, col=2)
fig.show()
