"""Print a convergence table and show two charts: price vs n, and |error| vs n."""
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from optionlab.convergence import binomial_convergence

S, K, r, sigma, T = 100, 100, 0.05, 0.2, 1.0

# 1) Table at a few chosen n
table = binomial_convergence(S, K, r, sigma, T)
print(table.to_string(index=False, float_format=lambda x: f"{x:.5f}"))

# 2) Dense study: every integer n, to reveal the zigzag
dense = binomial_convergence(S, K, r, sigma, T, n_values=range(5, 301))
bs = dense["black_scholes"].iloc[0]
wide = binomial_convergence(S, K, r, sigma, T, n_values=range(10, 1001))

fig = make_subplots(rows=1, cols=2, subplot_titles=("Binomial price vs n", "|error| vs n (log-log)"))
fig.add_trace(go.Scatter(x=dense["n"], y=dense["binomial"], mode="lines+markers",
                         name="binomial"), row=1, col=1)
fig.add_hline(y=bs, line_dash="dash", annotation_text="Black-Scholes", row=1, col=1)
fig.add_trace(go.Scatter(x=wide["n"], y=wide["abs_error"], mode="markers",
                         name="|error|", marker=dict(size=3)), row=1, col=2)
ref = wide["abs_error"].iloc[0] * wide["n"].iloc[0] / wide["n"]
fig.add_trace(go.Scatter(x=wide["n"], y=ref, mode="lines", name="reference c/n",
                         line=dict(dash="dash")), row=1, col=2)
fig.update_xaxes(title_text="n (tree steps)", row=1, col=1)
fig.update_xaxes(title_text="n (tree steps)", type="log", row=1, col=2)
fig.update_yaxes(title_text="price", row=1, col=1)
fig.update_yaxes(title_text="absolute error", type="log", row=1, col=2)
fig.show()
