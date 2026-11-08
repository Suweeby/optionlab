"""Fan chart of GBM paths + terminal histogram with the theoretical lognormal density."""
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import lognorm

from optionlab.simulation.gbm import simulate_gbm, time_grid
from optionlab.simulation.stats import (
    gbm_log_return_theory, log_return_stats, terminal_summary, value_at_risk,
)

S0, K, mu, sigma, T = 100.0, 100.0, 0.08, 0.2, 1.0   # real-world mode: drift = mu
n_steps, n_paths, n_draw = 252, 20_000, 100

paths = simulate_gbm(S0, mu, sigma, T, n_steps, n_paths, seed=42)
ST = paths[:, -1]

print("Summary:", {k: round(v, 3) for k, v in terminal_summary(ST, K).items()})
print("VaR 95%:", round(value_at_risk(ST, S0, 0.95), 3))
print("ln(S_T/S0) sample:", {k: round(v, 5) for k, v in log_return_stats(ST, S0).items()})
print("ln(S_T/S0) theory:", {k: round(v, 5) for k, v in gbm_log_return_theory(mu, sigma, T).items()})

fig = make_subplots(rows=1, cols=2, column_widths=[0.6, 0.4],
                    subplot_titles=(f"{n_draw} of {n_paths:,} simulated paths", "Terminal price S_T"))
t = time_grid(T, n_steps)
for path in paths[:n_draw]:
    fig.add_trace(go.Scatter(x=t, y=path, mode="lines", line=dict(width=1),
                             opacity=0.4, showlegend=False), row=1, col=1)
fig.add_hline(y=K, line_dash="dash", annotation_text="strike K", row=1, col=1)

fig.add_trace(go.Histogram(y=ST, histnorm="probability density", nbinsy=80,
                           name="simulated"), row=1, col=2)
grid = np.linspace(ST.min(), ST.max(), 300)
pdf = lognorm.pdf(grid, s=sigma * np.sqrt(T), scale=S0 * np.exp((mu - 0.5 * sigma**2) * T))
fig.add_trace(go.Scatter(x=pdf, y=grid, mode="lines", name="theory (lognormal)"), row=1, col=2)
fig.add_hline(y=K, line_dash="dash", row=1, col=2)
fig.update_xaxes(title_text="time (years)", row=1, col=1)
fig.update_yaxes(title_text="price", row=1, col=1)
fig.update_xaxes(title_text="density", row=1, col=2)
fig.update_layout(showlegend=True)
fig.show()
