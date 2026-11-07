"""Compare analytic, finite-difference and Monte Carlo Greeks, and plot Greeks against spot."""
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from optionlab.greeks import fd_greeks, greeks, mc_delta

S, K, r, sigma, T = 100, 100, 0.05, 0.2, 1.0

analytic = greeks(S, K, r, sigma, T)
numeric = fd_greeks(S, K, r, sigma, T)
print(f"{'':8}{'analytic':>14}{'finite diff':>16}")
for name in ("delta", "gamma", "vega", "theta", "rho"):
    print(f"{name:8}{getattr(analytic, name):>14.6f}{getattr(numeric, name):>16.6f}")
est, se = mc_delta(S, K, r, sigma, T, n_paths=200_000, seed=1)
print(f"\nMonte Carlo delta: {est:.4f} ± {se:.4f}   (analytic {analytic.delta:.4f})")

spots = list(range(50, 151))
panels = [("delta", "Delta"), ("gamma", "Gamma"), ("vega", "Vega"), ("theta", "Theta (per year)")]
fig = make_subplots(rows=2, cols=2, subplot_titles=[p[1] for p in panels])
for i, (field, _) in enumerate(panels):
    row, col = divmod(i, 2)
    for option_type in ("call", "put"):
        ys = [getattr(greeks(s, K, r, sigma, T, option_type), field) for s in spots]
        fig.add_trace(go.Scatter(x=spots, y=ys, mode="lines", name=option_type,
                                 legendgroup=option_type, showlegend=(i == 0)),
                      row=row + 1, col=col + 1)
    fig.add_vline(x=K, line_dash="dot", row=row + 1, col=col + 1)
fig.update_xaxes(title_text="spot price S")
fig.show()