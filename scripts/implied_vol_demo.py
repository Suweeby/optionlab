"""Implied volatility: recover a known sigma, compare Newton with bisection, and plot price vs sigma."""
import numpy as np
import plotly.graph_objects as go

from optionlab.implied_vol import implied_vol_detailed
from optionlab.pricing.black_scholes import bs_price

S, r, T, true_vol = 100, 0.05, 1.0, 0.25

print(f"True sigma = {true_vol}\n")
print(f"{'K':>5} {'price':>9} {'implied':>10} {'Newton its':>11} {'bisection its':>14}")
for K in (70, 85, 100, 115, 130):
    price = bs_price(S, K, r, true_vol, T)
    newton = implied_vol_detailed(price, S, K, r, T)
    bisect = implied_vol_detailed(price, S, K, r, T, max_newton_iter=0)
    print(f"{K:>5} {price:>9.4f} {newton.vol:>10.6f} {newton.iterations:>11} {bisect.iterations:>14}")

sigmas = np.linspace(0.01, 1.0, 200)
fig = go.Figure()
for K in (70, 100, 130):
    fig.add_trace(go.Scatter(x=sigmas, y=[bs_price(S, K, r, s, T) for s in sigmas],
                             mode="lines", name=f"call, K = {K}"))
    fig.add_trace(go.Scatter(x=[true_vol], y=[bs_price(S, K, r, true_vol, T)], mode="markers",
                             marker=dict(size=10), name=f"K = {K}, sigma = {true_vol}",
                             showlegend=False))
fig.update_layout(title="Black-Scholes call price against volatility (always increasing)",
                  xaxis_title="volatility sigma", yaxis_title="option price")
fig.show()