"""Simulator page: possible futures of a stock under geometric Brownian motion."""
import math

import numpy as np
import plotly.graph_objects as go
import pandas as pd
import streamlit as st
from scipy.stats import lognorm, norm

from optionlab.pricing.black_scholes import d2
from optionlab.simulation.scenario import simulate_scenario
from optionlab.simulation.stats import (
    gbm_log_return_theory, gbm_quantile, gbm_terminal_mean, log_return_stats,
    terminal_summary, value_at_risk,
)

st.set_page_config(page_title="Simulator · OptionLab", page_icon="📈", layout="wide")


@st.cache_data(show_spinner=False)
def cached_scenario(S0, drift, sigma, T, n_steps, n_paths, n_draw, seed):
    return simulate_scenario(S0, drift, sigma, T, n_steps, n_paths, n_draw, seed)


# ---------------------------------------------------------------- sidebar inputs
with st.sidebar:
    st.header("Inputs")
    mode = st.radio(
        "Simulation mode", ["Real-world (drift = μ)", "Risk-neutral (drift = r)"], key="sim_mode",
        help="Real-world: 'what could plausibly happen' with an expected return you choose. "
             "Risk-neutral: the artificial world used for pricing options.",
    )
    real_world = mode.startswith("Real")
    S0 = st.number_input("Starting price S₀", min_value=0.01, value=100.0, step=1.0, key="sim_S0",
                         help="The stock's price today.")
    K = st.number_input("Strike K (marker only)", min_value=0.01, value=100.0, step=1.0, key="sim_K",
                        help="Drawn as a line on the charts; used for P(S_T > K).")
    if real_world:
        drift_pct = st.number_input(
            "Expected return μ (% per year)", value=8.0, step=0.5, format="%.2f", key="sim_mu_pct",
            help="The average yearly growth you assume. Past data estimates it very poorly, so "
                 "treat it as an assumption, not a forecast.")
    else:
        drift_pct = st.number_input(
            "Risk-free rate r (% per year)", value=5.0, step=0.25, format="%.2f", key="sim_r_pct",
            help="In risk-neutral mode the stock grows at the risk-free rate.")
    sigma_pct = st.number_input("Volatility σ (% per year)", min_value=0.1, value=20.0, step=1.0,
                                key="sim_sigma_pct", help="How much the price fluctuates in a year.")
    T = st.number_input("Horizon T (years)", min_value=0.01, value=1.0, step=0.25, key="sim_T")
    st.divider()
    n_steps = st.number_input("Time steps per path", min_value=1, max_value=1000, value=252,
                              step=21, key="sim_steps", help="252 = one step per trading day for a year.")
    n_paths = st.number_input("Simulated paths N", min_value=10, max_value=500_000, value=20_000,
                              step=10_000, key="sim_paths",
                              help="Used for the histogram and statistics.")
    n_draw = st.number_input("Paths drawn on the chart", min_value=1, max_value=500, value=100,
                             step=10, key="sim_draw",
                             help="Drawing thousands of lines is unreadable, so only a few are shown.")
    seed = st.number_input("Random seed", min_value=0, value=42, step=1, key="sim_seed")

# ---------------------------------------------------------------- main page
st.title("Stock simulator")
if real_world:
    st.caption("Real-world mode: the stock drifts at μ. These are possible futures under the model's "
               "assumptions, not a forecast.")
else:
    st.warning("Risk-neutral mode: the stock drifts at the risk-free rate r. This is the world used to "
               "PRICE options. Its probabilities are pricing weights, not predictions of what will happen.")

drift, sigma = drift_pct / 100.0, sigma_pct / 100.0
try:
    sc = cached_scenario(S0, drift, sigma, T, int(n_steps), int(n_paths), int(n_draw), int(seed))
except ValueError as exc:
    st.error(f"Could not run the simulation: {exc}")
    st.stop()

ST = sc.terminal_prices
summary = terminal_summary(ST, K)
times = sc.times

# ---- fan chart
fig = go.Figure()
lo, hi = gbm_quantile(S0, drift, sigma, times, 0.05), gbm_quantile(S0, drift, sigma, times, 0.95)
fig.add_trace(go.Scatter(x=times, y=hi, mode="lines", line=dict(width=0), showlegend=False,
                         hoverinfo="skip"))
fig.add_trace(go.Scatter(x=times, y=lo, mode="lines", line=dict(width=0), fill="tonexty",
                         fillcolor="rgba(99,110,250,0.15)", name="5%–95% range (theory)",
                         hoverinfo="skip"))
rows = sc.sample_paths.shape[0]
x = np.tile(np.append(times, np.nan), rows)                       # one trace, gaps between paths
y = np.column_stack([sc.sample_paths, np.full(rows, np.nan)]).ravel()
fig.add_trace(go.Scatter(x=x, y=y, mode="lines", line=dict(width=1), opacity=0.45,
                         name=f"{rows} sample paths", connectgaps=False, hoverinfo="skip"))
fig.add_trace(go.Scatter(x=times, y=S0 * np.exp(drift * times), mode="lines",
                         line=dict(dash="dash", width=2), name="mean path S₀·e^(drift·t)"))
fig.add_trace(go.Scatter(x=times, y=S0 * np.exp((drift - 0.5 * sigma**2) * times), mode="lines",
                         line=dict(dash="dot", width=2), name="median path"))
fig.add_hline(y=K, line_dash="dash", annotation_text="strike K")
fig.update_layout(title=f"{rows} of {len(ST):,} simulated paths", xaxis_title="time (years)",
                  yaxis_title="price", legend=dict(orientation="h", y=-0.2), height=480)
st.plotly_chart(fig, width="stretch")

# ---- terminal distribution
col_hist, col_stats = st.columns([3, 2])
with col_hist:
    hist = go.Figure()
    hist.add_trace(go.Histogram(x=ST, histnorm="probability density", nbinsx=80, name="simulated"))
    grid = np.linspace(max(float(ST.min()), 1e-9), float(ST.max()), 400)
    pdf = lognorm.pdf(grid, s=sigma * math.sqrt(T), scale=S0 * math.exp((drift - 0.5 * sigma**2) * T))
    hist.add_trace(go.Scatter(x=grid, y=pdf, mode="lines", name="theory (lognormal)"))
    hist.add_vline(x=K, line_dash="dash", annotation_text="strike K")
    hist.update_layout(title="Terminal price S_T", xaxis_title="price at horizon",
                       yaxis_title="density", height=380)
    st.plotly_chart(hist, width="stretch")

with col_stats:
    p_above = summary["prob_above_K"]
    st.metric("P(S_T > K)" + ("" if real_world else "  (risk-neutral)"), f"{p_above:.3f}")
    st.metric("Mean of S_T", f"{summary['mean']:.2f}",
              delta=f"{summary['mean'] - gbm_terminal_mean(S0, drift, T):+.2f} vs theory", delta_color="off")
    st.metric("Median of S_T", f"{summary['p50']:.2f}")
    if real_world:
        var95 = value_at_risk(ST, S0, 0.95)
        st.metric("95% value at risk", f"{var95:.2f}",
                  help="A loss (S₀ − S_T) that is exceeded in only 5% of simulated scenarios.")
    else:
        n_d2 = float(norm.cdf(d2(S0, K, drift, sigma, T)))
        st.caption(f"Check against Black-Scholes: N(d2) = {n_d2:.3f}. "
                   "In risk-neutral mode P(S_T > K) is exactly this number.")

# ---- summary table and theory check
table = pd.DataFrame(
    {
        "Statistic": ["Mean", "Standard deviation", "Minimum", "5th percentile", "25th percentile",
                      "Median", "75th percentile", "95th percentile", "Maximum"],
        "Value": [summary["mean"], summary["std"], summary["min"], summary["p5"], summary["p25"],
                  summary["p50"], summary["p75"], summary["p95"], summary["max"]],
    }
)
sample_lr = log_return_stats(ST, S0)
theory_lr = gbm_log_return_theory(drift, sigma, T)
check = pd.DataFrame(
    {
        "Quantity": ["Mean of ln(S_T/S₀)", "Variance of ln(S_T/S₀)", "Mean of S_T"],
        "Simulated": [sample_lr["mean"], sample_lr["variance"], summary["mean"]],
        "Theory": [theory_lr["mean"], theory_lr["variance"], gbm_terminal_mean(S0, drift, T)],
    }
)
t1, t2 = st.columns(2)
t1.subheader("Terminal price statistics")
t1.dataframe(table, hide_index=True, width="stretch",
             column_config={"Value": st.column_config.NumberColumn(format="%.2f")})
t2.subheader("Simulation vs theory")
t2.dataframe(check, hide_index=True, width="stretch",
             column_config={"Simulated": st.column_config.NumberColumn(format="%.4f"),
                            "Theory": st.column_config.NumberColumn(format="%.4f")})
t2.caption("Small gaps are ordinary sampling noise; they shrink as you add paths.")

with st.expander("How to read this"):
    st.markdown(
        """
- Each thin line is one possible future for the stock under **geometric Brownian motion**.
- The shaded band is where 90% of outcomes fall (theory). The **mean** path is above the
  **median** path: prices are lognormal, so a few big winners pull the average up.
- The histogram shows where the stock ends up at the horizon; the dashed vertical line is the strike.
- **Real-world mode** uses an expected return μ you choose and is for "what could happen" views.
  **Risk-neutral mode** uses the risk-free rate and is the world used for pricing options.
- This is a model of randomness with assumed parameters, not a prediction.
"""
    )
