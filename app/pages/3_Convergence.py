"""Convergence page: how the binomial tree and Monte Carlo approach Black-Scholes."""
import numpy as np
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from optionlab.convergence import (
    binomial_convergence, monte_carlo_convergence, monte_carlo_single_runs,
)
from optionlab.pricing.black_scholes import bs_price

st.set_page_config(page_title="Convergence · OptionLab", page_icon="📈", layout="wide")


@st.cache_data(show_spinner="Pricing the binomial tree for many step counts...")
def cached_binomial(S, K, r, sigma, T, option_type, max_n):
    dense = binomial_convergence(S, K, r, sigma, T, option_type, n_values=range(5, min(200, max_n) + 1))
    grid = np.unique(np.round(np.geomspace(10, max_n, 25)).astype(int))
    ns = sorted({int(n) for n in grid} | {int(n) + 1 for n in grid})     # n and n+1: shows the zigzag
    wide = binomial_convergence(S, K, r, sigma, T, option_type, n_values=ns)
    return dense, wide


@st.cache_data(show_spinner="Running Monte Carlo many times...")
def cached_mc(S, K, r, sigma, T, option_type, counts, replications, seed):
    plain = monte_carlo_convergence(S, K, r, sigma, T, option_type, counts, replications, False, seed)
    anti = monte_carlo_convergence(S, K, r, sigma, T, option_type, counts, replications, True, seed)
    return plain, anti


@st.cache_data(show_spinner=False)
def cached_runs(S, K, r, sigma, T, option_type, counts, antithetic, seed):
    return monte_carlo_single_runs(S, K, r, sigma, T, option_type, counts, antithetic, seed)


# ---------------------------------------------------------------- sidebar inputs
with st.sidebar:
    st.header("Inputs")
    option_type = st.radio("Option type", ["call", "put"], key="conv_type", horizontal=True)
    S = st.number_input("Spot price S", min_value=0.01, value=100.0, step=1.0, key="conv_S")
    K = st.number_input("Strike price K", min_value=0.01, value=100.0, step=1.0, key="conv_K")
    r_pct = st.number_input("Risk-free rate r (% per year)", value=5.0, step=0.25, format="%.2f",
                            key="conv_r")
    sigma_pct = st.number_input("Volatility σ (% per year)", min_value=0.1, value=20.0, step=1.0,
                                key="conv_sigma")
    T = st.number_input("Time to expiry T (years)", min_value=0.01, value=1.0, step=0.25, key="conv_T")
    st.divider()
    max_n = st.select_slider("Largest tree size n", options=[100, 200, 500, 1000, 2000], value=1000,
                             key="conv_max_n", help="Bigger trees are more accurate but slower.")
    max_exp = st.select_slider("Largest Monte Carlo size N", options=[4, 5, 6], value=5,
                               format_func=lambda e: f"{10**e:,}", key="conv_max_exp",
                               help="Number of simulated paths in the biggest run.")
    replications = st.number_input("Repetitions per size", min_value=5, max_value=50, value=20,
                                   step=5, key="conv_reps",
                                   help="Each Monte Carlo size is repeated this many times to measure its real error.")
    antithetic = st.checkbox("Antithetic variates in the single-run chart", value=True, key="conv_anti")
    seed = st.number_input("Random seed", min_value=0, value=42, step=1, key="conv_seed")

# ---------------------------------------------------------------- main page
st.title("Convergence to Black-Scholes")
r, sigma = r_pct / 100.0, sigma_pct / 100.0
try:
    bs = bs_price(S, K, r, sigma, T, option_type)
    dense, wide = cached_binomial(S, K, r, sigma, T, option_type, int(max_n))
    counts = tuple(10**e for e in range(2, int(max_exp) + 1))
    plain, anti = cached_mc(S, K, r, sigma, T, option_type, counts, int(replications), int(seed))
    runs = cached_runs(S, K, r, sigma, T, option_type, counts, bool(antithetic), int(seed))
except ValueError as exc:
    st.error(f"Could not run the study: {exc}")
    st.stop()

st.metric("Black-Scholes price (the target)", f"{bs:.4f}")
tab_tree, tab_mc = st.tabs(["Binomial tree", "Monte Carlo"])

# ---------------------------------------------------------------- binomial tab
with tab_tree:
    last = wide[wide["n"] == int(max_n)].iloc[0]
    c1, c2 = st.columns(2)
    c1.metric(f"Tree price at n = {int(last['n']):,}", f"{last['binomial']:.4f}",
              delta=f"{last['error']:+.5f} vs Black-Scholes", delta_color="off")
    c2.metric("n × |error| at the largest n", f"{last['n'] * last['abs_error']:.3f}",
              help="Roughly constant as n grows: the error shrinks like 1/n.")

    fig = make_subplots(rows=1, cols=2, subplot_titles=("Tree price vs n (zigzag)", "|error| vs n (log-log)"))
    fig.add_trace(go.Scatter(x=dense["n"], y=dense["binomial"], mode="lines+markers", name="binomial"),
                  row=1, col=1)
    fig.add_hline(y=bs, line_dash="dash", annotation_text="Black-Scholes", row=1, col=1)
    positive = wide[wide["abs_error"] > 0]
    fig.add_trace(go.Scatter(x=positive["n"], y=positive["abs_error"], mode="markers", name="|error|"),
                  row=1, col=2)
    ref = positive["abs_error"].iloc[0] * positive["n"].iloc[0] / positive["n"]
    fig.add_trace(go.Scatter(x=positive["n"], y=ref, mode="lines", name="reference c/n",
                             line=dict(dash="dash")), row=1, col=2)
    fig.update_xaxes(title_text="n (tree steps)", row=1, col=1)
    fig.update_xaxes(title_text="n (tree steps)", type="log", row=1, col=2)
    fig.update_yaxes(title_text="price", row=1, col=1)
    fig.update_yaxes(title_text="absolute error", type="log", row=1, col=2)
    fig.update_layout(height=460, legend=dict(orientation="h", y=-0.25))
    st.plotly_chart(fig, width="stretch")
    st.markdown(
        "The tree price **zigzags** around Black-Scholes (odd and even `n` err in opposite directions) "
        "while the error shrinks like `1/n`: 10x more steps gives about 10x less error. "
        "The cost, though, grows like `n²`."
    )
    st.dataframe(wide[["n", "binomial", "error", "abs_error"]], hide_index=True, width="stretch")

# ---------------------------------------------------------------- Monte Carlo tab
with tab_mc:
    top = runs.iloc[-1]
    c1, c2 = st.columns(2)
    c1.metric(f"Monte Carlo at N = {int(top['n_paths']):,}", f"{top['price']:.4f} ± {top['std_error']:.4f}",
              delta=f"{top['error']:+.4f} vs Black-Scholes", delta_color="off")
    c2.metric("Intervals containing Black-Scholes", f"{int(runs['covers'].sum())} of {len(runs)}",
              help="About 95% of confidence intervals should contain the true price.")

    fig = make_subplots(rows=1, cols=2, subplot_titles=("One run per size, with 95% interval",
                                                        "Typical error vs N (log-log)"))
    fig.add_trace(go.Scatter(
        x=runs["n_paths"], y=runs["price"], mode="markers", name="MC price",
        error_y=dict(type="data", symmetric=False, array=runs["ci_high"] - runs["price"],
                     arrayminus=runs["price"] - runs["ci_low"])), row=1, col=1)
    fig.add_hline(y=bs, line_dash="dash", annotation_text="Black-Scholes", row=1, col=1)
    fig.add_trace(go.Scatter(x=plain["n_paths"], y=plain["rmse"], mode="lines+markers",
                             name="RMSE (plain)"), row=1, col=2)
    fig.add_trace(go.Scatter(x=anti["n_paths"], y=anti["rmse"], mode="lines+markers",
                             name="RMSE (antithetic)"), row=1, col=2)
    n = plain["n_paths"].to_numpy(dtype=float)
    fig.add_trace(go.Scatter(x=n, y=plain["rmse"].iloc[0] * np.sqrt(n[0] / n), mode="lines",
                             name="reference c/√N", line=dict(dash="dash")), row=1, col=2)
    fig.update_xaxes(title_text="paths N", type="log", row=1, col=1)
    fig.update_xaxes(title_text="paths N", type="log", row=1, col=2)
    fig.update_yaxes(title_text="price", row=1, col=1)
    fig.update_yaxes(title_text="root-mean-square error", type="log", row=1, col=2)
    fig.update_layout(height=460, legend=dict(orientation="h", y=-0.25))
    st.plotly_chart(fig, width="stretch")
    st.markdown(
        "Monte Carlo has a **random** error that shrinks like `1/√N`: 100x more paths gives 10x less "
        "error. Antithetic variates (pairing each random shock with its mirror image) lower the error "
        "at the same N. A 95% interval misses the true price about 1 time in 20, by design."
    )
    st.dataframe(plain[["n_paths", "mean_price", "mean_std_error", "rmse"]].rename(
        columns={"mean_std_error": "average reported SE", "rmse": "actual RMSE"}),
        hide_index=True, width="stretch")