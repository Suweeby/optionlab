"""Pricer page: Black-Scholes, binomial and Monte Carlo side by side."""
import pandas as pd
import streamlit as st
from optionlab.greeks import greeks, to_trader_units
from optionlab.payoffs import moneyness
from optionlab.payoffs import moneyness
from optionlab.pricing.compare import price_all

st.set_page_config(page_title="Pricer · OptionLab", page_icon="📈", layout="wide")


@st.cache_data(show_spinner=False)
def cached_price_all(S, K, r, sigma, T, option_type, n_steps, n_paths, seed, antithetic):
    return price_all(S, K, r, sigma, T, option_type, n_steps, n_paths, seed, antithetic)


# ---------------------------------------------------------------- sidebar inputs
with st.sidebar:
    st.header("Inputs")
    option_type = st.radio(
        "Option type", ["call", "put"], key="option_type", horizontal=True,
        help="A call pays off if the stock ends ABOVE the strike; a put if it ends BELOW.",
    )
    S = st.number_input(
        "Spot price S", min_value=0.01, value=100.0, step=1.0, key="S",
        help="The stock's current price.",
    )
    K = st.number_input(
        "Strike price K", min_value=0.01, value=100.0, step=1.0, key="K",
        help="The price at which the option holder may buy (call) or sell (put) at expiry.",
    )
    r_pct = st.number_input(
        "Risk-free rate r (% per year)", value=5.0, step=0.25, format="%.2f", key="r_pct",
        help="Continuously compounded interest rate on a risk-free investment.",
    )
    sigma_pct = st.number_input(
        "Volatility σ (% per year)", min_value=0.1, value=20.0, step=1.0, key="sigma_pct",
        help="How much the stock price typically fluctuates in a year. Higher = riskier.",
    )
    T = st.number_input(
        "Time to expiry T (years)", min_value=0.01, value=1.0, step=0.25, key="T",
        help="0.5 means six months.",
    )
    st.divider()
    n_steps = st.number_input(
        "Binomial tree steps n", min_value=1, max_value=5000, value=500, step=50, key="n_steps",
        help="More steps = closer to Black-Scholes, but slower.",
    )
    n_paths = st.number_input(
        "Monte Carlo simulations N", min_value=2, max_value=2_000_000, value=100_000,
        step=10_000, key="n_paths",
        help="Number of simulated futures. 100x more simulations = 10x less random error.",
    )
    antithetic = st.checkbox(
        "Antithetic variates", value=True, key="antithetic",
        help="A variance-reduction trick: simulate each random shock together with its "
             "mirror image. Needs an even number of simulations.",
    )
    seed = st.number_input(
        "Random seed", min_value=0, value=42, step=1, key="seed",
        help="Same seed gives the same Monte Carlo result every time.",
    )

# ---------------------------------------------------------------- main page
st.title("Option pricer")
st.caption("European option · three independent methods · "
           "pricing mode: risk-neutral (the simulated stock drifts at r)")

r, sigma = r_pct / 100.0, sigma_pct / 100.0
try:
    res = cached_price_all(S, K, r, sigma, T, option_type, int(n_steps), int(n_paths),
                           int(seed), antithetic)
    status = moneyness(S, K, option_type)
    greek = to_trader_units(greeks(S, K, r, sigma, T, option_type))
        
except ValueError as exc:
    st.error(f"Could not price this option: {exc}")
    st.stop()

st.markdown(f"This {option_type} is **{status}** (S = {S:g}, K = {K:g}).")

mc = res.monte_carlo
col1, col2, col3 = st.columns(3)
col1.metric("Black-Scholes (exact formula)", f"{res.black_scholes:.4f}")
col2.metric(f"Binomial tree (n = {int(n_steps):,})", f"{res.binomial:.4f}",
            delta=f"{res.binomial_minus_bs:+.4f} vs Black-Scholes", delta_color="off")
col3.metric(f"Monte Carlo (N = {int(n_paths):,})",
            f"{mc.price:.4f} ± {mc.std_error:.4f}",
            delta=f"{res.mc_minus_bs:+.4f} vs Black-Scholes", delta_color="off")

table = pd.DataFrame(
    {
        "Method": ["Black-Scholes", "Binomial tree", "Monte Carlo"],
        "Price": [res.black_scholes, res.binomial, mc.price],
        "Difference vs Black-Scholes": [0.0, res.binomial_minus_bs, res.mc_minus_bs],
        "Uncertainty": [
            "exact",
            "discretization error (shrinks as n grows)",
            f"± {mc.std_error:.4f} standard error (random)",
        ],
    }
)
st.dataframe(table, hide_index=True, width="stretch",
             column_config={"Price": st.column_config.NumberColumn(format="%.4f"),
                            "Difference vs Black-Scholes": st.column_config.NumberColumn(format="%+.4f")})

st.write(f"Monte Carlo {int(mc.confidence * 100)}% confidence interval: "
         f"**[{mc.ci_low:.4f}, {mc.ci_high:.4f}]**")
if res.bs_inside_mc_interval:
    st.success("The Black-Scholes price lies inside the Monte Carlo interval.")
else:
    st.info("The Black-Scholes price lies outside this Monte Carlo interval. "
            "That happens in about 1 run in 20 by design; try another seed or more simulations.")

st.subheader("Greeks (Black-Scholes)")
st.caption("How much the option price changes when one input changes slightly, all else equal.")
g1, g2, g3, g4, g5 = st.columns(5)
g1.metric("Delta", f"{greek.delta:.4f}",
          help="Price change per $1 rise in the stock. Also roughly the number of shares "
               "that hedges the option.")
g2.metric("Gamma", f"{greek.gamma:.4f}",
          help="How much delta changes per $1 rise in the stock (the curvature).")
g3.metric("Vega (per 1% vol)", f"{greek.vega:.4f}",
          help="Price change when volatility rises by one percentage point.")
g4.metric("Theta (per day)", f"{greek.theta:.4f}",
          help="Price change from one day passing (time decay). Usually negative for a bought option.")
g5.metric("Rho (per 1% rate)", f"{greek.rho:.4f}",
          help="Price change when the interest rate rises by one percentage point.")

with st.expander("How to read this"):
    st.markdown(
        """
- **Black-Scholes** is the exact answer under the model's assumptions.
- The **binomial tree** approximates it step by step; its error shrinks as you add steps.
- **Monte Carlo** averages many random futures, so it carries a random error (± standard
  error). 100x more simulations gives about 10x less error.
- The three should agree up to those errors. If they don't, something is wrong.
"""
    )

with st.expander("Assumptions and limitations"):
    st.markdown(
        """
- European exercise only (exercise at expiry), no dividends, constant volatility and rate.
- Real listed stock options are often American-style, many stocks pay dividends, and real
  volatility changes over time, so real market quotes will differ from these prices.
- This is a pricing model, not a forecast: it says nothing about where the stock will go.
"""
    )