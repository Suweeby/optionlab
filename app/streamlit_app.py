"""OptionLab: entry point. Run with:  streamlit run app/streamlit_app.py"""
import streamlit as st

st.set_page_config(page_title="OptionLab", page_icon="📈", layout="wide")

st.title("OptionLab")
st.subheader("Options pricer and stock simulator")

st.markdown(
    """
OptionLab prices **European options** three independent ways and checks them
against each other:

- **Black-Scholes**: an exact closed-form formula
- **Binomial tree**: step-by-step backward induction
- **Monte Carlo**: the average of many simulated futures

Use the page list in the sidebar:

- **Pricer**: the three prices side by side, plus the Greeks
- **Simulator**: possible futures of a stock, in real-world or risk-neutral mode
- **Convergence**: how the tree and Monte Carlo approach Black-Scholes

**What this is not:** a stock-price predictor, a trading signal or financial advice.
It shows what a standard pricing model implies for the inputs you choose.
"""
)