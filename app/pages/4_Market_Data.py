"""Market Data page: historical volatility, model prices vs real option quotes, implied volatility."""
from datetime import date

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from optionlab.data import DataUnavailableError, TickerNotFoundError, load_prices, normalize_ticker
from optionlab.market import (
    atm_implied_vol, clean_quotes, compare_quotes, list_expiries, load_option_chain, only_otm,
    time_to_expiry,
)
from optionlab.volatility import ewma_volatility, historical_volatility

st.set_page_config(page_title="Market Data · OptionLab", page_icon="📈", layout="wide")

WINDOWS = {"21 days (1 month)": 21, "63 days (3 months)": 63, "126 days (6 months)": 126,
           "252 days (1 year)": 252, "All available": None}


@st.cache_data(ttl=900, show_spinner="Loading prices...")
def cached_prices(ticker):
    return load_prices(ticker, lookback_days=730)


@st.cache_data(ttl=900, show_spinner="Loading option expiries...")
def cached_expiries(ticker):
    return list_expiries(ticker)


@st.cache_data(ttl=900, show_spinner="Loading option quotes...")
def cached_chain(ticker, expiry):
    return load_option_chain(ticker, expiry)


# ---------------------------------------------------------------- sidebar inputs
with st.sidebar:
    st.header("Inputs")
    ticker_in = st.text_input("Ticker", value="AAPL", key="mkt_ticker",
                              help="A Yahoo Finance symbol, e.g. AAPL, MSFT, ^GSPC or RELIANCE.NS.")
    window_label = st.selectbox("Volatility lookback", list(WINDOWS), index=3, key="mkt_window",
                                help="How much history the historical volatility uses. "
                                     "There is no single correct choice.")
    r_pct = st.number_input("Risk-free rate r (% per year)", value=4.0, step=0.25, format="%.2f",
                            key="mkt_r", help="Set this to the current short-term interest rate; "
                                              "the app does not look it up.")
    option_type = st.radio("Option type", ["call", "put"], key="mkt_type", horizontal=True)
    otm_only = st.checkbox("Out-of-the-money options only", value=True, key="mkt_otm",
                           help="Implied volatility is most reliable for out-of-the-money options.")

# ---------------------------------------------------------------- load prices and expiries
st.title("Market data")
try:
    ticker = normalize_ticker(ticker_in)
    pdata = cached_prices(ticker)
    expiries = cached_expiries(ticker)
except TickerNotFoundError as exc:
    st.error(f"{exc}")
    st.stop()
except (DataUnavailableError, ValueError) as exc:
    st.error(f"Could not load market data: {exc}")
    st.stop()

if not pdata.is_real:
    st.error("Only SYNTHETIC sample prices are available (the data source could not be reached). "
             "This page compares the model with real option quotes, so it needs real data. "
             "Check your internet connection and try again.")
    st.stop()

with st.sidebar:
    expiry = st.selectbox("Option expiry", expiries, key="mkt_expiry")
    if st.button("Refresh data", help="Clear cached downloads and fetch again."):
        st.cache_data.clear()
        st.rerun()

prices = pdata.prices
spot = float(prices.iloc[-1])
spot_date = prices.index[-1].date()
window = WINDOWS[window_label]
n_returns = len(prices) - 1
used_window = None if (window is None or window > n_returns) else window
hist_vol = historical_volatility(prices, window=used_window)
ewma_vol = float(ewma_volatility(prices).iloc[-1])

try:
    chain = cached_chain(ticker, expiry)
    T = time_to_expiry(expiry)
except (DataUnavailableError, ValueError) as exc:
    st.error(f"Could not load option quotes: {exc}")
    st.stop()

if pdata.source == "stale-cache":
    st.warning(f"Prices: {pdata.note}")
if chain.source == "stale-cache":
    st.warning(f"Option quotes: {chain.note}")
if T < 7 / 365:
    st.info("This expiry is less than a week away. Very short-dated quotes are noisy.")

r = r_pct / 100.0
quotes, report = clean_quotes(chain.chain, option_type=option_type)
if otm_only:
    quotes = only_otm(quotes, spot)
if quotes.empty:
    st.warning("No usable quotes for these settings. Try another expiry, the other option type, "
               "or switch off 'out-of-the-money only'.")
    st.stop()

compared = compare_quotes(quotes, spot, r, T, hist_vol)
atm = atm_implied_vol(compared, spot)

# ---------------------------------------------------------------- headline numbers
c1, c2, c3, c4 = st.columns(4)
c1.metric("Spot price (last close)", f"{spot:,.2f}", help=f"Close of {spot_date}.")
c2.metric("Historical volatility", f"{hist_vol * 100:.1f}%",
          help=f"Annualized, from the {window_label.lower()} of log returns.")
c3.metric("EWMA volatility (λ = 0.94)", f"{ewma_vol * 100:.1f}%")
if atm is not None:
    c4.metric("Implied volatility near the money", f"{atm[1] * 100:.1f}%",
              delta=f"{(atm[1] - hist_vol) * 100:+.1f} pts vs historical", delta_color="off",
              help=f"From the quote with strike {atm[0]:g}.")
else:
    c4.metric("Implied volatility near the money", "n/a")

finite = compared["implied_vol"].dropna()
if len(finite):
    gap = float(finite.mean()) - hist_vol
    direction = "above" if gap > 0 else "below"
    st.write(f"Across the {len(finite)} quotes shown, the market's implied volatility averages "
             f"**{abs(gap) * 100:.1f} points {direction}** the historical volatility. "
             f"Time to expiry: {T * 365:.0f} days.")

# ---------------------------------------------------------------- charts
left, right = st.columns(2)
with left:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=compared["strike"], y=compared["ask"], mode="lines",
                             line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=compared["strike"], y=compared["bid"], mode="lines", line=dict(width=0),
                             fill="tonexty", fillcolor="rgba(99,110,250,0.25)", name="bid–ask range"))
    fig.add_trace(go.Scatter(x=compared["strike"], y=compared["mid"], mode="markers", name="market mid"))
    fig.add_trace(go.Scatter(x=compared["strike"], y=compared["model_price"], mode="lines+markers",
                             name="model (historical vol)"))
    fig.add_vline(x=spot, line_dash="dot", annotation_text="spot")
    fig.update_layout(title="Option price by strike", xaxis_title="strike", yaxis_title="price",
                      legend=dict(orientation="h", y=-0.25), height=420)
    st.plotly_chart(fig, width="stretch")
with right:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=compared["strike"], y=compared["implied_vol"], mode="lines+markers",
                             name="implied volatility (our solver)"))
    fig.add_trace(go.Scatter(x=compared["strike"], y=compared["vendor_iv"], mode="markers",
                             name="Yahoo's reported IV", marker=dict(symbol="x", opacity=0.6)))
    fig.add_hline(y=hist_vol, line_dash="dash", annotation_text="historical vol")
    fig.add_vline(x=spot, line_dash="dot", annotation_text="spot")
    fig.update_layout(title="Implied volatility by strike", xaxis_title="strike",
                      yaxis_title="annualized volatility", legend=dict(orientation="h", y=-0.25),
                      height=420)
    st.plotly_chart(fig, width="stretch")

# ---------------------------------------------------------------- table and data quality
st.subheader("Quotes versus model")
table = compared[["option_type", "strike", "bid", "ask", "mid", "model_price", "model_minus_mid",
                  "model_in_spread", "implied_vol", "volume", "open_interest"]]
st.dataframe(table, hide_index=True, width="stretch", column_config={
    "mid": st.column_config.NumberColumn("mid", format="%.3f"),
    "bid": st.column_config.NumberColumn(format="%.3f"),
    "ask": st.column_config.NumberColumn(format="%.3f"),
    "model_price": st.column_config.NumberColumn("model price", format="%.3f"),
    "model_minus_mid": st.column_config.NumberColumn("model − mid", format="%+.3f"),
    "model_in_spread": st.column_config.CheckboxColumn("model inside bid–ask?"),
    "implied_vol": st.column_config.NumberColumn("implied vol", format="%.3f"),
})
st.caption(
    f"Quotes: {report['kept']} of {report['total']} {option_type}s passed the quality filter "
    f"(dropped: {report['missing_or_zero_bid']} with a missing or zero bid, "
    f"{report['crossed_quote']} crossed, {report['wide_spread']} with a very wide spread, "
    f"{report['stale_last_trade']} stale). Source: {chain.source}. Prices source: {pdata.source}."
)

with st.expander("How to read this"):
    st.markdown(
        """
- The **market mid** is the average of the bid and ask. The **model price** is Black-Scholes
  using the *historical* volatility you chose.
- If the model price is **below** the mid, the market is charging for more volatility than the
  past showed (and the other way round). This is a statement about the model and the market,
  not a trading signal.
- **Implied volatility** is the volatility that makes Black-Scholes match the market mid.
  Plotted against strike it is usually not flat (the "smile" or "skew"), which contradicts
  Black-Scholes's constant-volatility assumption.
- Out-of-the-money options give the most reliable implied volatility; deep in-the-money
  options are mostly intrinsic value and barely depend on volatility.
"""
    )
with st.expander("Assumptions and limitations"):
    st.markdown(
        """
- Black-Scholes assumes **European** exercise and **no dividends**. Most listed stock options
  are **American**, and many stocks pay dividends, so gaps from market prices are expected.
- Volatility is **not constant** in reality, and historical volatility depends on the window.
- Yahoo data can be **delayed or incomplete**. The spot price is the last close, which may not
  match the moment the option quotes were taken.
- The risk-free rate is **your input**. Time to expiry is counted in calendar days / 365.
- Mid-prices are not tradable prices. This page is an educational comparison, not advice.
"""
    )