"""Shared helpers for pages that need a stock. Lives in app/ (not the engine): it uses Streamlit.

- which dataset the user picked on the Data page (else config.DEFAULT_TICKER)
- loading its prices (cached for 15 minutes)
- the "Use market data" sidebar block: spot price and historical volatility from those prices
Pages import it after adding app/ to sys.path (see the top of each page).
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import streamlit as st

from optionlab.config import DEFAULT_TICKER, MIN_VOL_RETURNS
from optionlab.data import DataUnavailableError, PriceData, load_prices, require_min_history
from optionlab.market import purge_expired_chains
from optionlab.volatility import historical_volatility

SELECTED_KEY = "selected_ticker"          # label of the dataset other pages use
SELECTED_UPLOAD_KEY = "selected_is_upload"
UPLOADS_KEY = "uploads"                   # {label: PriceData}, this browser session only
UPLOAD_SUFFIX = " (upload)"
VOL_WINDOWS = {"21 days (1 month)": 21, "63 days (3 months)": 63, "126 days (6 months)": 126,
               "252 days (1 year)": 252, "All available": None}


@dataclass(frozen=True)
class Selection:
    """The dataset other pages should use: a ticker label, and whether it is a user upload."""

    label: str
    is_upload: bool

    @property
    def display(self) -> str:
        """The label as shown to the user, e.g. 'AAPL' or 'MYDATA (upload)'."""
        return self.label + (UPLOAD_SUFFIX if self.is_upload else "")


def current_selection() -> Selection:
    """The dataset chosen on the Data page, else the default ticker.

    Uploads live only in the browser session; if a selected upload is gone, use the default.
    """
    label = st.session_state.get(SELECTED_KEY) or DEFAULT_TICKER
    if st.session_state.get(SELECTED_UPLOAD_KEY):
        if label in st.session_state.get(UPLOADS_KEY, {}):
            return Selection(label, True)
        return Selection(DEFAULT_TICKER, False)
    return Selection(label, False)


@st.cache_data(ttl=900, show_spinner="Loading prices...")
def cached_prices(ticker: str) -> PriceData:
    """load_prices for a ticker, cached for 15 minutes across reruns."""
    return load_prices(ticker)


def selected_prices(selection: Selection) -> PriceData:
    """Prices for the selection: the uploaded data, or the ticker's market data.

    Raises DataUnavailableError, TickerNotFoundError or ValueError like load_prices.
    """
    if selection.is_upload:
        return st.session_state[UPLOADS_KEY][selection.label]
    return cached_prices(selection.label)


def purge_chains_once() -> None:
    """Delete long-expired cached option chains, at most once per browser session."""
    if "chains_purged" not in st.session_state:
        st.session_state["chains_purged"] = len(purge_expired_chains())


@dataclass(frozen=True)
class MarketInputs:
    """Spot price and historical volatility taken from a price series, with their provenance."""

    label: str
    spot: float
    spot_date: date
    sigma: float
    window_used: str          # e.g. "252 days (1 year)" or "all 39 available returns"
    data: PriceData


def market_inputs(data: PriceData, window_label: str) -> MarketInputs:
    """Spot = last adjusted close; sigma = historical volatility over the chosen lookback.

    Raises ValueError for SYNTHETIC data or fewer than MIN_VOL_RETURNS daily returns.
    If the lookback is longer than the history, all available returns are used (and named).
    """
    if data.source == "sample":
        raise ValueError(f"only SYNTHETIC sample prices are available for {data.ticker} (the data "
                         "source could not be reached), and they cannot stand in for a real spot "
                         "price and volatility.")
    prices = data.prices
    require_min_history(prices, MIN_VOL_RETURNS, "a volatility estimate")
    n_returns = len(prices) - 1
    window = VOL_WINDOWS[window_label]
    if window is None or window > n_returns:
        window, used = None, f"all {n_returns} available returns"
    else:
        used = window_label
    return MarketInputs(data.ticker, float(prices.iloc[-1]), prices.index[-1].date(),
                        historical_volatility(prices, window=window), used, data)


def market_toggle(key_prefix: str) -> MarketInputs | None:
    """Sidebar block: a 'Use market data for <ticker>' toggle and a volatility lookback.

    Returns the market inputs, or None when the toggle is off or the data cannot be used
    (the reason is shown with st.error, and the page falls back to the typed inputs).
    Call it inside `with st.sidebar:`.
    """
    selection = current_selection()
    on = st.toggle(f"Use market data for {selection.display}", key=f"{key_prefix}_use_market",
                   help="Fill the spot price and volatility from price data. Choose another stock "
                        "or your own upload on the Data page.")
    if not on:
        return None
    window_label = st.selectbox("Volatility lookback", list(VOL_WINDOWS), index=3,
                                key=f"{key_prefix}_vol_window",
                                help="How much history the historical volatility uses. There is "
                                     "no single correct choice.")
    try:
        inputs = market_inputs(selected_prices(selection), window_label)
    except (DataUnavailableError, ValueError) as exc:
        st.error(f"Market data not used: {exc} Using the typed inputs instead.")
        return None
    source = "your upload" if inputs.data.is_user_supplied else inputs.data.source
    st.caption(f"S = {inputs.spot:,.2f} (adjusted close of {inputs.spot_date}) · "
               f"σ = {inputs.sigma:.1%} from {inputs.window_used} · source: {source}")
    if inputs.data.source == "stale-cache":
        st.warning(inputs.data.note)
    return inputs
