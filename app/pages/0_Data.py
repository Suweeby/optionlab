"""Data page: download, refresh, inspect, audit, upload and export price data."""
import io
import re
from datetime import date, timedelta

import plotly.graph_objects as go
import streamlit as st

from optionlab.audit.data_audit import audit_price_data, count_statuses
from optionlab.config import DEFAULT_LOOKBACK_DAYS, DEFAULT_TICKER, PRESET_TICKERS
from optionlab.data import (
    DataUnavailableError, TickerNotFoundError, list_cached, load_cached_prices,
    load_prices_from_csv, normalize_ticker, prices_to_csv, refresh_prices,
)
from optionlab.market import purge_expired_chains

st.set_page_config(page_title="Data · OptionLab", page_icon="📈", layout="wide")

SELECTED_KEY = "selected_ticker"          # the dataset other pages use (label of a ticker or upload)
SELECTED_UPLOAD_KEY = "selected_is_upload"
UPLOADS_KEY = "uploads"                   # {label: PriceData}, this browser session only
MAX_UPLOAD_MB = 5
UPLOAD_SUFFIX = " (upload)"

st.session_state.setdefault(UPLOADS_KEY, {})
if "chains_purged" not in st.session_state:          # housekeeping, once per session
    st.session_state["chains_purged"] = len(purge_expired_chains())


def parse_tickers(presets: list[str], extra: str) -> tuple[list[str], list[str]]:
    """Valid tickers (de-duplicated, in order) and the entries that are not valid tickers."""
    valid, invalid = [], []
    for raw in list(presets) + [t for t in re.split(r"[,\s]+", extra) if t]:
        try:
            ticker = normalize_ticker(raw)
        except ValueError:
            invalid.append(raw)
            continue
        if ticker not in valid:
            valid.append(ticker)
    return valid, invalid


st.title("Data")
st.caption("Download, refresh, inspect and export the price data the other pages use. Prices are "
           "adjusted closes from Yahoo Finance (via yfinance), cached on this computer.")

# ---------------------------------------------------------------- 1. download / refresh
st.subheader("1. Download or refresh")
selected = st.session_state.get(SELECTED_KEY)
left, right = st.columns([2, 1])
with left:
    chosen = st.multiselect(
        "Tickers", PRESET_TICKERS, key="data_presets",
        default=[selected if selected in PRESET_TICKERS else DEFAULT_TICKER],
        help="Well-known symbols. Add any other Yahoo Finance symbol in the box below.")
    extra = st.text_input("Other tickers (comma-separated)", key="data_extra",
                          placeholder="e.g. TSLA, RELIANCE.NS, ^FTSE")
with right:
    range_mode = st.radio("History to download", ["Last N days", "Date range"],
                          key="data_range_mode", horizontal=True)
    today = date.today()
    if range_mode == "Last N days":
        lookback = st.number_input("Calendar days", min_value=30, max_value=36_500,
                                   value=DEFAULT_LOOKBACK_DAYS, step=30, key="data_lookback")
        window = {"lookback_days": int(lookback)}
    else:
        start = st.date_input("Start", value=today - timedelta(days=DEFAULT_LOOKBACK_DAYS),
                              max_value=today, key="data_start")
        end = st.date_input("End", value=today, max_value=today, key="data_end")
        window = {"start": start, "end": end}

if st.button("Download / refresh", type="primary", key="data_download"):
    tickers, invalid = parse_tickers(chosen, extra)
    for bad in invalid:
        st.error(f"'{bad}' is not a valid ticker symbol.")
    if not tickers and not invalid:
        st.warning("Choose at least one ticker.")
    for ticker in tickers:                        # each ticker succeeds or fails on its own
        with st.spinner(f"Downloading {ticker}..."):
            try:
                data = refresh_prices(ticker, **window)
            except TickerNotFoundError as exc:
                st.error(f"{ticker}: {exc}")
            except DataUnavailableError as exc:
                st.error(f"{ticker}: {exc} Any cached data was left unchanged.")
            except ValueError as exc:
                st.error(f"{ticker}: {exc}")
            else:
                st.success(f"{ticker}: {len(data.prices)} prices from {data.prices.index[0].date()} "
                           f"to {data.prices.index[-1].date()}. {data.note}")
    st.cache_data.clear()                         # other pages re-read the refreshed cache

# ---------------------------------------------------------------- 2. cached datasets
st.subheader("2. Cached datasets")
cached = list_cached()
if cached.empty:
    st.info("Nothing is cached yet. Download a ticker above.")
else:
    st.dataframe(cached, hide_index=True, width="stretch", column_config={
        "first_date": "first date", "last_date": "last date",
        "last_refresh_utc": "last refresh (UTC)", "history_start": "first trading day (if known)",
    })
if st.session_state["chains_purged"]:
    st.caption(f"Removed {st.session_state['chains_purged']} cached option chains that expired "
               "more than a week ago.")

# ---------------------------------------------------------------- 3. upload
st.subheader("3. Upload your own prices (optional)")
st.caption("A CSV with a Date column (YYYY-MM-DD) and an Adj Close or Close column. It stays in this "
           "browser session only, is never written to the cache, and is never used for the "
           "model-vs-market comparison.")
up_left, up_right = st.columns([2, 1])
upload = up_left.file_uploader("CSV file", type=["csv"], key="data_upload")
label = up_right.text_input("Label for this data", value="MYDATA", key="data_upload_label",
                            help="A short name such as MYDATA; letters, digits, '.', '-' only.")
if upload is not None:
    if upload.size > MAX_UPLOAD_MB * 1024 * 1024:
        st.error(f"The file is larger than {MAX_UPLOAD_MB} MB. Daily prices for decades fit in "
                 "well under 1 MB; please check the file.")
    else:
        try:
            uploaded = load_prices_from_csv(io.BytesIO(upload.getvalue()), label)
        except ValueError as exc:
            st.error(f"Could not use this file: {exc}")
        else:
            st.session_state[UPLOADS_KEY][uploaded.ticker] = uploaded
            st.success(uploaded.note)

# ---------------------------------------------------------------- 4. inspect, audit, export
st.subheader("4. Inspect, audit and export")
uploads = st.session_state[UPLOADS_KEY]
options = [t for t, n in zip(cached["ticker"], cached["rows"]) if n] + \
          [name + UPLOAD_SUFFIX for name in uploads]
if not options:
    st.info("Download a ticker or upload a file to inspect it here.")
    st.stop()

current = (selected + UPLOAD_SUFFIX) if st.session_state.get(SELECTED_UPLOAD_KEY) else selected
choice = st.selectbox("Dataset", options, key="data_inspect",
                      index=options.index(current) if current in options else 0)
is_upload = choice.endswith(UPLOAD_SUFFIX)
name = choice.removesuffix(UPLOAD_SUFFIX)
try:
    data = uploads[name] if is_upload else load_cached_prices(name)
except (DataUnavailableError, ValueError) as exc:
    st.error(f"Could not read {name}: {exc}")
    st.stop()

if data.source == "stale-cache":
    st.warning(data.note)
elif is_upload:
    st.info(data.note)

prices = data.prices
currency = f" {data.currency}" if data.currency else ""
m1, m2, m3, m4 = st.columns(4)
m1.metric("Prices", f"{len(prices):,}")
m2.metric("First date", f"{prices.index[0].date()}")
m3.metric("Last date", f"{prices.index[-1].date()}")
m4.metric("Last adjusted close", f"{float(prices.iloc[-1]):,.2f}{currency}")

fig = go.Figure(go.Scatter(x=prices.index, y=prices, mode="lines", name=name))
fig.update_layout(title=f"{name} adjusted close" + (" (user-supplied)" if is_upload else ""),
                  xaxis_title="date", yaxis_title="price", height=380)
st.plotly_chart(fig, width="stretch")

audit = audit_price_data(data)
counts = count_statuses(audit)
st.markdown(f"**Data quality audit:** {counts['PASS']} PASS · {counts['WARN']} WARN · "
            f"{counts['FAIL']} FAIL")
st.dataframe(audit, hide_index=True, width="stretch")

b1, b2 = st.columns(2)
b1.download_button("Download this data as CSV", data=prices_to_csv(data),
                   file_name=f"{name}_adj_close.csv", mime="text/csv", key="data_export")
if b2.button(f"Use {name} on the other pages", key="data_use"):
    st.session_state[SELECTED_KEY] = name
    st.session_state[SELECTED_UPLOAD_KEY] = is_upload
active = st.session_state.get(SELECTED_KEY)
if active:
    kind = " (your upload)" if st.session_state.get(SELECTED_UPLOAD_KEY) else ""
    st.caption(f"Dataset selected for the other pages: **{active}{kind}**.")

with st.expander("How to read the audit"):
    st.markdown(
        """
- **PASS**: the check holds. **WARN**: usable, but look before trusting it.
  **FAIL**: do not use the data as it is.
- **Dates / Prices**: duplicates, missing or non-positive prices. If cleaning removed some rows,
  the check shows WARN so the problem is not hidden.
- **History**: one year (252 returns) is needed for a drift estimate; 21 returns is the
  minimum for a volatility estimate.
- **Largest gap**: weekdays with no price. Exchange holidays count as one missing day each.
- **Big moves**: a one-day move above about 28% (log return 0.25) may be a split the data
  source did not adjust for, or a bad price. Real crashes and jumps also trigger it.
- **Staleness**: weekdays since the last price. **Source**: downloaded data passes; old
  caches and uploads get a WARN; SYNTHETIC data fails.
- The audit checks the data's internal consistency. It cannot tell whether Yahoo's prices
  are correct.
"""
    )
