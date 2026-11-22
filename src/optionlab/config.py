"""Project-wide defaults in one place. Pages, scripts and the engine read them from here.

This module imports nothing from optionlab, so any other module can import it without
creating an import cycle. Rates are decimals (0.05 = 5% per year); pages show percentages.
"""

# ---------------------------------------------------------------- tickers
DEFAULT_TICKER = "AAPL"

# Well-known Yahoo Finance symbols offered as presets on the Data page. Add more freely:
# anything that passes data.normalize_ticker works (e.g. "RELIANCE.NS", "EURUSD=X").
PRESET_TICKERS: tuple[str, ...] = ("AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "JPM", "^GSPC")

# ---------------------------------------------------------------- market inputs
DEFAULT_RISK_FREE_RATE = 0.05     # continuously compounded, per year; the app never looks it up
DEFAULT_LOOKBACK_DAYS = 730       # calendar days of price history to load (about 2 years)

# ---------------------------------------------------------------- time conventions
TRADING_DAYS = 252                # trading days per year, used to annualize daily statistics

# ---------------------------------------------------------------- minimum history (daily returns)
# Drift: its standard error is sigma / sqrt(years), so below one year it is larger than sigma
# itself (20-30 %/yr for a typical stock) and the estimate says nothing useful.
MIN_HISTORY_DAYS = 252
# Volatility: about one month of returns is the shortest window that gives a usable estimate.
MIN_VOL_RETURNS = 21

# ---------------------------------------------------------------- cache housekeeping
# Cached option chains are deleted this many days after their expiry date. Expired chains are
# never used for pricing; the grace period only keeps last week's quotes around for a look back.
EXPIRED_CHAIN_GRACE_DAYS = 7
