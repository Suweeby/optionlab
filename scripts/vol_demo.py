"""Download prices for a ticker and compare volatility estimators.

Usage:  python scripts/vol_demo.py AAPL
"""
import sys

import plotly.graph_objects as go

from optionlab.data import load_prices
from optionlab.volatility import (
    annualized_drift, drift_standard_error, ewma_volatility, rolling_volatility,
    volatility_by_lookback,
)

ticker = sys.argv[1] if len(sys.argv) > 1 else "AAPL"
data = load_prices(ticker, lookback_days=730)
prices = data.prices

print(f"{data.ticker}: {len(prices)} prices, {prices.index[0].date()} to {prices.index[-1].date()}")
print(f"source: {data.source}. {data.note}\n")

table = volatility_by_lookback(prices)
print("Historical volatility by lookback window (annualized):")
print(table.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

ewma = ewma_volatility(prices)
print(f"\nEWMA volatility (lambda = 0.94), latest: {ewma.iloc[-1]:.4f}")

mu, se = annualized_drift(prices), drift_standard_error(prices)
print(f"Estimated drift mu: {mu:.3f} with standard error {se:.3f}  (very noisy: do not trust it)")

fig = go.Figure()
fig.add_trace(go.Scatter(x=rolling_volatility(prices, 21).index, y=rolling_volatility(prices, 21),
                         name="21-day window"))
fig.add_trace(go.Scatter(x=rolling_volatility(prices, 63).index, y=rolling_volatility(prices, 63),
                         name="63-day window"))
fig.add_trace(go.Scatter(x=ewma.index, y=ewma, name="EWMA (0.94)"))
title = f"{data.ticker} annualized volatility estimates"
if not data.is_real:
    title += "  (SYNTHETIC sample data)"
fig.update_layout(title=title, xaxis_title="date", yaxis_title="annualized volatility")
fig.show()