"""
Data loading for the backtesting framework.

Two sources:
  - `load_yahoo`: real daily prices via yfinance (requires network access)
  - `load_synthetic`: reproducible GBM-simulated prices, so the whole repo
    runs offline and every result in the README is exactly reproducible.

Design note: the backtester never touches the network itself. Data loading is
isolated here so that strategy logic and risk metrics can be unit-tested
deterministically.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def load_synthetic(
    tickers: list[str] | None = None,
    n_days: int = 2500,
    start: str = "2016-01-01",
    mu: float = 0.07,
    sigma: float = 0.22,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Simulate daily close prices under GBM for a small universe of assets.

    Returns a DataFrame indexed by business day, one column per ticker.
    Assets share a common market factor plus idiosyncratic noise, so that
    cross-sectional strategies (momentum ranking) have something to bite on.
    """
    if tickers is None:
        tickers = ["ASSET_A", "ASSET_B", "ASSET_C", "ASSET_D", "ASSET_E"]

    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start=start, periods=n_days)
    dt = 1 / 252

    # Common market factor + idiosyncratic component
    market = rng.standard_normal(n_days)
    prices = {}
    for i, ticker in enumerate(tickers):
        beta = 0.6 + 0.2 * i
        idio = rng.standard_normal(n_days)
        z = beta * market + np.sqrt(max(1 - beta**2, 0.1)) * idio
        drift = (mu - 0.5 * sigma**2) * dt
        log_returns = drift + sigma * np.sqrt(dt) * z
        prices[ticker] = 100 * np.exp(np.cumsum(log_returns))

    return pd.DataFrame(prices, index=dates)


def load_yahoo(tickers: list[str], start: str = "2016-01-01", end: str | None = None) -> pd.DataFrame:
    """
    Download adjusted close prices from Yahoo Finance.

    Requires `yfinance` and network access. Kept deliberately thin: any
    cleaning happens downstream so the raw-vs-clean boundary stays visible.
    """
    import yfinance as yf

    raw = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)
    prices = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]]
    prices = prices.dropna(how="all").ffill()
    return prices


def to_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Simple daily arithmetic returns. First row is dropped (no prior price)."""
    return prices.pct_change().dropna(how="all")
