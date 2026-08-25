"""
Performance and risk metrics.

All annualised figures assume 252 trading days. Every function takes a
pandas Series of *portfolio* returns (already net of costs) so that the
metrics layer stays independent of how those returns were produced.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

TRADING_DAYS = 252


def annualised_return(returns: pd.Series) -> float:
    """Geometric (compounded) annualised return."""
    total_growth = (1 + returns).prod()
    years = len(returns) / TRADING_DAYS
    if years <= 0 or total_growth <= 0:
        return np.nan
    return total_growth ** (1 / years) - 1


def annualised_vol(returns: pd.Series) -> float:
    return returns.std(ddof=1) * np.sqrt(TRADING_DAYS)


def sharpe_ratio(returns: pd.Series, rf: float = 0.0) -> float:
    """Annualised Sharpe. `rf` is an annual risk-free rate."""
    excess = returns - rf / TRADING_DAYS
    vol = excess.std(ddof=1)
    if vol == 0:
        return np.nan
    return np.sqrt(TRADING_DAYS) * excess.mean() / vol


def sortino_ratio(returns: pd.Series, rf: float = 0.0) -> float:
    """
    Like Sharpe, but penalises only downside deviation. Relevant when the
    return distribution is skewed and upside volatility should not be
    treated as risk.
    """
    excess = returns - rf / TRADING_DAYS
    downside = excess[excess < 0]
    if len(downside) < 2:
        return np.nan
    downside_dev = np.sqrt((downside**2).mean())
    if downside_dev == 0:
        return np.nan
    return np.sqrt(TRADING_DAYS) * excess.mean() / downside_dev


def max_drawdown(returns: pd.Series) -> float:
    """Largest peak-to-trough decline of the cumulative equity curve (negative)."""
    equity = (1 + returns).cumprod()
    running_max = equity.cummax()
    drawdown = equity / running_max - 1
    return drawdown.min()


def drawdown_series(returns: pd.Series) -> pd.Series:
    equity = (1 + returns).cumprod()
    return equity / equity.cummax() - 1


def calmar_ratio(returns: pd.Series) -> float:
    """Annualised return divided by the absolute max drawdown."""
    mdd = abs(max_drawdown(returns))
    if mdd == 0:
        return np.nan
    return annualised_return(returns) / mdd


def historical_var(returns: pd.Series, alpha: float = 0.05) -> float:
    """
    Historical (non-parametric) VaR at confidence 1-alpha, expressed as a
    positive loss figure. Makes no distributional assumption, but is bounded
    by the worst loss actually observed in the sample.
    """
    return -np.quantile(returns.dropna(), alpha)


def parametric_var(returns: pd.Series, alpha: float = 0.05) -> float:
    """
    Gaussian VaR. Reported alongside the historical figure precisely because
    the gap between the two is informative: financial returns are fat-tailed,
    so parametric VaR systematically understates tail risk.
    """
    mu, sigma = returns.mean(), returns.std(ddof=1)
    return -(mu + sigma * stats.norm.ppf(alpha))


def expected_shortfall(returns: pd.Series, alpha: float = 0.05) -> float:
    """
    Conditional VaR: average loss *given* that the VaR threshold is breached.
    Coherent as a risk measure (subadditive), unlike VaR.
    """
    var_threshold = -historical_var(returns, alpha)
    tail = returns[returns <= var_threshold]
    if len(tail) == 0:
        return np.nan
    return -tail.mean()


def hit_ratio(returns: pd.Series) -> float:
    """Fraction of days with a strictly positive return."""
    clean = returns.dropna()
    if len(clean) == 0:
        return np.nan
    return (clean > 0).mean()


def summary(returns: pd.Series, rf: float = 0.0) -> dict:
    """Full metric table for a single return stream."""
    return {
        "ann_return": annualised_return(returns),
        "ann_vol": annualised_vol(returns),
        "sharpe": sharpe_ratio(returns, rf),
        "sortino": sortino_ratio(returns, rf),
        "max_drawdown": max_drawdown(returns),
        "calmar": calmar_ratio(returns),
        "var_95_hist": historical_var(returns, 0.05),
        "var_95_gauss": parametric_var(returns, 0.05),
        "es_95": expected_shortfall(returns, 0.05),
        "hit_ratio": hit_ratio(returns),
        "skew": stats.skew(returns.dropna()),
        "excess_kurtosis": stats.kurtosis(returns.dropna()),
    }
