"""
Signal generation.

CRITICAL DESIGN RULE — no look-ahead bias:
every signal is computed from information available strictly *before* the
return it is meant to capture. Concretely, a signal built from data up to
close of day t is shifted by one day and applied to the return of day t+1.
The `.shift(1)` in each function below is what enforces this; removing it
would silently inflate every performance metric in the repo.
"""

from __future__ import annotations

import pandas as pd


def _rank_to_neutral_weights(scores: pd.DataFrame, sign: int = 1) -> pd.DataFrame:
    """
    Turn a cross-sectional score matrix into dollar-neutral portfolio weights.

    Note on the construction: percentile ranks are NOT symmetric around 0.5
    for a finite universe (with 5 assets they are 0.2 ... 1.0, whose mean is
    0.6), so mapping them directly onto [-1, 1] leaves a systematic net long
    exposure. Demeaning the raw ranks cross-sectionally is what actually
    forces the weights to sum to zero on every date.

    Weights are then scaled to unit gross exposure (sum of |w| = 1) so that
    turnover and transaction costs are comparable across strategies.
    """
    ranks = scores.rank(axis=1)
    centred = sign * ranks.sub(ranks.mean(axis=1), axis=0)

    gross = centred.abs().sum(axis=1)
    weights = centred.div(gross.where(gross > 0), axis=0).fillna(0.0)

    # Lag by one day: weights on date t may only use information up to t-1
    return weights.shift(1).fillna(0.0)


def momentum_signal(prices: pd.DataFrame, lookback: int = 126, skip: int = 21) -> pd.DataFrame:
    """
    Cross-sectional momentum: rank assets by their trailing return over
    `lookback` days, excluding the most recent `skip` days (the classic
    1-month reversal exclusion of Jegadeesh & Titman).

    Returns weights in [-1, 1] that sum to zero (dollar-neutral long/short).
    """
    trailing = prices.shift(skip) / prices.shift(skip + lookback) - 1
    return _rank_to_neutral_weights(trailing, sign=1)


def mean_reversion_signal(prices: pd.DataFrame, lookback: int = 5) -> pd.DataFrame:
    """
    Short-horizon cross-sectional mean reversion: go long the recent losers,
    short the recent winners, over a `lookback`-day window.

    Same dollar-neutral construction as momentum, with the sign flipped.
    """
    trailing = prices / prices.shift(lookback) - 1
    return _rank_to_neutral_weights(trailing, sign=-1)


def buy_and_hold(prices: pd.DataFrame) -> pd.DataFrame:
    """Equal-weight long-only benchmark, rebalanced daily."""
    n = prices.shape[1]
    weights = pd.DataFrame(1.0 / n, index=prices.index, columns=prices.columns)
    return weights.shift(1).fillna(0.0)
