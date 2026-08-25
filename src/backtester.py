"""
Backtest engine.

Two things this module is deliberately strict about, because they are the
two most common ways a backtest lies:

1. TRANSACTION COSTS. Turnover is charged on every rebalance. A dollar-neutral
   daily-rebalanced signal can look excellent gross and be worthless net; the
   cost model is on by default rather than opt-in.

2. CHRONOLOGICAL SPLIT. `train_test_split` cuts the sample by date, never by
   random sampling. Shuffling a time series leaks future information into the
   training set and produces Sharpe ratios that cannot be reproduced live.
   There is no `shuffle=True` option anywhere in this file, by design.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from risk_metrics import summary, drawdown_series


@dataclass
class BacktestResult:
    returns: pd.Series          # net portfolio returns
    gross_returns: pd.Series
    equity_curve: pd.Series
    weights: pd.DataFrame
    turnover: pd.Series
    metrics: dict = field(default_factory=dict)

    def drawdowns(self) -> pd.Series:
        return drawdown_series(self.returns)


def run_backtest(
    prices: pd.DataFrame,
    weights: pd.DataFrame,
    cost_bps: float = 5.0,
) -> BacktestResult:
    """
    Apply a weight matrix to a price matrix and return the net performance.

    Parameters
    ----------
    prices : daily prices, one column per asset
    weights : portfolio weights, already lagged by the strategy module so
              that weights on row t are applied to the return of row t
    cost_bps : one-way transaction cost in basis points of traded notional
    """
    asset_returns = prices.pct_change().reindex(weights.index).fillna(0.0)
    aligned_weights = weights.reindex(columns=asset_returns.columns).fillna(0.0)

    gross = (aligned_weights * asset_returns).sum(axis=1)

    # Turnover = sum of absolute weight changes between consecutive days
    turnover = aligned_weights.diff().abs().sum(axis=1).fillna(0.0)
    costs = turnover * (cost_bps / 10_000)
    net = gross - costs

    equity = (1 + net).cumprod()

    return BacktestResult(
        returns=net,
        gross_returns=gross,
        equity_curve=equity,
        weights=aligned_weights,
        turnover=turnover,
        metrics=summary(net),
    )


def train_test_split(prices: pd.DataFrame, split_date: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Chronological split — everything before `split_date` is in-sample,
    everything from `split_date` onward is out-of-sample.

    Deliberately NOT sklearn's train_test_split: random shuffling of a price
    series lets the model see future prices while fitting, which is the single
    most common source of a backtest that works on paper and fails live.
    """
    split_ts = pd.Timestamp(split_date)
    train = prices.loc[prices.index < split_ts]
    test = prices.loc[prices.index >= split_ts]
    if len(train) == 0 or len(test) == 0:
        raise ValueError(f"split_date {split_date} leaves one side of the split empty")
    return train, test


def walk_forward(
    prices: pd.DataFrame,
    signal_fn,
    n_folds: int = 4,
    cost_bps: float = 5.0,
) -> pd.DataFrame:
    """
    Expanding-window walk-forward evaluation: each fold trains on everything
    up to a cut point and evaluates on the following block only. This is the
    time-series analogue of cross-validation, and the only honest way to get
    several out-of-sample estimates from one price history.
    """
    cut_points = pd.Series(prices.index).quantile(
        [i / (n_folds + 1) for i in range(1, n_folds + 1)]
    )

    rows = []
    for fold, cut in enumerate(cut_points, start=1):
        test_prices = prices.loc[prices.index >= cut]
        if len(test_prices) < 30:
            continue
        weights = signal_fn(prices).reindex(test_prices.index)
        result = run_backtest(test_prices, weights, cost_bps=cost_bps)
        rows.append(
            {
                "fold": fold,
                "oos_start": str(cut.date()),
                "n_days": len(test_prices),
                "sharpe": result.metrics["sharpe"],
                "ann_return": result.metrics["ann_return"],
                "max_drawdown": result.metrics["max_drawdown"],
            }
        )

    return pd.DataFrame(rows)
