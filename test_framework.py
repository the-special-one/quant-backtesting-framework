"""
Run with: pytest tests/ -v

The tests worth reading are `test_no_lookahead_in_signals` and
`test_split_is_chronological` — they encode the two invariants that make the
rest of the numbers meaningful.
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from data_loader import load_synthetic, to_returns
from strategies import momentum_signal, mean_reversion_signal, buy_and_hold
from backtester import run_backtest, train_test_split, walk_forward
from risk_metrics import (
    sharpe_ratio,
    sortino_ratio,
    max_drawdown,
    historical_var,
    expected_shortfall,
    annualised_return,
)


@pytest.fixture(scope="module")
def prices():
    return load_synthetic(n_days=1500, seed=7)


def test_synthetic_data_is_reproducible():
    a = load_synthetic(n_days=200, seed=123)
    b = load_synthetic(n_days=200, seed=123)
    pd.testing.assert_frame_equal(a, b)


def test_no_lookahead_in_signals(prices):
    """
    Truncating the price history must not change any signal value on the
    dates that both versions share. If a signal peeked at future prices,
    removing the tail of the series would change earlier signal values.
    """
    full = momentum_signal(prices)
    truncated = momentum_signal(prices.iloc[:-100])

    common = truncated.index
    pd.testing.assert_frame_equal(full.loc[common], truncated.loc[common])


def test_split_is_chronological(prices):
    train, test = train_test_split(prices, "2019-01-01")
    assert train.index.max() < test.index.min()
    assert len(train) + len(test) == len(prices)


def test_costs_reduce_returns(prices):
    weights = mean_reversion_signal(prices)
    free = run_backtest(prices, weights, cost_bps=0.0)
    costly = run_backtest(prices, weights, cost_bps=20.0)
    assert costly.returns.sum() < free.returns.sum()
    assert (costly.turnover >= 0).all()


def test_dollar_neutral_weights_sum_to_zero(prices):
    weights = momentum_signal(prices)
    row_sums = weights.sum(axis=1).abs()
    # Ignore the warm-up period where the trailing window is not yet filled
    assert row_sums.iloc[200:].max() < 1e-9


def test_max_drawdown_is_non_positive(prices):
    result = run_backtest(prices, buy_and_hold(prices))
    assert result.metrics["max_drawdown"] <= 0


def test_gaussian_var_understates_deep_tail():
    """
    On a fat-tailed sample (Student t, 3 df), Gaussian VaR understates the
    DEEP tail — but only there.

    At 95% the Gaussian estimate is actually the larger of the two: fat tails
    inflate the sample standard deviation, which pushes the Gaussian quantile
    out further than the empirical one at a level that shallow. The two cross
    over around the 98% level. This is exactly why the summary table reports
    historical VaR, Gaussian VaR and expected shortfall side by side rather
    than any one of them alone.
    """
    from risk_metrics import parametric_var

    rng = np.random.default_rng(0)
    fat_tailed = pd.Series(rng.standard_t(df=3, size=200_000) / 100)

    # Deep tail: the empirical loss is far worse than the Gaussian estimate
    assert historical_var(fat_tailed, alpha=0.001) > 1.5 * parametric_var(fat_tailed, alpha=0.001)
    # Shallow tail: the ordering reverses
    assert historical_var(fat_tailed, alpha=0.05) < parametric_var(fat_tailed, alpha=0.05)


def test_expected_shortfall_exceeds_var(prices):
    returns = to_returns(prices).iloc[:, 0]
    assert expected_shortfall(returns) >= historical_var(returns)


def test_sharpe_of_constant_series_is_nan_or_inf():
    flat = pd.Series([0.0] * 100)
    assert np.isnan(sharpe_ratio(flat))


def test_walk_forward_returns_multiple_folds(prices):
    table = walk_forward(prices, momentum_signal, n_folds=3)
    assert len(table) >= 2
    assert {"fold", "sharpe", "max_drawdown"}.issubset(table.columns)
