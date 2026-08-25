"""
End-to-end demo: loads data, runs three strategies through the backtester,
prints the metric table and saves two figures used in the README.

Run with: python notebooks/demo_backtest.py
"""

import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from data_loader import load_synthetic
from strategies import momentum_signal, mean_reversion_signal, buy_and_hold
from backtester import run_backtest, train_test_split, walk_forward

ROOT = os.path.join(os.path.dirname(__file__), "..")
COST_BPS = 5.0
SPLIT_DATE = "2021-01-01"

prices = load_synthetic(n_days=2500, seed=42)
train, test = train_test_split(prices, SPLIT_DATE)

print(f"In-sample:     {train.index.min().date()} -> {train.index.max().date()}  ({len(train)} days)")
print(f"Out-of-sample: {test.index.min().date()} -> {test.index.max().date()}  ({len(test)} days)")
print()

strategies = {
    "Momentum": momentum_signal,
    "Mean reversion": mean_reversion_signal,
    "Buy & hold": buy_and_hold,
}

results = {}
for name, fn in strategies.items():
    weights = fn(prices).reindex(test.index)
    results[name] = run_backtest(test, weights, cost_bps=COST_BPS)

# ---- Metric table (out-of-sample only) ----
table = pd.DataFrame({name: res.metrics for name, res in results.items()}).T
display_cols = ["ann_return", "ann_vol", "sharpe", "sortino", "max_drawdown", "var_95_hist", "es_95", "hit_ratio"]
print("Out-of-sample performance (net of transaction costs):")
print(table[display_cols].round(4).to_string())
print()

# ---- Gross vs net, to show the cost drag ----
print("Cost drag (annualised, out-of-sample):")
for name, res in results.items():
    gross_ann = (1 + res.gross_returns).prod() ** (252 / len(res.gross_returns)) - 1
    net_ann = res.metrics["ann_return"]
    print(f"  {name:<16} gross {gross_ann:+.2%}   net {net_ann:+.2%}   drag {gross_ann - net_ann:.2%}")
print()

# ---- Walk-forward stability ----
print("Walk-forward folds (momentum):")
print(walk_forward(prices, momentum_signal, n_folds=4, cost_bps=COST_BPS).round(3).to_string(index=False))

# ---- Figures ----
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7), sharex=True, height_ratios=[2, 1])

for name, res in results.items():
    ax1.plot(res.equity_curve.index, res.equity_curve.values, label=name, linewidth=1.4)
ax1.axhline(1.0, color="black", linewidth=0.8, linestyle=":")
ax1.set_ylabel("Growth of 1 unit")
ax1.set_title(f"Out-of-sample equity curves (net of {COST_BPS:.0f} bps costs)")
ax1.legend(frameon=False)

for name, res in results.items():
    dd = res.drawdowns()
    ax2.fill_between(dd.index, dd.values, 0, alpha=0.3)
    ax2.plot(dd.index, dd.values, linewidth=1.0, label=name)
ax2.set_ylabel("Drawdown")
ax2.set_xlabel("Date")

fig.tight_layout()
fig.savefig(os.path.join(ROOT, "equity_and_drawdown.png"), dpi=150)

# ---- Return distribution with VaR markers ----
fig2, ax = plt.subplots(figsize=(8, 4.5))
mom = results["Momentum"].returns
ax.hist(mom, bins=80, alpha=0.75, edgecolor="none")
ax.axvline(-results["Momentum"].metrics["var_95_hist"], color="darkorange", linestyle="--", label="Historical VaR 95%")
ax.axvline(-results["Momentum"].metrics["var_95_gauss"], color="steelblue", linestyle="--", label="Gaussian VaR 95%")
ax.axvline(-results["Momentum"].metrics["es_95"], color="crimson", linestyle="-", label="Expected shortfall 95%")
ax.set_title("Momentum: daily return distribution and tail risk measures")
ax.set_xlabel("Daily return")
ax.legend(frameon=False)
fig2.tight_layout()
fig2.savefig(os.path.join(ROOT, "return_distribution.png"), dpi=150)

print(f"\nSaved figures to {os.path.abspath(ROOT)}")
