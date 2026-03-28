"""Vectorized backtesting engine for rapid signal evaluation."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.backtesting.performance_metrics import PerformanceMetrics


class VectorizedBacktest:
    """Fast, vectorized backtesting for portfolio signals.

    Computes strategy returns, applies transaction costs, and produces
    summary statistics without event-driven loops.

    Attributes:
        config: Configuration dictionary.
        metrics: PerformanceMetrics instance.
    """

    def __init__(self, config: dict | None = None) -> None:
        """Initializes VectorizedBacktest.

        Args:
            config: Optional configuration overrides. Recognized keys:
                'risk_free_rate', 'annualization'.
        """
        self.config = config or {}
        self.metrics = PerformanceMetrics(
            risk_free_rate=self.config.get("risk_free_rate", 0.05),
            annualization=self.config.get("annualization", 252),
        )

    def run(
        self,
        signals: pd.DataFrame,
        returns: pd.DataFrame,
        transaction_costs: float = 0.001,
    ) -> dict[str, Any]:
        """Executes a full vectorized backtest.

        Args:
            signals: DataFrame of position weights (assets as columns,
                dates as index). Values should sum to 1 or be normalized.
            returns: DataFrame of asset returns aligned with signals.
            transaction_costs: One-way transaction cost in decimal
                (e.g. 0.001 = 10 bps).

        Returns:
            Dictionary containing:
                'strategy_returns': net strategy returns after costs,
                'gross_returns': gross strategy returns,
                'costs': per-period cost series,
                'turnover': per-period turnover,
                'statistics': summary statistics dict.
        """
        cost_bps = transaction_costs * 10_000
        gross_returns = self.compute_strategy_returns(signals, returns)
        costs = self.apply_transaction_costs(signals, cost_bps)
        net_returns = gross_returns - costs
        stats = self.summary_statistics(net_returns, costs)
        return {
            "strategy_returns": net_returns,
            "gross_returns": gross_returns,
            "costs": costs,
            "turnover": self.compute_turnover(signals),
            "statistics": stats,
        }

    def compute_strategy_returns(
        self, signals: pd.DataFrame, returns: pd.DataFrame
    ) -> pd.Series:
        """Computes vectorized portfolio return as weighted sum of asset returns.

        Args:
            signals: Position weight DataFrame (dates × assets).
            returns: Asset return DataFrame (dates × assets).

        Returns:
            Portfolio daily return series.
        """
        aligned_signals, aligned_returns = signals.align(returns, join="inner", axis=None)
        # Lag signals by one period to avoid look-ahead bias
        lagged = aligned_signals.shift(1).fillna(0)
        portfolio_returns = (lagged * aligned_returns).sum(axis=1)
        return portfolio_returns.rename("strategy_returns")

    def apply_transaction_costs(self, signals: pd.DataFrame, cost_bps: float) -> pd.Series:
        """Computes per-period transaction costs from portfolio turnover.

        Args:
            signals: Position weight DataFrame.
            cost_bps: One-way transaction cost in basis points.

        Returns:
            Per-period cost series (in return units).
        """
        turnover = self.compute_turnover(signals)
        return (turnover * cost_bps / 10_000).rename("costs")

    def compute_turnover(self, signals: pd.DataFrame) -> pd.Series:
        """Computes per-period portfolio turnover as sum of absolute weight changes.

        Args:
            signals: Position weight DataFrame.

        Returns:
            Turnover series (0 to 2, where 2 = full portfolio rotation).
        """
        delta = signals.diff().abs()
        return delta.sum(axis=1).rename("turnover")

    def screen_signals(
        self,
        signals: pd.DataFrame,
        returns: pd.DataFrame,
        metric: str = "sharpe",
    ) -> pd.DataFrame:
        """Quickly screens individual asset signals by a performance metric.

        Args:
            signals: Position weight DataFrame (dates × assets).
            returns: Asset return DataFrame (dates × assets).
            metric: Metric to rank by. One of 'sharpe', 'sortino', 'calmar'.

        Returns:
            DataFrame sorted by the chosen metric (descending) with columns
            for each computed metric per asset.
        """
        results = []
        for asset in signals.columns:
            if asset not in returns.columns:
                continue
            single_signal = signals[[asset]]
            single_returns = returns[[asset]]
            strat_ret = self.compute_strategy_returns(single_signal, single_returns)
            row: dict[str, Any] = {"asset": asset}
            row["sharpe"] = self.metrics.sharpe_ratio(strat_ret)
            row["sortino"] = self.metrics.sortino_ratio(strat_ret)
            row["calmar"] = self.metrics.calmar_ratio(strat_ret)
            row["max_drawdown"] = self.metrics.max_drawdown(strat_ret)
            row["annual_return"] = float(strat_ret.mean() * self.metrics.annualization)
            results.append(row)

        df = pd.DataFrame(results).set_index("asset")
        if metric not in df.columns:
            metric = "sharpe"
        return df.sort_values(metric, ascending=False)

    def summary_statistics(
        self, strategy_returns: pd.Series, costs: pd.Series
    ) -> dict[str, float]:
        """Produces a comprehensive summary statistics dictionary.

        Args:
            strategy_returns: Net strategy return series.
            costs: Per-period cost series.

        Returns:
            Dictionary of metric name → value.
        """
        stats = self.metrics.compute_all(strategy_returns)
        stats["total_cost"] = float(costs.sum())
        stats["avg_daily_cost_bps"] = float(costs.mean() * 10_000)
        stats["avg_turnover"] = float(self.compute_turnover(
            pd.DataFrame({"placeholder": np.zeros(len(strategy_returns))})
        ).mean())
        return stats
