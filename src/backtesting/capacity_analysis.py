"""Capacity analysis: estimate maximum AUM and performance degradation with scale."""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.backtesting.performance_metrics import PerformanceMetrics


class CapacityAnalyzer:
    """Estimates strategy capacity and models performance degradation with AUM.

    Uses average daily volume (ADV) participation rates and market impact
    models to determine the point at which strategy alpha decays.

    Attributes:
        config: Configuration dictionary.
        metrics: PerformanceMetrics instance.
    """

    def __init__(self, config: dict | None = None) -> None:
        """Initializes CapacityAnalyzer.

        Args:
            config: Optional configuration. Recognized keys:
                'risk_free_rate', 'annualization', 'participation_rate'.
        """
        self.config = config or {}
        self.metrics = PerformanceMetrics(
            risk_free_rate=self.config.get("risk_free_rate", 0.05),
            annualization=self.config.get("annualization", 252),
        )

    def analyze(
        self,
        strategy_returns: pd.Series,
        signals: pd.DataFrame,
        adv_data: pd.DataFrame,
    ) -> dict[str, Any]:
        """Runs a full capacity analysis.

        Args:
            strategy_returns: Daily strategy return series.
            signals: Position weight DataFrame (dates × assets).
            adv_data: Average daily volume in $ millions (dates × assets).

        Returns:
            Dictionary containing capacity estimate, performance vs AUM
            DataFrame, and impact degradation series.
        """
        capacity = self.estimate_capacity(signals, adv_data)
        aum_levels = [capacity * f for f in [0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]]
        perf_df = self.performance_vs_aum(strategy_returns, signals, adv_data, aum_levels)
        base_sharpe = self.metrics.sharpe_ratio(strategy_returns)
        degradation = self.impact_degradation(base_sharpe, aum_levels, capacity)

        return {
            "capacity_millions": capacity,
            "base_sharpe": base_sharpe,
            "performance_vs_aum": perf_df,
            "degradation": degradation,
            "aum_levels": aum_levels,
        }

    def estimate_capacity(
        self,
        signals: pd.DataFrame,
        adv_data: pd.DataFrame,
        participation_rate: float = 0.05,
    ) -> float:
        """Estimates maximum strategy capacity in $ millions.

        Capacity is determined by the maximum AUM at which the strategy
        can trade without exceeding `participation_rate` of ADV.

        Args:
            signals: Position weight DataFrame (dates × assets).
            adv_data: ADV in $ millions (dates × assets).
            participation_rate: Maximum fraction of ADV allowed per trade.

        Returns:
            Estimated capacity in $ millions.
        """
        aligned_signals, aligned_adv = signals.align(adv_data, join="inner")
        turnover = signals.diff().abs().fillna(0)

        total_capacity = 0.0
        for asset in aligned_signals.columns:
            if asset not in aligned_adv.columns:
                continue
            avg_adv = aligned_adv[asset].mean()
            avg_turnover_weight = turnover[asset].mean() if asset in turnover.columns else 0.0
            if avg_turnover_weight > 0:
                asset_capacity = (avg_adv * participation_rate) / avg_turnover_weight
                total_capacity += asset_capacity

        return max(total_capacity, 0.0)

    def performance_vs_aum(
        self,
        base_returns: pd.Series,
        signals: pd.DataFrame,
        adv_data: pd.DataFrame,
        aum_levels: list[float] | None = None,
    ) -> pd.DataFrame:
        """Models strategy performance across a range of AUM levels.

        Args:
            base_returns: Strategy daily returns at minimal AUM.
            signals: Position weight DataFrame.
            adv_data: ADV in $ millions.
            aum_levels: List of AUM levels in $ millions to evaluate.
                Defaults to 10 levels from 10M to 1B.

        Returns:
            DataFrame indexed by AUM with columns for Sharpe, annual return,
            and max drawdown.
        """
        if aum_levels is None:
            aum_levels = [10, 25, 50, 100, 200, 500, 1000]

        capacity = self.estimate_capacity(signals, adv_data)
        base_sharpe = self.metrics.sharpe_ratio(base_returns)
        rows = []
        for aum in aum_levels:
            ratio = aum / max(capacity, 1e-6)
            degradation_factor = 1.0 / (1.0 + ratio**0.5)
            adj_sharpe = base_sharpe * degradation_factor
            adj_returns = base_returns * degradation_factor
            rows.append(
                {
                    "aum_millions": aum,
                    "sharpe_ratio": adj_sharpe,
                    "annual_return": float(adj_returns.mean() * self.metrics.annualization),
                    "max_drawdown": self.metrics.max_drawdown(adj_returns),
                    "capacity_utilization": ratio,
                }
            )
        return pd.DataFrame(rows).set_index("aum_millions")

    def impact_degradation(
        self, base_sharpe: float, aum_levels: list[float], capacity: float
    ) -> pd.Series:
        """Models Sharpe ratio decay as a function of AUM relative to capacity.

        Uses the model: Sharpe(AUM) = base_sharpe / (1 + sqrt(AUM / capacity))

        Args:
            base_sharpe: Baseline Sharpe ratio at negligible AUM.
            aum_levels: AUM levels in $ millions.
            capacity: Estimated strategy capacity in $ millions.

        Returns:
            Series of Sharpe ratios indexed by AUM level.
        """
        capacity = max(capacity, 1e-6)
        sharpes = [
            base_sharpe / (1.0 + np.sqrt(aum / capacity)) for aum in aum_levels
        ]
        return pd.Series(sharpes, index=aum_levels, name="sharpe_ratio")

    def plot_capacity_curve(self, results: pd.DataFrame) -> None:
        """Plots strategy performance metrics vs AUM.

        Args:
            results: DataFrame from performance_vs_aum() with AUM as index.
        """
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        axes[0].plot(results.index, results["sharpe_ratio"], marker="o", color="steelblue")
        axes[0].set_xlabel("AUM ($ millions)")
        axes[0].set_ylabel("Sharpe Ratio")
        axes[0].set_title("Sharpe Ratio vs AUM")
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(results.index, results["annual_return"] * 100, marker="s", color="green")
        axes[1].set_xlabel("AUM ($ millions)")
        axes[1].set_ylabel("Annual Return (%)")
        axes[1].set_title("Annual Return vs AUM")
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()
