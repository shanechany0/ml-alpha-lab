"""Cost decomposition: explicit, implicit, and total cost attribution."""

from __future__ import annotations

from typing import Any

import pandas as pd

from src.costs.market_impact import SquareRootImpactModel
from src.costs.transaction_costs import TransactionCostModel


class CostDecomposer:
    """Decomposes total trading costs into explicit and implicit components.

    Explicit costs include spread and commission. Implicit costs include
    market impact and execution delay costs.

    Attributes:
        config: Configuration dictionary.
        tcm: TransactionCostModel instance.
        impact_model: SquareRootImpactModel instance.
    """

    def __init__(self, config: dict | None = None) -> None:
        """Initializes CostDecomposer.

        Args:
            config: Optional configuration. Recognized keys:
                'spread_bps', 'commission_bps', 'impact_coefficient'.
        """
        self.config = config or {}
        self.tcm = TransactionCostModel(config)
        self.impact_model = SquareRootImpactModel(
            coefficient=self.config.get("impact_coefficient", 0.1)
        )

    def decompose(
        self,
        signals: pd.DataFrame,
        prices: pd.DataFrame,
        adv_data: pd.DataFrame,
        volatility: pd.DataFrame,
    ) -> pd.DataFrame:
        """Decomposes per-period costs into explicit, implicit, and total.

        Args:
            signals: Position weight DataFrame (dates × assets).
            prices: Asset price DataFrame (dates × assets).
            adv_data: Average daily volume DataFrame (dates × assets).
            volatility: Daily return volatility DataFrame (dates × assets).

        Returns:
            DataFrame with columns 'explicit', 'implicit', 'total' and
            date index.
        """
        spread_bps = self.config.get("spread_bps", 5.0)
        commission_bps = self.config.get("commission_bps", 10.0)

        explicit = self.explicit_costs(signals, prices, spread_bps, commission_bps)
        implicit = self.implicit_costs(signals, prices, adv_data, volatility)

        all_index = explicit.index.union(implicit.index)
        explicit = explicit.reindex(all_index, fill_value=0.0)
        implicit = implicit.reindex(all_index, fill_value=0.0)
        total = explicit + implicit

        return pd.DataFrame(
            {"explicit": explicit, "implicit": implicit, "total": total},
            index=all_index,
        )

    def explicit_costs(
        self,
        signals: pd.DataFrame,
        prices: pd.DataFrame,
        spread_bps: float,
        commission_bps: float,
    ) -> pd.Series:
        """Computes per-period explicit costs (spread + commission).

        Args:
            signals: Position weight DataFrame.
            prices: Asset price DataFrame.
            spread_bps: Bid-ask spread in basis points.
            commission_bps: Commission in basis points.

        Returns:
            Per-period explicit cost series.
        """
        return self.tcm.total_cost(signals, prices, spread_bps, commission_bps).rename("explicit")

    def implicit_costs(
        self,
        signals: pd.DataFrame,
        prices: pd.DataFrame,
        adv_data: pd.DataFrame,
        volatility: pd.DataFrame,
    ) -> pd.Series:
        """Computes per-period implicit costs (market impact + delay).

        Args:
            signals: Position weight DataFrame.
            prices: Asset price DataFrame.
            adv_data: Average daily volume DataFrame.
            volatility: Daily return volatility DataFrame.

        Returns:
            Per-period implicit cost series.
        """
        impact = self.impact_model.expected_cost(signals, adv_data, volatility)
        # Delay cost requires forward-looking returns; approximate with zeros if unavailable
        delay = pd.Series(0.0, index=impact.index, name="delay")
        return (impact + delay).rename("implicit")

    def delay_cost(
        self,
        signals: pd.DataFrame,
        returns: pd.DataFrame,
        delay_periods: int = 1,
    ) -> pd.Series:
        """Estimates the cost of execution delay relative to signal generation.

        Measures the return missed by executing one or more periods after
        the signal was generated.

        Args:
            signals: Position weight DataFrame.
            returns: Asset return DataFrame.
            delay_periods: Number of periods of execution delay.

        Returns:
            Per-period delay cost series (positive = cost from missed return).
        """
        aligned_signals, aligned_returns = signals.align(returns, join="inner")
        immediate = (aligned_signals.shift(1).fillna(0) * aligned_returns).sum(axis=1)
        delayed = (aligned_signals.shift(1 + delay_periods).fillna(0) * aligned_returns).sum(axis=1)
        return (immediate - delayed).rename("delay_cost")

    def cost_attribution_report(self, results: pd.DataFrame) -> dict[str, float]:
        """Produces a summary attribution report from decomposed costs.

        Args:
            results: DataFrame from decompose() with explicit/implicit/total columns.

        Returns:
            Dictionary of summary statistics including mean, total, and
            fraction breakdown per cost type.
        """
        report: dict[str, float] = {}
        for col in ["explicit", "implicit", "total"]:
            if col in results.columns:
                report[f"mean_{col}"] = float(results[col].mean())
                report[f"total_{col}"] = float(results[col].sum())
                report[f"annualized_{col}"] = float(results[col].mean() * 252)

        if "total" in results.columns and results["total"].sum() != 0:
            for col in ["explicit", "implicit"]:
                if col in results.columns:
                    report[f"fraction_{col}"] = float(
                        results[col].sum() / results["total"].sum()
                    )
        return report
