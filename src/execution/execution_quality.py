"""Execution quality analysis: market impact, timing, and cost breakdown."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class ExecutionQualityAnalyzer:
    """Analyzes the quality of trade execution relative to benchmarks.

    Provides market impact analysis, timing analysis, cost breakdown, and
    consolidated execution reports.
    """

    def __init__(self, config: dict | None = None) -> None:
        """Initialize ExecutionQualityAnalyzer.

        Args:
            config: Optional configuration dict (reserved for future use).
        """
        self._config = config or {}

    def analyze(
        self,
        fills: pd.DataFrame,
        benchmarks: pd.DataFrame,
    ) -> dict[str, Any]:
        """Run a comprehensive execution quality analysis.

        Args:
            fills: DataFrame of fill records with columns: fill_price,
                order_size, side (buy/sell), timestamp.
            benchmarks: DataFrame of benchmark prices with columns: vwap,
                twap, arrival_price aligned to fills.

        Returns:
            Dict containing: avg_slippage, avg_market_impact,
            implementation_shortfall, total_cost_bps, fill_rate.
        """
        results: dict[str, Any] = {}

        if "fill_price" in fills.columns and "vwap" in benchmarks.columns:
            slippage = fills["fill_price"] - benchmarks["vwap"].reindex(fills.index)
            results["avg_slippage_bps"] = float(
                (slippage / benchmarks["vwap"].reindex(fills.index) * 1e4).mean()
            )
        else:
            results["avg_slippage_bps"] = np.nan

        if "arrival_price" in benchmarks.columns and "fill_price" in fills.columns:
            is_cost = fills["fill_price"] - benchmarks["arrival_price"].reindex(
                fills.index
            )
            results["implementation_shortfall_bps"] = float(
                (
                    is_cost
                    / benchmarks["arrival_price"].reindex(fills.index)
                    * 1e4
                ).mean()
            )
        else:
            results["implementation_shortfall_bps"] = np.nan

        results["fill_rate"] = (
            float(fills["fill_ratio"].mean())
            if "fill_ratio" in fills.columns
            else 1.0
        )

        if "total_cost" in fills.columns and "order_size" in fills.columns:
            results["total_cost_bps"] = float(
                (fills["total_cost"] / (fills["order_size"] + 1e-12) * 1e4).mean()
            )
        else:
            results["total_cost_bps"] = np.nan

        return results

    def market_impact_analysis(
        self,
        fills: pd.DataFrame,
        benchmarks: pd.DataFrame,
    ) -> pd.DataFrame:
        """Analyze market impact per fill.

        Args:
            fills: Fill records with fill_price and order_size.
            benchmarks: Benchmark records with arrival_price.

        Returns:
            DataFrame with columns: order_size, fill_price, arrival_price,
            market_impact_bps, participation_rate.
        """
        df = fills.copy()
        if "arrival_price" in benchmarks.columns:
            df["arrival_price"] = benchmarks["arrival_price"].reindex(df.index)
            df["market_impact_bps"] = (
                (df["fill_price"] - df["arrival_price"])
                / (df["arrival_price"] + 1e-12)
                * 1e4
            )
        else:
            df["market_impact_bps"] = np.nan

        return df[
            [c for c in ["order_size", "fill_price", "arrival_price",
                          "market_impact_bps"] if c in df.columns]
        ]

    def timing_analysis(
        self,
        fills: pd.DataFrame,
        market_prices: pd.DataFrame,
    ) -> pd.DataFrame:
        """Analyse price drift before and after execution.

        Args:
            fills: Fill records with fill_price and timestamp.
            market_prices: Market prices with columns for each asset.

        Returns:
            DataFrame with columns: pre_trade_drift_bps, post_trade_drift_bps.
        """
        df = fills.copy()
        if "fill_price" in df.columns and not market_prices.empty:
            last_market = market_prices.iloc[-1]
            first_market = market_prices.iloc[0]

            df["pre_trade_drift_bps"] = np.nan
            df["post_trade_drift_bps"] = np.nan

            for asset in df.index:
                if asset in market_prices.columns:
                    prices = market_prices[asset].dropna()
                    if len(prices) >= 2:
                        df.loc[asset, "pre_trade_drift_bps"] = float(
                            (prices.iloc[0] - prices.iloc[-1])
                            / (prices.iloc[-1] + 1e-12)
                            * 1e4
                        )
        return df[
            [c for c in ["pre_trade_drift_bps", "post_trade_drift_bps"]
             if c in df.columns]
        ]

    def cost_breakdown(
        self,
        fills: pd.DataFrame,
        market_data: pd.DataFrame,
    ) -> dict[str, float]:
        """Break down total execution costs into components.

        Args:
            fills: Fill records with total_cost, market_impact, spread_cost.
            market_data: Market data (unused directly; reserved for extensions).

        Returns:
            Dict with keys: market_impact_bps, spread_cost_bps,
            other_cost_bps, total_cost_bps.
        """
        breakdown: dict[str, float] = {}
        notional = float((fills.get("order_size", pd.Series([1.0]))).sum())

        for col, key in [
            ("market_impact", "market_impact_bps"),
            ("spread_cost", "spread_cost_bps"),
            ("total_cost", "total_cost_bps"),
        ]:
            if col in fills.columns:
                breakdown[key] = float(
                    fills[col].sum() / (notional + 1e-12) * 1e4
                )
            else:
                breakdown[key] = np.nan

        if "total_cost_bps" in breakdown and "market_impact_bps" in breakdown:
            breakdown["other_cost_bps"] = float(
                breakdown["total_cost_bps"]
                - breakdown.get("market_impact_bps", 0.0)
                - breakdown.get("spread_cost_bps", 0.0)
            )

        return breakdown

    def generate_execution_report(
        self,
        fills: pd.DataFrame,
        market_data: pd.DataFrame,
    ) -> str:
        """Generate a human-readable execution quality report.

        Args:
            fills: Fill records DataFrame.
            market_data: Market data DataFrame.

        Returns:
            Formatted multi-line string report.
        """
        breakdown = self.cost_breakdown(fills, market_data)
        n_trades = len(fills)
        total_notional = float(fills.get("order_size", pd.Series([0])).sum())

        lines = [
            "=" * 60,
            "EXECUTION QUALITY REPORT",
            "=" * 60,
            f"Total Trades        : {n_trades}",
            f"Total Notional      : {total_notional:,.0f}",
            "",
            "Cost Breakdown:",
        ]
        for key, val in breakdown.items():
            lines.append(f"  {key:<28}: {val:.2f} bps")

        fill_rate = (
            float(fills["fill_ratio"].mean())
            if "fill_ratio" in fills.columns
            else 1.0
        )
        lines.extend(
            [
                "",
                f"Average Fill Rate   : {fill_rate:.1%}",
                "=" * 60,
            ]
        )
        return "\n".join(lines)
