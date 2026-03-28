"""Execution benchmark calculations: VWAP, TWAP, arrival price, IS."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class ExecutionBenchmarks:
    """Computes standard execution quality benchmarks.

    Provides VWAP, TWAP, arrival price, implementation shortfall, and
    VWAP slippage calculations for trade analysis.
    """

    def __init__(self, config: dict | None = None) -> None:
        """Initialize ExecutionBenchmarks.

        Args:
            config: Optional configuration dict (reserved for future use).
        """
        self._config = config or {}

    def vwap(
        self,
        prices: pd.Series,
        volumes: pd.Series,
        window: int | None = None,
    ) -> float | pd.Series:
        """Compute Volume-Weighted Average Price.

        Args:
            prices: Price series.
            volumes: Volume series aligned with prices.
            window: Rolling window for rolling VWAP. If None, returns scalar
                VWAP over the full series.

        Returns:
            Scalar VWAP if window is None, else rolling VWAP Series.
        """
        if window is None:
            total_volume = volumes.sum()
            if total_volume == 0:
                return float(prices.mean())
            return float((prices * volumes).sum() / total_volume)

        pv = prices * volumes
        return pv.rolling(window).sum() / volumes.rolling(window).sum()

    def twap(
        self,
        prices: pd.Series,
        window: int | None = None,
    ) -> float | pd.Series:
        """Compute Time-Weighted Average Price.

        Args:
            prices: Price series.
            window: Rolling window. If None, returns scalar mean.

        Returns:
            Scalar TWAP or rolling TWAP Series.
        """
        if window is None:
            return float(prices.mean())
        return prices.rolling(window).mean()

    def arrival_price(self, prices: pd.Series) -> float:
        """Return the arrival (decision) price — first price in the series.

        Args:
            prices: Price series starting at trade decision time.

        Returns:
            First price in the series.
        """
        return float(prices.iloc[0])

    def implementation_shortfall(
        self,
        decision_price: float,
        fill_prices: pd.Series,
        shares: pd.Series,
    ) -> float:
        """Compute implementation shortfall relative to decision price.

        IS = Σ (fill_price_i - decision_price) * shares_i / Σ shares_i

        Args:
            decision_price: Price at the time of the trading decision.
            fill_prices: Prices at which fills were executed.
            shares: Number of shares for each fill.

        Returns:
            Implementation shortfall as a fraction of decision price.
        """
        total_shares = shares.sum()
        if total_shares == 0:
            return 0.0
        weighted_fill = (fill_prices * shares).sum() / total_shares
        return float((weighted_fill - decision_price) / (decision_price + 1e-12))

    def vwap_slippage(
        self,
        fill_price: float,
        vwap: float,
        direction: str = "buy",
    ) -> float:
        """Compute fill price slippage relative to VWAP benchmark.

        Positive slippage means the fill was worse than VWAP.

        Args:
            fill_price: Actual execution price.
            vwap: VWAP benchmark price.
            direction: "buy" or "sell". Defaults to "buy".

        Returns:
            Slippage as a fraction of VWAP (positive = worse than benchmark).
        """
        if vwap == 0:
            return 0.0
        slippage = (fill_price - vwap) / vwap
        if direction == "sell":
            slippage = -slippage
        return float(slippage)
