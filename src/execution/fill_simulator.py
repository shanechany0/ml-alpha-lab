"""Trade fill simulation including market impact and partial fills."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class FillSimulator:
    """Simulates trade fills with market impact, partial fills, and VWAP execution.

    Models realistic execution costs including the square-root market impact
    model, bid-ask spread costs, and volume-participation-based partial fills.

    Attributes:
        impact_coeff: Linear market impact coefficient.
        spread_bps: Half-spread in basis points.
        partial_fill_prob: Probability that an order is only partially filled.
    """

    def __init__(self, config: dict | None = None) -> None:
        """Initialize FillSimulator.

        Args:
            config: Optional configuration dict with keys:
                - impact_coeff (float): Square-root impact coefficient.
                    Defaults to 0.1.
                - spread_bps (float): Half-spread in bps. Defaults to 5.0.
                - partial_fill_prob (float): Probability of partial fill.
                    Defaults to 0.1.
        """
        cfg = config or {}
        self.impact_coeff: float = cfg.get("impact_coeff", 0.1)
        self.spread_bps: float = cfg.get("spread_bps", 5.0)
        self.partial_fill_prob: float = cfg.get("partial_fill_prob", 0.1)

    def simulate_fill(
        self,
        order_size: float,
        adv: float,
        price: float,
        volatility: float,
    ) -> dict[str, float]:
        """Simulate a single order fill with market impact.

        Uses a square-root market impact model:
            impact = σ · η · sqrt(order_size / ADV)

        Args:
            order_size: Order size in shares.
            adv: Average daily volume in shares.
            price: Current mid price.
            volatility: Daily return volatility (as decimal).

        Returns:
            Dict with keys: filled_size, fill_price, market_impact, spread_cost,
            total_cost.
        """
        participation = order_size / (adv + 1e-12)
        market_impact = (
            self.impact_coeff * volatility * np.sqrt(participation) * price
        )
        spread_cost = self.spread_bps * 1e-4 * price
        fill_price = price + market_impact + spread_cost
        total_cost = (fill_price - price) * order_size

        return {
            "filled_size": order_size,
            "fill_price": float(fill_price),
            "market_impact": float(market_impact),
            "spread_cost": float(spread_cost),
            "total_cost": float(total_cost),
        }

    def simulate_partial_fill(
        self,
        order_size: float,
        market_depth: float,
    ) -> float:
        """Simulate a partial fill based on available market depth.

        Args:
            order_size: Requested order size.
            market_depth: Available liquidity at current price level.

        Returns:
            Actual filled order size (≤ order_size).
        """
        if market_depth <= 0:
            return 0.0
        fill_ratio = min(1.0, market_depth / order_size)
        if np.random.random() < self.partial_fill_prob:
            fill_ratio *= np.random.uniform(0.5, 1.0)
        return float(order_size * fill_ratio)

    def simulate_vwap_fill(
        self,
        order_size: float,
        volume_profile: pd.Series,
        prices: pd.Series,
    ) -> dict[str, float]:
        """Simulate VWAP execution by distributing the order over the day.

        Args:
            order_size: Total order size to execute.
            volume_profile: Intraday volume weights (should sum to 1).
            prices: Intraday prices aligned with volume_profile.

        Returns:
            Dict with keys: filled_size, vwap_price, slippage.
        """
        vol_weights = volume_profile / (volume_profile.sum() + 1e-12)
        slice_sizes = vol_weights * order_size
        vwap_price = float((prices * vol_weights).sum())
        arrival_price = float(prices.iloc[0])
        slippage = (vwap_price - arrival_price) / (arrival_price + 1e-12)

        return {
            "filled_size": float(slice_sizes.sum()),
            "vwap_price": vwap_price,
            "slippage": float(slippage),
        }

    def simulate_portfolio_execution(
        self,
        target_weights: pd.Series,
        current_weights: pd.Series,
        prices: pd.Series,
        adv: pd.Series,
        volatility: pd.Series | None = None,
    ) -> pd.DataFrame:
        """Simulate execution of a full portfolio rebalance.

        Args:
            target_weights: Desired portfolio weights.
            current_weights: Current portfolio weights.
            prices: Current prices per asset.
            adv: Average daily volumes per asset.
            volatility: Per-asset daily volatilities. If None, defaults to 0.02.

        Returns:
            DataFrame with columns: order_size, fill_price, market_impact,
            spread_cost, total_cost, fill_ratio indexed by asset.
        """
        if volatility is None:
            volatility = pd.Series(0.02, index=prices.index)

        delta_weights = target_weights - current_weights
        records = []

        for asset in delta_weights.index:
            if abs(delta_weights[asset]) < 1e-6:
                records.append(
                    {
                        "asset": asset,
                        "order_size": 0.0,
                        "fill_price": float(prices.get(asset, 0.0)),
                        "market_impact": 0.0,
                        "spread_cost": 0.0,
                        "total_cost": 0.0,
                        "fill_ratio": 1.0,
                    }
                )
                continue

            order_size = abs(float(delta_weights[asset])) * 1e6  # notional
            fill = self.simulate_fill(
                order_size=order_size,
                adv=float(adv.get(asset, 1e6)),
                price=float(prices.get(asset, 1.0)),
                volatility=float(volatility.get(asset, 0.02)),
            )
            records.append(
                {
                    "asset": asset,
                    "order_size": order_size,
                    **fill,
                    "fill_ratio": 1.0,
                }
            )

        df = pd.DataFrame(records).set_index("asset")
        return df
