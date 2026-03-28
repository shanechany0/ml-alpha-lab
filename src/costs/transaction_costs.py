"""Transaction cost models for strategy evaluation."""

from __future__ import annotations

import pandas as pd


class TransactionCostModel:
    """Models explicit transaction costs: spread, commission, and slippage.

    Attributes:
        config: Configuration dictionary.
        default_spread_bps: Default bid-ask spread in basis points.
        default_commission_bps: Default commission in basis points.
    """

    def __init__(self, config: dict | None = None) -> None:
        """Initializes TransactionCostModel.

        Args:
            config: Optional configuration. Recognized keys:
                'spread_bps', 'commission_bps', 'slippage_factor'.
        """
        self.config = config or {}
        self.default_spread_bps: float = self.config.get("spread_bps", 5.0)
        self.default_commission_bps: float = self.config.get("commission_bps", 10.0)
        self.slippage_factor: float = self.config.get("slippage_factor", 0.1)

    def compute_spread_cost(
        self, trade_value: float | pd.Series, spread_bps: float
    ) -> float | pd.Series:
        """Computes the bid-ask spread cost for a given trade value.

        Args:
            trade_value: Trade value in dollars (scalar or Series).
            spread_bps: Bid-ask spread in basis points.

        Returns:
            Spread cost in dollars (same type as trade_value).
        """
        return trade_value * spread_bps / 10_000 / 2

    def compute_commission(
        self, trade_value: float | pd.Series, commission_bps: float
    ) -> float | pd.Series:
        """Computes brokerage commission for a given trade value.

        Args:
            trade_value: Trade value in dollars.
            commission_bps: Commission rate in basis points.

        Returns:
            Commission in dollars.
        """
        return trade_value * commission_bps / 10_000

    def compute_slippage(
        self,
        trade_size: float | pd.Series,
        volatility: float | pd.Series,
        slippage_factor: float = 0.1,
    ) -> float | pd.Series:
        """Estimates slippage cost proportional to volatility and trade size.

        Uses the model: slippage = factor * volatility * trade_size

        Args:
            trade_size: Normalized trade size (fraction of portfolio).
            volatility: Asset daily volatility.
            slippage_factor: Linear slippage coefficient.

        Returns:
            Slippage cost (same type as inputs).
        """
        return slippage_factor * volatility * trade_size

    def total_cost(
        self,
        signals: pd.DataFrame,
        prices: pd.DataFrame,
        spread_bps: float = 5.0,
        commission_bps: float = 10.0,
    ) -> pd.Series:
        """Computes total transaction costs per period across all assets.

        Args:
            signals: Position weight DataFrame (dates × assets).
            prices: Asset price DataFrame (dates × assets).
            spread_bps: Bid-ask spread in basis points.
            commission_bps: Commission in basis points.

        Returns:
            Per-period total cost series (in fractional return units).
        """
        aligned_signals, aligned_prices = signals.align(prices, join="inner")
        turnover = aligned_signals.diff().abs().fillna(0)

        # Costs are applied to the dollar value of each trade (weight × price)
        trade_value = turnover * aligned_prices
        spread_cost = self.compute_spread_cost(trade_value, spread_bps)
        commission = self.compute_commission(trade_value, commission_bps)
        total_dollar_cost = (spread_cost + commission).sum(axis=1)

        # Normalise by total portfolio value (sum of |weights| × prices)
        portfolio_value = (aligned_signals.abs() * aligned_prices).sum(axis=1).replace(0, 1)
        return (total_dollar_cost / portfolio_value).rename("total_cost")

    def cost_per_trade(
        self,
        trade_value: float,
        spread_bps: float,
        commission_bps: float,
        slippage_bps: float,
    ) -> dict[str, float]:
        """Decomposes the total cost of a single trade into components.

        Args:
            trade_value: Trade value in dollars.
            spread_bps: Spread cost in basis points.
            commission_bps: Commission in basis points.
            slippage_bps: Slippage cost in basis points.

        Returns:
            Dictionary with keys 'spread', 'commission', 'slippage', 'total'
            all in dollar terms.
        """
        spread = float(self.compute_spread_cost(trade_value, spread_bps))
        commission = float(self.compute_commission(trade_value, commission_bps))
        slippage = float(trade_value * slippage_bps / 10_000)
        return {
            "spread": spread,
            "commission": commission,
            "slippage": slippage,
            "total": spread + commission + slippage,
        }

    def apply_costs_to_returns(
        self, strategy_returns: pd.Series, turnover: pd.Series, cost_bps: float
    ) -> pd.Series:
        """Subtracts transaction costs from strategy returns.

        Args:
            strategy_returns: Gross strategy daily returns.
            turnover: Per-period portfolio turnover (sum of |Δw|).
            cost_bps: One-way cost in basis points applied per unit of turnover.

        Returns:
            Net strategy returns after transaction costs.
        """
        aligned_returns, aligned_turnover = strategy_returns.align(turnover, join="inner")
        costs = aligned_turnover * cost_bps / 10_000
        return (aligned_returns - costs).rename("net_returns")
