"""Capital allocation across strategies and assets."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class CapitalAllocator:
    """Allocates capital across strategies using various sizing methodologies.

    Supports Kelly criterion, fractional Kelly, equal weighting, Sharpe-
    weighted, and dynamic rolling allocation schemes.

    Attributes:
        max_kelly: Upper bound on Kelly fraction to prevent over-sizing.
    """

    def __init__(self, config: dict | None = None) -> None:
        """Initialize CapitalAllocator.

        Args:
            config: Optional configuration dict with keys:
                - max_kelly (float): Cap on Kelly fraction. Defaults to 1.0.
        """
        cfg = config or {}
        self.max_kelly: float = cfg.get("max_kelly", 1.0)

    def kelly_fraction(self, expected_return: float, variance: float) -> float:
        """Compute full Kelly fraction.

        Uses the formula:  f* = μ / σ²

        Args:
            expected_return: Expected strategy return.
            variance: Variance of strategy returns.

        Returns:
            Kelly fraction clipped to [0, max_kelly].
        """
        if variance <= 0:
            return 0.0
        fraction = expected_return / variance
        return float(np.clip(fraction, 0.0, self.max_kelly))

    def fractional_kelly(
        self,
        expected_return: float,
        variance: float,
        fraction: float = 0.5,
    ) -> float:
        """Compute fractional Kelly position size.

        Args:
            expected_return: Expected strategy return.
            variance: Variance of strategy returns.
            fraction: Fraction of full Kelly to use. Defaults to 0.5.

        Returns:
            Fractional Kelly position size.
        """
        return fraction * self.kelly_fraction(expected_return, variance)

    def allocate_strategies(
        self,
        strategy_returns: pd.DataFrame,
        method: str = "kelly",
    ) -> pd.Series:
        """Allocate capital weights across strategies.

        Args:
            strategy_returns: DataFrame with strategy returns as columns.
            method: Allocation method — one of "kelly", "equal", "sharpe".

        Returns:
            Normalized allocation weights indexed by strategy names.

        Raises:
            ValueError: If an unsupported method is specified.
        """
        if method == "kelly":
            weights = {}
            for col in strategy_returns.columns:
                r = strategy_returns[col].dropna()
                mu = float(r.mean())
                var = float(r.var())
                weights[col] = self.kelly_fraction(mu, var)
            w = pd.Series(weights)
        elif method == "equal":
            w = self.equal_weight(len(strategy_returns.columns))
            w.index = strategy_returns.columns
        elif method == "sharpe":
            w = self.sharpe_weighted(strategy_returns)
        else:
            raise ValueError(f"Unsupported allocation method: {method}")

        total = w.sum()
        if total > 0:
            w = w / total
        return w

    def equal_weight(self, n_strategies: int) -> pd.Series:
        """Produce equal-weight allocation.

        Args:
            n_strategies: Number of strategies.

        Returns:
            Equal weights of length n_strategies.
        """
        weights = np.ones(n_strategies) / n_strategies
        return pd.Series(weights, name="weight")

    def sharpe_weighted(self, strategy_returns: pd.DataFrame) -> pd.Series:
        """Weight strategies proportionally to their Sharpe ratios.

        Strategies with non-positive Sharpe receive zero weight.

        Args:
            strategy_returns: DataFrame with strategy returns as columns.

        Returns:
            Normalized Sharpe-weighted allocation.
        """
        sharpes = {}
        for col in strategy_returns.columns:
            r = strategy_returns[col].dropna()
            std = float(r.std())
            if std > 0:
                sharpes[col] = max(0.0, float(r.mean()) / std)
            else:
                sharpes[col] = 0.0

        w = pd.Series(sharpes)
        total = w.sum()
        if total > 0:
            w = w / total
        return w

    def dynamic_allocation(
        self,
        strategy_returns: pd.DataFrame,
        lookback: int = 63,
    ) -> pd.DataFrame:
        """Compute rolling capital allocation weights over time.

        Args:
            strategy_returns: DataFrame with strategy returns as columns,
                indexed by date.
            lookback: Rolling window length in periods. Defaults to 63.

        Returns:
            DataFrame of the same shape as strategy_returns containing
            rolling allocation weights.
        """
        allocations = pd.DataFrame(
            index=strategy_returns.index,
            columns=strategy_returns.columns,
            dtype=float,
        )

        for i in range(lookback, len(strategy_returns)):
            window = strategy_returns.iloc[i - lookback : i]
            sharpes = {}
            for col in window.columns:
                r = window[col].dropna()
                std = float(r.std())
                sharpes[col] = max(0.0, float(r.mean()) / std) if std > 0 else 0.0

            w = pd.Series(sharpes)
            total = w.sum()
            if total > 0:
                w = w / total
            else:
                w = pd.Series(
                    np.ones(len(w)) / len(w), index=w.index
                )
            allocations.iloc[i] = w.values

        allocations.iloc[:lookback] = np.nan
        return allocations
