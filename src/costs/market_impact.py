"""Market impact models: Almgren-Chriss and square-root impact."""

from __future__ import annotations

import numpy as np
import pandas as pd


class AlmgrenChrissModel:
    """Implements the Almgren-Chriss (2000) optimal execution model.

    Models both temporary and permanent market impact for a single
    asset traded over a finite horizon.

    Attributes:
        eta: Temporary impact coefficient (ε in A-C notation).
        gamma: Permanent impact coefficient (γ).
        sigma: Asset daily volatility.
        tau: Single trading period duration in days.
    """

    def __init__(
        self,
        eta: float = 0.1,
        gamma: float = 0.1,
        sigma: float = 0.3,
        tau: float = 1.0,
    ) -> None:
        """Initializes AlmgrenChrissModel.

        Args:
            eta: Temporary impact coefficient.
            gamma: Permanent impact coefficient.
            sigma: Annualized asset volatility.
            tau: Period length in days.
        """
        self.eta = eta
        self.gamma = gamma
        self.sigma = sigma
        self.tau = tau

    def temporary_impact(self, trade_rate: float, adv: float) -> float:
        """Computes temporary (instantaneous) price impact.

        Uses the model: ε * σ * sqrt(v / V) where v is trade rate
        and V is average daily volume.

        Args:
            trade_rate: Shares (or units) traded per period.
            adv: Average daily volume in the same units.

        Returns:
            Temporary impact as a fraction of price.
        """
        if adv <= 0:
            return 0.0
        participation = abs(trade_rate) / adv
        return self.eta * self.sigma * np.sqrt(participation)

    def permanent_impact(self, total_shares: float, adv: float) -> float:
        """Computes permanent (lasting) price impact from total position.

        Uses the model: γ * σ * (Q / V)

        Args:
            total_shares: Total shares to be traded.
            adv: Average daily volume.

        Returns:
            Permanent impact as a fraction of price.
        """
        if adv <= 0:
            return 0.0
        return self.gamma * self.sigma * abs(total_shares) / adv

    def total_impact(self, trade_rate: float, total_shares: float, adv: float) -> float:
        """Computes combined temporary and permanent market impact.

        Args:
            trade_rate: Shares traded per period.
            total_shares: Total position to unwind.
            adv: Average daily volume.

        Returns:
            Total impact as a fraction of price.
        """
        return self.temporary_impact(trade_rate, adv) + self.permanent_impact(total_shares, adv)

    def optimal_trajectory(
        self, total_shares: float, adv: float, n_periods: int = 10
    ) -> np.ndarray:
        """Computes the optimal linear-rate execution trajectory.

        Minimizes total expected cost (temporary impact + risk) using the
        Almgren-Chriss result: optimal strategy is to trade at a constant
        rate when ignoring risk aversion.

        Args:
            total_shares: Total shares to execute.
            adv: Average daily volume.
            n_periods: Number of trading periods.

        Returns:
            Array of remaining inventory at each period (length n_periods + 1).
        """
        if n_periods <= 0:
            return np.array([total_shares, 0.0])

        kappa = np.sqrt(self.gamma / (2 * self.eta)) if self.eta > 0 else 1.0
        t = np.linspace(0, 1, n_periods + 1)

        if kappa == 0:
            trajectory = total_shares * (1 - t)
        else:
            trajectory = total_shares * np.sinh(kappa * (1 - t)) / np.sinh(kappa)

        return trajectory


class SquareRootImpactModel:
    """Simple square-root market impact model (Grinold & Kahn style).

    Attributes:
        coefficient: Impact scaling coefficient.
    """

    def __init__(self, coefficient: float = 0.1) -> None:
        """Initializes SquareRootImpactModel.

        Args:
            coefficient: Scaling coefficient for the impact formula.
        """
        self.coefficient = coefficient

    def impact(self, trade_size: float, adv: float, volatility: float) -> float:
        """Estimates market impact using the square-root model.

        Uses: h = coefficient * sigma * sqrt(|Q| / V)

        Args:
            trade_size: Size of the trade (shares or notional).
            adv: Average daily volume in same units as trade_size.
            volatility: Daily return volatility of the asset.

        Returns:
            Estimated market impact as a fraction of price.
        """
        if adv <= 0:
            return 0.0
        return self.coefficient * volatility * np.sqrt(abs(trade_size) / adv)

    def expected_cost(
        self,
        signals: pd.DataFrame,
        adv_data: pd.DataFrame,
        volatility: pd.DataFrame,
    ) -> pd.Series:
        """Computes expected market impact costs per period across all assets.

        Args:
            signals: Position weight DataFrame (dates × assets).
            adv_data: Average daily volume DataFrame (dates × assets).
            volatility: Daily volatility DataFrame (dates × assets).

        Returns:
            Per-period total expected impact cost series.
        """
        aligned = signals.align(adv_data, join="inner")[0]
        aligned, vol_aligned = aligned.align(volatility, join="inner")
        _, adv_aligned = signals.align(adv_data, join="inner")
        adv_aligned = adv_aligned.reindex(aligned.index)

        turnover = aligned.diff().abs().fillna(0)
        total_cost = pd.Series(0.0, index=aligned.index)

        for asset in aligned.columns:
            if asset not in adv_aligned.columns or asset not in vol_aligned.columns:
                continue
            for date in aligned.index:
                t_size = turnover.loc[date, asset] if asset in turnover.columns else 0.0
                adv = adv_aligned.loc[date, asset] if not pd.isna(adv_aligned.loc[date, asset]) else 1.0
                vol = vol_aligned.loc[date, asset] if not pd.isna(vol_aligned.loc[date, asset]) else 0.01
                total_cost[date] += self.impact(t_size, adv, vol)

        return total_cost.rename("expected_impact_cost")
