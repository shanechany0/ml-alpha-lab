"""Portfolio risk control and limit enforcement."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import norm

logger = logging.getLogger(__name__)


class RiskController:
    """Applies risk controls and limits to portfolio weights.

    Supports position limits, drawdown circuit breakers, sector
    concentration limits, VaR checks, and consolidated risk reporting.

    Attributes:
        position_limit: Default maximum weight per asset.
        max_drawdown_limit: Drawdown threshold that triggers circuit breaker.
        sector_limit: Maximum aggregate weight per sector.
        var_limit: Maximum allowable portfolio VaR (as fraction).
    """

    def __init__(self, config: dict | None = None) -> None:
        """Initialize RiskController.

        Args:
            config: Optional configuration dict with keys:
                - position_limit (float): Defaults to 0.1.
                - max_drawdown_limit (float): Defaults to 0.15.
                - sector_limit (float): Defaults to 0.3.
                - var_limit (float): Defaults to 0.05.
        """
        cfg = config or {}
        self.position_limit: float = cfg.get("position_limit", 0.1)
        self.max_drawdown_limit: float = cfg.get("max_drawdown_limit", 0.15)
        self.sector_limit: float = cfg.get("sector_limit", 0.3)
        self.var_limit: float = cfg.get("var_limit", 0.05)

    def apply_position_limits(
        self,
        weights: pd.Series,
        max_weight: float = 0.1,
    ) -> pd.Series:
        """Clip individual asset weights to a maximum and renormalize.

        Uses an iterative water-filling algorithm so that after renormalization
        all weights are guaranteed to satisfy the cap exactly.

        Args:
            weights: Portfolio weights indexed by asset names.
            max_weight: Maximum allowable weight per asset. Defaults to 0.1.

        Returns:
            Weight-capped and renormalized portfolio weights.
        """
        w = weights.copy().astype(float)
        total = w.sum()
        if total <= 0:
            return w
        w = w / total  # normalize to sum to 1

        capped = pd.Series(False, index=w.index)
        for _ in range(len(w) + 1):
            n_capped = int(capped.sum())
            free = ~capped
            free_sum = w[free].sum()
            if free_sum <= 0:
                break
            free_budget = 1.0 - n_capped * max_weight
            # Renormalized weights for uncapped assets
            renorm = w[free] / free_sum * free_budget
            newly_capped = renorm > max_weight
            if not newly_capped.any():
                w[free] = renorm
                break
            capped[free] = capped[free] | newly_capped

        w[capped] = max_weight
        return w

    def check_drawdown_circuit_breaker(
        self,
        returns: pd.Series,
        threshold: float = 0.15,
    ) -> bool:
        """Check whether current drawdown exceeds the halt threshold.

        Args:
            returns: Time series of portfolio returns.
            threshold: Maximum tolerable drawdown. Defaults to 0.15.

        Returns:
            True if trading should halt (drawdown >= threshold), else False.
        """
        cumulative = (1 + returns).cumprod()
        rolling_max = cumulative.cummax()
        drawdown = (cumulative - rolling_max) / rolling_max
        current_dd = float(drawdown.iloc[-1])
        return current_dd <= -threshold

    def apply_sector_limits(
        self,
        weights: pd.Series,
        sector_map: dict[str, str],
        max_sector: float = 0.3,
    ) -> pd.Series:
        """Reduce sector concentration to comply with per-sector cap.

        Iteratively scales down over-weight sectors and renormalizes.

        Args:
            weights: Portfolio weights indexed by asset names.
            sector_map: Mapping from asset name to sector label.
            max_sector: Maximum aggregate weight per sector. Defaults to 0.3.

        Returns:
            Sector-constrained and renormalized portfolio weights.
        """
        adjusted = weights.copy().astype(float)

        sectors: dict[str, list[str]] = {}
        for asset, sector in sector_map.items():
            if asset in adjusted.index:
                sectors.setdefault(sector, []).append(asset)

        for sector, assets in sectors.items():
            sector_weight = adjusted[assets].sum()
            if sector_weight > max_sector:
                scale = max_sector / sector_weight
                adjusted[assets] *= scale

        total = adjusted.sum()
        if total > 0:
            adjusted = adjusted / total
        return adjusted

    def compute_portfolio_var(
        self,
        weights: pd.Series,
        covariance_matrix: pd.DataFrame,
        confidence: float = 0.95,
    ) -> float:
        """Compute parametric portfolio Value-at-Risk.

        Args:
            weights: Portfolio weights indexed by asset names.
            covariance_matrix: Asset covariance matrix.
            confidence: VaR confidence level. Defaults to 0.95.

        Returns:
            One-period VaR as a positive loss fraction.
        """
        w = weights.values
        sigma = covariance_matrix.values
        port_vol = float(np.sqrt(w @ sigma @ w))
        z = norm.ppf(confidence)
        return float(z * port_vol)

    def apply_all_controls(
        self,
        weights: pd.Series,
        returns_history: pd.Series,
        sector_map: dict[str, str] | None = None,
    ) -> pd.Series:
        """Apply position, sector, and circuit-breaker controls sequentially.

        Args:
            weights: Raw portfolio weights.
            returns_history: Historical portfolio returns for drawdown check.
            sector_map: Optional mapping from asset to sector.

        Returns:
            Risk-controlled portfolio weights, or zero weights if circuit
            breaker is triggered.
        """
        if self.check_drawdown_circuit_breaker(
            returns_history, self.max_drawdown_limit
        ):
            logger.warning("Drawdown circuit breaker triggered — zeroing weights.")
            return pd.Series(0.0, index=weights.index)

        controlled = self.apply_position_limits(weights, self.position_limit)

        if sector_map is not None:
            controlled = self.apply_sector_limits(
                controlled, sector_map, self.sector_limit
            )

        return controlled

    def generate_risk_report(
        self,
        weights: pd.Series,
        returns_history: pd.Series,
    ) -> dict[str, Any]:
        """Generate a summary risk report for the portfolio.

        Args:
            weights: Current portfolio weights.
            returns_history: Historical portfolio returns.

        Returns:
            Dict containing: max_weight, n_positions, hhi (Herfindahl),
            current_drawdown, circuit_breaker_triggered, annualized_vol.
        """
        cumulative = (1 + returns_history).cumprod()
        rolling_max = cumulative.cummax()
        drawdown = (cumulative - rolling_max) / rolling_max

        ann_vol = float(returns_history.std() * np.sqrt(252))
        max_dd = float(drawdown.min())
        current_dd = float(drawdown.iloc[-1])
        hhi = float((weights ** 2).sum())

        return {
            "max_weight": float(weights.max()),
            "n_positions": int((weights > 1e-6).sum()),
            "hhi": hhi,
            "current_drawdown": current_dd,
            "max_drawdown": max_dd,
            "circuit_breaker_triggered": current_dd <= -self.max_drawdown_limit,
            "annualized_vol": ann_vol,
        }
