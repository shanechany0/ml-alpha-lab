"""Risk parity portfolio optimization."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy.optimize import minimize

logger = logging.getLogger(__name__)


class RiskParityOptimizer:
    """Risk parity (equal risk contribution) portfolio optimizer.

    Each asset contributes equally to total portfolio risk. Supports the
    classic ERC formulation as well as target risk contribution and
    inverse-volatility weighting.

    Attributes:
        max_iter: Maximum optimizer iterations.
        tol: Optimizer tolerance.
    """

    def __init__(self, config: dict | None = None) -> None:
        """Initialize RiskParityOptimizer.

        Args:
            config: Optional configuration dict with keys:
                - max_iter (int): Defaults to 1000.
                - tol (float): Defaults to 1e-9.
        """
        cfg = config or {}
        self.max_iter: int = cfg.get("max_iter", 1000)
        self.tol: float = cfg.get("tol", 1e-9)

    def optimize(self, covariance_matrix: pd.DataFrame) -> pd.Series:
        """Compute equal risk contribution (ERC) portfolio weights.

        Args:
            covariance_matrix: Asset covariance matrix.

        Returns:
            Portfolio weights indexed by asset names.
        """
        n = len(covariance_matrix)
        cov = covariance_matrix.values
        w0 = np.ones(n) / n
        bounds = [(1e-6, 1.0)] * n
        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]

        result = minimize(
            self._objective,
            w0,
            args=(cov,),
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"ftol": self.tol, "maxiter": self.max_iter},
        )

        weights = result.x / result.x.sum()
        return pd.Series(weights, index=covariance_matrix.index)

    def risk_contribution(
        self, weights: np.ndarray, covariance_matrix: np.ndarray
    ) -> np.ndarray:
        """Compute marginal risk contribution for each asset.

        Args:
            weights: Portfolio weight vector.
            covariance_matrix: Covariance matrix as numpy array.

        Returns:
            Array of risk contributions (as fraction of total portfolio risk).
        """
        sigma_w = covariance_matrix @ weights
        portfolio_vol = np.sqrt(weights @ sigma_w)
        if portfolio_vol < 1e-12:
            return np.ones(len(weights)) / len(weights)
        return weights * sigma_w / portfolio_vol

    def _objective(self, weights: np.ndarray, cov: np.ndarray) -> float:
        """Objective function: minimize sum of squared pairwise RC differences.

        Args:
            weights: Portfolio weight vector.
            cov: Covariance matrix as numpy array.

        Returns:
            Sum of squared pairwise differences in risk contributions.
        """
        rc = self.risk_contribution(weights, cov)
        rc_mean = rc.mean()
        return float(np.sum((rc - rc_mean) ** 2))

    def optimize_inverse_vol(self, volatilities: pd.Series) -> pd.Series:
        """Compute simple inverse-volatility weights.

        Args:
            volatilities: Per-asset volatility estimates.

        Returns:
            Normalized inverse-volatility portfolio weights.
        """
        inv_vol = 1.0 / (volatilities.values + 1e-12)
        weights = inv_vol / inv_vol.sum()
        return pd.Series(weights, index=volatilities.index)

    def target_risk_contribution(
        self,
        covariance_matrix: pd.DataFrame,
        target_contributions: pd.Series,
    ) -> pd.Series:
        """Find weights that achieve specified risk contributions.

        Args:
            covariance_matrix: Asset covariance matrix.
            target_contributions: Desired risk contributions (must sum to 1).

        Returns:
            Portfolio weights indexed by asset names.
        """
        n = len(covariance_matrix)
        cov = covariance_matrix.values
        targets = target_contributions.values
        targets = targets / targets.sum()

        def objective(weights: np.ndarray) -> float:
            rc = self.risk_contribution(weights, cov)
            return float(np.sum((rc - targets) ** 2))

        w0 = np.ones(n) / n
        bounds = [(1e-6, 1.0)] * n
        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]

        result = minimize(
            objective,
            w0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"ftol": self.tol, "maxiter": self.max_iter},
        )

        weights = result.x / result.x.sum()
        return pd.Series(weights, index=covariance_matrix.index)
