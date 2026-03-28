"""Mean-variance portfolio optimization using Modern Portfolio Theory."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize

logger = logging.getLogger(__name__)


class MeanVarianceOptimizer:
    """Mean-variance portfolio optimizer based on Markowitz framework.

    Supports efficient frontier computation, max-Sharpe, min-variance, and
    target-return optimization using scipy.optimize with linear and quadratic
    constraints.

    Attributes:
        risk_aversion: Risk aversion coefficient for utility maximization.
        max_weight: Maximum weight for any single asset.
        min_weight: Minimum weight for any single asset.
        target_volatility: Target portfolio volatility (annualized).
    """

    def __init__(self, config: dict | None = None) -> None:
        """Initialize MeanVarianceOptimizer.

        Args:
            config: Optional configuration dict with keys:
                - risk_aversion (float): Defaults to 1.0.
                - max_weight (float): Defaults to 1.0.
                - min_weight (float): Defaults to 0.0.
                - target_volatility (float | None): Defaults to None.
        """
        cfg = config or {}
        self.risk_aversion: float = cfg.get("risk_aversion", 1.0)
        self.max_weight: float = cfg.get("max_weight", 1.0)
        self.min_weight: float = cfg.get("min_weight", 0.0)
        self.target_volatility: float | None = cfg.get("target_volatility", None)

    def optimize(
        self,
        expected_returns: pd.Series,
        covariance_matrix: pd.DataFrame,
    ) -> pd.Series:
        """Optimize portfolio weights using mean-variance utility.

        Maximizes  w'μ - (λ/2) w'Σw  subject to sum(w)=1 and weight bounds.

        Args:
            expected_returns: Expected returns per asset.
            covariance_matrix: Asset covariance matrix.

        Returns:
            Optimal portfolio weights indexed by asset names.
        """
        n = len(expected_returns)
        mu = expected_returns.values
        sigma = covariance_matrix.values

        def neg_utility(w: np.ndarray) -> float:
            ret = w @ mu
            var = w @ sigma @ w
            return -(ret - 0.5 * self.risk_aversion * var)

        w0 = np.ones(n) / n
        bounds = [(self.min_weight, self.max_weight)] * n
        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]

        result = minimize(
            neg_utility,
            w0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"ftol": 1e-9, "maxiter": 1000},
        )

        weights = self._apply_constraints(result.x)
        return pd.Series(weights, index=expected_returns.index)

    def optimize_max_sharpe(
        self,
        expected_returns: pd.Series,
        covariance_matrix: pd.DataFrame,
        risk_free_rate: float = 0.0,
    ) -> pd.Series:
        """Find maximum Sharpe ratio portfolio weights.

        Args:
            expected_returns: Expected returns per asset.
            covariance_matrix: Asset covariance matrix.
            risk_free_rate: Risk-free rate (annualized). Defaults to 0.0.

        Returns:
            Portfolio weights that maximize the Sharpe ratio.
        """
        n = len(expected_returns)
        mu = expected_returns.values
        sigma = covariance_matrix.values

        def neg_sharpe(w: np.ndarray) -> float:
            ret = w @ mu - risk_free_rate
            vol = np.sqrt(w @ sigma @ w)
            return -ret / (vol + 1e-12)

        w0 = np.ones(n) / n
        bounds = [(self.min_weight, self.max_weight)] * n
        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]

        result = minimize(
            neg_sharpe,
            w0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"ftol": 1e-9, "maxiter": 1000},
        )

        weights = self._apply_constraints(result.x)
        return pd.Series(weights, index=expected_returns.index)

    def optimize_min_variance(
        self,
        covariance_matrix: pd.DataFrame,
    ) -> pd.Series:
        """Find the global minimum variance portfolio.

        Args:
            covariance_matrix: Asset covariance matrix.

        Returns:
            Portfolio weights minimizing portfolio variance.
        """
        n = len(covariance_matrix)
        sigma = covariance_matrix.values

        def portfolio_variance(w: np.ndarray) -> float:
            return w @ sigma @ w

        w0 = np.ones(n) / n
        bounds = [(self.min_weight, self.max_weight)] * n
        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]

        result = minimize(
            portfolio_variance,
            w0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"ftol": 1e-9, "maxiter": 1000},
        )

        weights = self._apply_constraints(result.x)
        return pd.Series(weights, index=covariance_matrix.index)

    def optimize_target_return(
        self,
        expected_returns: pd.Series,
        covariance_matrix: pd.DataFrame,
        target_return: float,
    ) -> pd.Series:
        """Find minimum variance portfolio achieving a target return.

        Args:
            expected_returns: Expected returns per asset.
            covariance_matrix: Asset covariance matrix.
            target_return: Required portfolio expected return.

        Returns:
            Portfolio weights minimizing variance subject to target return.
        """
        n = len(expected_returns)
        mu = expected_returns.values
        sigma = covariance_matrix.values

        def portfolio_variance(w: np.ndarray) -> float:
            return w @ sigma @ w

        w0 = np.ones(n) / n
        bounds = [(self.min_weight, self.max_weight)] * n
        constraints = [
            {"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
            {"type": "eq", "fun": lambda w: w @ mu - target_return},
        ]

        result = minimize(
            portfolio_variance,
            w0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"ftol": 1e-9, "maxiter": 1000},
        )

        weights = self._apply_constraints(result.x)
        return pd.Series(weights, index=expected_returns.index)

    def efficient_frontier(
        self,
        expected_returns: pd.Series,
        covariance_matrix: pd.DataFrame,
        n_points: int = 50,
    ) -> pd.DataFrame:
        """Compute the efficient frontier.

        Args:
            expected_returns: Expected returns per asset.
            covariance_matrix: Asset covariance matrix.
            n_points: Number of points on the frontier. Defaults to 50.

        Returns:
            DataFrame with columns: return, volatility, and one column per
            asset containing the optimal weights at each frontier point.
        """
        mu = expected_returns.values
        sigma = covariance_matrix.values

        min_ret = mu.min()
        max_ret = mu.max()
        target_returns = np.linspace(min_ret, max_ret, n_points)

        records: list[dict[str, Any]] = []
        for target in target_returns:
            try:
                weights = self.optimize_target_return(
                    expected_returns, covariance_matrix, target
                )
                vol = float(np.sqrt(weights.values @ sigma @ weights.values))
                row: dict[str, Any] = {"return": target, "volatility": vol}
                row.update(dict(zip(expected_returns.index, weights.values)))
                records.append(row)
            except Exception:
                logger.debug("Failed frontier point at target_return=%.4f", target)

        return pd.DataFrame(records)

    def _apply_constraints(self, weights: np.ndarray) -> np.ndarray:
        """Clip weights to [min_weight, max_weight] and renormalize.

        Args:
            weights: Raw weight array from optimizer.

        Returns:
            Clipped and renormalized weight array.
        """
        weights = np.clip(weights, self.min_weight, self.max_weight)
        total = weights.sum()
        if total > 0:
            weights /= total
        return weights
