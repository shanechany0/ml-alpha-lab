"""Black-Litterman portfolio model."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.portfolio.mean_variance import MeanVarianceOptimizer

logger = logging.getLogger(__name__)


class BlackLittermanModel:
    """Black-Litterman model for incorporating investor views into optimization.

    Combines equilibrium market returns with explicit investor views to produce
    posterior expected returns, which are then used in mean-variance optimization.

    Attributes:
        tau: Uncertainty scaling parameter (typically 0.025–0.10).
    """

    def __init__(self, tau: float = 0.05, config: dict | None = None) -> None:
        """Initialize BlackLittermanModel.

        Args:
            tau: Uncertainty in prior (scales prior covariance). Defaults to 0.05.
            config: Optional configuration dict forwarded to MeanVarianceOptimizer.
        """
        self.tau = tau
        self._mv_optimizer = MeanVarianceOptimizer(config)

    def compute_equilibrium_returns(
        self,
        market_weights: pd.Series,
        covariance_matrix: pd.DataFrame,
        risk_aversion: float = 2.5,
    ) -> pd.Series:
        """Compute CAPM implied equilibrium excess returns.

        Uses the reverse optimization formula:  π = λ · Σ · w_mkt

        Args:
            market_weights: Market-capitalization weights.
            covariance_matrix: Asset covariance matrix.
            risk_aversion: Market risk aversion coefficient λ. Defaults to 2.5.

        Returns:
            Equilibrium expected excess returns per asset.
        """
        pi = risk_aversion * covariance_matrix.values @ market_weights.values
        return pd.Series(pi, index=market_weights.index)

    def incorporate_views(
        self,
        equilibrium_returns: pd.Series,
        covariance_matrix: pd.DataFrame,
        P: np.ndarray,
        Q: np.ndarray,
        omega: np.ndarray | None = None,
    ) -> pd.Series:
        """Compute Black-Litterman posterior expected returns.

        Posterior:  μ_BL = [(τΣ)⁻¹ + P'Ω⁻¹P]⁻¹ [(τΣ)⁻¹π + P'Ω⁻¹Q]

        Args:
            equilibrium_returns: CAPM equilibrium returns π.
            covariance_matrix: Asset covariance matrix Σ.
            P: Views pick matrix of shape (k, n).
            Q: Views return vector of shape (k,).
            omega: View uncertainty matrix (k × k). If None, defaults to
                proportional to P Σ P'.

        Returns:
            Posterior expected returns per asset.
        """
        sigma = covariance_matrix.values
        pi = equilibrium_returns.values
        tau_sigma = self.tau * sigma

        if omega is None:
            omega = np.diag(np.diag(P @ tau_sigma @ P.T))

        tau_sigma_inv = np.linalg.inv(tau_sigma)
        omega_inv = np.linalg.inv(omega)

        M = tau_sigma_inv + P.T @ omega_inv @ P
        M_inv = np.linalg.inv(M)

        mu_bl = M_inv @ (tau_sigma_inv @ pi + P.T @ omega_inv @ Q)
        return pd.Series(mu_bl, index=equilibrium_returns.index)

    def optimize(
        self,
        equilibrium_returns: pd.Series,
        covariance_matrix: pd.DataFrame,
        views: dict[str, float] | None = None,
    ) -> pd.Series:
        """Full Black-Litterman optimization pipeline.

        Incorporates views (if any) into equilibrium returns and performs
        mean-variance optimization on the posterior.

        Args:
            equilibrium_returns: CAPM equilibrium excess returns.
            covariance_matrix: Asset covariance matrix.
            views: Optional dict mapping asset name to expected view return.

        Returns:
            Optimal portfolio weights.
        """
        if views:
            assets = equilibrium_returns.index.tolist()
            view_assets = [a for a in views if a in assets]
            k = len(view_assets)
            if k > 0:
                P = np.zeros((k, len(assets)))
                Q = np.zeros(k)
                for i, asset in enumerate(view_assets):
                    P[i, assets.index(asset)] = 1.0
                    Q[i] = views[asset]
                posterior = self.incorporate_views(
                    equilibrium_returns, covariance_matrix, P, Q
                )
            else:
                posterior = equilibrium_returns
        else:
            posterior = equilibrium_returns

        return self._mv_optimizer.optimize(posterior, covariance_matrix)

    def views_from_ml_signals(
        self,
        signals: pd.DataFrame,
        confidence: float = 0.1,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Convert ML alpha signals to Black-Litterman P and Q matrices.

        Each asset with a non-zero signal gets an absolute view.  The view
        magnitude is scaled by the signal value; ``confidence`` scales Ω.

        Args:
            signals: DataFrame of shape (1, n_assets) or (n_assets,) with
                signal values.
            confidence: Confidence scaling for the view uncertainty.

        Returns:
            Tuple of (P, Q) where P is the views pick matrix and Q is the
            views vector.
        """
        if isinstance(signals, pd.DataFrame):
            sig = signals.iloc[-1] if len(signals) > 1 else signals.iloc[0]
        else:
            sig = signals

        non_zero = sig[sig != 0]
        k = len(non_zero)
        n = len(sig)
        assets = sig.index.tolist()

        P = np.zeros((k, n))
        Q = np.zeros(k)
        for i, (asset, val) in enumerate(non_zero.items()):
            P[i, assets.index(asset)] = 1.0
            Q[i] = float(val) * confidence

        return P, Q
