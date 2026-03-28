"""Deflated Sharpe Ratio (DSR) implementation (Bailey & de Prado, 2014)."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy.stats import norm

logger = logging.getLogger(__name__)


class DeflatedSharpeRatio:
    """Computes the Deflated Sharpe Ratio to correct for selection bias.

    The DSR adjusts the observed Sharpe ratio for the number of strategy
    trials and the statistical properties of the backtest, following
    Bailey & de Prado (2014).
    """

    def __init__(self) -> None:
        """Initialize DeflatedSharpeRatio."""

    def compute(
        self,
        returns: pd.Series,
        n_trials: int,
        annualization: int = 252,
    ) -> dict[str, float]:
        """Compute the Deflated Sharpe Ratio.

        Args:
            returns: Strategy daily return series.
            n_trials: Number of strategy configurations / backtests evaluated.
            annualization: Trading days per year. Defaults to 252.

        Returns:
            Dict with keys: observed_sr, expected_max_sr, dsr, p_value,
            annualized_sr.
        """
        r = returns.dropna()
        n = len(r)
        sr_daily = float(r.mean() / (r.std() + 1e-12))
        ann_sr = sr_daily * np.sqrt(annualization)
        sr_std = float(
            np.sqrt((1.0 + 0.5 * sr_daily ** 2) / n)
        )

        e_max_sr = self.expected_max_sharpe(n_trials, n, sr_std)
        dsr = self.deflated_sr(sr_daily, n_trials, n)
        p_value = float(norm.cdf(dsr))

        return {
            "observed_sr": ann_sr,
            "expected_max_sr": float(e_max_sr * np.sqrt(annualization)),
            "dsr": float(dsr),
            "p_value": p_value,
            "annualized_sr": ann_sr,
        }

    def expected_max_sharpe(
        self,
        n_trials: int,
        n_obs: int,
        sr_std: float = 1.0,
    ) -> float:
        """Estimate the expected maximum Sharpe ratio under the null hypothesis.

        Uses the expected maximum of n_trials i.i.d. normal variates.

        Args:
            n_trials: Number of independent trials.
            n_obs: Number of observations per trial.
            sr_std: Standard deviation of Sharpe ratios under null. Defaults to 1.0.

        Returns:
            Expected maximum Sharpe ratio under null.
        """
        if n_trials <= 1:
            return 0.0
        euler_mascheroni = 0.5772156649
        expected_max = (
            (1 - euler_mascheroni) * norm.ppf(1 - 1.0 / n_trials)
            + euler_mascheroni * norm.ppf(1 - 1.0 / (n_trials * np.e))
        )
        return float(expected_max * sr_std)

    def deflated_sr(
        self,
        observed_sr: float,
        n_trials: int,
        n_obs: int,
    ) -> float:
        """Compute the DSR test statistic (z-score after deflation).

        Args:
            observed_sr: Observed Sharpe ratio (daily scale).
            n_trials: Number of trials / configurations tested.
            n_obs: Number of observations.

        Returns:
            DSR test statistic.
        """
        sr_std = float(np.sqrt((1.0 + 0.5 * observed_sr ** 2) / n_obs))
        e_max_sr = self.expected_max_sharpe(n_trials, n_obs, sr_std)
        dsr = (observed_sr - e_max_sr) / (sr_std + 1e-12)
        return float(dsr)

    def probability_overfitting(
        self,
        is_sharpes: list[float],
        oos_returns: pd.Series,
    ) -> float:
        """Estimate probability of overfitting via combinatorial selection.

        Computes the fraction of in-sample Sharpe ratios that exceed the
        median OOS Sharpe ratio, as a simple overfitting diagnostic.

        Args:
            is_sharpes: List of in-sample Sharpe ratios across all trials.
            oos_returns: Out-of-sample returns for the best IS strategy.

        Returns:
            Probability of overfitting in [0, 1].
        """
        if not is_sharpes:
            return 0.0

        r_oos = oos_returns.dropna()
        oos_sr = float(r_oos.mean() / (r_oos.std() + 1e-12)) * np.sqrt(252)
        np.median(is_sharpes)

        n_overfit = sum(1 for sr in is_sharpes if sr > oos_sr)
        return float(n_overfit / len(is_sharpes))


def compute_dsr(
    returns: pd.Series,
    n_trials: int,
    annualization: int = 252,
) -> dict[str, float]:
    """Module-level convenience function for DSR computation.

    Args:
        returns: Strategy daily returns.
        n_trials: Number of strategy configurations tested.
        annualization: Trading days per year. Defaults to 252.

    Returns:
        Dict with keys: observed_sr, expected_max_sr, dsr, p_value,
        annualized_sr.
    """
    return DeflatedSharpeRatio().compute(returns, n_trials, annualization)
