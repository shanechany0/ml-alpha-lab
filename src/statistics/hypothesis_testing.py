"""Statistical hypothesis testing with multiple comparison corrections."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


class HypothesisTester:
    """Statistical hypothesis tests for trading strategy evaluation.

    Provides t-tests for return significance, Sharpe significance, and
    multiple comparison corrections (Bonferroni, Holm, Benjamini-Hochberg).
    """

    def __init__(self, config: dict | None = None) -> None:
        """Initialize HypothesisTester.

        Args:
            config: Optional configuration dict (reserved for future use).
        """
        self._config = config or {}

    def bonferroni_correction(
        self,
        p_values: list[float],
        alpha: float = 0.05,
    ) -> tuple[list[bool], list[float]]:
        """Apply Bonferroni correction to a list of p-values.

        Args:
            p_values: Observed p-values.
            alpha: Family-wise error rate. Defaults to 0.05.

        Returns:
            Tuple of (reject_list, corrected_p_values).
        """
        n = len(p_values)
        corrected = [min(1.0, p * n) for p in p_values]
        reject = [p <= alpha for p in corrected]
        return reject, corrected

    def holm_correction(
        self,
        p_values: list[float],
        alpha: float = 0.05,
    ) -> tuple[list[bool], list[float]]:
        """Apply Holm–Bonferroni step-down correction.

        Args:
            p_values: Observed p-values.
            alpha: Family-wise error rate. Defaults to 0.05.

        Returns:
            Tuple of (reject_list, adjusted_p_values).
        """
        n = len(p_values)
        indexed = sorted(enumerate(p_values), key=lambda x: x[1])
        reject = [False] * n
        adjusted = [0.0] * n

        for rank, (orig_idx, p) in enumerate(indexed):
            adj_p = min(1.0, p * (n - rank))
            adjusted[orig_idx] = adj_p

        # Step-down: once we fail to reject, all subsequent fail too
        for rank, (orig_idx, _) in enumerate(indexed):
            if adjusted[orig_idx] <= alpha:
                reject[orig_idx] = True
            else:
                # All remaining p-values are larger; stop rejecting
                break

        return reject, adjusted

    def fdr_correction(
        self,
        p_values: list[float],
        alpha: float = 0.05,
    ) -> tuple[list[bool], list[float]]:
        """Apply Benjamini-Hochberg FDR correction.

        Args:
            p_values: Observed p-values.
            alpha: FDR threshold. Defaults to 0.05.

        Returns:
            Tuple of (reject_list, adjusted_p_values).
        """
        n = len(p_values)
        indexed = sorted(enumerate(p_values), key=lambda x: x[1])
        adjusted = [0.0] * n
        reject = [False] * n

        for rank, (orig_idx, p) in enumerate(indexed):
            adj_p = min(1.0, p * n / (rank + 1))
            adjusted[orig_idx] = adj_p

        # Ensure monotonicity (BH step-up)
        max_adj = 0.0
        for _, (orig_idx, _) in enumerate(reversed(indexed)):
            max_adj = max(max_adj, adjusted[orig_idx])
            adjusted[orig_idx] = max_adj

        reject = [p <= alpha for p in adjusted]
        return reject, adjusted

    def t_test_returns(self, returns: pd.Series) -> dict[str, float]:
        """Test whether strategy mean return is significantly greater than zero.

        Args:
            returns: Daily return series.

        Returns:
            Dict with keys: mean_return, t_statistic, p_value, significant.
        """
        r = returns.dropna()
        t_stat, p_val = stats.ttest_1samp(r, 0.0)
        return {
            "mean_return": float(r.mean()),
            "t_stat": float(t_stat),
            "t_statistic": float(t_stat),
            "p_value": float(p_val / 2),  # one-sided
            "significant": bool(t_stat > 0 and p_val / 2 < 0.05),
        }

    def sharpe_significance(
        self,
        returns: pd.Series,
        n_obs: int | None = None,
    ) -> dict[str, float]:
        """Test significance of the annualized Sharpe ratio.

        Uses the Jobson-Korkie / Lo (2002) asymptotic test statistic:
            z = SR_hat * sqrt(T) / sqrt(1 + 0.5 * SR_hat^2)

        Args:
            returns: Daily return series.
            n_obs: Number of observations used for the test. Defaults to
                len(returns).

        Returns:
            Dict with keys: sharpe_ratio, z_statistic, p_value, significant.
        """
        r = returns.dropna()
        t = n_obs if n_obs is not None else len(r)
        sr = float(r.mean() / (r.std() + 1e-12)) * np.sqrt(252)
        sr_daily = float(r.mean() / (r.std() + 1e-12))

        z = sr_daily * np.sqrt(t) / np.sqrt(1.0 + 0.5 * sr_daily ** 2)
        p_val = float(1.0 - stats.norm.cdf(z))

        return {
            "sharpe_ratio": sr,
            "z_statistic": float(z),
            "p_value": p_val,
            "significant": p_val < 0.05,
        }

    def multiple_testing_correction(
        self,
        results: dict[str, float],
        method: str = "holm",
    ) -> dict[str, dict]:
        """Apply multiple testing correction to a collection of p-values.

        Args:
            results: Dict mapping strategy/test name to p-value.
            method: Correction method — "bonferroni", "holm", or "fdr".

        Returns:
            Dict mapping name to sub-dict with keys: p_value, adjusted_p,
            reject.

        Raises:
            ValueError: If an unsupported method is provided.
        """
        names = list(results.keys())
        p_vals = [results[n] for n in names]

        if method == "bonferroni":
            reject, adjusted = self.bonferroni_correction(p_vals)
        elif method == "holm":
            reject, adjusted = self.holm_correction(p_vals)
        elif method == "fdr":
            reject, adjusted = self.fdr_correction(p_vals)
        else:
            raise ValueError(f"Unsupported method: {method}")

        return {
            name: {
                "p_value": float(p_vals[i]),
                "adjusted_p": float(adjusted[i]),
                "reject": bool(reject[i]),
            }
            for i, name in enumerate(names)
        }
