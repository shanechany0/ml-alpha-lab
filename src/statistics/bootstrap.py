"""Circular block bootstrap for time-series inference."""

from __future__ import annotations

import logging
from typing import Callable

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class Bootstrap:
    """Circular block bootstrap for dependent time-series data.

    Provides bootstrap confidence intervals, p-values, and IC distributions
    while preserving the autocorrelation structure of financial returns.

    Attributes:
        n_bootstrap: Number of bootstrap replications.
        block_size: Block length for the block bootstrap. Auto-selected if None.
        random_state: Seed for reproducibility.
    """

    def __init__(
        self,
        n_bootstrap: int = 1000,
        block_size: int | None = None,
        random_state: int = 42,
    ) -> None:
        """Initialize Bootstrap.

        Args:
            n_bootstrap: Number of bootstrap samples. Defaults to 1000.
            block_size: Block length. If None, uses floor(T^(1/3)). Defaults to None.
            random_state: Random seed. Defaults to 42.
        """
        self.n_bootstrap = n_bootstrap
        self.block_size = block_size
        self.random_state = random_state
        self._rng = np.random.default_rng(random_state)

    def block_bootstrap(
        self,
        series: pd.Series,
        statistic_fn: Callable[[pd.Series], float],
    ) -> np.ndarray:
        """Draw circular block bootstrap samples and compute the statistic.

        Args:
            series: Time series to bootstrap.
            statistic_fn: Callable that maps a pd.Series to a scalar float.

        Returns:
            Array of bootstrap statistic values, shape (n_bootstrap,).
        """
        data = series.dropna().values
        n = len(data)
        b = self.block_size if self.block_size is not None else max(1, int(n ** (1 / 3)))

        results = np.empty(self.n_bootstrap)
        for i in range(self.n_bootstrap):
            n_blocks = int(np.ceil(n / b))
            start_indices = self._rng.integers(0, n, size=n_blocks)
            resampled = np.concatenate(
                [np.roll(data, -s)[:b] for s in start_indices]
            )[:n]
            results[i] = statistic_fn(pd.Series(resampled))

        return results

    def confidence_interval(
        self,
        series: pd.Series,
        statistic_fn: Callable[[pd.Series], float],
        alpha: float = 0.05,
    ) -> tuple[float, float]:
        """Compute percentile bootstrap confidence interval.

        Args:
            series: Time series.
            statistic_fn: Statistic to bootstrap.
            alpha: Significance level. Defaults to 0.05.

        Returns:
            Tuple of (lower_bound, upper_bound).
        """
        samples = self.block_bootstrap(series, statistic_fn)
        lower = float(np.percentile(samples, 100 * alpha / 2))
        upper = float(np.percentile(samples, 100 * (1 - alpha / 2)))
        return lower, upper

    def sharpe_confidence_interval(
        self,
        returns: pd.Series,
        alpha: float = 0.05,
    ) -> tuple[float, float]:
        """Compute bootstrap confidence interval for the annualized Sharpe ratio.

        Args:
            returns: Daily return series.
            alpha: Significance level. Defaults to 0.05.

        Returns:
            Tuple of (lower, upper) annualized Sharpe ratio bounds.
        """
        def ann_sharpe(r: pd.Series) -> float:
            std = float(r.std())
            return float(r.mean() * np.sqrt(252)) / (std * np.sqrt(252) + 1e-12)

        return self.confidence_interval(returns, ann_sharpe, alpha)

    def p_value_bootstrap(
        self,
        observed_statistic: float,
        bootstrap_samples: np.ndarray,
    ) -> float:
        """Compute one-sided bootstrap p-value.

        Args:
            observed_statistic: Observed value of the test statistic.
            bootstrap_samples: Bootstrap distribution under null.

        Returns:
            Fraction of bootstrap samples ≥ observed statistic.
        """
        return float(np.mean(bootstrap_samples >= observed_statistic))

    def bootstrap_ic(
        self,
        signals: pd.DataFrame,
        returns: pd.DataFrame,
    ) -> pd.DataFrame:
        """Bootstrap confidence intervals for information coefficients.

        Computes IC (rank correlation) for each signal column and provides
        bootstrap 95% confidence intervals.

        Args:
            signals: DataFrame of signal values with shape (T, n_signals).
            returns: DataFrame of forward returns with shape (T, n_assets).

        Returns:
            DataFrame indexed by signal name with columns:
            ic_mean, ic_std, ci_lower, ci_upper.
        """
        records = []
        for col in signals.columns:
            sig = signals[col].dropna()
            ret = returns.reindex(sig.index)

            if ret.ndim > 1:
                ret_series = ret.mean(axis=1).dropna()
            else:
                ret_series = ret.dropna()

            aligned = pd.concat([sig, ret_series], axis=1).dropna()
            if len(aligned) < 10:
                records.append(
                    {"signal": col, "ic_mean": np.nan, "ic_std": np.nan,
                     "ci_lower": np.nan, "ci_upper": np.nan}
                )
                continue

            def ic_fn(x: pd.Series) -> float:
                half = len(x) // 2
                s = x.iloc[:half]
                r = x.iloc[half:]
                if len(s) != len(r):
                    r = r.iloc[: len(s)]
                return float(s.corr(r, method="spearman"))

            raw_ic = float(
                aligned.iloc[:, 0].corr(aligned.iloc[:, 1], method="spearman")
            )

            samples = self.block_bootstrap(aligned.iloc[:, 0], lambda s: float(
                s.corr(aligned.iloc[:len(s), 1], method="spearman")
            ))

            records.append(
                {
                    "signal": col,
                    "ic_mean": raw_ic,
                    "ic_std": float(np.std(samples)),
                    "ci_lower": float(np.percentile(samples, 2.5)),
                    "ci_upper": float(np.percentile(samples, 97.5)),
                }
            )

        return pd.DataFrame(records).set_index("signal")
