"""Strategy stability and performance consistency analysis."""

from __future__ import annotations

import logging
from typing import Any, Callable

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class StabilityAnalyzer:
    """Analyzes the temporal stability of strategy performance.

    Provides rolling risk/return metrics, parameter sensitivity analysis,
    performance consistency statistics, and regime-conditional breakdowns.
    """

    def __init__(self, config: dict | None = None) -> None:
        """Initialize StabilityAnalyzer.

        Args:
            config: Optional configuration dict (reserved for future use).
        """
        self._config = config or {}

    def rolling_sharpe(
        self, returns: pd.Series, window: int = 252
    ) -> pd.Series:
        """Compute rolling annualized Sharpe ratio.

        Args:
            returns: Daily return series.
            window: Rolling window in days. Defaults to 252.

        Returns:
            Rolling Sharpe ratio series.
        """
        roll_mean = returns.rolling(window).mean()
        roll_std = returns.rolling(window).std()
        return (roll_mean * np.sqrt(252)) / (roll_std * np.sqrt(252) + 1e-12)

    def rolling_max_drawdown(
        self, returns: pd.Series, window: int = 252
    ) -> pd.Series:
        """Compute rolling maximum drawdown.

        Args:
            returns: Daily return series.
            window: Rolling window in days. Defaults to 252.

        Returns:
            Rolling maximum drawdown series (negative values).
        """
        def max_dd_in_window(x: np.ndarray) -> float:
            cum = (1 + x).cumprod()
            return float(((cum - cum.cummax()) / (cum.cummax() + 1e-12)).min())

        return returns.rolling(window).apply(max_dd_in_window, raw=True)

    def rolling_beta(
        self,
        returns: pd.Series,
        benchmark: pd.Series,
        window: int = 252,
    ) -> pd.Series:
        """Compute rolling beta relative to a benchmark.

        Args:
            returns: Strategy daily returns.
            benchmark: Benchmark daily returns.
            window: Rolling window in days. Defaults to 252.

        Returns:
            Rolling beta series.
        """
        combined = pd.DataFrame({"strategy": returns, "benchmark": benchmark}).dropna()

        def beta_in_window(idx: int) -> float:
            if idx < window:
                return np.nan
            w = combined.iloc[idx - window : idx]
            cov = np.cov(w["strategy"], w["benchmark"])
            bench_var = cov[1, 1]
            return float(cov[0, 1] / (bench_var + 1e-12))

        betas = [beta_in_window(i) for i in range(len(combined))]
        return pd.Series(betas, index=combined.index, name="beta")

    def parameter_sensitivity(
        self,
        param_name: str,
        param_values: list,
        base_returns: pd.Series,
        compute_fn: Callable,
    ) -> pd.DataFrame:
        """Evaluate Sharpe sensitivity to a parameter sweep.

        Args:
            param_name: Name of the parameter being varied.
            param_values: List of parameter values to test.
            base_returns: Baseline return series passed to compute_fn.
            compute_fn: Callable ``(param_value, base_returns) -> pd.Series``
                that returns a return series for each parameter value.

        Returns:
            DataFrame with columns: param_value, sharpe, annualized_return,
            annualized_vol.
        """
        records = []
        for val in param_values:
            try:
                r = compute_fn(val, base_returns)
                ann_vol = float(r.std() * np.sqrt(252))
                ann_ret = float(r.mean() * 252)
                sharpe = ann_ret / (ann_vol + 1e-12)
                records.append(
                    {
                        param_name: val,
                        "sharpe": sharpe,
                        "annualized_return": ann_ret,
                        "annualized_vol": ann_vol,
                    }
                )
            except Exception as exc:
                logger.warning("Parameter %s=%s failed: %s", param_name, val, exc)

        return pd.DataFrame(records)

    def performance_consistency(
        self, returns: pd.Series, window: int = 63
    ) -> dict[str, float]:
        """Measure how consistently the strategy generates positive risk-adjusted returns.

        Args:
            returns: Daily return series.
            window: Rolling window to compute sub-period Sharpe. Defaults to 63.

        Returns:
            Dict with keys: pct_positive_sharpe, pct_positive_return,
            avg_rolling_sharpe, std_rolling_sharpe.
        """
        roll_sharpe = self.rolling_sharpe(returns, window).dropna()

        return {
            "pct_positive_sharpe": float((roll_sharpe > 0).mean()),
            "pct_positive_return": float((returns > 0).mean()),
            "avg_rolling_sharpe": float(roll_sharpe.mean()),
            "std_rolling_sharpe": float(roll_sharpe.std()),
        }

    def regime_stability(
        self, returns: pd.Series, regimes: pd.Series
    ) -> pd.DataFrame:
        """Compute performance metrics broken down by regime label.

        Args:
            returns: Daily return series.
            regimes: Integer or categorical regime labels aligned with returns.

        Returns:
            DataFrame indexed by regime with columns: mean_return,
            annualized_vol, sharpe, max_drawdown, n_days.
        """
        df = pd.DataFrame({"returns": returns, "regime": regimes}).dropna()
        records = {}

        for regime, group in df.groupby("regime"):
            r = group["returns"]
            n = len(r)
            ann_vol = float(r.std() * np.sqrt(252))
            ann_ret = float(r.mean() * 252)
            cum = (1 + r).cumprod()
            max_dd = float(
                ((cum - cum.cummax()) / (cum.cummax() + 1e-12)).min()
            )
            records[regime] = {
                "mean_return": float(r.mean()),
                "annualized_return": ann_ret,
                "annualized_vol": ann_vol,
                "sharpe": ann_ret / (ann_vol + 1e-12),
                "max_drawdown": max_dd,
                "n_days": int(n),
            }

        return pd.DataFrame(records).T
