"""Historical and synthetic stress testing for trading strategies."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)

CRISIS_PERIODS: dict[str, tuple[str, str]] = {
    "GFC_2008": ("2007-10-01", "2009-03-31"),
    "COVID_2020": ("2020-02-01", "2020-06-30"),
    "RATE_HIKES_2022": ("2022-01-01", "2022-12-31"),
    "DOTCOM_2000": ("2000-03-01", "2002-10-31"),
    "EURO_CRISIS_2011": ("2010-04-01", "2012-12-31"),
}


class StressTester:
    """Stress tests strategies against historical crises and synthetic shocks.

    Attributes:
        random_state: Seed for reproducible Monte Carlo simulations.
    """

    def __init__(self, config: dict | None = None) -> None:
        """Initialize StressTester.

        Args:
            config: Optional configuration dict with keys:
                - random_state (int): Defaults to 42.
        """
        cfg = config or {}
        self.random_state: int = cfg.get("random_state", 42)
        np.random.seed(self.random_state)

    def historical_crisis_replay(
        self,
        strategy_returns: pd.Series,
        crisis_name: str,
    ) -> dict[str, float]:
        """Compute strategy performance metrics during a historical crisis period.

        Args:
            strategy_returns: Full strategy daily return history.
            crisis_name: Key from CRISIS_PERIODS.

        Returns:
            Dict with keys: total_return, annualized_return, max_drawdown,
            annualized_vol, sharpe_ratio, n_days. Returns empty dict if no
            overlapping data.

        Raises:
            ValueError: If crisis_name is not in CRISIS_PERIODS.
        """
        if crisis_name not in CRISIS_PERIODS:
            raise ValueError(
                f"Unknown crisis '{crisis_name}'. "
                f"Valid options: {list(CRISIS_PERIODS.keys())}"
            )

        start, end = CRISIS_PERIODS[crisis_name]
        subset = strategy_returns.loc[start:end]

        if len(subset) == 0:
            logger.warning("No data for crisis period %s", crisis_name)
            return {}

        return self._compute_metrics(subset)

    def run_all_crises(self, strategy_returns: pd.Series) -> pd.DataFrame:
        """Run replay for all defined crisis periods.

        Args:
            strategy_returns: Full strategy daily return history.

        Returns:
            DataFrame indexed by crisis name with performance metric columns.
        """
        records = {}
        for name in CRISIS_PERIODS:
            metrics = self.historical_crisis_replay(strategy_returns, name)
            if metrics:
                records[name] = metrics

        return pd.DataFrame(records).T

    def synthetic_shock(
        self,
        strategy_returns: pd.Series,
        shock_magnitude: float = -0.3,
        shock_duration: int = 20,
    ) -> dict[str, float]:
        """Apply a synthetic shock and measure recovery characteristics.

        Distributes the shock_magnitude uniformly over shock_duration days
        appended to the historical series and computes combined metrics.

        Args:
            strategy_returns: Historical daily returns.
            shock_magnitude: Total return shock (negative = loss). Defaults to -0.3.
            shock_duration: Number of days over which shock occurs. Defaults to 20.

        Returns:
            Dict with keys: total_return, max_drawdown, recovery_days,
            annualized_vol, sharpe_ratio.
        """
        daily_shock = shock_magnitude / shock_duration
        shock_returns = pd.Series(
            [daily_shock] * shock_duration,
            index=range(shock_duration),
        )

        combined = pd.concat(
            [strategy_returns.reset_index(drop=True), shock_returns],
            ignore_index=True,
        )
        metrics = self._compute_metrics(combined)

        # Estimate recovery days (days after shock where cumulative returns > pre-shock peak)
        cum = (1 + combined).cumprod()
        pre_shock_peak = float(cum.iloc[: len(strategy_returns)].max())
        post_shock = cum.iloc[len(strategy_returns):]
        recovered = post_shock[post_shock >= pre_shock_peak]
        metrics["recovery_days"] = int(
            recovered.index[0] - len(strategy_returns) if len(recovered) else shock_duration
        )

        return metrics

    def monte_carlo_stress(
        self,
        strategy_returns: pd.Series,
        n_simulations: int = 1000,
        horizon: int = 252,
    ) -> pd.DataFrame:
        """Simulate future return paths via parametric Monte Carlo.

        Args:
            strategy_returns: Historical returns used to fit distribution.
            n_simulations: Number of simulation paths. Defaults to 1000.
            horizon: Simulation horizon in days. Defaults to 252.

        Returns:
            DataFrame of shape (horizon, n_simulations) with simulated returns.
        """
        mu = float(strategy_returns.mean())
        sigma = float(strategy_returns.std())

        rng = np.random.default_rng(self.random_state)
        paths = rng.normal(mu, sigma, size=(horizon, n_simulations))

        return pd.DataFrame(
            paths,
            columns=[f"sim_{i}" for i in range(n_simulations)],
        )

    def tail_risk_analysis(
        self, strategy_returns: pd.Series
    ) -> dict[str, float]:
        """Compute tail risk metrics.

        Args:
            strategy_returns: Strategy daily returns.

        Returns:
            Dict with keys: var_95, var_99, cvar_95, cvar_99,
            max_drawdown, skewness, kurtosis.
        """
        r = strategy_returns.dropna()
        var_95 = float(np.percentile(r, 5))
        var_99 = float(np.percentile(r, 1))
        cvar_95 = float(r[r <= var_95].mean())
        cvar_99 = float(r[r <= var_99].mean())

        cum = (1 + r).cumprod()
        max_dd = float(((cum - cum.cummax()) / cum.cummax()).min())

        return {
            "var_95": var_95,
            "var_99": var_99,
            "cvar_95": cvar_95,
            "cvar_99": cvar_99,
            "max_drawdown": max_dd,
            "skewness": float(stats.skew(r)),
            "kurtosis": float(stats.kurtosis(r)),
        }

    def _compute_metrics(self, returns: pd.Series) -> dict[str, float]:
        """Compute standard performance metrics for a return series.

        Args:
            returns: Daily return series.

        Returns:
            Dict with keys: total_return, annualized_return, max_drawdown,
            annualized_vol, sharpe_ratio, n_days.
        """
        r = returns.dropna()
        n = len(r)
        if n == 0:
            return {}

        cum = (1 + r).prod() - 1
        ann_factor = 252 / n
        ann_ret = (1 + cum) ** ann_factor - 1
        ann_vol = float(r.std() * np.sqrt(252))

        cumulative = (1 + r).cumprod()
        max_dd = float(((cumulative - cumulative.cummax()) / cumulative.cummax()).min())
        sharpe = ann_ret / (ann_vol + 1e-12)

        return {
            "total_return": float(cum),
            "annualized_return": float(ann_ret),
            "max_drawdown": float(max_dd),
            "annualized_vol": float(ann_vol),
            "sharpe_ratio": float(sharpe),
            "n_days": int(n),
        }
