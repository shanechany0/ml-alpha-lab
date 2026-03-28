"""Signal monetization utilities for ML Alpha Lab."""

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class SignalMonetizer:
    """Estimates signal economic value and capacity.

    Provides Sharpe estimation, capacity analysis, breakeven cost
    computation, PnL attribution, and a comprehensive summary.

    Attributes:
        config: Optional configuration dictionary.
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        """Initialise SignalMonetizer.

        Args:
            config: Optional configuration overrides.
        """
        self.config: dict[str, Any] = config or {}

    def estimate_sharpe(
        self,
        signal_returns: pd.Series,
        annualization: int = 252,
    ) -> float:
        """Estimate the annualised Sharpe ratio of a signal return stream.

        Args:
            signal_returns: Daily P&L or return series.
            annualization: Number of trading periods per year.

        Returns:
            Annualised Sharpe ratio (0.0 if std is zero).
        """
        clean = signal_returns.dropna()
        if clean.empty or clean.std() == 0:
            return 0.0
        sharpe = (clean.mean() / clean.std()) * np.sqrt(annualization)
        logger.info("Estimated Sharpe: %.3f", sharpe)
        return float(sharpe)

    def capacity_analysis(
        self,
        signal: pd.Series,
        adv_data: pd.DataFrame,
        target_sharpe: float = 1.0,
    ) -> dict[str, float]:
        """Estimate maximum AUM the signal can support at a target Sharpe.

        Capacity is approximated as the maximum dollar amount that can be
        traded without exceeding a participation rate threshold of the
        average daily volume.

        Args:
            signal: Composite signal Series (used for turnover estimation).
            adv_data: DataFrame of average daily volumes per ticker ($ value).
            target_sharpe: Minimum acceptable Sharpe ratio.

        Returns:
            Dictionary with:
                - ``max_aum_usd``: Estimated maximum AUM.
                - ``avg_daily_volume_usd``: Mean ADV across the universe.
                - ``participation_rate_pct``: Assumed max participation rate.
                - ``estimated_sharpe``: Sharpe ratio from signal.
        """
        participation_rate = self.config.get("participation_rate", 0.10)
        total_adv = float(adv_data.sum().sum()) if not adv_data.empty else 0.0

        # Turnover proxy: average absolute day-over-day signal change
        turnover_proxy = signal.diff().abs().mean()
        if turnover_proxy == 0 or np.isnan(turnover_proxy):
            turnover_proxy = 1.0

        max_aum = (total_adv * participation_rate) / (turnover_proxy + 1e-9)

        return {
            "max_aum_usd": float(max_aum),
            "avg_daily_volume_usd": total_adv,
            "participation_rate_pct": participation_rate * 100,
            "estimated_sharpe": self.estimate_sharpe(signal),
        }

    def breakeven_cost(
        self, gross_pnl: pd.Series, turnover: pd.Series
    ) -> float:
        """Compute the breakeven transaction cost in basis points.

        The breakeven cost is the level of transaction costs at which the
        strategy would break even (i.e. net PnL = 0).

        Args:
            gross_pnl: Daily gross P&L series.
            turnover: Daily portfolio turnover series (fraction 0–1).

        Returns:
            Breakeven cost in basis points.
        """
        common_idx = gross_pnl.index.intersection(turnover.index)
        gross = gross_pnl.loc[common_idx].dropna()
        tv = turnover.loc[common_idx].dropna()

        total_gross = gross.sum()
        total_turnover = tv.sum()

        if total_turnover == 0:
            logger.warning("Zero turnover; cannot compute breakeven cost.")
            return 0.0

        bps = (total_gross / total_turnover) * 10_000
        logger.info("Breakeven cost: %.2f bps", bps)
        return float(bps)

    def signal_pnl_attribution(
        self,
        combined_returns: pd.Series,
        individual_signals: pd.DataFrame,
    ) -> pd.DataFrame:
        """Attribute combined strategy P&L to individual signals via OLS.

        Fits a linear regression of the combined return on each individual
        signal return to compute contribution coefficients.

        Args:
            combined_returns: Daily returns of the combined signal strategy.
            individual_signals: Daily returns (or signal values) for each
                individual signal component.

        Returns:
            DataFrame with columns ``['coefficient', 'contribution_pct']``
            indexed by signal name.
        """
        common_idx = combined_returns.index.intersection(individual_signals.index)
        y = combined_returns.loc[common_idx].fillna(0)
        X = individual_signals.loc[common_idx].fillna(0)

        XtX = X.T @ X
        XtY = X.T @ y
        try:
            coeffs = np.linalg.lstsq(XtX.values, XtY.values, rcond=None)[0]
        except np.linalg.LinAlgError:
            coeffs = np.zeros(X.shape[1])

        coeff_series = pd.Series(coeffs, index=X.columns, name="coefficient")
        predicted = X @ coeff_series
        total_var = predicted.var()
        contribution_pct = pd.Series(
            {
                col: float(
                    (coeff_series[col] * X[col]).cov(predicted) / (total_var + 1e-9)
                )
                for col in X.columns
            },
            name="contribution_pct",
        )
        return pd.DataFrame({"coefficient": coeff_series, "contribution_pct": contribution_pct})

    def monetization_summary(
        self,
        signals: pd.DataFrame,
        returns: pd.DataFrame,
    ) -> dict[str, Any]:
        """Generate a comprehensive monetization summary for a set of signals.

        Args:
            signals: Wide signal DataFrame (rows=dates, cols=signal names).
            returns: Wide asset return DataFrame.

        Returns:
            Dictionary with:
                - ``signal_sharpes``: Per-signal Sharpe ratios.
                - ``combined_sharpe``: Equal-weight combined Sharpe.
                - ``avg_turnover``: Average signal turnover.
                - ``breakeven_bps``: Breakeven cost in bps.
        """
        signal_sharpes: dict[str, float] = {}
        combined_rets = pd.Series(0.0, index=signals.index)

        for col in signals.columns:
            if col in returns.columns:
                sig_ret = signals[col] * returns[col]
            else:
                sig_ret = signals[col] * returns.mean(axis=1)
            signal_sharpes[col] = self.estimate_sharpe(sig_ret)
            combined_rets += sig_ret

        n = max(len(signals.columns), 1)
        combined_rets /= n
        combined_sharpe = self.estimate_sharpe(combined_rets)

        turnover = signals.diff().abs().mean(axis=1)
        avg_turnover = float(turnover.mean())

        bps = self.breakeven_cost(combined_rets, turnover)

        return {
            "signal_sharpes": signal_sharpes,
            "combined_sharpe": combined_sharpe,
            "avg_turnover": avg_turnover,
            "breakeven_bps": bps,
        }
