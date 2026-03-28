"""Signal evaluation utilities for ML Alpha Lab."""

import logging
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


class SignalEvaluator:
    """Evaluates alpha signal quality.

    Metrics include Information Coefficient (IC), IC decay, turnover,
    hit rate, and a comprehensive full evaluation suite.

    Attributes:
        config: Optional configuration dictionary.
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        """Initialise SignalEvaluator.

        Args:
            config: Optional configuration overrides.
        """
        self.config: dict[str, Any] = config or {}

    def information_coefficient(
        self,
        signals: pd.DataFrame,
        forward_returns: pd.DataFrame,
        periods: list[int] | None = None,
    ) -> pd.DataFrame:
        """Compute the rank IC between signals and forward returns.

        For each signal column and each prediction horizon, calculates the
        Spearman rank correlation between the signal and subsequent returns.

        Args:
            signals: Wide DataFrame of alpha signals (rows=dates, cols=signals).
            forward_returns: Wide DataFrame of forward returns. Columns
                should match tickers; must be shifted appropriately for
                each ``period``.
            periods: Forward return horizons in days. Defaults to
                ``[1, 5, 20]``.

        Returns:
            DataFrame with IC values indexed by date, columns as
            ``{signal}_{period}d_ic``.
        """
        if periods is None:
            periods = [1, 5, 20]

        results: dict[str, pd.Series] = {}
        for period in periods:
            fwd = forward_returns.shift(-period)
            common_cols = signals.columns.intersection(fwd.columns)
            for sig_col in signals.columns:
                if sig_col not in common_cols:
                    continue
                fwd_col = fwd[sig_col]

                def _ic_apply(x: pd.Series, _fwd: pd.Series = fwd_col) -> float:
                    aligned = _fwd.reindex(x.index)
                    valid = x.notna() & aligned.notna()
                    if valid.sum() < 5:
                        return np.nan
                    rho, _ = stats.spearmanr(x[valid], aligned[valid])
                    return float(rho)

                ic_series = signals[sig_col].rolling(60).apply(
                    _ic_apply, raw=False
                )
                results[f"{sig_col}_{period}d_ic"] = ic_series

        return pd.DataFrame(results, index=signals.index)

    def information_ratio(self, ic_series: pd.Series) -> float:
        """Compute the Information Ratio from an IC time series.

        Args:
            ic_series: Series of daily IC values.

        Returns:
            IR = mean(IC) / std(IC). Returns 0.0 if std is zero.
        """
        ic_clean = ic_series.dropna()
        std = ic_clean.std()
        if std == 0 or np.isnan(std):
            return 0.0
        return float(ic_clean.mean() / std)

    def ic_decay(
        self,
        signals: pd.DataFrame,
        forward_returns: pd.DataFrame,
        max_lag: int = 20,
    ) -> pd.DataFrame:
        """Compute IC at multiple forward horizons to measure signal decay.

        Args:
            signals: Wide signal DataFrame.
            forward_returns: Wide forward return DataFrame.
            max_lag: Maximum number of periods ahead to evaluate.

        Returns:
            DataFrame with lag on the index and signal names as columns,
            showing mean IC at each horizon.
        """
        common_cols = signals.columns.intersection(forward_returns.columns)
        lags = range(1, max_lag + 1)
        records: list[dict[str, Any]] = []
        for lag in lags:
            fwd = forward_returns.shift(-lag)
            row: dict[str, Any] = {"lag": lag}
            for col in common_cols:
                valid = signals[col].dropna().index.intersection(fwd[col].dropna().index)
                if len(valid) < 10:
                    row[col] = np.nan
                    continue
                ic, _ = stats.spearmanr(signals[col][valid], fwd[col][valid])
                row[col] = ic
            records.append(row)
        return pd.DataFrame(records).set_index("lag")

    def turnover_analysis(self, signals: pd.DataFrame) -> dict[str, float]:
        """Analyse signal turnover (fraction of rank changes day-over-day).

        Args:
            signals: Wide signal DataFrame.

        Returns:
            Dictionary mapping signal column name → average daily turnover.
        """
        result: dict[str, float] = {}
        for col in signals.columns:
            ranked = signals[col].rank(pct=True)
            daily_change = ranked.diff().abs()
            result[col] = float(daily_change.mean())
        logger.info("Turnover analysis: %s", result)
        return result

    def hit_rate(
        self,
        signals: pd.DataFrame,
        forward_returns: pd.DataFrame,
        period: int = 5,
    ) -> dict[str, float]:
        """Compute the fraction of correctly predicted return directions.

        Args:
            signals: Wide signal DataFrame.
            forward_returns: Wide forward return DataFrame.
            period: Forward return horizon.

        Returns:
            Dictionary mapping signal column name → hit rate (0 to 1).
        """
        fwd = forward_returns.shift(-period)
        common_cols = signals.columns.intersection(fwd.columns)
        result: dict[str, float] = {}
        for col in common_cols:
            valid = signals[col].dropna().index.intersection(fwd[col].dropna().index)
            if len(valid) == 0:
                result[col] = np.nan
                continue
            sig_sign = np.sign(signals[col][valid])
            ret_sign = np.sign(fwd[col][valid])
            result[col] = float((sig_sign == ret_sign).mean())
        logger.info("Hit rates: %s", result)
        return result

    def full_evaluation(
        self,
        signals: pd.DataFrame,
        forward_returns: pd.DataFrame,
    ) -> dict[str, Any]:
        """Run a comprehensive evaluation of signal quality.

        Args:
            signals: Wide signal DataFrame.
            forward_returns: Wide forward return DataFrame.

        Returns:
            Dictionary containing:
                - ``ic_by_period``: IC DataFrame.
                - ``ic_decay``: IC decay DataFrame.
                - ``turnover``: Turnover dict.
                - ``hit_rate``: Hit-rate dict.
                - ``information_ratios``: Dict of IR per signal.
        """
        logger.info("Running full signal evaluation.")
        ic = self.information_coefficient(signals, forward_returns)
        decay = self.ic_decay(signals, forward_returns)
        turnover = self.turnover_analysis(signals)
        hr = self.hit_rate(signals, forward_returns)
        ir: dict[str, float] = {}
        for col in signals.columns:
            ic_cols = [c for c in ic.columns if c.startswith(col)]
            for ic_col in ic_cols:
                ir[ic_col] = self.information_ratio(ic[ic_col])
        return {
            "ic_by_period": ic,
            "ic_decay": decay,
            "turnover": turnover,
            "hit_rate": hr,
            "information_ratios": ir,
        }
