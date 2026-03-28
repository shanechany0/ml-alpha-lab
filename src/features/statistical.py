"""Statistical features for ML Alpha Lab."""

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class StatisticalFeatures:
    """Computes statistical features from return series.

    Attributes:
        config: Optional configuration dictionary.
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        """Initialise StatisticalFeatures.

        Args:
            config: Optional parameter overrides dictionary.
        """
        self.config: dict[str, Any] = config or {}

    def compute_all(self, returns: pd.DataFrame) -> pd.DataFrame:
        """Compute all statistical features.

        Args:
            returns: DataFrame of asset returns with DatetimeIndex.

        Returns:
            Wide DataFrame of statistical features.
        """
        logger.info("Computing all statistical features on shape %s", returns.shape)
        frames: list[pd.DataFrame] = [
            self.rolling_stats(returns),
            self.realized_volatility(returns),
            self.autocorrelation(returns),
        ]
        for col in returns.columns:
            zs = self.zscore(returns[col]).rename(f"{col}_zscore")
            frames.append(zs.to_frame())
            hurst = self.hurst_exponent(returns[col]).rename(f"{col}_hurst")
            frames.append(hurst.to_frame())
        return pd.concat(frames, axis=1)

    def rolling_stats(
        self,
        returns: pd.DataFrame,
        windows: list[int] | None = None,
    ) -> pd.DataFrame:
        """Compute rolling mean, std, skewness, and kurtosis.

        Args:
            returns: Return DataFrame.
            windows: List of rolling window sizes. Defaults to
                ``[5, 10, 20, 60]``.

        Returns:
            DataFrame with columns ``{ticker}_{stat}_{window}``.
        """
        if windows is None:
            windows = [5, 10, 20, 60]
        parts: dict[str, pd.Series] = {}
        for col in returns.columns:
            s = returns[col]
            for w in windows:
                roll = s.rolling(w)
                parts[f"{col}_mean_{w}"] = roll.mean()
                parts[f"{col}_std_{w}"] = roll.std()
                parts[f"{col}_skew_{w}"] = roll.skew()
                parts[f"{col}_kurt_{w}"] = roll.kurt()
        return pd.DataFrame(parts, index=returns.index)

    def zscore(self, series: pd.Series, window: int = 20) -> pd.Series:
        """Compute rolling z-score of a series.

        Args:
            series: Input series.
            window: Rolling window size.

        Returns:
            Rolling z-score series.
        """
        roll = series.rolling(window)
        z = (series - roll.mean()) / (roll.std() + 1e-9)
        z.name = f"{series.name}_zscore_{window}"
        return z

    def autocorrelation(
        self,
        returns: pd.DataFrame,
        lags: list[int] | None = None,
    ) -> pd.DataFrame:
        """Compute rolling autocorrelation at specified lags.

        Args:
            returns: Return DataFrame.
            lags: List of lag values. Defaults to ``[1, 5, 10]``.

        Returns:
            DataFrame with columns ``{ticker}_autocorr_lag{lag}``.
        """
        if lags is None:
            lags = [1, 5, 10]
        parts: dict[str, pd.Series] = {}
        window = max(lags) * 5
        for col in returns.columns:
            s = returns[col]
            for lag in lags:
                parts[f"{col}_autocorr_lag{lag}"] = s.rolling(window).apply(
                    lambda x, lag_=lag: x.autocorr(lag=lag_) if len(x) > lag_ + 1 else np.nan,
                    raw=False,
                )
        return pd.DataFrame(parts, index=returns.index)

    def realized_volatility(
        self,
        returns: pd.DataFrame,
        windows: list[int] | None = None,
    ) -> pd.DataFrame:
        """Compute rolling realized volatility (annualised).

        Args:
            returns: Return DataFrame.
            windows: Rolling window sizes. Defaults to ``[5, 20, 60]``.

        Returns:
            DataFrame with columns ``{ticker}_realized_vol_{window}``.
        """
        if windows is None:
            windows = [5, 20, 60]
        parts: dict[str, pd.Series] = {}
        for col in returns.columns:
            s = returns[col]
            for w in windows:
                rv = s.rolling(w).std() * np.sqrt(252)
                parts[f"{col}_realized_vol_{w}"] = rv
        return pd.DataFrame(parts, index=returns.index)

    def hurst_exponent(
        self,
        series: pd.Series,
        lags: list[int] | None = None,
    ) -> pd.Series:
        """Estimate the rolling Hurst exponent.

        Uses R/S analysis over a trailing window. Values near 0.5 indicate
        random walk, <0.5 mean-reversion, >0.5 trending.

        Args:
            series: Return or price series.
            lags: Lag sizes for R/S analysis. Defaults to
                ``[2, 4, 8, 16, 32]``.

        Returns:
            Rolling Hurst exponent series.
        """
        if lags is None:
            lags = [2, 4, 8, 16, 32]

        window = max(lags) * 4

        def _hurst(x: np.ndarray) -> float:
            if len(x) < max(lags) + 1:
                return np.nan
            rs_vals = []
            for lag in lags:
                chunks = len(x) // lag
                if chunks < 1:
                    continue
                rs_chunk = []
                for i in range(chunks):
                    sub = x[i * lag : (i + 1) * lag]
                    mean_sub = np.mean(sub)
                    dev = np.cumsum(sub - mean_sub)
                    r = dev.max() - dev.min()
                    s = np.std(sub, ddof=1)
                    if s > 0:
                        rs_chunk.append(r / s)
                if rs_chunk:
                    rs_vals.append((lag, np.mean(rs_chunk)))
            if len(rs_vals) < 2:
                return np.nan
            lags_arr = np.log([v[0] for v in rs_vals])
            rs_arr = np.log([v[1] for v in rs_vals])
            hurst = np.polyfit(lags_arr, rs_arr, 1)[0]
            return float(hurst)

        result = series.rolling(window).apply(lambda x: _hurst(x), raw=True)
        result.name = f"{series.name}_hurst"
        return result

    def information_ratio_feature(
        self,
        returns: pd.Series,
        benchmark_returns: pd.Series,
        window: int = 60,
    ) -> pd.Series:
        """Compute rolling Information Ratio relative to a benchmark.

        Args:
            returns: Asset return series.
            benchmark_returns: Benchmark return series.
            window: Rolling window size.

        Returns:
            Rolling IR series (mean active return / std active return).
        """
        active = returns - benchmark_returns
        roll = active.rolling(window)
        ir = roll.mean() / (roll.std() + 1e-9)
        ir.name = f"{returns.name}_ir_{window}"
        return ir
