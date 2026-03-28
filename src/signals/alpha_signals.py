"""Alpha signal generation for ML Alpha Lab."""

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class AlphaSignals:
    """Generates alpha signals from price and return data.

    All signals are cross-sectionally standardised (z-scored) before being
    returned to ensure comparability and consistent scaling.

    Attributes:
        config: Optional configuration dictionary.
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        """Initialise AlphaSignals.

        Args:
            config: Optional parameter overrides.
        """
        self.config: dict[str, Any] = config or {}

    @staticmethod
    def _cs_zscore(df: pd.DataFrame) -> pd.DataFrame:
        """Apply cross-sectional z-score normalisation row-wise.

        Args:
            df: Wide DataFrame (rows=dates, cols=tickers).

        Returns:
            Cross-sectionally z-scored DataFrame.
        """
        cs_mean = df.mean(axis=1)
        cs_std = df.std(axis=1).replace(0, np.nan)
        return df.sub(cs_mean, axis=0).div(cs_std, axis=0)

    def compute_all(
        self, prices: pd.DataFrame, returns: pd.DataFrame
    ) -> pd.DataFrame:
        """Compute all alpha signals and return a combined DataFrame.

        Args:
            prices: Wide DataFrame of close prices.
            returns: Wide DataFrame of asset returns.

        Returns:
            DataFrame of cross-sectionally normalised alpha signals.
        """
        logger.info("Computing all alpha signals.")
        frames: list[pd.DataFrame] = [
            self.time_series_momentum(returns).add_prefix("ts_mom_"),
            self.cross_sectional_momentum(returns).add_prefix("cs_mom_"),
            self.mean_reversion(prices).add_prefix("mean_rev_"),
            self.short_term_reversal(returns).add_prefix("str_"),
            self.value_signal(prices).add_prefix("value_"),
            self.quality_signal(returns).add_prefix("quality_"),
        ]
        return pd.concat(frames, axis=1)

    def time_series_momentum(
        self,
        returns: pd.DataFrame,
        windows: list[int] | None = None,
    ) -> pd.DataFrame:
        """Compute time-series momentum as the sign of past cumulative return.

        Args:
            returns: Wide return DataFrame.
            windows: Look-back windows. Defaults to ``[20, 60, 120]``.

        Returns:
            Cross-sectionally z-scored TS momentum DataFrame.
        """
        if windows is None:
            windows = [20, 60, 120]
        parts: dict[str, pd.Series] = {}
        for w in windows:
            signal = np.sign(returns.rolling(w).sum())
            zs = self._cs_zscore(signal)
            for col in zs.columns:
                parts[f"{col}_{w}d"] = zs[col]
        return pd.DataFrame(parts, index=returns.index)

    def cross_sectional_momentum(
        self, returns: pd.DataFrame, window: int = 60
    ) -> pd.DataFrame:
        """Compute cross-sectional rank-based momentum.

        Args:
            returns: Wide return DataFrame.
            window: Look-back window for cumulative return.

        Returns:
            Cross-sectionally z-scored CS momentum DataFrame.
        """
        cum_ret = returns.rolling(window).sum()
        ranked = cum_ret.rank(axis=1, pct=True)
        return self._cs_zscore(ranked)

    def mean_reversion(
        self, prices: pd.DataFrame, window: int = 20
    ) -> pd.DataFrame:
        """Compute mean-reversion signal as the negative z-score of price.

        A stock trading above its rolling mean is expected to revert down,
        so the signal is the negative of the rolling z-score.

        Args:
            prices: Wide price DataFrame.
            window: Rolling window for mean and std.

        Returns:
            Cross-sectionally z-scored mean-reversion DataFrame.
        """
        roll_mean = prices.rolling(window).mean()
        roll_std = prices.rolling(window).std().replace(0, np.nan)
        zscore = (prices - roll_mean) / roll_std
        signal = -zscore
        return self._cs_zscore(signal)

    def short_term_reversal(
        self, returns: pd.DataFrame, window: int = 5
    ) -> pd.DataFrame:
        """Compute short-term reversal signal (negative recent return).

        Args:
            returns: Wide return DataFrame.
            window: Short look-back window.

        Returns:
            Cross-sectionally z-scored short-term reversal DataFrame.
        """
        recent_return = returns.rolling(window).sum()
        signal = -recent_return
        return self._cs_zscore(signal)

    def value_signal(
        self,
        prices: pd.DataFrame,
        fundamental_data: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """Compute a value signal based on price relative to rolling average.

        When ``fundamental_data`` is not provided, uses the ratio of the
        current price to its 252-day rolling mean as a valuation proxy
        (lower is cheaper).

        Args:
            prices: Wide price DataFrame.
            fundamental_data: Optional fundamental data (e.g. P/E ratios).
                If provided, used directly as the value signal.

        Returns:
            Cross-sectionally z-scored value DataFrame.
        """
        if fundamental_data is not None:
            return self._cs_zscore(-fundamental_data)

        long_avg = prices.rolling(252, min_periods=63).mean()
        price_to_avg = prices / (long_avg + 1e-9)
        signal = -price_to_avg  # cheaper stocks have lower price-to-avg
        return self._cs_zscore(signal)

    def quality_signal(
        self, returns: pd.DataFrame, window: int = 252
    ) -> pd.DataFrame:
        """Compute a quality signal as inverse realised volatility.

        Low-volatility stocks are considered higher quality in this proxy.

        Args:
            returns: Wide return DataFrame.
            window: Look-back window for volatility estimation.

        Returns:
            Cross-sectionally z-scored quality (low-vol) DataFrame.
        """
        realized_vol = returns.rolling(window, min_periods=63).std() * np.sqrt(252)
        signal = -realized_vol  # lower vol → higher quality
        return self._cs_zscore(signal)
