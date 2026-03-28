"""Data cleaning utilities for ML Alpha Lab."""

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class DataCleaner:
    """Cleans and preprocesses raw market data.

    Attributes:
        config: Optional configuration dictionary.
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        """Initialise the DataCleaner.

        Args:
            config: Optional dictionary with cleaning parameters. Supported
                keys: ``n_std`` (float), ``min_volume`` (float),
                ``winsorize_limits`` (list[float, float]).
        """
        self.config: dict[str, Any] = config or {}

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """Run all cleaning steps in sequence.

        Steps applied:
        1. ``adjust_for_splits``
        2. ``remove_low_volume``
        3. ``handle_missing_values``
        4. ``detect_outliers``
        5. ``winsorize``

        Args:
            df: Raw OHLCV DataFrame.

        Returns:
            Cleaned DataFrame.
        """
        logger.info("Running full cleaning pipeline on shape %s", df.shape)
        df = self.adjust_for_splits(df)
        min_vol = self.config.get("min_volume", 1e6)
        df = self.remove_low_volume(df, min_volume=min_vol)
        df = self.handle_missing_values(df)
        n_std = self.config.get("n_std", 4.0)
        df = self.detect_outliers(df, n_std=n_std)
        limits_cfg = self.config.get("winsorize_limits", [0.01, 0.01])
        df = self.winsorize(df, limits=tuple(limits_cfg))  # type: ignore[arg-type]
        return df

    def handle_missing_values(
        self, df: pd.DataFrame, method: str = "ffill"
    ) -> pd.DataFrame:
        """Impute missing values in the DataFrame.

        Args:
            df: Input DataFrame with potential NaN values.
            method: Imputation method. One of ``'ffill'`` (forward fill),
                ``'bfill'`` (backward fill), or ``'interpolate'`` (linear
                interpolation).

        Returns:
            DataFrame with missing values handled.

        Raises:
            ValueError: If an unknown method is specified.
        """
        logger.debug("Handling missing values with method='%s'", method)
        if method == "ffill":
            df = df.ffill()
        elif method == "bfill":
            df = df.bfill()
        elif method == "interpolate":
            df = df.interpolate(method="linear")
        else:
            raise ValueError(
                f"Unknown method '{method}'. Choose from 'ffill', 'bfill', 'interpolate'."
            )
        # Fill any remaining NaNs at the start with back-fill
        df = df.bfill()
        return df

    def detect_outliers(
        self, df: pd.DataFrame, n_std: float = 4.0
    ) -> pd.DataFrame:
        """Replace z-score outliers with NaN, then forward-fill.

        Values more than ``n_std`` standard deviations from the rolling mean
        are flagged as outliers and replaced with NaN.

        Args:
            df: Input DataFrame.
            n_std: Number of standard deviations threshold.

        Returns:
            DataFrame with outliers replaced.
        """
        logger.debug("Detecting outliers with n_std=%.1f", n_std)
        numeric = df.select_dtypes(include=[np.number])
        zscore = (numeric - numeric.mean()) / (numeric.std() + 1e-9)
        mask = zscore.abs() > n_std
        cleaned = df.copy()
        cleaned[mask] = np.nan
        cleaned = cleaned.ffill()
        n_outliers = int(mask.values.sum())
        logger.info("Replaced %d outlier values", n_outliers)
        return cleaned

    def winsorize(
        self, df: pd.DataFrame, limits: tuple[float, float] = (0.01, 0.01)
    ) -> pd.DataFrame:
        """Winsorize the DataFrame column-wise.

        Args:
            df: Input DataFrame (typically containing returns).
            limits: Lower and upper tail fractions to clip.

        Returns:
            Winsorized DataFrame.
        """
        logger.debug("Winsorizing with limits=%s", limits)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        result = df.copy()
        lower_q, upper_q = limits[0], 1.0 - limits[1]
        for col in numeric_cols:
            lo = df[col].quantile(lower_q)
            hi = df[col].quantile(upper_q)
            result[col] = df[col].clip(lower=lo, upper=hi)
        return result

    def adjust_for_splits(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure prices are split-adjusted using the adjusted close column.

        When an ``'adj close'`` column is present, computes a split/dividend
        adjustment ratio and applies it to ``open``, ``high``, ``low``, and
        ``close`` columns.

        Args:
            df: OHLCV DataFrame, optionally with ``'adj close'`` column.

        Returns:
            DataFrame with prices adjusted for splits.
        """
        adj_col = None
        for candidate in ["adj close", "Adj Close", "adjclose"]:
            if candidate in df.columns:
                adj_col = candidate
                break

        if adj_col is None:
            logger.debug("No 'adj close' column found; skipping split adjustment.")
            return df

        result = df.copy()
        ratio = result[adj_col] / result.get("close", result.get("Close", result[adj_col]))
        for col in ["open", "high", "low", "close", "Open", "High", "Low", "Close"]:
            if col in result.columns:
                result[col] = result[col] * ratio
        logger.info("Applied split adjustment using '%s' column.", adj_col)
        return result

    def remove_low_volume(
        self, df: pd.DataFrame, min_volume: float = 1e6
    ) -> pd.DataFrame:
        """Replace rows with volume below a minimum threshold with NaN.

        Args:
            df: OHLCV DataFrame.
            min_volume: Minimum acceptable average daily volume.

        Returns:
            DataFrame with low-volume periods replaced by NaN.
        """
        vol_col = None
        for candidate in ["volume", "Volume"]:
            if candidate in df.columns:
                vol_col = candidate
                break

        if vol_col is None:
            logger.debug("No volume column found; skipping low-volume filter.")
            return df

        result = df.copy()
        low_vol_mask = result[vol_col] < min_volume
        n_low = int(low_vol_mask.sum())
        result[low_vol_mask] = np.nan
        logger.info("Removed %d low-volume rows (min_volume=%.0f)", n_low, min_volume)
        return result

    def compute_returns(
        self, df: pd.DataFrame, method: str = "log"
    ) -> pd.DataFrame:
        """Compute period returns from price data.

        Args:
            df: DataFrame of prices (each column is a price series).
            method: ``'log'`` for log returns or ``'simple'`` for simple
                percentage returns.

        Returns:
            DataFrame of returns with the same columns, NaN for the first row.

        Raises:
            ValueError: If an unknown method is given.
        """
        logger.debug("Computing %s returns", method)
        if method == "log":
            returns = np.log(df / df.shift(1))
        elif method == "simple":
            returns = df.pct_change()
        else:
            raise ValueError(
                f"Unknown method '{method}'. Choose from 'log' or 'simple'."
            )
        return returns.dropna()
