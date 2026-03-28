"""Data validation utilities for ML Alpha Lab."""

import logging
from typing import Any

import numpy as np
import pandas as pd
from pandas.tseries.offsets import BDay

logger = logging.getLogger(__name__)


class DataValidator:
    """Validates market data quality.

    Runs a suite of checks covering completeness, date continuity,
    price reasonableness, and return distribution.
    """

    def __init__(self) -> None:
        """Initialise the DataValidator."""

    def validate(self, df: pd.DataFrame) -> dict[str, Any]:
        """Run all validation checks and return a summary dictionary.

        Args:
            df: DataFrame to validate (DatetimeIndex expected).

        Returns:
            Dictionary with keys:
                - ``completeness`` (bool)
                - ``date_continuity`` (bool)
                - ``price_reasonableness`` (bool)
                - ``return_distribution`` (dict[str, float])
                - ``passed`` (bool): True only if all boolean checks pass.
        """
        logger.info("Running full validation on shape %s", df.shape)
        completeness = self.check_completeness(df)
        date_continuity = self.check_date_continuity(df)
        price_ok = self.check_price_reasonableness(df)
        ret_dist = self.check_return_distribution(df)
        passed = completeness and date_continuity and price_ok
        return {
            "completeness": completeness,
            "date_continuity": date_continuity,
            "price_reasonableness": price_ok,
            "return_distribution": ret_dist,
            "passed": passed,
        }

    def check_completeness(
        self, df: pd.DataFrame, max_missing_pct: float = 0.05
    ) -> bool:
        """Check that the fraction of missing values is below a threshold.

        Args:
            df: Input DataFrame.
            max_missing_pct: Maximum allowable fraction of NaN values.

        Returns:
            True if missing fraction is within the acceptable threshold.
        """
        total = df.size
        if total == 0:
            return True
        missing_pct = df.isna().sum().sum() / total
        ok = missing_pct <= max_missing_pct
        logger.info(
            "Completeness check: missing=%.2f%% (threshold=%.2f%%), passed=%s",
            missing_pct * 100,
            max_missing_pct * 100,
            ok,
        )
        return bool(ok)

    def check_date_continuity(self, df: pd.DataFrame) -> bool:
        """Check that the index has no unexpected gaps in business days.

        Args:
            df: DataFrame with a DatetimeIndex.

        Returns:
            True if no business-day gaps are detected.

        Raises:
            TypeError: If the index is not a DatetimeIndex.
        """
        if not isinstance(df.index, pd.DatetimeIndex):
            logger.warning("Index is not a DatetimeIndex; skipping continuity check.")
            return True
        if len(df) < 2:
            return True
        expected = pd.date_range(df.index[0], df.index[-1], freq=BDay())
        actual_set = set(df.index.normalize())
        expected_set = set(expected.normalize())
        gaps = expected_set - actual_set
        ok = len(gaps) == 0
        if not ok:
            logger.warning("Found %d business-day gaps in index.", len(gaps))
        else:
            logger.info("Date continuity check passed.")
        return ok

    def check_price_reasonableness(self, df: pd.DataFrame) -> bool:
        """Check that no prices are zero or negative.

        Args:
            df: DataFrame of price values.

        Returns:
            True if all prices are strictly positive.
        """
        numeric = df.select_dtypes(include=[np.number])
        if numeric.empty:
            return True
        has_non_positive = (numeric <= 0).any().any()
        ok = not has_non_positive
        if not ok:
            n_bad = int((numeric <= 0).sum().sum())
            logger.warning("Price reasonableness: found %d non-positive prices.", n_bad)
        else:
            logger.info("Price reasonableness check passed.")
        return ok

    def check_return_distribution(self, df: pd.DataFrame) -> dict[str, float]:
        """Compute descriptive statistics of returns derived from the data.

        Args:
            df: DataFrame of prices or returns. If values are large (>5)
                they are assumed to be prices and returns are computed.

        Returns:
            Dictionary of return statistics:
                ``mean``, ``std``, ``skew``, ``kurt``, ``min``, ``max``.
        """
        numeric = df.select_dtypes(include=[np.number])
        if numeric.empty:
            return {}
        # Heuristic: treat as prices if typical values are large
        if numeric.abs().median().median() > 5:
            returns = numeric.pct_change().dropna()
        else:
            returns = numeric.dropna()

        flat = returns.values.flatten()
        flat = flat[~np.isnan(flat)]
        if len(flat) == 0:
            return {}

        stats: dict[str, float] = {
            "mean": float(np.mean(flat)),
            "std": float(np.std(flat)),
            "skew": float(pd.Series(flat).skew()),
            "kurt": float(pd.Series(flat).kurt()),
            "min": float(np.min(flat)),
            "max": float(np.max(flat)),
        }
        logger.info("Return distribution stats: %s", stats)
        return stats

    def generate_report(self, df: pd.DataFrame) -> str:
        """Generate a human-readable data quality report.

        Args:
            df: DataFrame to report on.

        Returns:
            Multi-line string containing the data quality report.
        """
        results = self.validate(df)
        lines = [
            "=" * 60,
            "ML Alpha Lab — Data Quality Report",
            "=" * 60,
            f"Shape:              {df.shape}",
            f"Date range:         {df.index[0]} to {df.index[-1]}" if hasattr(df.index, "min") else "",
            f"Completeness:       {'PASS' if results['completeness'] else 'FAIL'}",
            f"Date continuity:    {'PASS' if results['date_continuity'] else 'FAIL'}",
            f"Price reasonableness: {'PASS' if results['price_reasonableness'] else 'FAIL'}",
            "",
            "Return distribution:",
        ]
        for k, v in results["return_distribution"].items():
            lines.append(f"  {k:<10}: {v:.6f}")
        lines += [
            "",
            f"Overall:            {'PASS' if results['passed'] else 'FAIL'}",
            "=" * 60,
        ]
        report = "\n".join(lines)
        logger.info("Generated data quality report.")
        return report
