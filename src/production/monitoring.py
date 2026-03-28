"""Production model monitoring: drift detection and performance alerts."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


class ModelMonitor:
    """Monitors ML model and strategy performance in production.

    Detects feature distribution drift (KS test), performance degradation,
    and raises alerts for Sharpe and drawdown breaches.

    Attributes:
        alert_threshold: KS p-value threshold for drift alerts.
        lookback_window: Default window for rolling metric computation.
    """

    def __init__(self, config: dict | None = None) -> None:
        """Initialize ModelMonitor.

        Args:
            config: Optional configuration dict with keys:
                - alert_threshold (float): Defaults to 0.05.
                - lookback_window (int): Defaults to 63.
        """
        cfg = config or {}
        self.alert_threshold: float = cfg.get("alert_threshold", 0.05)
        self.lookback_window: int = cfg.get("lookback_window", 63)

    def detect_drift(
        self,
        current_features: pd.DataFrame,
        reference_features: pd.DataFrame,
    ) -> dict[str, float]:
        """Run Kolmogorov-Smirnov test for distribution drift per feature.

        Args:
            current_features: Recent feature matrix.
            reference_features: Reference (training-period) feature matrix.

        Returns:
            Dict mapping feature name to KS test p-value.
        """
        drift_scores: dict[str, float] = {}
        shared_cols = [
            c for c in current_features.columns if c in reference_features.columns
        ]
        for col in shared_cols:
            cur = current_features[col].dropna().values
            ref = reference_features[col].dropna().values
            if len(cur) < 5 or len(ref) < 5:
                drift_scores[col] = 1.0
                continue
            _, p_val = stats.ks_2samp(cur, ref)
            drift_scores[col] = float(p_val)

        return drift_scores

    def detect_performance_degradation(
        self,
        recent_returns: pd.Series,
        reference_returns: pd.Series,
    ) -> dict[str, bool]:
        """Compare recent vs. reference performance on key metrics.

        Args:
            recent_returns: Returns from the recent evaluation window.
            reference_returns: Returns from the reference (backtest) period.

        Returns:
            Dict with bool flags: sharpe_degraded, vol_increased,
            mean_return_degraded.
        """
        def ann_sharpe(r: pd.Series) -> float:
            std = float(r.std())
            return float(r.mean() * np.sqrt(252)) / (std * np.sqrt(252) + 1e-12)

        recent_sr = ann_sharpe(recent_returns.dropna())
        ref_sr = ann_sharpe(reference_returns.dropna())
        recent_vol = float(recent_returns.std() * np.sqrt(252))
        ref_vol = float(reference_returns.std() * np.sqrt(252))

        return {
            "sharpe_degraded": recent_sr < 0.5 * ref_sr,
            "vol_increased": recent_vol > 1.5 * ref_vol,
            "mean_return_degraded": float(recent_returns.mean()) < float(
                reference_returns.mean()
            ) * 0.5,
        }

    def sharpe_alert(
        self,
        returns: pd.Series,
        window: int = 63,
        threshold: float = 0.0,
    ) -> bool:
        """Alert when rolling Sharpe ratio falls below threshold.

        Args:
            returns: Daily return series.
            window: Rolling window length. Defaults to 63.
            threshold: Minimum acceptable Sharpe. Defaults to 0.0.

        Returns:
            True if the latest rolling Sharpe is below threshold.
        """
        r = returns.dropna()
        if len(r) < window:
            return False
        recent = r.iloc[-window:]
        roll_sr = float(recent.mean() * np.sqrt(252)) / (
            float(recent.std() * np.sqrt(252)) + 1e-12
        )
        return roll_sr < threshold

    def drawdown_alert(
        self,
        returns: pd.Series,
        threshold: float = 0.1,
    ) -> bool:
        """Alert when current drawdown exceeds threshold.

        Args:
            returns: Daily return series.
            threshold: Maximum tolerable drawdown (positive). Defaults to 0.1.

        Returns:
            True if current drawdown ≥ threshold.
        """
        cum = (1 + returns.dropna()).cumprod()
        rolling_max = cum.cummax()
        current_dd = float((cum.iloc[-1] - rolling_max.iloc[-1]) / rolling_max.iloc[-1])
        return current_dd <= -threshold

    def feature_drift_alert(
        self,
        current_features: pd.DataFrame,
        reference_features: pd.DataFrame,
        threshold: float = 0.05,
    ) -> list[str]:
        """Return names of features with statistically significant drift.

        Args:
            current_features: Recent feature values.
            reference_features: Reference feature values.
            threshold: KS p-value threshold. Defaults to 0.05.

        Returns:
            List of feature names flagged for drift.
        """
        drift_scores = self.detect_drift(current_features, reference_features)
        return [feat for feat, p in drift_scores.items() if p < threshold]

    def log_monitoring_metrics(self, metrics: dict[str, Any]) -> None:
        """Log monitoring metrics to MLflow (if available), else to logger.

        Args:
            metrics: Dict of metric name to scalar value.
        """
        try:
            import mlflow  # noqa: PLC0415

            for key, val in metrics.items():
                if isinstance(val, (int, float)):
                    mlflow.log_metric(key, val)
                else:
                    mlflow.log_param(key, str(val))
        except Exception:
            logger.info("Monitoring metrics: %s", metrics)
