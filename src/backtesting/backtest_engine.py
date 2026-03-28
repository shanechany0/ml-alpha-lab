"""Walk-forward backtesting engine with purged cross-validation."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from src.backtesting.performance_metrics import PerformanceMetrics

logger = logging.getLogger(__name__)


class WalkForwardBacktest:
    """Walk-forward backtesting engine with purged cross-validation.

    Implements expanding-window or rolling-window walk-forward analysis
    with a configurable embargo/gap to prevent leakage.

    Attributes:
        config: Configuration dictionary.
        metrics: PerformanceMetrics instance.
    """

    def __init__(self, config_path: str | None = None) -> None:
        """Initializes WalkForwardBacktest, optionally loading YAML config.

        Args:
            config_path: Path to a YAML configuration file. If None,
                defaults are used.
        """
        self.config: dict[str, Any] = {}
        if config_path is not None:
            path = Path(config_path)
            if path.exists():
                with path.open() as fh:
                    self.config = yaml.safe_load(fh) or {}
            else:
                logger.warning("Config path %s not found; using defaults.", config_path)

        self.metrics = PerformanceMetrics(
            risk_free_rate=self.config.get("risk_free_rate", 0.05),
            annualization=self.config.get("annualization", 252),
        )
        self.n_splits: int = self.config.get("n_splits", 5)
        self.train_size: int = self.config.get("train_size", 504)  # ~2 years
        self.test_size: int = self.config.get("test_size", 126)  # ~6 months
        self.gap: int = self.config.get("gap", 21)  # ~1 month embargo

    def run(
        self,
        features: pd.DataFrame,
        returns: pd.DataFrame,
        model: Any,
    ) -> dict[str, Any]:
        """Executes a full walk-forward backtest.

        Args:
            features: Feature DataFrame (dates × features).
            returns: Asset or portfolio returns (dates × assets or single column).
            model: Sklearn-compatible model with fit() and predict() methods.

        Returns:
            Aggregated results dictionary containing OOS predictions,
            performance metrics, and per-fold details.
        """
        n_samples = len(features)
        splits = self._create_splits(n_samples)
        fold_results: list[dict[str, Any]] = []

        common_idx = features.index.intersection(returns.index)
        features = features.loc[common_idx]
        returns = returns.loc[common_idx]

        for fold_num, (train_idx, test_idx) in enumerate(splits):
            train_idx = self._purge_overlap(train_idx, test_idx, self.gap)
            X_train = features.iloc[train_idx]
            y_train = returns.iloc[train_idx].squeeze()
            X_test = features.iloc[test_idx]
            y_test = returns.iloc[test_idx].squeeze()

            try:
                model.fit(X_train, y_train)
                predictions = model.predict(X_test)
            except Exception as exc:
                logger.error("Fold %d failed: %s", fold_num, exc)
                continue

            pred_series = pd.Series(predictions, index=features.index[test_idx], name="predictions")
            fold_metrics = self.metrics.compute_all(y_test)
            fold_results.append(
                {
                    "fold": fold_num,
                    "predictions": pred_series,
                    "actual": y_test,
                    "metrics": fold_metrics,
                    "train_size": len(train_idx),
                    "test_size": len(test_idx),
                }
            )

        return self.aggregate_results(fold_results)

    def _create_splits(self, n_samples: int) -> list[tuple[np.ndarray, np.ndarray]]:
        """Creates purged walk-forward cross-validation splits.

        Args:
            n_samples: Total number of samples.

        Returns:
            List of (train_indices, test_indices) tuples.
        """
        splits: list[tuple[np.ndarray, np.ndarray]] = []
        step = (n_samples - self.train_size) // max(self.n_splits, 1)
        step = max(step, self.test_size)

        start = self.train_size
        while start + self.test_size <= n_samples:
            train_end = start
            train_start = max(0, train_end - self.train_size)
            test_start = train_end
            test_end = min(n_samples, test_start + self.test_size)

            train_idx = np.arange(train_start, train_end)
            test_idx = np.arange(test_start, test_end)
            splits.append((train_idx, test_idx))
            start += step
            if len(splits) >= self.n_splits:
                break

        return splits

    def _purge_overlap(
        self, train_idx: np.ndarray, test_idx: np.ndarray, gap: int
    ) -> np.ndarray:
        """Removes training samples that overlap with the test window (embargo).

        Args:
            train_idx: Training period indices.
            test_idx: Test period indices.
            gap: Number of periods to exclude before the test start.

        Returns:
            Purged training indices with the embargo zone removed.
        """
        if len(test_idx) == 0:
            return train_idx
        test_start = test_idx[0]
        cutoff = test_start - gap
        return train_idx[train_idx < cutoff]

    def compute_oos_predictions(
        self, features: pd.DataFrame, returns: pd.DataFrame, model: Any
    ) -> pd.Series:
        """Computes concatenated out-of-sample predictions across all folds.

        Args:
            features: Feature DataFrame.
            returns: Returns DataFrame.
            model: Sklearn-compatible model.

        Returns:
            Series of OOS predictions indexed by date.
        """
        result = self.run(features, returns, model)
        oos_preds: list[pd.Series] = []
        for fold in result.get("fold_details", []):
            oos_preds.append(fold["predictions"])
        if not oos_preds:
            return pd.Series(dtype=float)
        return pd.concat(oos_preds).sort_index()

    def aggregate_results(self, fold_results: list[dict[str, Any]]) -> dict[str, Any]:
        """Aggregates per-fold results into a summary.

        Args:
            fold_results: List of per-fold result dictionaries.

        Returns:
            Aggregated dictionary with mean metrics, OOS returns,
            and per-fold details.
        """
        if not fold_results:
            return {"fold_details": [], "aggregate_metrics": {}, "oos_returns": pd.Series(dtype=float)}

        all_preds: list[pd.Series] = [f["predictions"] for f in fold_results]
        all_actual: list[pd.Series] = [f["actual"] for f in fold_results]
        oos_returns = pd.concat(all_actual).sort_index()
        oos_preds = pd.concat(all_preds).sort_index()

        metric_keys = list(fold_results[0]["metrics"].keys())
        agg_metrics: dict[str, float] = {}
        for key in metric_keys:
            vals = [f["metrics"].get(key, np.nan) for f in fold_results]
            agg_metrics[f"mean_{key}"] = float(np.nanmean(vals))
            agg_metrics[f"std_{key}"] = float(np.nanstd(vals))

        oos_full_metrics = self.metrics.compute_all(oos_returns)

        return {
            "fold_details": fold_results,
            "aggregate_metrics": agg_metrics,
            "oos_metrics": oos_full_metrics,
            "oos_returns": oos_returns,
            "oos_predictions": oos_preds,
        }

    def plot_equity_curve(self, returns: pd.Series) -> None:
        """Plots the cumulative equity curve and drawdown.

        Args:
            returns: Strategy daily return series.
        """
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

        ax1.plot(cumulative.index, cumulative.values, linewidth=1.5, color="steelblue")
        ax1.set_title("Equity Curve")
        ax1.set_ylabel("Cumulative Return")
        ax1.grid(True, alpha=0.3)

        ax2.fill_between(drawdown.index, drawdown.values, 0, color="red", alpha=0.4)
        ax2.set_title("Drawdown")
        ax2.set_ylabel("Drawdown")
        ax2.set_xlabel("Date")
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Walk-forward backtesting engine")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config file")
    parser.add_argument("--features", type=str, required=True, help="Path to features CSV")
    parser.add_argument("--returns", type=str, required=True, help="Path to returns CSV")
    args = parser.parse_args()

    features_df = pd.read_csv(args.features, index_col=0, parse_dates=True)
    returns_df = pd.read_csv(args.returns, index_col=0, parse_dates=True)

    from sklearn.linear_model import Ridge

    engine = WalkForwardBacktest(config_path=args.config)
    results = engine.run(features_df, returns_df, Ridge())
    print("OOS Metrics:")
    for k, v in results["oos_metrics"].items():
        print(f"  {k}: {v:.4f}")
    engine.plot_equity_curve(results["oos_returns"])
