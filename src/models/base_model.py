"""Abstract base class for all ML models in the trading system."""

from __future__ import annotations

import logging
import pickle
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class BaseModel(ABC):
    """Abstract base class for all trading ML models.

    Provides a common interface for fitting, predicting, scoring,
    saving/loading, and logging metrics to MLflow.

    Attributes:
        config: Configuration dictionary for the model.
        model: The underlying model object (set by subclasses).
        feature_names: Optional list of feature column names.
    """

    def __init__(self, config: dict | None = None) -> None:
        """Initialize the base model.

        Args:
            config: Model configuration dictionary. Defaults to empty dict.
        """
        self.config: dict = config or {}
        self.model: Any = None
        self.feature_names: list[str] | None = None
        self._mlflow_run_id: str | None = None

        mlflow_uri = self.config.get("mlflow_tracking_uri", "mlruns")
        mlflow.set_tracking_uri(mlflow_uri)

    @abstractmethod
    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
    ) -> "BaseModel":
        """Fit the model on training data.

        Args:
            X_train: Training features of shape (n_samples, n_features).
            y_train: Training targets of shape (n_samples,).
            X_val: Optional validation features.
            y_val: Optional validation targets.

        Returns:
            Self, to allow method chaining.
        """

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generate predictions for input features.

        Args:
            X: Feature matrix of shape (n_samples, n_features).

        Returns:
            Predictions array of shape (n_samples,).
        """

    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Generate probability predictions for input features.

        Args:
            X: Feature matrix of shape (n_samples, n_features).

        Returns:
            Probability predictions array.
        """

    def score(self, X: np.ndarray, y: np.ndarray) -> dict[str, float]:
        """Compute evaluation metrics on data.

        Computes Information Coefficient (IC / Spearman correlation),
        directional accuracy, mean absolute error, and RMSE.

        Args:
            X: Feature matrix of shape (n_samples, n_features).
            y: True targets of shape (n_samples,).

        Returns:
            Dictionary mapping metric names to float values.
        """
        from scipy.stats import spearmanr
        from sklearn.metrics import mean_absolute_error

        preds = self.predict(X)

        ic, _ = spearmanr(preds, y)
        mae = mean_absolute_error(y, preds)
        rmse = float(np.sqrt(np.mean((preds - y) ** 2)))

        # Directional accuracy: fraction of times predicted sign matches true sign
        direction_acc = float(np.mean(np.sign(preds) == np.sign(y)))

        metrics = {
            "ic": float(ic) if not np.isnan(ic) else 0.0,
            "mae": float(mae),
            "rmse": float(rmse),
            "directional_accuracy": direction_acc,
        }
        return metrics

    def save(self, path: str) -> None:
        """Save the model to disk using pickle.

        Args:
            path: File system path to save the model.
        """
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "wb") as f:
            pickle.dump(self, f)
        logger.info("Model saved to %s", path)

    def load(self, path: str) -> "BaseModel":
        """Load a model from disk.

        Args:
            path: File system path to load the model from.

        Returns:
            Loaded model instance.

        Raises:
            FileNotFoundError: If the file does not exist.
        """
        load_path = Path(path)
        if not load_path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")
        with open(load_path, "rb") as f:
            loaded: BaseModel = pickle.load(f)
        logger.info("Model loaded from %s", path)
        return loaded

    def log_metrics(self, metrics: dict[str, float]) -> None:
        """Log metrics to the active MLflow run.

        If no active run exists, starts a new one. Metrics with NaN
        or infinite values are skipped with a warning.

        Args:
            metrics: Dictionary mapping metric names to float values.
        """
        active_run = mlflow.active_run()
        if active_run is None:
            mlflow.start_run()

        for key, value in metrics.items():
            if np.isnan(value) or np.isinf(value):
                logger.warning("Skipping metric %s with invalid value %s", key, value)
                continue
            mlflow.log_metric(key, value)

    def get_feature_importance(self) -> pd.Series | None:
        """Return feature importance scores if available.

        Returns:
            Series indexed by feature name with importance scores,
            or None if not available.
        """
        return None
