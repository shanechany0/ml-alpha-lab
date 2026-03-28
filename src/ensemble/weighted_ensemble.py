"""Weighted ensemble for the ML Alpha Lab trading system."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from src.models.base_model import BaseModel

logger = logging.getLogger(__name__)


class WeightedEnsemble:
    """Ensemble that combines model predictions using IC-based weights.

    Supports three weighting strategies controlled by the ``weighting``
    parameter:

    * ``"performance"`` — weights proportional to absolute validation IC.
    * ``"equal"`` — uniform weights across all models.
    * ``"inverse_error"`` — weights inversely proportional to validation
      RMSE (better models contribute more).

    Weights are dynamically updatable at inference time via
    ``update_weights``.

    Attributes:
        models: List of ``BaseModel`` instances.
        weighting: Weighting strategy string.
        weights: Current weight array of shape (n_models,).
        is_fitted: Whether the ensemble has been fitted.
        model_names: Human-readable model identifiers.
    """

    def __init__(
        self,
        models: list[BaseModel],
        weighting: str = "performance",
    ) -> None:
        """Initialize the weighted ensemble.

        Args:
            models: List of ``BaseModel`` instances.
            weighting: Weighting strategy. One of ``"performance"``,
                ``"equal"``, or ``"inverse_error"``.
        """
        self.models = models
        self.weighting = weighting
        self.weights: np.ndarray = np.ones(len(models)) / len(models)
        self.is_fitted: bool = False
        self.model_names: list[str] = [f"{type(m).__name__}_{i}" for i, m in enumerate(models)]

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
    ) -> "WeightedEnsemble":
        """Fit all base models and compute ensemble weights from validation IC.

        Args:
            X_train: Training feature matrix of shape (n_samples, n_features).
            y_train: Training targets of shape (n_samples,).
            X_val: Optional validation features used to compute weights.
            y_val: Optional validation targets.

        Returns:
            Self, for method chaining.
        """
        if isinstance(X_train, pd.DataFrame):
            X_train = X_train.values
        if isinstance(X_val, pd.DataFrame):
            X_val = X_val.values

        for i, model in enumerate(self.models):
            logger.info("Fitting model %d/%d: %s", i + 1, len(self.models), self.model_names[i])
            model.fit(X_train, y_train, X_val, y_val)

        if X_val is not None and y_val is not None and self.weighting != "equal":
            val_preds = [m.predict(X_val) for m in self.models]
            self.weights = self._compute_weights(val_preds, np.array(y_val))
        else:
            self.weights = np.ones(len(self.models)) / len(self.models)

        self.is_fitted = True
        logger.info("WeightedEnsemble fitted. Weights: %s", dict(zip(self.model_names, self.weights)))
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generate weighted-average predictions.

        Args:
            X: Feature matrix of shape (n_samples, n_features).

        Returns:
            Weighted predictions of shape (n_samples,).

        Raises:
            RuntimeError: If the ensemble has not been fitted.
        """
        if not self.is_fitted:
            raise RuntimeError("WeightedEnsemble has not been fitted. Call fit() first.")

        if isinstance(X, pd.DataFrame):
            X = X.values

        preds_matrix = np.column_stack([m.predict(X) for m in self.models])
        return preds_matrix @ self.weights

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Generate weighted-average probability predictions.

        For regression models (no ``predict_proba``), falls back to
        ``predict``.

        Args:
            X: Feature matrix of shape (n_samples, n_features).

        Returns:
            Weighted predictions array.
        """
        if isinstance(X, pd.DataFrame):
            X = X.values

        all_preds = []
        for model in self.models:
            if hasattr(model, "predict_proba"):
                all_preds.append(model.predict_proba(X))
            else:
                all_preds.append(model.predict(X))

        # Handle 1-D (regression) and 2-D (classification) outputs uniformly
        if all_preds[0].ndim == 1:
            preds_matrix = np.column_stack(all_preds)
            return preds_matrix @ self.weights
        else:
            # Weighted sum of probability matrices
            return sum(w * p for w, p in zip(self.weights, all_preds))

    def _compute_weights(
        self,
        predictions: list[np.ndarray],
        y_val: np.ndarray,
    ) -> np.ndarray:
        """Compute normalised weights from model performance on validation data.

        For ``"performance"`` strategy uses absolute Spearman IC;
        for ``"inverse_error"`` uses inverse RMSE. Falls back to
        uniform weights when all models have zero performance.

        Args:
            predictions: List of prediction arrays, one per model.
            y_val: True validation targets.

        Returns:
            Normalised weight array of shape (n_models,).
        """
        raw_weights = np.zeros(len(predictions))

        for i, preds in enumerate(predictions):
            if self.weighting == "performance":
                ic, _ = spearmanr(preds, y_val)
                raw_weights[i] = abs(ic) if not np.isnan(ic) else 0.0
            elif self.weighting == "inverse_error":
                rmse = float(np.sqrt(np.mean((preds - y_val) ** 2)))
                raw_weights[i] = 1.0 / (rmse + 1e-10)
            else:
                raw_weights[i] = 1.0

        total = raw_weights.sum()
        if total == 0 or np.isnan(total):
            logger.warning("All model weights are zero or NaN; falling back to uniform weights.")
            return np.ones(len(predictions)) / len(predictions)

        return raw_weights / total

    def update_weights(self, new_performance: dict[str, float]) -> None:
        """Dynamically update weights from an external performance dictionary.

        Useful for online/live trading scenarios where model performance
        is tracked over a rolling window.

        Args:
            new_performance: Dictionary mapping model name to its latest
                performance score (e.g. rolling IC). Unknown model names
                are ignored; missing models receive zero weight.
        """
        raw_weights = np.array([abs(new_performance.get(name, 0.0)) for name in self.model_names], dtype=float)
        total = raw_weights.sum()
        if total == 0 or np.isnan(total):
            logger.warning("update_weights: all new weights are zero; keeping current weights.")
            return
        self.weights = raw_weights / total
        logger.info("Weights updated: %s", dict(zip(self.model_names, self.weights)))
