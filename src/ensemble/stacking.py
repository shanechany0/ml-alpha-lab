"""Stacking ensemble for the ML Alpha Lab trading system."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, cross_val_predict

from src.models.base_model import BaseModel

logger = logging.getLogger(__name__)


class StackingEnsemble:
    """Stacking ensemble that trains a meta-learner on out-of-fold predictions.

    Base models are fitted using *k*-fold cross-validation to generate
    out-of-fold (OOF) predictions. The meta-learner (default: Ridge
    regression) is then trained on the concatenated OOF predictions.

    Attributes:
        base_models: List of fitted or unfitted ``BaseModel`` instances.
        meta_model: Sklearn-compatible meta-learner.
        n_folds: Number of CV folds for OOF generation.
        oof_predictions: Array of OOF predictions from base models.
        is_fitted: Whether the ensemble has been fitted.
    """

    def __init__(
        self,
        base_models: list[BaseModel],
        meta_model: Any | None = None,
        n_folds: int = 5,
    ) -> None:
        """Initialize the stacking ensemble.

        Args:
            base_models: List of ``BaseModel`` instances to use as
                the first-level learners.
            meta_model: Optional sklearn-compatible second-level learner.
                Defaults to ``Ridge(alpha=1.0)``.
            n_folds: Number of folds for cross-validated OOF predictions.
        """
        self.base_models = base_models
        self.meta_model: Any = meta_model if meta_model is not None else Ridge(alpha=1.0)
        self.n_folds = n_folds
        self.oof_predictions: np.ndarray | None = None
        self.is_fitted: bool = False

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
    ) -> "StackingEnsemble":
        """Fit base models via CV OOF and train the meta-learner.

        Base models that expose an sklearn-compatible ``fit``/``predict``
        API are wrapped transparently. Each base model is also re-fitted
        on the full training set after OOF generation so that it is
        ready for inference.

        Args:
            X_train: Training feature matrix of shape (n_samples, n_features).
            y_train: Training targets of shape (n_samples,).
            X_val: Unused (kept for API consistency).
            y_val: Unused.

        Returns:
            Self, for method chaining.
        """
        if isinstance(X_train, pd.DataFrame):
            X_train = X_train.values
        y_train = np.array(y_train)

        n_samples = X_train.shape[0]
        oof_matrix = np.zeros((n_samples, len(self.base_models)))

        for col_idx, base_model in enumerate(self.base_models):
            logger.info("Generating OOF predictions for base model %d/%d", col_idx + 1, len(self.base_models))
            wrapper = _BaseModelWrapper(base_model)
            oof_preds = cross_val_predict(wrapper, X_train, y_train, cv=self.n_folds)
            oof_matrix[:, col_idx] = oof_preds

            # Re-fit on full training data for inference
            base_model.fit(X_train, y_train)

        self.oof_predictions = oof_matrix
        self.meta_model.fit(oof_matrix, y_train)
        self.is_fitted = True
        logger.info("Stacking ensemble fitted. Meta-learner: %s", type(self.meta_model).__name__)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generate stacked predictions.

        Collects base model predictions into a matrix then passes them
        through the meta-learner.

        Args:
            X: Feature matrix of shape (n_samples, n_features).

        Returns:
            Meta-learner predictions of shape (n_samples,).

        Raises:
            RuntimeError: If the ensemble has not been fitted.
        """
        if not self.is_fitted:
            raise RuntimeError("StackingEnsemble has not been fitted. Call fit() first.")

        if isinstance(X, pd.DataFrame):
            X = X.values

        base_preds = self._base_predictions(X)
        return self.meta_model.predict(base_preds)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Generate probability predictions via the meta-learner.

        Falls back to ``predict`` if the meta-model has no
        ``predict_proba`` method.

        Args:
            X: Feature matrix of shape (n_samples, n_features).

        Returns:
            Probability array or regression predictions.
        """
        if hasattr(self.meta_model, "predict_proba"):
            if isinstance(X, pd.DataFrame):
                X = X.values
            base_preds = self._base_predictions(X)
            return self.meta_model.predict_proba(base_preds)
        return self.predict(X)

    def get_model_weights(self) -> dict[str, float]:
        """Return meta-learner coefficients as a weight dictionary.

        Works for linear meta-models (e.g. ``Ridge``) that expose a
        ``coef_`` attribute.

        Returns:
            Dictionary mapping ``"model_{i}"`` to the corresponding
            meta-learner coefficient.

        Raises:
            RuntimeError: If the ensemble has not been fitted or the
                meta-learner does not have ``coef_`` attribute.
        """
        if not self.is_fitted:
            raise RuntimeError("StackingEnsemble has not been fitted.")
        if not hasattr(self.meta_model, "coef_"):
            raise RuntimeError(f"Meta-model {type(self.meta_model).__name__} does not expose coef_.")

        coefs = np.atleast_1d(self.meta_model.coef_)
        return {f"model_{i}": float(c) for i, c in enumerate(coefs)}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _base_predictions(self, X: np.ndarray) -> np.ndarray:
        """Collect predictions from all base models.

        Args:
            X: Feature matrix.

        Returns:
            Matrix of shape (n_samples, n_base_models).
        """
        return np.column_stack([m.predict(X) for m in self.base_models])


class _BaseModelWrapper(BaseEstimator):
    """Thin sklearn-compatible wrapper around a ``BaseModel``.

    Required because ``cross_val_predict`` expects objects that implement
    ``fit`` and ``predict`` with the standard sklearn signature.
    """

    def __init__(self, model: BaseModel) -> None:
        self.model = model

    def fit(self, X: np.ndarray, y: np.ndarray) -> "_BaseModelWrapper":
        self.model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)
