"""XGBoost model implementation for the ML Alpha Lab trading system."""

from __future__ import annotations

import logging
from typing import Any

import mlflow
import numpy as np
import pandas as pd
import xgboost as xgb

from src.models.base_model import BaseModel

logger = logging.getLogger(__name__)

_DEFAULT_PARAMS: dict[str, Any] = {
    "n_estimators": 1000,
    "learning_rate": 0.01,
    "max_depth": 6,
    "min_child_weight": 5,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "tree_method": "hist",
    "n_jobs": -1,
    "random_state": 42,
}


class XGBoostModel(BaseModel):
    """XGBoost gradient boosting model for return prediction.

    Supports regression (default) and classification tasks.
    Feature importance is computed in a SHAP-compatible manner
    (``weight``, ``gain``, ``cover``).

    Attributes:
        config: Configuration dictionary.
        model: Fitted ``xgb.XGBRegressor`` or ``xgb.XGBClassifier``.
        task: Either ``"regression"`` or ``"classification"``.
        early_stopping_rounds: Rounds without improvement before stop.
        feature_names: Column names set at fit time.
    """

    def __init__(self, config: dict | None = None) -> None:
        """Initialize the XGBoost model.

        Args:
            config: Configuration dictionary. Recognises top-level key
                ``task`` (``"regression"`` | ``"classification"``) and
                an ``xgboost`` sub-dict of model hyperparameters.
        """
        super().__init__(config)
        self.task: str = self.config.get("task", "regression")
        xgb_cfg = dict(self.config.get("xgboost", {}))
        self.early_stopping_rounds: int = xgb_cfg.pop("early_stopping_rounds", 50)
        self._eval_metric: str | None = xgb_cfg.pop("eval_metric", None)
        self._base_params: dict[str, Any] = {**_DEFAULT_PARAMS, **xgb_cfg}
        if self._eval_metric is not None:
            self._base_params["eval_metric"] = self._eval_metric
        # model is built lazily in fit() so early_stopping_rounds is only
        # injected when an eval_set is actually available.
        self.model: xgb.XGBRegressor | xgb.XGBClassifier | None = None

    def _build_model(self, with_early_stopping: bool) -> Any:
        """Build an XGBRegressor/Classifier with optional early stopping.

        Args:
            with_early_stopping: If True, add early_stopping_rounds to params.

        Returns:
            Uninitialised XGBoost model instance.
        """
        params = dict(self._base_params)
        if with_early_stopping:
            params["early_stopping_rounds"] = self.early_stopping_rounds
        if self.task == "classification":
            return xgb.XGBClassifier(**params)
        return xgb.XGBRegressor(**params)

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
    ) -> "XGBoostModel":
        """Train the XGBoost model with optional early stopping.

        Args:
            X_train: Training feature matrix of shape (n_samples, n_features).
            y_train: Training target array of shape (n_samples,).
            X_val: Validation feature matrix used for early stopping.
            y_val: Validation target array.

        Returns:
            Self, for method chaining.
        """
        if isinstance(X_train, pd.DataFrame):
            self.feature_names = list(X_train.columns)

        has_val = X_val is not None and y_val is not None
        self.model = self._build_model(with_early_stopping=has_val)

        fit_kwargs: dict[str, Any] = {"verbose": False}
        if has_val:
            fit_kwargs["eval_set"] = [(X_val, y_val)]

        with mlflow.start_run(nested=True):
            mlflow.log_params(self.model.get_params())
            self.model.fit(X_train, y_train, **fit_kwargs)

            train_metrics = self.score(X_train, y_train)
            mlflow.log_metrics({f"train_{k}": v for k, v in train_metrics.items()})

            if has_val:
                val_metrics = self.score(X_val, y_val)
                mlflow.log_metrics({f"val_{k}": v for k, v in val_metrics.items()})

        best_iter = getattr(self.model, "best_iteration", "N/A")
        logger.info("XGBoost training complete. Best iteration: %s", best_iter)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generate point predictions.

        Args:
            X: Feature matrix of shape (n_samples, n_features).

        Returns:
            Predictions array of shape (n_samples,).
        """
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Generate probability predictions.

        For regression models, returns ``predict()`` output directly.
        For classifiers, returns class probability matrix.

        Args:
            X: Feature matrix of shape (n_samples, n_features).

        Returns:
            Probability array or regression predictions.
        """
        if self.task == "classification":
            return self.model.predict_proba(X)
        return self.predict(X)

    def get_feature_importance(self) -> pd.Series:
        """Return SHAP-ready feature importance (gain).

        Returns the ``"gain"`` importance type, which is the most
        suitable for SHAP-style interpretation.

        Returns:
            Series indexed by feature name, values are gain importance
            scores sorted descending.

        Raises:
            RuntimeError: If the model has not been fitted yet.
        """
        if self.model is None or not hasattr(self.model, "get_booster"):
            raise RuntimeError("Model has not been fitted yet.")

        booster = self.model.get_booster()
        scores = booster.get_score(importance_type="gain")

        names = self.feature_names or list(scores.keys())
        importance = pd.Series(scores, name="gain_importance")

        # Reindex to include zero-importance features
        if self.feature_names:
            importance = importance.reindex(names, fill_value=0.0)

        return importance.sort_values(ascending=False)
