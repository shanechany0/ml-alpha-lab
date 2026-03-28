"""LightGBM model implementation for the ML Alpha Lab trading system."""

from __future__ import annotations

import logging
from typing import Any

import lightgbm as lgb
import mlflow
import numpy as np
import pandas as pd

from src.models.base_model import BaseModel

logger = logging.getLogger(__name__)

_DEFAULT_PARAMS: dict[str, Any] = {
    "n_estimators": 1000,
    "learning_rate": 0.01,
    "num_leaves": 63,
    "max_depth": -1,
    "min_child_samples": 20,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 0.1,
    "verbose": -1,
    "n_jobs": -1,
}


class LightGBMModel(BaseModel):
    """LightGBM gradient boosting model for return prediction.

    Supports both regression (default) and classification tasks,
    with optional Optuna-based hyperparameter optimisation.

    Attributes:
        config: Configuration dict. May contain a ``lightgbm`` sub-key
            with LightGBM-specific parameters.
        model: Fitted ``lgb.LGBMRegressor`` or ``lgb.LGBMClassifier``.
        task: Either ``"regression"`` or ``"classification"``.
        early_stopping_rounds: Number of rounds without improvement
            before training stops.
        feature_names: Column names passed at fit time (if any).
    """

    def __init__(self, config: dict | None = None) -> None:
        """Initialize the LightGBM model.

        Args:
            config: Configuration dictionary. Recognises top-level key
                ``task`` (``"regression"`` | ``"classification"``) and
                a ``lightgbm`` sub-dict of model hyperparameters.
        """
        super().__init__(config)
        self.task: str = self.config.get("task", "regression")
        lgbm_cfg = self.config.get("lightgbm", {})
        self.early_stopping_rounds: int = lgbm_cfg.pop("early_stopping_rounds", 50)

        params = {**_DEFAULT_PARAMS, **lgbm_cfg}

        if self.task == "classification":
            self.model = lgb.LGBMClassifier(**params)
        else:
            self.model = lgb.LGBMRegressor(**params)

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
    ) -> "LightGBMModel":
        """Train the LightGBM model with optional early stopping.

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

        callbacks = [lgb.log_evaluation(period=100)]
        fit_kwargs: dict[str, Any] = {"callbacks": callbacks}

        if X_val is not None and y_val is not None:
            fit_kwargs["eval_set"] = [(X_val, y_val)]
            callbacks.append(lgb.early_stopping(stopping_rounds=self.early_stopping_rounds, verbose=False))

        with mlflow.start_run(nested=True):
            mlflow.log_params(self.model.get_params())
            self.model.fit(X_train, y_train, **fit_kwargs)

            train_metrics = self.score(X_train, y_train)
            mlflow.log_metrics({f"train_{k}": v for k, v in train_metrics.items()})

            if X_val is not None and y_val is not None:
                val_metrics = self.score(X_val, y_val)
                mlflow.log_metrics({f"val_{k}": v for k, v in val_metrics.items()})

        logger.info("LightGBM training complete. Best iteration: %s", self.model.best_iteration_)
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
        """Return feature importance by both split count and gain.

        Returns:
            Series indexed by feature name, values are gain-based
            importance scores sorted descending.

        Raises:
            RuntimeError: If the model has not been fitted yet.
        """
        if self.model is None or not hasattr(self.model, "booster_"):
            raise RuntimeError("Model has not been fitted yet.")

        booster = self.model.booster_
        names = booster.feature_name()
        gain_imp = booster.feature_importance(importance_type="gain")
        importance = pd.Series(gain_imp, index=names, name="gain_importance")
        return importance.sort_values(ascending=False)

    def tune_hyperparameters(self, X: np.ndarray, y: np.ndarray, n_trials: int = 50) -> dict:
        """Tune hyperparameters using Optuna.

        Requires ``optuna`` to be installed (optional dependency).

        Args:
            X: Feature matrix for cross-validated tuning.
            y: Target array.
            n_trials: Number of Optuna trials to run.

        Returns:
            Dictionary of best hyperparameter values.

        Raises:
            ImportError: If ``optuna`` is not installed.
        """
        try:
            import optuna  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError("optuna is required for hyperparameter tuning: pip install optuna") from exc

        from sklearn.model_selection import cross_val_score  # noqa: PLC0415

        optuna.logging.set_verbosity(optuna.logging.WARNING)

        def objective(trial: optuna.Trial) -> float:
            params = {
                "num_leaves": trial.suggest_int("num_leaves", 20, 300),
                "learning_rate": trial.suggest_float("learning_rate", 1e-4, 0.3, log=True),
                "n_estimators": trial.suggest_int("n_estimators", 100, 2000),
                "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
                "verbose": -1,
            }
            if self.task == "classification":
                mdl = lgb.LGBMClassifier(**params)
                scoring = "roc_auc"
            else:
                mdl = lgb.LGBMRegressor(**params)
                scoring = "neg_mean_squared_error"

            scores = cross_val_score(mdl, X, y, cv=5, scoring=scoring, n_jobs=-1)
            return float(np.mean(scores))

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

        best_params = study.best_params
        logger.info("Best HPO params: %s", best_params)
        return best_params
