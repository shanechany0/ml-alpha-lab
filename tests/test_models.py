"""Tests for ML models."""

import numpy as np
import pandas as pd
import pytest
from unittest.mock import MagicMock, patch
from src.models.lightgbm_model import LightGBMModel
from src.models.xgboost_model import XGBoostModel


@pytest.fixture
def regression_data():
    """Create synthetic regression data."""
    np.random.seed(42)
    n, d = 500, 20
    X = np.random.randn(n, d)
    y = X[:, 0] * 0.5 + X[:, 1] * 0.3 + np.random.randn(n) * 0.1
    split = int(n * 0.8)
    return X[:split], y[:split], X[split:], y[split:]


class TestLightGBMModel:
    def test_fit_predict_shape(self, regression_data):
        X_train, y_train, X_val, y_val = regression_data
        with patch("mlflow.start_run"), patch("mlflow.log_params"), patch("mlflow.log_metrics"):
            model = LightGBMModel(config={"n_estimators": 50, "learning_rate": 0.1, "verbose": -1})
            model.fit(X_train, y_train, X_val, y_val)
            preds = model.predict(X_val)
        assert preds.shape == (len(X_val),)

    def test_feature_importance_not_none(self, regression_data):
        X_train, y_train, X_val, y_val = regression_data
        with patch("mlflow.start_run"), patch("mlflow.log_params"), patch("mlflow.log_metrics"):
            model = LightGBMModel(config={"n_estimators": 50, "verbose": -1})
            model.fit(X_train, y_train, X_val, y_val)
            importance = model.get_feature_importance()
        assert importance is not None
        assert len(importance) == X_train.shape[1]

    def test_score_returns_dict(self, regression_data):
        X_train, y_train, X_val, y_val = regression_data
        with patch("mlflow.start_run"), patch("mlflow.log_params"), patch("mlflow.log_metrics"):
            model = LightGBMModel(config={"n_estimators": 50, "verbose": -1})
            model.fit(X_train, y_train, X_val, y_val)
            scores = model.score(X_val, y_val)
        assert isinstance(scores, dict)


class TestXGBoostModel:
    def test_fit_predict_shape(self, regression_data):
        X_train, y_train, X_val, y_val = regression_data
        with patch("mlflow.start_run"), patch("mlflow.log_params"), patch("mlflow.log_metrics"):
            model = XGBoostModel(config={"n_estimators": 50, "learning_rate": 0.1})
            model.fit(X_train, y_train, X_val, y_val)
            preds = model.predict(X_val)
        assert preds.shape == (len(X_val),)

    def test_feature_importance(self, regression_data):
        X_train, y_train, X_val, y_val = regression_data
        with patch("mlflow.start_run"), patch("mlflow.log_params"), patch("mlflow.log_metrics"):
            model = XGBoostModel(config={"n_estimators": 50})
            model.fit(X_train, y_train)
            importance = model.get_feature_importance()
        assert importance is not None
