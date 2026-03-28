"""Tests for portfolio optimization modules."""

import numpy as np
import pandas as pd
import pytest

from src.portfolio.mean_variance import MeanVarianceOptimizer
from src.portfolio.risk_controls import RiskController
from src.portfolio.risk_parity import RiskParityOptimizer


@pytest.fixture
def portfolio_data():
    np.random.seed(42)
    n_assets = 5
    tickers = [f"ASSET_{i}" for i in range(n_assets)]
    returns_matrix = np.random.normal(0.0005, 0.01, (252, n_assets))
    expected_returns = pd.Series(np.random.uniform(0.0003, 0.001, n_assets), index=tickers)
    cov_matrix = pd.DataFrame(
        np.cov(returns_matrix.T),
        index=tickers,
        columns=tickers,
    )
    return expected_returns, cov_matrix


class TestMeanVarianceOptimizer:
    def test_weights_sum_to_one(self, portfolio_data):
        expected_returns, cov_matrix = portfolio_data
        optimizer = MeanVarianceOptimizer(config={"max_weight": 0.5, "min_weight": 0.0})
        weights = optimizer.optimize(expected_returns, cov_matrix)
        assert abs(weights.sum() - 1.0) < 1e-4

    def test_weights_within_bounds(self, portfolio_data):
        expected_returns, cov_matrix = portfolio_data
        max_w = 0.4
        optimizer = MeanVarianceOptimizer(config={"max_weight": max_w, "min_weight": 0.0})
        weights = optimizer.optimize(expected_returns, cov_matrix)
        assert (weights <= max_w + 1e-4).all()
        assert (weights >= -1e-4).all()

    def test_min_variance_portfolio(self, portfolio_data):
        _, cov_matrix = portfolio_data
        optimizer = MeanVarianceOptimizer()
        weights = optimizer.optimize_min_variance(cov_matrix)
        assert abs(weights.sum() - 1.0) < 1e-4


class TestRiskParityOptimizer:
    def test_weights_sum_to_one(self, portfolio_data):
        _, cov_matrix = portfolio_data
        optimizer = RiskParityOptimizer()
        weights = optimizer.optimize(cov_matrix)
        assert abs(weights.sum() - 1.0) < 1e-4

    def test_weights_positive(self, portfolio_data):
        _, cov_matrix = portfolio_data
        optimizer = RiskParityOptimizer()
        weights = optimizer.optimize(cov_matrix)
        assert (weights >= -1e-6).all()

    def test_inverse_vol_weights(self, portfolio_data):
        _, cov_matrix = portfolio_data
        vols = pd.Series(np.sqrt(np.diag(cov_matrix.values)), index=cov_matrix.index)
        optimizer = RiskParityOptimizer()
        weights = optimizer.optimize_inverse_vol(vols)
        assert abs(weights.sum() - 1.0) < 1e-4


class TestRiskController:
    def test_position_limits_enforced(self):
        tickers = [f"ASSET_{i}" for i in range(5)]
        weights = pd.Series([0.5, 0.2, 0.15, 0.1, 0.05], index=tickers)
        controller = RiskController()
        capped = controller.apply_position_limits(weights, max_weight=0.3)
        assert (capped <= 0.3 + 1e-6).all()

    def test_drawdown_circuit_breaker(self):
        np.random.seed(42)
        returns = pd.Series([-0.01] * 100)
        controller = RiskController()
        result = controller.check_drawdown_circuit_breaker(returns, threshold=0.20)
        assert isinstance(result, bool)
