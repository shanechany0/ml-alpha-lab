"""Tests for backtesting engine and performance metrics."""

import numpy as np
import pandas as pd
import pytest

from src.backtesting.performance_metrics import PerformanceMetrics
from src.backtesting.vectorized_backtest import VectorizedBacktest


@pytest.fixture
def strategy_returns():
    """Create sample strategy returns."""
    np.random.seed(42)
    dates = pd.date_range("2015-01-01", periods=500, freq="B")
    returns = pd.Series(np.random.normal(0.0005, 0.01, 500), index=dates)
    return returns


@pytest.fixture
def multi_asset_signals_returns():
    np.random.seed(42)
    n, k = 300, 5
    dates = pd.date_range("2018-01-01", periods=n, freq="B")
    tickers = [f"STK_{i}" for i in range(k)]
    signals = pd.DataFrame(np.random.randn(n, k), index=dates, columns=tickers)
    signals = signals.div(signals.abs().sum(axis=1), axis=0)
    returns = pd.DataFrame(np.random.normal(0.0005, 0.01, (n, k)), index=dates, columns=tickers)
    return signals, returns


class TestPerformanceMetrics:
    def test_sharpe_ratio_type(self, strategy_returns):
        pm = PerformanceMetrics(risk_free_rate=0.02)
        sr = pm.sharpe_ratio(strategy_returns)
        assert isinstance(sr, float)

    def test_sharpe_positive_for_good_strategy(self):
        np.random.seed(42)
        returns = pd.Series(np.random.normal(0.002, 0.005, 252))
        pm = PerformanceMetrics(risk_free_rate=0.0)
        assert pm.sharpe_ratio(returns) > 0

    def test_max_drawdown_negative_or_zero(self, strategy_returns):
        pm = PerformanceMetrics()
        dd = pm.max_drawdown(strategy_returns)
        assert dd <= 0

    def test_sortino_ratio(self, strategy_returns):
        pm = PerformanceMetrics()
        sr = pm.sortino_ratio(strategy_returns)
        assert isinstance(sr, float)

    def test_compute_all_returns_dict(self, strategy_returns):
        pm = PerformanceMetrics()
        results = pm.compute_all(strategy_returns)
        assert isinstance(results, dict)
        assert "sharpe_ratio" in results
        assert "max_drawdown" in results

    def test_var_cvar_relationship(self, strategy_returns):
        pm = PerformanceMetrics()
        var = pm.var(strategy_returns)
        cvar = pm.cvar(strategy_returns)
        assert cvar <= var

    def test_hit_rate_between_0_1(self, strategy_returns):
        pm = PerformanceMetrics()
        hr = pm.hit_rate(strategy_returns)
        assert 0 <= hr <= 1


class TestVectorizedBacktest:
    def test_run_returns_dict(self, multi_asset_signals_returns):
        signals, returns = multi_asset_signals_returns
        vbt = VectorizedBacktest()
        result = vbt.run(signals, returns)
        assert isinstance(result, dict)

    def test_strategy_returns_shape(self, multi_asset_signals_returns):
        signals, returns = multi_asset_signals_returns
        vbt = VectorizedBacktest()
        strat_returns = vbt.compute_strategy_returns(signals, returns)
        assert isinstance(strat_returns, pd.Series)
        assert len(strat_returns) <= len(returns)

    def test_turnover_non_negative(self, multi_asset_signals_returns):
        signals, returns = multi_asset_signals_returns
        vbt = VectorizedBacktest()
        turnover = vbt.compute_turnover(signals)
        assert (turnover.dropna() >= 0).all()
