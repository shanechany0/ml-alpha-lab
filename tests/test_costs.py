"""Tests for transaction cost models."""

import pandas as pd
import numpy as np
import pytest
from src.costs.transaction_costs import TransactionCostModel
from src.costs.market_impact import AlmgrenChrissModel, SquareRootImpactModel


@pytest.fixture
def trade_data():
    np.random.seed(42)
    n = 100
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    tickers = ["AAPL", "MSFT", "GOOGL"]
    signals = pd.DataFrame(np.random.randn(n, 3) / 10, index=dates, columns=tickers)
    prices = pd.DataFrame(
        {t: 100 * np.cumprod(1 + np.random.normal(0.0005, 0.01, n)) for t in tickers},
        index=dates,
    )
    return signals, prices


class TestTransactionCostModel:
    def test_spread_cost_positive(self):
        model = TransactionCostModel()
        cost = model.compute_spread_cost(100000, spread_bps=5.0)
        assert cost > 0

    def test_commission_positive(self):
        model = TransactionCostModel()
        cost = model.compute_commission(100000, commission_bps=10.0)
        assert cost > 0

    def test_total_cost_series(self, trade_data):
        signals, prices = trade_data
        model = TransactionCostModel()
        costs = model.total_cost(signals, prices)
        assert isinstance(costs, pd.Series)
        assert (costs >= 0).all()


class TestAlmgrenChrissModel:
    def test_temporary_impact_positive(self):
        model = AlmgrenChrissModel()
        impact = model.temporary_impact(trade_rate=1000, adv=100000)
        assert impact >= 0

    def test_permanent_impact_positive(self):
        model = AlmgrenChrissModel()
        impact = model.permanent_impact(total_shares=5000, adv=100000)
        assert impact >= 0

    def test_optimal_trajectory_sums_to_total(self):
        model = AlmgrenChrissModel()
        total_shares = 10000.0
        trajectory = model.optimal_trajectory(total_shares, adv=100000, n_periods=10)
        assert abs(trajectory.sum() - total_shares) < 1.0


class TestSquareRootImpactModel:
    def test_impact_increases_with_size(self):
        model = SquareRootImpactModel(coefficient=0.1)
        adv = 1e6
        vol = 0.01
        impact_small = model.impact(trade_size=1000, adv=adv, volatility=vol)
        impact_large = model.impact(trade_size=10000, adv=adv, volatility=vol)
        assert impact_large > impact_small
