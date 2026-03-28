"""Tests for alpha signal generation and evaluation."""

import numpy as np
import pandas as pd
import pytest

from src.signals.alpha_signals import AlphaSignals
from src.signals.signal_combination import SignalCombiner
from src.signals.signal_evaluation import SignalEvaluator


@pytest.fixture
def price_return_data():
    n = 300
    dates = pd.date_range("2018-01-01", periods=n, freq="B")
    np.random.seed(42)
    tickers = [f"STOCK_{i}" for i in range(10)]
    prices = pd.DataFrame(
        {t: 100 * np.cumprod(1 + np.random.normal(0.0003, 0.01, n)) for t in tickers},
        index=dates
    )
    returns = prices.pct_change().dropna()
    return prices, returns


class TestAlphaSignals:
    def test_momentum_signal_shape(self, price_return_data):
        prices, returns = price_return_data
        signals = AlphaSignals()
        mom = signals.time_series_momentum(returns, windows=[20])
        assert isinstance(mom, pd.DataFrame)
        assert mom.shape == returns.shape

    def test_mean_reversion_signal(self, price_return_data):
        prices, returns = price_return_data
        signals = AlphaSignals()
        mr = signals.mean_reversion(prices)
        assert isinstance(mr, pd.DataFrame)

    def test_signals_cross_sectional_mean_zero(self, price_return_data):
        prices, returns = price_return_data
        signals = AlphaSignals()
        mom = signals.cross_sectional_momentum(returns, window=20).dropna()
        assert abs(mom.mean().mean()) < 0.5


class TestSignalEvaluator:
    def test_ic_computation(self, price_return_data):
        prices, returns = price_return_data
        signals_obj = AlphaSignals()
        signals = signals_obj.time_series_momentum(returns).dropna()
        fwd_returns = returns.shift(-5)
        evaluator = SignalEvaluator()
        ic = evaluator.information_coefficient(signals, fwd_returns, periods=[5])
        assert isinstance(ic, pd.DataFrame)

    def test_information_ratio(self):
        np.random.seed(42)
        ic_series = pd.Series(np.random.normal(0.05, 0.1, 100))
        evaluator = SignalEvaluator()
        ir = evaluator.information_ratio(ic_series)
        assert isinstance(ir, float)
        assert ir > 0

    def test_turnover_analysis(self, price_return_data):
        prices, returns = price_return_data
        signals_obj = AlphaSignals()
        signals = signals_obj.cross_sectional_momentum(returns)
        evaluator = SignalEvaluator()
        result = evaluator.turnover_analysis(signals)
        assert isinstance(result, dict)
        assert "mean_turnover" in result


class TestSignalCombiner:
    def test_equal_weight_combination(self, price_return_data):
        prices, returns = price_return_data
        signals_obj = AlphaSignals()
        s1 = signals_obj.time_series_momentum(returns, windows=[20])
        s2 = signals_obj.mean_reversion(prices)
        signals_df = pd.concat([s1.iloc[:, 0].rename("mom"), s2.iloc[:, 0].rename("mr")], axis=1).dropna()
        combiner = SignalCombiner(method="equal_weight")
        combined = combiner.combine(signals_df)
        assert isinstance(combined, pd.Series)
        assert len(combined) == len(signals_df)
