"""
Critical tests to verify no look-ahead bias in the feature pipeline.

Look-ahead bias occurs when future information is used at time t.
This is the most critical correctness test for quantitative trading systems.
"""

import pandas as pd
import numpy as np
import pytest
from src.features.feature_pipeline import FeaturePipeline
from src.features.technical import TechnicalFeatures
from src.features.statistical import StatisticalFeatures
from src.signals.alpha_signals import AlphaSignals


@pytest.fixture
def price_data():
    """Create deterministic price data for look-ahead testing."""
    n = 100
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    np.random.seed(0)
    prices = pd.DataFrame(
        {
            "AAPL": 100 + np.cumsum(np.random.randn(n)),
            "MSFT": 200 + np.cumsum(np.random.randn(n)),
        },
        index=dates,
    )
    return prices


class TestNoLookaheadBias:
    """Verify that features computed at time t do not use information from t+1 or later."""

    def test_feature_pipeline_shifts_features(self, price_data):
        """Feature pipeline must shift all features by 1 period to prevent look-ahead."""
        pipeline = FeaturePipeline()
        features = pipeline.fit_transform(price_data)
        extended_prices = price_data.copy()
        new_row = pd.DataFrame(
            {"AAPL": [999999.0], "MSFT": [999999.0]},
            index=[price_data.index[-1] + pd.Timedelta(days=1)],
        )
        extended_prices = pd.concat([extended_prices, new_row])
        features_extended = pipeline.transform(extended_prices)
        common_idx = features.index.intersection(features_extended.index[:-1])
        if len(common_idx) > 0:
            diff = (features.loc[common_idx] - features_extended.loc[common_idx]).abs()
            assert diff.max().max() < 1e-6, "Feature pipeline has look-ahead bias!"

    def test_rsi_no_lookahead(self, price_data):
        """RSI computed on truncated data must match RSI on full data for shared dates."""
        tech = TechnicalFeatures()
        close = price_data["AAPL"]
        rsi_short = tech.rsi(close.iloc[:50], period=14)
        rsi_full = tech.rsi(close, period=14)
        common_idx = rsi_short.index.intersection(rsi_full.index)
        np.testing.assert_allclose(
            rsi_short.loc[common_idx].values,
            rsi_full.loc[common_idx].values,
            rtol=1e-5,
            err_msg="RSI has look-ahead bias!",
        )

    def test_macd_no_lookahead(self, price_data):
        """MACD should not use future data."""
        tech = TechnicalFeatures()
        close = price_data["AAPL"]
        macd_short = tech.macd(close.iloc[:60])
        macd_full = tech.macd(close)
        common_idx = macd_short.index.intersection(macd_full.index)
        if len(common_idx) > 0:
            np.testing.assert_allclose(
                macd_short.loc[common_idx].values,
                macd_full.loc[common_idx].values,
                rtol=1e-4,
                err_msg="MACD has look-ahead bias!",
            )

    def test_bollinger_bands_no_lookahead(self, price_data):
        """Bollinger Bands should not use future data."""
        tech = TechnicalFeatures()
        close = price_data["AAPL"]
        bb_short = tech.bollinger_bands(close.iloc[:60])
        bb_full = tech.bollinger_bands(close)
        common_idx = bb_short.index.intersection(bb_full.index)
        if len(common_idx) > 0:
            diff = (bb_short.loc[common_idx] - bb_full.loc[common_idx]).abs()
            assert diff.max().max() < 1e-5, "Bollinger Bands has look-ahead bias!"

    def test_zscore_no_lookahead(self, price_data):
        """Rolling z-score should not use future returns."""
        stat = StatisticalFeatures()
        returns = price_data.pct_change().dropna()
        zscore_short = stat.zscore(returns["AAPL"].iloc[:50], window=10)
        zscore_full = stat.zscore(returns["AAPL"], window=10)
        common_idx = zscore_short.index.intersection(zscore_full.index)
        if len(common_idx) > 0:
            np.testing.assert_allclose(
                zscore_short.loc[common_idx].dropna().values,
                zscore_full.loc[common_idx].dropna().values,
                rtol=1e-4,
                err_msg="Z-score has look-ahead bias!",
            )

    def test_momentum_signal_no_lookahead(self, price_data):
        """Momentum signals should not use future returns."""
        returns = price_data.pct_change().dropna()
        signals_obj = AlphaSignals()
        mom_short = signals_obj.time_series_momentum(returns.iloc[:50], windows=[10])
        mom_full = signals_obj.time_series_momentum(returns, windows=[10])
        common_idx = mom_short.index.intersection(mom_full.index)
        if len(common_idx) > 0:
            diff = (mom_short.loc[common_idx] - mom_full.loc[common_idx]).abs()
            assert diff.max().max() < 1e-6, "Momentum signal has look-ahead bias!"
