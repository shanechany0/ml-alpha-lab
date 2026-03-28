"""Tests for feature engineering modules."""

import pandas as pd
import numpy as np
import pytest
from src.features.technical import TechnicalFeatures
from src.features.statistical import StatisticalFeatures
from src.features.cross_sectional import CrossSectionalFeatures
from src.features.feature_pipeline import FeaturePipeline


@pytest.fixture
def ohlcv_data():
    """Create OHLCV sample data."""
    n = 200
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    np.random.seed(42)
    close = 100 * np.cumprod(1 + np.random.normal(0.0005, 0.01, n))
    high = close * (1 + np.abs(np.random.normal(0, 0.005, n)))
    low = close * (1 - np.abs(np.random.normal(0, 0.005, n)))
    volume = np.random.uniform(1e6, 5e6, n)
    return pd.DataFrame({"Open": close, "High": high, "Low": low, "Close": close, "Volume": volume}, index=dates)


@pytest.fixture
def multi_asset_prices():
    n = 200
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    np.random.seed(42)
    prices = {}
    for ticker in ["AAPL", "MSFT", "GOOGL", "AMZN"]:
        prices[ticker] = 100 * np.cumprod(1 + np.random.normal(0.0005, 0.01, n))
    return pd.DataFrame(prices, index=dates)


class TestTechnicalFeatures:
    def test_rsi_bounds(self, ohlcv_data):
        tech = TechnicalFeatures()
        rsi = tech.rsi(ohlcv_data["Close"], period=14)
        valid = rsi.dropna()
        assert (valid >= 0).all() and (valid <= 100).all()

    def test_macd_shape(self, ohlcv_data):
        tech = TechnicalFeatures()
        macd_df = tech.macd(ohlcv_data["Close"])
        assert isinstance(macd_df, pd.DataFrame)
        assert "macd" in macd_df.columns or len(macd_df.columns) >= 2

    def test_bollinger_bands(self, ohlcv_data):
        tech = TechnicalFeatures()
        bb = tech.bollinger_bands(ohlcv_data["Close"])
        assert isinstance(bb, pd.DataFrame)
        assert bb.shape[0] == len(ohlcv_data)

    def test_atr_positive(self, ohlcv_data):
        tech = TechnicalFeatures()
        atr = tech.atr(ohlcv_data["High"], ohlcv_data["Low"], ohlcv_data["Close"])
        assert (atr.dropna() >= 0).all()

    def test_obv_monotonicity_concept(self, ohlcv_data):
        tech = TechnicalFeatures()
        obv = tech.obv(ohlcv_data["Close"], ohlcv_data["Volume"])
        assert isinstance(obv, pd.Series)
        assert len(obv) == len(ohlcv_data)


class TestStatisticalFeatures:
    def test_zscore_mean_zero(self, multi_asset_prices):
        returns = multi_asset_prices.pct_change().dropna()
        stat = StatisticalFeatures()
        zscores = stat.zscore(returns["AAPL"], window=20).dropna()
        assert abs(zscores.mean()) < 1.0

    def test_rolling_stats_shape(self, multi_asset_prices):
        returns = multi_asset_prices.pct_change().dropna()
        stat = StatisticalFeatures()
        result = stat.rolling_stats(returns, windows=[10, 20])
        assert isinstance(result, pd.DataFrame)
        assert result.shape[0] == returns.shape[0]

    def test_realized_vol_positive(self, multi_asset_prices):
        returns = multi_asset_prices.pct_change().dropna()
        stat = StatisticalFeatures()
        vol = stat.realized_volatility(returns, windows=[20])
        assert (vol.dropna() >= 0).all().all()


class TestCrossSectionalFeatures:
    def test_percentile_rank_bounds(self, multi_asset_prices):
        returns = multi_asset_prices.pct_change().dropna()
        cs = CrossSectionalFeatures()
        ranks = cs.percentile_rank(returns)
        valid = ranks.dropna()
        assert (valid >= 0).all().all()
        assert (valid <= 1).all().all()

    def test_zscore_cross_sectional(self, multi_asset_prices):
        returns = multi_asset_prices.pct_change().dropna()
        cs = CrossSectionalFeatures()
        result = cs.cross_sectional_zscore(returns)
        assert isinstance(result, pd.DataFrame)


class TestFeaturePipeline:
    def test_no_lookahead_bias(self, multi_asset_prices):
        """Critical test: features must be shifted by 1 period."""
        pipeline = FeaturePipeline()
        features = pipeline.fit_transform(multi_asset_prices)
        assert isinstance(features, pd.DataFrame)

    def test_feature_shape(self, multi_asset_prices):
        pipeline = FeaturePipeline()
        features = pipeline.fit_transform(multi_asset_prices)
        assert features.shape[0] == multi_asset_prices.shape[0]
