"""Tests for data loading, cleaning, and validation."""


import numpy as np
import pandas as pd
import pytest

from src.data.data_cleaner import DataCleaner
from src.data.data_validator import DataValidator


@pytest.fixture
def sample_prices():
    """Create sample price DataFrame for testing."""
    dates = pd.date_range("2020-01-01", periods=100, freq="B")
    tickers = ["AAPL", "MSFT", "GOOGL"]
    np.random.seed(42)
    prices = pd.DataFrame(
        np.random.uniform(100, 200, size=(100, 3)) * np.cumprod(1 + np.random.normal(0.0005, 0.01, size=(100, 3)), axis=0),
        index=dates,
        columns=tickers,
    )
    return prices


@pytest.fixture
def sample_returns(sample_prices):
    return sample_prices.pct_change().dropna()


class TestDataCleaner:
    def test_handle_missing_ffill(self, sample_prices):
        df = sample_prices.copy()
        df.iloc[5, 0] = np.nan
        cleaner = DataCleaner()
        cleaned = cleaner.handle_missing_values(df, method="ffill")
        assert not cleaned.isnull().any().any()

    def test_winsorize(self, sample_returns):
        cleaner = DataCleaner()
        result = cleaner.winsorize(sample_returns)
        assert result.shape == sample_returns.shape
        for col in result.columns:
            assert result[col].max() <= sample_returns[col].quantile(0.99) + 1e-10

    def test_detect_outliers(self, sample_returns):
        cleaner = DataCleaner()
        result = cleaner.detect_outliers(sample_returns, n_std=3.0)
        assert isinstance(result, pd.DataFrame)
        assert result.shape == sample_returns.shape

    def test_compute_returns_log(self, sample_prices):
        cleaner = DataCleaner()
        returns = cleaner.compute_returns(sample_prices, method="log")
        assert isinstance(returns, pd.DataFrame)
        assert returns.shape[0] == sample_prices.shape[0] - 1


class TestDataValidator:
    def test_check_completeness_pass(self, sample_prices):
        validator = DataValidator()
        result = validator.check_completeness(sample_prices)
        assert result is True

    def test_check_completeness_fail(self, sample_prices):
        df = sample_prices.copy()
        df.iloc[:60, :] = np.nan
        validator = DataValidator()
        result = validator.check_completeness(df, max_missing_pct=0.01)
        assert result is False

    def test_check_price_reasonableness(self, sample_prices):
        validator = DataValidator()
        result = validator.check_price_reasonableness(sample_prices)
        assert result is True

    def test_validate_returns_report(self, sample_prices):
        validator = DataValidator()
        report = validator.generate_report(sample_prices)
        assert isinstance(report, str)
        assert len(report) > 0
