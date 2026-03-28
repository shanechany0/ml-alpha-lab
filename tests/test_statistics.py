"""Tests for statistical analysis modules."""

import numpy as np
import pandas as pd
import pytest

from src.statistics.bootstrap import Bootstrap
from src.statistics.deflated_sharpe import DeflatedSharpeRatio, compute_dsr
from src.statistics.hypothesis_testing import HypothesisTester


@pytest.fixture
def return_series():
    np.random.seed(42)
    return pd.Series(np.random.normal(0.0005, 0.01, 252))


class TestHypothesisTester:
    def test_bonferroni_length(self):
        tester = HypothesisTester()
        p_values = [0.001, 0.01, 0.05, 0.1, 0.5]
        rejected, adjusted = tester.bonferroni_correction(p_values)
        assert len(rejected) == len(p_values)
        assert len(adjusted) == len(p_values)

    def test_holm_more_powerful_than_bonferroni(self):
        tester = HypothesisTester()
        p_values = [0.001, 0.01, 0.04, 0.045, 0.5]
        rej_bonf, _ = tester.bonferroni_correction(p_values, alpha=0.05)
        rej_holm, _ = tester.holm_correction(p_values, alpha=0.05)
        assert sum(rej_holm) >= sum(rej_bonf)

    def test_fdr_correction(self):
        tester = HypothesisTester()
        p_values = [0.001, 0.01, 0.02, 0.04, 0.2, 0.5]
        rejected, adjusted = tester.fdr_correction(p_values)
        assert len(rejected) == len(p_values)

    def test_t_test_returns(self, return_series):
        tester = HypothesisTester()
        result = tester.t_test_returns(return_series)
        assert "t_stat" in result
        assert "p_value" in result


class TestDeflatedSharpeRatio:
    def test_dsr_less_than_observed(self, return_series):
        dsr = DeflatedSharpeRatio()
        result = dsr.compute(return_series, n_trials=100)
        assert "dsr" in result
        assert "observed_sr" in result
        assert result["dsr"] <= result["observed_sr"]

    def test_compute_dsr_function(self, return_series):
        result = compute_dsr(return_series, n_trials=50)
        assert isinstance(result, dict)
        assert "dsr" in result

    def test_more_trials_lower_dsr(self, return_series):
        dsr = DeflatedSharpeRatio()
        result_few = dsr.compute(return_series, n_trials=10)
        result_many = dsr.compute(return_series, n_trials=1000)
        assert result_many["dsr"] <= result_few["dsr"]


class TestBootstrap:
    def test_confidence_interval_bounds(self, return_series):
        boot = Bootstrap(n_bootstrap=200, random_state=42)
        lo, hi = boot.confidence_interval(return_series, statistic_fn=np.mean)
        assert lo < hi

    def test_sharpe_ci(self, return_series):
        boot = Bootstrap(n_bootstrap=200, random_state=42)
        lo, hi = boot.sharpe_confidence_interval(return_series)
        assert lo < hi
        assert isinstance(lo, float)

    def test_block_bootstrap_length(self, return_series):
        boot = Bootstrap(n_bootstrap=100, block_size=20, random_state=42)
        samples = boot.block_bootstrap(return_series, statistic_fn=np.mean)
        assert len(samples) == 100
