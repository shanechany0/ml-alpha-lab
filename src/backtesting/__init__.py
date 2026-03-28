"""Backtesting module for ML Alpha Lab."""

from src.backtesting.backtest_engine import WalkForwardBacktest
from src.backtesting.capacity_analysis import CapacityAnalyzer
from src.backtesting.performance_metrics import (
    PerformanceMetrics,
    beta_alpha,
    calmar_ratio,
    cvar,
    drawdown_series,
    hit_rate,
    information_ratio,
    max_drawdown,
    profit_factor,
    sharpe_ratio,
    sortino_ratio,
    var,
    volatility,
)
from src.backtesting.report_generator import ReportGenerator
from src.backtesting.vectorized_backtest import VectorizedBacktest

__all__ = [
    "PerformanceMetrics",
    "VectorizedBacktest",
    "WalkForwardBacktest",
    "CapacityAnalyzer",
    "ReportGenerator",
    # convenience functions
    "sharpe_ratio",
    "sortino_ratio",
    "calmar_ratio",
    "max_drawdown",
    "drawdown_series",
    "hit_rate",
    "profit_factor",
    "volatility",
    "var",
    "cvar",
    "information_ratio",
    "beta_alpha",
]
