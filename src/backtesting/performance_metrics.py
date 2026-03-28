"""Performance metrics for backtesting and strategy evaluation."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


class PerformanceMetrics:
    """Computes standard performance metrics for strategy evaluation.

    Attributes:
        risk_free_rate: Annual risk-free rate (e.g. 0.05 for 5%).
        annualization: Trading days per year for annualization.
    """

    def __init__(self, risk_free_rate: float = 0.05, annualization: int = 252) -> None:
        """Initializes PerformanceMetrics.

        Args:
            risk_free_rate: Annual risk-free rate.
            annualization: Number of periods per year for annualization.
        """
        self.risk_free_rate = risk_free_rate
        self.annualization = annualization

    def sharpe_ratio(self, returns: pd.Series) -> float:
        """Computes annualized Sharpe ratio.

        Args:
            returns: Daily strategy returns.

        Returns:
            Annualized Sharpe ratio.
        """
        if returns.empty or returns.std() == 0:
            return 0.0
        daily_rf = self.risk_free_rate / self.annualization
        excess = returns - daily_rf
        return float(excess.mean() / excess.std() * np.sqrt(self.annualization))

    def sortino_ratio(self, returns: pd.Series) -> float:
        """Computes annualized Sortino ratio using downside deviation.

        Args:
            returns: Daily strategy returns.

        Returns:
            Annualized Sortino ratio.
        """
        if returns.empty:
            return 0.0
        daily_rf = self.risk_free_rate / self.annualization
        excess = returns - daily_rf
        downside = excess[excess < 0]
        if len(downside) == 0 or downside.std() == 0:
            return 0.0
        downside_std = np.sqrt((downside**2).mean())
        return float(excess.mean() / downside_std * np.sqrt(self.annualization))

    def calmar_ratio(self, returns: pd.Series) -> float:
        """Computes Calmar ratio (annualized return / max drawdown).

        Args:
            returns: Daily strategy returns.

        Returns:
            Calmar ratio (positive is better).
        """
        mdd = self.max_drawdown(returns)
        if mdd == 0:
            return 0.0
        annual_return = returns.mean() * self.annualization
        return float(annual_return / abs(mdd))

    def max_drawdown(self, returns: pd.Series) -> float:
        """Computes maximum drawdown as a negative number.

        Args:
            returns: Daily strategy returns.

        Returns:
            Maximum drawdown (negative value, e.g. -0.25 for 25% drawdown).
        """
        if returns.empty:
            return 0.0
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max
        return float(drawdown.min())

    def drawdown_series(self, returns: pd.Series) -> pd.Series:
        """Computes the drawdown time series.

        Args:
            returns: Daily strategy returns.

        Returns:
            Series of drawdown values (negative).
        """
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.cummax()
        return (cumulative - running_max) / running_max

    def hit_rate(self, returns: pd.Series) -> float:
        """Computes win rate (fraction of positive return periods).

        Args:
            returns: Daily strategy returns.

        Returns:
            Fraction of periods with positive returns.
        """
        if returns.empty:
            return 0.0
        return float((returns > 0).mean())

    def profit_factor(self, returns: pd.Series) -> float:
        """Computes profit factor (gross profit / gross loss).

        Args:
            returns: Daily strategy returns.

        Returns:
            Profit factor; inf if no losses, 0 if no profits.
        """
        gains = returns[returns > 0].sum()
        losses = returns[returns < 0].abs().sum()
        if losses == 0:
            return float("inf")
        return float(gains / losses)

    def average_win_loss(self, returns: pd.Series) -> dict[str, float]:
        """Computes average win, average loss, and their ratio.

        Args:
            returns: Daily strategy returns.

        Returns:
            Dictionary with keys 'avg_win', 'avg_loss', 'win_loss_ratio'.
        """
        wins = returns[returns > 0]
        losses = returns[returns < 0]
        avg_win = float(wins.mean()) if len(wins) > 0 else 0.0
        avg_loss = float(losses.mean()) if len(losses) > 0 else 0.0
        ratio = abs(avg_win / avg_loss) if avg_loss != 0 else float("inf")
        return {"avg_win": avg_win, "avg_loss": avg_loss, "win_loss_ratio": ratio}

    def volatility(self, returns: pd.Series, annualized: bool = True) -> float:
        """Computes return volatility (standard deviation).

        Args:
            returns: Daily strategy returns.
            annualized: Whether to annualize the volatility.

        Returns:
            Volatility (annualized if requested).
        """
        if returns.empty:
            return 0.0
        vol = float(returns.std())
        if annualized:
            vol *= np.sqrt(self.annualization)
        return vol

    def var(self, returns: pd.Series, confidence: float = 0.95) -> float:
        """Computes historical Value at Risk (VaR).

        Args:
            returns: Daily strategy returns.
            confidence: Confidence level (e.g. 0.95 for 95% VaR).

        Returns:
            VaR as a negative number representing potential loss.
        """
        if returns.empty:
            return 0.0
        return float(np.percentile(returns, (1 - confidence) * 100))

    def cvar(self, returns: pd.Series, confidence: float = 0.95) -> float:
        """Computes Conditional Value at Risk (CVaR / Expected Shortfall).

        Args:
            returns: Daily strategy returns.
            confidence: Confidence level.

        Returns:
            CVaR as a negative number (average loss beyond VaR threshold).
        """
        if returns.empty:
            return 0.0
        var_threshold = self.var(returns, confidence)
        tail = returns[returns <= var_threshold]
        if tail.empty:
            return var_threshold
        return float(tail.mean())

    def information_ratio(self, returns: pd.Series, benchmark: pd.Series) -> float:
        """Computes the Information Ratio relative to a benchmark.

        Args:
            returns: Strategy daily returns.
            benchmark: Benchmark daily returns.

        Returns:
            Annualized information ratio.
        """
        active = returns - benchmark
        if active.std() == 0:
            return 0.0
        return float(active.mean() / active.std() * np.sqrt(self.annualization))

    def beta_alpha(self, returns: pd.Series, benchmark: pd.Series) -> tuple[float, float]:
        """Computes beta and annualized alpha via OLS regression.

        Args:
            returns: Strategy daily returns.
            benchmark: Benchmark daily returns.

        Returns:
            Tuple of (beta, annualized_alpha).
        """
        aligned = pd.concat([returns, benchmark], axis=1).dropna()
        if len(aligned) < 2:
            return 0.0, 0.0
        slope, intercept, _, _, _ = stats.linregress(aligned.iloc[:, 1], aligned.iloc[:, 0])
        annualized_alpha = intercept * self.annualization
        return float(slope), float(annualized_alpha)

    def compute_all(
        self, returns: pd.Series, benchmark: pd.Series | None = None
    ) -> dict[str, float]:
        """Computes all available metrics and returns them as a flat dictionary.

        Args:
            returns: Strategy daily returns.
            benchmark: Optional benchmark daily returns for relative metrics.

        Returns:
            Dictionary of metric name to value.
        """
        wl = self.average_win_loss(returns)
        results: dict[str, float] = {
            "sharpe_ratio": self.sharpe_ratio(returns),
            "sortino_ratio": self.sortino_ratio(returns),
            "calmar_ratio": self.calmar_ratio(returns),
            "max_drawdown": self.max_drawdown(returns),
            "hit_rate": self.hit_rate(returns),
            "profit_factor": self.profit_factor(returns),
            "avg_win": wl["avg_win"],
            "avg_loss": wl["avg_loss"],
            "win_loss_ratio": wl["win_loss_ratio"],
            "volatility": self.volatility(returns),
            "var_95": self.var(returns, 0.95),
            "cvar_95": self.cvar(returns, 0.95),
            "annual_return": float(returns.mean() * self.annualization),
        }
        if benchmark is not None:
            results["information_ratio"] = self.information_ratio(returns, benchmark)
            beta, alpha = self.beta_alpha(returns, benchmark)
            results["beta"] = beta
            results["alpha"] = alpha
        return results


# ---------------------------------------------------------------------------
# Module-level convenience functions
# ---------------------------------------------------------------------------

def sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.05, annualization: int = 252) -> float:
    """Module-level convenience wrapper for Sharpe ratio."""
    return PerformanceMetrics(risk_free_rate, annualization).sharpe_ratio(returns)


def sortino_ratio(returns: pd.Series, risk_free_rate: float = 0.05, annualization: int = 252) -> float:
    """Module-level convenience wrapper for Sortino ratio."""
    return PerformanceMetrics(risk_free_rate, annualization).sortino_ratio(returns)


def calmar_ratio(returns: pd.Series, risk_free_rate: float = 0.05, annualization: int = 252) -> float:
    """Module-level convenience wrapper for Calmar ratio."""
    return PerformanceMetrics(risk_free_rate, annualization).calmar_ratio(returns)


def max_drawdown(returns: pd.Series) -> float:
    """Module-level convenience wrapper for max drawdown."""
    return PerformanceMetrics().max_drawdown(returns)


def drawdown_series(returns: pd.Series) -> pd.Series:
    """Module-level convenience wrapper for drawdown series."""
    return PerformanceMetrics().drawdown_series(returns)


def hit_rate(returns: pd.Series) -> float:
    """Module-level convenience wrapper for hit rate."""
    return PerformanceMetrics().hit_rate(returns)


def profit_factor(returns: pd.Series) -> float:
    """Module-level convenience wrapper for profit factor."""
    return PerformanceMetrics().profit_factor(returns)


def volatility(returns: pd.Series, annualized: bool = True, annualization: int = 252) -> float:
    """Module-level convenience wrapper for volatility."""
    return PerformanceMetrics(annualization=annualization).volatility(returns, annualized)


def var(returns: pd.Series, confidence: float = 0.95) -> float:
    """Module-level convenience wrapper for VaR."""
    return PerformanceMetrics().var(returns, confidence)


def cvar(returns: pd.Series, confidence: float = 0.95) -> float:
    """Module-level convenience wrapper for CVaR."""
    return PerformanceMetrics().cvar(returns, confidence)


def information_ratio(
    returns: pd.Series, benchmark: pd.Series, risk_free_rate: float = 0.05, annualization: int = 252
) -> float:
    """Module-level convenience wrapper for information ratio."""
    return PerformanceMetrics(risk_free_rate, annualization).information_ratio(returns, benchmark)


def beta_alpha(
    returns: pd.Series, benchmark: pd.Series, risk_free_rate: float = 0.05, annualization: int = 252
) -> tuple[float, float]:
    """Module-level convenience wrapper for beta and alpha."""
    return PerformanceMetrics(risk_free_rate, annualization).beta_alpha(returns, benchmark)
