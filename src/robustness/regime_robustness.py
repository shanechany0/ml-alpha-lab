"""Regime-conditional robustness testing."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class RegimeRobustnessTester:
    """Tests strategy robustness across different market regimes.

    Classifies market states (bull/bear, volatility regimes, HMM-based)
    and evaluates strategy performance metrics within each regime.
    """

    def __init__(self, config: dict | None = None) -> None:
        """Initialize RegimeRobustnessTester.

        Args:
            config: Optional configuration dict (reserved for future use).
        """
        self._config = config or {}

    def analyze_by_regime(
        self,
        strategy_returns: pd.Series,
        market_returns: pd.Series,
        n_regimes: int = 3,
    ) -> pd.DataFrame:
        """Compute performance metrics per market regime.

        Regimes are identified by quantile-bucketing rolling market volatility.

        Args:
            strategy_returns: Strategy daily returns.
            market_returns: Market (benchmark) daily returns.
            n_regimes: Number of volatility-based regimes. Defaults to 3.

        Returns:
            DataFrame indexed by regime (0..n_regimes-1) with columns:
            mean_return, annualized_vol, sharpe, max_drawdown, n_days,
            market_return.
        """
        vol = market_returns.rolling(21).std().dropna()
        labels = pd.qcut(vol, n_regimes, labels=False)

        df = pd.DataFrame(
            {"strategy": strategy_returns, "market": market_returns, "regime": labels}
        ).dropna()

        records = {}
        for regime, group in df.groupby("regime"):
            r = group["strategy"]
            mkt = group["market"]
            ann_vol = float(r.std() * np.sqrt(252))
            ann_ret = float(r.mean() * 252)
            cum = (1 + r).cumprod()
            max_dd = float(
                ((cum - cum.cummax()) / (cum.cummax() + 1e-12)).min()
            )
            records[regime] = {
                "mean_return": float(r.mean()),
                "annualized_return": ann_ret,
                "annualized_vol": ann_vol,
                "sharpe": ann_ret / (ann_vol + 1e-12),
                "max_drawdown": max_dd,
                "n_days": int(len(r)),
                "market_return": float(mkt.mean()),
            }

        return pd.DataFrame(records).T

    def bull_bear_analysis(
        self,
        strategy_returns: pd.Series,
        market_returns: pd.Series,
    ) -> dict[str, dict]:
        """Compute strategy performance in bull and bear market conditions.

        Bull regime: rolling 63-day market return > 0.
        Bear regime: rolling 63-day market return ≤ 0.

        Args:
            strategy_returns: Strategy daily returns.
            market_returns: Market daily returns.

        Returns:
            Dict with keys "bull" and "bear", each containing:
            mean_return, annualized_vol, sharpe, max_drawdown, n_days.
        """
        trend = market_returns.rolling(63).mean()
        is_bull = trend > 0

        df = pd.DataFrame(
            {"strategy": strategy_returns, "is_bull": is_bull}
        ).dropna()

        result = {}
        for label, flag in [("bull", True), ("bear", False)]:
            r = df.loc[df["is_bull"] == flag, "strategy"]
            if len(r) == 0:
                result[label] = {}
                continue
            ann_vol = float(r.std() * np.sqrt(252))
            ann_ret = float(r.mean() * 252)
            cum = (1 + r).cumprod()
            max_dd = float(
                ((cum - cum.cummax()) / (cum.cummax() + 1e-12)).min()
            )
            result[label] = {
                "mean_return": float(r.mean()),
                "annualized_return": ann_ret,
                "annualized_vol": ann_vol,
                "sharpe": ann_ret / (ann_vol + 1e-12),
                "max_drawdown": max_dd,
                "n_days": int(len(r)),
            }

        return result

    def volatility_regime_analysis(
        self,
        strategy_returns: pd.Series,
        market_returns: pd.Series,
    ) -> dict[str, dict]:
        """Classify into low / medium / high volatility regimes.

        Thresholds are based on terciles of realized 21-day rolling volatility.

        Args:
            strategy_returns: Strategy daily returns.
            market_returns: Market daily returns.

        Returns:
            Dict with keys "low_vol", "med_vol", "high_vol", each containing:
            mean_return, annualized_vol, sharpe, max_drawdown, n_days.
        """
        vol = market_returns.rolling(21).std()
        q33, q66 = vol.quantile(1 / 3), vol.quantile(2 / 3)

        def label_vol(v: float) -> str:
            if v <= q33:
                return "low_vol"
            elif v <= q66:
                return "med_vol"
            return "high_vol"

        df = pd.DataFrame(
            {"strategy": strategy_returns, "vol": vol}
        ).dropna()
        df["vol_regime"] = df["vol"].apply(label_vol)

        result = {}
        for regime, group in df.groupby("vol_regime"):
            r = group["strategy"]
            ann_vol = float(r.std() * np.sqrt(252))
            ann_ret = float(r.mean() * 252)
            cum = (1 + r).cumprod()
            max_dd = float(
                ((cum - cum.cummax()) / (cum.cummax() + 1e-12)).min()
            )
            result[str(regime)] = {
                "mean_return": float(r.mean()),
                "annualized_return": ann_ret,
                "annualized_vol": ann_vol,
                "sharpe": ann_ret / (ann_vol + 1e-12),
                "max_drawdown": max_dd,
                "n_days": int(len(r)),
            }

        return result

    def conditional_performance(
        self,
        strategy_returns: pd.Series,
        condition: pd.Series,
        quantiles: int = 4,
    ) -> pd.DataFrame:
        """Analyze strategy performance conditional on an external factor.

        Args:
            strategy_returns: Strategy daily returns.
            condition: External factor series aligned with strategy_returns.
            quantiles: Number of quantile buckets. Defaults to 4.

        Returns:
            DataFrame indexed by quantile with columns: mean_return,
            annualized_vol, sharpe, n_days.
        """
        df = pd.DataFrame(
            {"strategy": strategy_returns, "condition": condition}
        ).dropna()

        df["bucket"] = pd.qcut(df["condition"], quantiles, labels=False, duplicates="drop")
        records = {}

        for bucket, group in df.groupby("bucket"):
            r = group["strategy"]
            ann_vol = float(r.std() * np.sqrt(252))
            ann_ret = float(r.mean() * 252)
            records[bucket] = {
                "mean_return": float(r.mean()),
                "annualized_return": ann_ret,
                "annualized_vol": ann_vol,
                "sharpe": ann_ret / (ann_vol + 1e-12),
                "n_days": int(len(r)),
            }

        return pd.DataFrame(records).T

    def robustness_score(
        self,
        strategy_returns: pd.Series,
        market_returns: pd.Series,
    ) -> float:
        """Compute a composite robustness score from 0 to 100.

        Combines: bull/bear Sharpe consistency, low/high vol Sharpe ratio,
        max drawdown, and % of rolling windows with positive Sharpe.

        Args:
            strategy_returns: Strategy daily returns.
            market_returns: Market daily returns.

        Returns:
            Composite robustness score in [0, 100].
        """
        score = 0.0
        max_score = 0.0

        # Bull/bear Sharpe consistency (25 pts)
        max_score += 25
        bb = self.bull_bear_analysis(strategy_returns, market_returns)
        if bb.get("bull") and bb.get("bear"):
            bull_sr = bb["bull"].get("sharpe", 0)
            bear_sr = bb["bear"].get("sharpe", 0)
            if bull_sr > 0:
                score += 12.5
            if bear_sr > 0:
                score += 12.5

        # Vol regime Sharpe (25 pts)
        max_score += 25
        vr = self.volatility_regime_analysis(strategy_returns, market_returns)
        n_positive = sum(1 for v in vr.values() if v.get("sharpe", 0) > 0)
        score += 25 * (n_positive / max(len(vr), 1))

        # Max drawdown (25 pts)
        max_score += 25
        cum = (1 + strategy_returns).cumprod()
        max_dd = float(((cum - cum.cummax()) / (cum.cummax() + 1e-12)).min())
        dd_score = max(0.0, 1.0 + max_dd / 0.5)  # full score if DD < 0
        score += 25 * dd_score

        # % positive Sharpe windows (25 pts)
        max_score += 25
        roll_mean = strategy_returns.rolling(63).mean()
        roll_std = strategy_returns.rolling(63).std()
        roll_sharpe = (roll_mean * np.sqrt(252)) / (roll_std * np.sqrt(252) + 1e-12)
        pct_positive = float((roll_sharpe.dropna() > 0).mean())
        score += 25 * pct_positive

        return float(min(100.0, 100.0 * score / max_score))
