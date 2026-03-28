"""Liquidity analysis: Amihud illiquidity, volume profiles, and filtering."""

from __future__ import annotations

import numpy as np
import pandas as pd


class LiquidityAnalyzer:
    """Analyzes asset liquidity across multiple dimensions.

    Supports Amihud illiquidity ratio, VWAP-based volume profiles,
    turnover analysis, and composite liquidity scoring.

    Attributes:
        config: Configuration dictionary.
    """

    def __init__(self, config: dict | None = None) -> None:
        """Initializes LiquidityAnalyzer.

        Args:
            config: Optional configuration. Recognized keys:
                'adv_window' (default 20), 'liquidity_window' (default 21).
        """
        self.config = config or {}

    def amihud_illiquidity(
        self,
        returns: pd.DataFrame,
        volume: pd.DataFrame,
        window: int = 21,
    ) -> pd.DataFrame:
        """Computes rolling Amihud (2002) illiquidity ratio.

        ILLIQ = mean(|r_t| / (P_t * V_t)) where V_t is share volume and
        P_t * V_t is dollar volume. Here we expect dollar volume directly.

        Args:
            returns: Daily return DataFrame (dates × assets).
            volume: Dollar volume DataFrame in millions (dates × assets).
            window: Rolling window in trading days.

        Returns:
            DataFrame of rolling Amihud illiquidity ratios.
        """
        aligned_returns, aligned_volume = returns.align(volume, join="inner")
        # Avoid division by zero; small epsilon for zero-volume days
        safe_volume = aligned_volume.replace(0, np.nan)
        illiq_daily = aligned_returns.abs() / safe_volume
        return illiq_daily.rolling(window).mean().rename(columns=lambda c: c)

    def volume_profile(
        self, volume: pd.DataFrame, prices: pd.DataFrame
    ) -> pd.DataFrame:
        """Computes VWAP and basic volume distribution statistics.

        Args:
            volume: Share volume DataFrame (dates × assets).
            prices: Asset price DataFrame (dates × assets).

        Returns:
            DataFrame with VWAP and total dollar volume per asset (columns
            as MultiIndex or flat, indexed by date).
        """
        aligned_vol, aligned_prices = volume.align(prices, join="inner")
        dollar_volume = aligned_vol * aligned_prices
        cumulative_dv = dollar_volume.cumsum()
        cumulative_v = aligned_vol.cumsum()

        vwap = cumulative_dv / cumulative_v.replace(0, np.nan)
        vwap.columns = [f"{c}_vwap" for c in vwap.columns]
        dv = dollar_volume
        dv.columns = [f"{c}_dollar_volume" for c in dv.columns]

        return pd.concat([vwap, dv], axis=1)

    def turnover_analysis(
        self,
        volume: pd.DataFrame,
        shares_outstanding: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """Computes share turnover (volume / shares outstanding).

        Args:
            volume: Daily share volume DataFrame.
            shares_outstanding: Shares outstanding DataFrame. If None,
                uses normalized volume (fraction of mean daily volume).

        Returns:
            DataFrame of daily turnover ratios.
        """
        if shares_outstanding is not None:
            aligned_vol, aligned_so = volume.align(shares_outstanding, join="inner")
            return (aligned_vol / aligned_so.replace(0, np.nan)).rename(
                columns=lambda c: c
            )
        # Fallback: turnover as fraction of rolling mean volume
        mean_vol = volume.rolling(20).mean().replace(0, np.nan)
        return (volume / mean_vol).rename(columns=lambda c: c)

    def average_daily_volume(
        self, volume: pd.DataFrame, window: int = 20
    ) -> pd.DataFrame:
        """Computes rolling average daily volume.

        Args:
            volume: Daily volume DataFrame (share or dollar).
            window: Rolling window in trading days.

        Returns:
            Rolling ADV DataFrame.
        """
        return volume.rolling(window).mean()

    def liquidity_score(
        self,
        returns: pd.DataFrame,
        volume: pd.DataFrame,
        prices: pd.DataFrame,
    ) -> pd.DataFrame:
        """Computes a composite liquidity score for each asset.

        Score is based on: low Amihud ratio (high liquidity) and
        high dollar volume, normalized to [0, 1] per date.

        Args:
            returns: Daily return DataFrame.
            volume: Daily volume DataFrame.
            prices: Price DataFrame.

        Returns:
            DataFrame of composite liquidity scores (0 = least liquid,
            1 = most liquid).
        """
        dollar_vol = (volume * prices).rolling(20).mean()
        illiq = self.amihud_illiquidity(returns, dollar_vol)

        # Normalize: higher dollar volume → higher score; lower illiq → higher score
        dv_rank = dollar_vol.rank(axis=1, pct=True)
        illiq_rank = (1 - illiq.rank(axis=1, pct=True))
        score = (dv_rank + illiq_rank) / 2
        return score.rename(columns=lambda c: c)

    def filter_by_liquidity(
        self,
        returns: pd.DataFrame,
        volume: pd.DataFrame,
        prices: pd.DataFrame,
        min_adv_millions: float = 5.0,
    ) -> list[str]:
        """Returns tickers meeting a minimum average dollar volume threshold.

        Args:
            returns: Daily return DataFrame.
            volume: Daily volume DataFrame (shares).
            prices: Asset price DataFrame.
            min_adv_millions: Minimum average daily dollar volume in $ millions.

        Returns:
            List of ticker symbols meeting the liquidity threshold.
        """
        aligned_vol, aligned_prices = volume.align(prices, join="inner")
        dollar_vol = aligned_vol * aligned_prices
        mean_adv = dollar_vol.mean()
        threshold = min_adv_millions * 1_000_000
        liquid = mean_adv[mean_adv >= threshold].index.tolist()
        return [t for t in liquid if t in returns.columns]
