"""Bid-ask spread estimation models."""

from __future__ import annotations

import numpy as np
import pandas as pd


class SpreadModel:
    """Estimates bid-ask spreads from price data using various microstructure models.

    Supports Roll (1984) covariance model, effective spread, and
    quoted spread calculations.

    Attributes:
        config: Configuration dictionary.
        window: Rolling window size for spread estimation.
    """

    def __init__(self, config: dict | None = None) -> None:
        """Initializes SpreadModel.

        Args:
            config: Optional configuration. Recognized keys:
                'window' (rolling estimation window, default 21).
        """
        self.config = config or {}
        self.window: int = self.config.get("window", 21)

    def roll_model(self, prices: pd.Series) -> float:
        """Estimates bid-ask spread using the Roll (1984) model.

        Computes: 2 * sqrt(-cov(ΔP_t, ΔP_{t-1})) when the covariance is negative.

        Args:
            prices: Asset price time series.

        Returns:
            Roll spread estimate as a fraction of price (0 if covariance
            is non-negative, indicating the model assumption is violated).
        """
        delta = prices.diff().dropna()
        if len(delta) < 2:
            return 0.0
        cov = delta.cov(delta.shift(1).dropna().reindex(delta.index))
        if cov >= 0:
            return 0.0
        return float(2 * np.sqrt(-cov))

    def effective_spread(
        self, trade_prices: pd.Series, mid_prices: pd.Series
    ) -> float:
        """Estimates the effective spread from trade prices relative to mid.

        Effective spread = 2 * |trade_price - mid_price| / mid_price

        Args:
            trade_prices: Prices at which trades occurred.
            mid_prices: Contemporaneous mid-quote prices.

        Returns:
            Average effective spread as a fraction of mid-price.
        """
        aligned_trades, aligned_mid = trade_prices.align(mid_prices, join="inner")
        if aligned_mid.empty or (aligned_mid == 0).all():
            return 0.0
        spreads = 2 * (aligned_trades - aligned_mid).abs() / aligned_mid.replace(0, np.nan)
        return float(spreads.mean())

    def quoted_spread(self, bid: pd.Series, ask: pd.Series) -> pd.Series:
        """Computes the quoted (proportional) spread from bid and ask series.

        Quoted spread = (ask - bid) / mid, where mid = (ask + bid) / 2

        Args:
            bid: Best bid price series.
            ask: Best ask price series.

        Returns:
            Proportional quoted spread series.
        """
        mid = (bid + ask) / 2
        spread = (ask - bid) / mid.replace(0, np.nan)
        return spread.rename("quoted_spread")

    def estimate_spread(
        self, prices: pd.Series, method: str = "roll"
    ) -> pd.Series:
        """Computes a rolling spread estimate using the selected method.

        Args:
            prices: Asset price time series.
            method: Estimation method. Options: 'roll'. More methods can
                be added; unrecognized methods fall back to 'roll'.

        Returns:
            Rolling spread estimate series.
        """
        def _roll_window(window_prices: pd.Series) -> float:
            return self.roll_model(window_prices)

        return (
            prices.rolling(self.window)
            .apply(_roll_window, raw=False)
            .rename("spread_estimate")
        )

    def spread_to_cost_bps(
        self, spread: float | pd.Series
    ) -> float | pd.Series:
        """Converts fractional spread to basis points.

        Args:
            spread: Spread as a fraction of price (e.g. 0.001 for 10 bps).

        Returns:
            Spread in basis points.
        """
        return spread * 10_000
