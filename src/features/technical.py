"""Technical indicator features for ML Alpha Lab."""

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class TechnicalFeatures:
    """Computes technical analysis features using only pandas/numpy.

    Attributes:
        config: Optional configuration dictionary.
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        """Initialise TechnicalFeatures.

        Args:
            config: Optional dictionary of parameter overrides. Supported
                keys mirror method parameter names (e.g. ``'rsi_period'``).
        """
        self.config: dict[str, Any] = config or {}

    def compute_all(self, prices: pd.DataFrame) -> pd.DataFrame:
        """Compute all technical features for each ticker column.

        Expects ``prices`` to have columns ``['open', 'high', 'low', 'close',
        'volume']`` or a MultiIndex with those field names at level 0.

        Args:
            prices: DataFrame of OHLCV prices. Simple column DataFrame where
                each column is a price series is also accepted (treated as
                close prices).

        Returns:
            Wide DataFrame of technical features for all tickers, with
            column names in the form ``{ticker}_{feature}``.
        """
        frames: list[pd.DataFrame] = []

        # Handle MultiIndex columns (field, ticker)
        if isinstance(prices.columns, pd.MultiIndex):
            close = prices["close"] if "close" in prices.columns.get_level_values(0) else prices.iloc[:, 0]
            high = prices.get("high", close)
            low = prices.get("low", close)
            volume = prices.get("volume", pd.DataFrame(np.nan, index=prices.index, columns=close.columns))
            tickers = close.columns.tolist()
            for ticker in tickers:
                feat = self._compute_for_ticker(
                    close[ticker], high[ticker], low[ticker], volume[ticker], ticker
                )
                frames.append(feat)
        else:
            # Treat each column as a close price series
            for col in prices.columns:
                feat = self._compute_for_ticker(
                    prices[col],
                    prices[col],
                    prices[col],
                    pd.Series(np.nan, index=prices.index),
                    str(col),
                )
                frames.append(feat)

        if not frames:
            return pd.DataFrame(index=prices.index)
        return pd.concat(frames, axis=1)

    def _compute_for_ticker(
        self,
        close: pd.Series,
        high: pd.Series,
        low: pd.Series,
        volume: pd.Series,
        ticker: str,
    ) -> pd.DataFrame:
        """Compute all features for a single ticker.

        Args:
            close: Close price series.
            high: High price series.
            low: Low price series.
            volume: Volume series.
            ticker: Ticker label for column naming.

        Returns:
            DataFrame of features for the given ticker.
        """
        parts: dict[str, pd.Series] = {}

        macd_df = self.macd(close)
        for c in macd_df.columns:
            parts[f"{ticker}_macd_{c}"] = macd_df[c]

        parts[f"{ticker}_rsi"] = self.rsi(close)

        bb_df = self.bollinger_bands(close)
        for c in bb_df.columns:
            parts[f"{ticker}_bb_{c}"] = bb_df[c]

        parts[f"{ticker}_atr"] = self.atr(high, low, close)
        parts[f"{ticker}_obv"] = self.obv(close, volume)
        parts[f"{ticker}_ema_12"] = self.ema(close, span=12)
        parts[f"{ticker}_ema_26"] = self.ema(close, span=26)
        parts[f"{ticker}_momentum_20"] = self.momentum(close, period=20)
        parts[f"{ticker}_volume_ratio"] = self.volume_ratio(volume, period=20)

        return pd.DataFrame(parts, index=close.index)

    def macd(
        self,
        prices: pd.Series,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9,
    ) -> pd.DataFrame:
        """Compute MACD, signal line, and histogram.

        Args:
            prices: Close price series.
            fast: Fast EMA period.
            slow: Slow EMA period.
            signal: Signal EMA period.

        Returns:
            DataFrame with columns ``['macd', 'signal', 'histogram']``.
        """
        ema_fast = prices.ewm(span=fast, adjust=False).mean()
        ema_slow = prices.ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line
        return pd.DataFrame(
            {"macd": macd_line, "signal": signal_line, "histogram": histogram},
            index=prices.index,
        )

    def rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Compute the Relative Strength Index.

        Args:
            prices: Close price series.
            period: Look-back period.

        Returns:
            RSI series in the range [0, 100].
        """
        delta = prices.diff()
        gain = delta.clip(lower=0)
        loss = (-delta).clip(lower=0)
        avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
        rs = avg_gain / (avg_loss + 1e-9)
        rsi_series = 100 - (100 / (1 + rs))
        rsi_series.name = "rsi"
        return rsi_series

    def bollinger_bands(
        self, prices: pd.Series, period: int = 20, n_std: float = 2.0
    ) -> pd.DataFrame:
        """Compute Bollinger Bands.

        Args:
            prices: Close price series.
            period: Rolling window period.
            n_std: Number of standard deviations for band width.

        Returns:
            DataFrame with columns ``['upper', 'lower', 'pct_b', 'bandwidth']``.
        """
        rolling_mean = prices.rolling(period).mean()
        rolling_std = prices.rolling(period).std()
        upper = rolling_mean + n_std * rolling_std
        lower = rolling_mean - n_std * rolling_std
        pct_b = (prices - lower) / (upper - lower + 1e-9)
        bandwidth = (upper - lower) / (rolling_mean + 1e-9)
        return pd.DataFrame(
            {"upper": upper, "lower": lower, "pct_b": pct_b, "bandwidth": bandwidth},
            index=prices.index,
        )

    def atr(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        period: int = 14,
    ) -> pd.Series:
        """Compute Average True Range.

        Args:
            high: High price series.
            low: Low price series.
            close: Close price series.
            period: Smoothing period.

        Returns:
            ATR series.
        """
        prev_close = close.shift(1)
        tr = pd.concat(
            [
                high - low,
                (high - prev_close).abs(),
                (low - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        atr_series = tr.ewm(alpha=1 / period, adjust=False).mean()
        atr_series.name = "atr"
        return atr_series

    def obv(self, close: pd.Series, volume: pd.Series) -> pd.Series:
        """Compute On Balance Volume.

        Args:
            close: Close price series.
            volume: Volume series.

        Returns:
            OBV series.
        """
        direction = np.sign(close.diff()).fillna(0)
        obv_series = (direction * volume).cumsum()
        obv_series.name = "obv"
        return obv_series

    def ema(self, prices: pd.Series, span: int) -> pd.Series:
        """Compute Exponential Moving Average.

        Args:
            prices: Price series.
            span: EMA span (decay = 2 / (span + 1)).

        Returns:
            EMA series.
        """
        ema_series = prices.ewm(span=span, adjust=False).mean()
        ema_series.name = f"ema_{span}"
        return ema_series

    def momentum(self, prices: pd.Series, period: int = 20) -> pd.Series:
        """Compute price momentum as the rate of change over a period.

        Args:
            prices: Price series.
            period: Look-back period.

        Returns:
            Momentum series (price / price[t-period] - 1).
        """
        mom = prices / prices.shift(period) - 1
        mom.name = f"momentum_{period}"
        return mom

    def volume_ratio(self, volume: pd.Series, period: int = 20) -> pd.Series:
        """Compute ratio of current volume to rolling average volume.

        Args:
            volume: Volume series.
            period: Rolling average window.

        Returns:
            Volume ratio series (volume / rolling_mean_volume).
        """
        avg_vol = volume.rolling(period).mean()
        ratio = volume / (avg_vol + 1e-9)
        ratio.name = f"volume_ratio_{period}"
        return ratio
