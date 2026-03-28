"""Cross-sectional features for ML Alpha Lab."""

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class CrossSectionalFeatures:
    """Computes cross-sectional features across the asset universe.

    Attributes:
        config: Optional configuration dictionary.
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        """Initialise CrossSectionalFeatures.

        Args:
            config: Optional configuration overrides.
        """
        self.config: dict[str, Any] = config or {}

    def compute_all(
        self, returns: pd.DataFrame, prices: pd.DataFrame
    ) -> pd.DataFrame:
        """Compute all cross-sectional features.

        Args:
            returns: Wide DataFrame of asset returns (rows=dates, cols=tickers).
            prices: Wide DataFrame of asset prices (rows=dates, cols=tickers).

        Returns:
            Concatenated DataFrame of all cross-sectional features.
        """
        logger.info("Computing all cross-sectional features.")
        frames: list[pd.DataFrame] = [
            self.cross_sectional_momentum(returns),
            self.cross_sectional_zscore(returns),
            self.percentile_rank(returns),
            self.relative_strength(returns),
        ]
        return pd.concat(frames, axis=1)

    def cross_sectional_momentum(
        self,
        returns: pd.DataFrame,
        windows: list[int] | None = None,
    ) -> pd.DataFrame:
        """Compute cross-sectional momentum as universe-relative rank.

        For each window, compute the cumulative return and rank stocks
        within the cross-section.

        Args:
            returns: Wide return DataFrame.
            windows: Look-back windows. Defaults to ``[20, 60, 120]``.

        Returns:
            DataFrame with columns ``{ticker}_cs_mom_{window}``.
        """
        if windows is None:
            windows = [20, 60, 120]
        parts: dict[str, pd.Series] = {}
        for w in windows:
            cum_ret = returns.rolling(w).sum()
            ranked = cum_ret.rank(axis=1, pct=True)
            for col in ranked.columns:
                parts[f"{col}_cs_mom_{w}"] = ranked[col]
        return pd.DataFrame(parts, index=returns.index)

    def cross_sectional_zscore(self, df: pd.DataFrame) -> pd.DataFrame:
        """Demean and scale values cross-sectionally (z-score).

        For each row, subtract the cross-sectional mean and divide by
        the cross-sectional standard deviation.

        Args:
            df: Wide DataFrame (rows=dates, cols=tickers).

        Returns:
            Cross-sectionally z-scored DataFrame with ``_cs_zscore`` suffix.
        """
        cs_mean = df.mean(axis=1)
        cs_std = df.std(axis=1).replace(0, np.nan)
        z = df.sub(cs_mean, axis=0).div(cs_std, axis=0)
        z.columns = [f"{c}_cs_zscore" for c in df.columns]
        return z

    def percentile_rank(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute cross-sectional percentile rank (0 to 1).

        Args:
            df: Wide DataFrame (rows=dates, cols=tickers).

        Returns:
            DataFrame of percentile ranks with ``_pct_rank`` suffix.
        """
        ranked = df.rank(axis=1, pct=True)
        ranked.columns = [f"{c}_pct_rank" for c in df.columns]
        return ranked

    def sector_relative_momentum(
        self,
        returns: pd.DataFrame,
        sector_map: dict[str, str] | None = None,
    ) -> pd.DataFrame:
        """Compute stock return minus its sector's average return.

        Args:
            returns: Wide return DataFrame.
            sector_map: Mapping from ticker → sector name. If None, all
                tickers are treated as belonging to one sector.

        Returns:
            DataFrame of sector-relative returns with ``_sector_rel`` suffix.
        """
        if sector_map is None:
            # Treat all as same sector
            sector_map = {c: "all" for c in returns.columns}

        sectors = pd.Series(sector_map)
        result = pd.DataFrame(index=returns.index, columns=returns.columns, dtype=float)

        for sector in sectors.unique():
            tickers_in_sector = sectors[sectors == sector].index.tolist()
            valid = [t for t in tickers_in_sector if t in returns.columns]
            if not valid:
                continue
            sector_ret = returns[valid].mean(axis=1)
            for t in valid:
                result[t] = returns[t] - sector_ret

        result.columns = [f"{c}_sector_rel" for c in result.columns]
        return result

    def relative_strength(
        self, returns: pd.DataFrame, window: int = 60
    ) -> pd.DataFrame:
        """Compute relative strength: stock cumulative return / index return.

        The index return is the equal-weight average of all stocks.

        Args:
            returns: Wide return DataFrame.
            window: Look-back window for cumulative return.

        Returns:
            DataFrame of relative strength with ``_rel_strength`` suffix.
        """
        cum_ret = returns.rolling(window).sum()
        index_ret = cum_ret.mean(axis=1)
        rs = cum_ret.sub(index_ret, axis=0)
        rs.columns = [f"{c}_rel_strength" for c in returns.columns]
        return rs
