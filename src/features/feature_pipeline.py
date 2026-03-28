"""Feature pipeline orchestration for ML Alpha Lab."""

import logging
import os
from typing import Any

import numpy as np
import pandas as pd
import yaml

from src.features.cross_sectional import CrossSectionalFeatures
from src.features.statistical import StatisticalFeatures
from src.features.technical import TechnicalFeatures

logger = logging.getLogger(__name__)


class FeaturePipeline:
    """Orchestrates computation of all features for the ML pipeline.

    Handles technical, statistical, and cross-sectional features;
    removes look-ahead bias; normalises; and manages missing values.

    Attributes:
        config: Configuration dictionary from model_config.yaml.
        technical: TechnicalFeatures instance.
        statistical: StatisticalFeatures instance.
        cross_sectional: CrossSectionalFeatures instance.
        _feature_names: Cached list of feature column names after fitting.
    """

    def __init__(self, config_path: str | None = None) -> None:
        """Initialise the FeaturePipeline.

        Args:
            config_path: Path to model_config.yaml. Defaults to
                ``configs/model_config.yaml`` relative to the repo root.
        """
        self.config: dict[str, Any] = {}
        if config_path is None:
            default = os.path.join(
                os.path.dirname(__file__), "..", "..", "configs", "model_config.yaml"
            )
            config_path = default
        if os.path.exists(config_path):
            with open(config_path, "r", encoding="utf-8") as fh:
                self.config = yaml.safe_load(fh) or {}
            logger.info("Loaded model config from %s", config_path)
        else:
            logger.warning("Model config not found at %s; using defaults.", config_path)

        feat_cfg = self.config.get("features", {})
        self.technical = TechnicalFeatures(feat_cfg.get("technical"))
        self.statistical = StatisticalFeatures(feat_cfg.get("statistical"))
        self.cross_sectional = CrossSectionalFeatures(feat_cfg.get("cross_sectional"))
        self._feature_names: list[str] = []

    def build_features(
        self,
        prices: pd.DataFrame,
        returns: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """Orchestrate computation of all feature groups.

        Args:
            prices: Wide DataFrame of close prices (cols=tickers).
            returns: Optional pre-computed return DataFrame. If None,
                log returns are computed from ``prices``.

        Returns:
            Wide feature DataFrame (not yet normalised or shifted).
        """
        if returns is None:
            returns = np.log(prices / prices.shift(1))

        logger.info("Building technical features …")
        tech = self.technical.compute_all(prices)

        logger.info("Building statistical features …")
        stat = self.statistical.compute_all(returns)

        logger.info("Building cross-sectional features …")
        cs = self.cross_sectional.compute_all(returns, prices)

        all_features = pd.concat([tech, stat, cs], axis=1)
        self._feature_names = all_features.columns.tolist()
        logger.info("Total features built: %d", len(self._feature_names))
        return all_features

    def remove_lookahead_bias(self, features: pd.DataFrame) -> pd.DataFrame:
        """Shift all features forward by one period to prevent look-ahead bias.

        This is CRITICAL: features computed on day t must only be available
        to the model when predicting day t+1 forward returns.

        Args:
            features: Feature DataFrame aligned to dates.

        Returns:
            Features shifted by one period (day t → available on day t+1).
        """
        logger.info("Removing look-ahead bias (shifting features by 1 period).")
        return features.shift(1)

    def handle_missing(
        self, features: pd.DataFrame, method: str = "ffill"
    ) -> pd.DataFrame:
        """Handle missing values in the feature DataFrame.

        Args:
            features: Feature DataFrame.
            method: Imputation method: ``'ffill'``, ``'bfill'``, or
                ``'zero'`` (fill with 0).

        Returns:
            Feature DataFrame with NaN values handled.

        Raises:
            ValueError: If an unknown method is specified.
        """
        if method == "ffill":
            features = features.ffill().bfill()
        elif method == "bfill":
            features = features.bfill().ffill()
        elif method == "zero":
            features = features.fillna(0.0)
        else:
            raise ValueError(
                f"Unknown missing-value method '{method}'. "
                "Choose from 'ffill', 'bfill', 'zero'."
            )
        return features

    def normalize_features(
        self, features: pd.DataFrame, method: str = "zscore"
    ) -> pd.DataFrame:
        """Normalise features column-wise.

        Args:
            features: Feature DataFrame.
            method: Normalisation method. One of ``'zscore'`` (standardise
                to zero mean / unit variance) or ``'minmax'`` (scale to
                [0, 1]).

        Returns:
            Normalised feature DataFrame.

        Raises:
            ValueError: If an unknown method is specified.
        """
        if method == "zscore":
            mean = features.mean()
            std = features.std().replace(0, 1)
            return (features - mean) / std
        elif method == "minmax":
            min_ = features.min()
            max_ = features.max()
            denom = (max_ - min_).replace(0, 1)
            return (features - min_) / denom
        else:
            raise ValueError(
                f"Unknown normalisation method '{method}'. "
                "Choose from 'zscore' or 'minmax'."
            )

    def get_feature_names(self) -> list[str]:
        """Return the list of feature column names from the last build.

        Returns:
            List of feature names. Empty list if ``build_features`` has not
            been called yet.
        """
        return self._feature_names

    def fit_transform(self, prices: pd.DataFrame) -> pd.DataFrame:
        """Compute, shift, clean, and normalise features in one call.

        Args:
            prices: Wide DataFrame of close prices.

        Returns:
            Fully processed feature DataFrame ready for model training.
        """
        logger.info("Running fit_transform pipeline.")
        features = self.build_features(prices)
        features = self.remove_lookahead_bias(features)
        features = self.handle_missing(features)
        features = self.normalize_features(features)
        return features

    def transform(self, prices: pd.DataFrame) -> pd.DataFrame:
        """Transform new prices using the same steps as fit_transform.

        Identical to ``fit_transform`` for stateless normalisation. For
        production use, normalisation parameters should be stored from the
        training set and applied here.

        Args:
            prices: Wide DataFrame of close prices.

        Returns:
            Fully processed feature DataFrame.
        """
        return self.fit_transform(prices)
