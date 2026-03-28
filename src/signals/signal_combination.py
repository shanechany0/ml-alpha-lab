"""Signal combination methods for ML Alpha Lab."""

import logging

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

logger = logging.getLogger(__name__)


class SignalCombiner:
    """Combines multiple alpha signals into a single composite signal.

    Supports equal-weighting, IC-weighting, ML meta-learning, and
    rank-based combination.

    Attributes:
        method: Default combination method name.
    """

    def __init__(self, method: str = "equal_weight") -> None:
        """Initialise the SignalCombiner.

        Args:
            method: Default combination method. One of ``'equal_weight'``,
                ``'ic_weighted'``, ``'ml_combined'``, or ``'rank_based'``.
        """
        self.method = method

    def combine(
        self,
        signals: pd.DataFrame,
        weights: dict[str, float] | None = None,
    ) -> pd.Series:
        """Combine signals using the instance's default method.

        Args:
            signals: Wide signal DataFrame (rows=dates, cols=signal names).
            weights: Optional signal weights dict for ``'equal_weight'``
                with explicit weights or ``'ic_weighted'`` (passed as
                ``ic_scores``).

        Returns:
            Combined signal Series.
        """
        if self.method == "equal_weight":
            return self.equal_weight(signals)
        elif self.method == "ic_weighted":
            ic_scores = weights or {col: 1.0 for col in signals.columns}
            return self.ic_weighted(signals, ic_scores)
        elif self.method == "ml_combined":
            logger.warning(
                "ml_combined requires a target; falling back to equal_weight."
            )
            return self.equal_weight(signals)
        elif self.method == "rank_based":
            return self.rank_based(signals)
        else:
            raise ValueError(
                f"Unknown combination method '{self.method}'. "
                "Choose from 'equal_weight', 'ic_weighted', "
                "'ml_combined', 'rank_based'."
            )

    def equal_weight(self, signals: pd.DataFrame) -> pd.Series:
        """Combine signals with equal weights.

        NaN values are excluded from the average on a per-row basis.

        Args:
            signals: Wide signal DataFrame.

        Returns:
            Equal-weighted composite signal Series.
        """
        composite = signals.mean(axis=1)
        composite.name = "composite_signal"
        return composite

    def ic_weighted(
        self,
        signals: pd.DataFrame,
        ic_scores: dict[str, float],
    ) -> pd.Series:
        """Combine signals weighted by their Information Coefficients.

        Signals with higher IC receive larger weights. Weights are
        normalised so they sum to 1.

        Args:
            signals: Wide signal DataFrame.
            ic_scores: Mapping of signal column name → IC score.

        Returns:
            IC-weighted composite signal Series.
        """
        weights = pd.Series(ic_scores).reindex(signals.columns).fillna(0)
        total = weights.abs().sum()
        if total == 0:
            return self.equal_weight(signals)
        weights = weights / total
        composite = signals.multiply(weights, axis=1).sum(axis=1)
        composite.name = "composite_signal"
        return composite

    def ml_combined(
        self, signals: pd.DataFrame, target: pd.Series
    ) -> pd.Series:
        """Combine signals using a Ridge regression meta-learner.

        Fits a Ridge regression with ``signals`` as features and ``target``
        (forward returns) as the label, then predicts the composite signal.

        Args:
            signals: Wide signal DataFrame (must overlap with target index).
            target: Forward return target Series.

        Returns:
            Composite signal Series (Ridge predictions).
        """
        common_idx = signals.index.intersection(target.index)
        X = signals.loc[common_idx].ffill().fillna(0)
        y = target.loc[common_idx].fillna(0)
        model = Ridge(alpha=1.0)
        model.fit(X.values, y.values)
        predictions = model.predict(signals.ffill().fillna(0).values)
        composite = pd.Series(predictions, index=signals.index, name="composite_signal")
        logger.info(
            "ML combined: fitted Ridge with %d features, %d samples.",
            X.shape[1],
            X.shape[0],
        )
        return composite

    def rank_based(self, signals: pd.DataFrame) -> pd.Series:
        """Combine signals by averaging cross-sectional percentile ranks.

        Each signal is converted to a [0, 1] percentile rank before
        averaging to reduce the influence of outliers.

        Args:
            signals: Wide signal DataFrame.

        Returns:
            Rank-averaged composite signal Series.
        """
        ranked = signals.rank(pct=True)
        composite = ranked.mean(axis=1)
        composite.name = "composite_signal"
        return composite
