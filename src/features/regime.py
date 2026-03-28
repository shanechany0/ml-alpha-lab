"""Hidden Markov Model-based market regime detection for ML Alpha Lab."""

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class RegimeDetector:
    """Detects market regimes using a Gaussian HMM.

    Regimes are labelled 0 (bear), 1 (sideways), 2 (bull) after sorting
    by the mean return of each hidden state.

    Attributes:
        n_regimes: Number of hidden states.
        config: Optional configuration dictionary.
        model: Fitted GaussianHMM instance (None before fitting).
        _label_map: Mapping from raw HMM state → semantic label.
    """

    def __init__(
        self, n_regimes: int = 3, config: dict[str, Any] | None = None
    ) -> None:
        """Initialise the RegimeDetector.

        Args:
            n_regimes: Number of market regimes to model.
            config: Optional configuration overrides (e.g. ``n_iter``,
                ``covariance_type``).
        """
        self.n_regimes = n_regimes
        self.config: dict[str, Any] = config or {}
        self.model: Any = None
        self._label_map: dict[int, int] = {}
        self._fitted = False

    def _build_model(self) -> Any:
        """Construct a GaussianHMM from hmmlearn.

        Returns:
            Unfitted GaussianHMM instance.

        Raises:
            ImportError: If hmmlearn is not installed.
        """
        try:
            from hmmlearn import hmm
        except ImportError as exc:
            raise ImportError(
                "hmmlearn is required: pip install hmmlearn"
            ) from exc

        cov_type = self.config.get("covariance_type", "full")
        n_iter = self.config.get("n_iter", 200)
        random_state = self.config.get("random_state", 42)
        return hmm.GaussianHMM(
            n_components=self.n_regimes,
            covariance_type=cov_type,
            n_iter=n_iter,
            random_state=random_state,
        )

    @staticmethod
    def _prepare_observations(returns: pd.DataFrame) -> np.ndarray:
        """Flatten returns DataFrame into a (T, n_features) observation array.

        Args:
            returns: Return DataFrame.

        Returns:
            2-D numpy array of shape ``(T, n_features)``.
        """
        arr = returns.ffill().fillna(0).values
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        return arr

    def fit(self, returns: pd.DataFrame) -> "RegimeDetector":
        """Fit the HMM on market return observations.

        Args:
            returns: DataFrame of asset returns. A single column or multiple
                columns are both accepted.

        Returns:
            Self (fitted RegimeDetector).
        """
        obs = self._prepare_observations(returns)
        self.model = self._build_model()
        try:
            self.model.fit(obs)
            self._fitted = True
            # Sort states by mean return of the first feature
            means = self.model.means_[:, 0]
            sorted_states = np.argsort(means)
            self._label_map = {
                int(raw): int(label) for label, raw in enumerate(sorted_states)
            }
            logger.info("HMM fitted with %d regimes.", self.n_regimes)
        except Exception as exc:  # noqa: BLE001
            logger.warning("HMM fitting failed: %s. Using neutral regime.", exc)
            self._fitted = False
        return self

    def predict(self, returns: pd.DataFrame) -> pd.Series:
        """Predict regime labels for the given returns.

        Args:
            returns: DataFrame of asset returns.

        Returns:
            Integer Series of regime labels
            (0=bear, 1=sideways, 2=bull).
        """
        if not self._fitted or self.model is None:
            logger.warning("Model not fitted; returning neutral regime (1).")
            return pd.Series(1, index=returns.index, name="regime")

        obs = self._prepare_observations(returns)
        raw_labels = self.model.predict(obs)
        labels = np.array([self._label_map.get(int(r), 1) for r in raw_labels])
        return pd.Series(labels, index=returns.index, name="regime")

    def predict_proba(self, returns: pd.DataFrame) -> pd.DataFrame:
        """Compute posterior regime probabilities.

        Args:
            returns: DataFrame of asset returns.

        Returns:
            DataFrame of shape ``(T, n_regimes)`` with columns
            ``['regime_0_prob', 'regime_1_prob', ...]``.
        """
        if not self._fitted or self.model is None:
            logger.warning("Model not fitted; returning uniform probabilities.")
            uniform = 1.0 / self.n_regimes
            cols = [f"regime_{i}_prob" for i in range(self.n_regimes)]
            return pd.DataFrame(uniform, index=returns.index, columns=cols)

        obs = self._prepare_observations(returns)
        _, posteriors = self.model.score_samples(obs)
        # Reorder columns to semantic labels
        n = self.n_regimes
        reordered = np.zeros_like(posteriors)
        for raw, label in self._label_map.items():
            reordered[:, label] = posteriors[:, raw]
        cols = [f"regime_{i}_prob" for i in range(n)]
        return pd.DataFrame(reordered, index=returns.index, columns=cols)

    def get_regime_features(self, returns: pd.DataFrame) -> pd.DataFrame:
        """Return a combined DataFrame of regime label and probabilities.

        Args:
            returns: DataFrame of asset returns.

        Returns:
            DataFrame with columns ``['regime', 'regime_0_prob', ...]``.
        """
        labels = self.predict(returns)
        probas = self.predict_proba(returns)
        return pd.concat([labels.to_frame(), probas], axis=1)
