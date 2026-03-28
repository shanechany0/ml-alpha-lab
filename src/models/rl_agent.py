"""Reinforcement learning agent for position sizing using PPO."""

from __future__ import annotations

import logging
from typing import Any

import mlflow
import numpy as np

from src.models.base_model import BaseModel

# ---------------------------------------------------------------------------
# Optional gym import — degrade gracefully so the package can be imported
# even when gymnasium/gym is not installed.
# ---------------------------------------------------------------------------
try:
    import gymnasium as _gym_module
    from gymnasium import spaces as _spaces
except ImportError:
    try:
        import gym as _gym_module  # type: ignore[no-redef]
        from gym import spaces as _spaces  # type: ignore[assignment]
    except ImportError:
        _gym_module = None  # type: ignore[assignment]
        _spaces = None  # type: ignore[assignment]

_GymEnv: type = object if _gym_module is None else _gym_module.Env

logger = logging.getLogger(__name__)


class TradingEnvironment(_GymEnv):  # type: ignore[valid-type]
    """Gymnasium environment for continuous position-sizing trading.

    At each step the agent selects a position in ``[-1, 1]`` (short to long),
    and receives a reward equal to the realised PnL minus a transaction-cost
    penalty proportional to the change in position.

    Attributes:
        features: Feature matrix of shape (n_steps, n_features).
        returns: Return array of shape (n_steps,).
        config: Environment configuration dictionary.
        current_step: Index of the current time step.
        prev_action: Previous action (position) for cost computation.
        observation_space: Continuous box over feature space.
        action_space: Continuous box ``[-1, 1]`` for position sizing.
    """

    metadata: dict[str, Any] = {"render_modes": []}

    def __init__(
        self,
        features: np.ndarray,
        returns: np.ndarray,
        config: dict,
    ) -> None:
        """Initialize the trading environment.

        Args:
            features: Feature matrix of shape (n_steps, n_features).
            returns: Forward returns of shape (n_steps,). The reward at
                step ``t`` is ``action * returns[t]``.
            config: Configuration dictionary. Recognises:
                ``transaction_cost`` (float, default 0.001).
        """
        super().__init__()
        self.features = np.array(features, dtype=np.float32)
        self.returns = np.array(returns, dtype=np.float32)
        self.config = config
        self.transaction_cost: float = config.get("transaction_cost", 0.001)
        self.current_step: int = 0
        self.prev_action: float = 0.0

        n_features = self.features.shape[1]
        self.observation_space = _spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(n_features,),
            dtype=np.float32,
        )
        self.action_space = _spaces.Box(
            low=np.array([-1.0], dtype=np.float32),
            high=np.array([1.0], dtype=np.float32),
            dtype=np.float32,
        )

    def reset(self, *, seed: int | None = None, options: dict | None = None) -> tuple[np.ndarray, dict]:
        """Reset the environment to the initial state.

        Args:
            seed: Random seed for reproducibility (passed to super).
            options: Unused extra options.

        Returns:
            Tuple of ``(observation, info)`` where observation is the
            first feature row and info is an empty dict.
        """
        if _gym_module is not None:
            super().reset(seed=seed)
        self.current_step = 0
        self.prev_action = 0.0
        return self.features[0], {}

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        """Advance the environment by one step.

        Computes realised PnL less a transaction cost proportional to
        the absolute change in position (turnover penalty).

        Args:
            action: Position array of shape ``(1,)`` clipped to ``[-1, 1]``.

        Returns:
            Tuple of ``(next_obs, reward, terminated, truncated, info)``.
            ``terminated`` is ``True`` when all steps are exhausted.
        """
        position = float(np.clip(action[0], -1.0, 1.0))
        ret = float(self.returns[self.current_step])

        pnl = position * ret
        cost = self.transaction_cost * abs(position - self.prev_action)
        reward = pnl - cost

        self.prev_action = position
        self.current_step += 1
        terminated = self.current_step >= len(self.features) - 1

        next_obs = self.features[min(self.current_step, len(self.features) - 1)]
        info: dict[str, Any] = {"pnl": pnl, "cost": cost, "position": position}
        return next_obs, reward, terminated, False, info

    def render(self) -> None:
        """Render the environment (no-op)."""


class RLAgent(BaseModel):
    """Reinforcement learning agent using PPO for position sizing.

    Wraps ``stable-baselines3`` PPO inside the ``BaseModel`` interface.
    Requires ``stable-baselines3`` and ``gymnasium`` to be installed.

    Attributes:
        config: Configuration dictionary.
        rl_model: Trained ``stable_baselines3.PPO`` instance.
        rl_config: RL-specific hyperparameter sub-dict.
    """

    def __init__(self, config: dict | None = None) -> None:
        """Initialize the RL agent.

        Args:
            config: Configuration dictionary. Recognises an ``rl_agent``
                sub-dict with SB3 PPO hyperparameters: ``policy``,
                ``learning_rate``, ``n_steps``, ``batch_size``,
                ``n_epochs``, ``gamma``, ``gae_lambda``, ``clip_range``,
                ``ent_coef``, ``total_timesteps``.
        """
        super().__init__(config)
        self.rl_config: dict = self.config.get("rl_agent", {})
        self.rl_model: Any = None

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
    ) -> "RLAgent":
        """Create a TradingEnvironment and train a PPO agent.

        Args:
            X_train: Training feature matrix of shape (n_samples, n_features).
            y_train: Forward return targets used as environment rewards.
            X_val: Unused (included for API compatibility).
            y_val: Unused.

        Returns:
            Self, for method chaining.

        Raises:
            ImportError: If ``stable-baselines3`` is not installed.
        """
        try:
            from stable_baselines3 import PPO  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError(
                "stable-baselines3 is required for RLAgent: pip install stable-baselines3"
            ) from exc

        import pandas as pd  # noqa: PLC0415

        if isinstance(X_train, pd.DataFrame):
            self.feature_names = list(X_train.columns)
            X_train = X_train.values

        env = TradingEnvironment(
            features=X_train,
            returns=np.array(y_train, dtype=np.float32),
            config=self.config,
        )

        total_timesteps: int = self.rl_config.get("total_timesteps", 100_000)
        ppo_params = {
            "policy": self.rl_config.get("policy", "MlpPolicy"),
            "learning_rate": self.rl_config.get("learning_rate", 3e-4),
            "n_steps": self.rl_config.get("n_steps", 2048),
            "batch_size": self.rl_config.get("batch_size", 64),
            "n_epochs": self.rl_config.get("n_epochs", 10),
            "gamma": self.rl_config.get("gamma", 0.99),
            "gae_lambda": self.rl_config.get("gae_lambda", 0.95),
            "clip_range": self.rl_config.get("clip_range", 0.2),
            "ent_coef": self.rl_config.get("ent_coef", 0.01),
            "verbose": 0,
        }

        with mlflow.start_run(nested=True):
            mlflow.log_params({**ppo_params, "total_timesteps": total_timesteps})
            self.rl_model = PPO(env=env, **ppo_params)
            self.rl_model.learn(total_timesteps=total_timesteps)
            logger.info("PPO training complete.")

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return position-sizing array for each observation.

        Args:
            X: Feature matrix of shape (n_samples, n_features).

        Returns:
            Position array of shape (n_samples,) with values in ``[-1, 1]``.

        Raises:
            RuntimeError: If the agent has not been trained yet.
        """
        if self.rl_model is None:
            raise RuntimeError("RLAgent has not been trained. Call fit() first.")

        import pandas as pd  # noqa: PLC0415

        if isinstance(X, pd.DataFrame):
            X = X.values

        positions = []
        for obs in X:
            action, _ = self.rl_model.predict(obs.astype(np.float32), deterministic=True)
            positions.append(float(np.clip(action[0], -1.0, 1.0)))
        return np.array(positions, dtype=np.float32)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return position predictions (alias for predict).

        Args:
            X: Feature matrix of shape (n_samples, n_features).

        Returns:
            Same as ``predict(X)``.
        """
        return self.predict(X)

    def save(self, path: str) -> None:
        """Save the SB3 model to disk.

        Args:
            path: File path (without extension; SB3 appends ``.zip``).

        Raises:
            RuntimeError: If the agent has not been trained.
        """
        from pathlib import Path  # noqa: PLC0415

        if self.rl_model is None:
            raise RuntimeError("RLAgent has not been trained. Call fit() first.")

        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        self.rl_model.save(str(save_path))
        logger.info("RLAgent saved to %s", path)

    def load(self, path: str) -> "RLAgent":
        """Load a previously saved SB3 PPO model.

        Args:
            path: File path (with or without ``.zip`` extension).

        Returns:
            Self with the loaded model.

        Raises:
            ImportError: If ``stable-baselines3`` is not installed.
        """
        try:
            from stable_baselines3 import PPO  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError(
                "stable-baselines3 is required: pip install stable-baselines3"
            ) from exc

        self.rl_model = PPO.load(path)
        logger.info("RLAgent loaded from %s", path)
        return self
