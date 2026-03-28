"""Strategy lifecycle state machine with transition history."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class StrategyState(str, Enum):
    """Lifecycle states for a trading strategy."""

    RESEARCH = "RESEARCH"
    VALIDATION = "VALIDATION"
    PAPER_TRADING = "PAPER_TRADING"
    LIVE = "LIVE"
    DEPRECATED = "DEPRECATED"


@dataclass
class StrategyTransition:
    """Record of a single state transition.

    Attributes:
        from_state: Origin state.
        to_state: Destination state.
        timestamp: ISO-format timestamp string.
        reason: Human-readable reason for the transition.
        approved_by: Identifier of who approved the transition.
    """

    from_state: str
    to_state: str
    timestamp: str
    reason: str
    approved_by: str


_VALID_TRANSITIONS: dict[StrategyState, list[StrategyState]] = {
    StrategyState.RESEARCH: [StrategyState.VALIDATION, StrategyState.DEPRECATED],
    StrategyState.VALIDATION: [
        StrategyState.PAPER_TRADING,
        StrategyState.RESEARCH,
        StrategyState.DEPRECATED,
    ],
    StrategyState.PAPER_TRADING: [
        StrategyState.LIVE,
        StrategyState.VALIDATION,
        StrategyState.DEPRECATED,
    ],
    StrategyState.LIVE: [StrategyState.DEPRECATED, StrategyState.PAPER_TRADING],
    StrategyState.DEPRECATED: [],
}

_TRANSITION_REQUIREMENTS: dict[StrategyState, list[str]] = {
    StrategyState.VALIDATION: ["oos_sharpe", "backtest_period"],
    StrategyState.PAPER_TRADING: ["oos_sharpe", "robustness_passed", "risk_controls"],
    StrategyState.LIVE: [
        "oos_sharpe",
        "robustness_passed",
        "risk_controls",
        "paper_trading_days",
    ],
    StrategyState.DEPRECATED: [],
    StrategyState.RESEARCH: [],
}


class StrategyLifecycle:
    """Manages the lifecycle of a trading strategy through defined states.

    Args:
        strategy_name: Unique name for the strategy.
        initial_state: Starting state. Defaults to StrategyState.RESEARCH.
    """

    def __init__(
        self,
        strategy_name: str,
        initial_state: StrategyState = StrategyState.RESEARCH,
    ) -> None:
        """Initialize StrategyLifecycle.

        Args:
            strategy_name: Unique identifier for the strategy.
            initial_state: Initial lifecycle state. Defaults to RESEARCH.
        """
        self.strategy_name = strategy_name
        self._state = initial_state
        self._history: list[StrategyTransition] = []

    def transition(
        self,
        new_state: StrategyState,
        reason: str,
        approved_by: str = "system",
    ) -> bool:
        """Attempt a state transition.

        Args:
            new_state: Target state.
            reason: Reason for the transition.
            approved_by: Approver identifier. Defaults to "system".

        Returns:
            True if the transition succeeded, False otherwise.
        """
        if not self.can_transition(self._state, new_state):
            logger.warning(
                "Invalid transition %s -> %s for strategy %s",
                self._state,
                new_state,
                self.strategy_name,
            )
            return False

        self._history.append(
            StrategyTransition(
                from_state=self._state.value,
                to_state=new_state.value,
                timestamp=datetime.utcnow().isoformat(),
                reason=reason,
                approved_by=approved_by,
            )
        )
        self._state = new_state
        logger.info(
            "Strategy %s transitioned to %s (reason: %s)",
            self.strategy_name,
            new_state.value,
            reason,
        )
        return True

    def can_transition(
        self,
        from_state: StrategyState,
        to_state: StrategyState,
    ) -> bool:
        """Check whether a transition is permitted.

        Args:
            from_state: Current state.
            to_state: Desired next state.

        Returns:
            True if the transition is in the allowed transitions map.
        """
        return to_state in _VALID_TRANSITIONS.get(from_state, [])

    def get_history(self) -> list[StrategyTransition]:
        """Return the full transition history.

        Returns:
            List of StrategyTransition records in chronological order.
        """
        return list(self._history)

    def current_state(self) -> StrategyState:
        """Return the current lifecycle state.

        Returns:
            Current StrategyState.
        """
        return self._state

    def validate_transition_requirements(
        self,
        new_state: StrategyState,
        backtest_results: dict | None = None,
    ) -> tuple[bool, list[str]]:
        """Check that all required fields are present for the target state.

        Args:
            new_state: Desired next state.
            backtest_results: Optional dict of backtest metrics/flags.

        Returns:
            Tuple of (all_satisfied: bool, missing_fields: list[str]).
        """
        required = _TRANSITION_REQUIREMENTS.get(new_state, [])
        if not required:
            return True, []

        results = backtest_results or {}
        missing = [field for field in required if field not in results]
        return len(missing) == 0, missing

    def save_state(self, path: str) -> None:
        """Persist lifecycle state and history to a JSON file.

        Args:
            path: Absolute file path to write to.
        """
        data: dict[str, Any] = {
            "strategy_name": self.strategy_name,
            "current_state": self._state.value,
            "history": [asdict(t) for t in self._history],
        }
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2)

    def load_state(self, path: str) -> "StrategyLifecycle":
        """Load lifecycle state and history from a JSON file.

        Args:
            path: Absolute file path to read from.

        Returns:
            A new StrategyLifecycle instance populated from the file.
        """
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)

        instance = StrategyLifecycle(
            strategy_name=data["strategy_name"],
            initial_state=StrategyState(data["current_state"]),
        )
        instance._history = [StrategyTransition(**t) for t in data.get("history", [])]
        return instance
