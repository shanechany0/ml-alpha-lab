"""Live trading readiness checklist and scoring."""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

CHECKLIST_ITEMS: list[str] = [
    "no_lookahead_bias",
    "oos_performance_acceptable",
    "robustness_checks_passed",
    "risk_controls_defined",
    "model_stability_validated",
    "realistic_cost_model",
    "position_limits_enforced",
    "drawdown_circuit_breaker",
    "monitoring_in_place",
    "strategy_documented",
]


class LiveReadinessChecker:
    """Evaluates whether a trading strategy is ready for live deployment.

    Runs a structured set of checks covering look-ahead bias, OOS performance,
    robustness, risk controls, model stability, and cost modeling.
    """

    def __init__(self) -> None:
        """Initialize LiveReadinessChecker."""

    def check_all(
        self,
        strategy: dict,
        model: Any,
        backtest_results: dict,
    ) -> dict[str, Any]:
        """Run all readiness checks.

        Args:
            strategy: Strategy configuration dict.
            model: Trained model object (inspected for stability flags).
            backtest_results: Backtest output dict from the backtesting engine.

        Returns:
            Dict mapping each check name (from CHECKLIST_ITEMS) to its bool
            result, plus "overall_ready" and "readiness_score" keys.
        """
        results: dict[str, Any] = {
            "no_lookahead_bias": self.check_no_lookahead(strategy),
            "oos_performance_acceptable": self.check_oos_performance(backtest_results),
            "robustness_checks_passed": self.check_robustness(backtest_results),
            "risk_controls_defined": self.check_risk_controls(strategy),
            "model_stability_validated": self.check_model_stability(backtest_results),
            "realistic_cost_model": self.check_cost_model(strategy),
            "position_limits_enforced": bool(
                strategy.get("position_limit") or strategy.get("max_weight")
            ),
            "drawdown_circuit_breaker": bool(
                strategy.get("max_drawdown_limit")
            ),
            "monitoring_in_place": bool(strategy.get("monitoring")),
            "strategy_documented": bool(strategy.get("description")),
        }
        score = self.readiness_score(results)
        results["readiness_score"] = score
        results["overall_ready"] = all(
            v for k, v in results.items()
            if k in CHECKLIST_ITEMS
        )
        return results

    def check_no_lookahead(self, strategy: dict) -> bool:
        """Verify the strategy has been reviewed for look-ahead bias.

        Args:
            strategy: Strategy configuration dict.

        Returns:
            True if the strategy explicitly declares no look-ahead bias.
        """
        return bool(strategy.get("no_lookahead", False))

    def check_oos_performance(
        self,
        backtest_results: dict,
        min_sharpe: float = 0.5,
    ) -> bool:
        """Check that OOS Sharpe ratio meets the minimum threshold.

        Args:
            backtest_results: Dict containing "oos_sharpe" key.
            min_sharpe: Minimum acceptable annualized Sharpe. Defaults to 0.5.

        Returns:
            True if OOS Sharpe >= min_sharpe.
        """
        oos_sharpe = backtest_results.get("oos_sharpe", 0.0)
        return float(oos_sharpe) >= min_sharpe

    def check_robustness(self, backtest_results: dict) -> bool:
        """Check whether robustness tests have been run and passed.

        Args:
            backtest_results: Dict that may contain "robustness_passed" key.

        Returns:
            True if robustness checks were performed and passed.
        """
        return bool(backtest_results.get("robustness_passed", False))

    def check_risk_controls(self, strategy: dict) -> bool:
        """Verify that risk control parameters are configured.

        Args:
            strategy: Strategy configuration dict.

        Returns:
            True if at minimum position_limit and max_drawdown_limit are set.
        """
        has_position_limit = (
            "position_limit" in strategy or "max_weight" in strategy
        )
        has_dd_limit = "max_drawdown_limit" in strategy
        return has_position_limit and has_dd_limit

    def check_model_stability(self, backtest_results: dict) -> bool:
        """Check whether model stability metrics are within acceptable bounds.

        Args:
            backtest_results: Dict that may contain "model_stability_score"
                (float in [0, 1]) or "model_stable" (bool).

        Returns:
            True if stability is confirmed.
        """
        if "model_stable" in backtest_results:
            return bool(backtest_results["model_stable"])
        stability = float(backtest_results.get("model_stability_score", 0.0))
        return stability >= 0.6

    def check_cost_model(self, strategy: dict) -> bool:
        """Check that a realistic cost model is configured.

        Args:
            strategy: Strategy configuration dict.

        Returns:
            True if transaction cost or market impact settings are present.
        """
        return bool(
            strategy.get("transaction_costs")
            or strategy.get("market_impact")
            or strategy.get("cost_model")
        )

    def generate_readiness_report(self, check_results: dict) -> str:
        """Produce a human-readable readiness report.

        Args:
            check_results: Output of check_all().

        Returns:
            Formatted multi-line string report.
        """
        lines = [
            "=" * 60,
            "LIVE TRADING READINESS REPORT",
            "=" * 60,
        ]
        for item in CHECKLIST_ITEMS:
            status = "✓ PASS" if check_results.get(item) else "✗ FAIL"
            lines.append(f"  {item:<40} {status}")

        score = check_results.get("readiness_score", 0.0)
        ready = check_results.get("overall_ready", False)
        lines += [
            "",
            f"Readiness Score    : {score:.1f} / 100",
            f"Overall Ready      : {'YES' if ready else 'NO'}",
            "=" * 60,
        ]
        return "\n".join(lines)

    def readiness_score(self, check_results: dict) -> float:
        """Compute a 0–100 readiness score based on check results.

        Args:
            check_results: Dict mapping check names to bool outcomes.

        Returns:
            Percentage of checklist items that passed.
        """
        total = len(CHECKLIST_ITEMS)
        passed = sum(
            1 for item in CHECKLIST_ITEMS if check_results.get(item)
        )
        return float(100.0 * passed / total) if total > 0 else 0.0
