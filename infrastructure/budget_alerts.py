"""Azure cost management and budget alert utilities for ML Alpha Lab."""
from __future__ import annotations

import logging
import smtplib
from datetime import datetime, timezone
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)


class BudgetAlertManager:
    """Manages Azure cost budgets and spend alerts for ML Alpha Lab.

    Args:
        config_path: Path to ``azure_config.yaml``.
    """

    def __init__(self, config_path: str | Path) -> None:
        """Load configuration and initialise Azure cost management clients.

        Args:
            config_path: Path to the Azure configuration YAML file.
        """
        with open(config_path) as fh:
            self.config = yaml.safe_load(fh)

        azure_cfg = self.config["azure"]
        self.subscription_id: str = azure_cfg["subscription_id"]
        self.resource_group: str = azure_cfg["resource_group"]
        self.budget_config: dict[str, Any] = self.config.get("budget", {})

        # Lazy import so the rest of the codebase doesn't require azure-mgmt-*
        try:
            from azure.identity import DefaultAzureCredential
            from azure.mgmt.consumption import ConsumptionManagementClient
            from azure.mgmt.costmanagement import CostManagementClient

            credential = DefaultAzureCredential()
            self._cost_client = CostManagementClient(credential, self.subscription_id)
            self._consumption_client = ConsumptionManagementClient(credential, self.subscription_id)
        except ImportError as exc:
            logger.warning(
                "Azure management packages not installed; budget operations will be unavailable. %s",
                exc,
            )
            self._cost_client = None
            self._consumption_client = None

    # ── Budget management ─────────────────────────────────────────────────────

    def create_budget(
        self,
        name: str,
        amount: float,
        time_grain: str = "Monthly",
    ) -> dict[str, Any]:
        """Create or update an Azure subscription budget.

        Args:
            name: Unique name for the budget.
            amount: Spending limit in USD.
            time_grain: Billing period granularity (``"Monthly"``, ``"Quarterly"``, ``"Annually"``).

        Returns:
            The created budget resource as a dictionary.

        Raises:
            RuntimeError: If the Azure cost management client is unavailable.
        """
        if self._cost_client is None:
            raise RuntimeError("Azure cost management client is not initialised.")

        from azure.mgmt.costmanagement.models import Budget, BudgetTimePeriod

        now = datetime.now(tz=timezone.utc)
        start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)

        budget_body = Budget(
            category="Cost",
            amount=amount,
            time_grain=time_grain,
            time_period=BudgetTimePeriod(start_date=start),
        )

        scope = f"/subscriptions/{self.subscription_id}"
        result = self._cost_client.budgets.create_or_update(scope, name, budget_body)
        logger.info("Budget '%s' created/updated: $%.2f %s", name, amount, time_grain)
        return result.as_dict()

    def set_alert_thresholds(
        self,
        budget_name: str,
        thresholds: list[float],
    ) -> dict[str, Any]:
        """Configure percentage-based alert notifications on an existing budget.

        Args:
            budget_name: Name of the budget to configure.
            thresholds: List of percentage thresholds (e.g. ``[50, 80, 100]``).

        Returns:
            The updated budget resource as a dictionary.

        Raises:
            RuntimeError: If the Azure cost management client is unavailable.
        """
        if self._cost_client is None:
            raise RuntimeError("Azure cost management client is not initialised.")

        from azure.mgmt.costmanagement.models import Notification

        scope = f"/subscriptions/{self.subscription_id}"
        budget = self._cost_client.budgets.get(scope, budget_name)

        notifications: dict[str, Notification] = {}
        for pct in thresholds:
            key = f"alert_{int(pct)}_pct"
            notifications[key] = Notification(
                enabled=True,
                operator="GreaterThanOrEqualTo",
                threshold=pct,
                contact_emails=[self.budget_config.get("alert_email", "")],
                contact_roles=["Owner", "Contributor"],
            )

        budget.notifications = notifications
        result = self._cost_client.budgets.create_or_update(scope, budget_name, budget)
        logger.info(
            "Alert thresholds %s set on budget '%s'.", thresholds, budget_name
        )
        return result.as_dict()

    # ── Spend queries ─────────────────────────────────────────────────────────

    def get_current_spend(self) -> float:
        """Query the current month's billed spend for the subscription.

        Returns:
            Total spend in USD for the current billing month. Returns ``0.0``
            if the client is unavailable or the query returns no rows.
        """
        if self._consumption_client is None:
            logger.warning("Consumption client unavailable; returning 0.0.")
            return 0.0

        now = datetime.now(tz=timezone.utc)
        month_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)

        usage = self._consumption_client.usage_details.list(
            scope=f"/subscriptions/{self.subscription_id}",
            expand="properties/additionalInfo",
            filter=(
                f"properties/usageStart ge '{month_start.isoformat()}' and "
                f"properties/usageEnd le '{now.isoformat()}'"
            ),
        )

        total: float = 0.0
        for item in usage:
            total += getattr(item.properties, "pretax_cost", 0.0) or 0.0

        logger.info("Current month spend: $%.2f", total)
        return total

    # ── Alerting ──────────────────────────────────────────────────────────────

    def send_alert(self, subject: str, message: str) -> None:
        """Send an email alert via SMTP.

        Reads SMTP credentials from environment variables:
        ``SMTP_HOST``, ``SMTP_PORT``, ``SMTP_USER``, ``SMTP_PASSWORD``.

        Args:
            subject: Email subject line.
            message: Plain-text email body.
        """
        import os

        smtp_host = os.environ.get("SMTP_HOST", "smtp.gmail.com")
        smtp_port = int(os.environ.get("SMTP_PORT", "587"))
        smtp_user = os.environ.get("SMTP_USER", "")
        smtp_password = os.environ.get("SMTP_PASSWORD", "")
        recipient = self.budget_config.get("alert_email", smtp_user)

        if not smtp_user or not smtp_password:
            logger.error("SMTP credentials not configured; cannot send alert.")
            return

        msg = MIMEMultipart("alternative")
        msg["Subject"] = f"[ML Alpha Lab] {subject}"
        msg["From"] = smtp_user
        msg["To"] = recipient
        msg.attach(MIMEText(message, "plain"))

        try:
            with smtplib.SMTP(smtp_host, smtp_port) as server:
                server.ehlo()
                server.starttls()
                server.login(smtp_user, smtp_password)
                server.sendmail(smtp_user, [recipient], msg.as_string())
            logger.info("Alert email sent to %s: %s", recipient, subject)
        except smtplib.SMTPException as exc:
            logger.error("Failed to send alert email: %s", exc)

    # ── Convenience ───────────────────────────────────────────────────────────

    def check_and_alert(self) -> None:
        """Check current spend against configured thresholds and fire alerts.

        Reads ``budget.monthly_limit_usd`` and ``budget.alert_thresholds``
        from the loaded configuration.
        """
        limit: float = float(self.budget_config.get("monthly_limit_usd", 100))
        thresholds: list[float] = [float(t) for t in self.budget_config.get("alert_thresholds", [])]
        spend = self.get_current_spend()
        spend_pct = (spend / limit * 100) if limit > 0 else 0.0

        for threshold in sorted(thresholds, reverse=True):
            if spend_pct >= threshold:
                subject = f"Budget alert: {spend_pct:.1f}% of ${limit:.0f} monthly limit used"
                body = (
                    f"Current month spend: ${spend:.2f}\n"
                    f"Monthly limit:       ${limit:.2f}\n"
                    f"Usage:               {spend_pct:.1f}%\n\n"
                    f"Threshold triggered: {threshold}%\n"
                )
                self.send_alert(subject, body)
                break  # Only fire the highest applicable threshold


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Azure budget alert manager for ML Alpha Lab.")
    parser.add_argument("--config", default="configs/azure_config.yaml")
    parser.add_argument("--check", action="store_true", help="Check current spend and fire alerts.")
    parser.add_argument("--create-budget", action="store_true", help="Create/update the monthly budget.")
    args = parser.parse_args()

    manager = BudgetAlertManager(args.config)

    if args.create_budget:
        limit = float(manager.budget_config.get("monthly_limit_usd", 100))
        thresholds = [float(t) for t in manager.budget_config.get("alert_thresholds", [])]
        budget = manager.create_budget("ml-alpha-lab-monthly", limit)
        manager.set_alert_thresholds("ml-alpha-lab-monthly", thresholds)
        print(f"Budget created: {budget}")

    if args.check:
        manager.check_and_alert()
