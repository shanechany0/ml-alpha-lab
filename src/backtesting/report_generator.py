"""HTML report generator for backtesting results."""

from __future__ import annotations

import base64
import io
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from jinja2 import BaseLoader, Environment

from src.backtesting.performance_metrics import PerformanceMetrics

matplotlib.use("Agg")

_HTML_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>{{ title }}</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background: #f9f9f9; }
        h1 { color: #2c3e50; }
        h2 { color: #34495e; border-bottom: 2px solid #bdc3c7; padding-bottom: 4px; }
        table { border-collapse: collapse; width: 100%; margin-bottom: 20px; }
        th { background: #2c3e50; color: white; padding: 8px 12px; text-align: left; }
        td { padding: 6px 12px; border-bottom: 1px solid #ddd; }
        tr:nth-child(even) { background: #ecf0f1; }
        .chart { margin: 20px 0; }
        img { max-width: 100%; border: 1px solid #ddd; border-radius: 4px; }
        .metric-value { font-weight: bold; }
        .positive { color: #27ae60; }
        .negative { color: #e74c3c; }
    </style>
</head>
<body>
    <h1>{{ title }}</h1>
    <p>Generated: {{ generated_at }}</p>

    <h2>Performance Metrics</h2>
    {{ metrics_table }}

    <h2>Equity Curve</h2>
    <div class="chart">
        <img src="data:image/png;base64,{{ equity_chart }}" alt="Equity Curve" />
    </div>

    <h2>Drawdown</h2>
    <div class="chart">
        <img src="data:image/png;base64,{{ drawdown_chart }}" alt="Drawdown" />
    </div>

    <h2>Monthly Returns Heatmap</h2>
    <div class="chart">
        <img src="data:image/png;base64,{{ monthly_heatmap }}" alt="Monthly Returns" />
    </div>
</body>
</html>
"""


class ReportGenerator:
    """Generates HTML performance reports with embedded charts.

    Attributes:
        config: Configuration dictionary.
        metrics: PerformanceMetrics instance.
    """

    def __init__(self, config: dict | None = None) -> None:
        """Initializes ReportGenerator.

        Args:
            config: Optional configuration. Recognized keys:
                'title', 'risk_free_rate', 'annualization'.
        """
        self.config = config or {}
        self.metrics = PerformanceMetrics(
            risk_free_rate=self.config.get("risk_free_rate", 0.05),
            annualization=self.config.get("annualization", 252),
        )

    def generate_html_report(
        self,
        results: dict[str, Any],
        output_path: str = "outputs/report.html",
    ) -> str:
        """Generates a full HTML report and writes it to disk.

        Args:
            results: Backtesting results dictionary. Should contain at minimum
                'strategy_returns' (pd.Series) and optionally 'statistics'
                (dict of metrics).
            output_path: File path for the HTML output.

        Returns:
            Absolute path to the generated HTML file.
        """
        returns: pd.Series = results.get("strategy_returns", pd.Series(dtype=float))
        metrics_dict: dict[str, float] = results.get(
            "statistics", self.metrics.compute_all(returns)
        )

        context = {
            "title": self.config.get("title", "ML Alpha Lab — Backtest Report"),
            "generated_at": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
            "metrics_table": self.metrics_table(metrics_dict),
            "equity_chart": self.equity_curve_chart(returns),
            "drawdown_chart": self.drawdown_chart(returns),
            "monthly_heatmap": self.monthly_returns_heatmap(returns),
        }

        html = self._render_template(_HTML_TEMPLATE, context)
        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(html, encoding="utf-8")
        return str(out_path.resolve())

    def equity_curve_chart(self, returns: pd.Series) -> str:
        """Renders a cumulative equity curve chart as base64 PNG.

        Args:
            returns: Daily strategy return series.

        Returns:
            Base64-encoded PNG string.
        """
        fig, ax = plt.subplots(figsize=(10, 4))
        if not returns.empty:
            cumulative = (1 + returns).cumprod()
            ax.plot(cumulative.index, cumulative.values, color="steelblue", linewidth=1.5)
            ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.8)
        ax.set_title("Equity Curve")
        ax.set_ylabel("Cumulative Return")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        return self._fig_to_base64(fig)

    def drawdown_chart(self, returns: pd.Series) -> str:
        """Renders a drawdown chart as base64 PNG.

        Args:
            returns: Daily strategy return series.

        Returns:
            Base64-encoded PNG string.
        """
        fig, ax = plt.subplots(figsize=(10, 3))
        if not returns.empty:
            dd = self.metrics.drawdown_series(returns)
            ax.fill_between(dd.index, dd.values, 0, color="red", alpha=0.5)
            ax.plot(dd.index, dd.values, color="darkred", linewidth=0.8)
        ax.set_title("Drawdown")
        ax.set_ylabel("Drawdown")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        return self._fig_to_base64(fig)

    def monthly_returns_heatmap(self, returns: pd.Series) -> str:
        """Renders a monthly returns heatmap as base64 PNG.

        Args:
            returns: Daily strategy return series.

        Returns:
            Base64-encoded PNG string.
        """
        fig, ax = plt.subplots(figsize=(12, 5))
        if returns.empty or not hasattr(returns.index, "month"):
            ax.text(0.5, 0.5, "No data", ha="center", va="center")
            return self._fig_to_base64(fig)

        monthly = returns.resample("ME").apply(lambda x: (1 + x).prod() - 1)
        monthly_df = monthly.to_frame("return")
        monthly_df["year"] = monthly_df.index.year
        monthly_df["month"] = monthly_df.index.month

        pivot = monthly_df.pivot(index="year", columns="month", values="return")
        pivot.columns = [
            "Jan", "Feb", "Mar", "Apr", "May", "Jun",
            "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
        ][: len(pivot.columns)]

        sns.heatmap(
            pivot * 100,
            annot=True,
            fmt=".1f",
            center=0,
            cmap="RdYlGn",
            linewidths=0.5,
            ax=ax,
            cbar_kws={"label": "Return (%)"},
        )
        ax.set_title("Monthly Returns (%)")
        plt.tight_layout()
        return self._fig_to_base64(fig)

    def metrics_table(self, metrics: dict[str, float]) -> str:
        """Renders a performance metrics dictionary as an HTML table.

        Args:
            metrics: Dictionary of metric name → float value.

        Returns:
            HTML string for the metrics table.
        """
        rows = []
        for key, value in metrics.items():
            label = key.replace("_", " ").title()
            if isinstance(value, float):
                css = "positive" if value >= 0 else "negative"
                formatted = f"{value:.4f}"
            else:
                css = ""
                formatted = str(value)
            rows.append(
                f'<tr><td>{label}</td>'
                f'<td class="metric-value {css}">{formatted}</td></tr>'
            )
        return (
            "<table><thead><tr><th>Metric</th><th>Value</th></tr></thead>"
            f"<tbody>{''.join(rows)}</tbody></table>"
        )

    def _render_template(self, template_str: str, context: dict) -> str:
        """Renders a Jinja2 template string with the given context.

        Args:
            template_str: Jinja2 template as a string.
            context: Template variable dictionary.

        Returns:
            Rendered HTML string.
        """
        env = Environment(loader=BaseLoader(), autoescape=False)
        template = env.from_string(template_str)
        return template.render(**context)

    @staticmethod
    def _fig_to_base64(fig: plt.Figure) -> str:
        """Converts a matplotlib Figure to a base64-encoded PNG string.

        Args:
            fig: Matplotlib figure to encode.

        Returns:
            Base64 string of the PNG image.
        """
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        return base64.b64encode(buf.read()).decode("utf-8")
