"""MLflow-backed model registry for the ML Alpha Lab trading system."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import mlflow
import mlflow.pyfunc
import pandas as pd
import yaml

from src.models.base_model import BaseModel

logger = logging.getLogger(__name__)


class ModelRegistry:
    """Centralised MLflow model registry with comparison utilities.

    Wraps the MLflow Model Registry API to provide a higher-level
    interface for registering, listing, and comparing model versions.

    Attributes:
        config: Configuration dictionary loaded from ``config_path``.
        client: ``mlflow.MlflowClient`` instance.
    """

    def __init__(self, config_path: str | None = None) -> None:
        """Initialize the registry and configure MLflow.

        Args:
            config_path: Optional path to a YAML configuration file.
                If provided and the file exists, it is loaded and the
                ``mlflow_tracking_uri`` key (if present) is used.
        """
        self.config: dict = {}
        if config_path is not None:
            cfg_file = Path(config_path)
            if cfg_file.exists():
                with open(cfg_file) as f:
                    self.config = yaml.safe_load(f) or {}

        tracking_uri = self.config.get("mlflow_tracking_uri", "mlruns")
        mlflow.set_tracking_uri(tracking_uri)
        self.client = mlflow.MlflowClient()

    def register_model(
        self,
        model: BaseModel,
        name: str,
        metrics: dict[str, float],
    ) -> str:
        """Log and register a model in the MLflow registry.

        Starts a new MLflow run, logs all provided metrics, logs the
        model as a ``pyfunc`` artifact, then registers it under ``name``.

        Args:
            model: Fitted ``BaseModel`` instance to register.
            name: Registry model name.
            metrics: Dictionary of metric names to float values.

        Returns:
            The registered model version string.
        """
        with mlflow.start_run() as run:
            mlflow.log_metrics({k: v for k, v in metrics.items() if not _is_invalid(v)})
            mlflow.sklearn.log_model(model, artifact_path="model", registered_model_name=name)
            run_id = run.info.run_id

        registered = mlflow.register_model(f"runs:/{run_id}/model", name)
        version = registered.version
        logger.info("Registered model '%s' version %s", name, version)
        return str(version)

    def get_best_model(self, metric: str = "sharpe_ratio") -> tuple[str, Any]:
        """Retrieve the registered model version with the highest metric value.

        Searches all runs that have the specified metric logged and
        returns the name and loaded model for the run with the maximum
        value.

        Args:
            metric: Name of the metric to rank by.

        Returns:
            Tuple of ``(model_name, loaded_model)`` for the best version.

        Raises:
            ValueError: If no runs with the specified metric are found.
        """
        runs = mlflow.search_runs(order_by=[f"metrics.{metric} DESC"], max_results=1)
        if runs.empty or f"metrics.{metric}" not in runs.columns:
            raise ValueError(f"No runs found with metric '{metric}'.")

        best_run = runs.iloc[0]
        run_id = best_run["run_id"]

        # Determine which registered model name corresponds to this run
        model_name = self._name_for_run(run_id)
        loaded = mlflow.pyfunc.load_model(f"runs:/{run_id}/model")
        logger.info("Best model: %s (run %s, %s=%.4f)", model_name, run_id, metric, best_run[f"metrics.{metric}"])
        return model_name, loaded

    def list_models(self) -> pd.DataFrame:
        """List all registered models and their latest versions.

        Returns:
            DataFrame with columns ``name``, ``version``,
            ``creation_timestamp``, ``last_updated_timestamp``,
            and ``current_stage``.
        """
        registered = self.client.search_registered_models()
        rows = []
        for rm in registered:
            for mv in rm.latest_versions:
                rows.append(
                    {
                        "name": rm.name,
                        "version": mv.version,
                        "creation_timestamp": mv.creation_timestamp,
                        "last_updated_timestamp": mv.last_updated_timestamp,
                        "current_stage": mv.current_stage,
                    }
                )
        return pd.DataFrame(rows)

    def load_model(self, name: str, version: str | None = None) -> Any:
        """Load a model from the MLflow registry.

        Args:
            name: Registered model name.
            version: Specific version string. If ``None``, loads the
                latest version (using the ``"None"`` stage alias).

        Returns:
            Loaded ``mlflow.pyfunc.PythonModel`` instance.
        """
        if version is None:
            uri = f"models:/{name}/latest"
        else:
            uri = f"models:/{name}/{version}"
        loaded = mlflow.pyfunc.load_model(uri)
        logger.info("Loaded model '%s' version %s", name, version or "latest")
        return loaded

    def compare_models(self, metric: str = "sharpe_ratio") -> pd.DataFrame:
        """Compare all registered model versions by a given metric.

        Args:
            metric: Metric name to compare across all versions.

        Returns:
            DataFrame sorted descending by ``metric`` with columns
            ``run_id``, ``model_name``, ``version``, and the metric value.
        """
        col = f"metrics.{metric}"
        runs = mlflow.search_runs(order_by=[f"{col} DESC"])
        if col not in runs.columns:
            logger.warning("Metric '%s' not found in any run.", metric)
            return pd.DataFrame()

        result = runs[["run_id", col]].dropna(subset=[col]).copy()
        result.rename(columns={col: metric}, inplace=True)

        # Attach registered model names where available
        result["model_name"] = result["run_id"].apply(self._name_for_run)
        return result.sort_values(metric, ascending=False).reset_index(drop=True)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _name_for_run(self, run_id: str) -> str:
        """Return the registered model name associated with a run ID.

        Falls back to the run ID string if no registered model is found.

        Args:
            run_id: MLflow run identifier.

        Returns:
            Registered model name or the run ID string.
        """
        try:
            versions = self.client.search_model_versions(f"run_id='{run_id}'")
            if versions:
                return versions[0].name
        except Exception:  # noqa: BLE001
            pass
        return run_id


def _is_invalid(value: float) -> bool:
    """Return True if value is NaN or infinite."""
    import math

    return math.isnan(value) or math.isinf(value)


def main() -> None:
    """CLI entry point for model registry operations."""
    parser = argparse.ArgumentParser(description="ML Alpha Lab — Model Registry CLI")
    parser.add_argument("--config", default=None, help="Path to YAML config file")
    subparsers = parser.add_subparsers(dest="command")

    list_p = subparsers.add_parser("list", help="List all registered models")
    list_p.add_argument("--output", default=None, help="CSV output path")

    compare_p = subparsers.add_parser("compare", help="Compare model versions by metric")
    compare_p.add_argument("--metric", default="sharpe_ratio")
    compare_p.add_argument("--output", default=None)

    best_p = subparsers.add_parser("best", help="Print the best model by metric")
    best_p.add_argument("--metric", default="sharpe_ratio")

    args = parser.parse_args()
    registry = ModelRegistry(config_path=args.config)

    if args.command == "list":
        df = registry.list_models()
        print(df.to_string(index=False))
        if args.output:
            df.to_csv(args.output, index=False)

    elif args.command == "compare":
        df = registry.compare_models(metric=args.metric)
        print(df.to_string(index=False))
        if args.output:
            df.to_csv(args.output, index=False)

    elif args.command == "best":
        name, _ = registry.get_best_model(metric=args.metric)
        print(f"Best model: {name}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
