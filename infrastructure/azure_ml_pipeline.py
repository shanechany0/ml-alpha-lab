"""Azure ML pipeline definitions for ML Alpha Lab."""
from __future__ import annotations

import logging
from pathlib import Path

import mlflow
import yaml
from azure.ai.ml import Input, MLClient, Output, command
from azure.ai.ml.constants import AssetTypes
from azure.ai.ml.dsl import pipeline
from azure.identity import DefaultAzureCredential

logger = logging.getLogger(__name__)


class MLAlphaPipeline:
    """Builds and submits the end-to-end ML Alpha Lab Azure ML pipeline.

    Args:
        config_path: Path to ``azure_config.yaml``.
    """

    def __init__(self, config_path: str | Path) -> None:
        """Initialise the pipeline manager and connect to Azure ML.

        Args:
            config_path: Path to the Azure configuration YAML file.
        """
        with open(config_path) as fh:
            self.config = yaml.safe_load(fh)

        azure_cfg = self.config["azure"]
        self.ml_client = MLClient(
            credential=DefaultAzureCredential(),
            subscription_id=azure_cfg["subscription_id"],
            resource_group_name=azure_cfg["resource_group"],
            workspace_name=azure_cfg["workspace_name"],
        )

        mlflow_cfg = self.config.get("mlflow", {})
        mlflow.set_tracking_uri(mlflow_cfg.get("tracking_uri", "azureml"))
        mlflow.set_experiment(mlflow_cfg.get("experiment_name", "ml-alpha-lab"))
        logger.info("Connected to Azure ML workspace: %s", azure_cfg["workspace_name"])

    # ── Pipeline steps ────────────────────────────────────────────────────────

    def create_data_ingestion_step(self) -> command:
        """Create the data ingestion CommandJob step.

        Returns:
            A configured :class:`azure.ai.ml.command` component for data ingestion.
        """
        compute = self.config["compute"]["cpu_cluster"]["name"]
        storage = self.config["storage"]

        return command(
            name="data_ingestion",
            display_name="Data Ingestion",
            description="Download and validate raw market data.",
            command="python -m src.data.data_ingestion --output ${{outputs.raw_data}}",
            environment="azureml:ml-alpha-lab-env:latest",
            compute=compute,
            outputs={
                "raw_data": Output(
                    type=AssetTypes.URI_FOLDER,
                    path=f"azureml://datastores/workspaceblobstore/paths/{storage['paths']['raw_data']}",
                )
            },
        )

    def create_feature_engineering_step(self) -> command:
        """Create the feature engineering CommandJob step.

        Returns:
            A configured :class:`azure.ai.ml.command` component for feature engineering.
        """
        compute = self.config["compute"]["cpu_cluster"]["name"]
        storage = self.config["storage"]

        return command(
            name="feature_engineering",
            display_name="Feature Engineering",
            description="Compute technical indicators and alpha factors.",
            command=(
                "python -m src.features.feature_engineering "
                "--input ${{inputs.raw_data}} "
                "--output ${{outputs.features}}"
            ),
            environment="azureml:ml-alpha-lab-env:latest",
            compute=compute,
            inputs={
                "raw_data": Input(type=AssetTypes.URI_FOLDER),
            },
            outputs={
                "features": Output(
                    type=AssetTypes.URI_FOLDER,
                    path=f"azureml://datastores/workspaceblobstore/paths/{storage['paths']['features']}",
                )
            },
        )

    def create_model_training_step(self) -> command:
        """Create the model training CommandJob step.

        Returns:
            A configured :class:`azure.ai.ml.command` component for model training.
        """
        compute = self.config["compute"]["gpu_cluster"]["name"]
        storage = self.config["storage"]

        return command(
            name="model_training",
            display_name="Model Training",
            description="Train LightGBM, XGBoost, LSTM, Transformer and RL models.",
            command=(
                "python -m src.models.model_registry "
                "--input ${{inputs.features}} "
                "--output ${{outputs.models}} "
                "--config configs/model_config.yaml"
            ),
            environment="azureml:ml-alpha-lab-env:latest",
            compute=compute,
            inputs={
                "features": Input(type=AssetTypes.URI_FOLDER),
            },
            outputs={
                "models": Output(
                    type=AssetTypes.URI_FOLDER,
                    path=f"azureml://datastores/workspaceblobstore/paths/{storage['paths']['models']}",
                )
            },
        )

    def create_backtest_step(self) -> command:
        """Create the backtesting CommandJob step.

        Returns:
            A configured :class:`azure.ai.ml.command` component for backtesting.
        """
        compute = self.config["compute"]["cpu_cluster"]["name"]
        storage = self.config["storage"]

        return command(
            name="backtesting",
            display_name="Walk-Forward Backtesting",
            description="Run walk-forward backtest and generate performance reports.",
            command=(
                "python -m src.backtesting.backtest_engine "
                "--features ${{inputs.features}} "
                "--models ${{inputs.models}} "
                "--output ${{outputs.backtests}} "
                "--config configs/backtest_config.yaml"
            ),
            environment="azureml:ml-alpha-lab-env:latest",
            compute=compute,
            inputs={
                "features": Input(type=AssetTypes.URI_FOLDER),
                "models": Input(type=AssetTypes.URI_FOLDER),
            },
            outputs={
                "backtests": Output(
                    type=AssetTypes.URI_FOLDER,
                    path=f"azureml://datastores/workspaceblobstore/paths/{storage['paths']['backtests']}",
                )
            },
        )

    # ── Pipeline assembly ─────────────────────────────────────────────────────

    def build_pipeline(self):
        """Assemble all steps into a single Azure ML pipeline.

        Returns:
            A compiled Azure ML pipeline function ready for submission.
        """
        ingest = self.create_data_ingestion_step()
        feature_eng = self.create_feature_engineering_step()
        train = self.create_model_training_step()
        backtest = self.create_backtest_step()

        @pipeline(
            name="ml_alpha_lab_pipeline",
            description="End-to-end ML Alpha Lab pipeline: ingest → features → train → backtest",
        )
        def _pipeline():
            ingest_step = ingest()
            feature_step = feature_eng(raw_data=ingest_step.outputs.raw_data)
            train_step = train(features=feature_step.outputs.features)
            backtest(
                features=feature_step.outputs.features,
                models=train_step.outputs.models,
            )

        return _pipeline

    def submit_pipeline(self) -> str:
        """Build and submit the pipeline to Azure ML.

        Returns:
            The run ID of the submitted pipeline job.
        """
        built_pipeline = self.build_pipeline()
        pipeline_job = self.ml_client.jobs.create_or_update(
            built_pipeline(),
            experiment_name=self.config["mlflow"]["experiment_name"],
        )
        logger.info("Pipeline submitted. Job name: %s", pipeline_job.name)
        return pipeline_job.name


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Submit the ML Alpha Lab pipeline to Azure ML.")
    parser.add_argument(
        "--config",
        default="configs/azure_config.yaml",
        help="Path to azure_config.yaml",
    )
    args = parser.parse_args()

    manager = MLAlphaPipeline(args.config)
    job_name = manager.submit_pipeline()
    print(f"Submitted pipeline job: {job_name}")
