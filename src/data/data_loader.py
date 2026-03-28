"""Data loading utilities for ML Alpha Lab."""

import logging
import os
from typing import Any

import pandas as pd
import yaml

logger = logging.getLogger(__name__)


class DataLoader:
    """Loads market data from various sources.

    Supports Yahoo Finance, CSV files, Azure Blob Storage, and
    config-defined universes.

    Attributes:
        config: Configuration dictionary loaded from backtest_config.yaml.
    """

    def __init__(self, config_path: str | None = None) -> None:
        """Initialise the DataLoader.

        Args:
            config_path: Path to backtest_config.yaml. If None, searches
                for the file relative to the current working directory.
        """
        self.config: dict[str, Any] = {}
        if config_path is None:
            default = os.path.join(
                os.path.dirname(__file__), "..", "..", "configs", "backtest_config.yaml"
            )
            config_path = default
        if os.path.exists(config_path):
            with open(config_path, "r", encoding="utf-8") as fh:
                self.config = yaml.safe_load(fh) or {}
            logger.info("Loaded config from %s", config_path)
        else:
            logger.warning("Config file not found at %s; using empty config", config_path)

    def load_yahoo_finance(
        self, tickers: list[str], start_date: str, end_date: str
    ) -> pd.DataFrame:
        """Download OHLCV data from Yahoo Finance.

        Args:
            tickers: List of ticker symbols.
            start_date: Start date string in 'YYYY-MM-DD' format.
            end_date: End date string in 'YYYY-MM-DD' format.

        Returns:
            Multi-index DataFrame with (field, ticker) columns.

        Raises:
            ImportError: If yfinance is not installed.
        """
        try:
            import yfinance as yf
        except ImportError as exc:
            raise ImportError("yfinance is required: pip install yfinance") from exc

        logger.info(
            "Downloading %d tickers from Yahoo Finance (%s to %s)",
            len(tickers),
            start_date,
            end_date,
        )
        data: pd.DataFrame = yf.download(
            tickers,
            start=start_date,
            end=end_date,
            auto_adjust=False,
            progress=False,
        )
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.set_levels(
                [lvl.lower() for lvl in data.columns.levels[0]], level=0
            )
        logger.info("Downloaded data shape: %s", data.shape)
        return data

    def load_csv(self, filepath: str) -> pd.DataFrame:
        """Load market data from a CSV file.

        Args:
            filepath: Absolute or relative path to the CSV file.

        Returns:
            DataFrame loaded from the CSV, with the first column parsed
            as a DatetimeIndex when it contains date-like values.

        Raises:
            FileNotFoundError: If the file does not exist.
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"CSV file not found: {filepath}")
        logger.info("Loading CSV from %s", filepath)
        df = pd.read_csv(filepath, index_col=0, parse_dates=True)
        logger.info("Loaded CSV shape: %s", df.shape)
        return df

    def load_from_blob(self, blob_path: str) -> pd.DataFrame:
        """Load market data from Azure Blob Storage.

        Expects the environment variables ``AZURE_STORAGE_CONNECTION_STRING``
        or ``AZURE_STORAGE_ACCOUNT`` + ``AZURE_STORAGE_KEY`` to be set.

        Args:
            blob_path: Blob path in the format ``container/blob_name``.

        Returns:
            DataFrame loaded from the blob (CSV format assumed).

        Raises:
            ImportError: If azure-storage-blob is not installed.
            ValueError: If blob_path format is invalid.
        """
        try:
            from azure.storage.blob import BlobServiceClient
        except ImportError as exc:
            raise ImportError(
                "azure-storage-blob is required: pip install azure-storage-blob"
            ) from exc

        parts = blob_path.split("/", 1)
        if len(parts) != 2:
            raise ValueError(
                "blob_path must be in the format 'container/blob_name', "
                f"got: {blob_path!r}"
            )
        container_name, blob_name = parts

        conn_str = os.environ.get("AZURE_STORAGE_CONNECTION_STRING")
        if conn_str:
            client = BlobServiceClient.from_connection_string(conn_str)
        else:
            account = os.environ.get("AZURE_STORAGE_ACCOUNT", "")
            key = os.environ.get("AZURE_STORAGE_KEY", "")
            account_url = f"https://{account}.blob.core.windows.net"
            from azure.storage.blob import StorageSharedKeyCredential  # type: ignore[attr-defined]

            credential = StorageSharedKeyCredential(account, key)
            client = BlobServiceClient(account_url=account_url, credential=credential)

        logger.info("Downloading blob: %s", blob_path)
        blob_client = client.get_blob_client(container=container_name, blob=blob_name)
        stream = blob_client.download_blob()
        import io

        content = stream.readall()
        df = pd.read_csv(io.BytesIO(content), index_col=0, parse_dates=True)
        logger.info("Loaded blob shape: %s", df.shape)
        return df

    def get_universe(
        self, start_date: str | None = None, end_date: str | None = None
    ) -> pd.DataFrame:
        """Load the universe defined in the backtest configuration.

        Args:
            start_date: Override start date (YYYY-MM-DD). Falls back to
                config value.
            end_date: Override end date (YYYY-MM-DD). Falls back to config
                value.

        Returns:
            Multi-index DataFrame of OHLCV data for the configured universe.

        Raises:
            KeyError: If the universe configuration is missing.
        """
        universe_cfg = self.config.get("universe", {})
        tickers: list[str] = universe_cfg.get("tickers", self.get_sp500_tickers())
        sd = start_date or self.config.get("backtest", {}).get("start_date", "2018-01-01")
        ed = end_date or self.config.get("backtest", {}).get("end_date", "2023-12-31")
        logger.info(
            "Loading universe of %d tickers from %s to %s", len(tickers), sd, ed
        )
        return self.load_yahoo_finance(tickers, sd, ed)

    @staticmethod
    def get_sp500_tickers() -> list[str]:
        """Return a representative subset of S&P 500 tickers.

        Returns:
            List of S&P 500 ticker symbols (subset of ~50 liquid names).
        """
        return [
            "AAPL", "MSFT", "AMZN", "GOOGL", "META", "NVDA", "TSLA", "JPM",
            "JNJ", "V", "PG", "UNH", "HD", "MA", "DIS", "BAC", "ADBE", "CRM",
            "NFLX", "CMCSA", "VZ", "KO", "PEP", "ABT", "TMO", "ACN", "COST",
            "AVGO", "TXN", "NKE", "MRK", "WMT", "LLY", "CVX", "ABBV", "XOM",
            "INTC", "QCOM", "MDT", "UPS", "HON", "AMGN", "BMY", "LIN", "SBUX",
            "GS", "BLK", "MMM", "CAT", "DE",
        ]
