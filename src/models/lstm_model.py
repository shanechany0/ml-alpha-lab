"""LSTM model implementation for the ML Alpha Lab trading system."""

from __future__ import annotations

import logging
from typing import Any

import mlflow
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from src.models.base_model import BaseModel

logger = logging.getLogger(__name__)


class LSTMDataset(Dataset):
    """PyTorch Dataset that generates sliding-window sequences.

    Attributes:
        X: Sequence input tensor of shape (n_windows, seq_len, n_features).
        y: Target tensor of shape (n_windows,).
    """

    def __init__(self, X: torch.Tensor, y: torch.Tensor | None = None) -> None:
        """Initialize with pre-built sequence tensors.

        Args:
            X: Input tensor of shape (n_windows, seq_len, n_features).
            y: Optional target tensor of shape (n_windows,).
        """
        self.X = X
        self.y = y

    def __len__(self) -> int:
        """Return the number of windows."""
        return len(self.X)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, ...]:
        """Retrieve a single window.

        Args:
            idx: Window index.

        Returns:
            Tuple of ``(X_window,)`` or ``(X_window, y_target)`` if
            targets are provided.
        """
        if self.y is not None:
            return self.X[idx], self.y[idx]
        return (self.X[idx],)


class _LSTMNet(nn.Module):
    """Internal LSTM network with a linear prediction head.

    Attributes:
        lstm: Multi-layer LSTM with dropout.
        head: Linear layer projecting hidden state to scalar output.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        output_size: int = 1,
    ) -> None:
        """Build the LSTM + linear head.

        Args:
            input_size: Number of input features per time step.
            hidden_size: Number of hidden units in the LSTM.
            num_layers: Number of stacked LSTM layers.
            dropout: Dropout probability between LSTM layers.
            output_size: Dimensionality of the scalar output.
        """
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch, seq_len, n_features).

        Returns:
            Output tensor of shape (batch,).
        """
        out, _ = self.lstm(x)
        # Use the last time-step hidden state
        last = out[:, -1, :]
        last = self.dropout(last)
        return self.head(last).squeeze(-1)


class LSTMModel(BaseModel):
    """LSTM-based sequence model for return prediction.

    Implements sliding-window sequence creation, mini-batch training
    with an Adam optimizer and MSE loss, and early stopping on
    validation loss.

    Attributes:
        config: Configuration dictionary.
        network: Trained ``_LSTMNet`` module.
        device: Torch device (cuda if available, else cpu).
        seq_len: Sequence window length.
        hidden_size: LSTM hidden dimension.
        num_layers: Number of LSTM layers.
        dropout: Dropout rate.
        batch_size: Training batch size.
        lr: Adam learning rate.
        epochs: Maximum training epochs.
        patience: Early-stopping patience.
    """

    def __init__(self, config: dict | None = None) -> None:
        """Initialize the LSTM model.

        Args:
            config: Configuration dictionary. Recognises an ``lstm``
                sub-dict with keys: ``hidden_size``, ``num_layers``,
                ``dropout``, ``sequence_length``, ``batch_size``,
                ``learning_rate``, ``epochs``, ``patience``.
        """
        super().__init__(config)
        lstm_cfg = self.config.get("lstm", {})
        self.seq_len: int = lstm_cfg.get("sequence_length", 60)
        self.hidden_size: int = lstm_cfg.get("hidden_size", 128)
        self.num_layers: int = lstm_cfg.get("num_layers", 2)
        self.dropout: float = lstm_cfg.get("dropout", 0.3)
        self.batch_size: int = lstm_cfg.get("batch_size", 64)
        self.lr: float = lstm_cfg.get("learning_rate", 0.001)
        self.epochs: int = lstm_cfg.get("epochs", 100)
        self.patience: int = lstm_cfg.get("patience", 10)
        self.network: _LSTMNet | None = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _build_network(self, input_size: int) -> nn.Module:
        """Build and return the LSTM network.

        Args:
            input_size: Number of features per time step.

        Returns:
            Initialised ``_LSTMNet`` module moved to ``self.device``.
        """
        net = _LSTMNet(
            input_size=input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
        )
        return net.to(self.device)

    def _prepare_sequences(
        self,
        X: np.ndarray,
        y: np.ndarray | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Create sliding-window sequences from a flat feature matrix.

        For an input of shape ``(T, F)``, produces windows of shape
        ``(T - seq_len, seq_len, F)`` and targets of shape
        ``(T - seq_len,)``.

        Args:
            X: Feature matrix of shape (n_samples, n_features).
            y: Optional target array of shape (n_samples,).

        Returns:
            Tuple of ``(X_seq, y_seq)`` tensors. ``y_seq`` is ``None``
            if ``y`` was not provided.
        """
        X_arr = np.array(X, dtype=np.float32)
        n = len(X_arr)
        n_windows = n - self.seq_len

        X_windows = np.stack([X_arr[i : i + self.seq_len] for i in range(n_windows)])
        X_tensor = torch.tensor(X_windows, dtype=torch.float32)

        y_tensor = None
        if y is not None:
            y_arr = np.array(y, dtype=np.float32)
            y_seq = y_arr[self.seq_len :]
            y_tensor = torch.tensor(y_seq, dtype=torch.float32)

        return X_tensor, y_tensor

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
    ) -> "LSTMModel":
        """Train the LSTM with Adam, MSE loss, and early stopping.

        Args:
            X_train: Training feature matrix of shape (n_samples, n_features).
            y_train: Training targets of shape (n_samples,).
            X_val: Optional validation features.
            y_val: Optional validation targets.

        Returns:
            Self, for method chaining.
        """
        if isinstance(X_train, pd.DataFrame):
            self.feature_names = list(X_train.columns)
            X_train = X_train.values
        if isinstance(X_val, pd.DataFrame):
            X_val = X_val.values

        input_size = X_train.shape[1]
        self.network = self._build_network(input_size)
        optimizer = torch.optim.Adam(self.network.parameters(), lr=self.lr)
        criterion = nn.MSELoss()

        X_seq, y_seq = self._prepare_sequences(X_train, y_train)
        train_loader = DataLoader(
            LSTMDataset(X_seq, y_seq),
            batch_size=self.batch_size,
            shuffle=True,
        )

        val_loader: DataLoader | None = None
        if X_val is not None and y_val is not None:
            X_val_seq, y_val_seq = self._prepare_sequences(X_val, y_val)
            val_loader = DataLoader(
                LSTMDataset(X_val_seq, y_val_seq),
                batch_size=self.batch_size,
                shuffle=False,
            )

        best_val_loss = float("inf")
        no_improve = 0
        best_state: dict[str, Any] = {}

        with mlflow.start_run(nested=True):
            mlflow.log_params(
                {
                    "hidden_size": self.hidden_size,
                    "num_layers": self.num_layers,
                    "dropout": self.dropout,
                    "sequence_length": self.seq_len,
                    "learning_rate": self.lr,
                }
            )

            for epoch in range(1, self.epochs + 1):
                self.network.train()
                train_losses = []
                for batch in train_loader:
                    xb, yb = batch
                    xb, yb = xb.to(self.device), yb.to(self.device)
                    optimizer.zero_grad()
                    preds = self.network(xb)
                    loss = criterion(preds, yb)
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=1.0)
                    optimizer.step()
                    train_losses.append(loss.item())

                train_loss = float(np.mean(train_losses))
                mlflow.log_metric("train_loss", train_loss, step=epoch)

                if val_loader is not None:
                    val_loss = self._eval_loss(val_loader, criterion)
                    mlflow.log_metric("val_loss", val_loss, step=epoch)

                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        no_improve = 0
                        best_state = {k: v.cpu().clone() for k, v in self.network.state_dict().items()}
                    else:
                        no_improve += 1
                        if no_improve >= self.patience:
                            logger.info("Early stopping at epoch %d", epoch)
                            break
                else:
                    best_state = {k: v.cpu().clone() for k, v in self.network.state_dict().items()}

                if epoch % 10 == 0:
                    logger.debug("Epoch %d/%d  train_loss=%.6f", epoch, self.epochs, train_loss)

            if best_state:
                self.network.load_state_dict({k: v.to(self.device) for k, v in best_state.items()})

        return self

    def _eval_loss(self, loader: DataLoader, criterion: nn.Module) -> float:
        """Evaluate mean loss on a DataLoader.

        Args:
            loader: DataLoader to evaluate on.
            criterion: Loss function.

        Returns:
            Mean loss value over all batches.
        """
        self.network.eval()
        losses = []
        with torch.no_grad():
            for batch in loader:
                xb, yb = batch
                xb, yb = xb.to(self.device), yb.to(self.device)
                preds = self.network(xb)
                losses.append(criterion(preds, yb).item())
        return float(np.mean(losses))

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Run inference on feature matrix.

        Args:
            X: Feature matrix of shape (n_samples, n_features).

        Returns:
            Predictions array of shape (n_samples - seq_len,).
        """
        if isinstance(X, pd.DataFrame):
            X = X.values

        X_seq, _ = self._prepare_sequences(X)
        dataset = LSTMDataset(X_seq)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        self.network.eval()
        results = []
        with torch.no_grad():
            for (xb,) in loader:
                xb = xb.to(self.device)
                results.append(self.network(xb).cpu().numpy())

        return np.concatenate(results)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return predictions (regression task alias for predict).

        Args:
            X: Feature matrix of shape (n_samples, n_features).

        Returns:
            Same as ``predict(X)``.
        """
        return self.predict(X)
