"""Transformer-based model for the ML Alpha Lab trading system."""

from __future__ import annotations

import logging
import math
from typing import Any

import mlflow
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.models.base_model import BaseModel

logger = logging.getLogger(__name__)


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for Transformer inputs.

    Adds fixed sinusoidal position embeddings to the input sequence,
    following the formulation in *Attention Is All You Need*.

    Attributes:
        pe: Pre-computed positional encoding buffer of shape
            ``(1, max_len, d_model)``.
    """

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000) -> None:
        """Initialise positional encoding.

        Args:
            d_model: Model embedding dimension.
            dropout: Dropout rate applied after adding the encoding.
            max_len: Maximum sequence length supported.
        """
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # Shape: (1, max_len, d_model) — broadcast over batch
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input embeddings.

        Args:
            x: Input tensor of shape (batch, seq_len, d_model).

        Returns:
            Tensor of the same shape with positional encoding added.
        """
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class TransformerPredictor(nn.Module):
    """Transformer encoder followed by a linear prediction head.

    Projects features to ``d_model``, adds positional encodings, passes
    through stacked encoder layers, then mean-pools and maps to a scalar.

    Attributes:
        input_projection: Linear layer mapping n_features → d_model.
        pos_encoder: Sinusoidal positional encoding module.
        transformer_encoder: Stack of ``TransformerEncoderLayer`` modules.
        head: Linear layer producing the scalar output.
    """

    def __init__(
        self,
        n_features: int,
        d_model: int,
        nhead: int,
        num_encoder_layers: int,
        dim_feedforward: int,
        dropout: float,
    ) -> None:
        """Build the Transformer predictor.

        Args:
            n_features: Number of input features per time step.
            d_model: Transformer model dimension (embedding size).
            nhead: Number of attention heads (must divide ``d_model``).
            num_encoder_layers: Number of stacked encoder layers.
            dim_feedforward: Feedforward sublayer hidden dimension.
            dropout: Dropout rate throughout the model.
        """
        super().__init__()
        self.input_projection = nn.Linear(n_features, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        self.head = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the Transformer predictor.

        Args:
            x: Input tensor of shape (batch, seq_len, n_features).

        Returns:
            Output tensor of shape (batch,).
        """
        x = self.input_projection(x)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        # Mean-pool over the sequence dimension
        x = x.mean(dim=1)
        return self.head(x).squeeze(-1)


class TransformerModel(BaseModel):
    """Transformer encoder model for return prediction.

    Uses a cosine-annealing LR scheduler and early stopping on
    validation loss. All metrics are logged to MLflow per epoch.

    Attributes:
        config: Configuration dictionary.
        network: Trained ``TransformerPredictor`` module.
        device: Torch device.
        seq_len: Input sequence window length.
        d_model: Transformer model dimension.
        nhead: Number of attention heads.
        num_encoder_layers: Number of encoder layers.
        dim_feedforward: Feedforward hidden dimension.
        dropout: Dropout rate.
        batch_size: Training batch size.
        lr: Adam learning rate.
        epochs: Maximum training epochs.
        patience: Early-stopping patience.
    """

    def __init__(self, config: dict | None = None) -> None:
        """Initialize the Transformer model.

        Args:
            config: Configuration dictionary. Recognises a
                ``transformer`` sub-dict with keys: ``d_model``,
                ``nhead``, ``num_encoder_layers``, ``dim_feedforward``,
                ``dropout``, ``sequence_length``, ``batch_size``,
                ``learning_rate``, ``epochs``, ``patience``.
        """
        super().__init__(config)
        tr_cfg = self.config.get("transformer", {})
        self.seq_len: int = tr_cfg.get("sequence_length", 60)
        self.d_model: int = tr_cfg.get("d_model", 64)
        self.nhead: int = tr_cfg.get("nhead", 4)
        self.num_encoder_layers: int = tr_cfg.get("num_encoder_layers", 3)
        self.dim_feedforward: int = tr_cfg.get("dim_feedforward", 256)
        self.dropout: float = tr_cfg.get("dropout", 0.1)
        self.batch_size: int = tr_cfg.get("batch_size", 64)
        self.lr: float = tr_cfg.get("learning_rate", 0.0001)
        self.epochs: int = tr_cfg.get("epochs", 100)
        self.patience: int = tr_cfg.get("patience", 10)
        self.network: TransformerPredictor | None = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _build_network(self, n_features: int) -> TransformerPredictor:
        """Instantiate and move the Transformer network to the device.

        Args:
            n_features: Number of input feature dimensions.

        Returns:
            Initialised ``TransformerPredictor`` on ``self.device``.
        """
        net = TransformerPredictor(
            n_features=n_features,
            d_model=self.d_model,
            nhead=self.nhead,
            num_encoder_layers=self.num_encoder_layers,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout,
        )
        return net.to(self.device)

    def _prepare_sequences(
        self,
        X: np.ndarray,
        y: np.ndarray | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Build sliding-window sequences from a flat feature matrix.

        Args:
            X: Feature matrix of shape (n_samples, n_features).
            y: Optional targets of shape (n_samples,).

        Returns:
            Tuple ``(X_seq, y_seq)`` where ``X_seq`` has shape
            ``(n_windows, seq_len, n_features)`` and ``y_seq`` has
            shape ``(n_windows,)`` (or ``None``).
        """
        X_arr = np.array(X, dtype=np.float32)
        n_windows = len(X_arr) - self.seq_len
        X_windows = np.stack([X_arr[i : i + self.seq_len] for i in range(n_windows)])
        X_tensor = torch.tensor(X_windows, dtype=torch.float32)

        y_tensor = None
        if y is not None:
            y_arr = np.array(y, dtype=np.float32)
            y_tensor = torch.tensor(y_arr[self.seq_len :], dtype=torch.float32)

        return X_tensor, y_tensor

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
    ) -> "TransformerModel":
        """Train the Transformer with Adam, cosine LR schedule, and early stopping.

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

        n_features = X_train.shape[1]
        self.network = self._build_network(n_features)
        optimizer = torch.optim.Adam(self.network.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs)
        criterion = nn.MSELoss()

        X_seq, y_seq = self._prepare_sequences(X_train, y_train)
        train_loader = DataLoader(
            TensorDataset(X_seq, y_seq),
            batch_size=self.batch_size,
            shuffle=True,
        )

        val_loader: DataLoader | None = None
        if X_val is not None and y_val is not None:
            X_val_seq, y_val_seq = self._prepare_sequences(X_val, y_val)
            val_loader = DataLoader(
                TensorDataset(X_val_seq, y_val_seq),
                batch_size=self.batch_size,
                shuffle=False,
            )

        best_val_loss = float("inf")
        no_improve = 0
        best_state: dict[str, Any] = {}

        with mlflow.start_run(nested=True):
            mlflow.log_params(
                {
                    "d_model": self.d_model,
                    "nhead": self.nhead,
                    "num_encoder_layers": self.num_encoder_layers,
                    "dim_feedforward": self.dim_feedforward,
                    "dropout": self.dropout,
                    "sequence_length": self.seq_len,
                    "learning_rate": self.lr,
                }
            )

            for epoch in range(1, self.epochs + 1):
                self.network.train()
                train_losses = []
                for xb, yb in train_loader:
                    xb, yb = xb.to(self.device), yb.to(self.device)
                    optimizer.zero_grad()
                    preds = self.network(xb)
                    loss = criterion(preds, yb)
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=1.0)
                    optimizer.step()
                    train_losses.append(loss.item())

                scheduler.step()
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
        """Compute mean loss over a DataLoader without gradient tracking.

        Args:
            loader: DataLoader to evaluate on.
            criterion: Loss function.

        Returns:
            Mean loss over all batches.
        """
        self.network.eval()
        losses = []
        with torch.no_grad():
            for xb, yb in loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                losses.append(criterion(self.network(xb), yb).item())
        return float(np.mean(losses))

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Run inference on a feature matrix.

        Args:
            X: Feature matrix of shape (n_samples, n_features).

        Returns:
            Predictions array of shape (n_samples - seq_len,).
        """
        if isinstance(X, pd.DataFrame):
            X = X.values

        X_seq, _ = self._prepare_sequences(X)
        loader = DataLoader(TensorDataset(X_seq), batch_size=self.batch_size, shuffle=False)

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
