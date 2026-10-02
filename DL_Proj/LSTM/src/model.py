"""LSTM for RUL regression.

The window (batch, time, sensors) is read one cycle at a time by a stacked LSTM. Its
hidden state after the last cycle summarises the whole window and is mapped to a RUL value.
"""
import torch
from torch import nn


class LSTMRegressor(nn.Module):
    def __init__(self, n_features: int, hidden_size: int = 64, n_layers: int = 1,
                 dropout: float = 0.2, fc_units: int = 32):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden_size, num_layers=n_layers, batch_first=True,
                            dropout=dropout if n_layers > 1 else 0.0)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, fc_units),
            nn.ReLU(),
            nn.Linear(fc_units, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)              # (B, W, hidden)
        return self.head(out[:, -1]).squeeze(-1)  # hidden state at the last cycle


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
