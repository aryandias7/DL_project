"""1D-CNN for RUL regression.

Input (batch, window, sensors) is treated as a multichannel 1D signal: each sensor is a
channel and convolutions slide along the time axis, learning local degradation patterns.
"""
import torch
from torch import nn


class CNN1D(nn.Module):
    def __init__(self, n_features: int, window: int, n_filters: int = 32, kernel_size: int = 5,
                 n_layers: int = 3, dropout: float = 0.2, fc_units: int = 64,
                 pooling: str = "flatten"):
        super().__init__()
        layers, in_ch = [], n_features
        for _ in range(n_layers):
            layers += [nn.Conv1d(in_ch, n_filters, kernel_size, padding="same"),
                       nn.BatchNorm1d(n_filters),
                       nn.ReLU()]
            in_ch = n_filters
        self.features = nn.Sequential(*layers)

        if pooling == "gap":       # global average pooling over time
            self.pool = nn.AdaptiveAvgPool1d(1)
            flat = n_filters
        elif pooling == "flatten":  # keep temporal position information
            self.pool = nn.Identity()
            flat = n_filters * window
        else:
            raise ValueError(f"unknown pooling {pooling!r}")

        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(flat, fc_units),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fc_units, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)  # (B, W, F) -> (B, F, W)
        return self.head(self.pool(self.features(x))).squeeze(-1)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
