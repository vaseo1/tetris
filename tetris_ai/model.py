from __future__ import annotations

from .features import FEATURE_SIZE


def require_torch():
    try:
        import torch
        import torch.nn as nn
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "PyTorch is required for training/export. Install it with:\n"
            "  python3 -m pip install torch tensorboard\n"
            "On Apple silicon, PyTorch will use the MPS backend when available."
        ) from exc
    return torch, nn


def best_device(torch):
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def make_value_net(input_size: int = FEATURE_SIZE):
    torch, nn = require_torch()

    class ValueNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.Sequential(
                nn.Linear(input_size, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, 1),
            )

        def forward(self, x):
            return self.layers(x).squeeze(-1)

    return ValueNet()
