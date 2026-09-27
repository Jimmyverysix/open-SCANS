from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import nn


def build_predictor_optimizer(
    parameters: Iterable[nn.Parameter],
    *,
    name: str,
    learning_rate: float,
    weight_decay: float,
) -> torch.optim.Optimizer:
    normalized = str(name).lower()
    if normalized == "rmsprop":
        return torch.optim.RMSprop(
            parameters,
            lr=float(learning_rate),
            weight_decay=float(weight_decay),
        )
    if normalized == "adamw":
        return torch.optim.AdamW(
            parameters,
            lr=float(learning_rate),
            weight_decay=float(weight_decay),
        )
    raise ValueError(f"unknown predictor optimizer: {name}")
