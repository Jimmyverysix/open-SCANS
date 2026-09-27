from __future__ import annotations

import torch
from torch import nn


class DenseOrSparseLinear(nn.Module):
    """Linear projection that keeps sparse input features sparse until multiplication."""

    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.layout in {torch.sparse_coo, torch.sparse_csr}:
            output = torch.sparse.mm(inputs, self.linear.weight.T)
            if self.linear.bias is not None:
                output = output + self.linear.bias
            return output
        return self.linear(inputs)
