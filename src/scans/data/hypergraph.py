from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import Iterable

import torch

Hyperedge = tuple[int, ...]


def normalize_hyperedge(nodes: Iterable[int]) -> Hyperedge:
    edge = tuple(sorted(set(int(node) for node in nodes)))
    if not edge:
        raise ValueError("hyperedge must contain at least one node")
    return edge


@dataclass(frozen=True)
class HypergraphDataset:
    num_nodes: int
    train_edges: list[Hyperedge]
    val_edges: list[Hyperedge]
    test_edges: list[Hyperedge]
    node_features: torch.Tensor | None = None

    @cached_property
    def positive_edges(self) -> set[Hyperedge]:
        return set(self.train_edges) | set(self.val_edges) | set(self.test_edges)

    @cached_property
    def train_positive_edges(self) -> set[Hyperedge]:
        return set(self.train_edges)

    @cached_property
    def future_positive_edges(self) -> set[Hyperedge]:
        return set(self.val_edges) | set(self.test_edges)

    def validate(self) -> None:
        if self.num_nodes <= 0:
            raise ValueError("num_nodes must be positive")

        seen: set[Hyperedge] = set()
        for split_name, edges in (
            ("train", self.train_edges),
            ("val", self.val_edges),
            ("test", self.test_edges),
        ):
            if not edges:
                raise ValueError(f"{split_name} split is empty")
            for edge in edges:
                if len(edge) == 0:
                    raise ValueError(f"{split_name} contains an empty hyperedge")
                if len(edge) != len(set(edge)):
                    raise ValueError(f"{split_name} contains duplicate nodes: {edge}")
                if min(edge) < 0 or max(edge) >= self.num_nodes:
                    raise ValueError(f"{split_name} contains node outside [0, num_nodes): {edge}")
                if edge in seen:
                    raise ValueError(f"hyperedge appears in multiple splits: {edge}")
                seen.add(edge)

        if self.node_features is not None:
            if self.node_features.ndim != 2:
                raise ValueError("node_features must be a 2D tensor")
            if self.node_features.shape[0] < self.num_nodes:
                raise ValueError("node_features must contain at least num_nodes rows")
