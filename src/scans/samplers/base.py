from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Protocol

from scans.data.hypergraph import Hyperedge


@dataclass(frozen=True)
class NegativeSampleBatch:
    edges: list[Hyperedge]
    source_edges: list[Hyperedge]
    metadata: dict[str, float] = field(default_factory=dict)
    candidate_labels: list[str] = field(default_factory=list)


class NegativeSampler(Protocol):
    name: str

    def sample(
        self,
        source_edges: list[Hyperedge],
        num_nodes: int,
        positive_edges: set[Hyperedge],
        rng: random.Random,
        negatives_per_positive: int,
    ) -> NegativeSampleBatch:
        ...


def random_hyperedge(rng: random.Random, num_nodes: int, edge_size: int) -> Hyperedge:
    return tuple(sorted(rng.sample(range(num_nodes), edge_size)))


def jaccard(edge_a: Hyperedge, edge_b: Hyperedge) -> float:
    set_a = set(edge_a)
    set_b = set(edge_b)
    union_size = len(set_a | set_b)
    if union_size == 0:
        return 0.0
    return len(set_a & set_b) / union_size
