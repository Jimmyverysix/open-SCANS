from __future__ import annotations

import random
from dataclasses import dataclass

from scans.data.hypergraph import Hyperedge
from scans.samplers.base import NegativeSampleBatch, random_hyperedge


@dataclass
class RandomSampler:
    min_edge_size: int
    max_edge_size: int
    max_attempts: int = 100

    name: str = "random"

    def sample(
        self,
        source_edges: list[Hyperedge],
        num_nodes: int,
        positive_edges: set[Hyperedge],
        rng: random.Random,
        negatives_per_positive: int,
    ) -> NegativeSampleBatch:
        negatives: list[Hyperedge] = []
        sources: list[Hyperedge] = []
        for source_edge in source_edges:
            for _ in range(negatives_per_positive):
                edge = self._sample_one(num_nodes, positive_edges, rng)
                negatives.append(edge)
                sources.append(source_edge)
        return NegativeSampleBatch(edges=negatives, source_edges=sources)

    def _sample_one(
        self,
        num_nodes: int,
        positive_edges: set[Hyperedge],
        rng: random.Random,
    ) -> Hyperedge:
        for _ in range(self.max_attempts):
            edge_size = rng.randint(self.min_edge_size, self.max_edge_size)
            edge = random_hyperedge(rng, num_nodes, edge_size)
            if edge not in positive_edges:
                return edge
        raise RuntimeError("failed to sample a random negative hyperedge")
