from __future__ import annotations

import random
from dataclasses import dataclass

from scans.data.hypergraph import Hyperedge
from scans.samplers.base import NegativeSampleBatch, random_hyperedge


@dataclass
class SizeMatchedSampler:
    max_attempts: int = 100

    name: str = "size_matched"

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
                edge = self._sample_one(len(source_edge), num_nodes, positive_edges, rng)
                negatives.append(edge)
                sources.append(source_edge)
        return NegativeSampleBatch(edges=negatives, source_edges=sources)

    def _sample_one(
        self,
        edge_size: int,
        num_nodes: int,
        positive_edges: set[Hyperedge],
        rng: random.Random,
    ) -> Hyperedge:
        for _ in range(self.max_attempts):
            edge = random_hyperedge(rng, num_nodes, edge_size)
            if edge not in positive_edges:
                return edge
        raise RuntimeError("failed to sample a size-matched negative hyperedge")
