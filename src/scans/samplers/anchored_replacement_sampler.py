from __future__ import annotations

import math
import random
from dataclasses import dataclass

from scans.data.hypergraph import Hyperedge
from scans.samplers.base import NegativeSampleBatch, jaccard


@dataclass
class AnchoredReplacementSampler:
    anchor_ratio: float = 0.5
    jaccard_upper_bound: float = 0.75
    max_attempts: int = 100

    name: str = "anchored_random"

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
                edge = self._sample_one(source_edge, num_nodes, positive_edges, rng)
                negatives.append(edge)
                sources.append(source_edge)
        return NegativeSampleBatch(edges=negatives, source_edges=sources)

    def _sample_one(
        self,
        source_edge: Hyperedge,
        num_nodes: int,
        positive_edges: set[Hyperedge],
        rng: random.Random,
    ) -> Hyperedge:
        if len(source_edge) < 2:
            raise ValueError("anchored replacement requires source hyperedges with size >= 2")

        anchor_size = self._anchor_size(len(source_edge))
        anchor_nodes = tuple(sorted(rng.sample(source_edge, anchor_size)))
        source_set = set(source_edge)
        forbidden_replacements = source_set
        replacement_size = len(source_edge) - anchor_size

        for _ in range(self.max_attempts):
            candidate_pool = [node for node in range(num_nodes) if node not in forbidden_replacements]
            replacements = rng.sample(candidate_pool, replacement_size)
            edge = tuple(sorted(set(anchor_nodes) | set(replacements)))
            if len(edge) != len(source_edge):
                continue
            if edge in positive_edges:
                continue
            if jaccard(edge, source_edge) > self.jaccard_upper_bound:
                continue
            return edge

        raise RuntimeError("failed to sample an anchored replacement negative hyperedge")

    def _anchor_size(self, edge_size: int) -> int:
        raw_size = int(math.floor(edge_size * self.anchor_ratio))
        return min(edge_size - 1, max(1, raw_size))
