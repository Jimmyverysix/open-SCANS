from __future__ import annotations

import math
import random
from dataclasses import dataclass

from scans.data.hypergraph import Hyperedge
from scans.samplers.base import NegativeSampleBatch, jaccard, random_hyperedge


@dataclass
class FastSafeSampler:
    anchor_ratio: float = 0.5
    source_jaccard_upper_bound: float = 0.85
    max_attempts: int = 100

    name: str = "fast_safe"

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
        source_jaccards: list[float] = []
        fallbacks = 0
        for source_edge in source_edges:
            for _ in range(negatives_per_positive):
                edge, fallback = self._sample_one(source_edge, num_nodes, positive_edges, rng)
                negatives.append(edge)
                sources.append(source_edge)
                source_jaccards.append(jaccard(edge, source_edge))
                fallbacks += int(fallback)
        metadata = {
            "fast_safe_enabled": 1.0,
            "fast_safe_anchor_ratio": float(self.anchor_ratio),
            "fast_safe_source_jaccard_upper_bound": float(self.source_jaccard_upper_bound),
            "fast_safe_source_jaccard_mean": _mean(source_jaccards),
            "fast_safe_fallback_rate": fallbacks / len(negatives) if negatives else 0.0,
        }
        return NegativeSampleBatch(edges=negatives, source_edges=sources, metadata=metadata)

    def _sample_one(
        self,
        source_edge: Hyperedge,
        num_nodes: int,
        positive_edges: set[Hyperedge],
        rng: random.Random,
    ) -> tuple[Hyperedge, bool]:
        edge_size = len(source_edge)
        if edge_size <= 0:
            raise ValueError("source hyperedges must be non-empty")
        if self.anchor_ratio <= 0.0 or edge_size == 1:
            for _ in range(self.max_attempts):
                edge = random_hyperedge(rng, num_nodes, edge_size)
                if edge not in positive_edges and jaccard(edge, source_edge) <= self.source_jaccard_upper_bound:
                    return edge, False
            return self._fallback(edge_size, num_nodes, positive_edges, rng, source_edge), True

        anchor_size = self._anchor_size(edge_size)
        source_set = set(source_edge)
        replacement_size = edge_size - anchor_size
        for _ in range(self.max_attempts):
            anchor_nodes = set(rng.sample(source_edge, anchor_size))
            replacement_nodes: set[int] = set()
            draws = 0
            while len(replacement_nodes) < replacement_size and draws < self.max_attempts * max(2, replacement_size):
                node = rng.randrange(num_nodes)
                draws += 1
                if node in source_set or node in replacement_nodes:
                    continue
                replacement_nodes.add(node)
            if len(replacement_nodes) != replacement_size:
                continue
            edge = tuple(sorted(anchor_nodes | replacement_nodes))
            if edge in positive_edges:
                continue
            if jaccard(edge, source_edge) > self.source_jaccard_upper_bound:
                continue
            return edge, False
        return self._fallback(edge_size, num_nodes, positive_edges, rng, source_edge), True

    def _fallback(
        self,
        edge_size: int,
        num_nodes: int,
        positive_edges: set[Hyperedge],
        rng: random.Random,
        source_edge: Hyperedge | None = None,
    ) -> Hyperedge:
        if source_edge is not None:
            for _ in range(self.max_attempts):
                edge = random_hyperedge(rng, num_nodes, edge_size)
                if edge not in positive_edges and jaccard(edge, source_edge) <= self.source_jaccard_upper_bound:
                    return edge
        for _ in range(self.max_attempts):
            edge = random_hyperedge(rng, num_nodes, edge_size)
            if edge not in positive_edges:
                return edge
        raise RuntimeError("failed to sample a fast safe negative hyperedge")

    def _anchor_size(self, edge_size: int) -> int:
        raw_size = int(math.floor(edge_size * self.anchor_ratio))
        return min(edge_size - 1, max(1, raw_size))


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))
