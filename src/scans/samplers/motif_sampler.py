from __future__ import annotations

import random

from scans.data.hypergraph import Hyperedge
from scans.samplers.base import NegativeSampleBatch
from scans.samplers.frontier_accel import sample_motif_batch, sample_motif_one
from scans.samplers.size_matched_sampler import SizeMatchedSampler
from scans.samplers.two_section_index import LazyTwoSectionIndex, shared_two_section_index


class MotifNegativeSampler:
    """SEHP-style MNS baseline using bounded frontier expansion on the two-section graph."""

    name = "mns"

    def __init__(self, max_attempts: int = 100) -> None:
        self.max_attempts = int(max_attempts)
        self._cached_positive_edges: set[Hyperedge] | None = None
        self._cached_adjacency: LazyTwoSectionIndex | None = None
        self._cached_starts: list[int] | None = None

    def sample(
        self,
        source_edges: list[Hyperedge],
        num_nodes: int,
        positive_edges: set[Hyperedge],
        rng: random.Random,
        negatives_per_positive: int,
    ) -> NegativeSampleBatch:
        adjacency, starts = self._index(positive_edges)
        fallback = SizeMatchedSampler(max_attempts=self.max_attempts)
        batch_sizes = [
            len(source)
            for source in source_edges
            for _ in range(negatives_per_positive)
        ]
        native_batch = sample_motif_batch(
            batch_sizes,
            adjacency,
            starts,
            positive_edges,
            rng,
            self.max_attempts,
        )
        if native_batch is not NotImplemented:
            negatives = list(native_batch)
            sources = [
                source
                for source in source_edges
                for _ in range(negatives_per_positive)
            ]
            return NegativeSampleBatch(
                negatives,
                sources,
                {"mns_fallback_rate": 0.0},
            )

        negatives: list[Hyperedge] = []
        sources: list[Hyperedge] = []
        fallbacks = 0
        for source in source_edges:
            for _ in range(negatives_per_positive):
                negative = self._sample_one(len(source), adjacency, starts, positive_edges, rng)
                if negative is None:
                    fallbacks += 1
                    batch = fallback.sample([source], num_nodes, positive_edges, rng, 1)
                    negative = batch.edges[0]
                negatives.append(negative)
                sources.append(source)
        return NegativeSampleBatch(
            negatives,
            sources,
            {"mns_fallback_rate": fallbacks / max(1, len(negatives))},
        )

    def _sample_one(
        self,
        size: int,
        adjacency: LazyTwoSectionIndex,
        starts: list[int],
        positive_edges: set[Hyperedge],
        rng: random.Random,
    ) -> Hyperedge | None:
        return sample_motif_one(size, adjacency, starts, positive_edges, rng, self.max_attempts)

    def _index(self, positive_edges: set[Hyperedge]) -> tuple[LazyTwoSectionIndex, list[int]]:
        if self._cached_positive_edges is positive_edges:
            if self._cached_adjacency is None or self._cached_starts is None:
                raise RuntimeError("incomplete MNS adjacency cache")
            return self._cached_adjacency, self._cached_starts
        adjacency = shared_two_section_index(positive_edges)
        starts = adjacency.starts
        self._cached_positive_edges = positive_edges
        self._cached_adjacency = adjacency
        self._cached_starts = starts
        return adjacency, starts
