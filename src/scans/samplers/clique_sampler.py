from __future__ import annotations

import random
from collections import defaultdict

from scans.data.hypergraph import Hyperedge
from scans.samplers.base import NegativeSampleBatch
from scans.samplers.size_matched_sampler import SizeMatchedSampler
from scans.samplers.two_section_index import LazyTwoSectionIndex, shared_two_section_index


class CliqueNegativeSampler:
    """SEHP-style CNS baseline using common-neighbor replacement."""

    name = "cns"

    def __init__(self, max_attempts: int = 100) -> None:
        self.max_attempts = int(max_attempts)
        self._cached_positive_edges: set[Hyperedge] | None = None
        self._cached_adjacency: LazyTwoSectionIndex | None = None
        self._candidate_cache_adjacency: LazyTwoSectionIndex | None = None
        self._cached_candidate_pools: dict[
            Hyperedge,
            tuple[list[int], dict[int, list[int]], list[int]],
        ] = {}

    def sample(
        self,
        source_edges: list[Hyperedge],
        num_nodes: int,
        positive_edges: set[Hyperedge],
        rng: random.Random,
        negatives_per_positive: int,
    ) -> NegativeSampleBatch:
        adjacency = self._index(positive_edges)
        fallback = SizeMatchedSampler(max_attempts=self.max_attempts)
        negatives: list[Hyperedge] = []
        sources: list[Hyperedge] = []
        fallbacks = 0
        for source in source_edges:
            for _ in range(negatives_per_positive):
                negative = self._sample_one(source, adjacency, positive_edges, rng)
                if negative is None:
                    fallbacks += 1
                    negative = fallback.sample([source], num_nodes, positive_edges, rng, 1).edges[0]
                negatives.append(negative)
                sources.append(source)
        return NegativeSampleBatch(
            negatives,
            sources,
            {"cns_fallback_rate": fallbacks / max(1, len(negatives))},
        )

    def _sample_one(
        self,
        source: Hyperedge,
        adjacency: LazyTwoSectionIndex,
        positive_edges: set[Hyperedge],
        rng: random.Random,
    ) -> Hyperedge | None:
        universal, missing_by_removed, feasible_removed = self._candidate_pool(source, adjacency)
        source_set = set(source)
        for _ in range(min(self.max_attempts, max(1, len(feasible_removed)))):
            if not feasible_removed:
                break
            removed = rng.choice(feasible_removed)
            candidates = universal + missing_by_removed.get(removed, [])
            edge = tuple(sorted((source_set - {removed}) | {rng.choice(candidates)}))
            if edge not in positive_edges:
                return edge
        return None

    def _candidate_pool(
        self,
        source: Hyperedge,
        adjacency: LazyTwoSectionIndex,
    ) -> tuple[list[int], dict[int, list[int]], list[int]]:
        if self._candidate_cache_adjacency is not adjacency:
            self._candidate_cache_adjacency = adjacency
            self._cached_candidate_pools.clear()
        cached = self._cached_candidate_pools.get(source)
        if cached is not None:
            return cached

        if len(source) >= 4:
            candidate_pool = self._candidate_pool_from_pivots(source, adjacency)
        else:
            candidate_pool = self._candidate_pool_reference(source, adjacency)
        self._cached_candidate_pools[source] = candidate_pool
        return candidate_pool

    @staticmethod
    def _candidate_pool_from_pivots(
        source: Hyperedge,
        adjacency: LazyTwoSectionIndex,
    ) -> tuple[list[int], dict[int, list[int]], list[int]]:
        source_set = set(source)
        pivot_nodes = source[:4]
        pivot_neighbors = [adjacency.neighbors(node) for node in pivot_nodes]
        pivot_counts: dict[int, int] = defaultdict(int)
        candidate_order: list[int] = []
        seen: set[int] = set()
        for neighbors in pivot_neighbors:
            for candidate in neighbors:
                if candidate in source_set:
                    continue
                pivot_counts[candidate] += 1
                if candidate not in seen:
                    seen.add(candidate)
                    candidate_order.append(candidate)

        active = [candidate for candidate in candidate_order if pivot_counts[candidate] >= 3]
        missing_count: dict[int, int] = {}
        missing_node: dict[int, int] = {}
        for candidate in active:
            missing_count[candidate] = 4 - pivot_counts[candidate]
            if missing_count[candidate] == 1:
                missing_node[candidate] = next(
                    node for node, neighbors in zip(pivot_nodes, pivot_neighbors) if candidate not in neighbors
                )

        for node in source[4:]:
            neighbors = adjacency.neighbors(node)
            next_active: list[int] = []
            for candidate in active:
                if candidate not in neighbors:
                    missing_count[candidate] += 1
                    if missing_count[candidate] == 1:
                        missing_node[candidate] = node
                if missing_count[candidate] <= 1:
                    next_active.append(candidate)
            active = next_active
            if not active:
                break

        active_set = set(active)
        universal: list[int] = []
        missing_by_removed: dict[int, list[int]] = defaultdict(list)
        for candidate in candidate_order:
            if candidate not in active_set:
                continue
            if missing_count[candidate] == 0:
                universal.append(candidate)
            else:
                missing_by_removed[missing_node[candidate]].append(candidate)
        feasible_removed = [removed for removed in source if universal or missing_by_removed.get(removed)]
        return universal, dict(missing_by_removed), feasible_removed

    @staticmethod
    def _candidate_pool_reference(
        source: Hyperedge,
        adjacency: LazyTwoSectionIndex,
    ) -> tuple[list[int], dict[int, list[int]], list[int]]:

        source_set = set(source)
        required = max(0, len(source) - 1)
        counts: dict[int, int] = defaultdict(int)
        adjacent_xor: dict[int, int] = defaultdict(int)
        source_xor = 0
        for node in source:
            source_xor ^= node
            for candidate in adjacency.neighbors(node):
                if candidate in source_set:
                    continue
                counts[candidate] += 1
                adjacent_xor[candidate] ^= node

        universal: list[int] = []
        missing_by_removed: dict[int, list[int]] = defaultdict(list)
        for candidate, count in counts.items():
            if count == len(source):
                universal.append(candidate)
            elif count == required:
                missing_by_removed[source_xor ^ adjacent_xor[candidate]].append(candidate)

        feasible_removed = [removed for removed in source if universal or missing_by_removed.get(removed)]
        return universal, dict(missing_by_removed), feasible_removed

    def _index(self, positive_edges: set[Hyperedge]) -> LazyTwoSectionIndex:
        if self._cached_positive_edges is positive_edges:
            if self._cached_adjacency is None:
                raise RuntimeError("incomplete CNS adjacency cache")
            return self._cached_adjacency
        adjacency = shared_two_section_index(positive_edges)
        self._cached_positive_edges = positive_edges
        self._cached_adjacency = adjacency
        return adjacency
