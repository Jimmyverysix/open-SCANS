from __future__ import annotations

from array import array
from collections import defaultdict

from scans.data.hypergraph import Hyperedge


class LazyTwoSectionIndex:
    """Exact two-section neighbors backed by the sparse hypergraph incidence list."""

    def __init__(
        self,
        positive_edges: set[Hyperedge],
        max_cached_memberships: int = 2_000_000,
        max_cached_sequence_memberships: int = 128_000_000,
    ) -> None:
        self.positive_edges = list(positive_edges)
        incident_edge_ids: dict[int, list[int]] = defaultdict(list)
        starts: set[int] = set()
        for edge_id, edge in enumerate(self.positive_edges):
            for node in edge:
                incident_edge_ids[node].append(edge_id)
            if len(edge) > 1:
                starts.update(edge)
        self.incident_edge_ids = {
            node: array("I", edge_ids)
            for node, edge_ids in incident_edge_ids.items()
        }
        self.starts = sorted(starts)
        self.max_cached_memberships = max(0, int(max_cached_memberships))
        self._neighbor_cache: dict[int, set[int]] = {}
        self._cached_memberships = 0
        self.max_cached_sequence_memberships = max(0, int(max_cached_sequence_memberships))
        self._neighbor_sequence_cache: dict[int, array[int]] = {}
        self._frontier_sequence_cache: dict[int, array[int]] = {}
        self._cached_sequence_memberships = 0

    def neighbors(self, node: int) -> set[int]:
        cached = self._neighbor_cache.get(node)
        if cached is not None:
            return cached
        neighbors: set[int] = set()
        for edge_id in self.incident_edge_ids.get(node, ()):
            neighbors.update(self.positive_edges[edge_id])
        neighbors.discard(node)
        self._remember(node, neighbors)
        return neighbors

    def neighbor_sequence(self, node: int) -> array[int]:
        cached = self._neighbor_sequence_cache.get(node)
        if cached is not None:
            return cached
        sequence = array("I", self.neighbors(node))
        self._remember_sequence(self._neighbor_sequence_cache, node, sequence)
        return sequence

    def incident_sequence(self, node: int) -> array[int]:
        return self.incident_edge_ids.get(node, array("I"))

    def frontier_sequence(self, node: int) -> array[int]:
        cached = self._frontier_sequence_cache.get(node)
        if cached is not None:
            return cached
        sequence = array("I", set(self.neighbors(node)))
        self._remember_sequence(self._frontier_sequence_cache, node, sequence)
        return sequence

    def _remember(self, node: int, neighbors: set[int]) -> None:
        size = len(neighbors)
        if self.max_cached_memberships <= 0 or size > self.max_cached_memberships:
            return
        if self._cached_memberships + size > self.max_cached_memberships:
            return
        self._neighbor_cache[node] = neighbors
        self._cached_memberships += size

    def _remember_sequence(
        self,
        cache: dict[int, array[int]],
        node: int,
        sequence: array[int],
    ) -> None:
        size = len(sequence)
        if (
            self.max_cached_sequence_memberships <= 0
            or size > self.max_cached_sequence_memberships
            or self._cached_sequence_memberships + size > self.max_cached_sequence_memberships
        ):
            return
        cache[node] = sequence
        self._cached_sequence_memberships += size


_INDEX_CACHE: dict[int, tuple[set[Hyperedge], LazyTwoSectionIndex]] = {}


def shared_two_section_index(positive_edges: set[Hyperedge]) -> LazyTwoSectionIndex:
    key = id(positive_edges)
    cached = _INDEX_CACHE.get(key)
    if cached is not None and cached[0] is positive_edges:
        return cached[1]
    index = LazyTwoSectionIndex(positive_edges)
    _INDEX_CACHE[key] = (positive_edges, index)
    return index
