from __future__ import annotations

import math
from collections import Counter
from itertools import combinations

from scans.data.hypergraph import Hyperedge


class CoWalkRiskIndex:
    """Local truncated co-walk risk on the clique expansion of a hypergraph."""

    def __init__(
        self,
        positive_edges: set[Hyperedge],
        max_neighbors_per_node: int = 128,
        max_pairs_per_edge: int = 2000,
    ) -> None:
        self.max_neighbors_per_node = max_neighbors_per_node
        self.max_pairs_per_edge = max_pairs_per_edge
        adjacency: dict[int, Counter[int]] = {}

        for edge in positive_edges:
            if len(edge) < 2:
                continue
            weight = 1.0 / max(1, len(edge) - 1)
            for node_a, node_b in _bounded_pairs(edge, max_pairs_per_edge):
                adjacency.setdefault(node_a, Counter())[node_b] += weight
                adjacency.setdefault(node_b, Counter())[node_a] += weight

        self.neighbors: dict[int, dict[int, float]] = {}
        self.norms: dict[int, float] = {}
        for node, counts in adjacency.items():
            top_items = counts.most_common(max_neighbors_per_node)
            weights = {neighbor: float(value) for neighbor, value in top_items}
            self.neighbors[node] = weights
            self.norms[node] = math.sqrt(sum(value * value for value in weights.values()))

    def risk(self, edge: Hyperedge) -> float:
        if len(edge) < 2:
            return 0.0
        scores = [self._pair_risk(node_a, node_b) for node_a, node_b in combinations(edge, 2)]
        return sum(scores) / len(scores)

    def _pair_risk(self, node_a: int, node_b: int) -> float:
        neighbors_a = self.neighbors.get(node_a)
        neighbors_b = self.neighbors.get(node_b)
        if not neighbors_a or not neighbors_b:
            return 0.0

        norm = self.norms[node_a] * self.norms[node_b]
        if norm <= 0.0:
            return 0.0

        if len(neighbors_a) <= len(neighbors_b):
            small, large = neighbors_a, neighbors_b
        else:
            small, large = neighbors_b, neighbors_a

        dot = 0.0
        for shared_node, weight in small.items():
            dot += weight * large.get(shared_node, 0.0)
        return float(dot / norm)


def _bounded_pairs(edge: Hyperedge, max_pairs: int) -> list[tuple[int, int]]:
    pair_count = len(edge) * (len(edge) - 1) // 2
    if pair_count <= max_pairs:
        return list(combinations(edge, 2))

    nodes = tuple(edge)
    pairs: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    stride = max(1, len(nodes) // int(math.sqrt(max_pairs)))
    offset = 1
    while len(pairs) < max_pairs and offset < len(nodes):
        for index, node_a in enumerate(nodes):
            node_b = nodes[(index + offset) % len(nodes)]
            if node_a == node_b:
                continue
            pair = (node_a, node_b) if node_a < node_b else (node_b, node_a)
            if pair not in seen:
                seen.add(pair)
                pairs.append(pair)
                if len(pairs) >= max_pairs:
                    break
        offset += stride
    return pairs
