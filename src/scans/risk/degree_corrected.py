from __future__ import annotations

import math
from collections import Counter
from itertools import combinations

from scans.data.hypergraph import Hyperedge


class DegreeCorrectedResidualRiskIndex:
    """Positive-support residual risk under a degree-corrected hypergraph null model."""

    def __init__(self, positive_edges: set[Hyperedge], max_pairs_per_edge: int = 2000) -> None:
        self.node_degrees: Counter[int] = Counter()
        self.pair_counts: Counter[tuple[int, int]] = Counter()
        self.total_degree = 0.0
        self.pair_exposure = 0.0

        for edge in positive_edges:
            edge_size = len(edge)
            if edge_size == 0:
                continue
            self.total_degree += edge_size
            self.pair_exposure += edge_size * max(0, edge_size - 1)
            for node in edge:
                self.node_degrees[node] += 1
            if edge_size < 2:
                continue
            for node_a, node_b in _bounded_pairs(edge, max_pairs_per_edge):
                self.pair_counts[_pair_key(node_a, node_b)] += 1

    def risk(self, edge: Hyperedge) -> float:
        if len(edge) < 2:
            return 0.0
        scores = [self._pair_residual(node_a, node_b) for node_a, node_b in combinations(edge, 2)]
        return sum(scores) / len(scores)

    def _pair_residual(self, node_a: int, node_b: int) -> float:
        observed = float(self.pair_counts[_pair_key(node_a, node_b)])
        expected = self._expected_pair_count(node_a, node_b)
        if observed <= expected:
            return 0.0
        z_score = (observed - expected) / math.sqrt(expected + 1e-9)
        return float(math.log1p(z_score))

    def _expected_pair_count(self, node_a: int, node_b: int) -> float:
        if self.total_degree <= 1.0 or self.pair_exposure <= 0.0:
            return 0.0
        degree_a = float(self.node_degrees[node_a])
        degree_b = float(self.node_degrees[node_b])
        if degree_a <= 0.0 or degree_b <= 0.0:
            return 0.0
        return self.pair_exposure * degree_a * degree_b / (self.total_degree * (self.total_degree - 1.0))


def _pair_key(node_a: int, node_b: int) -> tuple[int, int]:
    return (node_a, node_b) if node_a <= node_b else (node_b, node_a)


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
            pair = _pair_key(node_a, node_b)
            if pair not in seen:
                seen.add(pair)
                pairs.append(pair)
                if len(pairs) >= max_pairs:
                    break
        offset += stride
    return pairs
