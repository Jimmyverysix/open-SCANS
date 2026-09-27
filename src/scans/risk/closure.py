from __future__ import annotations

import math
from collections import Counter
from functools import lru_cache
from itertools import combinations

from scans.data.hypergraph import Hyperedge


class ClosureRiskIndex:
    def __init__(self, positive_edges: set[Hyperedge], max_pairs_per_edge: int = 0) -> None:
        self.max_pairs_per_edge = max(0, int(max_pairs_per_edge))
        self.pair_counts: Counter[tuple[int, int]] = Counter()
        for edge in positive_edges:
            for index_a, index_b in _bounded_pair_indices(len(edge), self.max_pairs_per_edge):
                self.pair_counts[_pair_key(edge[index_a], edge[index_b])] += 1
        self.pair_scores = {
            pair: math.log1p(count)
            for pair, count in self.pair_counts.items()
        }

    def risk(self, edge: Hyperedge) -> float:
        if len(edge) < 2:
            return 0.0
        pair_indices = _bounded_pair_indices(len(edge), self.max_pairs_per_edge)
        total = 0.0
        pair_scores = self.pair_scores
        for index_a, index_b in pair_indices:
            total += pair_scores.get(_pair_key(edge[index_a], edge[index_b]), 0.0)
        return total / len(pair_indices)


def _pair_key(node_a: int, node_b: int) -> tuple[int, int]:
    return (node_a, node_b) if node_a <= node_b else (node_b, node_a)


def _bounded_pairs(edge: Hyperedge, max_pairs: int) -> list[tuple[int, int]]:
    """Return a deterministic structural sketch for very large hyperedges.

    Recipe hyperedges can contain almost two thousand nodes, so materializing
    every pair creates hundreds of millions of Python Counter entries.  Small
    edges retain the exact closure statistic; large edges use evenly spaced
    cyclic offsets with no duplicate pairs.
    """
    return [
        _pair_key(edge[index_a], edge[index_b])
        for index_a, index_b in _bounded_pair_indices(len(edge), max_pairs)
    ]


@lru_cache(maxsize=2048)
def _bounded_pair_indices(edge_size: int, max_pairs: int) -> tuple[tuple[int, int], ...]:
    pair_count = edge_size * (edge_size - 1) // 2
    if max_pairs <= 0 or pair_count <= max_pairs:
        return tuple(combinations(range(edge_size), 2))

    pairs: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    stride = max(1, edge_size // max(1, int(math.sqrt(max_pairs))))
    offset = 1
    while len(pairs) < max_pairs and offset < edge_size:
        for index_a in range(edge_size):
            index_b = (index_a + offset) % edge_size
            if index_a == index_b:
                continue
            pair = _pair_key(index_a, index_b)
            if pair not in seen:
                seen.add(pair)
                pairs.append(pair)
                if len(pairs) >= max_pairs:
                    break
        offset += stride
    return tuple(pairs)
