from __future__ import annotations

import math
from collections import Counter
from itertools import combinations

from scans.data.hypergraph import Hyperedge


class HittingRiskIndex:
    """Bounded truncated hitting probability on the clique expansion."""

    def __init__(
        self,
        positive_edges: set[Hyperedge],
        max_neighbors_per_node: int = 128,
        max_pairs_per_edge: int = 2000,
        two_step_weight: float = 0.5,
    ) -> None:
        self.two_step_weight = max(0.0, float(two_step_weight))
        adjacency: dict[int, Counter[int]] = {}
        for edge in positive_edges:
            if len(edge) < 2:
                continue
            weight = 1.0 / max(1, len(edge) - 1)
            for node_a, node_b in _bounded_pairs(edge, max_pairs_per_edge):
                adjacency.setdefault(node_a, Counter())[node_b] += weight
                adjacency.setdefault(node_b, Counter())[node_a] += weight

        self.transitions: dict[int, dict[int, float]] = {}
        for node, counts in adjacency.items():
            top_items = counts.most_common(max_neighbors_per_node)
            total = float(sum(value for _, value in top_items))
            if total <= 0.0:
                continue
            self.transitions[node] = {neighbor: float(value) / total for neighbor, value in top_items}

    def risk(self, edge: Hyperedge) -> float:
        if len(edge) < 2:
            return 0.0
        scores = [self._pair_hitting(node_a, node_b) for node_a, node_b in combinations(edge, 2)]
        return sum(scores) / len(scores)

    def _pair_hitting(self, node_a: int, node_b: int) -> float:
        forward = self._directed_hitting(node_a, node_b)
        backward = self._directed_hitting(node_b, node_a)
        return 0.5 * (forward + backward)

    def _directed_hitting(self, source: int, target: int) -> float:
        first_step = self.transitions.get(source)
        if not first_step:
            return 0.0
        score = first_step.get(target, 0.0)
        if self.two_step_weight <= 0.0:
            return float(score)
        two_step = 0.0
        for middle, source_prob in first_step.items():
            middle_step = self.transitions.get(middle)
            if middle_step:
                two_step += source_prob * middle_step.get(target, 0.0)
        return float(score + self.two_step_weight * two_step)


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
