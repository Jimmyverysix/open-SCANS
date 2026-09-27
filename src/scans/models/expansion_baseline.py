from __future__ import annotations

import math
import random
from collections import Counter
from itertools import combinations

import numpy as np

from scans.data.hypergraph import Hyperedge


EXPANSION_FEATURES = ("gm", "hm", "am", "cn", "jc", "aa")


class ExpansionProjectionFeatures:
    """Second- and third-order projected-graph features from Expansion."""

    def __init__(self, train_edges: list[Hyperedge], num_nodes: int) -> None:
        self.train_edges = train_edges
        self.num_nodes = int(num_nodes)
        self.pair_support: Counter[tuple[int, int]] = Counter()
        self.pair_neighbors = [set() for _ in range(num_nodes)]
        for edge in train_edges:
            for low, high in combinations(edge, 2):
                pair = (low, high)
                self.pair_support[pair] += 1
                self.pair_neighbors[low].add(high)
                self.pair_neighbors[high].add(low)

        self.triple_support: Counter[tuple[int, int, int]] = Counter()
        self.third_neighbors: dict[tuple[int, int], set[tuple[int, int]]] = {}
        self.third_common_degrees: dict[tuple[int, int], int] = {}
        self._feature_cache: dict[Hyperedge, tuple[float, ...]] = {}
        self.last_star_sampling_stats: dict[str, float] = {}
        self._prepared = False

    def sample_star_negatives(
        self,
        source_edges: list[Hyperedge],
        positive_edges: set[Hyperedge],
        rng: random.Random,
        *,
        max_attempts_per_edge: int = 5000,
    ) -> list[Hyperedge]:
        """Sample one unique size-matched 2-PG star for every source edge."""

        eligible_by_size: dict[int, list[int]] = {}
        output: list[Hyperedge] = []
        seen: set[Hyperedge] = set()
        size_two_pool = [
            pair for pair in self.pair_support if pair not in positive_edges
        ]
        rng.shuffle(size_two_pool)
        size_two_index = 0
        for source in source_edges:
            size = len(source)
            if size == 2:
                if not size_two_pool:
                    raise RuntimeError(
                        "the 2-PG contains no size-2 edge outside the positive set"
                    )
                if size_two_index < len(size_two_pool):
                    sampled = size_two_pool[size_two_index]
                    size_two_index += 1
                else:
                    sampled = rng.choice(size_two_pool)
                seen.add(sampled)
                output.append(sampled)
                continue
            eligible = eligible_by_size.get(size)
            if eligible is None:
                eligible = [
                    node
                    for node, neighbors in enumerate(self.pair_neighbors)
                    if len(neighbors) >= size - 1
                ]
                eligible_by_size[size] = eligible
            if not eligible:
                raise RuntimeError(f"no 2-PG star center can support a size-{size} negative")

            sampled = None
            for _ in range(max_attempts_per_edge):
                center = rng.choice(eligible)
                leaves = rng.sample(tuple(self.pair_neighbors[center]), size - 1)
                candidate = tuple(sorted((center, *leaves)))
                if candidate not in positive_edges and candidate not in seen:
                    sampled = candidate
                    break
            if sampled is None:
                for _ in range(max_attempts_per_edge):
                    center = rng.choice(eligible)
                    leaves = rng.sample(tuple(self.pair_neighbors[center]), size - 1)
                    candidate = tuple(sorted((center, *leaves)))
                    if candidate not in positive_edges:
                        sampled = candidate
                        break
            if sampled is None:
                raise RuntimeError(
                    f"failed to sample a size-{size} 2-PG star outside the positive "
                    f"set after {2 * max_attempts_per_edge} attempts"
                )
            seen.add(sampled)
            output.append(sampled)
        unique_count = len(set(output))
        self.last_star_sampling_stats = {
            "star_negative_unique_count": float(unique_count),
            "star_negative_unique_rate": unique_count / max(1, len(output)),
            "star_negative_duplicate_count": float(len(output) - unique_count),
        }
        return output

    def prepare(self, candidate_edges: list[Hyperedge]) -> None:
        """Materialize only the third-order neighborhoods needed by candidates."""

        target_pairs: set[tuple[int, int]] = set()
        for edge in candidate_edges:
            endpoints = _projected_neighbor_endpoints(edge, 3)
            if endpoints is not None:
                target_pairs.update(endpoints)
        self.third_neighbors = {pair: set() for pair in target_pairs}

        for edge in self.train_edges:
            for triple in combinations(edge, 3):
                self.triple_support[triple] += 1
                for left, right in _third_projection_links(triple):
                    if left in self.third_neighbors:
                        self.third_neighbors[left].add(right)
                    if right in self.third_neighbors:
                        self.third_neighbors[right].add(left)

        common_nodes: set[tuple[int, int]] = set()
        for edge in candidate_edges:
            endpoints = _projected_neighbor_endpoints(edge, 3)
            if endpoints is None:
                continue
            common_nodes.update(
                self.third_neighbors.get(endpoints[0], set()).intersection(
                    self.third_neighbors.get(endpoints[1], set())
                )
            )

        common_neighbors = {pair: set() for pair in common_nodes}
        if common_neighbors:
            for edge in self.train_edges:
                for triple in combinations(edge, 3):
                    for left, right in _third_projection_links(triple):
                        if left in common_neighbors:
                            common_neighbors[left].add(right)
                        if right in common_neighbors:
                            common_neighbors[right].add(left)
        self.third_common_degrees = {
            pair: len(neighbors) for pair, neighbors in common_neighbors.items()
        }
        self._feature_cache.clear()
        self._prepared = True

    def transform(self, edges: list[Hyperedge]) -> dict[str, np.ndarray]:
        if not self._prepared:
            raise RuntimeError("prepare() must be called before transform()")
        matrices = {
            name: np.zeros((len(edges), 2), dtype=np.float64)
            for name in EXPANSION_FEATURES
        }
        for row, edge in enumerate(edges):
            values = self._feature_cache.get(edge)
            if values is None:
                values = self._edge_features(edge)
                self._feature_cache[edge] = values
            for index, name in enumerate(EXPANSION_FEATURES):
                matrices[name][row] = values[index * 2 : index * 2 + 2]
        return matrices

    def _edge_features(self, edge: Hyperedge) -> tuple[float, ...]:
        second_means = _means(
            [self.pair_support[pair] for pair in combinations(edge, 2)]
        )
        third_means = _means(
            [self.triple_support[triple] for triple in combinations(edge, 3)]
        )
        second_neighbors = _neighbor_features(
            edge,
            order=2,
            neighbors=lambda node: self.pair_neighbors[int(node)],
            degree=lambda node: len(self.pair_neighbors[int(node)]),
        )
        third_neighbors = _neighbor_features(
            edge,
            order=3,
            neighbors=lambda node: self.third_neighbors.get(node, set()),
            degree=lambda node: self.third_common_degrees.get(node, 0),
        )
        output = []
        for value2, value3 in zip(second_means, third_means):
            output.extend((value2, value3))
        for value2, value3 in zip(second_neighbors, third_neighbors):
            output.extend((value2, value3))
        return tuple(output)

    def stats(self) -> dict[str, int]:
        return {
            "pair_projection_edges": len(self.pair_support),
            "third_projection_hyperedges": len(self.triple_support),
            "third_target_nodes": len(self.third_neighbors),
            "third_common_nodes": len(self.third_common_degrees),
        }


def is_second_projection_star(
    edge: Hyperedge,
    pair_neighbors: list[set[int]],
) -> bool:
    nodes = set(edge)
    return any(nodes - {center} <= pair_neighbors[center] for center in edge)


def _means(weights: list[int]) -> tuple[float, float, float]:
    if not weights:
        return 0.0, 0.0, 0.0
    count = len(weights)
    arithmetic = sum(weights) / float(count)
    if any(weight == 0 for weight in weights):
        geometric = 0.0
    else:
        geometric = math.exp(sum(math.log(weight) for weight in weights) / count)
    inverse_sum = sum(1.0 / weight for weight in weights if weight != 0)
    harmonic = count / inverse_sum if inverse_sum else 0.0
    return geometric, harmonic, arithmetic


def _neighbor_features(
    edge: Hyperedge,
    *,
    order: int,
    neighbors,
    degree,
) -> tuple[float, float, float]:
    endpoints = _projected_neighbor_endpoints(edge, order)
    if endpoints is None:
        return 0.0, 0.0, 0.0
    left_neighbors = set(neighbors(endpoints[0]))
    right_neighbors = set(neighbors(endpoints[1]))
    common = left_neighbors.intersection(right_neighbors)
    union = left_neighbors.union(right_neighbors)
    cn = float(len(common))
    jaccard = cn / len(union) if union else 0.0
    adamic_adar = 0.0
    for node in common:
        node_degree = int(degree(node))
        if node_degree > 1:
            adamic_adar += 1.0 / math.log(node_degree)
    return cn, jaccard, adamic_adar


def _projected_neighbor_endpoints(
    edge: Hyperedge,
    order: int,
) -> tuple[int, int] | tuple[tuple[int, int], tuple[int, int]] | None:
    if order == 2:
        if len(edge) < 2:
            return None
        return edge[0], edge[-1]
    if order == 3:
        if len(edge) < 3:
            return None
        return (edge[0], edge[1]), (edge[-2], edge[-1])
    raise ValueError(f"unsupported projection order: {order}")


def _third_projection_links(
    triple: tuple[int, int, int],
) -> tuple[
    tuple[tuple[int, int], tuple[int, int]],
    tuple[tuple[int, int], tuple[int, int]],
    tuple[tuple[int, int], tuple[int, int]],
]:
    low, middle, high = triple
    first = (low, middle)
    second = (low, high)
    third = (middle, high)
    return (first, second), (first, third), (second, third)
