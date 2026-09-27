from __future__ import annotations

import math
import random
from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from scans.data.hypergraph import Hyperedge
from scans.samplers.base import NegativeSampleBatch, jaccard


@dataclass
class AnchoredSafeSampler:
    anchor_ratio: float = 0.25
    jaccard_upper_bound: float = 0.65
    nearest_positive_upper_bound: float = 0.25
    max_attempts: int = 300

    name: str = "anchored_safe"

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
        positive_index = _PositiveEdgeIndex(positive_edges)
        for source_edge in source_edges:
            for _ in range(negatives_per_positive):
                edge = self._sample_one(source_edge, num_nodes, positive_edges, positive_index, rng)
                negatives.append(edge)
                sources.append(source_edge)
        return NegativeSampleBatch(edges=negatives, source_edges=sources)

    def _sample_one(
        self,
        source_edge: Hyperedge,
        num_nodes: int,
        positive_edges: set[Hyperedge],
        positive_index: "_PositiveEdgeIndex",
        rng: random.Random,
    ) -> Hyperedge:
        if len(source_edge) < 2:
            raise ValueError("anchored safe sampling requires source hyperedges with size >= 2")

        anchor_size = self._anchor_size(len(source_edge))
        source_set = set(source_edge)
        replacement_size = len(source_edge) - anchor_size
        candidate_pool = [node for node in range(num_nodes) if node not in source_set]

        best_edge: Hyperedge | None = None
        best_nearest = float("inf")
        for _ in range(self.max_attempts):
            anchor_nodes = rng.sample(source_edge, anchor_size)
            replacements = rng.sample(candidate_pool, replacement_size)
            edge = tuple(sorted(set(anchor_nodes) | set(replacements)))
            if len(edge) != len(source_edge):
                continue
            if edge in positive_edges:
                continue
            if jaccard(edge, source_edge) > self.jaccard_upper_bound:
                continue

            nearest = positive_index.nearest_similarity(edge, excluded_edge=source_edge)
            if nearest < best_nearest:
                best_edge = edge
                best_nearest = nearest
            if nearest <= self.nearest_positive_upper_bound:
                return edge

        if best_edge is not None:
            return best_edge
        raise RuntimeError("failed to sample a nearest-risk-controlled anchored hyperedge")

    def _anchor_size(self, edge_size: int) -> int:
        raw_size = int(math.floor(edge_size * self.anchor_ratio))
        return min(edge_size - 1, max(1, raw_size))


class _PositiveEdgeIndex:
    def __init__(self, positive_edges: set[Hyperedge]) -> None:
        self.positive_edges = list(positive_edges)
        self.edge_sizes = [len(edge) for edge in self.positive_edges]
        self.edge_sizes_array = np.asarray(self.edge_sizes, dtype=np.int64)
        self.edge_ids = {edge: edge_id for edge_id, edge in enumerate(self.positive_edges)}
        self.by_node: dict[int, list[int]] = defaultdict(list)
        for edge_id, edge in enumerate(self.positive_edges):
            for node in edge:
                self.by_node[node].append(edge_id)
        self.by_node_arrays = {
            node: np.asarray(edge_ids, dtype=np.int32)
            for node, edge_ids in self.by_node.items()
        }

    def nearest_similarity(self, edge: Hyperedge, excluded_edge: Hyperedge | None = None) -> float:
        incidence_arrays = [
            self.by_node_arrays[node]
            for node in edge
            if node in self.by_node_arrays
        ]
        incidence_count = sum(array.size for array in incidence_arrays)
        if incidence_count >= 64:
            return self._nearest_similarity_numpy(edge, excluded_edge, incidence_arrays)

        overlap_counts: dict[int, int] = {}
        for node in edge:
            for edge_id in self.by_node.get(node, ()):
                overlap_counts[edge_id] = overlap_counts.get(edge_id, 0) + 1
        if not overlap_counts:
            return 0.0
        edge_size = len(edge)
        excluded_edge_id = self.edge_ids.get(excluded_edge) if excluded_edge is not None else None
        best = 0.0
        for edge_id, intersection_size in overlap_counts.items():
            if edge_id == excluded_edge_id:
                continue
            union_size = edge_size + self.edge_sizes[edge_id] - intersection_size
            if union_size > 0:
                best = max(best, intersection_size / union_size)
        return best

    def _nearest_similarity_numpy(
        self,
        edge: Hyperedge,
        excluded_edge: Hyperedge | None,
        incidence_arrays: list[np.ndarray],
    ) -> float:
        incident_edge_ids = (
            incidence_arrays[0]
            if len(incidence_arrays) == 1
            else np.concatenate(incidence_arrays)
        )
        edge_ids, intersection_sizes = np.unique(incident_edge_ids, return_counts=True)
        excluded_edge_id = self.edge_ids.get(excluded_edge) if excluded_edge is not None else None
        if excluded_edge_id is not None:
            keep = edge_ids != excluded_edge_id
            edge_ids = edge_ids[keep]
            intersection_sizes = intersection_sizes[keep]
        if edge_ids.size == 0:
            return 0.0
        union_sizes = len(edge) + self.edge_sizes_array[edge_ids] - intersection_sizes
        similarities = intersection_sizes / union_sizes
        return float(np.max(similarities, initial=0.0))
