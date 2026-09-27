from __future__ import annotations

import random
from collections.abc import Iterator

from scans.data.hypergraph import Hyperedge


def shuffled_index_batches(
    count: int,
    batch_size: int,
    rng: random.Random,
) -> Iterator[list[int]]:
    if count < 0:
        raise ValueError("count must be non-negative")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    indices = list(range(count))
    rng.shuffle(indices)
    for start in range(0, count, batch_size):
        yield indices[start : start + batch_size]


def source_aligned_index_batches(
    positive_edges: list[Hyperedge],
    negative_edges: list[Hyperedge],
    negative_source_edges: list[Hyperedge],
    batch_size: int,
    rng: random.Random,
) -> Iterator[tuple[list[int], list[int]]]:
    """Batch positives while retaining every negative attached to each source."""
    if len(negative_edges) != len(negative_source_edges):
        raise ValueError("negative edges and source edges must have equal length")
    if len(set(positive_edges)) != len(positive_edges):
        raise ValueError("source-aligned mini-batching requires unique positive hyperedges")

    source_to_negative_indices: dict[Hyperedge, list[int]] = {
        edge: [] for edge in positive_edges
    }
    for index, source in enumerate(negative_source_edges):
        if source not in source_to_negative_indices:
            raise ValueError("negative batch contains a source outside the positive batch")
        source_to_negative_indices[source].append(index)

    missing = [edge for edge, indices in source_to_negative_indices.items() if not indices]
    if missing:
        raise ValueError(f"negative batch is missing {len(missing)} positive sources")

    for positive_indices in shuffled_index_batches(len(positive_edges), batch_size, rng):
        negative_indices = [
            negative_index
            for positive_index in positive_indices
            for negative_index in source_to_negative_indices[positive_edges[positive_index]]
        ]
        yield positive_indices, negative_indices
