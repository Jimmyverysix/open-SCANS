from __future__ import annotations

import pickle
from pathlib import Path
from typing import Iterable

from scans.data.hypergraph import Hyperedge, HypergraphDataset


def load_protocol_batches(
    path: str | Path,
    dataset: HypergraphDataset,
    protocols: Iterable[str],
    *,
    seed: int,
    source_edges: list[Hyperedge],
    negatives_per_positive: int = 1,
    split: str,
) -> dict[str, dict[str, object]]:
    bank_path = Path(path)
    with bank_path.open("rb") as handle:
        payload = pickle.load(handle)
    if payload.get("format") != "recipe_protocol_bank_v1":
        raise ValueError(f"unsupported protocol bank format: {bank_path}")
    if int(payload.get("seed", -1)) != int(seed):
        raise ValueError(f"protocol bank seed mismatch: expected {seed}, got {payload.get('seed')}")
    if payload.get("split", "validation") != split:
        raise ValueError(f"protocol bank split mismatch: expected {split}, got {payload.get('split')}")
    if int(payload.get("num_nodes", -1)) != int(dataset.num_nodes):
        raise ValueError("protocol bank num_nodes mismatch")
    if int(payload.get("positive_count", -1)) != len(dataset.positive_edges):
        raise ValueError("protocol bank positive-edge count mismatch")
    if int(payload.get("negatives_per_positive", -1)) != int(negatives_per_positive):
        raise ValueError("protocol bank negatives-per-positive mismatch")
    stored_sources = [tuple(int(node) for node in edge) for edge in payload.get("source_edges", [])]
    if stored_sources != source_edges:
        raise ValueError(f"protocol bank source edges do not match the current {split} split")

    stored_protocols = payload.get("protocols", {})
    expected_sizes = [len(source) for source in source_edges for _ in range(int(negatives_per_positive))]
    batches: dict[str, dict[str, object]] = {}
    for protocol in protocols:
        if protocol not in stored_protocols:
            raise ValueError(f"protocol bank is missing protocol {protocol}")
        record = stored_protocols[protocol]
        negative_edges = [tuple(int(node) for node in edge) for edge in record["edges"]]
        if len(negative_edges) != len(expected_sizes):
            raise ValueError(f"protocol bank has an invalid {protocol} edge count")
        if any(len(edge) != size for edge, size in zip(negative_edges, expected_sizes)):
            raise ValueError(f"protocol bank changes hyperedge cardinality for {protocol}")
        if any(edge in dataset.positive_edges for edge in negative_edges):
            raise ValueError(f"protocol bank contains a known positive for {protocol}")
        batches[protocol] = {
            "edges": source_edges + negative_edges,
            "labels": [1] * len(source_edges) + [0] * len(negative_edges),
            "metadata": dict(record.get("metadata", {})),
        }
    return batches
