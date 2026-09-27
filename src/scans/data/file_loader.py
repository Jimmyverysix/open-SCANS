from __future__ import annotations

import importlib
import random
import pickle
import sys
import heapq
from collections import Counter
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import torch

from scans.data.hypergraph import HypergraphDataset, Hyperedge, normalize_hyperedge


def _install_numpy_pickle_aliases() -> None:
    # NumPy 2.x pickles may reference numpy._core; NumPy 1.x exposes numpy.core.
    try:
        importlib.import_module("numpy._core")
        importlib.import_module("numpy._core.numeric")
    except ModuleNotFoundError:
        sys.modules.setdefault("numpy._core", np.core)
        sys.modules.setdefault("numpy._core.numeric", np.core.numeric)


def load_hyperedges_from_txt(config: Mapping[str, object], seed: int) -> HypergraphDataset:
    path = Path(str(config["path"]))
    separator = str(config.get("separator", " "))
    split_config = config["split"]
    rng = random.Random(seed)

    edges = _read_edges(path, separator)
    num_nodes = max(max(edge) for edge in edges) + 1
    node_features = None
    if "features_path" in config:
        node_features = load_node_features(Path(str(config["features_path"])))

    train_ratio = float(split_config["train"])
    val_ratio = float(split_config["val"])
    train_end = int(len(edges) * train_ratio)
    val_end = train_end + int(len(edges) * val_ratio)
    split_strategy = str(split_config.get("strategy", "random"))
    if split_strategy == "random":
        rng.shuffle(edges)
        train_edges = edges[:train_end]
        val_edges = edges[train_end:val_end]
        test_edges = edges[val_end:]
    elif split_strategy == "cover_aware":
        train_edges, val_edges, test_edges = _cover_aware_split(
            edges,
            train_count=train_end,
            val_count=val_end - train_end,
            rng=rng,
        )
    else:
        raise ValueError(f"unknown split strategy: {split_strategy}")

    dataset = HypergraphDataset(
        num_nodes=num_nodes,
        train_edges=train_edges,
        val_edges=val_edges,
        test_edges=test_edges,
        node_features=node_features,
    )
    dataset.validate()
    return dataset


def _cover_aware_split(
    edges: list[Hyperedge],
    *,
    train_count: int,
    val_count: int,
    rng: random.Random,
) -> tuple[list[Hyperedge], list[Hyperedge], list[Hyperedge]]:
    if train_count <= 0 or train_count + val_count >= len(edges):
        raise ValueError("cover-aware split requires non-empty train, validation, and test sets")

    ordered_edges = sorted(edges)
    all_nodes = {node for edge in ordered_edges for node in edge}
    node_counts = Counter(node for edge in ordered_edges for node in edge)
    selected: set[int] = {
        index
        for index, edge in enumerate(ordered_edges)
        if any(node_counts[node] == 1 for node in edge)
    }
    if len(selected) > train_count:
        raise ValueError(
            f"rare-node coverage needs {len(selected)} mandatory edges, exceeding train budget {train_count}"
        )
    covered = {node for index in selected for node in ordered_edges[index]}
    uncovered = all_nodes - covered
    heap: list[tuple[int, int, Hyperedge, int]] = [
        (-len(edge), len(edge), edge, index)
        for index, edge in enumerate(ordered_edges)
        if index not in selected
    ]
    heapq.heapify(heap)
    while uncovered and len(selected) < train_count:
        if not heap:
            raise RuntimeError("set-cover heap was exhausted before all nodes were covered")
        negative_gain, edge_size, edge, index = heapq.heappop(heap)
        if index in selected:
            continue
        gain = sum(node in uncovered for node in edge)
        if gain != -negative_gain:
            heapq.heappush(heap, (-gain, edge_size, edge, index))
            continue
        if gain == 0:
            raise RuntimeError("set-cover construction cannot cover the remaining nodes")
        selected.add(index)
        uncovered.difference_update(edge)

    cover_edges = [ordered_edges[index] for index in sorted(selected)]
    remaining = [edge for index, edge in enumerate(ordered_edges) if index not in selected]
    rng.shuffle(remaining)
    fill_count = train_count - len(cover_edges)
    train_edges = cover_edges + remaining[:fill_count]
    rng.shuffle(train_edges)
    validation_start = fill_count
    validation_end = validation_start + val_count
    val_edges = remaining[validation_start:validation_end]
    test_edges = remaining[validation_end:]
    return train_edges, val_edges, test_edges


def _read_edges(path: Path, separator: str) -> list[Hyperedge]:
    if not path.exists():
        raise FileNotFoundError(path)

    edges: set[Hyperedge] = set()
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            if separator == "whitespace":
                parts = stripped.split()
            else:
                parts = stripped.split(separator)
            try:
                edge = normalize_hyperedge(int(part) for part in parts if part != "")
            except ValueError as error:
                raise ValueError(f"invalid hyperedge at {path}:{line_number}") from error
            if len(edge) < 2:
                raise ValueError(f"hyperedge must contain at least two nodes at {path}:{line_number}")
            edges.add(edge)

    if not edges:
        raise ValueError(f"no hyperedges found in {path}")
    return list(edges)


def load_node_features(path: Path) -> torch.Tensor:
    if not path.exists():
        raise FileNotFoundError(path)

    with path.open("rb") as file:
        _install_numpy_pickle_aliases()
        features = pickle.load(file)
    if hasattr(features, "tocoo"):
        sparse = features.tocsr().astype(np.float32)
        row_sums = np.asarray(sparse.sum(axis=1)).reshape(-1)
        inverse = np.zeros_like(row_sums)
        nonzero = row_sums > 0
        inverse[nonzero] = 1.0 / row_sums[nonzero]
        sparse = sparse.multiply(inverse[:, None]).tocoo()
        indices = torch.from_numpy(np.vstack([sparse.row, sparse.col]).astype(np.int64, copy=False))
        values = torch.from_numpy(sparse.data.astype(np.float32, copy=False))
        return torch.sparse_coo_tensor(indices, values, sparse.shape).coalesce()
    array = np.asarray(features).astype(np.float32, copy=False)
    row_sums = array.sum(axis=1, keepdims=True)
    nonzero_rows = row_sums.squeeze(-1) > 0
    array[nonzero_rows] = array[nonzero_rows] / row_sums[nonzero_rows]
    return torch.from_numpy(array)
