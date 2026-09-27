from __future__ import annotations

import random
from collections.abc import Mapping

from scans.data.hypergraph import HypergraphDataset, Hyperedge, normalize_hyperedge


def generate_synthetic_hypergraph(config: Mapping[str, object], seed: int) -> HypergraphDataset:
    rng = random.Random(seed)
    num_nodes = int(config["num_nodes"])
    num_hyperedges = int(config["num_hyperedges"])
    num_communities = int(config["num_communities"])
    min_edge_size = int(config["min_edge_size"])
    max_edge_size = int(config["max_edge_size"])
    intra_probability = float(config["intra_community_probability"])

    if min_edge_size < 2:
        raise ValueError("min_edge_size must be at least 2")
    if max_edge_size > num_nodes:
        raise ValueError("max_edge_size cannot exceed num_nodes")
    if num_communities <= 0:
        raise ValueError("num_communities must be positive")

    communities = _make_communities(num_nodes, num_communities)
    edges: set[Hyperedge] = set()
    attempts = 0
    max_attempts = num_hyperedges * 50

    while len(edges) < num_hyperedges and attempts < max_attempts:
        attempts += 1
        edge_size = rng.randint(min_edge_size, max_edge_size)
        base_community = rng.randrange(num_communities)
        edge_nodes: set[int] = set()

        while len(edge_nodes) < edge_size:
            if rng.random() < intra_probability:
                pool = communities[base_community]
            else:
                pool = communities[rng.randrange(num_communities)]
            edge_nodes.add(rng.choice(pool))

        edges.add(normalize_hyperedge(edge_nodes))

    if len(edges) < num_hyperedges:
        raise RuntimeError("failed to generate enough unique synthetic hyperedges")

    edge_list = list(edges)
    rng.shuffle(edge_list)
    split_config = config["split"]
    train_ratio = float(split_config["train"])
    val_ratio = float(split_config["val"])
    train_end = int(len(edge_list) * train_ratio)
    val_end = train_end + int(len(edge_list) * val_ratio)

    dataset = HypergraphDataset(
        num_nodes=num_nodes,
        train_edges=edge_list[:train_end],
        val_edges=edge_list[train_end:val_end],
        test_edges=edge_list[val_end:],
    )
    dataset.validate()
    return dataset


def _make_communities(num_nodes: int, num_communities: int) -> list[list[int]]:
    communities = [[] for _ in range(num_communities)]
    for node_id in range(num_nodes):
        communities[node_id % num_communities].append(node_id)
    return communities
