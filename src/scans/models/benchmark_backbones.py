from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from scans.data.hypergraph import Hyperedge
from scans.models.feature_encoder import DenseOrSparseLinear
from scans.models.hyperedge_predictor import _pool_stat_edge_chunk


BENCHMARK_BACKBONES = (
    "hypergcn",
    "unigcnii",
    "hnhn",
    "allset_deepsets",
    "edhnn",
    "hds",
    "edgnn",
    "hypersagnn",
    "nhp",
)


def _two_layer_mlp(input_dim: int, hidden_dim: int, output_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.GELU(),
        nn.Linear(hidden_dim, output_dim),
    )


class BenchmarkBackbonePredictor(nn.Module):
    """Pure-PyTorch adapters for the representation baselines used by SEHP."""

    def __init__(
        self,
        *,
        backbone: str,
        num_nodes: int,
        train_edges: list[Hyperedge],
        embedding_dim: int,
        hidden_dim: int,
        dropout: float,
        node_features: torch.Tensor | None = None,
        message_passing_layers: int = 2,
        hnhn_alpha_e: float = 0.0,
        hnhn_alpha_v: float = 0.0,
    ) -> None:
        super().__init__()
        if backbone not in BENCHMARK_BACKBONES:
            raise ValueError(f"unknown benchmark backbone: {backbone}")
        self.backbone = backbone
        self.num_nodes = int(num_nodes)
        self.embedding_dim = int(embedding_dim)
        self.layers = max(1, int(message_passing_layers))
        self.hnhn_alpha_e = float(hnhn_alpha_e)
        self.hnhn_alpha_v = float(hnhn_alpha_v)
        self.dropout = nn.Dropout(dropout)
        if node_features is None:
            self.node_embeddings = nn.Embedding(num_nodes, embedding_dim)
            self.register_buffer("node_features", None)
            self.feature_encoder = None
        else:
            self.node_embeddings = None
            self.register_buffer("node_features", node_features.float())
            self.feature_encoder = DenseOrSparseLinear(node_features.shape[1], embedding_dim)

        node_ids, edge_ids, edge_sizes, node_degrees = _incidence_tensors(train_edges, num_nodes)
        self.register_buffer("incidence_node_ids", node_ids)
        self.register_buffer("incidence_edge_ids", edge_ids)
        self.register_buffer("edge_sizes", edge_sizes)
        self.register_buffer("node_degrees", node_degrees)
        hnhn_vertex_weights = node_degrees.clamp_min(1.0).pow(-self.hnhn_alpha_v)
        hnhn_edge_weights = edge_sizes.clamp_min(1.0).pow(-self.hnhn_alpha_e)
        hnhn_edge_denominators = edge_sizes.new_zeros(edge_sizes.shape)
        hnhn_edge_denominators.index_add_(0, edge_ids, hnhn_vertex_weights[node_ids])
        hnhn_node_denominators = node_degrees.new_zeros(node_degrees.shape)
        hnhn_node_denominators.index_add_(0, node_ids, hnhn_edge_weights[edge_ids])
        self.register_buffer("hnhn_vertex_weights", hnhn_vertex_weights)
        self.register_buffer("hnhn_edge_weights", hnhn_edge_weights)
        self.register_buffer("hnhn_edge_denominators", hnhn_edge_denominators)
        self.register_buffer("hnhn_node_denominators", hnhn_node_denominators)
        if backbone == "nhp":
            clique_indices, clique_values = _full_clique_expansion(
                train_edges,
                num_nodes,
            )
        else:
            clique_indices, clique_values = _clique_expansion(
                train_edges,
                num_nodes,
            )
        self.register_buffer("clique_indices", clique_indices)
        self.register_buffer("clique_values", clique_values)

        self.node_linears = nn.ModuleList(nn.Linear(embedding_dim, embedding_dim) for _ in range(self.layers))
        self.edge_linears = nn.ModuleList(nn.Linear(embedding_dim, embedding_dim) for _ in range(self.layers))
        self.message_linears = nn.ModuleList(
            nn.Sequential(
                nn.Linear(embedding_dim * 2, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, embedding_dim),
            )
            for _ in range(self.layers)
        )
        self.norms = nn.ModuleList(nn.LayerNorm(embedding_dim) for _ in range(self.layers))
        self.allset_node_phi = nn.ModuleList()
        self.allset_edge_rho = nn.ModuleList()
        self.allset_edge_phi = nn.ModuleList()
        self.allset_node_rho = nn.ModuleList()
        if backbone == "allset_deepsets":
            for _ in range(self.layers):
                self.allset_node_phi.append(_two_layer_mlp(embedding_dim, hidden_dim, embedding_dim))
                self.allset_edge_rho.append(_two_layer_mlp(embedding_dim * 2, hidden_dim, embedding_dim))
                self.allset_edge_phi.append(_two_layer_mlp(embedding_dim, hidden_dim, embedding_dim))
                self.allset_node_rho.append(_two_layer_mlp(embedding_dim * 2, hidden_dim, embedding_dim))

        self.hnhn_node_to_edge = nn.ModuleList()
        self.hnhn_edge_to_node = nn.ModuleList()
        if backbone == "hnhn":
            self.hnhn_node_to_edge.extend(nn.Linear(embedding_dim, embedding_dim) for _ in range(self.layers))
            self.hnhn_edge_to_node.extend(nn.Linear(embedding_dim, embedding_dim) for _ in range(self.layers))

        self.edhnn_edge_phi = nn.ModuleList()
        self.edhnn_incidence_updates = nn.ModuleList()
        self.edhnn_node_updates = nn.ModuleList()
        if backbone == "edhnn":
            for _ in range(self.layers):
                self.edhnn_edge_phi.append(_two_layer_mlp(embedding_dim, hidden_dim, embedding_dim))
                self.edhnn_incidence_updates.append(_two_layer_mlp(embedding_dim * 2, hidden_dim, embedding_dim))
                self.edhnn_node_updates.append(_two_layer_mlp(embedding_dim * 2, hidden_dim, embedding_dim))
        self.set_phi = nn.Sequential(nn.Linear(embedding_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, embedding_dim))
        self.attention = nn.MultiheadAttention(embedding_dim, num_heads=4, dropout=dropout, batch_first=True)
        self.hypersagnn_dynamic_norm = nn.LayerNorm(embedding_dim)
        self.hypersagnn_static_norm = nn.LayerNorm(embedding_dim)
        self.hypersagnn_output = nn.Linear(embedding_dim, 1)

        representation_dim = embedding_dim * 4 + 1
        if backbone in {"hnhn", "nhp"}:
            representation_dim = embedding_dim
        if backbone == "hnhn":
            self.classifier = nn.Sequential(
                nn.Linear(representation_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 8),
                nn.ReLU(),
                nn.Linear(8, 1),
            )
        else:
            self.classifier = nn.Sequential(
                nn.Linear(representation_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1),
            )

    def forward_edges(self, edges: list[Hyperedge]) -> torch.Tensor:
        nodes = self.encode_all_nodes()
        if self.backbone == "hypersagnn":
            chunk_size = _attention_chunk_size(edges)
            return torch.cat(
                [
                    self._hypersagnn_chunk(
                        edges[start : start + chunk_size],
                        nodes,
                    )
                    for start in range(0, len(edges), chunk_size)
                ],
                dim=0,
            )
        representations = self.edge_representations(edges, nodes)
        return self.logits_from_edge_representations(representations)

    def edge_representations(
        self,
        edges: list[Hyperedge],
        nodes: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self._encode_edge_batch(edges, self.encode_all_nodes() if nodes is None else nodes)

    def logits_from_edge_representations(self, representations: torch.Tensor) -> torch.Tensor:
        return self.classifier(representations).squeeze(-1)

    def predict_scores(self, edges: list[Hyperedge]) -> list[float]:
        self.eval()
        with torch.no_grad():
            return self.forward_edges(edges).sigmoid().cpu().tolist()

    def encode_edges(self, edges: list[Hyperedge]) -> list[list[float]]:
        self.eval()
        with torch.no_grad():
            if self.backbone == "hypersagnn":
                return self.forward_edges(edges).unsqueeze(1).cpu().tolist()
            return self._encode_edge_batch(edges, self.encode_all_nodes()).cpu().tolist()

    def encode_all_nodes(self) -> torch.Tensor:
        nodes = self._initial_nodes()
        if self.backbone in {"hds", "hypersagnn"}:
            return nodes
        initial = nodes
        for layer, (node_linear, edge_linear, message_linear, norm) in enumerate(
            zip(self.node_linears, self.edge_linears, self.message_linears, self.norms)
        ):
            if self.backbone in {"hypergcn", "nhp"}:
                adjacency = torch.sparse_coo_tensor(
                    self.clique_indices,
                    self.clique_values,
                    (self.num_nodes, self.num_nodes),
                    device=nodes.device,
                ).coalesce()
                updated = torch.sparse.mm(adjacency, nodes)
                nodes = norm(nodes + self.dropout(F.gelu(node_linear(updated))))
                continue

            if self.backbone == "allset_deepsets":
                nodes = self._allset_deepsets_layer(nodes, layer, norm)
                continue

            if self.backbone == "hnhn":
                nodes = self._hnhn_layer(nodes, layer, norm)
                continue

            if self.backbone == "edhnn":
                nodes = self._edhnn_layer(nodes, layer, norm)
                continue

            edge_context, node_context = self._incidence_propagation(nodes)
            if self.backbone == "unigcnii":
                alpha = 0.1
                beta = math.log(0.5 / (layer + 1) + 1.0)
                restarted = (1.0 - alpha) * node_context + alpha * initial
                nodes = norm((1.0 - beta) * restarted + beta * node_linear(restarted))
            else:
                edge_messages = edge_linear(edge_context)[self.incidence_edge_ids]
                node_inputs = torch.cat([nodes[self.incidence_node_ids], edge_messages], dim=1)
                messages = message_linear(node_inputs)
                aggregated = nodes.new_zeros(nodes.shape)
                aggregated.index_add_(0, self.incidence_node_ids, messages)
                aggregated = aggregated / self.node_degrees.clamp_min(1.0).unsqueeze(1)
                nodes = norm(0.5 * initial + 0.5 * self.dropout(F.gelu(aggregated)))
        return nodes

    def _allset_deepsets_layer(self, nodes: torch.Tensor, layer: int, norm: nn.LayerNorm) -> torch.Tensor:
        transformed_nodes = self.allset_node_phi[layer](nodes)
        edge_sum = nodes.new_zeros((int(self.edge_sizes.numel()), nodes.shape[1]))
        edge_sum.index_add_(0, self.incidence_edge_ids, transformed_nodes[self.incidence_node_ids])
        edge_sizes = self.edge_sizes.clamp_min(1.0).unsqueeze(1)
        edge_mean = edge_sum / edge_sizes
        edge_scaled_sum = edge_sum / edge_sizes.sqrt()
        edge_context = self.allset_edge_rho[layer](torch.cat([edge_mean, edge_scaled_sum], dim=1))

        transformed_edges = self.allset_edge_phi[layer](edge_context)
        node_sum = nodes.new_zeros(nodes.shape)
        node_sum.index_add_(0, self.incidence_node_ids, transformed_edges[self.incidence_edge_ids])
        node_context = node_sum / self.node_degrees.clamp_min(1.0).unsqueeze(1)
        update = self.allset_node_rho[layer](torch.cat([nodes, node_context], dim=1))
        return norm(nodes + self.dropout(update))

    def _hnhn_layer(self, nodes: torch.Tensor, layer: int, norm: nn.LayerNorm) -> torch.Tensor:
        # HNHN's two directional normalizers use independently tunable
        # cardinality exponents (alpha_v and alpha_e). The nonlinearities in
        # both directions are essential: without them this reduces to a
        # linear clique-style propagation and quickly over-smooths.
        node_messages = F.relu(self.hnhn_node_to_edge[layer](nodes))
        vertex_coefficients = (
            self.hnhn_vertex_weights[self.incidence_node_ids]
            / self.hnhn_edge_denominators[self.incidence_edge_ids].clamp_min(1e-12)
        ).unsqueeze(1)
        edge_context = nodes.new_zeros((int(self.edge_sizes.numel()), nodes.shape[1]))
        edge_context.index_add_(
            0,
            self.incidence_edge_ids,
            node_messages[self.incidence_node_ids] * vertex_coefficients,
        )
        edge_context = F.relu(self.hnhn_edge_to_node[layer](edge_context))

        edge_coefficients = (
            self.hnhn_edge_weights[self.incidence_edge_ids]
            / self.hnhn_node_denominators[self.incidence_node_ids].clamp_min(1e-12)
        ).unsqueeze(1)
        node_context = nodes.new_zeros(nodes.shape)
        node_context.index_add_(
            0,
            self.incidence_node_ids,
            edge_context[self.incidence_edge_ids] * edge_coefficients,
        )
        if layer + 1 < self.layers:
            node_context = self.dropout(node_context)
        return node_context

    def _edhnn_layer(self, nodes: torch.Tensor, layer: int, norm: nn.LayerNorm) -> torch.Tensor:
        edge_values = self.edhnn_edge_phi[layer](nodes)
        edge_context = nodes.new_zeros((int(self.edge_sizes.numel()), nodes.shape[1]))
        edge_context.index_add_(0, self.incidence_edge_ids, edge_values[self.incidence_node_ids])
        edge_context = edge_context / self.edge_sizes.clamp_min(1.0).unsqueeze(1)

        incidence_inputs = torch.cat(
            [nodes[self.incidence_node_ids], edge_context[self.incidence_edge_ids]],
            dim=1,
        )
        incidence_messages = self.edhnn_incidence_updates[layer](incidence_inputs)
        node_context = nodes.new_zeros(nodes.shape)
        node_context.index_add_(0, self.incidence_node_ids, incidence_messages)
        node_context = node_context / self.node_degrees.clamp_min(1.0).unsqueeze(1)
        update = self.edhnn_node_updates[layer](torch.cat([nodes, node_context], dim=1))
        return norm(nodes + self.dropout(update))

    def _incidence_propagation(self, nodes: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        edge_count = int(self.edge_sizes.numel())
        edge_context = nodes.new_zeros((edge_count, nodes.shape[1]))
        edge_context.index_add_(0, self.incidence_edge_ids, nodes[self.incidence_node_ids])
        edge_context = edge_context / self.edge_sizes.clamp_min(1.0).unsqueeze(1)
        node_context = nodes.new_zeros(nodes.shape)
        node_context.index_add_(0, self.incidence_node_ids, edge_context[self.incidence_edge_ids])
        node_context = node_context / self.node_degrees.clamp_min(1.0).unsqueeze(1)
        return edge_context, node_context

    def _encode_edge_batch(self, edges: list[Hyperedge], nodes: torch.Tensor) -> torch.Tensor:
        if self.backbone == "hds":
            transformed = self.set_phi(nodes)
            return torch.cat(
                [_pool_stat_edge_chunk(edges[start : start + 512], transformed) for start in range(0, len(edges), 512)],
                dim=0,
            )
        if self.backbone in {"hnhn", "nhp"}:
            return torch.cat(
                [_maxmin_edge_chunk(edges[start : start + 512], nodes) for start in range(0, len(edges), 512)],
                dim=0,
            )
        return torch.cat(
            [_pool_stat_edge_chunk(edges[start : start + 512], nodes) for start in range(0, len(edges), 512)],
            dim=0,
        )

    def _hypersagnn_chunk(self, edges: list[Hyperedge], nodes: torch.Tensor) -> torch.Tensor:
        max_size = max(len(edge) for edge in edges)
        ids = torch.zeros((len(edges), max_size), dtype=torch.long, device=nodes.device)
        padding = torch.ones((len(edges), max_size), dtype=torch.bool, device=nodes.device)
        for row, edge in enumerate(edges):
            ids[row, : len(edge)] = torch.tensor(edge, dtype=torch.long, device=nodes.device)
            padding[row, : len(edge)] = False
        static = nodes[ids]
        diagonal_mask = torch.eye(
            max_size,
            dtype=torch.bool,
            device=nodes.device,
        )
        dynamic, _ = self.attention(
            static,
            static,
            static,
            attn_mask=diagonal_mask,
            key_padding_mask=padding,
            need_weights=False,
        )
        dynamic = self.hypersagnn_dynamic_norm(dynamic)
        static = self.hypersagnn_static_norm(static)
        valid = (~padding).float().unsqueeze(-1)
        node_probabilities = torch.sigmoid(
            self.hypersagnn_output((dynamic - static).square())
        )
        edge_probabilities = (
            (node_probabilities * valid).sum(dim=1)
            / valid.sum(dim=1).clamp_min(1.0)
        ).squeeze(-1)
        return torch.logit(edge_probabilities.clamp(1e-6, 1.0 - 1e-6))

    def _initial_nodes(self) -> torch.Tensor:
        if self.node_features is not None:
            if self.feature_encoder is None:
                raise RuntimeError("feature encoder unavailable")
            return self.feature_encoder(self.node_features)
        if self.node_embeddings is None:
            raise RuntimeError("node embeddings unavailable")
        return self.node_embeddings.weight


def _incidence_tensors(
    edges: list[Hyperedge],
    num_nodes: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    node_ids = torch.tensor([node for edge in edges for node in edge], dtype=torch.long)
    edge_ids = torch.repeat_interleave(
        torch.arange(len(edges)),
        torch.tensor([len(edge) for edge in edges]),
    )
    edge_sizes = torch.bincount(edge_ids, minlength=len(edges)).float()
    node_degrees = torch.bincount(node_ids, minlength=num_nodes).float()
    return node_ids, edge_ids, edge_sizes, node_degrees


def _clique_expansion(edges: list[Hyperedge], num_nodes: int) -> tuple[torch.Tensor, torch.Tensor]:
    pairs: list[tuple[int, int]] = [(node, node) for node in range(num_nodes)]
    for edge in edges:
        if len(edge) < 2:
            continue
        low, high = min(edge), max(edge)
        pairs.extend([(low, high), (high, low)])
        for node in edge:
            if node not in {low, high}:
                pairs.extend([(low, node), (node, low), (high, node), (node, high)])
    indices = torch.tensor(pairs, dtype=torch.long).T
    flat = indices[0] * num_nodes + indices[1]
    unique, counts = torch.unique(flat, return_counts=True)
    rows, cols = unique.div(num_nodes, rounding_mode="floor"), unique.remainder(num_nodes)
    degrees = torch.zeros(num_nodes, dtype=torch.float32)
    degrees.index_add_(0, rows, counts.float())
    values = counts.float() / (degrees[rows].clamp_min(1.0).sqrt() * degrees[cols].clamp_min(1.0).sqrt())
    return torch.stack([rows, cols]), values


def _full_clique_expansion(
    edges: list[Hyperedge],
    num_nodes: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    pairs: list[tuple[int, int]] = [
        (node, node) for node in range(num_nodes)
    ]
    for edge in edges:
        for left_index, left in enumerate(edge):
            for right in edge[left_index + 1 :]:
                pairs.extend(((left, right), (right, left)))
    indices = torch.tensor(pairs, dtype=torch.long).T
    flat = indices[0] * num_nodes + indices[1]
    unique, counts = torch.unique(flat, return_counts=True)
    rows = unique.div(num_nodes, rounding_mode="floor")
    cols = unique.remainder(num_nodes)
    degrees = torch.zeros(num_nodes, dtype=torch.float32)
    degrees.index_add_(0, rows, counts.float())
    values = counts.float() / (
        degrees[rows].clamp_min(1.0).sqrt()
        * degrees[cols].clamp_min(1.0).sqrt()
    )
    return torch.stack([rows, cols]), values


def _attention_chunk_size(edges: list[Hyperedge]) -> int:
    if not edges:
        return 1
    max_size = max(len(edge) for edge in edges)
    if max_size >= 1024:
        return 1
    if max_size >= 512:
        return 2
    if max_size >= 256:
        return 4
    if max_size >= 128:
        return 16
    return 64


def _mean_edge_chunk(edges: list[Hyperedge], nodes: torch.Tensor) -> torch.Tensor:
    node_ids = torch.tensor([node for edge in edges for node in edge], dtype=torch.long, device=nodes.device)
    edge_ids = torch.repeat_interleave(
        torch.arange(len(edges), device=nodes.device),
        torch.tensor([len(edge) for edge in edges], device=nodes.device),
    )
    output = nodes.new_zeros((len(edges), nodes.shape[1]))
    output.index_add_(0, edge_ids, nodes[node_ids])
    return output / torch.bincount(edge_ids, minlength=len(edges)).to(nodes.dtype).clamp_min(1.0).unsqueeze(1)


def _maxmin_edge_chunk(
    edges: list[Hyperedge],
    nodes: torch.Tensor,
) -> torch.Tensor:
    return torch.stack(
        [
            nodes[torch.tensor(edge, dtype=torch.long, device=nodes.device)].amax(dim=0)
            - nodes[torch.tensor(edge, dtype=torch.long, device=nodes.device)].amin(dim=0)
            for edge in edges
        ],
        dim=0,
    )
