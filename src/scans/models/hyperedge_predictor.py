from __future__ import annotations

import torch
from torch import nn

from scans.data.hypergraph import Hyperedge
from scans.models.feature_encoder import DenseOrSparseLinear


class HyperedgePredictor(nn.Module):
    def __init__(
        self,
        num_nodes: int,
        embedding_dim: int,
        hidden_dim: int,
        dropout: float,
        node_features: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        if node_features is None:
            self.node_embeddings = nn.Embedding(num_nodes, embedding_dim)
            self.register_buffer("node_features", None)
            self.feature_encoder = None
        else:
            self.node_embeddings = None
            self.register_buffer("node_features", node_features.float())
            self.feature_encoder = DenseOrSparseLinear(node_features.shape[1], embedding_dim)
        representation_dim = embedding_dim
        self.classifier = nn.Sequential(
            nn.Linear(representation_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward_edges(self, edges: list[Hyperedge]) -> torch.Tensor:
        edge_representations = [self.encode_edge(edge) for edge in edges]
        stacked = torch.stack(edge_representations, dim=0)
        return self.classifier(stacked).squeeze(-1)

    def predict_scores(self, edges: list[Hyperedge]) -> list[float]:
        self.eval()
        with torch.no_grad():
            logits = self.forward_edges(edges)
            return torch.sigmoid(logits).detach().cpu().tolist()

    def encode_edges(self, edges: list[Hyperedge]) -> list[list[float]]:
        self.eval()
        with torch.no_grad():
            return [self.encode_edge(edge).detach().cpu().tolist() for edge in edges]

    def encode_edge(self, edge: Hyperedge) -> torch.Tensor:
        node_ids = torch.tensor(edge, dtype=torch.long, device=self._device())
        if self.node_features is None:
            if self.node_embeddings is None:
                raise RuntimeError("node_embeddings are not initialized")
            node_representations = self.node_embeddings(node_ids)
        else:
            if self.feature_encoder is None:
                raise RuntimeError("feature_encoder is not initialized")
            if self.node_features.layout in {torch.sparse_coo, torch.sparse_csr}:
                node_representations = self.feature_encoder(self.node_features)[node_ids]
            else:
                node_representations = self.feature_encoder(self.node_features[node_ids])
        return node_representations.mean(dim=0)

    def _device(self) -> torch.device:
        return next(self.parameters()).device


class SetHyperedgePredictor(nn.Module):
    def __init__(
        self,
        num_nodes: int,
        embedding_dim: int,
        hidden_dim: int,
        dropout: float,
        node_features: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        if node_features is None:
            self.node_embeddings = nn.Embedding(num_nodes, embedding_dim)
            self.register_buffer("node_features", None)
            self.feature_encoder = None
        else:
            self.node_embeddings = None
            self.register_buffer("node_features", node_features.float())
            self.feature_encoder = nn.Sequential(
                DenseOrSparseLinear(node_features.shape[1], hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, embedding_dim),
                nn.ReLU(),
            )

        representation_dim = embedding_dim * 4 + 1
        self.classifier = nn.Sequential(
            nn.Linear(representation_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward_edges(self, edges: list[Hyperedge]) -> torch.Tensor:
        edge_representations = [self.encode_edge(edge) for edge in edges]
        stacked = torch.stack(edge_representations, dim=0)
        return self.classifier(stacked).squeeze(-1)

    def predict_scores(self, edges: list[Hyperedge]) -> list[float]:
        self.eval()
        with torch.no_grad():
            logits = self.forward_edges(edges)
            return torch.sigmoid(logits).detach().cpu().tolist()

    def encode_edges(self, edges: list[Hyperedge]) -> list[list[float]]:
        self.eval()
        with torch.no_grad():
            return [self.encode_edge(edge).detach().cpu().tolist() for edge in edges]

    def encode_edge(self, edge: Hyperedge) -> torch.Tensor:
        node_ids = torch.tensor(edge, dtype=torch.long, device=self._device())
        node_representations = self._encode_nodes(node_ids)
        mean_pool = node_representations.mean(dim=0)
        max_pool = node_representations.max(dim=0).values
        min_pool = node_representations.min(dim=0).values
        std_pool = node_representations.std(dim=0, unbiased=False)
        size_feature = torch.tensor(
            [torch.log1p(torch.tensor(float(len(edge)), device=self._device()))],
            dtype=torch.float32,
            device=self._device(),
        )
        return torch.cat([mean_pool, max_pool, min_pool, std_pool, size_feature], dim=0)

    def _encode_nodes(self, node_ids: torch.Tensor) -> torch.Tensor:
        if self.node_features is None:
            if self.node_embeddings is None:
                raise RuntimeError("node_embeddings are not initialized")
            return self.node_embeddings(node_ids)
        if self.feature_encoder is None:
            raise RuntimeError("feature_encoder is not initialized")
        if self.node_features.layout in {torch.sparse_coo, torch.sparse_csr}:
            return self.feature_encoder(self.node_features)[node_ids]
        return self.feature_encoder(self.node_features[node_ids])

    def _device(self) -> torch.device:
        return next(self.parameters()).device


class BipartiteHyperedgePredictor(nn.Module):
    def __init__(
        self,
        num_nodes: int,
        embedding_dim: int,
        hidden_dim: int,
        dropout: float,
        node_features: torch.Tensor | None = None,
        train_edges: list[Hyperedge] | None = None,
        message_passing_layers: int = 1,
        use_residual: bool = False,
        use_layer_norm: bool = False,
        representation_fusion_layers: int = 1,
    ) -> None:
        super().__init__()
        self.num_nodes = num_nodes
        self.message_passing_layers = max(0, message_passing_layers)
        self.use_residual = bool(use_residual)
        self.use_layer_norm = bool(use_layer_norm)
        self.representation_fusion_layers = max(1, int(representation_fusion_layers))
        if node_features is None:
            self.node_embeddings = nn.Embedding(num_nodes, embedding_dim)
            self.register_buffer("node_features", None)
            self.feature_encoder = None
        else:
            self.node_embeddings = None
            self.register_buffer("node_features", node_features.float())
            self.feature_encoder = nn.Sequential(
                DenseOrSparseLinear(node_features.shape[1], hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, embedding_dim),
                nn.ReLU(),
            )

        self.node_updates = nn.ModuleList(
            nn.Linear(embedding_dim * 2, embedding_dim)
            for _ in range(self.message_passing_layers)
        )
        self.node_norms = nn.ModuleList(
            nn.LayerNorm(embedding_dim)
            for _ in range(self.message_passing_layers)
        )
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        incidence_node_ids, incidence_edge_ids, edge_sizes, node_degrees = self._build_incidence_tensors(
            train_edges or [],
            num_nodes,
        )
        self.register_buffer("incidence_node_ids", incidence_node_ids)
        self.register_buffer("incidence_edge_ids", incidence_edge_ids)
        self.register_buffer("train_edge_sizes", edge_sizes)
        self.register_buffer("node_degrees", node_degrees)

        representation_dim = embedding_dim * 4 + 1
        self.classifier = nn.Sequential(
            nn.Linear(representation_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward_edges(self, edges: list[Hyperedge]) -> torch.Tensor:
        node_representations = self.encode_all_nodes()
        return self.classifier(self._encode_edge_batch(edges, node_representations)).squeeze(-1)

    def predict_scores(self, edges: list[Hyperedge]) -> list[float]:
        self.eval()
        with torch.no_grad():
            logits = self.forward_edges(edges)
            return torch.sigmoid(logits).detach().cpu().tolist()

    def encode_edges(self, edges: list[Hyperedge]) -> list[list[float]]:
        self.eval()
        with torch.no_grad():
            node_representations = self.encode_all_nodes()
            return self._encode_edge_batch(edges, node_representations).detach().cpu().tolist()

    def _encode_edge_batch(self, edges: list[Hyperedge], node_representations: torch.Tensor) -> torch.Tensor:
        return torch.cat(
            [_pool_stat_edge_chunk(edges[start : start + 512], node_representations) for start in range(0, len(edges), 512)],
            dim=0,
        )

    def encode_all_nodes(self) -> torch.Tensor:
        node_ids = torch.arange(self.num_nodes, dtype=torch.long, device=self._device())
        node_representations = self._encode_nodes(node_ids)
        if self.message_passing_layers == 0 or self.incidence_node_ids.numel() == 0:
            return node_representations

        edge_ids = self.incidence_edge_ids
        incidence_node_ids = self.incidence_node_ids
        edge_count = int(self.train_edge_sizes.numel())
        layer_representations: list[torch.Tensor] = []
        for node_update, node_norm in zip(self.node_updates, self.node_norms):
            edge_sum = node_representations.new_zeros((edge_count, node_representations.shape[1]))
            edge_sum.index_add_(0, edge_ids, node_representations[incidence_node_ids])
            edge_context = edge_sum / self.train_edge_sizes.clamp_min(1.0).unsqueeze(1)

            node_sum = node_representations.new_zeros(node_representations.shape)
            node_sum.index_add_(0, incidence_node_ids, edge_context[edge_ids])
            node_context = node_sum / self.node_degrees.clamp_min(1.0).unsqueeze(1)

            combined = torch.cat([node_representations, node_context], dim=1)
            updated = self.dropout(self.activation(node_update(combined)))
            if self.use_residual:
                updated = node_representations + updated
            if self.use_layer_norm:
                updated = node_norm(updated)
            node_representations = updated
            layer_representations.append(node_representations)
        if self.representation_fusion_layers <= 1 or len(layer_representations) <= 1:
            return node_representations
        selected_layers = layer_representations[-self.representation_fusion_layers :]
        return torch.stack(selected_layers, dim=0).mean(dim=0)

    def encode_edge_with_nodes(
        self,
        edge: Hyperedge,
        node_representations: torch.Tensor,
    ) -> torch.Tensor:
        node_ids = torch.tensor(edge, dtype=torch.long, device=node_representations.device)
        edge_nodes = node_representations[node_ids]
        mean_pool = edge_nodes.mean(dim=0)
        max_pool = edge_nodes.max(dim=0).values
        min_pool = edge_nodes.min(dim=0).values
        std_pool = edge_nodes.std(dim=0, unbiased=False)
        size_feature = torch.tensor(
            [torch.log1p(torch.tensor(float(len(edge)), device=node_representations.device))],
            dtype=torch.float32,
            device=node_representations.device,
        )
        return torch.cat([mean_pool, max_pool, min_pool, std_pool, size_feature], dim=0)

    def _encode_nodes(self, node_ids: torch.Tensor) -> torch.Tensor:
        if self.node_features is None:
            if self.node_embeddings is None:
                raise RuntimeError("node_embeddings are not initialized")
            return self.node_embeddings(node_ids)
        if self.feature_encoder is None:
            raise RuntimeError("feature_encoder is not initialized")
        if self.node_features.layout in {torch.sparse_coo, torch.sparse_csr}:
            return self.feature_encoder(self.node_features)[node_ids]
        return self.feature_encoder(self.node_features[node_ids])

    def _device(self) -> torch.device:
        return next(self.parameters()).device

    @staticmethod
    def _build_incidence_tensors(
        train_edges: list[Hyperedge],
        num_nodes: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        incidence_node_ids: list[int] = []
        incidence_edge_ids: list[int] = []
        edge_sizes: list[float] = []
        node_degrees = torch.zeros(num_nodes, dtype=torch.float32)
        for edge_id, edge in enumerate(train_edges):
            edge_sizes.append(float(len(edge)))
            for node_id in edge:
                incidence_node_ids.append(node_id)
                incidence_edge_ids.append(edge_id)
                node_degrees[node_id] += 1.0
        return (
            torch.tensor(incidence_node_ids, dtype=torch.long),
            torch.tensor(incidence_edge_ids, dtype=torch.long),
            torch.tensor(edge_sizes, dtype=torch.float32),
            node_degrees,
        )


def _pool_stat_edge_chunk(edges: list[Hyperedge], nodes: torch.Tensor) -> torch.Tensor:
    node_ids = torch.tensor([node for edge in edges for node in edge], dtype=torch.long, device=nodes.device)
    edge_ids = torch.repeat_interleave(
        torch.arange(len(edges), device=nodes.device),
        torch.tensor([len(edge) for edge in edges], device=nodes.device),
    )
    values = nodes[node_ids]
    counts = torch.bincount(edge_ids, minlength=len(edges)).to(nodes.dtype).unsqueeze(1)
    sums = nodes.new_zeros((len(edges), nodes.shape[1]))
    sums.index_add_(0, edge_ids, values)
    means = sums / counts.clamp_min(1.0)
    square_sums = nodes.new_zeros((len(edges), nodes.shape[1]))
    square_sums.index_add_(0, edge_ids, values.square())
    standard_deviations = ((square_sums / counts.clamp_min(1.0) - means.square()).clamp_min(0.0) + 1e-8).sqrt()
    expanded_ids = edge_ids.unsqueeze(1).expand_as(values)
    maxima = nodes.new_full((len(edges), nodes.shape[1]), float("-inf"))
    minima = nodes.new_full((len(edges), nodes.shape[1]), float("inf"))
    maxima.scatter_reduce_(0, expanded_ids, values, reduce="amax", include_self=True)
    minima.scatter_reduce_(0, expanded_ids, values, reduce="amin", include_self=True)
    sizes = counts.log1p()
    return torch.cat([means, maxima, minima, standard_deviations, sizes], dim=1)
