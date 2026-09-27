from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from scans.data.hypergraph import Hyperedge
from scans.models.feature_encoder import DenseOrSparseLinear


class SparseIncidenceEncoder(nn.Module):
    """Sparse node-hyperedge encoder with no dense graph adjacency."""

    def __init__(
        self,
        num_nodes: int,
        train_edges: list[Hyperedge],
        embedding_dim: int = 128,
        hidden_dim: int = 256,
        layers: int = 3,
        dropout: float = 0.1,
        node_features: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.num_nodes = int(num_nodes)
        self.embedding_dim = int(embedding_dim)
        self.layers = int(layers)
        if node_features is None:
            self.node_embeddings = nn.Embedding(num_nodes, embedding_dim)
            self.register_buffer("node_features", None)
            self.feature_encoder = None
        else:
            self.node_embeddings = None
            self.register_buffer("node_features", node_features.float())
            self.feature_encoder = DenseOrSparseLinear(node_features.shape[1], embedding_dim)

        self.node_updates = nn.ModuleList(
            nn.Sequential(
                nn.Linear(embedding_dim * 2, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, embedding_dim),
            )
            for _ in range(layers)
        )
        self.edge_updates = nn.ModuleList(
            nn.Sequential(
                nn.Linear(embedding_dim * 2, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, embedding_dim),
            )
            for _ in range(layers)
        )
        self.node_norms = nn.ModuleList(nn.LayerNorm(embedding_dim) for _ in range(layers))
        self.edge_norms = nn.ModuleList(nn.LayerNorm(embedding_dim) for _ in range(layers))
        self.edge_seed = nn.Parameter(torch.zeros(embedding_dim))

        node_ids, edge_ids, edge_sizes, node_degrees = _incidence_tensors(train_edges, num_nodes)
        self.register_buffer("incidence_node_ids", node_ids)
        self.register_buffer("incidence_edge_ids", edge_ids)
        self.register_buffer("edge_sizes", edge_sizes)
        self.register_buffer("node_degrees", node_degrees)

    def forward(
        self,
        incidence_keep: torch.Tensor | None = None,
        incidence_indices: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        nodes = self._initial_nodes()
        edge_count = int(self.edge_sizes.numel())
        edges = self.edge_seed.unsqueeze(0).expand(edge_count, -1)
        if edge_count == 0:
            return nodes, edges

        node_ids = self.incidence_node_ids if incidence_indices is None else self.incidence_node_ids[incidence_indices]
        edge_ids = self.incidence_edge_ids if incidence_indices is None else self.incidence_edge_ids[incidence_indices]
        if incidence_keep is None:
            incidence_keep = torch.ones_like(node_ids, dtype=nodes.dtype)
        else:
            incidence_keep = incidence_keep.to(device=nodes.device, dtype=nodes.dtype)

        for node_update, edge_update, node_norm, edge_norm in zip(
            self.node_updates,
            self.edge_updates,
            self.node_norms,
            self.edge_norms,
        ):
            edge_sum = nodes.new_zeros((edge_count, self.embedding_dim))
            edge_sum.index_add_(0, edge_ids, nodes[node_ids] * incidence_keep.unsqueeze(1))
            kept_edge_sizes = nodes.new_zeros(edge_count)
            kept_edge_sizes.index_add_(0, edge_ids, incidence_keep)
            edge_context = edge_sum / kept_edge_sizes.clamp_min(1.0).unsqueeze(1)
            edges = edge_norm(edges + edge_update(torch.cat([edges, edge_context], dim=1)))

            node_sum = nodes.new_zeros((self.num_nodes, self.embedding_dim))
            node_sum.index_add_(0, node_ids, edges[edge_ids] * incidence_keep.unsqueeze(1))
            kept_degrees = nodes.new_zeros(self.num_nodes)
            kept_degrees.index_add_(0, node_ids, incidence_keep)
            node_context = node_sum / kept_degrees.clamp_min(1.0).unsqueeze(1)
            nodes = node_norm(nodes + node_update(torch.cat([nodes, node_context], dim=1)))
        return nodes, edges

    def encode_edges(self, edges: list[Hyperedge], nodes: torch.Tensor | None = None) -> torch.Tensor:
        if nodes is None:
            nodes, _ = self.forward()
        return torch.cat([_pool_edge_chunk(chunk, nodes) for chunk in _chunks(edges, 512)], dim=0)

    @property
    def edge_representation_dim(self) -> int:
        return self.embedding_dim * 3

    def _initial_nodes(self) -> torch.Tensor:
        if self.node_features is not None:
            if self.feature_encoder is None:
                raise RuntimeError("feature encoder is unavailable")
            return self.feature_encoder(self.node_features)
        if self.node_embeddings is None:
            raise RuntimeError("node embeddings are unavailable")
        return self.node_embeddings.weight


class StrongEncoderPretrainer(nn.Module):
    def __init__(self, encoder: SparseIncidenceEncoder, projection_dim: int = 128) -> None:
        super().__init__()
        self.encoder = encoder
        dim = encoder.embedding_dim
        self.incidence_temperature = nn.Parameter(torch.tensor(math.log(1 / math.sqrt(dim))))
        self.context_predictor = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, dim))
        self.projector = nn.Sequential(nn.Linear(dim, projection_dim), nn.GELU(), nn.Linear(projection_dim, projection_dim))

    def loss(
        self,
        *,
        mask_probability: float = 0.15,
        contrastive_temperature: float = 0.2,
        max_incidence_samples: int = 262_144,
        max_contrastive_edges: int = 1_024,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        device = self.encoder.edge_sizes.device
        count = int(self.encoder.incidence_node_ids.numel())
        if count == 0:
            raise ValueError("pretraining requires at least one incidence")

        incidence_indices = torch.arange(count, device=device)
        if count > max_incidence_samples:
            incidence_indices = torch.randperm(count, device=device, generator=generator)[:max_incidence_samples]
        view_count = int(incidence_indices.numel())
        random_values = torch.rand(view_count, device=device, generator=generator)
        keep_a = (random_values >= mask_probability).float()
        keep_b = (torch.rand(view_count, device=device, generator=generator) >= mask_probability).float()
        nodes_a, edges_a = self.encoder(keep_a, incidence_indices)
        nodes_b, edges_b = self.encoder(keep_b, incidence_indices)

        node_ids = self.encoder.incidence_node_ids[incidence_indices]
        edge_ids = self.encoder.incidence_edge_ids[incidence_indices]
        masked = keep_a == 0
        if not bool(masked.any()):
            masked[0] = True
        masked_indices = masked.nonzero(as_tuple=False).squeeze(1)
        sampled_node_ids = node_ids[masked_indices]
        sampled_edge_ids = edge_ids[masked_indices]
        pos_logits = (nodes_a[sampled_node_ids] * edges_a[sampled_edge_ids]).sum(dim=1)
        neg_node_ids = torch.randint(
            self.encoder.num_nodes,
            (masked_indices.numel(),),
            device=device,
            generator=generator,
        )
        neg_logits = (nodes_a[neg_node_ids] * edges_a[sampled_edge_ids]).sum(dim=1)
        scale = self.incidence_temperature.exp().clamp(1e-3, 100.0)
        incidence_loss = F.binary_cross_entropy_with_logits(
            torch.cat([pos_logits, neg_logits]) * scale,
            torch.cat([torch.ones_like(pos_logits), torch.zeros_like(neg_logits)]),
        )

        active_edges = torch.unique(edge_ids)
        context_loss = F.mse_loss(self.context_predictor(edges_a[active_edges]), edges_b[active_edges].detach())

        contrastive_indices = active_edges
        if contrastive_indices.numel() > max_contrastive_edges:
            contrastive_indices = contrastive_indices[
                torch.randperm(contrastive_indices.numel(), device=device, generator=generator)[:max_contrastive_edges]
            ]
        proj_a = F.normalize(self.projector(edges_a[contrastive_indices]), dim=1)
        proj_b = F.normalize(self.projector(edges_b[contrastive_indices]), dim=1)
        logits = proj_a @ proj_b.T / max(float(contrastive_temperature), 1e-6)
        labels = torch.arange(logits.shape[0], device=device)
        contrastive_loss = 0.5 * (
            F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)
        )
        total = incidence_loss + context_loss + contrastive_loss
        return total, {
            "encoder_loss": float(total.detach()),
            "masked_incidence_loss": float(incidence_loss.detach()),
            "context_loss": float(context_loss.detach()),
            "contrastive_loss": float(contrastive_loss.detach()),
        }


@dataclass(frozen=True)
class PretrainingSummary:
    epochs: int
    final_loss: float
    masked_incidence_loss: float
    context_loss: float
    contrastive_loss: float


def pretrain_encoder(
    pretrainer: StrongEncoderPretrainer,
    *,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    mask_probability: float,
    contrastive_temperature: float,
) -> PretrainingSummary:
    optimizer = torch.optim.AdamW(pretrainer.parameters(), lr=learning_rate, weight_decay=weight_decay)
    metrics: dict[str, float] = {}
    pretrainer.train()
    for _ in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        loss, metrics = pretrainer.loss(
            mask_probability=mask_probability,
            contrastive_temperature=contrastive_temperature,
        )
        loss.backward()
        optimizer.step()
    return PretrainingSummary(
        epochs=epochs,
        final_loss=metrics["encoder_loss"],
        masked_incidence_loss=metrics["masked_incidence_loss"],
        context_loss=metrics["context_loss"],
        contrastive_loss=metrics["contrastive_loss"],
    )


class StrongHyperedgePredictor(nn.Module):
    def __init__(
        self,
        num_nodes: int,
        train_edges: list[Hyperedge],
        embedding_dim: int,
        hidden_dim: int,
        dropout: float,
        node_features: torch.Tensor | None = None,
        message_passing_layers: int = 3,
        freeze_encoder: bool = True,
    ) -> None:
        super().__init__()
        self.encoder = SparseIncidenceEncoder(
            num_nodes=num_nodes,
            train_edges=train_edges,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            layers=message_passing_layers,
            dropout=dropout,
            node_features=node_features,
        )
        self.freeze_encoder = bool(freeze_encoder)
        self.classifier = nn.Sequential(
            nn.Linear(self.encoder.edge_representation_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def pretrain(
        self,
        *,
        epochs: int,
        learning_rate: float,
        weight_decay: float,
        mask_probability: float,
        contrastive_temperature: float,
    ) -> PretrainingSummary:
        pretrainer = StrongEncoderPretrainer(self.encoder, projection_dim=self.encoder.embedding_dim)
        summary = pretrain_encoder(
            pretrainer,
            epochs=epochs,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            mask_probability=mask_probability,
            contrastive_temperature=contrastive_temperature,
        )
        if self.freeze_encoder:
            for parameter in self.encoder.parameters():
                parameter.requires_grad_(False)
        return summary

    def forward_edges(self, edges: list[Hyperedge]) -> torch.Tensor:
        if self.freeze_encoder:
            with torch.no_grad():
                nodes, _ = self.encoder()
                representations = self.encoder.encode_edges(edges, nodes)
        else:
            nodes, _ = self.encoder()
            representations = self.encoder.encode_edges(edges, nodes)
        return self.classifier(representations).squeeze(-1)

    def predict_scores(self, edges: list[Hyperedge]) -> list[float]:
        self.eval()
        with torch.no_grad():
            return torch.sigmoid(self.forward_edges(edges)).cpu().tolist()

    def encode_edges(self, edges: list[Hyperedge]) -> list[list[float]]:
        self.eval()
        with torch.no_grad():
            nodes, _ = self.encoder()
            return self.encoder.encode_edges(edges, nodes).cpu().tolist()


def _incidence_tensors(
    train_edges: list[Hyperedge],
    num_nodes: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    node_ids: list[int] = []
    edge_ids: list[int] = []
    for edge_id, edge in enumerate(train_edges):
        node_ids.extend(edge)
        edge_ids.extend([edge_id] * len(edge))
    nodes = torch.tensor(node_ids, dtype=torch.long)
    edges = torch.tensor(edge_ids, dtype=torch.long)
    edge_sizes = torch.bincount(edges, minlength=len(train_edges)).float()
    node_degrees = torch.bincount(nodes, minlength=num_nodes).float()
    return nodes, edges, edge_sizes, node_degrees


def _chunks(edges: list[Hyperedge], size: int) -> list[list[Hyperedge]]:
    return [edges[start : start + size] for start in range(0, len(edges), size)]


def _pool_edge_chunk(edges: list[Hyperedge], nodes: torch.Tensor) -> torch.Tensor:
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
    maxima = nodes.new_full((len(edges), nodes.shape[1]), float("-inf"))
    maxima.scatter_reduce_(0, edge_ids.unsqueeze(1).expand_as(values), values, reduce="amax", include_self=True)
    return torch.cat([means, maxima, standard_deviations], dim=1)
