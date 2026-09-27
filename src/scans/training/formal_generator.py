from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from scans.data.hypergraph import Hyperedge
from scans.models import (
    DiscreteMembershipD3PM,
    LearnedCandidateRetriever,
    LearnedDualAnchor,
    MembershipDenoiser,
    PositiveSupportRiskEstimator,
    SparseIncidenceEncoder,
    StrongEncoderPretrainer,
    pretrain_encoder,
    primal_dual_update,
)


@dataclass(frozen=True)
class FormalGeneratorConfig:
    embedding_dim: int = 64
    hidden_dim: int = 128
    encoder_layers: int = 2
    encoder_epochs: int = 30
    encoder_mode: str = "strong_finetuned"
    generator_epochs: int = 30
    risk_epochs: int = 30
    candidate_size: int = 64
    diffusion_steps: int = 8
    batch_size: int = 64
    anchor_mode: str = "dual"
    diffusion_mode: str = "d3pm"
    risk_mode: str = "primal_dual"
    risk_budget: float = 0.5
    target_hardness: float = 0.5
    proposals_per_edge: int = 4
    representation_cache_path: str | None = None


class FormalDiscreteGenerator:
    def __init__(
        self,
        num_nodes: int,
        train_edges: list[Hyperedge],
        config: FormalGeneratorConfig,
        node_features: torch.Tensor | None = None,
    ) -> None:
        _validate_config(config)
        self.num_nodes = int(num_nodes)
        self.train_edges = train_edges
        self.config = config
        self.encoder = SparseIncidenceEncoder(
            num_nodes=num_nodes,
            train_edges=train_edges,
            embedding_dim=config.embedding_dim,
            hidden_dim=config.hidden_dim,
            layers=config.encoder_layers,
            dropout=0.1,
            node_features=node_features,
        )
        max_edge_size = max(len(edge) for edge in train_edges)
        self.anchor = LearnedDualAnchor(config.embedding_dim, config.hidden_dim, max_edge_size=max_edge_size)
        self.retriever = LearnedCandidateRetriever(config.embedding_dim)
        self.diffusion = DiscreteMembershipD3PM(
            MembershipDenoiser(config.embedding_dim, config.hidden_dim, max_steps=config.diffusion_steps),
            steps=config.diffusion_steps,
        )
        self.risk = PositiveSupportRiskEstimator(config.embedding_dim, config.hidden_dim)
        self.boundary = PositiveSupportRiskEstimator(config.embedding_dim, config.hidden_dim)
        self.risk_multiplier = torch.tensor(0.0)
        self.risk_budget = float(config.risk_budget)
        self.target_hardness = float(config.target_hardness)
        self.node_embeddings: torch.Tensor | None = None
        self.projected_retrieval_nodes: torch.Tensor | None = None

    def fit(self, rng: random.Random) -> dict[str, float]:
        representation_metrics = self.fit_representation(rng)
        return {
            **representation_metrics,
            **self._fit_generator(rng),
            "risk_budget": self.risk_budget,
            "target_hardness": self.target_hardness,
            "risk_multiplier": float(self.risk_multiplier),
        }

    def fit_representation(self, rng: random.Random) -> dict[str, float]:
        cache_path = Path(self.config.representation_cache_path) if self.config.representation_cache_path else None
        if cache_path is not None and cache_path.exists():
            payload = torch.load(cache_path, map_location="cpu", weights_only=False)
            self.encoder.load_state_dict(payload["encoder"])
            self.risk.load_state_dict(payload["risk"])
            self.boundary.load_state_dict(payload["boundary"])
            self.risk_budget = float(payload["risk_budget"])
            self.target_hardness = float(payload["target_hardness"])
            self._freeze_representation_modules()
            with torch.no_grad():
                self.node_embeddings, _ = self.encoder()
                self.projected_retrieval_nodes = self.retriever.project_nodes(self.node_embeddings)
            return {**payload["metrics"], "representation_cache_hit": 1.0}

        pretrainer = StrongEncoderPretrainer(self.encoder, projection_dim=self.config.embedding_dim)
        pretraining = pretrain_encoder(
            pretrainer,
            epochs=self.config.encoder_epochs,
            learning_rate=0.001,
            weight_decay=0.0001,
            mask_probability=0.15,
            contrastive_temperature=0.2,
        )
        finetuning = self._finetune_encoder() if self.config.encoder_mode == "strong_finetuned" else {}
        self._freeze_representation_modules(encoder_only=True)
        with torch.no_grad():
            self.node_embeddings, _ = self.encoder()
            self.projected_retrieval_nodes = self.retriever.project_nodes(self.node_embeddings)
        support_metrics = self._fit_support_heads(rng)
        metrics = {
            **{f"pretrain_{key}": float(value) for key, value in pretraining.__dict__.items()},
            **finetuning,
            **support_metrics,
            "representation_cache_hit": 0.0,
        }
        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = cache_path.with_suffix(cache_path.suffix + ".tmp")
            torch.save(
                {
                    "encoder": self.encoder.state_dict(),
                    "risk": self.risk.state_dict(),
                    "boundary": self.boundary.state_dict(),
                    "risk_budget": self.risk_budget,
                    "target_hardness": self.target_hardness,
                    "metrics": metrics,
                },
                temporary,
            )
            temporary.replace(cache_path)
        return metrics

    def _freeze_representation_modules(self, encoder_only: bool = False) -> None:
        modules = (self.encoder,) if encoder_only else (self.encoder, self.risk, self.boundary)
        for module in modules:
            for parameter in module.parameters():
                parameter.requires_grad_(False)
            module.eval()

    def _finetune_encoder(self) -> dict[str, float]:
        head = PositiveSupportRiskEstimator(self.config.embedding_dim, self.config.hidden_dim)
        optimizer = torch.optim.AdamW(
            list(self.encoder.parameters()) + list(head.parameters()),
            lr=0.001,
            weight_decay=0.0001,
        )
        metrics: dict[str, float] = {}
        self.encoder.train()
        for _ in range(self.config.encoder_epochs):
            nodes, _ = self.encoder()
            positive = _mean_edge_representations(self.train_edges, nodes)
            corrupted = _forward_diffusion_corrupted_representations(
                self.train_edges,
                nodes,
                self.diffusion,
                self.config.candidate_size,
                self.config.batch_size,
            )
            loss, metrics = head.loss(positive, corrupted)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        return {
            "encoder_finetune_loss": metrics.get("risk_loss", 0.0),
            "encoder_finetune_accuracy": metrics.get("risk_accuracy", 0.0),
        }

    @torch.no_grad()
    def generate(
        self,
        source_edges: list[Hyperedge],
        positive_edges: set[Hyperedge],
        rng: random.Random,
    ) -> tuple[list[Hyperedge], dict[str, float]]:
        if self.node_embeddings is None:
            raise RuntimeError("fit must be called before generate")
        outputs: list[Hyperedge] = []
        risk_values: list[float] = []
        source_jaccards: list[float] = []
        rejected_known = 0
        for start in range(0, len(source_edges), self.config.batch_size):
            batch = source_edges[start : start + self.config.batch_size]
            padded_ids, padding_mask = _pad_edges(batch)
            edge_nodes = self.node_embeddings[padded_ids]
            source_condition = _masked_mean(edge_nodes, padding_mask.float())
            source_anchor_mask, anchor_condition, _ = self._anchor_condition(
                edge_nodes, padding_mask, source_condition, stochastic=False, rng=rng
            )
            retrieval_indices, _ = self.retriever(
                self.node_embeddings,
                anchor_condition,
                source_condition,
                top_k=max(self.config.candidate_size, max(len(edge) for edge in batch)),
                projected_nodes=self.projected_retrieval_nodes,
            )
            candidate_ids, _, candidate_anchor_mask = _candidate_membership(
                batch, padded_ids, source_anchor_mask, retrieval_indices, self.config.candidate_size, self.num_nodes
            )
            candidate_embeddings = self.node_embeddings[candidate_ids]
            proposals: list[list[tuple[float, float, Hyperedge]]] = [[] for _ in batch]
            for _ in range(max(1, self.config.proposals_per_edge)):
                membership = self._sample_membership(
                    candidate_anchor_mask,
                    candidate_embeddings,
                    anchor_condition,
                    source_condition,
                    torch.tensor([len(edge) for edge in batch]),
                )
                soft_mean = _membership_mean(candidate_embeddings, membership.float())
                support = self.risk(soft_mean).sigmoid()
                hardness = self.boundary(soft_mean).sigmoid()
                for row, source in enumerate(batch):
                    edge = tuple(sorted(candidate_ids[row][membership[row].bool()].unique().tolist()))
                    if edge in positive_edges or edge == source or len(edge) != len(source):
                        rejected_known += int(edge in positive_edges or edge == source)
                        edge = self._repair(source, candidate_ids[row], candidate_anchor_mask[row], positive_edges, rng)
                    proposals[row].append(
                        (float(support[row]), abs(float(hardness[row]) - self.target_hardness), edge)
                    )
            for row, source in enumerate(batch):
                available = proposals[row]
                if self.config.risk_mode == "none":
                    selected = min(available, key=lambda item: item[1])
                else:
                    feasible = [item for item in available if item[0] <= self.risk_budget]
                    selected = min(feasible, key=lambda item: item[1]) if feasible else min(available, key=lambda item: item[0])
                outputs.append(selected[2])
                risk_values.append(selected[0])
                source_jaccards.append(_jaccard(source, selected[2]))
        return outputs, {
            "generated_known_positive_rejection_rate": rejected_known / max(1, len(outputs) * self.config.proposals_per_edge),
            "generated_risk_mean": float(np.mean(risk_values)) if risk_values else 0.0,
            "generated_source_jaccard_mean": float(np.mean(source_jaccards)) if source_jaccards else 0.0,
        }

    def _fit_support_heads(self, rng: random.Random) -> dict[str, float]:
        if self.node_embeddings is None:
            raise RuntimeError("node embeddings unavailable")
        positive = _mean_edge_representations(self.train_edges, self.node_embeddings)
        corrupted = _forward_diffusion_corrupted_representations(
            self.train_edges,
            self.node_embeddings,
            self.diffusion,
            self.config.candidate_size,
            self.config.batch_size,
        )
        optimizer = torch.optim.AdamW(
            list(self.risk.parameters()) + list(self.boundary.parameters()),
            lr=0.001,
            weight_decay=0.0001,
        )
        metrics: dict[str, float] = {}
        for _ in range(self.config.risk_epochs):
            risk_loss, risk_metrics = self.risk.loss(positive, corrupted)
            boundary_loss, boundary_metrics = self.boundary.loss(positive, corrupted)
            optimizer.zero_grad(set_to_none=True)
            (risk_loss + boundary_loss).backward()
            optimizer.step()
            metrics = {
                **risk_metrics,
                "boundary_loss": boundary_metrics["risk_loss"],
                "boundary_accuracy": boundary_metrics["risk_accuracy"],
            }
        with torch.no_grad():
            positive_support = self.risk(positive).sigmoid()
            corrupted_support = self.risk(corrupted).sigmoid()
            positive_boundary = self.boundary(positive).sigmoid()
            corrupted_boundary = self.boundary(corrupted).sigmoid()
        self.risk_budget = _youden_threshold(positive_support, corrupted_support)
        self.target_hardness = _youden_threshold(positive_boundary, corrupted_boundary)
        for module in (self.risk, self.boundary):
            for parameter in module.parameters():
                parameter.requires_grad_(False)
        return {
            **metrics,
            "positive_support_mean": float(positive_support.mean()),
            "corrupted_support_mean": float(corrupted_support.mean()),
            "calibrated_risk_budget": self.risk_budget,
            "calibrated_target_hardness": self.target_hardness,
        }

    def _fit_generator(self, rng: random.Random) -> dict[str, float]:
        if self.node_embeddings is None:
            raise RuntimeError("node embeddings unavailable")
        parameters = list(self.retriever.parameters())
        if self.config.anchor_mode in {"learned_discrete", "dual"}:
            parameters += list(self.anchor.parameters())
        if self.config.diffusion_mode == "d3pm":
            parameters += list(self.diffusion.parameters())
        optimizer = torch.optim.AdamW(parameters, lr=0.001, weight_decay=0.0001)
        metrics: dict[str, float] = {}
        for _ in range(self.config.generator_epochs):
            shuffled = list(self.train_edges)
            rng.shuffle(shuffled)
            rows: list[dict[str, float]] = []
            for start in range(0, len(shuffled), self.config.batch_size):
                batch = shuffled[start : start + self.config.batch_size]
                padded_ids, padding_mask = _pad_edges(batch)
                edge_nodes = self.node_embeddings[padded_ids]
                source_condition = _masked_mean(edge_nodes, padding_mask.float())
                source_anchor_mask, anchor_condition, anchor_semantic = self._anchor_condition(
                    edge_nodes, padding_mask, source_condition, stochastic=True, rng=rng
                )
                retrieval_indices, retrieval_scores = self.retriever(
                    self.node_embeddings,
                    anchor_condition,
                    source_condition,
                    top_k=max(self.config.candidate_size, max(len(edge) for edge in batch)),
                    projected_nodes=self.projected_retrieval_nodes,
                )
                candidate_ids, x_0, candidate_anchor_mask = _candidate_membership(
                    batch, padded_ids, source_anchor_mask, retrieval_indices, self.config.candidate_size, self.num_nodes
                )
                candidate_embeddings = self.node_embeddings[candidate_ids]
                retrieval_loss = -_source_retrieval_scores(retrieval_scores, retrieval_indices, batch).mean()
                semantic_loss = 1.0 - F.cosine_similarity(anchor_semantic, source_condition).mean()
                if self.config.diffusion_mode == "d3pm":
                    d3pm_loss, d3pm_metrics = self.diffusion.training_loss(
                        x_0=x_0,
                        anchor_mask=candidate_anchor_mask,
                        candidate_embeddings=candidate_embeddings,
                        anchor_condition=anchor_condition,
                        source_condition=source_condition,
                        target_hardness=torch.full((len(batch),), self.target_hardness),
                        risk_budget=torch.full((len(batch),), self.risk_budget),
                    )
                    probabilities = self._guided_membership_probabilities(
                        x_0, candidate_anchor_mask, candidate_embeddings, anchor_condition, source_condition
                    )
                else:
                    d3pm_loss = retrieval_loss.new_tensor(0.0)
                    d3pm_metrics = {"d3pm_loss": 0.0, "d3pm_membership_accuracy": 0.0}
                    probabilities = self._non_diffusion_probabilities(
                        candidate_anchor_mask,
                        candidate_embeddings,
                        anchor_condition,
                        source_condition,
                    )
                soft_mean = _membership_mean(candidate_embeddings, probabilities)
                support = self.risk(soft_mean).sigmoid()
                hardness = self.boundary(soft_mean).sigmoid()
                boundary_loss = ((hardness - self.target_hardness) ** 2).mean()
                violation = (support - self.risk_budget).clamp_min(0.0).mean()
                if self.config.risk_mode == "none":
                    risk_penalty = violation.detach() * 0.0
                elif self.config.risk_mode == "fixed":
                    risk_penalty = violation
                else:
                    risk_penalty = self.risk_multiplier * violation
                loss = d3pm_loss + 0.2 * retrieval_loss + 0.2 * semantic_loss + boundary_loss + risk_penalty
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                if self.config.risk_mode == "primal_dual":
                    self.risk_multiplier = primal_dual_update(self.risk_multiplier, support, self.risk_budget, 0.05)
                rows.append(
                    {
                        **d3pm_metrics,
                        "generator_loss": float(loss.detach()),
                        "retrieval_loss": float(retrieval_loss.detach()),
                        "anchor_semantic_loss": float(semantic_loss.detach()),
                        "boundary_loss": float(boundary_loss.detach()),
                        "support_mean": float(support.mean().detach()),
                        "risk_violation": float(violation.detach()),
                    }
                )
            metrics = _mean_metrics(rows)
        return metrics

    def _guided_membership_probabilities(
        self,
        x_0: torch.Tensor,
        anchor_mask: torch.Tensor,
        candidate_embeddings: torch.Tensor,
        anchor_condition: torch.Tensor,
        source_condition: torch.Tensor,
    ) -> torch.Tensor:
        timestep = torch.randint(1, self.config.diffusion_steps + 1, (x_0.shape[0],))
        x_t = self.diffusion.corrupt(x_0, timestep, anchor_mask)
        logits = self.diffusion.denoiser(
            x_t,
            candidate_embeddings,
            timestep,
            anchor_condition,
            source_condition,
            torch.full((x_0.shape[0],), self.target_hardness),
            torch.full((x_0.shape[0],), self.risk_budget),
        )
        return torch.where(anchor_mask.bool(), torch.ones_like(logits), logits.sigmoid())

    def _anchor_condition(
        self,
        edge_nodes: torch.Tensor,
        padding_mask: torch.Tensor,
        source_condition: torch.Tensor,
        *,
        stochastic: bool,
        rng: random.Random,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.config.anchor_mode == "none":
            return torch.zeros_like(padding_mask, dtype=torch.float32), source_condition, source_condition
        if self.config.anchor_mode == "random":
            mask = torch.zeros_like(padding_mask, dtype=torch.float32)
            for row in range(mask.shape[0]):
                valid = padding_mask[row].nonzero(as_tuple=False).squeeze(1).tolist()
                mask[row, rng.sample(valid, max(1, len(valid) // 2))] = 1.0
            condition = _masked_mean(edge_nodes, mask)
            return mask, condition, condition
        output = self.anchor(edge_nodes, padding_mask, temperature=0.7, stochastic=stochastic)
        condition = source_condition if self.config.anchor_mode == "learned_discrete" else output.semantic_condition
        return output.mask, condition, output.semantic_condition

    def _sample_membership(
        self,
        anchor_mask: torch.Tensor,
        candidate_embeddings: torch.Tensor,
        anchor_condition: torch.Tensor,
        source_condition: torch.Tensor,
        cardinality: torch.Tensor,
    ) -> torch.Tensor:
        if self.config.diffusion_mode == "d3pm":
            return self.diffusion.sample(
                anchor_mask=anchor_mask,
                candidate_embeddings=candidate_embeddings,
                anchor_condition=anchor_condition,
                source_condition=source_condition,
                target_hardness=torch.full((anchor_mask.shape[0],), self.target_hardness),
                risk_budget=torch.full((anchor_mask.shape[0],), self.risk_budget),
                target_cardinality=cardinality,
            )
        probabilities = self._non_diffusion_probabilities(
            anchor_mask,
            candidate_embeddings,
            anchor_condition,
            source_condition,
        )
        scores = probabilities.masked_fill(anchor_mask.bool(), float("-inf"))
        output = anchor_mask.long().clone()
        for row in range(output.shape[0]):
            remaining = max(0, int(cardinality[row] - anchor_mask[row].sum()))
            if remaining:
                output[row, scores[row].topk(remaining).indices] = 1
        return output

    def _non_diffusion_probabilities(
        self,
        anchor_mask: torch.Tensor,
        candidate_embeddings: torch.Tensor,
        anchor_condition: torch.Tensor,
        source_condition: torch.Tensor,
    ) -> torch.Tensor:
        source_scores = torch.einsum(
            "bkd,bd->bk",
            F.normalize(candidate_embeddings, dim=2),
            F.normalize(source_condition, dim=1),
        )
        if self.config.diffusion_mode == "none":
            return torch.where(anchor_mask.bool(), torch.ones_like(source_scores), source_scores.sigmoid())

        # Historical ablation: a manually specified annealed energy chain, not a learned diffusion model.
        anchor_scores = torch.einsum(
            "bkd,bd->bk",
            F.normalize(candidate_embeddings, dim=2),
            F.normalize(anchor_condition, dim=1),
        )
        energy = source_scores + anchor_scores
        state = anchor_mask.float()
        for temperature in torch.linspace(1.5, 0.35, self.config.diffusion_steps, device=energy.device):
            probabilities = torch.sigmoid(energy / temperature)
            sampled = torch.bernoulli(probabilities)
            state = torch.where(anchor_mask.bool(), torch.ones_like(sampled), sampled)
            energy = energy + 0.1 * (2.0 * state - 1.0)
        return torch.where(anchor_mask.bool(), torch.ones_like(energy), torch.sigmoid(energy / 0.35))

    def _repair(
        self,
        source: Hyperedge,
        candidate_ids: torch.Tensor,
        anchor_mask: torch.Tensor,
        positive_edges: set[Hyperedge],
        rng: random.Random,
    ) -> Hyperedge:
        anchor_ids = set(candidate_ids[anchor_mask.bool()].tolist())
        kept = list(anchor_ids)
        pool = [int(node) for node in candidate_ids.tolist() if node not in anchor_ids and node not in source]
        rng.shuffle(pool)
        for node in pool:
            kept.append(node)
            if len(set(kept)) == len(source):
                candidate = tuple(sorted(set(kept)))
                if candidate not in positive_edges:
                    return candidate
        while len(set(kept)) < len(source):
            node = rng.randrange(self.num_nodes)
            if node not in kept and node not in source:
                kept.append(node)
        return tuple(sorted(set(kept)))


def _validate_config(config: FormalGeneratorConfig) -> None:
    if config.encoder_mode not in {"strong_frozen", "strong_finetuned"}:
        raise ValueError(f"unknown encoder mode: {config.encoder_mode}")
    if config.anchor_mode not in {"none", "random", "learned_discrete", "dual"}:
        raise ValueError(f"unknown anchor mode: {config.anchor_mode}")
    if config.diffusion_mode not in {"none", "historical_energy", "d3pm"}:
        raise ValueError(f"unknown diffusion mode: {config.diffusion_mode}")
    if config.risk_mode not in {"none", "fixed", "primal_dual"}:
        raise ValueError(f"unknown risk mode: {config.risk_mode}")


def _pad_edges(edges: list[Hyperedge]) -> tuple[torch.Tensor, torch.Tensor]:
    max_size = max(len(edge) for edge in edges)
    ids = torch.zeros((len(edges), max_size), dtype=torch.long)
    mask = torch.zeros((len(edges), max_size), dtype=torch.bool)
    for row, edge in enumerate(edges):
        ids[row, : len(edge)] = torch.tensor(edge)
        mask[row, : len(edge)] = True
    return ids, mask


def _candidate_membership(
    edges: list[Hyperedge],
    padded_ids: torch.Tensor,
    source_anchor_mask: torch.Tensor,
    retrieval_indices: torch.Tensor,
    configured_size: int,
    num_nodes: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    candidate_size = max(configured_size, max(len(edge) for edge in edges))
    candidates, memberships, anchors = [], [], []
    for row, edge in enumerate(edges):
        ordered = list(dict.fromkeys(list(edge) + retrieval_indices[row].tolist()))
        if len(ordered) < candidate_size:
            for node in range(num_nodes):
                if node not in ordered:
                    ordered.append(node)
                if len(ordered) == candidate_size:
                    break
        ordered = ordered[:candidate_size]
        anchor_ids = {
            int(padded_ids[row, position])
            for position in range(len(edge))
            if source_anchor_mask[row, position] >= 0.5
        }
        source_set = set(edge)
        candidates.append(ordered)
        memberships.append([float(node in source_set) for node in ordered])
        anchors.append([float(node in anchor_ids) for node in ordered])
    return torch.tensor(candidates), torch.tensor(memberships), torch.tensor(anchors)


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return (values * mask.unsqueeze(-1)).sum(dim=1) / mask.sum(dim=1).clamp_min(1).unsqueeze(1)


def _membership_mean(candidate_embeddings: torch.Tensor, membership: torch.Tensor) -> torch.Tensor:
    return (candidate_embeddings * membership.unsqueeze(-1)).sum(dim=1) / membership.sum(dim=1).clamp_min(1).unsqueeze(1)


def _mean_edge_representations(edges: list[Hyperedge], nodes: torch.Tensor) -> torch.Tensor:
    return torch.stack([nodes[torch.tensor(edge)].mean(dim=0) for edge in edges])


def _source_retrieval_scores(scores: torch.Tensor, indices: torch.Tensor, edges: list[Hyperedge]) -> torch.Tensor:
    outputs = []
    for row, edge in enumerate(edges):
        source = set(edge)
        mask = torch.tensor([int(node) in source for node in indices[row].tolist()], dtype=torch.bool)
        outputs.append(scores[row][mask].mean() if bool(mask.any()) else scores[row].new_tensor(-1.0))
    return torch.stack(outputs)


def _forward_diffusion_corrupted_representations(
    edges: list[Hyperedge],
    nodes: torch.Tensor,
    diffusion: DiscreteMembershipD3PM,
    configured_candidate_size: int,
    batch_size: int,
) -> torch.Tensor:
    outputs = []
    for start in range(0, len(edges), batch_size):
        batch = edges[start : start + batch_size]
        batch_universe = sorted({node for edge in batch for node in edge})
        candidate_size = min(len(batch_universe), max(configured_candidate_size, max(len(edge) for edge in batch)))
        candidates = []
        memberships = []
        for edge in batch:
            ordered = list(edge) + [node for node in batch_universe if node not in edge]
            row = ordered[:candidate_size]
            source = set(edge)
            candidates.append(row)
            memberships.append([float(node in source) for node in row])
        candidate_ids = torch.tensor(candidates, dtype=torch.long, device=nodes.device)
        x_0 = torch.tensor(memberships, dtype=torch.float32, device=nodes.device)
        timestep = torch.randint(1, diffusion.steps + 1, (len(batch),), device=nodes.device)
        x_t = diffusion.corrupt(x_0, timestep, torch.zeros_like(x_0))
        outputs.append(_membership_mean(nodes[candidate_ids], x_t.float()))
    return torch.cat(outputs, dim=0)


def _youden_threshold(positive_scores: torch.Tensor, negative_scores: torch.Tensor) -> float:
    scores = torch.cat([positive_scores.detach(), negative_scores.detach()])
    labels = torch.cat([torch.ones_like(positive_scores), torch.zeros_like(negative_scores)])
    order = scores.argsort(descending=True)
    sorted_scores = scores[order]
    sorted_labels = labels[order]
    true_positive_rate = sorted_labels.cumsum(0) / max(1, int(positive_scores.numel()))
    false_positive_rate = (1.0 - sorted_labels).cumsum(0) / max(1, int(negative_scores.numel()))
    index = int((true_positive_rate - false_positive_rate).argmax())
    return float(sorted_scores[index].clamp(0.0, 1.0))


def _jaccard(edge_a: Hyperedge, edge_b: Hyperedge) -> float:
    a, b = set(edge_a), set(edge_b)
    return len(a & b) / max(1, len(a | b))


def _mean_metrics(rows: list[dict[str, float]]) -> dict[str, float]:
    if not rows:
        return {}
    return {key: float(np.mean([row[key] for row in rows])) for key in rows[0]}
