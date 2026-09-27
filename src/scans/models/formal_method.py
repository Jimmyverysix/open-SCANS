from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class AnchorOutput:
    mask: torch.Tensor
    cardinality: torch.Tensor
    semantic_condition: torch.Tensor
    node_logits: torch.Tensor
    cardinality_logits: torch.Tensor


class LearnedDualAnchor(nn.Module):
    def __init__(self, embedding_dim: int, hidden_dim: int, max_edge_size: int) -> None:
        super().__init__()
        self.max_edge_size = int(max_edge_size)
        self.node_scorer = nn.Sequential(
            nn.Linear(embedding_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.cardinality_head = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, max_edge_size - 1),
        )

    def forward(
        self,
        edge_node_embeddings: torch.Tensor,
        edge_padding_mask: torch.Tensor,
        *,
        temperature: float = 1.0,
        stochastic: bool = True,
    ) -> AnchorOutput:
        valid = edge_padding_mask.bool()
        counts = valid.sum(dim=1)
        pooled = (edge_node_embeddings * valid.unsqueeze(-1)).sum(dim=1) / counts.clamp_min(1).unsqueeze(1)
        context = pooled.unsqueeze(1).expand_as(edge_node_embeddings)
        node_logits = self.node_scorer(torch.cat([edge_node_embeddings, context], dim=-1)).squeeze(-1)
        node_logits = node_logits.masked_fill(~valid, float("-inf"))

        cardinality_logits = self.cardinality_head(pooled)
        card_positions = torch.arange(1, self.max_edge_size, device=pooled.device).unsqueeze(0)
        valid_cards = card_positions < counts.unsqueeze(1)
        cardinality_logits = cardinality_logits.masked_fill(~valid_cards, float("-inf"))
        if stochastic:
            card_one_hot = F.gumbel_softmax(cardinality_logits, tau=temperature, hard=True)
            differentiable_cardinality = (card_one_hot * card_positions).sum(dim=1)
            cardinality = differentiable_cardinality.detach().long()
            ranking_logits = node_logits + _gumbel_like(node_logits)
        else:
            cardinality = cardinality_logits.argmax(dim=1) + 1
            differentiable_cardinality = cardinality.float()
            ranking_logits = node_logits

        hard_mask = torch.zeros_like(node_logits)
        for batch_index in range(node_logits.shape[0]):
            k = int(cardinality[batch_index].item())
            selected = ranking_logits[batch_index].topk(k).indices
            hard_mask[batch_index, selected] = 1.0
        soft_scores = torch.sigmoid(node_logits / max(float(temperature), 1e-6)) * valid
        soft_mask = soft_scores * differentiable_cardinality.unsqueeze(1) / soft_scores.sum(dim=1, keepdim=True).clamp_min(1e-6)
        mask = hard_mask + soft_mask - soft_mask.detach()
        anchor_semantic = (edge_node_embeddings * mask.unsqueeze(-1)).sum(dim=1) / mask.sum(dim=1).clamp_min(1).unsqueeze(1)
        semantic = 0.5 * (anchor_semantic + pooled)
        return AnchorOutput(mask, cardinality, semantic, node_logits, cardinality_logits)


class LearnedCandidateRetriever(nn.Module):
    def __init__(self, embedding_dim: int) -> None:
        super().__init__()
        self.node_projection = nn.Linear(embedding_dim, embedding_dim, bias=False)
        with torch.no_grad():
            self.node_projection.weight.copy_(torch.eye(embedding_dim))
        self.node_projection.weight.requires_grad_(False)
        self.condition_projection = nn.Linear(embedding_dim * 2, embedding_dim, bias=False)

    def project_nodes(self, node_embeddings: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.node_projection(node_embeddings), dim=1)

    def forward(
        self,
        node_embeddings: torch.Tensor,
        anchor_condition: torch.Tensor,
        source_condition: torch.Tensor,
        *,
        top_k: int,
        forbidden_mask: torch.Tensor | None = None,
        chunk_size: int = 32_768,
        projected_nodes: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        query = F.normalize(self.condition_projection(torch.cat([anchor_condition, source_condition], dim=1)), dim=1)
        nodes = self.project_nodes(node_embeddings) if projected_nodes is None else projected_nodes
        kept = min(top_k, nodes.shape[0])
        best_values = query.new_full((query.shape[0], kept), float("-inf"))
        best_indices = torch.zeros((query.shape[0], kept), device=query.device, dtype=torch.long)
        for start in range(0, nodes.shape[0], chunk_size):
            stop = min(start + chunk_size, nodes.shape[0])
            scores = query @ nodes[start:stop].T
            if forbidden_mask is not None:
                scores = scores.masked_fill(forbidden_mask[:, start:stop], float("-inf"))
            chunk_indices = torch.arange(start, stop, device=query.device).unsqueeze(0).expand(query.shape[0], -1)
            merged_values = torch.cat([best_values, scores], dim=1)
            merged_indices = torch.cat([best_indices, chunk_indices], dim=1)
            best_values, positions = merged_values.topk(kept, dim=1)
            best_indices = merged_indices.gather(1, positions)
        return best_indices, best_values


class MembershipDenoiser(nn.Module):
    def __init__(self, embedding_dim: int, hidden_dim: int, max_steps: int) -> None:
        super().__init__()
        self.time_embedding = nn.Embedding(max_steps + 1, embedding_dim)
        self.state_embedding = nn.Embedding(2, embedding_dim)
        self.network = nn.Sequential(
            nn.Linear(embedding_dim * 4 + 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        x_t: torch.Tensor,
        candidate_embeddings: torch.Tensor,
        timestep: torch.Tensor,
        anchor_condition: torch.Tensor,
        source_condition: torch.Tensor,
        target_hardness: torch.Tensor,
        risk_budget: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, candidate_count = x_t.shape
        time = self.time_embedding(timestep).unsqueeze(1).expand(-1, candidate_count, -1)
        state = self.state_embedding(x_t.long())
        anchor = anchor_condition.unsqueeze(1).expand(-1, candidate_count, -1)
        source = source_condition.unsqueeze(1).expand(-1, candidate_count, -1)
        scalars = torch.stack([target_hardness, risk_budget], dim=1).unsqueeze(1).expand(-1, candidate_count, -1)
        features = torch.cat([candidate_embeddings, state, time, anchor + source, scalars], dim=-1)
        return self.network(features).squeeze(-1)


class DiscreteMembershipD3PM(nn.Module):
    def __init__(
        self,
        denoiser: MembershipDenoiser,
        *,
        steps: int,
        beta_start: float = 0.02,
        beta_end: float = 0.35,
    ) -> None:
        super().__init__()
        self.denoiser = denoiser
        self.steps = int(steps)
        betas = torch.zeros(steps + 1)
        betas[1:] = torch.linspace(beta_start, beta_end, steps)
        transitions = torch.eye(2).repeat(steps + 1, 1, 1)
        for step in range(1, steps + 1):
            beta = betas[step]
            transitions[step] = torch.tensor(
                [[1.0 - beta / 2.0, beta / 2.0], [beta / 2.0, 1.0 - beta / 2.0]]
            )
        cumulative = torch.eye(2).repeat(steps + 1, 1, 1)
        for step in range(1, steps + 1):
            cumulative[step] = cumulative[step - 1] @ transitions[step]
        self.register_buffer("betas", betas)
        self.register_buffer("transitions", transitions)
        self.register_buffer("cumulative_transitions", cumulative)

    def corrupt(
        self,
        x_0: torch.Tensor,
        timestep: torch.Tensor,
        anchor_mask: torch.Tensor,
    ) -> torch.Tensor:
        x_0_distribution = F.one_hot(x_0.long(), num_classes=2).float()
        cumulative = self.cumulative_transitions[timestep]
        probabilities = torch.einsum("bki,bij->bkj", x_0_distribution, cumulative)
        corrupted = torch.bernoulli(probabilities[..., 1])
        return torch.where(anchor_mask.bool(), torch.ones_like(corrupted), corrupted).long()

    def training_loss(
        self,
        x_0: torch.Tensor,
        anchor_mask: torch.Tensor,
        candidate_embeddings: torch.Tensor,
        anchor_condition: torch.Tensor,
        source_condition: torch.Tensor,
        target_hardness: torch.Tensor,
        risk_budget: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        timestep = torch.randint(1, self.steps + 1, (x_0.shape[0],), device=x_0.device)
        x_t = self.corrupt(x_0, timestep, anchor_mask)
        logits = self.denoiser(
            x_t,
            candidate_embeddings,
            timestep,
            anchor_condition,
            source_condition,
            target_hardness,
            risk_budget,
        )
        non_anchor = ~anchor_mask.bool()
        x_0_distribution = torch.stack([1.0 - logits.sigmoid(), logits.sigmoid()], dim=-1)
        predicted_reverse = self.reverse_posterior(x_t, x_0_distribution, timestep)
        true_reverse = self.reverse_posterior(
            x_t,
            F.one_hot(x_0.long(), num_classes=2).float(),
            timestep,
        )
        reverse_loss = -(true_reverse * predicted_reverse.clamp_min(1e-8).log()).sum(dim=-1)[non_anchor].mean()
        x_0_loss = F.binary_cross_entropy_with_logits(logits[non_anchor], x_0[non_anchor].float())
        loss = reverse_loss + x_0_loss
        accuracy = ((logits.sigmoid() >= 0.5) == x_0.bool())[non_anchor].float().mean()
        return loss, {
            "d3pm_loss": float(loss.detach()),
            "d3pm_reverse_loss": float(reverse_loss.detach()),
            "d3pm_x0_loss": float(x_0_loss.detach()),
            "d3pm_membership_accuracy": float(accuracy.detach()),
        }

    def reverse_posterior(
        self,
        x_t: torch.Tensor,
        x_0_distribution: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        transition = self.transitions[timestep]
        cumulative_previous = self.cumulative_transitions[(timestep - 1).clamp_min(0)]
        x_t_one_hot = F.one_hot(x_t.long(), num_classes=2).float()
        likelihood = torch.einsum("bij,bkj->bki", transition, x_t_one_hot)
        unnormalized = cumulative_previous.unsqueeze(1) * likelihood.unsqueeze(2)
        posterior_by_x0 = unnormalized / unnormalized.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        posterior = (posterior_by_x0 * x_0_distribution.unsqueeze(-1)).sum(dim=2)
        return posterior / posterior.sum(dim=-1, keepdim=True).clamp_min(1e-8)

    @torch.no_grad()
    def sample(
        self,
        anchor_mask: torch.Tensor,
        candidate_embeddings: torch.Tensor,
        anchor_condition: torch.Tensor,
        source_condition: torch.Tensor,
        target_hardness: torch.Tensor,
        risk_budget: torch.Tensor,
        target_cardinality: torch.Tensor,
    ) -> torch.Tensor:
        state = torch.bernoulli(torch.full(anchor_mask.shape, 0.5, device=anchor_mask.device)).long()
        state = torch.where(anchor_mask.bool(), torch.ones_like(state), state)
        last_logits = None
        for step in range(self.steps, 0, -1):
            timestep = torch.full((state.shape[0],), step, device=state.device, dtype=torch.long)
            last_logits = self.denoiser(
                state,
                candidate_embeddings,
                timestep,
                anchor_condition,
                source_condition,
                target_hardness,
                risk_budget,
            )
            x_0_distribution = torch.stack([1.0 - last_logits.sigmoid(), last_logits.sigmoid()], dim=-1)
            reverse = self.reverse_posterior(state, x_0_distribution, timestep)
            state = torch.bernoulli(reverse[..., 1]).long()
            state = torch.where(anchor_mask.bool(), torch.ones_like(state), state)
        if last_logits is None:
            raise RuntimeError("D3PM requires at least one step")
        output = anchor_mask.long().clone()
        for batch_index in range(output.shape[0]):
            remaining = max(0, int(target_cardinality[batch_index].item() - anchor_mask[batch_index].sum().item()))
            if remaining == 0:
                continue
            scores = last_logits[batch_index].masked_fill(anchor_mask[batch_index].bool(), float("-inf"))
            output[batch_index, scores.topk(min(remaining, scores.numel())).indices] = 1
        return output


class PositiveSupportRiskEstimator(nn.Module):
    def __init__(self, edge_representation_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(edge_representation_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, edge_representations: torch.Tensor) -> torch.Tensor:
        return self.network(edge_representations).squeeze(-1)

    def loss(self, positive: torch.Tensor, corrupted: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
        pos_logits = self(positive)
        neg_logits = self(corrupted)
        logits = torch.cat([pos_logits, neg_logits])
        labels = torch.cat([torch.ones_like(pos_logits), torch.zeros_like(neg_logits)])
        loss = F.binary_cross_entropy_with_logits(logits, labels)
        accuracy = ((logits.sigmoid() >= 0.5) == labels.bool()).float().mean()
        return loss, {"risk_loss": float(loss.detach()), "risk_accuracy": float(accuracy.detach())}


def primal_dual_update(multiplier: torch.Tensor, observed_risk: torch.Tensor, budget: float, learning_rate: float) -> torch.Tensor:
    return (multiplier + learning_rate * (observed_risk.detach().mean() - float(budget))).clamp_min(0.0)


def _gumbel_like(tensor: torch.Tensor) -> torch.Tensor:
    uniform = torch.rand_like(tensor).clamp_(1e-6, 1 - 1e-6)
    return -torch.log(-torch.log(uniform))
