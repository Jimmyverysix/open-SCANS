from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Protocol

from scans.samplers.base import NegativeSampleBatch


class EdgeScorer(Protocol):
    def predict_scores(self, edges: list[tuple[int, ...]]) -> list[float]:
        ...


class EdgeEncoder(Protocol):
    def encode_edges(self, edges: list[tuple[int, ...]]) -> list[list[float]]:
        ...


@dataclass(frozen=True)
class RerankConfig:
    candidate_multiplier: int = 2
    selection_strategy: str = "topk_sample"
    top_k: int = 2
    target_score: float = 0.5
    target_score_min: float = 0.5
    target_score_max: float = 0.5
    score_lower_bound: float = 0.3
    score_upper_bound: float = 0.7
    embedding_similarity_weight: float = 0.05

    @classmethod
    def from_mapping(cls, config: object) -> "RerankConfig":
        mapping = config if isinstance(config, dict) else {}
        target_score = float(mapping.get("rerank_target_score", 0.5))
        target_score_min = float(mapping.get("rerank_target_score_min", target_score))
        target_score_max = float(mapping.get("rerank_target_score_max", target_score))
        return cls(
            candidate_multiplier=max(1, int(mapping.get("rerank_candidate_multiplier", 2))),
            selection_strategy=str(mapping.get("rerank_selection_strategy", "topk_sample")),
            top_k=max(1, int(mapping.get("rerank_top_k", 2))),
            target_score=target_score,
            target_score_min=min(target_score_min, target_score_max),
            target_score_max=max(target_score_min, target_score_max),
            score_lower_bound=float(mapping.get("rerank_score_lower_bound", 0.3)),
            score_upper_bound=float(mapping.get("rerank_score_upper_bound", 0.7)),
            embedding_similarity_weight=float(mapping.get("rerank_embedding_similarity_weight", 0.05)),
        )


@dataclass
class ModelAwareReranker:
    config: RerankConfig

    def select(
        self,
        candidate_batch: NegativeSampleBatch,
        scorer: EdgeScorer,
        num_sources: int,
        negatives_per_positive: int,
        rng: random.Random,
    ) -> NegativeSampleBatch:
        scores = scorer.predict_scores(candidate_batch.edges)
        embedding_similarities = self._embedding_similarities(candidate_batch, scorer)
        selected_indices: list[int] = []
        group_targets: list[float] = []
        group_size = negatives_per_positive * self.config.candidate_multiplier
        for source_index in range(num_sources):
            start = source_index * group_size
            end = start + group_size
            group_indices = list(range(start, end))
            target_score = self._sample_target_score(rng)
            group_targets.append(target_score)
            group_indices = self._sort_group_indices(group_indices, scores, embedding_similarities, target_score)
            selected_indices.extend(
                self._select_group_indices(
                    group_indices=group_indices,
                    negatives_per_positive=negatives_per_positive,
                    rng=rng,
                )
            )

        selected_edges = [candidate_batch.edges[index] for index in selected_indices]
        selected_sources = [candidate_batch.source_edges[index] for index in selected_indices]
        metadata = dict(candidate_batch.metadata)
        metadata["rerank_candidate_multiplier"] = float(self.config.candidate_multiplier)
        metadata["rerank_top_k"] = float(self.config.top_k)
        metadata["rerank_target_score"] = float(self.config.target_score)
        metadata["rerank_target_score_min"] = float(self.config.target_score_min)
        metadata["rerank_target_score_max"] = float(self.config.target_score_max)
        metadata["rerank_sampled_target_score_mean"] = _mean(group_targets)
        metadata["rerank_score_lower_bound"] = float(self.config.score_lower_bound)
        metadata["rerank_score_upper_bound"] = float(self.config.score_upper_bound)
        metadata["rerank_selection_strategy"] = self.config.selection_strategy
        metadata["rerank_selected_score_mean"] = _mean([scores[index] for index in selected_indices])
        metadata["rerank_candidate_score_mean"] = _mean(scores)
        if embedding_similarities is not None:
            metadata["rerank_embedding_similarity_weight"] = float(self.config.embedding_similarity_weight)
            metadata["embedding_source_similarity_mean"] = _mean(
                [embedding_similarities[index] for index in selected_indices]
            )
            metadata["embedding_candidate_source_similarity_mean"] = _mean(embedding_similarities)
        metadata["rerank_selected_in_band_rate"] = _rate(
            [
                self.config.score_lower_bound <= scores[index] <= self.config.score_upper_bound
                for index in selected_indices
            ]
        )
        metadata["rerank_candidate_in_band_rate"] = _rate(
            [self.config.score_lower_bound <= score <= self.config.score_upper_bound for score in scores]
        )
        return NegativeSampleBatch(edges=selected_edges, source_edges=selected_sources, metadata=metadata)

    def _sample_target_score(self, rng: random.Random) -> float:
        if self.config.target_score_min == self.config.target_score_max:
            return self.config.target_score
        return rng.uniform(self.config.target_score_min, self.config.target_score_max)

    def _sort_group_indices(
        self,
        group_indices: list[int],
        scores: list[float],
        embedding_similarities: list[float] | None,
        target_score: float,
    ) -> list[int]:
        if self.config.selection_strategy in {"top_score", "topk_sample"}:
            return sorted(group_indices, key=lambda index: scores[index], reverse=True)
        if self.config.selection_strategy == "boundary_closest":
            lower = self.config.score_lower_bound
            upper = self.config.score_upper_bound
            return sorted(
                group_indices,
                key=lambda index: (
                    not (lower <= scores[index] <= upper),
                    abs(scores[index] - target_score),
                    -scores[index],
                ),
            )
        if self.config.selection_strategy == "embedding_boundary_closest":
            if embedding_similarities is None:
                raise ValueError("embedding_boundary_closest requires scorer.encode_edges")
            lower = self.config.score_lower_bound
            upper = self.config.score_upper_bound
            weight = self.config.embedding_similarity_weight
            return sorted(
                group_indices,
                key=lambda index: (
                    not (lower <= scores[index] <= upper),
                    abs(scores[index] - target_score) - weight * embedding_similarities[index],
                    -embedding_similarities[index],
                    -scores[index],
                ),
            )
        raise ValueError(f"unknown rerank selection strategy: {self.config.selection_strategy}")

    def _select_group_indices(
        self,
        group_indices: list[int],
        negatives_per_positive: int,
        rng: random.Random,
    ) -> list[int]:
        if self.config.selection_strategy == "top_score":
            return group_indices[:negatives_per_positive]
        if self.config.selection_strategy == "topk_sample":
            pool_size = min(len(group_indices), max(negatives_per_positive, self.config.top_k))
            return rng.sample(group_indices[:pool_size], negatives_per_positive)
        if self.config.selection_strategy in {"boundary_closest", "embedding_boundary_closest"}:
            return group_indices[:negatives_per_positive]
        raise ValueError(f"unknown selection strategy: {self.config.selection_strategy}")

    def _embedding_similarities(
        self,
        candidate_batch: NegativeSampleBatch,
        scorer: EdgeScorer,
    ) -> list[float] | None:
        if self.config.selection_strategy != "embedding_boundary_closest":
            return None
        if not hasattr(scorer, "encode_edges"):
            return None
        encoder = scorer  # type: ignore[assignment]
        candidate_embeddings = encoder.encode_edges(candidate_batch.edges)
        source_embeddings = encoder.encode_edges(candidate_batch.source_edges)
        return [
            _cosine_similarity(candidate_embedding, source_embedding)
            for candidate_embedding, source_embedding in zip(candidate_embeddings, source_embeddings)
        ]


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def _rate(values: list[bool]) -> float:
    if not values:
        return 0.0
    return float(sum(1 for value in values if value) / len(values))


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    dot = sum(left_value * right_value for left_value, right_value in zip(left, right))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return float(dot / (left_norm * right_norm))
