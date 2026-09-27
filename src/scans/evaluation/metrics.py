from __future__ import annotations

from statistics import mean
from collections import defaultdict

import numpy as np
from sklearn.metrics import average_precision_score, f1_score, ndcg_score, precision_score, roc_auc_score

from scans.data.hypergraph import Hyperedge
from scans.risk import ClosureRiskIndex, CoWalkRiskIndex, DegreeCorrectedResidualRiskIndex, HittingRiskIndex
from scans.samplers.base import jaccard


def binary_prediction_metrics(labels: list[int], scores: list[float]) -> dict[str, float]:
    if len(set(labels)) < 2:
        raise ValueError("binary metrics require both positive and negative labels")
    predictions = [int(score >= 0.5) for score in scores]
    return {
        "auc": float(roc_auc_score(labels, scores)),
        "aupr": float(average_precision_score(labels, scores)),
        "ndcg": float(ndcg_score([labels], [scores])),
        "mrr": _mean_reciprocal_rank(labels, scores),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
    }


def _mean_reciprocal_rank(labels: list[int], scores: list[float]) -> float:
    ranked = sorted(zip(scores, labels), key=lambda item: item[0], reverse=True)
    reciprocal_ranks = [1.0 / rank for rank, (_, label) in enumerate(ranked, start=1) if label == 1]
    return float(mean(reciprocal_ranks)) if reciprocal_ranks else 0.0


def negative_quality_metrics(
    negative_edges: list[Hyperedge],
    source_edges: list[Hyperedge],
    positive_edges: set[Hyperedge],
    hardness_scores: list[float],
    hard_negative_lower_bound: float = 0.3,
    hard_negative_upper_bound: float = 0.7,
) -> dict[str, float]:
    if len(negative_edges) != len(source_edges):
        raise ValueError("negative_edges and source_edges must have equal length")
    if len(negative_edges) != len(hardness_scores):
        raise ValueError("negative_edges and hardness_scores must have equal length")

    jaccard_values = [jaccard(negative, source) for negative, source in zip(negative_edges, source_edges)]
    positive_index = _PositiveSimilarityIndex(positive_edges)
    nearest_values = [positive_index.nearest(edge) for edge in negative_edges]
    nearest_other_values = [
        positive_index.nearest(edge, excluded_edge=source)
        for edge, source in zip(negative_edges, source_edges)
    ]
    closure_index = ClosureRiskIndex(positive_edges)
    closure_risk_values = [closure_index.risk(edge) for edge in negative_edges]
    cowalk_index = CoWalkRiskIndex(positive_edges)
    cowalk_risk_values = [cowalk_index.risk(edge) for edge in negative_edges]
    hitting_index = HittingRiskIndex(positive_edges)
    hitting_risk_values = [hitting_index.risk(edge) for edge in negative_edges]
    residual_index = DegreeCorrectedResidualRiskIndex(positive_edges)
    residual_risk_values = [residual_index.risk(edge) for edge in negative_edges]
    hard_hits = [
        score
        for score in hardness_scores
        if hard_negative_lower_bound <= score <= hard_negative_upper_bound
    ]
    return {
        "hardness_mean": float(np.mean(hardness_scores)) if hardness_scores else 0.0,
        "hard_negative_ratio": len(hard_hits) / len(hardness_scores) if hardness_scores else 0.0,
        "hard_negative_lower_bound": float(hard_negative_lower_bound),
        "hard_negative_upper_bound": float(hard_negative_upper_bound),
        "jaccard_mean": float(mean(jaccard_values)) if jaccard_values else 0.0,
        "nearest_positive_similarity_mean": float(mean(nearest_values)) if nearest_values else 0.0,
        "nearest_other_positive_similarity_mean": float(mean(nearest_other_values)) if nearest_other_values else 0.0,
        "closure_risk_mean": float(mean(closure_risk_values)) if closure_risk_values else 0.0,
        "cowalk_risk_mean": float(mean(cowalk_risk_values)) if cowalk_risk_values else 0.0,
        "hitting_risk_mean": float(mean(hitting_risk_values)) if hitting_risk_values else 0.0,
        "residual_risk_mean": float(mean(residual_risk_values)) if residual_risk_values else 0.0,
    }


def future_positive_hit_metrics(
    negative_edges: list[Hyperedge],
    future_positive_edges: set[Hyperedge],
    source_count: int,
) -> dict[str, float]:
    if not negative_edges:
        return {
            "future_positive_hit_rate": 0.0,
            "future_positive_hits": 0.0,
            "future_positive_eval_samples": 0.0,
            "future_positive_eval_sources": float(source_count),
        }
    hits = sum(1 for edge in negative_edges if edge in future_positive_edges)
    return {
        "future_positive_hit_rate": hits / len(negative_edges),
        "future_positive_hits": float(hits),
        "future_positive_eval_samples": float(len(negative_edges)),
        "future_positive_eval_sources": float(source_count),
    }


def nearest_positive_similarity_mean(edges: list[Hyperedge], positive_edges: set[Hyperedge]) -> float:
    if not edges:
        return 0.0
    positive_index = _PositiveSimilarityIndex(positive_edges)
    return float(mean(positive_index.nearest(edge) for edge in edges))


def nearest_positive_similarity(edge: Hyperedge, positive_edges: set[Hyperedge]) -> float:
    if not positive_edges:
        return 0.0
    return _PositiveSimilarityIndex(positive_edges).nearest(edge)


class _PositiveSimilarityIndex:
    def __init__(self, positive_edges: set[Hyperedge]) -> None:
        self.positive_edges = list(positive_edges)
        self.by_node: dict[int, list[int]] = defaultdict(list)
        for edge_id, edge in enumerate(self.positive_edges):
            for node in edge:
                self.by_node[node].append(edge_id)

    def nearest(self, edge: Hyperedge, excluded_edge: Hyperedge | None = None) -> float:
        candidate_ids: set[int] = set()
        for node in edge:
            candidate_ids.update(self.by_node.get(node, ()))
        if not candidate_ids:
            return 0.0
        best = 0.0
        for edge_id in candidate_ids:
            positive_edge = self.positive_edges[edge_id]
            if positive_edge == excluded_edge:
                continue
            best = max(best, jaccard(edge, positive_edge))
        return best
