from __future__ import annotations

import math
import random
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from itertools import combinations
from typing import NamedTuple, Protocol

from scans.data.hypergraph import Hyperedge
from scans.risk import ClosureRiskIndex, CoWalkRiskIndex, DegreeCorrectedResidualRiskIndex, HittingRiskIndex
from scans.samplers.anchored_safe_sampler import _PositiveEdgeIndex
from scans.samplers.base import NegativeSampleBatch


class EdgeScorer(Protocol):
    def predict_scores(self, edges: list[Hyperedge]) -> list[float]:
        ...


@dataclass
class RiskControlledSampler:
    anchor_ratio: float = 0.25
    nearest_positive_upper_bound: float = 0.25
    closure_risk_upper_bound: float = 0.35
    use_cowalk_risk: bool = False
    cowalk_risk_upper_bound: float = 0.35
    use_hitting_risk: bool = False
    hitting_risk_upper_bound: float = 0.35
    use_residual_risk: bool = False
    residual_risk_upper_bound: float = 0.35
    use_risk_budget: bool = False
    risk_budget: float = 3.0
    nearest_risk_weight: float = 1.0
    closure_risk_weight: float = 1.0
    cowalk_risk_weight: float = 1.0
    hitting_risk_weight: float = 1.0
    residual_risk_weight: float = 1.0
    use_budget_safety_caps: bool = False
    safety_nearest_positive_upper_bound: float = float("inf")
    safety_closure_risk_upper_bound: float = float("inf")
    safety_cowalk_risk_upper_bound: float = float("inf")
    safety_hitting_risk_upper_bound: float = float("inf")
    safety_residual_risk_upper_bound: float = float("inf")
    replacement_strategy: str = "random"
    neighbor_sample_probability: float = 0.8
    max_replacement_neighbors_per_node: int = 128
    max_replacement_pool_size: int = 512
    max_replacement_pairs_per_edge: int = 2000
    max_closure_pairs_per_edge: int = 0
    max_cowalk_neighbors: int = 128
    max_hitting_neighbors: int = 128
    hitting_two_step_weight: float = 0.5
    max_risk_cache_size: int = 200_000
    max_candidate_pool_cache_size: int = 100_000
    max_anchor_node_risk_cache_size: int = 200_000
    max_attempts: int = 300
    boundary_guided_probe_count: int = 16
    boundary_guided_max_rounds: int = 12
    boundary_guided_min_feasible_candidates: int = 4
    boundary_guided_elite_enabled: bool = False
    boundary_guided_elite_fraction: float = 0.35
    boundary_guided_elite_mix_probability: float = 0.70
    boundary_guided_elite_risk_penalty: float = 0.10
    risk_aware_pool_multiplier: int = 4
    risk_aware_nearest_weight: float = 0.0
    risk_aware_nearest_prefilter_multiplier: int = 4
    mixture_risk_aware_probability: float = 0.0
    mixture_residual_safe_probability: float = 0.0
    residual_safe_pool_multiplier: int = 4
    residual_safe_structural_quantile: float = 0.50
    residual_safe_residual_weight: float = 2.0
    residual_safe_random_pool_multiplier: int = 4
    diffusion_steps: int = 4
    diffusion_pool_multiplier: int = 6
    diffusion_temperature_start: float = 1.5
    diffusion_temperature_end: float = 0.35
    diffusion_structural_weight: float = 1.0
    diffusion_risk_weight: float = 1.0
    diffusion_random_pool_multiplier: int = 4
    primal_dual_enabled: bool = False
    primal_dual_learning_rate: float = 0.05
    primal_dual_max_lambda: float = 10.0
    primal_dual_hardness_scale: float = 1.0

    name: str = "risk_controlled"
    _cached_positive_edges_id: int | None = field(default=None, init=False, repr=False)
    _cached_positive_edges_len: int = field(default=0, init=False, repr=False)
    _cached_positive_index: _PositiveEdgeIndex | None = field(default=None, init=False, repr=False)
    _cached_closure_index: ClosureRiskIndex | None = field(default=None, init=False, repr=False)
    _cached_cowalk_index: CoWalkRiskIndex | None = field(default=None, init=False, repr=False)
    _cached_hitting_index: HittingRiskIndex | None = field(default=None, init=False, repr=False)
    _cached_residual_index: DegreeCorrectedResidualRiskIndex | None = field(default=None, init=False, repr=False)
    _cached_replacement_index: "_ReplacementNeighborhoodIndex | None" = field(default=None, init=False, repr=False)
    _risk_cache: dict[tuple[Hyperedge, Hyperedge], tuple[float, float, float, float, float]] = field(default_factory=dict, init=False, repr=False)
    _anchor_node_risk_cache: dict[tuple[object, ...], float] = field(default_factory=dict, init=False, repr=False)
    _dual_lambdas: dict[str, float] = field(default_factory=dict, init=False, repr=False)

    def sample(
        self,
        source_edges: list[Hyperedge],
        num_nodes: int,
        positive_edges: set[Hyperedge],
        rng: random.Random,
        negatives_per_positive: int,
    ) -> NegativeSampleBatch:
        negatives: list[Hyperedge] = []
        sources: list[Hyperedge] = []
        stats = _SamplingStats()
        positive_index, closure_index, cowalk_index, hitting_index, residual_index, replacement_index = self._indices_for(positive_edges)
        for source_edge in source_edges:
            for _ in range(negatives_per_positive):
                sample = self._sample_one(
                    source_edge,
                    num_nodes,
                    positive_edges,
                    positive_index,
                    closure_index,
                    cowalk_index,
                    hitting_index,
                    residual_index,
                    replacement_index,
                    rng,
                )
                negatives.append(sample.edge)
                sources.append(source_edge)
                stats.add(sample)
        return NegativeSampleBatch(edges=negatives, source_edges=sources, metadata=stats.to_metadata())

    def sample_boundary_guided(
        self,
        source_edges: list[Hyperedge],
        num_nodes: int,
        positive_edges: set[Hyperedge],
        scorer: EdgeScorer,
        rng: random.Random,
        negatives_per_positive: int,
        target_score: float = 0.5,
        target_score_min: float = 0.5,
        target_score_max: float = 0.5,
        score_lower_bound: float = 0.3,
        score_upper_bound: float = 0.7,
    ) -> NegativeSampleBatch:
        negatives: list[Hyperedge] = []
        sources: list[Hyperedge] = []
        stats = _SamplingStats()
        positive_index, closure_index, cowalk_index, hitting_index, residual_index, replacement_index = self._indices_for(positive_edges)
        for source_edge in source_edges:
            for _ in range(negatives_per_positive):
                sample = self._sample_one_boundary_guided(
                    source_edge=source_edge,
                    num_nodes=num_nodes,
                    positive_edges=positive_edges,
                    positive_index=positive_index,
                    closure_index=closure_index,
                    cowalk_index=cowalk_index,
                    hitting_index=hitting_index,
                    residual_index=residual_index,
                    replacement_index=replacement_index,
                    scorer=scorer,
                    rng=rng,
                    target_score=_sample_target_score(rng, target_score, target_score_min, target_score_max),
                    score_lower_bound=score_lower_bound,
                    score_upper_bound=score_upper_bound,
                )
                negatives.append(sample.edge)
                sources.append(source_edge)
                stats.add(sample)
        metadata = stats.to_metadata()
        metadata["adversarial_proposal_enabled"] = 1.0
        metadata["adversarial_probe_count"] = float(self.boundary_guided_probe_count)
        metadata["adversarial_max_rounds"] = float(self.boundary_guided_max_rounds)
        metadata["adversarial_elite_enabled"] = 1.0 if self.boundary_guided_elite_enabled else 0.0
        return NegativeSampleBatch(edges=negatives, source_edges=sources, metadata=metadata)

    def _indices_for(
        self,
        positive_edges: set[Hyperedge],
    ) -> tuple[_PositiveEdgeIndex, ClosureRiskIndex, CoWalkRiskIndex | None, HittingRiskIndex | None, DegreeCorrectedResidualRiskIndex | None, "_ReplacementNeighborhoodIndex | None"]:
        cache_hit = (
            self._cached_positive_edges_id == id(positive_edges)
            and self._cached_positive_edges_len == len(positive_edges)
            and self._cached_positive_index is not None
            and self._cached_closure_index is not None
        )
        if not cache_hit:
            self._cached_positive_edges_id = id(positive_edges)
            self._cached_positive_edges_len = len(positive_edges)
            self._cached_positive_index = _PositiveEdgeIndex(positive_edges)
            self._cached_closure_index = ClosureRiskIndex(
                positive_edges,
                max_pairs_per_edge=self.max_closure_pairs_per_edge,
            )
            self._cached_cowalk_index = None
            self._cached_hitting_index = None
            self._cached_residual_index = None
            self._cached_replacement_index = None
            self._risk_cache.clear()
            self._anchor_node_risk_cache.clear()

        if self.use_cowalk_risk and self._cached_cowalk_index is None:
            self._cached_cowalk_index = CoWalkRiskIndex(
                positive_edges,
                max_neighbors_per_node=self.max_cowalk_neighbors,
            )
        if self.use_hitting_risk and self._cached_hitting_index is None:
            self._cached_hitting_index = HittingRiskIndex(
                positive_edges,
                max_neighbors_per_node=self.max_hitting_neighbors,
                two_step_weight=self.hitting_two_step_weight,
            )
        if self.use_residual_risk and self._cached_residual_index is None:
            self._cached_residual_index = DegreeCorrectedResidualRiskIndex(
                positive_edges,
                max_pairs_per_edge=self.max_replacement_pairs_per_edge,
            )
        if self.replacement_strategy in {
            "neighborhood_mixed",
            "risk_aware_mixed",
            "residual_safe_mixed",
            "diffusion_mixed",
            "calibrated_mixture",
        } and self._cached_replacement_index is None:
            self._cached_replacement_index = _ReplacementNeighborhoodIndex(
                positive_edges,
                max_neighbors_per_node=self.max_replacement_neighbors_per_node,
                max_pairs_per_edge=self.max_replacement_pairs_per_edge,
                max_pool_cache_size=self.max_candidate_pool_cache_size,
            )

        if self._cached_positive_index is None or self._cached_closure_index is None:
            raise RuntimeError("risk index cache was not initialized")
        return (
            self._cached_positive_index,
            self._cached_closure_index,
            self._cached_cowalk_index,
            self._cached_hitting_index,
            self._cached_residual_index,
            self._cached_replacement_index,
        )

    def _sample_one(
        self,
        source_edge: Hyperedge,
        num_nodes: int,
        positive_edges: set[Hyperedge],
        positive_index: _PositiveEdgeIndex,
        closure_index: ClosureRiskIndex,
        cowalk_index: CoWalkRiskIndex | None,
        hitting_index: HittingRiskIndex | None,
        residual_index: DegreeCorrectedResidualRiskIndex | None,
        replacement_index: "_ReplacementNeighborhoodIndex | None",
        rng: random.Random,
    ) -> "_RiskSample":
        if len(source_edge) < 2:
            raise ValueError("risk-controlled sampling requires source hyperedges with size >= 2")

        anchor_size = self._anchor_size(len(source_edge))
        source_set = set(source_edge)
        random_replacement_pool = range(num_nodes)
        replacement_size = len(source_edge) - anchor_size

        best_edge: Hyperedge | None = None
        best_score = float("inf")
        attempts = 0
        valid_candidates = 0
        nearest_pass_candidates = 0
        closure_pass_candidates = 0
        cowalk_pass_candidates = 0
        hitting_pass_candidates = 0
        residual_pass_candidates = 0
        all_pass_candidates = 0
        budget_pass_candidates = 0
        emitted_combined_risk = 0.0
        candidate_pool_size_sum = 0
        neighbor_replacement_count = 0
        risk_aware_replacement_count = 0
        residual_safe_replacement_count = 0
        diffusion_replacement_count = 0
        total_replacement_count = 0
        risk_cache_hits = 0
        risk_cache_lookups = 0
        rejection_stats = _RejectionStats()
        for _ in range(self.max_attempts):
            attempts += 1
            anchor_nodes = rng.sample(source_edge, anchor_size)
            replacements, candidate_pool_size, neighbor_hits, proposal_strategy = self._sample_replacements(
                source_edge=source_edge,
                anchor_nodes=anchor_nodes,
                source_set=source_set,
                random_replacement_pool=random_replacement_pool,
                replacement_size=replacement_size,
                num_nodes=num_nodes,
                positive_index=positive_index,
                replacement_index=replacement_index,
                closure_index=closure_index,
                cowalk_index=cowalk_index,
                hitting_index=hitting_index,
                residual_index=residual_index,
                rng=rng,
            )
            candidate_pool_size_sum += candidate_pool_size
            neighbor_replacement_count += neighbor_hits
            if proposal_strategy == "risk_aware_mixed":
                risk_aware_replacement_count += len(replacements)
            if proposal_strategy == "residual_safe_mixed":
                residual_safe_replacement_count += len(replacements)
            if proposal_strategy == "diffusion_mixed":
                diffusion_replacement_count += len(replacements)
            total_replacement_count += len(replacements)
            edge = tuple(sorted(set(anchor_nodes) | set(replacements)))
            if len(edge) != len(source_edge):
                continue
            if edge in positive_edges:
                continue

            nearest, closure_risk, cowalk_risk, hitting_risk, residual_risk, cache_hit = self._risk_values(
                edge=edge,
                source_edge=source_edge,
                positive_index=positive_index,
                closure_index=closure_index,
                cowalk_index=cowalk_index,
                hitting_index=hitting_index,
                residual_index=residual_index,
            )
            risk_cache_lookups += 1
            if cache_hit:
                risk_cache_hits += 1
            valid_candidates += 1
            nearest_pass = nearest <= self.nearest_positive_upper_bound
            closure_pass = closure_risk <= self.closure_risk_upper_bound
            cowalk_pass = cowalk_index is None or cowalk_risk <= self.cowalk_risk_upper_bound
            hitting_pass = hitting_index is None or hitting_risk <= self.hitting_risk_upper_bound
            residual_pass = residual_index is None or residual_risk <= self.residual_risk_upper_bound
            if nearest_pass:
                nearest_pass_candidates += 1
            if closure_pass:
                closure_pass_candidates += 1
            if cowalk_pass:
                cowalk_pass_candidates += 1
            if hitting_pass:
                hitting_pass_candidates += 1
            if residual_pass:
                residual_pass_candidates += 1
            if nearest_pass and closure_pass and cowalk_pass and hitting_pass and residual_pass:
                all_pass_candidates += 1
            combined_risk = self._combined_risk(nearest, closure_risk, cowalk_risk, hitting_risk, residual_risk)
            budget_pass = combined_risk <= self.risk_budget
            safety_pass = self._safety_pass(nearest, closure_risk, cowalk_risk, hitting_risk, residual_risk)
            if budget_pass:
                budget_pass_candidates += 1
            score = combined_risk if self.use_risk_budget else nearest + closure_risk + cowalk_risk + hitting_risk + residual_risk
            if score < best_score:
                best_edge = edge
                best_score = score
                emitted_combined_risk = combined_risk
            accepted = budget_pass and safety_pass if self.use_risk_budget else nearest_pass and closure_pass and cowalk_pass and hitting_pass and residual_pass
            rejection_stats.add(
                _primary_rejection_reason(
                    nearest=nearest,
                    closure_risk=closure_risk,
                    cowalk_risk=cowalk_risk,
                    hitting_risk=hitting_risk,
                    residual_risk=residual_risk,
                    combined_risk=combined_risk,
                    nearest_pass=nearest_pass,
                    closure_pass=closure_pass,
                    cowalk_pass=cowalk_pass,
                    hitting_pass=hitting_pass,
                    residual_pass=residual_pass,
                    budget_pass=budget_pass,
                    safety_pass=safety_pass,
                    use_risk_budget=self.use_risk_budget,
                    use_cowalk_risk=self.use_cowalk_risk,
                    use_hitting_risk=self.use_hitting_risk,
                    use_residual_risk=self.use_residual_risk,
                    nearest_bound=self.nearest_positive_upper_bound,
                    closure_bound=self.closure_risk_upper_bound,
                    cowalk_bound=self.cowalk_risk_upper_bound,
                    hitting_bound=self.hitting_risk_upper_bound,
                    residual_bound=self.residual_risk_upper_bound,
                    budget_bound=self.risk_budget,
                )
            )
            if accepted:
                return _RiskSample(
                    edge=edge,
                    attempts=attempts,
                    valid_candidates=valid_candidates,
                    nearest_pass_candidates=nearest_pass_candidates,
                    closure_pass_candidates=closure_pass_candidates,
                    cowalk_pass_candidates=cowalk_pass_candidates,
                    hitting_pass_candidates=hitting_pass_candidates,
                    residual_pass_candidates=residual_pass_candidates,
                    all_pass_candidates=all_pass_candidates,
                    budget_pass_candidates=budget_pass_candidates,
                    emitted_combined_risk=combined_risk,
                    candidate_pool_size_mean=candidate_pool_size_sum / attempts,
                    neighbor_replacement_rate=_safe_ratio(neighbor_replacement_count, total_replacement_count),
                    risk_aware_replacement_rate=_safe_ratio(risk_aware_replacement_count, total_replacement_count),
                    residual_safe_replacement_rate=_safe_ratio(residual_safe_replacement_count, total_replacement_count),
                    diffusion_replacement_rate=_safe_ratio(diffusion_replacement_count, total_replacement_count),
                    risk_cache_hit_rate=_safe_ratio(risk_cache_hits, risk_cache_lookups),
                    used_fallback=False,
                    nearest_primary_rejections=rejection_stats.nearest,
                    closure_primary_rejections=rejection_stats.closure,
                    cowalk_primary_rejections=rejection_stats.cowalk,
                    hitting_primary_rejections=rejection_stats.hitting,
                    residual_primary_rejections=rejection_stats.residual,
                    budget_primary_rejections=rejection_stats.budget,
                    safety_primary_rejections=rejection_stats.safety,
                )

        if best_edge is not None:
            return _RiskSample(
                edge=best_edge,
                attempts=attempts,
                valid_candidates=valid_candidates,
                nearest_pass_candidates=nearest_pass_candidates,
                closure_pass_candidates=closure_pass_candidates,
                cowalk_pass_candidates=cowalk_pass_candidates,
                hitting_pass_candidates=hitting_pass_candidates,
                residual_pass_candidates=residual_pass_candidates,
                all_pass_candidates=all_pass_candidates,
                budget_pass_candidates=budget_pass_candidates,
                emitted_combined_risk=emitted_combined_risk,
                candidate_pool_size_mean=candidate_pool_size_sum / attempts,
                neighbor_replacement_rate=_safe_ratio(neighbor_replacement_count, total_replacement_count),
                risk_aware_replacement_rate=_safe_ratio(risk_aware_replacement_count, total_replacement_count),
                residual_safe_replacement_rate=_safe_ratio(residual_safe_replacement_count, total_replacement_count),
                diffusion_replacement_rate=_safe_ratio(diffusion_replacement_count, total_replacement_count),
                risk_cache_hit_rate=_safe_ratio(risk_cache_hits, risk_cache_lookups),
                used_fallback=True,
                nearest_primary_rejections=rejection_stats.nearest,
                closure_primary_rejections=rejection_stats.closure,
                cowalk_primary_rejections=rejection_stats.cowalk,
                hitting_primary_rejections=rejection_stats.hitting,
                residual_primary_rejections=rejection_stats.residual,
                budget_primary_rejections=rejection_stats.budget,
                safety_primary_rejections=rejection_stats.safety,
            )
        fallback = self._sample_audited_random_fallback(
            source_edge=source_edge,
            num_nodes=num_nodes,
            positive_edges=positive_edges,
            positive_index=positive_index,
            closure_index=closure_index,
            cowalk_index=cowalk_index,
            hitting_index=hitting_index,
            residual_index=residual_index,
            rng=rng,
        )
        if fallback is not None:
            return fallback
        raise RuntimeError("failed to sample a risk-controlled negative hyperedge")

    def _sample_one_boundary_guided(
        self,
        source_edge: Hyperedge,
        num_nodes: int,
        positive_edges: set[Hyperedge],
        positive_index: _PositiveEdgeIndex,
        closure_index: ClosureRiskIndex,
        cowalk_index: CoWalkRiskIndex | None,
        hitting_index: HittingRiskIndex | None,
        residual_index: DegreeCorrectedResidualRiskIndex | None,
        replacement_index: "_ReplacementNeighborhoodIndex | None",
        scorer: EdgeScorer,
        rng: random.Random,
        target_score: float,
        score_lower_bound: float,
        score_upper_bound: float,
    ) -> "_RiskSample":
        if len(source_edge) < 2:
            raise ValueError("risk-controlled sampling requires source hyperedges with size >= 2")

        anchor_size = self._anchor_size(len(source_edge))
        source_set = set(source_edge)
        random_replacement_pool = range(num_nodes)
        replacement_size = len(source_edge) - anchor_size
        max_rounds = max(1, min(self.max_attempts, self.boundary_guided_max_rounds))
        probe_count = max(1, self.boundary_guided_probe_count)
        min_feasible = max(1, self.boundary_guided_min_feasible_candidates)

        attempts = 0
        valid_candidates = 0
        nearest_pass_candidates = 0
        closure_pass_candidates = 0
        cowalk_pass_candidates = 0
        hitting_pass_candidates = 0
        residual_pass_candidates = 0
        all_pass_candidates = 0
        budget_pass_candidates = 0
        candidate_pool_size_sum = 0
        neighbor_replacement_count = 0
        risk_aware_replacement_count = 0
        residual_safe_replacement_count = 0
        diffusion_replacement_count = 0
        total_replacement_count = 0
        risk_cache_hits = 0
        risk_cache_lookups = 0
        rejection_stats = _RejectionStats()
        feasible: list[_PrimalDualCandidate] = []
        best_risk_edge: Hyperedge | None = None
        best_risk_score = float("inf")
        best_risk_value = 0.0
        elite_counts: Counter[int] = Counter()
        elite_updates = 0
        elite_replacement_count = 0

        for _ in range(max_rounds):
            attempts += 1
            anchor_nodes = rng.sample(source_edge, anchor_size)
            round_records: list[_PrimalDualCandidate] = []
            for _ in range(probe_count):
                replacements, candidate_pool_size, neighbor_hits, proposal_strategy = self._sample_replacements(
                    source_edge=source_edge,
                    anchor_nodes=anchor_nodes,
                    source_set=source_set,
                    random_replacement_pool=random_replacement_pool,
                    replacement_size=replacement_size,
                    num_nodes=num_nodes,
                    positive_index=positive_index,
                    replacement_index=replacement_index,
                    closure_index=closure_index,
                    cowalk_index=cowalk_index,
                    hitting_index=hitting_index,
                    residual_index=residual_index,
                    rng=rng,
                )
                if self.boundary_guided_elite_enabled and elite_counts:
                    replacements, elite_used = _mix_elite_replacements(
                        replacements=replacements,
                        elite_counts=elite_counts,
                        random_replacement_pool=random_replacement_pool,
                        excluded_nodes=set(anchor_nodes) | source_set,
                        replacement_size=replacement_size,
                        mix_probability=self.boundary_guided_elite_mix_probability,
                        rng=rng,
                    )
                    elite_replacement_count += elite_used
                candidate_pool_size_sum += candidate_pool_size
                neighbor_replacement_count += neighbor_hits
                if proposal_strategy == "risk_aware_mixed":
                    risk_aware_replacement_count += len(replacements)
                if proposal_strategy == "residual_safe_mixed":
                    residual_safe_replacement_count += len(replacements)
                if proposal_strategy == "diffusion_mixed":
                    diffusion_replacement_count += len(replacements)
                total_replacement_count += len(replacements)
                edge = tuple(sorted(set(anchor_nodes) | set(replacements)))
                if len(edge) != len(source_edge):
                    continue
                if edge in positive_edges:
                    continue

                nearest, closure_risk, cowalk_risk, hitting_risk, residual_risk, cache_hit = self._risk_values(
                    edge=edge,
                    source_edge=source_edge,
                    positive_index=positive_index,
                    closure_index=closure_index,
                    cowalk_index=cowalk_index,
                    hitting_index=hitting_index,
                    residual_index=residual_index,
                )
                risk_cache_lookups += 1
                if cache_hit:
                    risk_cache_hits += 1
                valid_candidates += 1
                nearest_pass = nearest <= self.nearest_positive_upper_bound
                closure_pass = closure_risk <= self.closure_risk_upper_bound
                cowalk_pass = cowalk_index is None or cowalk_risk <= self.cowalk_risk_upper_bound
                hitting_pass = hitting_index is None or hitting_risk <= self.hitting_risk_upper_bound
                residual_pass = residual_index is None or residual_risk <= self.residual_risk_upper_bound
                if nearest_pass:
                    nearest_pass_candidates += 1
                if closure_pass:
                    closure_pass_candidates += 1
                if cowalk_pass:
                    cowalk_pass_candidates += 1
                if hitting_pass:
                    hitting_pass_candidates += 1
                if residual_pass:
                    residual_pass_candidates += 1
                if nearest_pass and closure_pass and cowalk_pass and hitting_pass and residual_pass:
                    all_pass_candidates += 1
                combined_risk = self._combined_risk(nearest, closure_risk, cowalk_risk, hitting_risk, residual_risk)
                budget_pass = combined_risk <= self.risk_budget
                safety_pass = self._safety_pass(nearest, closure_risk, cowalk_risk, hitting_risk, residual_risk)
                if budget_pass:
                    budget_pass_candidates += 1
                risk_score = combined_risk if self.use_risk_budget else nearest + closure_risk + cowalk_risk + hitting_risk + residual_risk
                if risk_score < best_risk_score:
                    best_risk_edge = edge
                    best_risk_score = risk_score
                    best_risk_value = combined_risk
                if self.primal_dual_enabled:
                    accepted = True
                else:
                    accepted = budget_pass and safety_pass if self.use_risk_budget else nearest_pass and closure_pass and cowalk_pass and hitting_pass and residual_pass
                rejection_stats.add(
                    _primary_rejection_reason(
                        nearest=nearest,
                        closure_risk=closure_risk,
                        cowalk_risk=cowalk_risk,
                        hitting_risk=hitting_risk,
                        residual_risk=residual_risk,
                        combined_risk=combined_risk,
                        nearest_pass=nearest_pass,
                        closure_pass=closure_pass,
                        cowalk_pass=cowalk_pass,
                        hitting_pass=hitting_pass,
                        residual_pass=residual_pass,
                        budget_pass=budget_pass,
                        safety_pass=safety_pass,
                        use_risk_budget=self.use_risk_budget,
                        use_cowalk_risk=self.use_cowalk_risk,
                        use_hitting_risk=self.use_hitting_risk,
                        use_residual_risk=self.use_residual_risk,
                        nearest_bound=self.nearest_positive_upper_bound,
                        closure_bound=self.closure_risk_upper_bound,
                        cowalk_bound=self.cowalk_risk_upper_bound,
                        hitting_bound=self.hitting_risk_upper_bound,
                        residual_bound=self.residual_risk_upper_bound,
                        budget_bound=self.risk_budget,
                    )
                )
                if accepted:
                    record = _PrimalDualCandidate(
                        edge=edge,
                        combined_risk=combined_risk,
                        replacements=list(replacements),
                        nearest=nearest,
                        closure_risk=closure_risk,
                        cowalk_risk=cowalk_risk,
                        hitting_risk=hitting_risk,
                        residual_risk=residual_risk,
                    )
                    feasible.append(record)
                    round_records.append(record)
            if self.boundary_guided_elite_enabled and round_records:
                round_scores = scorer.predict_scores([record.edge for record in round_records])
                scored_records = [
                    (record.edge, record.combined_risk, record.replacements, float(score))
                    for record, score in zip(round_records, round_scores)
                ]
                elite_updates += _update_elite_counts(
                    elite_counts=elite_counts,
                    records=scored_records,
                    target_score=target_score,
                    score_lower_bound=score_lower_bound,
                    score_upper_bound=score_upper_bound,
                    risk_budget=self.risk_budget,
                    risk_penalty=self.boundary_guided_elite_risk_penalty,
                    elite_fraction=self.boundary_guided_elite_fraction,
                )
            if len(feasible) >= min_feasible:
                break

        if feasible:
            feasible_edges = [record.edge for record in feasible]
            scores = scorer.predict_scores(feasible_edges)
            if self.primal_dual_enabled:
                selected_index, objective, hardness, risk_penalty = self._select_primal_dual_candidate(
                    records=feasible,
                    scores=[float(score) for score in scores],
                    target_score=target_score,
                    score_lower_bound=score_lower_bound,
                    score_upper_bound=score_upper_bound,
                )
            else:
                selected_index = min(
                    range(len(feasible_edges)),
                    key=lambda index: (
                        not (score_lower_bound <= scores[index] <= score_upper_bound),
                        abs(scores[index] - target_score),
                        -scores[index],
                    ),
                )
                objective = 0.0
                hardness = 0.0
                risk_penalty = 0.0
            selected_score = float(scores[selected_index])
            selected_record = feasible[selected_index]
            selected_edge = selected_record.edge
            selected_risk = selected_record.combined_risk
            if self.primal_dual_enabled:
                self._update_dual_lambdas(selected_record)
            return _RiskSample(
                edge=selected_edge,
                attempts=attempts,
                valid_candidates=valid_candidates,
                nearest_pass_candidates=nearest_pass_candidates,
                closure_pass_candidates=closure_pass_candidates,
                cowalk_pass_candidates=cowalk_pass_candidates,
                hitting_pass_candidates=hitting_pass_candidates,
                residual_pass_candidates=residual_pass_candidates,
                all_pass_candidates=all_pass_candidates,
                budget_pass_candidates=budget_pass_candidates,
                emitted_combined_risk=selected_risk,
                candidate_pool_size_mean=_safe_ratio(candidate_pool_size_sum, attempts * probe_count),
                neighbor_replacement_rate=_safe_ratio(neighbor_replacement_count, total_replacement_count),
                risk_aware_replacement_rate=_safe_ratio(risk_aware_replacement_count, total_replacement_count),
                residual_safe_replacement_rate=_safe_ratio(residual_safe_replacement_count, total_replacement_count),
                diffusion_replacement_rate=_safe_ratio(diffusion_replacement_count, total_replacement_count),
                risk_cache_hit_rate=_safe_ratio(risk_cache_hits, risk_cache_lookups),
                used_fallback=False,
                boundary_probe_candidates=valid_candidates,
                boundary_feasible_candidates=len(feasible),
                boundary_selected_score=selected_score,
                boundary_selected_in_band=score_lower_bound <= selected_score <= score_upper_bound,
                boundary_elite_updates=elite_updates,
                boundary_elite_replacement_rate=_safe_ratio(elite_replacement_count, total_replacement_count),
                primal_dual_enabled=self.primal_dual_enabled,
                primal_dual_objective=objective,
                primal_dual_hardness=hardness,
                primal_dual_risk_penalty=risk_penalty,
                primal_dual_lambda_nearest=self._dual_lambda("nearest"),
                primal_dual_lambda_closure=self._dual_lambda("closure"),
                primal_dual_lambda_cowalk=self._dual_lambda("cowalk"),
                primal_dual_lambda_hitting=self._dual_lambda("hitting"),
                primal_dual_lambda_residual=self._dual_lambda("residual"),
                nearest_primary_rejections=rejection_stats.nearest,
                closure_primary_rejections=rejection_stats.closure,
                cowalk_primary_rejections=rejection_stats.cowalk,
                hitting_primary_rejections=rejection_stats.hitting,
                residual_primary_rejections=rejection_stats.residual,
                budget_primary_rejections=rejection_stats.budget,
                safety_primary_rejections=rejection_stats.safety,
            )

        if best_risk_edge is not None:
            selected_score = scorer.predict_scores([best_risk_edge])[0]
            return _RiskSample(
                edge=best_risk_edge,
                attempts=attempts,
                valid_candidates=valid_candidates,
                nearest_pass_candidates=nearest_pass_candidates,
                closure_pass_candidates=closure_pass_candidates,
                cowalk_pass_candidates=cowalk_pass_candidates,
                hitting_pass_candidates=hitting_pass_candidates,
                residual_pass_candidates=residual_pass_candidates,
                all_pass_candidates=all_pass_candidates,
                budget_pass_candidates=budget_pass_candidates,
                emitted_combined_risk=best_risk_value,
                candidate_pool_size_mean=_safe_ratio(candidate_pool_size_sum, attempts * probe_count),
                neighbor_replacement_rate=_safe_ratio(neighbor_replacement_count, total_replacement_count),
                risk_aware_replacement_rate=_safe_ratio(risk_aware_replacement_count, total_replacement_count),
                residual_safe_replacement_rate=_safe_ratio(residual_safe_replacement_count, total_replacement_count),
                diffusion_replacement_rate=_safe_ratio(diffusion_replacement_count, total_replacement_count),
                risk_cache_hit_rate=_safe_ratio(risk_cache_hits, risk_cache_lookups),
                used_fallback=True,
                boundary_probe_candidates=valid_candidates,
                boundary_feasible_candidates=0,
                boundary_selected_score=float(selected_score),
                boundary_selected_in_band=score_lower_bound <= selected_score <= score_upper_bound,
                boundary_elite_updates=elite_updates,
                boundary_elite_replacement_rate=_safe_ratio(elite_replacement_count, total_replacement_count),
                nearest_primary_rejections=rejection_stats.nearest,
                closure_primary_rejections=rejection_stats.closure,
                cowalk_primary_rejections=rejection_stats.cowalk,
                hitting_primary_rejections=rejection_stats.hitting,
                residual_primary_rejections=rejection_stats.residual,
                budget_primary_rejections=rejection_stats.budget,
                safety_primary_rejections=rejection_stats.safety,
            )
        return self._sample_one(
            source_edge=source_edge,
            num_nodes=num_nodes,
            positive_edges=positive_edges,
            positive_index=positive_index,
            closure_index=closure_index,
            cowalk_index=cowalk_index,
            hitting_index=hitting_index,
            residual_index=residual_index,
            replacement_index=replacement_index,
            rng=rng,
        )

    def _risk_values(
        self,
        edge: Hyperedge,
        source_edge: Hyperedge,
        positive_index: _PositiveEdgeIndex,
        closure_index: ClosureRiskIndex,
        cowalk_index: CoWalkRiskIndex | None,
        hitting_index: HittingRiskIndex | None,
        residual_index: DegreeCorrectedResidualRiskIndex | None,
    ) -> tuple[float, float, float, float, float, bool]:
        cache_key = (edge, source_edge)
        cached_values = self._risk_cache.get(cache_key)
        if cached_values is not None:
            return (*cached_values, True)

        nearest = positive_index.nearest_similarity(edge, excluded_edge=source_edge)
        closure_risk = closure_index.risk(edge)
        cowalk_risk = cowalk_index.risk(edge) if cowalk_index is not None else 0.0
        hitting_risk = hitting_index.risk(edge) if hitting_index is not None else 0.0
        residual_risk = residual_index.risk(edge) if residual_index is not None else 0.0
        values = (nearest, closure_risk, cowalk_risk, hitting_risk, residual_risk)
        if self.max_risk_cache_size > 0:
            if len(self._risk_cache) >= self.max_risk_cache_size:
                self._risk_cache.clear()
            self._risk_cache[cache_key] = values
        return (*values, False)

    def _sample_audited_random_fallback(
        self,
        source_edge: Hyperedge,
        num_nodes: int,
        positive_edges: set[Hyperedge],
        positive_index: _PositiveEdgeIndex,
        closure_index: ClosureRiskIndex,
        cowalk_index: CoWalkRiskIndex | None,
        hitting_index: HittingRiskIndex | None,
        residual_index: DegreeCorrectedResidualRiskIndex | None,
        rng: random.Random,
    ) -> "_RiskSample | None":
        anchor_size = self._anchor_size(len(source_edge))
        replacement_size = len(source_edge) - anchor_size
        source_set = set(source_edge)
        for attempt in range(1, self.max_attempts + 1):
            anchor_nodes = rng.sample(source_edge, anchor_size)
            replacements = _sample_random_nodes_from_pool(range(num_nodes), source_set, replacement_size, rng)
            edge = tuple(sorted(set(anchor_nodes) | set(replacements)))
            if len(edge) != len(source_edge):
                continue
            if edge in positive_edges:
                continue
            nearest, closure_risk, cowalk_risk, hitting_risk, residual_risk, _ = self._risk_values(
                edge=edge,
                source_edge=source_edge,
                positive_index=positive_index,
                closure_index=closure_index,
                cowalk_index=cowalk_index,
                hitting_index=hitting_index,
                residual_index=residual_index,
            )
            nearest_pass = nearest <= self.nearest_positive_upper_bound
            closure_pass = closure_risk <= self.closure_risk_upper_bound
            cowalk_pass = cowalk_index is None or cowalk_risk <= self.cowalk_risk_upper_bound
            hitting_pass = hitting_index is None or hitting_risk <= self.hitting_risk_upper_bound
            residual_pass = residual_index is None or residual_risk <= self.residual_risk_upper_bound
            combined_risk = self._combined_risk(nearest, closure_risk, cowalk_risk, hitting_risk, residual_risk)
            budget_pass = combined_risk <= self.risk_budget
            return _RiskSample(
                edge=edge,
                attempts=attempt,
                valid_candidates=1,
                nearest_pass_candidates=int(nearest_pass),
                closure_pass_candidates=int(closure_pass),
                cowalk_pass_candidates=int(cowalk_pass),
                hitting_pass_candidates=int(hitting_pass),
                residual_pass_candidates=int(residual_pass),
                all_pass_candidates=int(nearest_pass and closure_pass and cowalk_pass and hitting_pass and residual_pass),
                budget_pass_candidates=int(budget_pass),
                emitted_combined_risk=combined_risk,
                candidate_pool_size_mean=max(0.0, float(num_nodes - len(source_set))),
                neighbor_replacement_rate=0.0,
                risk_cache_hit_rate=0.0,
                used_fallback=True,
            )
        return None

    def _anchor_size(self, edge_size: int) -> int:
        raw_size = int(math.floor(edge_size * self.anchor_ratio))
        return min(edge_size - 1, max(1, raw_size))

    def _combined_risk(self, nearest: float, closure_risk: float, cowalk_risk: float, hitting_risk: float, residual_risk: float) -> float:
        nearest_term = self.nearest_risk_weight * _safe_ratio(nearest, self.nearest_positive_upper_bound)
        closure_term = self.closure_risk_weight * _safe_ratio(closure_risk, self.closure_risk_upper_bound)
        cowalk_term = 0.0
        if self.use_cowalk_risk:
            cowalk_term = self.cowalk_risk_weight * _safe_ratio(cowalk_risk, self.cowalk_risk_upper_bound)
        hitting_term = 0.0
        if self.use_hitting_risk:
            hitting_term = self.hitting_risk_weight * _safe_ratio(hitting_risk, self.hitting_risk_upper_bound)
        residual_term = 0.0
        if self.use_residual_risk:
            residual_term = self.residual_risk_weight * _safe_ratio(residual_risk, self.residual_risk_upper_bound)
        return nearest_term + closure_term + cowalk_term + hitting_term + residual_term

    def _safety_pass(self, nearest: float, closure_risk: float, cowalk_risk: float, hitting_risk: float, residual_risk: float) -> bool:
        if not self.use_budget_safety_caps:
            return True
        if nearest > self.safety_nearest_positive_upper_bound:
            return False
        if closure_risk > self.safety_closure_risk_upper_bound:
            return False
        if self.use_cowalk_risk and cowalk_risk > self.safety_cowalk_risk_upper_bound:
            return False
        if self.use_hitting_risk and hitting_risk > self.safety_hitting_risk_upper_bound:
            return False
        if self.use_residual_risk and residual_risk > self.safety_residual_risk_upper_bound:
            return False
        return True

    def _select_primal_dual_candidate(
        self,
        records: list["_PrimalDualCandidate"],
        scores: list[float],
        target_score: float,
        score_lower_bound: float,
        score_upper_bound: float,
    ) -> tuple[int, float, float, float]:
        if not records:
            raise ValueError("primal-dual candidate selection requires at least one record")

        best_index = 0
        best_objective = -float("inf")
        best_hardness = 0.0
        best_penalty = 0.0
        band_width = max(score_upper_bound - score_lower_bound, 1e-9)
        hardness_scale = max(self.primal_dual_hardness_scale, 1e-9)
        for index, (record, score) in enumerate(zip(records, scores)):
            hardness = -abs(float(score) - target_score) / (band_width * hardness_scale)
            risk_penalty = self._dual_risk_penalty(record)
            objective = hardness - risk_penalty
            if objective > best_objective:
                best_index = index
                best_objective = objective
                best_hardness = hardness
                best_penalty = risk_penalty
        return best_index, float(best_objective), float(best_hardness), float(best_penalty)

    def _dual_risk_penalty(self, record: "_PrimalDualCandidate") -> float:
        penalty = self._dual_lambda("nearest") * _safe_ratio(record.nearest, self.nearest_positive_upper_bound)
        penalty += self._dual_lambda("closure") * _safe_ratio(record.closure_risk, self.closure_risk_upper_bound)
        if self.use_cowalk_risk:
            penalty += self._dual_lambda("cowalk") * _safe_ratio(record.cowalk_risk, self.cowalk_risk_upper_bound)
        if self.use_hitting_risk:
            penalty += self._dual_lambda("hitting") * _safe_ratio(record.hitting_risk, self.hitting_risk_upper_bound)
        if self.use_residual_risk:
            penalty += self._dual_lambda("residual") * _safe_ratio(record.residual_risk, self.residual_risk_upper_bound)
        return float(penalty)

    def _update_dual_lambdas(self, record: "_PrimalDualCandidate") -> None:
        self._update_dual_lambda("nearest", _safe_ratio(record.nearest, self.nearest_positive_upper_bound) - 1.0)
        self._update_dual_lambda("closure", _safe_ratio(record.closure_risk, self.closure_risk_upper_bound) - 1.0)
        if self.use_cowalk_risk:
            self._update_dual_lambda("cowalk", _safe_ratio(record.cowalk_risk, self.cowalk_risk_upper_bound) - 1.0)
        if self.use_hitting_risk:
            self._update_dual_lambda("hitting", _safe_ratio(record.hitting_risk, self.hitting_risk_upper_bound) - 1.0)
        if self.use_residual_risk:
            self._update_dual_lambda("residual", _safe_ratio(record.residual_risk, self.residual_risk_upper_bound) - 1.0)

    def _update_dual_lambda(self, name: str, violation: float) -> None:
        value = self._dual_lambda(name) + max(0.0, self.primal_dual_learning_rate) * violation
        self._dual_lambdas[name] = min(max(0.0, value), max(0.0, self.primal_dual_max_lambda))

    def _dual_lambda(self, name: str) -> float:
        return float(self._dual_lambdas.get(name, 0.0))

    def _sample_replacements(
        self,
        source_edge: Hyperedge,
        anchor_nodes: list[int],
        source_set: set[int],
        random_replacement_pool: Sequence[int],
        replacement_size: int,
        num_nodes: int,
        positive_index: _PositiveEdgeIndex,
        replacement_index: "_ReplacementNeighborhoodIndex | None",
        closure_index: ClosureRiskIndex,
        cowalk_index: CoWalkRiskIndex | None,
        hitting_index: HittingRiskIndex | None,
        residual_index: DegreeCorrectedResidualRiskIndex | None,
        rng: random.Random,
    ) -> tuple[list[int], int, int, str]:
        strategy = self._effective_replacement_strategy(rng)
        if (
            strategy not in {"neighborhood_mixed", "risk_aware_mixed", "residual_safe_mixed", "diffusion_mixed"}
            or replacement_index is None
            or rng.random() > self.neighbor_sample_probability
        ):
            replacements = _sample_random_nodes_from_pool(random_replacement_pool, source_set, replacement_size, rng)
            return replacements, len(random_replacement_pool) - len(source_set), 0, "random"

        candidate_pool = replacement_index.candidate_pool(
            anchor_nodes=anchor_nodes,
            excluded_nodes=source_set,
            max_pool_size=self.max_replacement_pool_size,
        )
        if strategy == "risk_aware_mixed":
            candidate_pool = self._risk_aware_candidate_pool(
                candidate_pool=candidate_pool,
                random_replacement_pool=random_replacement_pool,
                source_edge=source_edge,
                anchor_nodes=anchor_nodes,
                source_set=source_set,
                replacement_size=replacement_size,
                positive_index=positive_index,
                closure_index=closure_index,
                cowalk_index=cowalk_index,
                hitting_index=hitting_index,
                residual_index=residual_index,
                rng=rng,
            )
        if strategy == "residual_safe_mixed":
            candidate_pool = self._residual_safe_candidate_pool(
                candidate_pool=candidate_pool,
                random_replacement_pool=random_replacement_pool,
                source_edge=source_edge,
                anchor_nodes=anchor_nodes,
                source_set=source_set,
                replacement_size=replacement_size,
                positive_index=positive_index,
                closure_index=closure_index,
                cowalk_index=cowalk_index,
                hitting_index=hitting_index,
                residual_index=residual_index,
                rng=rng,
            )
        if strategy == "diffusion_mixed":
            replacements, pool_size = self._diffusion_candidate_replacements(
                candidate_pool=candidate_pool,
                random_replacement_pool=random_replacement_pool,
                source_edge=source_edge,
                anchor_nodes=anchor_nodes,
                source_set=source_set,
                replacement_size=replacement_size,
                positive_index=positive_index,
                replacement_index=replacement_index,
                closure_index=closure_index,
                cowalk_index=cowalk_index,
                hitting_index=hitting_index,
                residual_index=residual_index,
                rng=rng,
            )
            return replacements, pool_size, len(replacements), strategy
        if len(candidate_pool) >= replacement_size:
            return rng.sample(candidate_pool, replacement_size), len(candidate_pool), replacement_size, strategy

        selected = list(candidate_pool)
        selected.extend(
            _sample_random_nodes_from_pool(
                random_replacement_pool,
                set(selected) | source_set,
                replacement_size - len(selected),
                rng,
            )
        )
        return selected, len(candidate_pool), len(candidate_pool), strategy

    def _effective_replacement_strategy(self, rng: random.Random) -> str:
        if self.replacement_strategy != "calibrated_mixture":
            return self.replacement_strategy
        risk_probability = min(1.0, max(0.0, self.mixture_risk_aware_probability))
        residual_probability = min(1.0, max(0.0, self.mixture_residual_safe_probability))
        scale = max(1.0, risk_probability + residual_probability)
        risk_probability /= scale
        residual_probability /= scale
        draw = rng.random()
        if draw < residual_probability:
            return "residual_safe_mixed"
        if draw < residual_probability + risk_probability:
            return "risk_aware_mixed"
        return "neighborhood_mixed"

    def _risk_aware_candidate_pool(
        self,
        candidate_pool: list[int],
        random_replacement_pool: Sequence[int],
        source_edge: Hyperedge,
        anchor_nodes: list[int],
        source_set: set[int],
        replacement_size: int,
        positive_index: _PositiveEdgeIndex,
        closure_index: ClosureRiskIndex,
        cowalk_index: CoWalkRiskIndex | None,
        hitting_index: HittingRiskIndex | None,
        residual_index: DegreeCorrectedResidualRiskIndex | None,
        rng: random.Random,
    ) -> list[int]:
        if len(candidate_pool) < replacement_size:
            sample_size = min(
                len(random_replacement_pool) - len(source_set),
                max(self.max_replacement_pool_size, replacement_size),
            )
            random_candidates = _sample_random_nodes_from_pool(random_replacement_pool, source_set, sample_size, rng)
            candidate_pool = list(dict.fromkeys(candidate_pool + random_candidates))
        keep_size = max(replacement_size, self.risk_aware_pool_multiplier * replacement_size)
        if self.risk_aware_nearest_weight > 0.0:
            prefilter_size = max(
                keep_size,
                self.risk_aware_nearest_prefilter_multiplier * keep_size,
            )
            base_scored = [
                (
                    _anchor_node_risk(
                        cache=self._anchor_node_risk_cache,
                        max_cache_size=self.max_anchor_node_risk_cache_size,
                        candidate=candidate,
                        anchor_nodes=anchor_nodes,
                        source_edge=source_edge,
                        positive_index=positive_index,
                        closure_index=closure_index,
                        cowalk_index=cowalk_index,
                        hitting_index=hitting_index,
                        residual_index=residual_index,
                        use_cowalk_risk=self.use_cowalk_risk,
                        use_hitting_risk=self.use_hitting_risk,
                        use_residual_risk=self.use_residual_risk,
                        nearest_weight=0.0,
                    ),
                    candidate,
                )
                for candidate in candidate_pool
                if candidate not in source_set
            ]
            base_scored.sort(key=lambda item: (item[0], item[1]))
            candidate_pool = [candidate for _, candidate in base_scored[:prefilter_size]]
        scored = [
            (
                _anchor_node_risk(
                    cache=self._anchor_node_risk_cache,
                    max_cache_size=self.max_anchor_node_risk_cache_size,
                    candidate=candidate,
                    anchor_nodes=anchor_nodes,
                    source_edge=source_edge,
                    positive_index=positive_index,
                    closure_index=closure_index,
                    cowalk_index=cowalk_index,
                    hitting_index=hitting_index,
                    residual_index=residual_index,
                    use_cowalk_risk=self.use_cowalk_risk,
                    use_hitting_risk=self.use_hitting_risk,
                    use_residual_risk=self.use_residual_risk,
                    nearest_weight=self.risk_aware_nearest_weight,
                ),
                candidate,
            )
            for candidate in candidate_pool
            if candidate not in source_set
        ]
        scored.sort(key=lambda item: (item[0], item[1]))
        return [candidate for _, candidate in scored[:keep_size]]

    def _residual_safe_candidate_pool(
        self,
        candidate_pool: list[int],
        random_replacement_pool: Sequence[int],
        source_edge: Hyperedge,
        anchor_nodes: list[int],
        source_set: set[int],
        replacement_size: int,
        positive_index: _PositiveEdgeIndex,
        closure_index: ClosureRiskIndex,
        cowalk_index: CoWalkRiskIndex | None,
        hitting_index: HittingRiskIndex | None,
        residual_index: DegreeCorrectedResidualRiskIndex | None,
        rng: random.Random,
    ) -> list[int]:
        random_sample_size = 0
        if len(candidate_pool) < self.max_replacement_pool_size:
            random_sample_size = min(
                len(random_replacement_pool) - len(source_set),
                self.max_replacement_pool_size - len(candidate_pool),
                max(replacement_size, self.residual_safe_random_pool_multiplier * max(replacement_size, 1)),
            )
        if random_sample_size > 0:
            random_candidates = _sample_random_nodes_from_pool(random_replacement_pool, source_set, random_sample_size, rng)
            candidate_pool = list(dict.fromkeys(candidate_pool + random_candidates))
        scored: list[tuple[float, float, int]] = []
        for candidate in candidate_pool:
            if candidate in source_set:
                continue
            structural_risk = _anchor_node_risk(
                cache=self._anchor_node_risk_cache,
                max_cache_size=self.max_anchor_node_risk_cache_size,
                candidate=candidate,
                anchor_nodes=anchor_nodes,
                source_edge=source_edge,
                positive_index=positive_index,
                closure_index=closure_index,
                cowalk_index=cowalk_index,
                hitting_index=hitting_index,
                residual_index=None,
                use_cowalk_risk=self.use_cowalk_risk,
                use_hitting_risk=self.use_hitting_risk,
                use_residual_risk=False,
                nearest_weight=0.0,
            )
            residual_risk = 0.0
            if self.use_residual_risk and residual_index is not None:
                residual_risk = _anchor_node_risk(
                    cache=self._anchor_node_risk_cache,
                    max_cache_size=self.max_anchor_node_risk_cache_size,
                    candidate=candidate,
                    anchor_nodes=anchor_nodes,
                    source_edge=source_edge,
                    positive_index=positive_index,
                    closure_index=closure_index,
                    cowalk_index=None,
                    hitting_index=None,
                    residual_index=residual_index,
                    use_cowalk_risk=False,
                    use_hitting_risk=False,
                    use_residual_risk=True,
                    nearest_weight=0.0,
                )
            scored.append((float(structural_risk), float(residual_risk), candidate))
        if not scored:
            return []

        structural_target = _quantile([item[0] for item in scored], self.residual_safe_structural_quantile)
        structural_scale = max(structural_target, max(item[0] for item in scored), 1e-9)
        residual_scale = max(self.residual_risk_upper_bound, max(item[1] for item in scored), 1e-9)
        keep_size = max(replacement_size, self.residual_safe_pool_multiplier * replacement_size)
        ranked = sorted(
            scored,
            key=lambda item: (
                abs(item[0] - structural_target) / structural_scale
                + max(0.0, self.residual_safe_residual_weight) * item[1] / residual_scale,
                -item[0],
                item[2],
            ),
        )
        return [candidate for _, _, candidate in ranked[:keep_size]]

    def _diffusion_candidate_replacements(
        self,
        candidate_pool: list[int],
        random_replacement_pool: Sequence[int],
        source_edge: Hyperedge,
        anchor_nodes: list[int],
        source_set: set[int],
        replacement_size: int,
        positive_index: _PositiveEdgeIndex,
        replacement_index: "_ReplacementNeighborhoodIndex",
        closure_index: ClosureRiskIndex,
        cowalk_index: CoWalkRiskIndex | None,
        hitting_index: HittingRiskIndex | None,
        residual_index: DegreeCorrectedResidualRiskIndex | None,
        rng: random.Random,
    ) -> tuple[list[int], int]:
        if replacement_size <= 0:
            return [], len(candidate_pool)

        target_pool_size = max(replacement_size, self.diffusion_pool_multiplier * replacement_size)
        candidate_pool = [candidate for candidate in candidate_pool if candidate not in source_set]
        random_sample_size = min(
            len(random_replacement_pool) - len(source_set),
            max(replacement_size, self.diffusion_random_pool_multiplier * replacement_size),
            max(0, target_pool_size - min(len(candidate_pool), target_pool_size // 2)),
        )
        local_keep_size = max(replacement_size, target_pool_size - random_sample_size)
        candidate_pool = candidate_pool[:local_keep_size]
        if random_sample_size > 0:
            random_candidates = _sample_random_nodes_from_pool(random_replacement_pool, source_set, random_sample_size, rng)
            candidate_pool = list(dict.fromkeys(candidate_pool + random_candidates))
        candidate_pool = candidate_pool[:target_pool_size]
        if len(candidate_pool) < replacement_size:
            selected = list(candidate_pool)
            selected.extend(
                _sample_random_nodes_from_pool(
                    random_replacement_pool,
                    set(selected) | source_set,
                    replacement_size - len(selected),
                    rng,
                )
            )
            return selected, len(candidate_pool)

        selected = _sample_random_nodes_from_pool(candidate_pool, set(), replacement_size, rng)
        steps = max(1, int(self.diffusion_steps))
        for step in range(steps):
            if steps == 1:
                temperature = self.diffusion_temperature_end
            else:
                progress = step / (steps - 1)
                temperature = (1.0 - progress) * self.diffusion_temperature_start + progress * self.diffusion_temperature_end
            temperature = max(float(temperature), 1e-6)
            for position in range(replacement_size):
                excluded = source_set | {node for index, node in enumerate(selected) if index != position}
                weighted_candidates = [
                    (
                        self._diffusion_node_logit(
                            candidate=candidate,
                            source_edge=source_edge,
                            anchor_nodes=anchor_nodes,
                            replacement_index=replacement_index,
                            positive_index=positive_index,
                            closure_index=closure_index,
                            cowalk_index=cowalk_index,
                            hitting_index=hitting_index,
                            residual_index=residual_index,
                        )
                        / temperature,
                        candidate,
                    )
                    for candidate in candidate_pool
                    if candidate not in excluded
                ]
                selected[position] = _weighted_choice(weighted_candidates, rng)
        return list(selected), len(candidate_pool)

    def _diffusion_node_logit(
        self,
        candidate: int,
        source_edge: Hyperedge,
        anchor_nodes: list[int],
        replacement_index: "_ReplacementNeighborhoodIndex",
        positive_index: _PositiveEdgeIndex,
        closure_index: ClosureRiskIndex,
        cowalk_index: CoWalkRiskIndex | None,
        hitting_index: HittingRiskIndex | None,
        residual_index: DegreeCorrectedResidualRiskIndex | None,
    ) -> float:
        structural_support = replacement_index.structural_support(candidate, anchor_nodes)
        risk = _anchor_node_risk(
            cache=self._anchor_node_risk_cache,
            max_cache_size=self.max_anchor_node_risk_cache_size,
            candidate=candidate,
            anchor_nodes=anchor_nodes,
            source_edge=source_edge,
            positive_index=positive_index,
            closure_index=closure_index,
            cowalk_index=cowalk_index,
            hitting_index=hitting_index,
            residual_index=residual_index,
            use_cowalk_risk=self.use_cowalk_risk,
            use_hitting_risk=self.use_hitting_risk,
            use_residual_risk=self.use_residual_risk,
            nearest_weight=self.risk_aware_nearest_weight,
        )
        return (
            max(0.0, self.diffusion_structural_weight) * math.log1p(structural_support)
            - max(0.0, self.diffusion_risk_weight) * risk
        )


class _PrimalDualCandidate(NamedTuple):
    edge: Hyperedge
    combined_risk: float
    replacements: list[int]
    nearest: float
    closure_risk: float
    cowalk_risk: float
    hitting_risk: float
    residual_risk: float


class _RiskSample(NamedTuple):
    edge: Hyperedge
    attempts: int
    valid_candidates: int
    nearest_pass_candidates: int
    closure_pass_candidates: int
    cowalk_pass_candidates: int
    hitting_pass_candidates: int
    residual_pass_candidates: int
    all_pass_candidates: int
    budget_pass_candidates: int
    emitted_combined_risk: float
    candidate_pool_size_mean: float
    neighbor_replacement_rate: float
    risk_cache_hit_rate: float
    used_fallback: bool
    risk_aware_replacement_rate: float = 0.0
    residual_safe_replacement_rate: float = 0.0
    diffusion_replacement_rate: float = 0.0
    boundary_probe_candidates: int = 0
    boundary_feasible_candidates: int = 0
    boundary_selected_score: float = 0.0
    boundary_selected_in_band: bool = False
    boundary_elite_updates: int = 0
    boundary_elite_replacement_rate: float = 0.0
    nearest_primary_rejections: int = 0
    closure_primary_rejections: int = 0
    cowalk_primary_rejections: int = 0
    hitting_primary_rejections: int = 0
    residual_primary_rejections: int = 0
    budget_primary_rejections: int = 0
    safety_primary_rejections: int = 0
    primal_dual_enabled: bool = False
    primal_dual_objective: float = 0.0
    primal_dual_hardness: float = 0.0
    primal_dual_risk_penalty: float = 0.0
    primal_dual_lambda_nearest: float = 0.0
    primal_dual_lambda_closure: float = 0.0
    primal_dual_lambda_cowalk: float = 0.0
    primal_dual_lambda_hitting: float = 0.0
    primal_dual_lambda_residual: float = 0.0


@dataclass
class _RejectionStats:
    nearest: int = 0
    closure: int = 0
    cowalk: int = 0
    hitting: int = 0
    residual: int = 0
    budget: int = 0
    safety: int = 0

    def add(self, reason: str | None) -> None:
        if reason == "nearest":
            self.nearest += 1
        elif reason == "closure":
            self.closure += 1
        elif reason == "cowalk":
            self.cowalk += 1
        elif reason == "hitting":
            self.hitting += 1
        elif reason == "residual":
            self.residual += 1
        elif reason == "budget":
            self.budget += 1
        elif reason == "safety":
            self.safety += 1


@dataclass
class _SamplingStats:
    samples: int = 0
    accepted: int = 0
    fallbacks: int = 0
    attempts: int = 0
    valid_candidates: int = 0
    nearest_pass_candidates: int = 0
    closure_pass_candidates: int = 0
    cowalk_pass_candidates: int = 0
    hitting_pass_candidates: int = 0
    residual_pass_candidates: int = 0
    all_pass_candidates: int = 0
    budget_pass_candidates: int = 0
    emitted_combined_risk: float = 0.0
    candidate_pool_size: float = 0.0
    neighbor_replacement_rate: float = 0.0
    risk_aware_replacement_rate: float = 0.0
    residual_safe_replacement_rate: float = 0.0
    diffusion_replacement_rate: float = 0.0
    risk_cache_hit_rate: float = 0.0
    boundary_probe_candidates: int = 0
    boundary_feasible_candidates: int = 0
    boundary_selected_score: float = 0.0
    boundary_selected_in_band: int = 0
    boundary_elite_updates: int = 0
    boundary_elite_replacement_rate: float = 0.0
    nearest_primary_rejections: int = 0
    closure_primary_rejections: int = 0
    cowalk_primary_rejections: int = 0
    hitting_primary_rejections: int = 0
    residual_primary_rejections: int = 0
    budget_primary_rejections: int = 0
    safety_primary_rejections: int = 0
    primal_dual_enabled: int = 0
    primal_dual_objective: float = 0.0
    primal_dual_hardness: float = 0.0
    primal_dual_risk_penalty: float = 0.0
    primal_dual_lambda_nearest: float = 0.0
    primal_dual_lambda_closure: float = 0.0
    primal_dual_lambda_cowalk: float = 0.0
    primal_dual_lambda_hitting: float = 0.0
    primal_dual_lambda_residual: float = 0.0

    def add(self, sample: _RiskSample) -> None:
        self.samples += 1
        self.attempts += sample.attempts
        self.valid_candidates += sample.valid_candidates
        self.nearest_pass_candidates += sample.nearest_pass_candidates
        self.closure_pass_candidates += sample.closure_pass_candidates
        self.cowalk_pass_candidates += sample.cowalk_pass_candidates
        self.hitting_pass_candidates += sample.hitting_pass_candidates
        self.residual_pass_candidates += sample.residual_pass_candidates
        self.all_pass_candidates += sample.all_pass_candidates
        self.budget_pass_candidates += sample.budget_pass_candidates
        self.emitted_combined_risk += sample.emitted_combined_risk
        self.candidate_pool_size += sample.candidate_pool_size_mean
        self.neighbor_replacement_rate += sample.neighbor_replacement_rate
        self.risk_aware_replacement_rate += sample.risk_aware_replacement_rate
        self.residual_safe_replacement_rate += sample.residual_safe_replacement_rate
        self.diffusion_replacement_rate += sample.diffusion_replacement_rate
        self.risk_cache_hit_rate += sample.risk_cache_hit_rate
        self.boundary_probe_candidates += sample.boundary_probe_candidates
        self.boundary_feasible_candidates += sample.boundary_feasible_candidates
        self.boundary_selected_score += sample.boundary_selected_score
        self.boundary_selected_in_band += int(sample.boundary_selected_in_band)
        self.boundary_elite_updates += sample.boundary_elite_updates
        self.boundary_elite_replacement_rate += sample.boundary_elite_replacement_rate
        self.nearest_primary_rejections += sample.nearest_primary_rejections
        self.closure_primary_rejections += sample.closure_primary_rejections
        self.cowalk_primary_rejections += sample.cowalk_primary_rejections
        self.hitting_primary_rejections += sample.hitting_primary_rejections
        self.residual_primary_rejections += sample.residual_primary_rejections
        self.budget_primary_rejections += sample.budget_primary_rejections
        self.safety_primary_rejections += sample.safety_primary_rejections
        self.primal_dual_enabled += int(sample.primal_dual_enabled)
        self.primal_dual_objective += sample.primal_dual_objective
        self.primal_dual_hardness += sample.primal_dual_hardness
        self.primal_dual_risk_penalty += sample.primal_dual_risk_penalty
        self.primal_dual_lambda_nearest += sample.primal_dual_lambda_nearest
        self.primal_dual_lambda_closure += sample.primal_dual_lambda_closure
        self.primal_dual_lambda_cowalk += sample.primal_dual_lambda_cowalk
        self.primal_dual_lambda_hitting += sample.primal_dual_lambda_hitting
        self.primal_dual_lambda_residual += sample.primal_dual_lambda_residual
        if sample.used_fallback:
            self.fallbacks += 1
        else:
            self.accepted += 1

    def to_metadata(self) -> dict[str, float]:
        if self.samples == 0:
            return {
                "risk_accept_rate": 0.0,
                "risk_fallback_rate": 0.0,
                "sampling_attempts_mean": 0.0,
                "valid_candidates_mean": 0.0,
                "nearest_constraint_pass_rate": 0.0,
                "closure_constraint_pass_rate": 0.0,
                "cowalk_constraint_pass_rate": 0.0,
                "hitting_constraint_pass_rate": 0.0,
                "residual_constraint_pass_rate": 0.0,
                "all_constraints_pass_rate": 0.0,
                "budget_constraint_pass_rate": 0.0,
                "combined_risk_mean": 0.0,
                "candidate_pool_size_mean": 0.0,
                "neighbor_replacement_rate": 0.0,
                "risk_aware_replacement_rate": 0.0,
                "residual_safe_replacement_rate": 0.0,
                "diffusion_replacement_rate": 0.0,
                "risk_cache_hit_rate": 0.0,
                "adversarial_probe_candidates_mean": 0.0,
                "adversarial_feasible_candidates_mean": 0.0,
                "adversarial_feasible_candidate_rate": 0.0,
                "adversarial_selected_score_mean": 0.0,
                "adversarial_selected_in_band_rate": 0.0,
                "adversarial_elite_updates_mean": 0.0,
                "adversarial_elite_replacement_rate": 0.0,
                "nearest_primary_rejection_rate": 0.0,
                "closure_primary_rejection_rate": 0.0,
                "cowalk_primary_rejection_rate": 0.0,
                "hitting_primary_rejection_rate": 0.0,
                "residual_primary_rejection_rate": 0.0,
                "budget_primary_rejection_rate": 0.0,
                "safety_primary_rejection_rate": 0.0,
                "primal_dual_enabled": 0.0,
                "primal_dual_objective_mean": 0.0,
                "primal_dual_hardness_mean": 0.0,
                "primal_dual_risk_penalty_mean": 0.0,
                "primal_dual_lambda_nearest_mean": 0.0,
                "primal_dual_lambda_closure_mean": 0.0,
                "primal_dual_lambda_cowalk_mean": 0.0,
                "primal_dual_lambda_hitting_mean": 0.0,
                "primal_dual_lambda_residual_mean": 0.0,
            }
        if self.valid_candidates == 0:
            nearest_rate = 0.0
            closure_rate = 0.0
            cowalk_rate = 0.0
            hitting_rate = 0.0
            residual_rate = 0.0
            all_rate = 0.0
            budget_rate = 0.0
        else:
            nearest_rate = self.nearest_pass_candidates / self.valid_candidates
            closure_rate = self.closure_pass_candidates / self.valid_candidates
            cowalk_rate = self.cowalk_pass_candidates / self.valid_candidates
            hitting_rate = self.hitting_pass_candidates / self.valid_candidates
            residual_rate = self.residual_pass_candidates / self.valid_candidates
            all_rate = self.all_pass_candidates / self.valid_candidates
            budget_rate = self.budget_pass_candidates / self.valid_candidates
        return {
            "risk_accept_rate": self.accepted / self.samples,
            "risk_fallback_rate": self.fallbacks / self.samples,
            "sampling_attempts_mean": self.attempts / self.samples,
            "valid_candidates_mean": self.valid_candidates / self.samples,
            "nearest_constraint_pass_rate": nearest_rate,
            "closure_constraint_pass_rate": closure_rate,
            "cowalk_constraint_pass_rate": cowalk_rate,
            "hitting_constraint_pass_rate": hitting_rate,
            "residual_constraint_pass_rate": residual_rate,
            "all_constraints_pass_rate": all_rate,
            "budget_constraint_pass_rate": budget_rate,
            "combined_risk_mean": self.emitted_combined_risk / self.samples,
            "candidate_pool_size_mean": self.candidate_pool_size / self.samples,
            "neighbor_replacement_rate": self.neighbor_replacement_rate / self.samples,
            "risk_aware_replacement_rate": self.risk_aware_replacement_rate / self.samples,
            "residual_safe_replacement_rate": self.residual_safe_replacement_rate / self.samples,
            "diffusion_replacement_rate": self.diffusion_replacement_rate / self.samples,
            "risk_cache_hit_rate": self.risk_cache_hit_rate / self.samples,
            "adversarial_probe_candidates_mean": self.boundary_probe_candidates / self.samples,
            "adversarial_feasible_candidates_mean": self.boundary_feasible_candidates / self.samples,
            "adversarial_feasible_candidate_rate": _safe_ratio(self.boundary_feasible_candidates, self.boundary_probe_candidates),
            "adversarial_selected_score_mean": self.boundary_selected_score / self.samples,
            "adversarial_selected_in_band_rate": self.boundary_selected_in_band / self.samples,
            "adversarial_elite_updates_mean": self.boundary_elite_updates / self.samples,
            "adversarial_elite_replacement_rate": self.boundary_elite_replacement_rate / self.samples,
            "nearest_primary_rejection_rate": _safe_ratio(self.nearest_primary_rejections, self.valid_candidates),
            "closure_primary_rejection_rate": _safe_ratio(self.closure_primary_rejections, self.valid_candidates),
            "cowalk_primary_rejection_rate": _safe_ratio(self.cowalk_primary_rejections, self.valid_candidates),
            "hitting_primary_rejection_rate": _safe_ratio(self.hitting_primary_rejections, self.valid_candidates),
            "residual_primary_rejection_rate": _safe_ratio(self.residual_primary_rejections, self.valid_candidates),
            "budget_primary_rejection_rate": _safe_ratio(self.budget_primary_rejections, self.valid_candidates),
            "safety_primary_rejection_rate": _safe_ratio(self.safety_primary_rejections, self.valid_candidates),
            "primal_dual_enabled": _safe_ratio(self.primal_dual_enabled, self.samples),
            "primal_dual_objective_mean": self.primal_dual_objective / self.samples,
            "primal_dual_hardness_mean": self.primal_dual_hardness / self.samples,
            "primal_dual_risk_penalty_mean": self.primal_dual_risk_penalty / self.samples,
            "primal_dual_lambda_nearest_mean": self.primal_dual_lambda_nearest / self.samples,
            "primal_dual_lambda_closure_mean": self.primal_dual_lambda_closure / self.samples,
            "primal_dual_lambda_cowalk_mean": self.primal_dual_lambda_cowalk / self.samples,
            "primal_dual_lambda_hitting_mean": self.primal_dual_lambda_hitting / self.samples,
            "primal_dual_lambda_residual_mean": self.primal_dual_lambda_residual / self.samples,
        }


def _safe_ratio(value: float, denominator: float) -> float:
    if denominator <= 0.0:
        return 0.0 if value <= 0.0 else float("inf")
    return value / denominator


def _quantile(values: list[float], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    clipped = min(1.0, max(0.0, quantile))
    index = int(math.floor(clipped * (len(ordered) - 1)))
    return float(ordered[index])


def _mix_elite_replacements(
    replacements: list[int],
    elite_counts: Counter[int],
    random_replacement_pool: Sequence[int],
    excluded_nodes: set[int],
    replacement_size: int,
    mix_probability: float,
    rng: random.Random,
) -> tuple[list[int], int]:
    if not elite_counts or rng.random() > max(0.0, min(1.0, mix_probability)):
        return replacements, 0
    selected: list[int] = []
    used = 0
    available = [
        (node, max(1, count))
        for node, count in elite_counts.items()
        if node not in excluded_nodes
    ]
    while available and len(selected) < replacement_size:
        total_weight = sum(weight for _, weight in available)
        draw = rng.uniform(0.0, float(total_weight))
        cumulative = 0.0
        chosen_index = 0
        for index, (_, weight) in enumerate(available):
            cumulative += weight
            if draw <= cumulative:
                chosen_index = index
                break
        node, _ = available.pop(chosen_index)
        selected.append(node)
        used += 1
    for node in replacements:
        if len(selected) >= replacement_size:
            break
        if node not in excluded_nodes and node not in selected:
            selected.append(node)
    if len(selected) < replacement_size:
        selected.extend(
            _sample_random_nodes_from_pool(
                random_replacement_pool,
                set(selected) | excluded_nodes,
                replacement_size - len(selected),
                rng,
            )
        )
    return selected, used


def _update_elite_counts(
    elite_counts: Counter[int],
    records: list[tuple[Hyperedge, float, list[int], float]],
    target_score: float,
    score_lower_bound: float,
    score_upper_bound: float,
    risk_budget: float,
    risk_penalty: float,
    elite_fraction: float,
) -> int:
    if not records:
        return 0
    elite_size = max(1, int(math.ceil(len(records) * max(0.0, min(1.0, elite_fraction)))))
    scored = sorted(
        records,
        key=lambda record: _elite_score(
            score=record[3],
            risk=record[1],
            target_score=target_score,
            score_lower_bound=score_lower_bound,
            score_upper_bound=score_upper_bound,
            risk_budget=risk_budget,
            risk_penalty=risk_penalty,
        ),
        reverse=True,
    )
    for _, _, replacements, _ in scored[:elite_size]:
        for node in replacements:
            elite_counts[node] += 1
    return 1


def _elite_score(
    score: float,
    risk: float,
    target_score: float,
    score_lower_bound: float,
    score_upper_bound: float,
    risk_budget: float,
    risk_penalty: float,
) -> float:
    in_band_bonus = 0.25 if score_lower_bound <= score <= score_upper_bound else 0.0
    normalized_risk = _safe_ratio(risk, risk_budget) if risk_budget > 0.0 else risk
    return in_band_bonus - abs(score - target_score) - max(0.0, risk_penalty) * normalized_risk


def _sample_target_score(
    rng: random.Random,
    target_score: float,
    target_score_min: float,
    target_score_max: float,
) -> float:
    lower = min(target_score_min, target_score_max)
    upper = max(target_score_min, target_score_max)
    if lower == upper:
        return target_score
    return rng.uniform(lower, upper)


def _primary_rejection_reason(
    *,
    nearest: float,
    closure_risk: float,
    cowalk_risk: float,
    hitting_risk: float,
    residual_risk: float,
    combined_risk: float,
    nearest_pass: bool,
    closure_pass: bool,
    cowalk_pass: bool,
    hitting_pass: bool,
    residual_pass: bool,
    budget_pass: bool,
    safety_pass: bool,
    use_risk_budget: bool,
    use_cowalk_risk: bool,
    use_hitting_risk: bool,
    use_residual_risk: bool,
    nearest_bound: float,
    closure_bound: float,
    cowalk_bound: float,
    hitting_bound: float,
    residual_bound: float,
    budget_bound: float,
) -> str | None:
    if use_risk_budget and budget_pass and safety_pass:
        return None
    if not use_risk_budget and nearest_pass and closure_pass and cowalk_pass and hitting_pass and residual_pass:
        return None

    component_excesses: list[tuple[float, str]] = []
    if not nearest_pass:
        component_excesses.append((_safe_ratio(nearest, nearest_bound), "nearest"))
    if not closure_pass:
        component_excesses.append((_safe_ratio(closure_risk, closure_bound), "closure"))
    if use_cowalk_risk and not cowalk_pass:
        component_excesses.append((_safe_ratio(cowalk_risk, cowalk_bound), "cowalk"))
    if use_hitting_risk and not hitting_pass:
        component_excesses.append((_safe_ratio(hitting_risk, hitting_bound), "hitting"))
    if use_residual_risk and not residual_pass:
        component_excesses.append((_safe_ratio(residual_risk, residual_bound), "residual"))
    if component_excesses:
        return max(component_excesses, key=lambda item: item[0])[1]
    if use_risk_budget and not budget_pass:
        return "budget"
    if not safety_pass:
        return "safety"
    return None


def _anchor_node_risk(
    *,
    cache: dict[tuple[object, ...], float],
    max_cache_size: int,
    candidate: int,
    anchor_nodes: list[int],
    source_edge: Hyperedge,
    positive_index: _PositiveEdgeIndex,
    closure_index: ClosureRiskIndex,
    cowalk_index: CoWalkRiskIndex | None,
    hitting_index: HittingRiskIndex | None,
    residual_index: DegreeCorrectedResidualRiskIndex | None,
    use_cowalk_risk: bool,
    use_hitting_risk: bool,
    use_residual_risk: bool,
    nearest_weight: float,
) -> float:
    cache_key = (
        candidate,
        tuple(sorted(anchor_nodes)),
        source_edge,
        bool(use_cowalk_risk),
        bool(use_hitting_risk),
        bool(use_residual_risk),
        round(float(nearest_weight), 8),
    )
    cached_risk = cache.get(cache_key)
    if cached_risk is not None:
        return cached_risk

    probe_edge = tuple(sorted(set(anchor_nodes) | {candidate}))
    risk = 0.0
    if nearest_weight > 0.0:
        risk += nearest_weight * positive_index.nearest_similarity(probe_edge, excluded_edge=source_edge)
    risk += closure_index.risk(probe_edge)
    if use_cowalk_risk and cowalk_index is not None:
        risk += cowalk_index.risk(probe_edge)
    if use_hitting_risk and hitting_index is not None:
        risk += hitting_index.risk(probe_edge)
    if use_residual_risk and residual_index is not None:
        risk += residual_index.risk(probe_edge)
    if max_cache_size > 0:
        if len(cache) >= max_cache_size:
            cache.clear()
        cache[cache_key] = risk
    return risk


class _ReplacementNeighborhoodIndex:
    def __init__(
        self,
        positive_edges: set[Hyperedge],
        max_neighbors_per_node: int,
        max_pairs_per_edge: int,
        max_pool_cache_size: int,
    ) -> None:
        self.max_pool_cache_size = max_pool_cache_size
        self._candidate_pool_cache: dict[tuple[tuple[int, ...], tuple[int, ...], int], list[int]] = {}
        adjacency: dict[int, Counter[int]] = {}
        for edge in positive_edges:
            if len(edge) < 2:
                continue
            for node_a, node_b in _bounded_pairs(edge, max_pairs_per_edge):
                adjacency.setdefault(node_a, Counter())[node_b] += 1
                adjacency.setdefault(node_b, Counter())[node_a] += 1

        self.neighbors: dict[int, tuple[int, ...]] = {}
        for node, counts in adjacency.items():
            self.neighbors[node] = tuple(
                neighbor
                for neighbor, _ in counts.most_common(max_neighbors_per_node)
            )

    def structural_support(self, candidate: int, anchor_nodes: list[int]) -> float:
        support = 0.0
        for anchor_node in anchor_nodes:
            if candidate in self.neighbors.get(anchor_node, ()):
                support += 1.0
        return support

    def candidate_pool(
        self,
        anchor_nodes: list[int],
        excluded_nodes: set[int],
        max_pool_size: int,
    ) -> list[int]:
        cache_key = (tuple(sorted(anchor_nodes)), tuple(sorted(excluded_nodes)), max_pool_size)
        cached_pool = self._candidate_pool_cache.get(cache_key)
        if cached_pool is not None:
            return list(cached_pool)

        scored: Counter[int] = Counter()
        for anchor_node in anchor_nodes:
            for neighbor in self.neighbors.get(anchor_node, ()):
                if neighbor not in excluded_nodes:
                    scored[neighbor] += 1
        candidate_pool = [
            node
            for node, _ in scored.most_common(max_pool_size)
        ]
        if self.max_pool_cache_size > 0:
            if len(self._candidate_pool_cache) >= self.max_pool_cache_size:
                self._candidate_pool_cache.clear()
            self._candidate_pool_cache[cache_key] = candidate_pool
        return list(candidate_pool)


def _bounded_pairs(edge: Hyperedge, max_pairs: int) -> list[tuple[int, int]]:
    pair_count = len(edge) * (len(edge) - 1) // 2
    if pair_count <= max_pairs:
        return list(combinations(edge, 2))

    nodes = tuple(edge)
    pairs: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    stride = max(1, len(nodes) // int(math.sqrt(max_pairs)))
    offset = 1
    while len(pairs) < max_pairs and offset < len(nodes):
        for index, node_a in enumerate(nodes):
            node_b = nodes[(index + offset) % len(nodes)]
            if node_a == node_b:
                continue
            pair = (node_a, node_b) if node_a < node_b else (node_b, node_a)
            if pair not in seen:
                seen.add(pair)
                pairs.append(pair)
                if len(pairs) >= max_pairs:
                    break
        offset += stride
    return pairs


def _sample_random_nodes(
    num_nodes: int,
    excluded_nodes: set[int],
    sample_size: int,
    rng: random.Random,
) -> list[int]:
    if num_nodes - len(excluded_nodes) < sample_size:
        raise ValueError("not enough nodes to sample replacements")
    selected: set[int] = set()
    while len(selected) < sample_size:
        candidate = rng.randrange(num_nodes)
        if candidate not in excluded_nodes and candidate not in selected:
            selected.add(candidate)
    return list(selected)


def _sample_random_nodes_from_pool(
    candidate_pool: Sequence[int],
    excluded_nodes: set[int],
    sample_size: int,
    rng: random.Random,
) -> list[int]:
    if len(candidate_pool) - len(excluded_nodes) < sample_size:
        raise ValueError("not enough nodes to sample replacements")
    if not excluded_nodes:
        return rng.sample(candidate_pool, sample_size)

    selected: set[int] = set()
    while len(selected) < sample_size:
        candidate = candidate_pool[rng.randrange(len(candidate_pool))]
        if candidate not in excluded_nodes and candidate not in selected:
            selected.add(candidate)
    return list(selected)


def _weighted_choice(weighted_items: list[tuple[float, int]], rng: random.Random) -> int:
    if not weighted_items:
        raise ValueError("weighted choice requires at least one candidate")
    max_logit = max(logit for logit, _ in weighted_items)
    weights = [(math.exp(logit - max_logit), item) for logit, item in weighted_items]
    total = sum(weight for weight, _ in weights)
    if total <= 0.0 or not math.isfinite(total):
        return weighted_items[rng.randrange(len(weighted_items))][1]
    draw = rng.uniform(0.0, total)
    cumulative = 0.0
    for weight, item in weights:
        cumulative += weight
        if draw <= cumulative:
            return item
    return weights[-1][1]
