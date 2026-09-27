from __future__ import annotations

from collections.abc import Mapping

from scans.samplers.anchored_replacement_sampler import AnchoredReplacementSampler
from scans.samplers.anchored_safe_sampler import AnchoredSafeSampler
from scans.samplers.base import NegativeSampler
from scans.samplers.fast_safe_sampler import FastSafeSampler
from scans.samplers.half_corruption_sampler import HalfCorruptionSampler
from scans.samplers.random_sampler import RandomSampler
from scans.samplers.risk_controlled_sampler import RiskControlledSampler
from scans.samplers.size_matched_sampler import SizeMatchedSampler
from scans.samplers.clique_sampler import CliqueNegativeSampler
from scans.samplers.mixed_sampler import MixedOfficialSampler
from scans.samplers.motif_sampler import MotifNegativeSampler


def build_sampler(name: str, config: Mapping[str, object], min_edge_size: int, max_edge_size: int) -> NegativeSampler:
    max_attempts = int(config.get("max_attempts", 100))
    adversarial_config = config.get("adversarial_proposal")
    adversarial_mapping = adversarial_config if isinstance(adversarial_config, Mapping) else {}
    if name == "random":
        return RandomSampler(
            min_edge_size=min_edge_size,
            max_edge_size=max_edge_size,
            max_attempts=max_attempts,
        )
    if name in {"size_matched", "sns"}:
        return SizeMatchedSampler(max_attempts=max_attempts)
    if name == "mns":
        return MotifNegativeSampler(max_attempts=max_attempts)
    if name == "cns":
        return CliqueNegativeSampler(max_attempts=max_attempts)
    if name == "mix":
        return MixedOfficialSampler(max_attempts=max_attempts)
    if name == "half_corruption":
        return HalfCorruptionSampler(max_attempts=max_attempts)
    if name == "fast_safe":
        return FastSafeSampler(
            anchor_ratio=float(config.get("fast_safe_anchor_ratio", 0.5)),
            source_jaccard_upper_bound=float(config.get("fast_safe_source_jaccard_upper_bound", 0.85)),
            max_attempts=max_attempts,
        )
    if name == "anchored_random":
        return AnchoredReplacementSampler(
            anchor_ratio=float(config.get("anchor_ratio", 0.5)),
            jaccard_upper_bound=float(config.get("jaccard_upper_bound", 0.75)),
            max_attempts=max_attempts,
        )
    if name == "anchored_safe":
        return AnchoredSafeSampler(
            anchor_ratio=float(config.get("anchor_ratio", 0.25)),
            jaccard_upper_bound=float(config.get("jaccard_upper_bound", 0.65)),
            nearest_positive_upper_bound=float(config.get("nearest_positive_upper_bound", 0.25)),
            max_attempts=max_attempts,
        )
    if name in {"risk_controlled", "model_aware_risk_controlled"}:
        return RiskControlledSampler(
            anchor_ratio=float(config.get("anchor_ratio", 0.25)),
            nearest_positive_upper_bound=float(config.get("nearest_positive_upper_bound", 0.25)),
            closure_risk_upper_bound=float(config.get("closure_risk_upper_bound", 0.35)),
            use_cowalk_risk=bool(config.get("use_cowalk_risk", False)),
            cowalk_risk_upper_bound=float(config.get("cowalk_risk_upper_bound", 0.35)),
            use_hitting_risk=bool(config.get("use_hitting_risk", False)),
            hitting_risk_upper_bound=float(config.get("hitting_risk_upper_bound", 0.35)),
            use_residual_risk=bool(config.get("use_residual_risk", False)),
            residual_risk_upper_bound=float(config.get("residual_risk_upper_bound", 0.35)),
            use_risk_budget=bool(config.get("use_risk_budget", False)),
            risk_budget=float(config.get("risk_budget", 3.0)),
            nearest_risk_weight=float(config.get("nearest_risk_weight", 1.0)),
            closure_risk_weight=float(config.get("closure_risk_weight", 1.0)),
            cowalk_risk_weight=float(config.get("cowalk_risk_weight", 1.0)),
            hitting_risk_weight=float(config.get("hitting_risk_weight", 1.0)),
            residual_risk_weight=float(config.get("residual_risk_weight", 1.0)),
            use_budget_safety_caps=bool(config.get("use_budget_safety_caps", False)),
            safety_nearest_positive_upper_bound=float(config.get("safety_nearest_positive_upper_bound", float("inf"))),
            safety_closure_risk_upper_bound=float(config.get("safety_closure_risk_upper_bound", float("inf"))),
            safety_cowalk_risk_upper_bound=float(config.get("safety_cowalk_risk_upper_bound", float("inf"))),
            safety_hitting_risk_upper_bound=float(config.get("safety_hitting_risk_upper_bound", float("inf"))),
            safety_residual_risk_upper_bound=float(config.get("safety_residual_risk_upper_bound", float("inf"))),
            replacement_strategy=str(config.get("replacement_strategy", "random")),
            neighbor_sample_probability=float(config.get("neighbor_sample_probability", 0.8)),
            max_replacement_neighbors_per_node=int(config.get("max_replacement_neighbors_per_node", 128)),
            max_replacement_pool_size=int(config.get("max_replacement_pool_size", 512)),
            max_replacement_pairs_per_edge=int(config.get("max_replacement_pairs_per_edge", 2000)),
            max_closure_pairs_per_edge=int(config.get("max_closure_pairs_per_edge", 0)),
            max_cowalk_neighbors=int(config.get("max_cowalk_neighbors", 128)),
            max_hitting_neighbors=int(config.get("max_hitting_neighbors", 128)),
            hitting_two_step_weight=float(config.get("hitting_two_step_weight", 0.5)),
            max_risk_cache_size=int(config.get("max_risk_cache_size", 200_000)),
            max_candidate_pool_cache_size=int(config.get("max_candidate_pool_cache_size", 100_000)),
            max_anchor_node_risk_cache_size=int(config.get("max_anchor_node_risk_cache_size", 200_000)),
            max_attempts=max_attempts,
            boundary_guided_probe_count=int(adversarial_mapping.get("probe_count", 16)),
            boundary_guided_max_rounds=int(adversarial_mapping.get("max_rounds", 12)),
            boundary_guided_min_feasible_candidates=int(adversarial_mapping.get("min_feasible_candidates", 4)),
            boundary_guided_elite_enabled=bool(adversarial_mapping.get("elite_enabled", False)),
            boundary_guided_elite_fraction=float(adversarial_mapping.get("elite_fraction", 0.35)),
            boundary_guided_elite_mix_probability=float(adversarial_mapping.get("elite_mix_probability", 0.70)),
            boundary_guided_elite_risk_penalty=float(adversarial_mapping.get("elite_risk_penalty", 0.10)),
            risk_aware_pool_multiplier=int(config.get("risk_aware_pool_multiplier", 4)),
            risk_aware_nearest_weight=float(config.get("risk_aware_nearest_weight", 0.0)),
            risk_aware_nearest_prefilter_multiplier=int(config.get("risk_aware_nearest_prefilter_multiplier", 4)),
            mixture_risk_aware_probability=float(config.get("mixture_risk_aware_probability", 0.0)),
            mixture_residual_safe_probability=float(config.get("mixture_residual_safe_probability", 0.0)),
            residual_safe_pool_multiplier=int(config.get("residual_safe_pool_multiplier", 4)),
            residual_safe_structural_quantile=float(config.get("residual_safe_structural_quantile", 0.50)),
            residual_safe_residual_weight=float(config.get("residual_safe_residual_weight", 2.0)),
            residual_safe_random_pool_multiplier=int(config.get("residual_safe_random_pool_multiplier", 4)),
            diffusion_steps=int(config.get("diffusion_steps", 4)),
            diffusion_pool_multiplier=int(config.get("diffusion_pool_multiplier", 6)),
            diffusion_temperature_start=float(config.get("diffusion_temperature_start", 1.5)),
            diffusion_temperature_end=float(config.get("diffusion_temperature_end", 0.35)),
            diffusion_structural_weight=float(config.get("diffusion_structural_weight", 1.0)),
            diffusion_risk_weight=float(config.get("diffusion_risk_weight", 1.0)),
            diffusion_random_pool_multiplier=int(config.get("diffusion_random_pool_multiplier", 4)),
            primal_dual_enabled=bool(config.get("primal_dual_enabled", False)),
            primal_dual_learning_rate=float(config.get("primal_dual_learning_rate", 0.05)),
            primal_dual_max_lambda=float(config.get("primal_dual_max_lambda", 10.0)),
            primal_dual_hardness_scale=float(config.get("primal_dual_hardness_scale", 1.0)),
        )
    raise ValueError(f"unknown sampler: {name}")
