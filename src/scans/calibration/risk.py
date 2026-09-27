from __future__ import annotations

import copy
import math
import random
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from scans.data.hypergraph import Hyperedge
from scans.risk import ClosureRiskIndex, CoWalkRiskIndex, DegreeCorrectedResidualRiskIndex, HittingRiskIndex
from scans.samplers.anchored_safe_sampler import _PositiveEdgeIndex
from scans.samplers.registry import build_sampler


@dataclass(frozen=True)
class _RiskObservation:
    nearest: float
    closure: float
    cowalk: float
    hitting: float
    residual: float


def calibrate_sampling_config(
    sampling_config: Mapping[str, Any],
    train_edges: list[Hyperedge],
    num_nodes: int,
    positive_edges: set[Hyperedge],
    seed: int,
) -> tuple[dict[str, Any], dict[str, float]]:
    calibrated_config = copy.deepcopy(dict(sampling_config))
    calibration_metrics: dict[str, float] = {}
    adaptive_config = sampling_config.get("adaptive_risk")
    if not isinstance(adaptive_config, Mapping) or not bool(adaptive_config.get("enabled", False)):
        proposal_metrics = _apply_proposal_calibration(
            sampling_config=calibrated_config,
            train_edges=train_edges,
            num_nodes=num_nodes,
            positive_edges=positive_edges,
            seed=seed,
        )
        calibration_metrics.update(proposal_metrics)
        return calibrated_config, calibration_metrics

    rng = random.Random(seed + int(adaptive_config.get("seed_offset", 1729)))
    observations = _collect_observations(
        sampling_config=calibrated_config,
        adaptive_config=adaptive_config,
        train_edges=train_edges,
        num_nodes=num_nodes,
        positive_edges=positive_edges,
        rng=rng,
    )
    if not observations:
        calibration_metrics["adaptive_calibration_candidates"] = 0.0
        proposal_metrics = _apply_proposal_calibration(
            sampling_config=calibrated_config,
            train_edges=train_edges,
            num_nodes=num_nodes,
            positive_edges=positive_edges,
            seed=seed,
        )
        calibration_metrics.update(proposal_metrics)
        return calibrated_config, calibration_metrics

    nearest_values = [observation.nearest for observation in observations]
    closure_values = [observation.closure for observation in observations]
    cowalk_values = [observation.cowalk for observation in observations]
    hitting_values = [observation.hitting for observation in observations]
    residual_values = [observation.residual for observation in observations]

    nearest_quantile = float(adaptive_config.get("nearest_quantile", adaptive_config.get("scale_quantile", 0.5)))
    closure_quantile = float(adaptive_config.get("closure_quantile", adaptive_config.get("scale_quantile", 0.5)))
    cowalk_quantile = float(adaptive_config.get("cowalk_quantile", adaptive_config.get("scale_quantile", 0.5)))
    hitting_quantile = float(adaptive_config.get("hitting_quantile", adaptive_config.get("scale_quantile", 0.5)))
    residual_quantile = float(adaptive_config.get("residual_quantile", adaptive_config.get("scale_quantile", 0.5)))
    budget_quantile = float(adaptive_config.get("budget_quantile", 0.03))
    quantile_method = str(adaptive_config.get("quantile_method", "linear"))

    nearest_bound = _calibrated_scale(
        nearest_values,
        quantile=nearest_quantile,
        positive_quantile=float(adaptive_config.get("nonzero_scale_quantile", nearest_quantile)),
        floor=float(adaptive_config.get("nearest_floor", 1e-6)),
        method=quantile_method,
    )
    closure_bound = _calibrated_scale(
        closure_values,
        quantile=closure_quantile,
        positive_quantile=float(adaptive_config.get("nonzero_scale_quantile", closure_quantile)),
        floor=float(adaptive_config.get("closure_floor", 1e-6)),
        method=quantile_method,
    )
    cowalk_bound = _calibrated_scale(
        cowalk_values,
        quantile=cowalk_quantile,
        positive_quantile=float(adaptive_config.get("nonzero_scale_quantile", cowalk_quantile)),
        floor=float(adaptive_config.get("cowalk_floor", 1e-6)),
        method=quantile_method,
    )
    hitting_bound = _calibrated_scale(
        hitting_values,
        quantile=hitting_quantile,
        positive_quantile=float(adaptive_config.get("nonzero_scale_quantile", hitting_quantile)),
        floor=float(adaptive_config.get("hitting_floor", 1e-6)),
        method=quantile_method,
    )
    residual_bound = _calibrated_scale(
        residual_values,
        quantile=residual_quantile,
        positive_quantile=float(adaptive_config.get("nonzero_scale_quantile", residual_quantile)),
        floor=float(adaptive_config.get("residual_floor", 1e-6)),
        method=quantile_method,
    )

    calibrated_config["nearest_positive_upper_bound"] = nearest_bound
    calibrated_config["closure_risk_upper_bound"] = closure_bound
    if bool(calibrated_config.get("use_cowalk_risk", False)):
        calibrated_config["cowalk_risk_upper_bound"] = cowalk_bound
    if bool(calibrated_config.get("use_hitting_risk", False)):
        calibrated_config["hitting_risk_upper_bound"] = hitting_bound
    if bool(calibrated_config.get("use_residual_risk", False)):
        calibrated_config["residual_risk_upper_bound"] = residual_bound

    combined_values = [
        _combined_risk(
            observation=observation,
            nearest_bound=nearest_bound,
            closure_bound=closure_bound,
            cowalk_bound=cowalk_bound,
            hitting_bound=hitting_bound,
            residual_bound=residual_bound,
            use_cowalk_risk=bool(calibrated_config.get("use_cowalk_risk", False)),
            use_hitting_risk=bool(calibrated_config.get("use_hitting_risk", False)),
            use_residual_risk=bool(calibrated_config.get("use_residual_risk", False)),
            nearest_weight=float(calibrated_config.get("nearest_risk_weight", 1.0)),
            closure_weight=float(calibrated_config.get("closure_risk_weight", 1.0)),
            cowalk_weight=float(calibrated_config.get("cowalk_risk_weight", 1.0)),
            hitting_weight=float(calibrated_config.get("hitting_risk_weight", 1.0)),
            residual_weight=float(calibrated_config.get("residual_risk_weight", 1.0)),
        )
        for observation in observations
    ]
    if bool(calibrated_config.get("use_risk_budget", False)):
        calibrated_config["risk_budget"] = _calibrated_scale(
            combined_values,
            quantile=budget_quantile,
            positive_quantile=float(adaptive_config.get("nonzero_budget_quantile", budget_quantile)),
            floor=float(adaptive_config.get("risk_budget_floor", 1e-6)),
            method=quantile_method,
        )
    if bool(adaptive_config.get("set_safety_caps", False)):
        calibrated_config["use_budget_safety_caps"] = True
        nearest_multiplier = float(adaptive_config.get("safety_nearest_multiplier", adaptive_config.get("safety_cap_multiplier", 1.0)))
        closure_multiplier = float(adaptive_config.get("safety_closure_multiplier", adaptive_config.get("safety_cap_multiplier", 1.0)))
        cowalk_multiplier = float(adaptive_config.get("safety_cowalk_multiplier", adaptive_config.get("safety_cap_multiplier", 1.0)))
        hitting_multiplier = float(adaptive_config.get("safety_hitting_multiplier", adaptive_config.get("safety_cap_multiplier", 1.0)))
        residual_multiplier = float(adaptive_config.get("safety_residual_multiplier", adaptive_config.get("safety_cap_multiplier", 1.0)))
        calibrated_config["safety_nearest_positive_upper_bound"] = nearest_bound * nearest_multiplier
        calibrated_config["safety_closure_risk_upper_bound"] = closure_bound * closure_multiplier
        calibrated_config["safety_cowalk_risk_upper_bound"] = cowalk_bound * cowalk_multiplier
        calibrated_config["safety_hitting_risk_upper_bound"] = hitting_bound * hitting_multiplier
        calibrated_config["safety_residual_risk_upper_bound"] = residual_bound * residual_multiplier

    calibration_metrics.update({
        "adaptive_calibration_candidates": float(len(observations)),
        "adaptive_nearest_positive_upper_bound": float(nearest_bound),
        "adaptive_closure_risk_upper_bound": float(closure_bound),
        "adaptive_cowalk_risk_upper_bound": float(cowalk_bound),
        "adaptive_hitting_risk_upper_bound": float(hitting_bound),
        "adaptive_residual_risk_upper_bound": float(residual_bound),
        "adaptive_risk_budget": float(calibrated_config.get("risk_budget", 0.0)),
        "adaptive_scale_quantile": float(adaptive_config.get("scale_quantile", 0.5)),
        "adaptive_budget_quantile": float(budget_quantile),
        "adaptive_order_statistic_quantile": 1.0 if quantile_method == "order_statistic" else 0.0,
        "adaptive_combined_risk_mean": float(sum(combined_values) / len(combined_values)),
        "adaptive_closure_zero_rate": _zero_rate(closure_values),
        "adaptive_cowalk_zero_rate": _zero_rate(cowalk_values),
        "adaptive_hitting_zero_rate": _zero_rate(hitting_values),
        "adaptive_residual_zero_rate": _zero_rate(residual_values),
        "adaptive_safety_nearest_positive_upper_bound": float(calibrated_config.get("safety_nearest_positive_upper_bound", 0.0)),
        "adaptive_safety_closure_risk_upper_bound": float(calibrated_config.get("safety_closure_risk_upper_bound", 0.0)),
        "adaptive_safety_cowalk_risk_upper_bound": float(calibrated_config.get("safety_cowalk_risk_upper_bound", 0.0)),
        "adaptive_safety_hitting_risk_upper_bound": float(calibrated_config.get("safety_hitting_risk_upper_bound", 0.0)),
        "adaptive_safety_residual_risk_upper_bound": float(calibrated_config.get("safety_residual_risk_upper_bound", 0.0)),
    })
    proposal_metrics = _apply_proposal_calibration(
        sampling_config=calibrated_config,
        train_edges=train_edges,
        num_nodes=num_nodes,
        positive_edges=positive_edges,
        seed=seed,
    )
    calibration_metrics.update(proposal_metrics)
    return calibrated_config, calibration_metrics


def _apply_proposal_calibration(
    sampling_config: dict[str, Any],
    train_edges: list[Hyperedge],
    num_nodes: int,
    positive_edges: set[Hyperedge],
    seed: int,
) -> dict[str, float]:
    if sampling_config.get("replacement_strategy") == "calibrated_mixture":
        return _apply_calibrated_mixture(
            sampling_config=sampling_config,
            train_edges=train_edges,
            num_nodes=num_nodes,
            positive_edges=positive_edges,
            seed=seed,
        )
    return _apply_proposal_gating(
        sampling_config=sampling_config,
        train_edges=train_edges,
        num_nodes=num_nodes,
        positive_edges=positive_edges,
        seed=seed,
    )


def _apply_proposal_gating(
    sampling_config: dict[str, Any],
    train_edges: list[Hyperedge],
    num_nodes: int,
    positive_edges: set[Hyperedge],
    seed: int,
) -> dict[str, float]:
    gating_config = sampling_config.get("proposal_gating")
    if not isinstance(gating_config, Mapping) or not bool(gating_config.get("enabled", False)):
        if sampling_config.get("replacement_strategy") == "auto_risk_aware":
            sampling_config["replacement_strategy"] = "neighborhood_mixed"
            return {"proposal_gating_enabled": 0.0, "proposal_gating_selected_risk_aware": 0.0}
        return {}

    if sampling_config.get("replacement_strategy") != "auto_risk_aware":
        return {"proposal_gating_enabled": 1.0, "proposal_gating_selected_risk_aware": 0.0}

    max_source_edges = min(len(train_edges), int(gating_config.get("max_source_edges", 128)))
    if max_source_edges <= 0:
        sampling_config["replacement_strategy"] = "neighborhood_mixed"
        return {"proposal_gating_enabled": 1.0, "proposal_gating_selected_risk_aware": 0.0}

    rng = random.Random(seed + int(gating_config.get("seed_offset", 3251)))
    source_edges = rng.sample(train_edges, max_source_edges) if len(train_edges) > max_source_edges else list(train_edges)
    min_edge_size = min(len(edge) for edge in train_edges)
    max_edge_size = max(len(edge) for edge in train_edges)

    neighborhood_metrics = _proposal_pilot_metrics(
        sampling_config=sampling_config,
        source_edges=source_edges,
        num_nodes=num_nodes,
        positive_edges=positive_edges,
        min_edge_size=min_edge_size,
        max_edge_size=max_edge_size,
        replacement_strategy="neighborhood_mixed",
        seed=seed + int(gating_config.get("neighborhood_seed_offset", 421)),
    )
    risk_aware_metrics = _proposal_pilot_metrics(
        sampling_config=sampling_config,
        source_edges=source_edges,
        num_nodes=num_nodes,
        positive_edges=positive_edges,
        min_edge_size=min_edge_size,
        max_edge_size=max_edge_size,
        replacement_strategy="risk_aware_mixed",
        seed=seed + int(gating_config.get("risk_aware_seed_offset", 422)),
    )

    nearest_activation_rate = float(gating_config.get("nearest_activation_rate", 0.45))
    fallback_activation_rate = float(gating_config.get("fallback_activation_rate", 0.30))
    max_fallback_increase = float(gating_config.get("max_fallback_increase", 0.15))
    min_candidate_pool_size = float(gating_config.get("min_candidate_pool_size", 0.5))
    risk_reduction_margin = float(gating_config.get("risk_reduction_margin", 0.05))

    neighborhood_fallback = neighborhood_metrics["risk_fallback_rate"]
    risk_fallback = risk_aware_metrics["risk_fallback_rate"]
    neighborhood_risk = neighborhood_metrics["combined_risk_mean"]
    risk_aware_risk = risk_aware_metrics["combined_risk_mean"]
    risk_reduction = _relative_drop(neighborhood_risk, risk_aware_risk)
    use_risk_aware = (
        neighborhood_metrics["nearest_primary_rejection_rate"] >= nearest_activation_rate
        and neighborhood_fallback >= fallback_activation_rate
        and risk_fallback <= neighborhood_fallback + max_fallback_increase
        and risk_aware_metrics["candidate_pool_size_mean"] >= min_candidate_pool_size
        and risk_reduction >= risk_reduction_margin
    )
    sampling_config["replacement_strategy"] = "risk_aware_mixed" if use_risk_aware else "neighborhood_mixed"

    return {
        "proposal_gating_enabled": 1.0,
        "proposal_gating_selected_risk_aware": 1.0 if use_risk_aware else 0.0,
        "proposal_gating_neighborhood_fallback_rate": neighborhood_fallback,
        "proposal_gating_risk_aware_fallback_rate": risk_fallback,
        "proposal_gating_neighborhood_nearest_primary_rate": neighborhood_metrics["nearest_primary_rejection_rate"],
        "proposal_gating_risk_aware_nearest_primary_rate": risk_aware_metrics["nearest_primary_rejection_rate"],
        "proposal_gating_neighborhood_combined_risk_mean": neighborhood_risk,
        "proposal_gating_risk_aware_combined_risk_mean": risk_aware_risk,
        "proposal_gating_combined_risk_relative_drop": risk_reduction,
        "proposal_gating_risk_aware_candidate_pool_size_mean": risk_aware_metrics["candidate_pool_size_mean"],
    }


def _apply_calibrated_mixture(
    sampling_config: dict[str, Any],
    train_edges: list[Hyperedge],
    num_nodes: int,
    positive_edges: set[Hyperedge],
    seed: int,
) -> dict[str, float]:
    mixture_config = sampling_config.get("proposal_mixture")
    mixture_mapping = mixture_config if isinstance(mixture_config, Mapping) else {}
    max_source_edges = min(len(train_edges), int(mixture_mapping.get("max_source_edges", 128)))
    if max_source_edges <= 0:
        sampling_config["mixture_risk_aware_probability"] = 0.0
        return {
            "proposal_mixture_enabled": 1.0,
            "proposal_mixture_risk_aware_probability": 0.0,
        }

    rng = random.Random(seed + int(mixture_mapping.get("seed_offset", 5279)))
    source_edges = rng.sample(train_edges, max_source_edges) if len(train_edges) > max_source_edges else list(train_edges)
    min_edge_size = min(len(edge) for edge in train_edges)
    max_edge_size = max(len(edge) for edge in train_edges)

    neighborhood_metrics = _proposal_pilot_metrics(
        sampling_config=sampling_config,
        source_edges=source_edges,
        num_nodes=num_nodes,
        positive_edges=positive_edges,
        min_edge_size=min_edge_size,
        max_edge_size=max_edge_size,
        replacement_strategy="neighborhood_mixed",
        seed=seed + int(mixture_mapping.get("neighborhood_seed_offset", 613)),
    )
    risk_aware_metrics = _proposal_pilot_metrics(
        sampling_config=sampling_config,
        source_edges=source_edges,
        num_nodes=num_nodes,
        positive_edges=positive_edges,
        min_edge_size=min_edge_size,
        max_edge_size=max_edge_size,
        replacement_strategy="risk_aware_mixed",
        seed=seed + int(mixture_mapping.get("risk_aware_seed_offset", 614)),
    )

    fallback_low = float(mixture_mapping.get("fallback_low", 0.20))
    fallback_high = float(mixture_mapping.get("fallback_high", 0.50))
    nearest_pressure_low = float(mixture_mapping.get("nearest_pressure_low", 0.10))
    nearest_pressure_high = float(mixture_mapping.get("nearest_pressure_high", 0.30))
    min_probability = float(mixture_mapping.get("min_risk_aware_probability", 0.0))
    max_probability = float(mixture_mapping.get("max_risk_aware_probability", 0.60))
    max_fallback_increase = float(mixture_mapping.get("max_fallback_increase", 0.15))
    min_candidate_pool_size = float(mixture_mapping.get("min_candidate_pool_size", 0.5))

    neighborhood_fallback = neighborhood_metrics["risk_fallback_rate"]
    risk_aware_fallback = risk_aware_metrics["risk_fallback_rate"]
    fallback_pressure = _clamped_ratio(neighborhood_fallback - fallback_low, fallback_high - fallback_low)
    nearest_pressure = _clamped_ratio(
        neighborhood_metrics["nearest_primary_rejection_rate"] - nearest_pressure_low,
        nearest_pressure_high - nearest_pressure_low,
    )
    pressure = max(fallback_pressure, nearest_pressure)
    fallback_penalty = _clamped_ratio(
        max_fallback_increase - max(0.0, risk_aware_fallback - neighborhood_fallback),
        max_fallback_increase,
    )
    diversity_ok = 1.0 if risk_aware_metrics["candidate_pool_size_mean"] >= min_candidate_pool_size else 0.0
    probability = min_probability + (max_probability - min_probability) * pressure * fallback_penalty * diversity_ok
    probability = min(max_probability, max(min_probability, probability))
    sampling_config["mixture_risk_aware_probability"] = float(probability)

    return {
        "proposal_mixture_enabled": 1.0,
        "proposal_mixture_risk_aware_probability": float(probability),
        "proposal_mixture_fallback_pressure": fallback_pressure,
        "proposal_mixture_nearest_pressure": nearest_pressure,
        "proposal_mixture_neighborhood_fallback_rate": neighborhood_fallback,
        "proposal_mixture_risk_aware_fallback_rate": risk_aware_fallback,
        "proposal_mixture_neighborhood_combined_risk_mean": neighborhood_metrics["combined_risk_mean"],
        "proposal_mixture_risk_aware_combined_risk_mean": risk_aware_metrics["combined_risk_mean"],
        "proposal_mixture_neighborhood_candidate_pool_size_mean": neighborhood_metrics["candidate_pool_size_mean"],
        "proposal_mixture_risk_aware_candidate_pool_size_mean": risk_aware_metrics["candidate_pool_size_mean"],
        "proposal_mixture_neighborhood_nearest_primary_rate": neighborhood_metrics["nearest_primary_rejection_rate"],
        "proposal_mixture_risk_aware_nearest_primary_rate": risk_aware_metrics["nearest_primary_rejection_rate"],
    }


def _proposal_pilot_metrics(
    sampling_config: Mapping[str, Any],
    source_edges: list[Hyperedge],
    num_nodes: int,
    positive_edges: set[Hyperedge],
    min_edge_size: int,
    max_edge_size: int,
    replacement_strategy: str,
    seed: int,
) -> dict[str, float]:
    pilot_config = copy.deepcopy(dict(sampling_config))
    pilot_config["replacement_strategy"] = replacement_strategy
    sampler = build_sampler("risk_controlled", pilot_config, min_edge_size, max_edge_size)
    batch = sampler.sample(
        source_edges=source_edges,
        num_nodes=num_nodes,
        positive_edges=positive_edges,
        rng=random.Random(seed),
        negatives_per_positive=1,
    )
    return batch.metadata


def _relative_drop(reference: float, value: float) -> float:
    if reference <= 0.0:
        return 0.0
    return max(0.0, (reference - value) / reference)


def _clamped_ratio(value: float, denominator: float) -> float:
    if denominator <= 0.0:
        return 0.0
    return min(1.0, max(0.0, value / denominator))


def _collect_observations(
    sampling_config: Mapping[str, Any],
    adaptive_config: Mapping[str, Any],
    train_edges: list[Hyperedge],
    num_nodes: int,
    positive_edges: set[Hyperedge],
    rng: random.Random,
) -> list[_RiskObservation]:
    max_source_edges = min(len(train_edges), int(adaptive_config.get("max_source_edges", 256)))
    candidates_per_edge = int(adaptive_config.get("candidates_per_edge", 8))
    max_attempts_per_candidate = int(adaptive_config.get("max_attempts_per_candidate", 20))
    if max_source_edges <= 0 or candidates_per_edge <= 0:
        return []

    source_edges = rng.sample(train_edges, max_source_edges) if len(train_edges) > max_source_edges else list(train_edges)
    anchor_ratio = float(sampling_config.get("anchor_ratio", 0.25))
    use_cowalk_risk = bool(sampling_config.get("use_cowalk_risk", False))
    use_hitting_risk = bool(sampling_config.get("use_hitting_risk", False))
    use_residual_risk = bool(sampling_config.get("use_residual_risk", False))
    max_cowalk_neighbors = int(sampling_config.get("max_cowalk_neighbors", 128))
    max_hitting_neighbors = int(sampling_config.get("max_hitting_neighbors", 128))
    hitting_two_step_weight = float(sampling_config.get("hitting_two_step_weight", 0.5))

    positive_index = _PositiveEdgeIndex(positive_edges)
    closure_index = ClosureRiskIndex(positive_edges)
    cowalk_index = CoWalkRiskIndex(positive_edges, max_neighbors_per_node=max_cowalk_neighbors) if use_cowalk_risk else None
    hitting_index = (
        HittingRiskIndex(
            positive_edges,
            max_neighbors_per_node=max_hitting_neighbors,
            two_step_weight=hitting_two_step_weight,
        )
        if use_hitting_risk
        else None
    )
    residual_index = (
        DegreeCorrectedResidualRiskIndex(
            positive_edges,
            max_pairs_per_edge=int(sampling_config.get("max_replacement_pairs_per_edge", 2000)),
        )
        if use_residual_risk
        else None
    )

    observations: list[_RiskObservation] = []
    for source_edge in source_edges:
        anchor_size = _anchor_size(len(source_edge), anchor_ratio)
        replacement_size = len(source_edge) - anchor_size
        source_set = set(source_edge)
        if num_nodes - len(source_set) < replacement_size:
            continue
        for _ in range(candidates_per_edge):
            edge = _sample_candidate(
                source_edge=source_edge,
                source_set=source_set,
                anchor_size=anchor_size,
                replacement_size=replacement_size,
                num_nodes=num_nodes,
                positive_edges=positive_edges,
                rng=rng,
                max_attempts=max_attempts_per_candidate,
            )
            if edge is None:
                continue
            observations.append(
                _RiskObservation(
                    nearest=positive_index.nearest_similarity(edge, excluded_edge=source_edge),
                    closure=closure_index.risk(edge),
                    cowalk=cowalk_index.risk(edge) if cowalk_index is not None else 0.0,
                    hitting=hitting_index.risk(edge) if hitting_index is not None else 0.0,
                    residual=residual_index.risk(edge) if residual_index is not None else 0.0,
                )
            )
    return observations


def _sample_candidate(
    source_edge: Hyperedge,
    source_set: set[int],
    anchor_size: int,
    replacement_size: int,
    num_nodes: int,
    positive_edges: set[Hyperedge],
    rng: random.Random,
    max_attempts: int,
) -> Hyperedge | None:
    for _ in range(max_attempts):
        anchor_nodes = rng.sample(source_edge, anchor_size)
        replacements = _sample_random_nodes(num_nodes, source_set, replacement_size, rng)
        edge = tuple(sorted(set(anchor_nodes) | set(replacements)))
        if len(edge) != len(source_edge):
            continue
        if edge in positive_edges:
            continue
        return edge
    return None


def _anchor_size(edge_size: int, anchor_ratio: float) -> int:
    raw_size = int(math.floor(edge_size * anchor_ratio))
    return min(edge_size - 1, max(1, raw_size))


def _sample_random_nodes(
    num_nodes: int,
    excluded_nodes: set[int],
    sample_size: int,
    rng: random.Random,
) -> list[int]:
    selected: set[int] = set()
    while len(selected) < sample_size:
        candidate = rng.randrange(num_nodes)
        if candidate not in excluded_nodes and candidate not in selected:
            selected.add(candidate)
    return list(selected)


def _combined_risk(
    observation: _RiskObservation,
    nearest_bound: float,
    closure_bound: float,
    cowalk_bound: float,
    hitting_bound: float,
    residual_bound: float,
    use_cowalk_risk: bool,
    use_hitting_risk: bool,
    use_residual_risk: bool,
    nearest_weight: float,
    closure_weight: float,
    cowalk_weight: float,
    hitting_weight: float,
    residual_weight: float,
) -> float:
    combined = nearest_weight * _safe_ratio(observation.nearest, nearest_bound)
    combined += closure_weight * _safe_ratio(observation.closure, closure_bound)
    if use_cowalk_risk:
        combined += cowalk_weight * _safe_ratio(observation.cowalk, cowalk_bound)
    if use_hitting_risk:
        combined += hitting_weight * _safe_ratio(observation.hitting, hitting_bound)
    if use_residual_risk:
        combined += residual_weight * _safe_ratio(observation.residual, residual_bound)
    return float(combined)


def _safe_ratio(value: float, denominator: float) -> float:
    if denominator <= 0.0:
        return 0.0 if value <= 0.0 else float("inf")
    return value / denominator


def _quantile(values: list[float], quantile: float, method: str = "linear") -> float:
    if not values:
        raise ValueError("cannot compute quantile of empty values")
    bounded_quantile = min(1.0, max(0.0, quantile))
    sorted_values = sorted(values)
    if method == "order_statistic":
        index = max(0, min(len(sorted_values) - 1, int(math.ceil(bounded_quantile * len(sorted_values))) - 1))
        return float(sorted_values[index])
    if method != "linear":
        raise ValueError(f"unknown quantile method: {method}")
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    position = bounded_quantile * (len(sorted_values) - 1)
    lower_index = int(math.floor(position))
    upper_index = int(math.ceil(position))
    if lower_index == upper_index:
        return float(sorted_values[lower_index])
    lower_weight = upper_index - position
    upper_weight = position - lower_index
    return float(sorted_values[lower_index] * lower_weight + sorted_values[upper_index] * upper_weight)


def _calibrated_scale(
    values: list[float],
    quantile: float,
    positive_quantile: float,
    floor: float,
    method: str = "linear",
) -> float:
    raw_value = _quantile(values, quantile, method=method)
    if raw_value > floor:
        return float(raw_value)

    positive_values = [value for value in values if value > floor]
    if positive_values:
        return _floor_positive(_quantile(positive_values, positive_quantile, method=method), floor)
    return _floor_positive(raw_value, floor)


def _floor_positive(value: float, floor: float) -> float:
    if math.isfinite(value) and value > floor:
        return float(value)
    return float(floor)


def _zero_rate(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(1 for value in values if value <= 0.0) / len(values)
