from __future__ import annotations

import random
import hashlib
import copy
import math
from dataclasses import dataclass
from collections.abc import Mapping

import torch
from torch import nn

from scans.calibration import calibrate_sampling_config
from scans.data import HypergraphDataset, generate_synthetic_hypergraph, load_hyperedges_from_txt
from scans.evaluation import binary_prediction_metrics, future_positive_hit_metrics, negative_quality_metrics
from scans.samplers.base import NegativeSampleBatch
from scans.risk import ClosureRiskIndex, CoWalkRiskIndex, DegreeCorrectedResidualRiskIndex, HittingRiskIndex
from scans.models import (
    BENCHMARK_BACKBONES,
    BenchmarkBackbonePredictor,
    BipartiteHyperedgePredictor,
    HyperedgePredictor,
    SetHyperedgePredictor,
    StrongHyperedgePredictor,
)
from scans.models.hyperedge_predictor import _pool_stat_edge_chunk
from scans.reporting import append_status
from scans.samplers.anchored_safe_sampler import _PositiveEdgeIndex
from scans.samplers import NegativeSampler, build_sampler
from scans.samplers.size_matched_sampler import SizeMatchedSampler
from scans.scoring import ModelAwareReranker, RerankConfig
from scans.training.seed import set_global_seed


def run_mvp_experiment(config: Mapping[str, object]) -> dict[str, object]:
    seed = int(config["experiment"]["seed"])
    set_global_seed(seed)
    dataset = _load_dataset(config, seed)
    runtime_config, calibration_metrics = _build_runtime_config(config, dataset, seed)
    min_edge_size, max_edge_size = _edge_size_range(dataset.train_edges)

    sampling_config = runtime_config["sampling"]
    sampler_names = list(sampling_config["samplers"])
    results: dict[str, object] = {}

    for sampler_name in sampler_names:
        append_status(_status_path(runtime_config), "sampler_start", sampler=sampler_name)
        run_seed = seed + _sampler_seed_offset(sampler_name)
        set_global_seed(run_seed)
        rng = random.Random(run_seed)
        sampler = build_sampler(sampler_name, sampling_config, min_edge_size, max_edge_size)
        model, training_summary = _train_for_sampler(runtime_config, dataset, sampler_name, sampler, rng)
        results[sampler_name] = _evaluate_sampler(runtime_config, dataset, sampler_name, sampler, model, random.Random(run_seed + 17))
        results[sampler_name].update(training_summary)
        results[sampler_name].update(calibration_metrics)
        append_status(_status_path(runtime_config), "sampler_finished", sampler=sampler_name)

    return {
        "dataset": {
            "source": config["data"]["source"],
            "num_nodes": dataset.num_nodes,
            "train_edges": len(dataset.train_edges),
            "val_edges": len(dataset.val_edges),
            "test_edges": len(dataset.test_edges),
            "has_node_features": dataset.node_features is not None,
        },
        "samplers": results,
    }


def _build_runtime_config(
    config: Mapping[str, object],
    dataset: HypergraphDataset,
    seed: int,
) -> tuple[dict[str, object], dict[str, float]]:
    runtime_config = copy.deepcopy(dict(config))
    sampling_config, calibration_metrics = calibrate_sampling_config(
        sampling_config=runtime_config["sampling"],
        train_edges=dataset.train_edges,
        num_nodes=dataset.num_nodes,
        positive_edges=dataset.train_positive_edges,
        seed=seed,
    )
    runtime_config["sampling"] = sampling_config
    return runtime_config, calibration_metrics


def _load_dataset(config: Mapping[str, object], seed: int) -> HypergraphDataset:
    data_config = config["data"]
    if data_config["source"] == "synthetic":
        return generate_synthetic_hypergraph(data_config, seed)
    if data_config["source"] == "txt":
        return load_hyperedges_from_txt(data_config, seed)
    raise ValueError(f"unsupported data source: {data_config['source']}")


def _train_for_sampler(
    config: Mapping[str, object],
    dataset: HypergraphDataset,
    sampler_name: str,
    sampler: NegativeSampler,
    rng: random.Random,
) -> tuple[HyperedgePredictor | SetHyperedgePredictor | BipartiteHyperedgePredictor | StrongHyperedgePredictor, dict[str, float]]:
    model_config = config["model"]
    training_config = config["training"]
    model = _build_model(model_config, dataset)
    pretraining_summary: dict[str, float] = {}
    if isinstance(model, StrongHyperedgePredictor):
        pretraining_config = model_config.get("pretraining", {})
        if not isinstance(pretraining_config, Mapping):
            raise TypeError("model.pretraining must be a mapping")
        summary = model.pretrain(
            epochs=int(pretraining_config.get("epochs", 30)),
            learning_rate=float(pretraining_config.get("learning_rate", 0.001)),
            weight_decay=float(pretraining_config.get("weight_decay", 0.0001)),
            mask_probability=float(pretraining_config.get("mask_probability", 0.15)),
            contrastive_temperature=float(pretraining_config.get("contrastive_temperature", 0.2)),
        )
        pretraining_summary = {f"pretrain_{key}": float(value) for key, value in summary.__dict__.items()}

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training_config["learning_rate"]),
        weight_decay=float(training_config["weight_decay"]),
    )
    criterion = nn.BCEWithLogitsLoss(reduction="none")
    negatives_per_positive = int(training_config["negatives_per_positive"])
    adaptive_hardness_state = _build_adaptive_hardness_state(config["sampling"])
    proposal_mixture_state = _build_proposal_mixture_state(config["sampling"])
    residual_safe_mixture_state = _build_residual_safe_mixture_state(config["sampling"])
    negative_loss_weighter = _build_negative_loss_weighter(config, dataset)
    negative_label_objective = _build_negative_label_objective(config)
    loss_weight_summaries: list[dict[str, float]] = []

    use_validation_checkpoint = bool(training_config.get("use_validation_checkpoint", True))
    validation_interval = max(1, int(training_config.get("validation_interval", 1)))
    early_stopping_patience = max(0, int(training_config.get("early_stopping_patience", 0)))
    epochs_without_improvement = 0
    best_state = copy.deepcopy(model.state_dict())
    best_epoch = 0
    best_val_metrics = {"val_auc": 0.0, "val_aupr": 0.0}
    validation_batch = _build_validation_batch(config, dataset, random.Random(7919))

    for epoch in range(int(training_config["epochs"])):
        model.train()
        _apply_proposal_mixture_to_sampler(sampler, proposal_mixture_state)
        _apply_residual_safe_mixture_to_sampler(sampler, residual_safe_mixture_state)
        negative_batch = _sample_training_negatives(
            config,
            sampler_name,
            sampler,
            model,
            validation_batch=validation_batch,
            source_edges=dataset.train_edges,
            num_nodes=dataset.num_nodes,
            positive_edges=dataset.train_positive_edges,
            rng=rng,
            negatives_per_positive=negatives_per_positive,
            epoch=epoch + 1,
            adaptive_hardness_state=adaptive_hardness_state,
        )
        model.train()
        positive_edges = dataset.train_edges
        negative_edges = negative_batch.edges
        edges = positive_edges + negative_edges
        labels = torch.tensor(
            [1.0] * len(positive_edges) + [0.0] * len(negative_edges),
            device=next(model.parameters()).device,
        )
        loss_weights = _loss_weights_for_batch(
            positive_count=len(positive_edges),
            negative_batch=negative_batch,
            negative_loss_weighter=negative_loss_weighter,
            epoch=epoch + 1,
        )
        if negative_loss_weighter is not None:
            loss_weight_summaries.append(negative_loss_weighter.last_summary())

        optimizer.zero_grad()
        logits = model.forward_edges(edges)
        loss = _training_loss(
            logits=logits,
            labels=labels,
            positive_count=len(positive_edges),
            loss_weights=loss_weights,
            criterion=criterion,
            objective=negative_label_objective,
            epoch=epoch + 1,
        )
        loss.backward()
        optimizer.step()

        if use_validation_checkpoint and (epoch + 1) % validation_interval == 0:
            val_metrics = _evaluate_fixed_edges(model, validation_batch["edges"], validation_batch["labels"])
            if val_metrics["aupr"] >= best_val_metrics["val_aupr"]:
                best_state = copy.deepcopy(model.state_dict())
                best_epoch = epoch + 1
                best_val_metrics = {
                    "val_auc": val_metrics["auc"],
                    "val_aupr": val_metrics["aupr"],
                }
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += validation_interval
            _write_epoch_status(
                config=config,
                sampler_name=sampler_name,
                epoch=epoch + 1,
                loss=float(loss.detach().cpu().item()),
                val_metrics=val_metrics,
                best_epoch=best_epoch,
                best_val_metrics=best_val_metrics,
                adaptive_hardness_state=adaptive_hardness_state,
                proposal_mixture_state=proposal_mixture_state,
                residual_safe_mixture_state=residual_safe_mixture_state,
            )
            if adaptive_hardness_state is not None:
                _update_adaptive_hardness_state(
                    adaptive_hardness_state,
                    val_aupr=val_metrics["aupr"],
                    negative_metadata=negative_batch.metadata,
                )
            if proposal_mixture_state is not None:
                _update_proposal_mixture_state(
                    proposal_mixture_state,
                    negative_metadata=negative_batch.metadata,
                )
            if residual_safe_mixture_state is not None:
                _update_residual_safe_mixture_state(
                    residual_safe_mixture_state,
                    negative_metadata=negative_batch.metadata,
                    val_aupr=val_metrics["aupr"],
                )
            if early_stopping_patience > 0 and epochs_without_improvement >= early_stopping_patience:
                append_status(
                    _status_path(config),
                    "early_stop",
                    sampler=sampler_name,
                    epoch=epoch + 1,
                    patience=early_stopping_patience,
                    best_epoch=best_epoch,
                    best_val_aupr=best_val_metrics["val_aupr"],
                )
                break
        elif not use_validation_checkpoint:
            _write_epoch_status(
                config=config,
                sampler_name=sampler_name,
                epoch=epoch + 1,
                loss=float(loss.detach().cpu().item()),
                val_metrics=None,
                best_epoch=epoch + 1,
                best_val_metrics=best_val_metrics,
                adaptive_hardness_state=adaptive_hardness_state,
                proposal_mixture_state=proposal_mixture_state,
                residual_safe_mixture_state=residual_safe_mixture_state,
            )
            if proposal_mixture_state is not None:
                _update_proposal_mixture_state(
                    proposal_mixture_state,
                    negative_metadata=negative_batch.metadata,
                )
            if residual_safe_mixture_state is not None:
                _update_residual_safe_mixture_state(
                    residual_safe_mixture_state,
                    negative_metadata=negative_batch.metadata,
                    val_aupr=None,
                )

    if use_validation_checkpoint:
        model.load_state_dict(best_state)
    else:
        best_epoch = int(training_config["epochs"])

    training_summary = {
        "best_epoch": float(best_epoch),
        "val_auc": float(best_val_metrics["val_auc"]),
        "val_aupr": float(best_val_metrics["val_aupr"]),
        **negative_label_objective.summary(),
        **pretraining_summary,
    }
    if negative_loss_weighter is not None:
        training_summary.update(_summarize_loss_weights(loss_weight_summaries))
    if adaptive_hardness_state is not None:
        _store_final_adaptive_hardness_config(config["sampling"], adaptive_hardness_state)
        training_summary.update(adaptive_hardness_state.summary())
    if proposal_mixture_state is not None:
        _store_final_proposal_mixture_config(config["sampling"], proposal_mixture_state)
        _apply_proposal_mixture_to_sampler(sampler, proposal_mixture_state)
        training_summary.update(proposal_mixture_state.summary())
    if residual_safe_mixture_state is not None:
        _store_final_residual_safe_mixture_config(config["sampling"], residual_safe_mixture_state)
        _apply_residual_safe_mixture_to_sampler(sampler, residual_safe_mixture_state)
        training_summary.update(residual_safe_mixture_state.summary())
    return model, training_summary


def _build_model(
    model_config: Mapping[str, object],
    dataset: HypergraphDataset,
) -> HyperedgePredictor | SetHyperedgePredictor | BipartiteHyperedgePredictor | StrongHyperedgePredictor:
    model_type = str(model_config.get("type", "mean"))
    model_class = _model_class(model_type)
    shared_kwargs = {
        "num_nodes": dataset.num_nodes,
        "embedding_dim": int(model_config["embedding_dim"]),
        "hidden_dim": int(model_config["hidden_dim"]),
        "dropout": float(model_config["dropout"]),
        "node_features": dataset.node_features,
    }
    device = _resolve_model_device(model_config)
    if model_type == "bipartite":
        model = model_class(
            **shared_kwargs,
            train_edges=dataset.train_edges,
            message_passing_layers=int(model_config.get("message_passing_layers", 1)),
            use_residual=bool(model_config.get("use_residual", False)),
            use_layer_norm=bool(model_config.get("use_layer_norm", False)),
            representation_fusion_layers=int(model_config.get("representation_fusion_layers", 1)),
        )
        return model.to(device)
    if model_type == "strong_sparse":
        model = model_class(
            **shared_kwargs,
            train_edges=dataset.train_edges,
            message_passing_layers=int(model_config.get("message_passing_layers", 3)),
            freeze_encoder=bool(model_config.get("freeze_encoder", True)),
        )
        return model.to(device)
    if model_type in BENCHMARK_BACKBONES:
        model = BenchmarkBackbonePredictor(
            **shared_kwargs,
            backbone=model_type,
            train_edges=dataset.train_edges,
            message_passing_layers=int(model_config.get("message_passing_layers", 2)),
        )
        return model.to(device)
    return model_class(**shared_kwargs).to(device)


def _resolve_model_device(model_config: Mapping[str, object]) -> torch.device:
    requested = str(model_config.get("device", "cpu"))
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"requested device {requested!r}, but CUDA is not available")
    return torch.device(requested)


def _model_class(
    model_type: str,
) -> type[HyperedgePredictor] | type[SetHyperedgePredictor] | type[BipartiteHyperedgePredictor] | type[StrongHyperedgePredictor]:
    if model_type == "mean":
        return HyperedgePredictor
    if model_type == "set_stats":
        return SetHyperedgePredictor
    if model_type == "bipartite":
        return BipartiteHyperedgePredictor
    if model_type == "strong_sparse":
        return StrongHyperedgePredictor
    if model_type in BENCHMARK_BACKBONES:
        return BenchmarkBackbonePredictor
    raise ValueError(f"unknown model type: {model_type}")


@dataclass
class _NegativeLabelObjective:
    loss_type: str
    q: float
    start_epoch: int
    positive_prior: float | None
    positive_prior_min: float
    positive_prior_max: float
    nnpu_floor: float
    _last_positive_prior: float = 0.0

    def loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        positive_count: int,
        loss_weights: torch.Tensor,
        criterion: nn.Module,
        epoch: int,
    ) -> torch.Tensor:
        if self.loss_type == "bce" or epoch < self.start_epoch:
            losses = criterion(logits, labels)
            return (losses * loss_weights.to(losses.device)).mean()

        if self.loss_type == "gce":
            positive_logits = logits[:positive_count]
            negative_logits = logits[positive_count:]
            positive_labels = labels[:positive_count]
            positive_losses = criterion(positive_logits, positive_labels)
            negative_prob = (1.0 - torch.sigmoid(negative_logits)).clamp(1e-7, 1.0)
            if self.q <= 1e-7:
                negative_losses = -torch.log(negative_prob)
            else:
                negative_losses = (1.0 - torch.pow(negative_prob, self.q)) / self.q
            weighted_losses = torch.cat([positive_losses, negative_losses]) * loss_weights.to(logits.device)
            return weighted_losses.mean()

        if self.loss_type == "nnpu":
            positive_logits = logits[:positive_count]
            negative_logits = logits[positive_count:]
            positive_losses = criterion(positive_logits, torch.ones_like(positive_logits))
            positive_as_negative_losses = criterion(positive_logits, torch.zeros_like(positive_logits))
            negative_losses = criterion(negative_logits, torch.zeros_like(negative_logits))
            negative_weights = loss_weights[positive_count:].to(logits.device)
            positive_prior = self._batch_positive_prior(negative_weights)
            positive_risk = positive_losses.mean()
            unlabeled_negative_risk = (negative_losses * negative_weights).mean()
            positive_negative_risk = positive_as_negative_losses.mean()
            corrected_negative_risk = unlabeled_negative_risk - positive_prior * positive_negative_risk
            non_negative_risk = torch.clamp(corrected_negative_risk, min=self.nnpu_floor)
            return positive_risk + non_negative_risk

        if self.loss_type == "pairwise_ranking":
            positive_scores = torch.sigmoid(logits[:positive_count])
            negative_scores = torch.sigmoid(logits[positive_count:])
            if negative_scores.numel() % positive_count != 0:
                raise ValueError(
                    "pairwise ranking requires an equal number of negatives "
                    "per positive"
                )
            negative_scores = negative_scores.view(positive_count, -1).mean(dim=1)
            return torch.nn.functional.softplus(
                negative_scores - positive_scores
            ).mean()

        raise ValueError(f"unknown negative label loss type: {self.loss_type}")

    def summary(self) -> dict[str, float]:
        return {
            "negative_label_loss_gce_enabled": 1.0 if self.loss_type == "gce" else 0.0,
            "negative_label_loss_nnpu_enabled": 1.0 if self.loss_type == "nnpu" else 0.0,
            "negative_label_loss_pairwise_ranking_enabled": 1.0 if self.loss_type == "pairwise_ranking" else 0.0,
            "negative_label_loss_q": float(self.q if self.loss_type == "gce" else 0.0),
            "negative_label_loss_positive_prior": float(self._last_positive_prior if self.loss_type == "nnpu" else 0.0),
            "negative_label_loss_start_epoch": float(self.start_epoch),
        }

    def _batch_positive_prior(self, negative_weights: torch.Tensor) -> torch.Tensor:
        if self.positive_prior is None:
            prior = (1.0 - negative_weights).mean().detach()
        else:
            prior = torch.tensor(float(self.positive_prior), device=negative_weights.device)
        prior = torch.clamp(prior, min=self.positive_prior_min, max=self.positive_prior_max)
        self._last_positive_prior = float(prior.detach().cpu().item())
        return prior


def _build_negative_label_objective(config: Mapping[str, object]) -> _NegativeLabelObjective:
    training_config = config["training"]
    objective_config = training_config.get("negative_label_loss", {})
    if not isinstance(objective_config, Mapping):
        objective_config = {}
    loss_type = str(objective_config.get("type", "bce")).lower()
    aliases = {
        "binary_cross_entropy": "bce",
        "generalized_cross_entropy": "gce",
        "non_negative_pu": "nnpu",
        "non_negative_pu_risk": "nnpu",
        "ranking": "pairwise_ranking",
        "nhp_ranking": "pairwise_ranking",
    }
    loss_type = aliases.get(loss_type, loss_type)
    if loss_type not in {"bce", "gce", "nnpu", "pairwise_ranking"}:
        raise ValueError(f"unsupported negative_label_loss.type: {loss_type}")
    q = float(objective_config.get("q", 0.7))
    if q < 0.0 or q > 1.0:
        raise ValueError("negative_label_loss.q must be in [0, 1]")
    start_epoch = int(objective_config.get("start_epoch", 1))
    if bool(objective_config.get("start_after_curriculum", False)):
        curriculum_config = config["sampling"].get("curriculum", {})
        if isinstance(curriculum_config, Mapping):
            start_epoch = max(start_epoch, int(curriculum_config.get("warmup_epochs", 0)) + 1)
    return _NegativeLabelObjective(
        loss_type=loss_type,
        q=q,
        start_epoch=max(1, start_epoch),
        positive_prior=_optional_float(objective_config.get("positive_prior")),
        positive_prior_min=float(objective_config.get("positive_prior_min", 0.0)),
        positive_prior_max=float(objective_config.get("positive_prior_max", 0.5)),
        nnpu_floor=float(objective_config.get("nnpu_floor", 0.0)),
    )


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, str) and value.strip().lower() in {"", "none", "auto"}:
        return None
    return float(value)


def _training_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    positive_count: int,
    loss_weights: torch.Tensor,
    criterion: nn.Module,
    objective: _NegativeLabelObjective,
    epoch: int = 1,
) -> torch.Tensor:
    return objective.loss(
        logits=logits,
        labels=labels,
        positive_count=positive_count,
        loss_weights=loss_weights,
        criterion=criterion,
        epoch=epoch,
    )


@dataclass
class _NegativeLossWeighter:
    positive_index: _PositiveEdgeIndex
    closure_index: ClosureRiskIndex
    cowalk_index: CoWalkRiskIndex | None
    hitting_index: HittingRiskIndex | None
    residual_index: DegreeCorrectedResidualRiskIndex | None
    nearest_bound: float
    closure_bound: float
    cowalk_bound: float
    hitting_bound: float
    residual_bound: float
    nearest_weight: float
    closure_weight: float
    cowalk_weight: float
    hitting_weight: float
    residual_weight: float
    use_cowalk_risk: bool
    use_hitting_risk: bool
    use_residual_risk: bool
    beta: float
    min_weight: float
    start_epoch: int
    cache: dict[tuple[tuple[int, ...], tuple[int, ...]], float]
    _last_weights: list[float]
    _last_risks: list[float]

    def weights(self, negative_batch: NegativeSampleBatch) -> list[float]:
        weights: list[float] = []
        risks: list[float] = []
        for edge, source_edge in zip(negative_batch.edges, negative_batch.source_edges):
            risk = self._risk(edge, source_edge)
            weight = max(self.min_weight, math.exp(-self.beta * risk))
            risks.append(float(risk))
            weights.append(float(weight))
        self._last_weights = weights
        self._last_risks = risks
        return weights

    def last_summary(self) -> dict[str, float]:
        if not self._last_weights:
            return {
                "weighted_negative_loss_enabled": 1.0,
                "negative_loss_weight_mean": 1.0,
                "negative_loss_weight_min": 1.0,
                "negative_loss_weight_max": 1.0,
                "negative_loss_positive_support_risk_mean": 0.0,
            }
        return {
            "weighted_negative_loss_enabled": 1.0,
            "negative_loss_weight_mean": float(sum(self._last_weights) / len(self._last_weights)),
            "negative_loss_weight_min": float(min(self._last_weights)),
            "negative_loss_weight_max": float(max(self._last_weights)),
            "negative_loss_positive_support_risk_mean": float(sum(self._last_risks) / len(self._last_risks)),
        }

    def _risk(self, edge: tuple[int, ...], source_edge: tuple[int, ...]) -> float:
        cache_key = (edge, source_edge)
        cached_risk = self.cache.get(cache_key)
        if cached_risk is not None:
            return cached_risk
        nearest = self.positive_index.nearest_similarity(edge, excluded_edge=source_edge)
        closure = self.closure_index.risk(edge)
        cowalk = self.cowalk_index.risk(edge) if self.cowalk_index is not None else 0.0
        hitting = self.hitting_index.risk(edge) if self.hitting_index is not None else 0.0
        residual = self.residual_index.risk(edge) if self.residual_index is not None else 0.0
        risk = self.nearest_weight * _safe_ratio(nearest, self.nearest_bound)
        risk += self.closure_weight * _safe_ratio(closure, self.closure_bound)
        if self.use_cowalk_risk:
            risk += self.cowalk_weight * _safe_ratio(cowalk, self.cowalk_bound)
        if self.use_hitting_risk:
            risk += self.hitting_weight * _safe_ratio(hitting, self.hitting_bound)
        if self.use_residual_risk:
            risk += self.residual_weight * _safe_ratio(residual, self.residual_bound)
        if len(self.cache) > 200_000:
            self.cache.clear()
        self.cache[cache_key] = float(risk)
        return float(risk)


def _build_negative_loss_weighter(
    config: Mapping[str, object],
    dataset: HypergraphDataset,
) -> _NegativeLossWeighter | None:
    training_config = config["training"]
    weighted_config = training_config.get("weighted_negative_loss")
    if not isinstance(weighted_config, Mapping) or not bool(weighted_config.get("enabled", False)):
        return None
    sampling_config = config["sampling"]
    risk_budget = float(sampling_config.get("risk_budget", 1.0))
    beta = float(weighted_config.get("beta", 0.0))
    if beta <= 0.0:
        half_life_multiplier = float(weighted_config.get("half_life_multiplier", 1.0))
        beta = math.log(2.0) / max(half_life_multiplier * risk_budget, 1e-9)
    min_weight = float(weighted_config.get("min_weight", 0.1))
    start_epoch = int(weighted_config.get("start_epoch", 1))
    if bool(weighted_config.get("start_after_curriculum", False)):
        curriculum_config = config["sampling"].get("curriculum", {})
        if isinstance(curriculum_config, Mapping) and bool(curriculum_config.get("enabled", False)):
            start_epoch = max(start_epoch, int(curriculum_config.get("warmup_epochs", 0)) + 1)
    max_cowalk_neighbors = int(sampling_config.get("max_cowalk_neighbors", 128))
    max_hitting_neighbors = int(sampling_config.get("max_hitting_neighbors", 128))
    train_positive_edges = dataset.train_positive_edges
    use_cowalk_risk = bool(sampling_config.get("use_cowalk_risk", False))
    use_hitting_risk = bool(sampling_config.get("use_hitting_risk", False))
    use_residual_risk = bool(sampling_config.get("use_residual_risk", False))
    return _NegativeLossWeighter(
        positive_index=_PositiveEdgeIndex(train_positive_edges),
        closure_index=ClosureRiskIndex(train_positive_edges),
        cowalk_index=CoWalkRiskIndex(train_positive_edges, max_neighbors_per_node=max_cowalk_neighbors) if use_cowalk_risk else None,
        hitting_index=HittingRiskIndex(
            train_positive_edges,
            max_neighbors_per_node=max_hitting_neighbors,
            two_step_weight=float(sampling_config.get("hitting_two_step_weight", 0.5)),
        ) if use_hitting_risk else None,
        residual_index=DegreeCorrectedResidualRiskIndex(
            train_positive_edges,
            max_pairs_per_edge=int(sampling_config.get("max_replacement_pairs_per_edge", 2000)),
        ) if use_residual_risk else None,
        nearest_bound=float(sampling_config.get("nearest_positive_upper_bound", 1.0)),
        closure_bound=float(sampling_config.get("closure_risk_upper_bound", 1.0)),
        cowalk_bound=float(sampling_config.get("cowalk_risk_upper_bound", 1.0)),
        hitting_bound=float(sampling_config.get("hitting_risk_upper_bound", 1.0)),
        residual_bound=float(sampling_config.get("residual_risk_upper_bound", 1.0)),
        nearest_weight=float(sampling_config.get("nearest_risk_weight", 1.0)),
        closure_weight=float(sampling_config.get("closure_risk_weight", 1.0)),
        cowalk_weight=float(sampling_config.get("cowalk_risk_weight", 1.0)),
        hitting_weight=float(sampling_config.get("hitting_risk_weight", 1.0)),
        residual_weight=float(sampling_config.get("residual_risk_weight", 1.0)),
        use_cowalk_risk=use_cowalk_risk,
        use_hitting_risk=use_hitting_risk,
        use_residual_risk=use_residual_risk,
        beta=beta,
        min_weight=max(0.0, min(1.0, min_weight)),
        start_epoch=max(1, start_epoch),
        cache={},
        _last_weights=[],
        _last_risks=[],
    )


def _loss_weights_for_batch(
    positive_count: int,
    negative_batch: NegativeSampleBatch,
    negative_loss_weighter: _NegativeLossWeighter | None,
    epoch: int = 1,
) -> torch.Tensor:
    if negative_loss_weighter is None or epoch < negative_loss_weighter.start_epoch:
        return torch.ones(positive_count + len(negative_batch.edges))
    negative_weights = negative_loss_weighter.weights(negative_batch)
    return torch.tensor([1.0] * positive_count + negative_weights, dtype=torch.float32)


def _summarize_loss_weights(summaries: list[dict[str, float]]) -> dict[str, float]:
    if not summaries:
        return {
            "weighted_negative_loss_enabled": 1.0,
            "negative_loss_weight_mean": 1.0,
            "negative_loss_weight_min": 1.0,
            "negative_loss_weight_max": 1.0,
            "negative_loss_positive_support_risk_mean": 0.0,
        }
    return {
        "weighted_negative_loss_enabled": 1.0,
        "negative_loss_weight_mean": float(sum(item["negative_loss_weight_mean"] for item in summaries) / len(summaries)),
        "negative_loss_weight_min": float(min(item["negative_loss_weight_min"] for item in summaries)),
        "negative_loss_weight_max": float(max(item["negative_loss_weight_max"] for item in summaries)),
        "negative_loss_positive_support_risk_mean": float(sum(item["negative_loss_positive_support_risk_mean"] for item in summaries) / len(summaries)),
    }


def _safe_ratio(value: float, denominator: float) -> float:
    if denominator <= 0.0:
        return 0.0 if value <= 0.0 else float("inf")
    return value / denominator


def _sampler_seed_offset(sampler_name: str) -> int:
    fixed_offsets = {
        "random": 0,
        "size_matched": 1000,
        "anchored_random": 2000,
        "anchored_safe": 3000,
        "risk_controlled": 5000,
        "model_aware_risk_controlled": 8000,
    }
    if sampler_name in fixed_offsets:
        return fixed_offsets[sampler_name]
    digest = hashlib.md5(sampler_name.encode("utf-8")).hexdigest()
    return 10_000 + int(digest[:6], 16) % 90_000


def _evaluate_sampler(
    config: Mapping[str, object],
    dataset: HypergraphDataset,
    sampler_name: str,
    sampler: NegativeSampler,
    model: HyperedgePredictor | SetHyperedgePredictor | BipartiteHyperedgePredictor,
    rng: random.Random,
) -> dict[str, float]:
    negatives_per_positive = int(config["training"]["negatives_per_positive"])
    eval_sampler = SizeMatchedSampler(max_attempts=int(config["sampling"].get("max_attempts", 100)))
    eval_negative_batch = eval_sampler.sample(
        source_edges=dataset.test_edges,
        num_nodes=dataset.num_nodes,
        positive_edges=dataset.positive_edges,
        rng=rng,
        negatives_per_positive=negatives_per_positive,
    )
    eval_edges = dataset.test_edges + eval_negative_batch.edges
    labels = [1] * len(dataset.test_edges) + [0] * len(eval_negative_batch.edges)
    scores = model.predict_scores(eval_edges)
    metrics = binary_prediction_metrics(labels, scores)
    evaluation_config = config.get("evaluation", {})
    if isinstance(evaluation_config, Mapping) and not bool(evaluation_config.get("quality_audit", True)):
        return metrics

    validation_batch = _build_validation_batch(config, dataset, random.Random(3571))
    quality_batch = _sample_training_negatives(
        config,
        sampler_name,
        sampler,
        model,
        validation_batch=validation_batch,
        source_edges=dataset.test_edges,
        num_nodes=dataset.num_nodes,
        positive_edges=dataset.positive_edges,
        rng=rng,
        negatives_per_positive=negatives_per_positive,
    )
    hardness_scores = model.predict_scores(quality_batch.edges)
    metrics.update(quality_batch.metadata)
    metrics.update(
        negative_quality_metrics(
            negative_edges=quality_batch.edges,
            source_edges=quality_batch.source_edges,
            positive_edges=dataset.positive_edges,
            hardness_scores=hardness_scores,
            hard_negative_lower_bound=float(config["sampling"].get("hard_negative_lower_bound", 0.3)),
            hard_negative_upper_bound=float(config["sampling"].get("hard_negative_upper_bound", 0.7)),
        )
    )
    metrics.update(
        _future_positive_probe_metrics(
            config=config,
            dataset=dataset,
            sampler_name=sampler_name,
            sampler=sampler,
            model=model,
            validation_batch=validation_batch,
            rng=rng,
            negatives_per_positive=negatives_per_positive,
        )
    )
    return metrics


def _sample_training_negatives(
    config: Mapping[str, object],
    sampler_name: str,
    sampler: NegativeSampler,
    model: HyperedgePredictor | SetHyperedgePredictor | BipartiteHyperedgePredictor,
    source_edges: list[tuple[int, ...]],
    num_nodes: int,
    positive_edges: set[tuple[int, ...]],
    rng: random.Random,
    negatives_per_positive: int,
    epoch: int | None = None,
    adaptive_hardness_state: "_AdaptiveHardnessState | None" = None,
    validation_batch: dict[str, list[tuple[int, ...]] | list[int]] | None = None,
) -> NegativeSampleBatch:
    sampling_config = _effective_sampling_config(config["sampling"], epoch, adaptive_hardness_state)
    if not _uses_model_aware_rerank(sampler_name, sampling_config):
        return sampler.sample(
            source_edges=source_edges,
            num_nodes=num_nodes,
            positive_edges=positive_edges,
            rng=rng,
            negatives_per_positive=negatives_per_positive,
        )

    scorer = _build_sampling_scorer(model, sampling_config, validation_batch)
    rerank_config = RerankConfig.from_mapping(sampling_config)
    candidate_count = negatives_per_positive * rerank_config.candidate_multiplier
    if _uses_adversarial_proposal(sampling_config) and hasattr(sampler, "sample_boundary_guided"):
        candidate_batch = sampler.sample_boundary_guided(
            source_edges=source_edges,
            num_nodes=num_nodes,
            positive_edges=positive_edges,
            scorer=scorer,
            rng=rng,
            negatives_per_positive=candidate_count,
            target_score=rerank_config.target_score,
            target_score_min=rerank_config.target_score_min,
            target_score_max=rerank_config.target_score_max,
            score_lower_bound=rerank_config.score_lower_bound,
            score_upper_bound=rerank_config.score_upper_bound,
        )
    else:
        candidate_batch = sampler.sample(
            source_edges=source_edges,
            num_nodes=num_nodes,
            positive_edges=positive_edges,
            rng=rng,
            negatives_per_positive=candidate_count,
        )
    reranker = ModelAwareReranker(rerank_config)
    return reranker.select(
        candidate_batch=candidate_batch,
        scorer=scorer,
        num_sources=len(source_edges),
        negatives_per_positive=negatives_per_positive,
        rng=rng,
    )


def _build_sampling_scorer(
    model: HyperedgePredictor | SetHyperedgePredictor | BipartiteHyperedgePredictor,
    sampling_config: Mapping[str, object] | None = None,
    validation_batch: dict[str, list[tuple[int, ...]] | list[int]] | None = None,
) -> HyperedgePredictor | SetHyperedgePredictor | "_CachedBipartiteScorer" | "_CachedHdsScorer":
    del sampling_config, validation_batch
    if isinstance(model, BipartiteHyperedgePredictor):
        return _CachedBipartiteScorer(model)
    if isinstance(model, BenchmarkBackbonePredictor) and model.backbone == "hds":
        return _CachedHdsScorer(model)
    return model


class _CachedBipartiteScorer:
    def __init__(self, model: BipartiteHyperedgePredictor) -> None:
        self.model = model
        self.model.eval()
        with torch.no_grad():
            self.node_representations = self.model.encode_all_nodes().detach()
        self._edge_cache: dict[tuple[int, ...], torch.Tensor] = {}

    def eval(self) -> "_CachedBipartiteScorer":
        self.model.eval()
        return self

    def predict_scores(self, edges: list[tuple[int, ...]]) -> list[float]:
        self.model.eval()
        with torch.no_grad():
            edge_representations = [self._encode_edge(edge) for edge in edges]
            if not edge_representations:
                return []
            stacked = torch.stack(edge_representations, dim=0)
            logits = self.model.classifier(stacked).squeeze(-1)
            return torch.sigmoid(logits).detach().cpu().tolist()

    def encode_edges(self, edges: list[tuple[int, ...]]) -> list[list[float]]:
        self.model.eval()
        with torch.no_grad():
            return [
                self._encode_edge(edge).detach().cpu().tolist()
                for edge in edges
            ]

    def _encode_edge(self, edge: tuple[int, ...]) -> torch.Tensor:
        cached = self._edge_cache.get(edge)
        if cached is not None:
            return cached
        encoded = self.model.encode_edge_with_nodes(edge, self.node_representations).detach()
        self._edge_cache[edge] = encoded
        return encoded


class _CachedHdsScorer:
    def __init__(self, model: BenchmarkBackbonePredictor) -> None:
        self.model = model
        self.model.eval()
        with torch.no_grad():
            nodes = self.model._initial_nodes()
            self.transformed_nodes = self.model.set_phi(nodes).detach()
        self._edge_cache: dict[tuple[int, ...], torch.Tensor] = {}

    def eval(self) -> "_CachedHdsScorer":
        self.model.eval()
        return self

    def predict_scores(self, edges: list[tuple[int, ...]]) -> list[float]:
        self.model.eval()
        with torch.no_grad():
            representations = self._encode_edges_tensor(edges)
            if representations.numel() == 0:
                return []
            logits = self.model.classifier(representations).squeeze(-1)
            return torch.sigmoid(logits).detach().cpu().tolist()

    def encode_edges(self, edges: list[tuple[int, ...]]) -> list[list[float]]:
        self.model.eval()
        with torch.no_grad():
            return self._encode_edges_tensor(edges).detach().cpu().tolist()

    def _encode_edges_tensor(self, edges: list[tuple[int, ...]]) -> torch.Tensor:
        if not edges:
            return self.transformed_nodes.new_empty((0, self.model.embedding_dim * 4 + 1))
        missing = [edge for edge in dict.fromkeys(edges) if edge not in self._edge_cache]
        for start in range(0, len(missing), 512):
            chunk = missing[start : start + 512]
            encoded_chunk = _pool_stat_edge_chunk(chunk, self.transformed_nodes).detach()
            for edge, encoded in zip(chunk, encoded_chunk):
                self._edge_cache[edge] = encoded
        return torch.stack([self._edge_cache[edge] for edge in edges], dim=0)


def _uses_model_aware_rerank(sampler_name: str, sampling_config: Mapping[str, object]) -> bool:
    if sampler_name == "model_aware_risk_controlled":
        return True
    return bool(sampling_config.get("model_aware_rerank", False))


def _uses_adversarial_proposal(sampling_config: Mapping[str, object]) -> bool:
    proposal_config = sampling_config.get("adversarial_proposal")
    if not isinstance(proposal_config, Mapping):
        return False
    return bool(proposal_config.get("enabled", False))


def _effective_sampling_config(
    sampling_config: Mapping[str, object],
    epoch: int | None,
    adaptive_hardness_state: "_AdaptiveHardnessState | None" = None,
) -> dict[str, object]:
    effective_config = dict(sampling_config)
    curriculum = sampling_config.get("curriculum")
    if epoch is None or not isinstance(curriculum, Mapping) or not bool(curriculum.get("enabled", False)):
        _apply_adaptive_hardness_config(effective_config, adaptive_hardness_state)
        return effective_config

    warmup_epochs = max(0, int(curriculum.get("warmup_epochs", 0)))
    if epoch <= warmup_epochs:
        effective_config["rerank_candidate_multiplier"] = max(
            1,
            int(curriculum.get("warmup_candidate_multiplier", effective_config.get("rerank_candidate_multiplier", 2))),
        )
        effective_config["rerank_selection_strategy"] = str(
            curriculum.get("warmup_selection_strategy", "topk_sample")
        )
        effective_config["rerank_top_k"] = max(
            1,
            int(curriculum.get("warmup_top_k", effective_config.get("rerank_top_k", 2))),
        )
        _apply_curriculum_rerank_targets(effective_config, curriculum, "warmup")
        effective_config["curriculum_phase"] = "warmup"
        _apply_adaptive_hardness_config(effective_config, adaptive_hardness_state)
        return effective_config

    effective_config["rerank_candidate_multiplier"] = max(
        1,
        int(curriculum.get("target_candidate_multiplier", effective_config.get("rerank_candidate_multiplier", 2))),
    )
    effective_config["rerank_selection_strategy"] = str(
        curriculum.get("target_selection_strategy", effective_config.get("rerank_selection_strategy", "top_score"))
    )
    effective_config["rerank_top_k"] = max(
        1,
        int(curriculum.get("target_top_k", effective_config.get("rerank_top_k", 1))),
    )
    _apply_curriculum_rerank_targets(effective_config, curriculum, "target")
    effective_config["curriculum_phase"] = "target"
    _apply_adaptive_hardness_config(effective_config, adaptive_hardness_state)
    return effective_config


def _apply_curriculum_rerank_targets(
    effective_config: dict[str, object],
    curriculum: Mapping[str, object],
    phase: str,
) -> None:
    target_key = f"{phase}_target_score"
    target_min_key = f"{phase}_target_score_min"
    target_max_key = f"{phase}_target_score_max"
    if target_key in curriculum:
        effective_config["rerank_target_score"] = float(curriculum[target_key])
    if target_min_key in curriculum:
        effective_config["rerank_target_score_min"] = float(curriculum[target_min_key])
    if target_max_key in curriculum:
        effective_config["rerank_target_score_max"] = float(curriculum[target_max_key])


@dataclass
class _AdaptiveHardnessState:
    center: float
    width: float
    min_center: float
    max_center: float
    step_size: float
    min_val_delta: float
    max_risk_fallback_rate: float
    min_in_band_rate: float
    previous_val_aupr: float | None = None
    updates: int = 0
    increase_updates: int = 0
    decrease_updates: int = 0
    last_risk_fallback_rate: float = 0.0
    last_in_band_rate: float = 0.0

    def target_bounds(self, score_lower_bound: float, score_upper_bound: float) -> tuple[float, float]:
        half_width = max(0.0, self.width) / 2.0
        lower = max(score_lower_bound, self.center - half_width)
        upper = min(score_upper_bound, self.center + half_width)
        if lower > upper:
            return self.center, self.center
        return lower, upper

    def summary(self) -> dict[str, float]:
        return {
            "adaptive_hardness_enabled": 1.0,
            "adaptive_hardness_final_center": float(self.center),
            "adaptive_hardness_width": float(self.width),
            "adaptive_hardness_updates": float(self.updates),
            "adaptive_hardness_increase_updates": float(self.increase_updates),
            "adaptive_hardness_decrease_updates": float(self.decrease_updates),
            "adaptive_hardness_last_risk_fallback_rate": float(self.last_risk_fallback_rate),
            "adaptive_hardness_last_in_band_rate": float(self.last_in_band_rate),
        }


def _build_adaptive_hardness_state(sampling_config: Mapping[str, object]) -> _AdaptiveHardnessState | None:
    controller = sampling_config.get("adaptive_hardness")
    if not isinstance(controller, Mapping) or not bool(controller.get("enabled", False)):
        return None
    initial_center = float(controller.get("initial_center", sampling_config.get("rerank_target_score", 0.4)))
    min_center = float(controller.get("min_center", 0.25))
    max_center = float(controller.get("max_center", 0.5))
    return _AdaptiveHardnessState(
        center=_clamp(initial_center, min_center, max_center),
        width=max(0.0, float(controller.get("target_width", 0.12))),
        min_center=min(min_center, max_center),
        max_center=max(min_center, max_center),
        step_size=max(0.0, float(controller.get("step_size", 0.03))),
        min_val_delta=float(controller.get("min_val_delta", -0.001)),
        max_risk_fallback_rate=float(controller.get("max_risk_fallback_rate", 0.35)),
        min_in_band_rate=float(controller.get("min_in_band_rate", 0.15)),
    )


def _apply_adaptive_hardness_config(
    effective_config: dict[str, object],
    adaptive_hardness_state: _AdaptiveHardnessState | None,
) -> None:
    if adaptive_hardness_state is None:
        return
    score_lower_bound = float(effective_config.get("rerank_score_lower_bound", 0.3))
    score_upper_bound = float(effective_config.get("rerank_score_upper_bound", 0.7))
    target_min, target_max = adaptive_hardness_state.target_bounds(score_lower_bound, score_upper_bound)
    effective_config["rerank_target_score"] = float(adaptive_hardness_state.center)
    effective_config["rerank_target_score_min"] = float(target_min)
    effective_config["rerank_target_score_max"] = float(target_max)
    effective_config["adaptive_hardness_target_center"] = float(adaptive_hardness_state.center)
    effective_config["adaptive_hardness_target_width"] = float(adaptive_hardness_state.width)


def _update_adaptive_hardness_state(
    adaptive_hardness_state: _AdaptiveHardnessState,
    val_aupr: float,
    negative_metadata: Mapping[str, object],
) -> None:
    fallback_rate = float(negative_metadata.get("risk_fallback_rate", 0.0))
    in_band_rate = float(
        negative_metadata.get(
            "rerank_selected_in_band_rate",
            negative_metadata.get("adversarial_selected_in_band_rate", 0.0),
        )
    )
    adaptive_hardness_state.last_risk_fallback_rate = fallback_rate
    adaptive_hardness_state.last_in_band_rate = in_band_rate
    previous_val_aupr = adaptive_hardness_state.previous_val_aupr
    adaptive_hardness_state.previous_val_aupr = float(val_aupr)
    if previous_val_aupr is None or adaptive_hardness_state.step_size == 0.0:
        return

    val_delta = float(val_aupr) - previous_val_aupr
    risk_ok = fallback_rate <= adaptive_hardness_state.max_risk_fallback_rate
    validation_stable = val_delta >= adaptive_hardness_state.min_val_delta
    hardness_feasible = in_band_rate >= adaptive_hardness_state.min_in_band_rate
    if risk_ok and validation_stable and hardness_feasible:
        adaptive_hardness_state.center = _clamp(
            adaptive_hardness_state.center + adaptive_hardness_state.step_size,
            adaptive_hardness_state.min_center,
            adaptive_hardness_state.max_center,
        )
        adaptive_hardness_state.increase_updates += 1
        adaptive_hardness_state.updates += 1
        return
    if not risk_ok or not validation_stable:
        adaptive_hardness_state.center = _clamp(
            adaptive_hardness_state.center - adaptive_hardness_state.step_size,
            adaptive_hardness_state.min_center,
            adaptive_hardness_state.max_center,
        )
        adaptive_hardness_state.decrease_updates += 1
        adaptive_hardness_state.updates += 1


def _store_final_adaptive_hardness_config(
    sampling_config: object,
    adaptive_hardness_state: _AdaptiveHardnessState,
) -> None:
    if not isinstance(sampling_config, dict):
        return
    lower = float(sampling_config.get("rerank_score_lower_bound", 0.3))
    upper = float(sampling_config.get("rerank_score_upper_bound", 0.7))
    target_min, target_max = adaptive_hardness_state.target_bounds(lower, upper)
    sampling_config["rerank_target_score"] = float(adaptive_hardness_state.center)
    sampling_config["rerank_target_score_min"] = float(target_min)
    sampling_config["rerank_target_score_max"] = float(target_max)
    sampling_config["adaptive_hardness_final_center"] = float(adaptive_hardness_state.center)


@dataclass
class _ProposalMixtureState:
    probability: float
    min_probability: float
    max_probability: float
    step_size: float
    target_in_band_rate: float
    max_risk_fallback_rate: float
    max_sampling_attempts_mean: float
    min_candidate_pool_size: float
    hardness_weight: float
    risk_weight: float
    cost_weight: float
    diversity_weight: float
    updates: int = 0
    increase_updates: int = 0
    decrease_updates: int = 0
    last_in_band_rate: float = 0.0
    last_risk_fallback_rate: float = 0.0
    last_sampling_attempts_mean: float = 0.0
    last_candidate_pool_size_mean: float = 0.0

    def summary(self) -> dict[str, float]:
        return {
            "proposal_controller_enabled": 1.0,
            "proposal_controller_final_probability": float(self.probability),
            "proposal_controller_updates": float(self.updates),
            "proposal_controller_increase_updates": float(self.increase_updates),
            "proposal_controller_decrease_updates": float(self.decrease_updates),
            "proposal_controller_last_in_band_rate": float(self.last_in_band_rate),
            "proposal_controller_last_risk_fallback_rate": float(self.last_risk_fallback_rate),
            "proposal_controller_last_sampling_attempts_mean": float(self.last_sampling_attempts_mean),
            "proposal_controller_last_candidate_pool_size_mean": float(self.last_candidate_pool_size_mean),
        }


def _build_proposal_mixture_state(sampling_config: Mapping[str, object]) -> _ProposalMixtureState | None:
    controller = sampling_config.get("proposal_controller")
    if not isinstance(controller, Mapping) or not bool(controller.get("enabled", False)):
        return None
    if sampling_config.get("replacement_strategy") != "calibrated_mixture":
        return None
    min_probability = float(controller.get("min_probability", 0.0))
    max_probability = float(controller.get("max_probability", 1.0))
    initial_probability = float(
        controller.get(
            "initial_probability",
            sampling_config.get("mixture_risk_aware_probability", min_probability),
        )
    )
    return _ProposalMixtureState(
        probability=_clamp(initial_probability, min_probability, max_probability),
        min_probability=min(min_probability, max_probability),
        max_probability=max(min_probability, max_probability),
        step_size=max(0.0, float(controller.get("step_size", 0.08))),
        target_in_band_rate=float(controller.get("target_in_band_rate", 0.20)),
        max_risk_fallback_rate=float(controller.get("max_risk_fallback_rate", 0.35)),
        max_sampling_attempts_mean=float(controller.get("max_sampling_attempts_mean", 6.0)),
        min_candidate_pool_size=float(controller.get("min_candidate_pool_size", 0.5)),
        hardness_weight=float(controller.get("hardness_weight", 1.0)),
        risk_weight=float(controller.get("risk_weight", 1.0)),
        cost_weight=float(controller.get("cost_weight", 0.5)),
        diversity_weight=float(controller.get("diversity_weight", 0.5)),
    )


def _apply_proposal_mixture_to_sampler(
    sampler: NegativeSampler,
    proposal_mixture_state: _ProposalMixtureState | None,
) -> None:
    if proposal_mixture_state is None:
        return
    if hasattr(sampler, "mixture_risk_aware_probability"):
        setattr(sampler, "mixture_risk_aware_probability", float(proposal_mixture_state.probability))


def _update_proposal_mixture_state(
    proposal_mixture_state: _ProposalMixtureState,
    negative_metadata: Mapping[str, object],
) -> None:
    if proposal_mixture_state.step_size == 0.0:
        return
    in_band_rate = float(
        negative_metadata.get(
            "rerank_selected_in_band_rate",
            negative_metadata.get("adversarial_selected_in_band_rate", 0.0),
        )
    )
    fallback_rate = float(negative_metadata.get("risk_fallback_rate", 0.0))
    attempts_mean = float(negative_metadata.get("sampling_attempts_mean", 0.0))
    candidate_pool_size = float(negative_metadata.get("candidate_pool_size_mean", 0.0))

    proposal_mixture_state.last_in_band_rate = in_band_rate
    proposal_mixture_state.last_risk_fallback_rate = fallback_rate
    proposal_mixture_state.last_sampling_attempts_mean = attempts_mean
    proposal_mixture_state.last_candidate_pool_size_mean = candidate_pool_size

    hardness_pressure = max(0.0, proposal_mixture_state.target_in_band_rate - in_band_rate)
    risk_pressure = max(0.0, fallback_rate - proposal_mixture_state.max_risk_fallback_rate)
    cost_pressure = _positive_relative_excess(attempts_mean, proposal_mixture_state.max_sampling_attempts_mean)
    diversity_pressure = _positive_relative_excess(
        proposal_mixture_state.min_candidate_pool_size,
        candidate_pool_size,
    )
    signed_pressure = (
        proposal_mixture_state.risk_weight * risk_pressure
        - proposal_mixture_state.hardness_weight * hardness_pressure
        - proposal_mixture_state.cost_weight * cost_pressure
        - proposal_mixture_state.diversity_weight * diversity_pressure
    )
    if signed_pressure == 0.0:
        return
    previous = proposal_mixture_state.probability
    proposal_mixture_state.probability = _clamp(
        previous + proposal_mixture_state.step_size * signed_pressure,
        proposal_mixture_state.min_probability,
        proposal_mixture_state.max_probability,
    )
    if proposal_mixture_state.probability > previous:
        proposal_mixture_state.increase_updates += 1
        proposal_mixture_state.updates += 1
    elif proposal_mixture_state.probability < previous:
        proposal_mixture_state.decrease_updates += 1
        proposal_mixture_state.updates += 1


def _store_final_proposal_mixture_config(
    sampling_config: object,
    proposal_mixture_state: _ProposalMixtureState,
) -> None:
    if not isinstance(sampling_config, dict):
        return
    sampling_config["mixture_risk_aware_probability"] = float(proposal_mixture_state.probability)
    sampling_config["proposal_controller_final_probability"] = float(proposal_mixture_state.probability)


@dataclass
class _ResidualSafeMixtureState:
    probability: float
    min_probability: float
    max_probability: float
    step_size: float
    target_in_band_rate: float
    max_risk_fallback_rate: float
    max_sampling_attempts_mean: float
    min_candidate_pool_size: float
    min_val_delta: float
    hardness_weight: float
    risk_weight: float
    cost_weight: float
    diversity_weight: float
    validation_weight: float
    previous_val_aupr: float | None = None
    updates: int = 0
    increase_updates: int = 0
    decrease_updates: int = 0
    last_in_band_rate: float = 0.0
    last_risk_fallback_rate: float = 0.0
    last_sampling_attempts_mean: float = 0.0
    last_candidate_pool_size_mean: float = 0.0
    last_val_delta: float = 0.0

    def summary(self) -> dict[str, float]:
        return {
            "residual_safe_controller_enabled": 1.0,
            "residual_safe_controller_final_probability": float(self.probability),
            "residual_safe_controller_updates": float(self.updates),
            "residual_safe_controller_increase_updates": float(self.increase_updates),
            "residual_safe_controller_decrease_updates": float(self.decrease_updates),
            "residual_safe_controller_last_in_band_rate": float(self.last_in_band_rate),
            "residual_safe_controller_last_risk_fallback_rate": float(self.last_risk_fallback_rate),
            "residual_safe_controller_last_sampling_attempts_mean": float(self.last_sampling_attempts_mean),
            "residual_safe_controller_last_candidate_pool_size_mean": float(self.last_candidate_pool_size_mean),
            "residual_safe_controller_last_val_delta": float(self.last_val_delta),
        }


def _build_residual_safe_mixture_state(sampling_config: Mapping[str, object]) -> _ResidualSafeMixtureState | None:
    controller = sampling_config.get("residual_safe_controller")
    if not isinstance(controller, Mapping) or not bool(controller.get("enabled", False)):
        return None
    if sampling_config.get("replacement_strategy") != "calibrated_mixture":
        return None
    min_probability = float(controller.get("min_probability", 0.0))
    max_probability = float(controller.get("max_probability", 0.60))
    initial_probability = float(
        controller.get(
            "initial_probability",
            sampling_config.get("mixture_residual_safe_probability", min_probability),
        )
    )
    return _ResidualSafeMixtureState(
        probability=_clamp(initial_probability, min_probability, max_probability),
        min_probability=min(min_probability, max_probability),
        max_probability=max(min_probability, max_probability),
        step_size=max(0.0, float(controller.get("step_size", 0.05))),
        target_in_band_rate=float(controller.get("target_in_band_rate", 0.12)),
        max_risk_fallback_rate=float(controller.get("max_risk_fallback_rate", 0.25)),
        max_sampling_attempts_mean=float(controller.get("max_sampling_attempts_mean", 6.0)),
        min_candidate_pool_size=float(controller.get("min_candidate_pool_size", 0.5)),
        min_val_delta=float(controller.get("min_val_delta", -0.002)),
        hardness_weight=float(controller.get("hardness_weight", 1.0)),
        risk_weight=float(controller.get("risk_weight", 1.0)),
        cost_weight=float(controller.get("cost_weight", 0.5)),
        diversity_weight=float(controller.get("diversity_weight", 0.5)),
        validation_weight=float(controller.get("validation_weight", 1.0)),
    )


def _apply_residual_safe_mixture_to_sampler(
    sampler: NegativeSampler,
    residual_safe_mixture_state: _ResidualSafeMixtureState | None,
) -> None:
    if residual_safe_mixture_state is None:
        return
    if hasattr(sampler, "mixture_residual_safe_probability"):
        setattr(sampler, "mixture_residual_safe_probability", float(residual_safe_mixture_state.probability))


def _update_residual_safe_mixture_state(
    residual_safe_mixture_state: _ResidualSafeMixtureState,
    negative_metadata: Mapping[str, object],
    val_aupr: float | None,
) -> None:
    if residual_safe_mixture_state.step_size == 0.0:
        return
    in_band_rate = float(
        negative_metadata.get(
            "rerank_selected_in_band_rate",
            negative_metadata.get("adversarial_selected_in_band_rate", 0.0),
        )
    )
    fallback_rate = float(negative_metadata.get("risk_fallback_rate", 0.0))
    attempts_mean = float(negative_metadata.get("sampling_attempts_mean", 0.0))
    candidate_pool_size = float(negative_metadata.get("candidate_pool_size_mean", 0.0))

    residual_safe_mixture_state.last_in_band_rate = in_band_rate
    residual_safe_mixture_state.last_risk_fallback_rate = fallback_rate
    residual_safe_mixture_state.last_sampling_attempts_mean = attempts_mean
    residual_safe_mixture_state.last_candidate_pool_size_mean = candidate_pool_size

    val_delta = 0.0
    if val_aupr is not None:
        previous_val_aupr = residual_safe_mixture_state.previous_val_aupr
        residual_safe_mixture_state.previous_val_aupr = float(val_aupr)
        if previous_val_aupr is not None:
            val_delta = float(val_aupr) - previous_val_aupr
    residual_safe_mixture_state.last_val_delta = val_delta

    hardness_pressure = max(0.0, residual_safe_mixture_state.target_in_band_rate - in_band_rate)
    risk_pressure = max(0.0, fallback_rate - residual_safe_mixture_state.max_risk_fallback_rate)
    cost_pressure = _positive_relative_excess(attempts_mean, residual_safe_mixture_state.max_sampling_attempts_mean)
    diversity_pressure = _positive_relative_excess(
        residual_safe_mixture_state.min_candidate_pool_size,
        candidate_pool_size,
    )
    validation_pressure = 0.0
    if val_aupr is not None and residual_safe_mixture_state.previous_val_aupr is not None:
        validation_pressure = max(0.0, residual_safe_mixture_state.min_val_delta - val_delta)
    signed_pressure = (
        residual_safe_mixture_state.hardness_weight * hardness_pressure
        - residual_safe_mixture_state.risk_weight * risk_pressure
        - residual_safe_mixture_state.cost_weight * cost_pressure
        - residual_safe_mixture_state.diversity_weight * diversity_pressure
        - residual_safe_mixture_state.validation_weight * validation_pressure
    )
    if signed_pressure == 0.0:
        return
    previous = residual_safe_mixture_state.probability
    residual_safe_mixture_state.probability = _clamp(
        previous + residual_safe_mixture_state.step_size * signed_pressure,
        residual_safe_mixture_state.min_probability,
        residual_safe_mixture_state.max_probability,
    )
    if residual_safe_mixture_state.probability > previous:
        residual_safe_mixture_state.increase_updates += 1
        residual_safe_mixture_state.updates += 1
    elif residual_safe_mixture_state.probability < previous:
        residual_safe_mixture_state.decrease_updates += 1
        residual_safe_mixture_state.updates += 1


def _store_final_residual_safe_mixture_config(
    sampling_config: object,
    residual_safe_mixture_state: _ResidualSafeMixtureState,
) -> None:
    if not isinstance(sampling_config, dict):
        return
    sampling_config["mixture_residual_safe_probability"] = float(residual_safe_mixture_state.probability)
    sampling_config["residual_safe_controller_final_probability"] = float(residual_safe_mixture_state.probability)


def _positive_relative_excess(value: float, bound: float) -> float:
    if bound <= 0.0:
        return 0.0
    return max(0.0, (value - bound) / bound)


def _clamp(value: float, lower: float, upper: float) -> float:
    return min(max(value, lower), upper)


def _future_positive_probe_metrics(
    config: Mapping[str, object],
    dataset: HypergraphDataset,
    sampler_name: str,
    sampler: NegativeSampler,
    model: HyperedgePredictor | SetHyperedgePredictor | BipartiteHyperedgePredictor,
    validation_batch: dict[str, list[tuple[int, ...]] | list[int]] | None,
    rng: random.Random,
    negatives_per_positive: int,
) -> dict[str, float]:
    sampling_config = config["sampling"]
    max_sources = int(sampling_config.get("future_positive_eval_max_sources", 1024))
    if max_sources <= 0 or not dataset.future_positive_edges:
        return future_positive_hit_metrics([], dataset.future_positive_edges, source_count=0)
    source_edges = _sample_source_edges(dataset.train_edges, max_sources, rng)
    if not source_edges:
        return future_positive_hit_metrics([], dataset.future_positive_edges, source_count=0)
    batch = _sample_training_negatives(
        config,
        sampler_name,
        sampler,
        model,
        validation_batch=validation_batch,
        source_edges=source_edges,
        num_nodes=dataset.num_nodes,
        positive_edges=dataset.train_positive_edges,
        rng=rng,
        negatives_per_positive=negatives_per_positive,
    )
    return future_positive_hit_metrics(
        negative_edges=batch.edges,
        future_positive_edges=dataset.future_positive_edges,
        source_count=len(source_edges),
    )


def _sample_source_edges(
    source_edges: list[tuple[int, ...]],
    max_sources: int,
    rng: random.Random,
) -> list[tuple[int, ...]]:
    if len(source_edges) <= max_sources:
        return list(source_edges)
    return rng.sample(source_edges, max_sources)


def _build_validation_batch(
    config: Mapping[str, object],
    dataset: HypergraphDataset,
    rng: random.Random,
) -> dict[str, list[tuple[int, ...]] | list[int]]:
    negatives_per_positive = int(config["training"]["negatives_per_positive"])
    eval_sampler = SizeMatchedSampler(max_attempts=int(config["sampling"].get("max_attempts", 100)))
    negative_batch = eval_sampler.sample(
        source_edges=dataset.val_edges,
        num_nodes=dataset.num_nodes,
        positive_edges=dataset.positive_edges,
        rng=rng,
        negatives_per_positive=negatives_per_positive,
    )
    edges = dataset.val_edges + negative_batch.edges
    labels = [1] * len(dataset.val_edges) + [0] * len(negative_batch.edges)
    return {"edges": edges, "labels": labels}


def _evaluate_fixed_edges(
    model: HyperedgePredictor | SetHyperedgePredictor | BipartiteHyperedgePredictor,
    edges: list[tuple[int, ...]],
    labels: list[int],
) -> dict[str, float]:
    scores = model.predict_scores(edges)
    return binary_prediction_metrics(labels, scores)


def _write_epoch_status(
    config: Mapping[str, object],
    sampler_name: str,
    epoch: int,
    loss: float,
    val_metrics: dict[str, float] | None,
    best_epoch: int,
    best_val_metrics: dict[str, float],
    adaptive_hardness_state: "_AdaptiveHardnessState | None" = None,
    proposal_mixture_state: "_ProposalMixtureState | None" = None,
    residual_safe_mixture_state: "_ResidualSafeMixtureState | None" = None,
) -> None:
    training_config = config["training"]
    total_epochs = int(training_config["epochs"])
    interval = max(1, int(training_config.get("status_interval", 5)))
    if epoch != 1 and epoch != total_epochs and epoch % interval != 0:
        return
    payload: dict[str, object] = {
        "sampler": sampler_name,
        "epoch": epoch,
        "total_epochs": total_epochs,
        "loss": loss,
        "best_epoch": best_epoch,
        "best_val_auc": best_val_metrics["val_auc"],
        "best_val_aupr": best_val_metrics["val_aupr"],
    }
    if adaptive_hardness_state is not None:
        payload["adaptive_hardness_center"] = adaptive_hardness_state.center
        payload["adaptive_hardness_last_in_band_rate"] = adaptive_hardness_state.last_in_band_rate
        payload["adaptive_hardness_last_risk_fallback_rate"] = adaptive_hardness_state.last_risk_fallback_rate
    if proposal_mixture_state is not None:
        payload["proposal_mixture_probability"] = proposal_mixture_state.probability
        payload["proposal_mixture_last_in_band_rate"] = proposal_mixture_state.last_in_band_rate
        payload["proposal_mixture_last_risk_fallback_rate"] = proposal_mixture_state.last_risk_fallback_rate
    if residual_safe_mixture_state is not None:
        payload["residual_safe_mixture_probability"] = residual_safe_mixture_state.probability
        payload["residual_safe_mixture_last_in_band_rate"] = residual_safe_mixture_state.last_in_band_rate
        payload["residual_safe_mixture_last_risk_fallback_rate"] = residual_safe_mixture_state.last_risk_fallback_rate
        payload["residual_safe_mixture_last_val_delta"] = residual_safe_mixture_state.last_val_delta
    if val_metrics is not None:
        payload["val_auc"] = val_metrics["auc"]
        payload["val_aupr"] = val_metrics["aupr"]
    append_status(_status_path(config), "epoch", **payload)


def _status_path(config: Mapping[str, object]) -> str | None:
    experiment_config = config.get("experiment", {})
    if not isinstance(experiment_config, Mapping):
        return None
    status_path = experiment_config.get("_status_path")
    return str(status_path) if status_path else None


def _edge_size_range(edges: list[tuple[int, ...]]) -> tuple[int, int]:
    sizes = [len(edge) for edge in edges]
    return min(sizes), max(sizes)
