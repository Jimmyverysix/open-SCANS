from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import pickle
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

REPO_ROOT = Path(os.environ.get("SCANS_REPO_ROOT", Path(__file__).resolve().parents[1]))
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from scans.data import load_hyperedges_from_txt
from scans.data.benchmark import SEHP_DATASETS, dataset_by_key, processed_edge_path, processed_feature_path
from scans.data.hypergraph import Hyperedge, HypergraphDataset
from scans.evaluation import binary_prediction_metrics, load_protocol_batches
from scans.models import BENCHMARK_BACKBONES, BenchmarkBackbonePredictor
from scans.samplers import build_sampler
from scans.samplers.base import NegativeSampleBatch, NegativeSampler
from scans.training.minibatch import shuffled_index_batches, source_aligned_index_batches
from scans.training.optimizers import build_predictor_optimizer
from scans.training.seed import set_global_seed


PROTOCOL_VERSION = "scans_v4_cover_aware_hnhn_minibatch"
ABLATION_SEMANTICS_VERSION = "scans_component_ablation_v2"
STRICT_ABLATION_SEMANTICS_VERSION = "scans_component_ablation_v3_strict_pairing"
MODES = (
    "sehp_diffusion",
    "scans_no_distill",
    "scans_no_risk",
    "scans_full",
    "scans_no_diffusion",
    "scans_no_boundary",
    "scans_hard_only",
    "scans_diffusion_only",
    "scans_no_teacher_target",
    "scans_no_diffusion_proposal",
    "scans_no_diffusion_signal",
    "scans_random_proposal_control",
    "scans_no_risk_control",
    "scans_no_discrete_retrieval",
    "scans_no_topk_soft",
)


class ConditionalResidualDDIMGenerator(nn.Module):
    def __init__(
        self,
        *,
        representation_dim: int,
        hidden_dim: int,
        steps: int,
        schedule: str,
        min_alpha_bar: float,
        cosine_s: float,
        beta_start: float,
        beta_end: float,
        residual_norm_clip: float,
    ) -> None:
        super().__init__()
        self.representation_dim = int(representation_dim)
        self.steps = int(steps)
        if self.steps < 2:
            raise ValueError("diffusion_steps must be >= 2 for conditional residual DDIM")
        self.residual_norm_clip = float(residual_norm_clip)
        alpha_bar = _build_alpha_bar_schedule(
            steps=self.steps,
            schedule=schedule,
            min_alpha_bar=float(min_alpha_bar),
            cosine_s=float(cosine_s),
            beta_start=float(beta_start),
            beta_end=float(beta_end),
        )
        self.register_buffer("alpha_bar", alpha_bar)
        self._validate_alpha_bar(float(min_alpha_bar))
        self.time_embedding = nn.Embedding(self.steps + 1, representation_dim)
        self.denoiser = nn.Sequential(
            nn.Linear(representation_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, representation_dim),
        )
        self.condition_projection = nn.Sequential(
            nn.Linear(representation_dim, representation_dim),
            nn.LayerNorm(representation_dim),
        )

    def training_loss(
        self,
        clean: torch.Tensor,
        condition: torch.Tensor,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        if clean.shape != condition.shape:
            raise ValueError(f"condition and teacher batch sizes must match: clean={tuple(clean.shape)} condition={tuple(condition.shape)}")
        condition_unit = F.normalize(condition, dim=1, eps=1e-8)
        teacher_unit = F.normalize(clean, dim=1, eps=1e-8)
        delta_clean = torch.nan_to_num(teacher_unit - condition_unit)
        batch = clean.shape[0]
        timestep = torch.randint(1, self.steps + 1, (batch,), device=clean.device, generator=generator)
        alpha_bar_t = self.alpha_bar.index_select(0, timestep).unsqueeze(1).to(clean.device)
        noise = torch.randn(
            delta_clean.shape,
            device=delta_clean.device,
            dtype=delta_clean.dtype,
            generator=generator,
        )
        noisy = alpha_bar_t.sqrt() * delta_clean + (1.0 - alpha_bar_t).sqrt() * noise
        predicted = self._predict_noise(noisy, condition_unit, timestep)
        loss = F.mse_loss(predicted, noise)
        residual_norm = (teacher_unit - condition_unit).detach().norm(dim=1)
        return loss, {
            "diffusion_denoise_mse": float(loss.detach().cpu().item()),
            "residual_target_norm_mean": float(residual_norm.mean().cpu().item()),
            "residual_target_norm_std": float(residual_norm.std(unbiased=False).cpu().item()) if residual_norm.numel() > 1 else 0.0,
            "sampled_timestep_mean": float(timestep.float().mean().detach().cpu().item()),
            "alpha_bar_sampled_mean": float(alpha_bar_t.detach().mean().cpu().item()),
            "alpha_bar_terminal": float(self.alpha_bar[-1].detach().cpu().item()),
        }

    def sample(
        self,
        condition: torch.Tensor,
        generator: torch.Generator | None = None,
        return_details: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, float]]:
        condition_unit = F.normalize(condition, dim=1, eps=1e-8)
        x = torch.randn(condition.shape, device=condition.device, generator=generator)
        initial_norm = x.detach().norm(dim=1)
        delta_0_hat = torch.zeros_like(x)
        for step in range(self.steps, 0, -1):
            timestep = torch.full((condition.shape[0],), step, dtype=torch.long, device=condition.device)
            alpha_bar_t = self.alpha_bar.index_select(0, timestep).unsqueeze(1).to(condition.device)
            epsilon_hat = self._predict_noise(x, condition_unit, timestep)
            delta_0_hat = (x - (1.0 - alpha_bar_t).sqrt() * epsilon_hat) / alpha_bar_t.sqrt().clamp_min(1e-8)
            delta_0_hat = torch.nan_to_num(delta_0_hat)
            delta_0_hat = self._clip_residual_norm(delta_0_hat)
            if step > 1:
                previous = torch.full((condition.shape[0],), step - 1, dtype=torch.long, device=condition.device)
                alpha_bar_prev = self.alpha_bar.index_select(0, previous).unsqueeze(1).to(condition.device)
                x = alpha_bar_prev.sqrt() * delta_0_hat + (1.0 - alpha_bar_prev).sqrt() * epsilon_hat
                x = torch.nan_to_num(x)
            else:
                proposal = F.normalize(condition_unit + delta_0_hat, dim=1, eps=1e-8)
                proposal = torch.nan_to_num(proposal)
                if proposal.shape != condition.shape:
                    raise RuntimeError(f"proposal shape mismatch: proposal={tuple(proposal.shape)} condition={tuple(condition.shape)}")
                if not torch.isfinite(proposal).all():
                    raise RuntimeError("conditional diffusion proposal contains non-finite values")
                proposal_norm = proposal.norm(dim=1)
                if not torch.allclose(proposal_norm, torch.ones_like(proposal_norm), atol=1e-4, rtol=1e-4):
                    raise RuntimeError("conditional diffusion proposal norm is not close to 1")
                if return_details:
                    details = {
                        "initial_noise_norm_mean": float(initial_norm.mean().detach().cpu().item()),
                        "predicted_residual_norm_mean": float(delta_0_hat.detach().norm(dim=1).mean().cpu().item()),
                        "proposal_norm_mean": float(proposal_norm.detach().mean().cpu().item()),
                        "proposal_condition_cosine_mean": float((proposal * condition_unit).sum(dim=1).detach().mean().cpu().item()),
                    }
                    return proposal, details
                return proposal
        raise RuntimeError("conditional diffusion sampling loop did not return a proposal")

    def _predict_noise(self, noisy_residual: torch.Tensor, condition_unit: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        projected = self.condition_projection(condition_unit)
        time = self.time_embedding(timestep)
        return self.denoiser(torch.cat([noisy_residual, projected, time], dim=1))

    def _clip_residual_norm(self, residual: torch.Tensor) -> torch.Tensor:
        if self.residual_norm_clip <= 0:
            return residual
        norm = residual.norm(dim=1, keepdim=True).clamp_min(1e-8)
        scale = torch.clamp(float(self.residual_norm_clip) / norm, max=1.0)
        return residual * scale

    def _validate_alpha_bar(self, min_alpha_bar: float) -> None:
        if self.alpha_bar.numel() != self.steps + 1:
            raise ValueError("conditional diffusion alpha_bar length must be T+1")
        if not torch.isclose(self.alpha_bar[0], torch.tensor(1.0, dtype=self.alpha_bar.dtype), atol=1e-7):
            raise ValueError("conditional diffusion alpha_bar[0] must equal 1")
        if not torch.all(self.alpha_bar[:-1] >= self.alpha_bar[1:] - 1e-8):
            raise ValueError("conditional diffusion alpha_bar must be monotonically non-increasing")
        terminal = float(self.alpha_bar[-1].item())
        if abs(terminal - float(min_alpha_bar)) > max(1e-6, 0.1 * float(min_alpha_bar)):
            raise ValueError(
                f"conditional diffusion terminal alpha_bar {terminal:g} "
                f"does not match min_alpha_bar {float(min_alpha_bar):g}"
            )


class ResidualMLPProposalGenerator(nn.Module):
    def __init__(
        self,
        representation_dim: int,
        hidden_dim: int,
        residual_norm_clip: float,
    ) -> None:
        super().__init__()
        self.residual_norm_clip = float(residual_norm_clip)
        self.mapper = nn.Sequential(
            nn.Linear(representation_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, representation_dim),
        )

    def training_loss(
        self,
        clean: torch.Tensor,
        condition: torch.Tensor,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        del generator
        if clean.shape != condition.shape:
            raise ValueError(f"condition and teacher batch sizes must match: clean={tuple(clean.shape)} condition={tuple(condition.shape)}")
        condition_unit = F.normalize(condition, dim=1, eps=1e-8)
        teacher_unit = F.normalize(clean, dim=1, eps=1e-8)
        residual_target = torch.nan_to_num(teacher_unit - condition_unit)
        residual_prediction = self._clip_residual_norm(self.mapper(condition_unit))
        loss = F.mse_loss(residual_prediction, residual_target)
        return loss, {
            "mlp_residual_mse": float(loss.detach().cpu().item()),
            "residual_target_norm_mean": float(residual_target.detach().norm(dim=1).mean().cpu().item()),
            "mlp_residual_norm_mean": float(residual_prediction.detach().norm(dim=1).mean().cpu().item()),
        }

    def sample(
        self,
        condition: torch.Tensor,
        generator: torch.Generator | None = None,
        return_details: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, float]]:
        del generator
        condition_unit = F.normalize(condition, dim=1, eps=1e-8)
        residual_prediction = self._clip_residual_norm(self.mapper(condition_unit))
        proposal = F.normalize(condition_unit + residual_prediction, dim=1, eps=1e-8)
        if return_details:
            return proposal, {
                "predicted_residual_norm_mean": float(residual_prediction.detach().norm(dim=1).mean().cpu().item()),
                "proposal_norm_mean": float(proposal.detach().norm(dim=1).mean().cpu().item()),
                "proposal_condition_cosine_mean": float((proposal * condition_unit).sum(dim=1).detach().mean().cpu().item()),
            }
        return proposal

    def _clip_residual_norm(self, residual: torch.Tensor) -> torch.Tensor:
        if self.residual_norm_clip <= 0:
            return residual
        norm = residual.norm(dim=1, keepdim=True).clamp_min(1e-8)
        scale = torch.clamp(float(self.residual_norm_clip) / norm, max=1.0)
        return residual * scale


class DirectMLPProposalGenerator(nn.Module):
    """One-shot conditional MLP that directly regresses a teacher-negative embedding."""

    def __init__(self, representation_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.mapper = nn.Sequential(
            nn.Linear(representation_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, representation_dim),
        )

    def training_loss(
        self,
        clean: torch.Tensor,
        condition: torch.Tensor,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        del generator
        if clean.shape != condition.shape:
            raise ValueError(f"condition and teacher batch sizes must match: clean={tuple(clean.shape)} condition={tuple(condition.shape)}")
        condition_unit = F.normalize(condition, dim=1, eps=1e-8)
        teacher_unit = F.normalize(clean, dim=1, eps=1e-8)
        proposal = F.normalize(self.mapper(condition_unit), dim=1, eps=1e-8)
        loss = F.mse_loss(proposal, teacher_unit)
        return loss, {
            "mlp_direct_mse": float(loss.detach().cpu().item()),
            "proposal_teacher_cosine_mean": float((proposal * teacher_unit).sum(dim=1).detach().mean().cpu().item()),
        }

    def sample(
        self,
        condition: torch.Tensor,
        generator: torch.Generator | None = None,
        return_details: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, float]]:
        del generator
        condition_unit = F.normalize(condition, dim=1, eps=1e-8)
        proposal = F.normalize(self.mapper(condition_unit), dim=1, eps=1e-8)
        if return_details:
            return proposal, {
                "proposal_norm_mean": float(proposal.detach().norm(dim=1).mean().cpu().item()),
                "proposal_condition_cosine_mean": float((proposal * condition_unit).sum(dim=1).detach().mean().cpu().item()),
            }
        return proposal


class UnconditionalVAEProposalGenerator(nn.Module):
    """Ordinary VAE over teacher-negative embeddings, without z+ conditioning."""

    def __init__(self, representation_dim: int, hidden_dim: int, latent_dim: int, kl_weight: float) -> None:
        super().__init__()
        self.representation_dim = int(representation_dim)
        self.latent_dim = int(latent_dim)
        self.kl_weight = float(kl_weight)
        self.encoder = nn.Sequential(
            nn.Linear(representation_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.mu = nn.Linear(hidden_dim, latent_dim)
        self.logvar = nn.Linear(hidden_dim, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, representation_dim),
        )

    def training_loss(
        self,
        clean: torch.Tensor,
        condition: torch.Tensor,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        if clean.shape[0] != condition.shape[0]:
            raise ValueError(f"teacher and reference batch sizes must match: clean={tuple(clean.shape)} reference={tuple(condition.shape)}")
        teacher_unit = F.normalize(clean, dim=1, eps=1e-8)
        hidden = self.encoder(teacher_unit)
        mu = self.mu(hidden)
        logvar = self.logvar(hidden).clamp(min=-8.0, max=8.0)
        latent = self._reparameterize(mu, logvar, generator=generator)
        reconstructed = self.decoder(latent)
        recon_loss = F.mse_loss(reconstructed, teacher_unit)
        kl = -0.5 * torch.mean(1.0 + logvar - mu.square() - logvar.exp())
        loss = recon_loss + self.kl_weight * kl
        return loss, {
            "vae_reconstruction_mse": float(recon_loss.detach().cpu().item()),
            "vae_kl": float(kl.detach().cpu().item()),
            "vae_loss": float(loss.detach().cpu().item()),
        }

    def sample(
        self,
        condition: torch.Tensor,
        generator: torch.Generator | None = None,
        return_details: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, float]]:
        latent = torch.randn(
            (condition.shape[0], self.latent_dim),
            device=condition.device,
            dtype=condition.dtype,
            generator=generator,
        )
        proposal = F.normalize(self.decoder(latent), dim=1, eps=1e-8)
        if return_details:
            return proposal, {
                "vae_latent_norm_mean": float(latent.detach().norm(dim=1).mean().cpu().item()),
                "proposal_norm_mean": float(proposal.detach().norm(dim=1).mean().cpu().item()),
            }
        return proposal

    @staticmethod
    def _reparameterize(
        mu: torch.Tensor,
        logvar: torch.Tensor,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        noise = torch.randn(std.shape, device=std.device, dtype=std.dtype, generator=generator)
        return mu + noise * std


def main() -> None:
    parser = argparse.ArgumentParser(description="Run SCANS for hyperedge prediction.")
    parser.add_argument("--datasets", default="cora,citeseer,ndc_class,dblp")
    parser.add_argument("--methods", default="hds,hypergcn,nhp")
    parser.add_argument("--modes", default="scans_full")
    parser.add_argument("--seeds", default="42,43,44,45,46")
    parser.add_argument("--epochs", "--max-epochs", dest="epochs", type=int, default=200)
    parser.add_argument("--min-epochs", type=int, default=20)
    parser.add_argument("--early-stopping-patience", type=int, default=20)
    parser.add_argument("--early-stopping-min-delta", type=float, default=1e-4)
    parser.add_argument("--validation-interval", type=int, default=1)
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--predictor-batch-size", type=int, default=128)
    parser.add_argument("--hnhn-alpha-e", type=float, default=0.0)
    parser.add_argument("--hnhn-alpha-v", type=float, default=0.0)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--message-passing-layers", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--predictor-optimizer", choices=("rmsprop", "adamw"), default="rmsprop")
    parser.add_argument(
        "--predictor-init-checkpoint",
        default="",
        help=(
            "Optional predictor checkpoint used as an epoch-0 safe baseline. The checkpoint must contain "
            "a 'predictor' state dict; validation checkpointing will retain it unless SCANS improves it."
        ),
    )
    parser.add_argument(
        "--predictor-replay-negative-bank",
        default="",
        help="Optional aligned AHP negative bank used to replay adversarial structural anchors during SCANS fine-tuning.",
    )
    parser.add_argument(
        "--predictor-replay-ratio",
        type=float,
        default=0.0,
        help="Fraction of SCANS training negatives replaced by aligned replay-bank negatives each epoch.",
    )
    parser.add_argument(
        "--predictor-protocol-replay-weights",
        default="",
        help=(
            "Optional auxiliary training-negative mixture, e.g. mns:0.2,cns:0.2. "
            "SCANS retrieval remains the primary negative distribution; these weights only redistribute "
            "the fixed total negative-loss mass toward protocol-robust replay banks."
        ),
    )
    parser.add_argument(
        "--predictor-protocol-replay-banks",
        type=int,
        default=0,
        help="Number of deterministic source-aligned training-negative banks built per replay protocol.",
    )
    parser.add_argument(
        "--predictor-protocol-replay-warmup-epochs",
        type=int,
        default=0,
        help="Linearly warm auxiliary protocol-replay weights from zero over this many epochs.",
    )
    parser.add_argument(
        "--predictor-protocol-replay-loss-mode",
        choices=("fixed_total", "additive"),
        default="fixed_total",
        help=(
            "fixed_total redistributes a fixed negative-loss mass across SCANS and replay negatives; "
            "additive preserves the full SCANS negative loss and adds weighted replay losses."
        ),
    )
    parser.add_argument("--split-strategy", choices=("random", "cover_aware"), default="cover_aware")
    parser.add_argument("--generator-learning-rate", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--proposal-generator", choices=("diffusion", "mlp", "mlp_direct", "vae"), default="diffusion")
    parser.add_argument(
        "--proposal-generator-training",
        choices=("shared", "native"),
        default="shared",
        help=(
            "Use the shared SCANS proposal auxiliaries, or train replacement generators only "
            "with their native objective (MLP regression or VAE ELBO)."
        ),
    )
    parser.add_argument("--vae-latent-dim", type=int, default=32)
    parser.add_argument("--vae-kl-weight", type=float, default=0.01)
    parser.add_argument("--diffusion-steps", type=int, default=4)
    parser.add_argument("--diffusion-schedule", choices=("cosine", "linear_beta"), default="cosine")
    parser.add_argument("--diffusion-min-alpha-bar", type=float, default=1e-4)
    parser.add_argument("--diffusion-cosine-s", type=float, default=0.008)
    parser.add_argument("--diffusion-beta-start", type=float, default=0.0001)
    parser.add_argument("--diffusion-beta-end", type=float, default=0.02)
    parser.add_argument("--residual-norm-clip", type=float, default=2.0)
    parser.add_argument("--generator-updates-per-epoch", type=int, default=4)
    parser.add_argument("--generator-batch-size", type=int, default=256)
    parser.add_argument("--generator-grad-clip", type=float, default=5.0)
    parser.add_argument("--risk-bank-size", type=int, default=1024)
    parser.add_argument("--generator-ema-decay", type=float, default=0.999)
    parser.add_argument(
        "--generator-ema-start-step",
        type=int,
        default=0,
        help=(
            "Use the online generator before this many optimizer updates, then initialize EMA from the learned "
            "online weights. Zero preserves the legacy EMA-from-random-initialization behavior."
        ),
    )
    parser.add_argument(
        "--strict-paired-ablation",
        action="store_true",
        help=(
            "Use mode-independent named RNG streams, retain the full teacher/retrieval candidate layout for "
            "w/o-diffusion, and delay checkpoint eligibility until diffusion/EMA warm-up is complete."
        ),
    )
    parser.add_argument(
        "--legacy-diffusion-rng-replay",
        action="store_true",
        help=(
            "For an MLP/VAE replacement compared with a completed legacy diffusion run, keep the replacement's "
            "stochastic generator on a private stream and consume the CUDA random draws that the legacy diffusion "
            "training path would have consumed before predictor dropout."
        ),
    )
    parser.add_argument("--boundary-target", type=float, default=0.5)
    parser.add_argument("--risk-margin", type=float, default=0.86)
    parser.add_argument("--boundary-weight", type=float, default=0.2)
    parser.add_argument("--risk-weight", type=float, default=0.2)
    parser.add_argument("--diversity-weight", type=float, default=0.02)
    parser.add_argument("--diffusion-weight", type=float, default=0.1)
    parser.add_argument("--teacher-candidate-multiplier", type=int, default=8)
    parser.add_argument("--retrieval-candidate-multiplier", type=int, default=16)
    parser.add_argument(
        "--retrieval-sampling-source-chunk-size",
        type=int,
        default=0,
        help="Generate candidate pools in source-edge chunks; 0 keeps the single-batch path.",
    )
    parser.add_argument("--retrieval-lambda", type=float, default=0.6)
    parser.add_argument("--retrieval-lambda-warmup-epochs", type=int, default=5)
    parser.add_argument("--retrieval-lambda-start", type=float, default=0.0)
    parser.add_argument("--retrieval-diffusion-weight", type=float, default=1.0)
    parser.add_argument("--retrieval-boundary-weight", type=float, default=0.5)
    parser.add_argument("--retrieval-hardness-weight", type=float, default=0.25)
    parser.add_argument(
        "--retrieval-selection-strategy",
        choices=("top_score", "topk_sample", "topk_multi"),
        default="topk_sample",
    )
    parser.add_argument("--retrieval-top-k", type=int, default=5)
    parser.add_argument("--retrieval-temperature", type=float, default=0.8)
    parser.add_argument(
        "--retrieval-normalization",
        choices=("none", "local_zscore", "robust_zscore", "local_rank", "pool_zscore", "pool_rank"),
        default="local_zscore",
    )
    parser.add_argument("--negatives-per-positive", type=int, default=1)
    parser.add_argument("--validation-protocols", default="sns,mns,cns,mix")
    parser.add_argument(
        "--validation-negative-bank",
        default="",
        help="Optional precomputed source-aligned validation negatives for the requested protocols.",
    )
    parser.add_argument(
        "--test-negative-bank",
        default="",
        help="Optional precomputed source-aligned test negatives for the requested protocols.",
    )
    parser.add_argument(
        "--checkpoint-objective",
        choices=("mean_auc_ap", "mean_auc", "mean_ap", "min_auc_ap", "min_auc", "min_ap"),
        default="mean_auc_ap",
    )
    parser.add_argument("--validation-protocol-weights", default="")
    parser.add_argument("--validation-auc-weight", type=float, default=1.0)
    parser.add_argument("--validation-ap-weight", type=float, default=1.0)
    parser.add_argument(
        "--retrieval-candidate-pool",
        choices=("risk_controlled", "risk_sns", "union", "union_hard", "uncontrolled_union"),
        default="risk_controlled",
    )
    parser.add_argument(
        "--retrieval-pool-source-weights",
        default="",
        help="Optional union-pool source quotas, e.g. risk_controlled:1,sns:0,mns:1,cns:3,mix:2.",
    )
    parser.add_argument(
        "--retrieval-pool-source-weights-start",
        default="",
        help="Optional initial source quotas, linearly warmed up to --retrieval-pool-source-weights.",
    )
    parser.add_argument("--retrieval-pool-source-weights-warmup-epochs", type=int, default=0)
    parser.add_argument(
        "--retrieval-pool-quota-target",
        choices=("both", "retrieval_only"),
        default="both",
        help="Apply union source quotas to both teacher/retrieval pools or retrieval only.",
    )
    parser.add_argument(
        "--retrieval-selected-source-weights",
        default="",
        help="Optional direct source mixture for final selected negatives; empty keeps score-only selection.",
    )
    parser.add_argument("--teacher-mode", choices=("safe_hard",), default="safe_hard")
    parser.add_argument("--sampling-max-attempts", type=int, default=300)
    parser.add_argument(
        "--sampling-replacement-strategy",
        choices=("random", "neighborhood_mixed", "risk_aware_mixed", "residual_safe_mixed", "diffusion_mixed", "calibrated_mixture"),
        default="random",
    )
    parser.add_argument("--sampling-anchor-ratio", type=float, default=0.5)
    parser.add_argument("--sampling-nearest-positive-upper-bound", type=float, default=0.90)
    parser.add_argument("--sampling-closure-risk-upper-bound", type=float, default=1.0)
    parser.add_argument(
        "--sampling-max-closure-pairs-per-edge",
        type=int,
        default=0,
        help="Deterministic closure-risk pair cap per hyperedge; 0 keeps the exact all-pairs index.",
    )
    parser.add_argument("--sampling-rerank-top-k", type=int, default=2)
    parser.add_argument(
        "--sampling-rerank-selection-strategy",
        choices=("top_score", "target_score", "band_pass", "random_top_k"),
        default="top_score",
    )
    parser.add_argument("--sampling-rerank-score-lower-bound", type=float, default=0.3)
    parser.add_argument("--sampling-rerank-score-upper-bound", type=float, default=0.8)
    parser.add_argument("--risk-aware-pool-multiplier", type=int, default=4)
    parser.add_argument("--risk-aware-nearest-weight", type=float, default=0.0)
    parser.add_argument("--residual-safe-pool-multiplier", type=int, default=4)
    parser.add_argument("--residual-safe-structural-quantile", type=float, default=0.5)
    parser.add_argument("--residual-safe-residual-weight", type=float, default=2.0)
    parser.add_argument("--max-parallel", type=int, default=1)
    parser.add_argument("--torch-threads", type=int, default=4)
    parser.add_argument("--device", default="cpu", help="Device for a worker, e.g. cpu, cuda, cuda:0, or auto.")
    parser.add_argument("--devices", default="", help="Comma-separated devices assigned round-robin to workers.")
    deterministic_group = parser.add_mutually_exclusive_group()
    deterministic_group.add_argument("--deterministic", dest="deterministic", action="store_true")
    deterministic_group.add_argument("--no-deterministic", dest="deterministic", action="store_false")
    parser.set_defaults(deterministic=True)
    parser.add_argument("--max-train-edges", type=int, default=0)
    parser.add_argument("--max-val-edges", type=int, default=0)
    parser.add_argument("--max-test-edges", type=int, default=0)
    parser.add_argument(
        "--skip-test",
        action="store_true",
        help="Skip test-set evaluation during validation-only hyperparameter screening.",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--self-check", action="store_true")
    parser.add_argument("--dataset", default="cora")
    parser.add_argument("--method", choices=BENCHMARK_BACKBONES, default="hds")
    parser.add_argument("--mode", choices=MODES, default="scans_full")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    _validate_training_control_args(args)

    if args.self_check:
        _run_self_check()
        return

    if args.worker:
        torch.set_num_threads(args.torch_threads)
        torch.set_num_interop_threads(1)
        run_one(args)
        return

    datasets = [value for value in args.datasets.split(",") if value]
    methods = [value for value in args.methods.split(",") if value]
    modes = [value for value in args.modes.split(",") if value]
    seeds = [int(value) for value in args.seeds.split(",") if value]
    known_datasets = {dataset.key for dataset in SEHP_DATASETS}
    if not set(datasets) <= known_datasets:
        raise ValueError(f"unknown datasets: {set(datasets) - known_datasets}")
    if not set(methods) <= set(BENCHMARK_BACKBONES):
        raise ValueError(f"unknown methods: {set(methods) - set(BENCHMARK_BACKBONES)}")
    if not set(modes) <= set(MODES):
        raise ValueError(f"unknown modes: {set(modes) - set(MODES)}")

    jobs = [(dataset, method, mode, seed) for dataset in datasets for method in methods for mode in modes for seed in seeds]
    if not args.force:
        jobs = [job for job in jobs if not _completed(*job, args)]
    _run_parallel(jobs, args)


def run_one(args: argparse.Namespace) -> None:
    started = time.perf_counter()
    _configure_reproducibility(args.seed, deterministic=bool(args.deterministic))
    output = _output_dir(args)
    output.mkdir(parents=True, exist_ok=True)
    config = _config(args)
    dataset = _limit_dataset_for_probe(load_hyperedges_from_txt(config["data"], args.seed), args)
    rng = random.Random(args.seed)
    model, training = _train_scans(config, dataset, args.mode, rng, checkpoint_dir=output)
    test = {} if args.skip_test else _evaluate_protocol_negative_sets(
        config, args.dataset, dataset, model, args.seed, str(args.test_negative_bank)
    )
    metrics = {
        "protocol_version": PROTOCOL_VERSION,
        "adapter_notice": (
            "SCANS validation-only screening: test evaluation was intentionally skipped."
            if args.skip_test
            else (
                "SCANS benchmark: the configured proposal generator produces a continuous proposal, "
                "then risk-feasible discrete negatives are retrieved for predictor training; "
                "test metrics are computed on standard discrete test hyperedges and protocol negative sets."
            )
        ),
        "dataset": args.dataset,
        "method": args.method,
        "mode": args.mode,
        "seed": args.seed,
        "training_epochs": args.epochs,
        "actual_training_epochs": int(training["actual_epochs"]),
        "test_evaluation_skipped": bool(args.skip_test),
        "training": training,
        "test": test,
        "probe_limits": _probe_limits(args),
        "scans_config": _scans_metadata(args),
        "runtime_seconds": time.perf_counter() - started,
    }
    (output / "config.json").write_text(json.dumps(_jsonable(config), indent=2), encoding="utf-8")
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"wrote {output / 'metrics.json'}")


def _validate_training_control_args(args: argparse.Namespace) -> None:
    if int(args.epochs) < 1:
        raise ValueError("max_epochs must be positive")
    if int(args.min_epochs) < 1 or int(args.min_epochs) > int(args.epochs):
        raise ValueError("min_epochs must be in [1, max_epochs]")
    if int(args.early_stopping_patience) < 1:
        raise ValueError("early_stopping_patience must be positive")
    if int(args.predictor_batch_size) < 1:
        raise ValueError("predictor_batch_size must be positive")
    if float(args.early_stopping_min_delta) < 0.0:
        raise ValueError("early_stopping_min_delta must be non-negative")
    if int(args.validation_interval) != 1:
        raise ValueError("the unified benchmark protocol requires validation_interval=1")
    if int(args.retrieval_pool_source_weights_warmup_epochs) < 0:
        raise ValueError("retrieval_pool_source_weights_warmup_epochs must be non-negative")
    if int(args.retrieval_sampling_source_chunk_size) < 0:
        raise ValueError("retrieval_sampling_source_chunk_size must be non-negative")
    if int(args.generator_ema_start_step) < 0:
        raise ValueError("generator_ema_start_step must be non-negative")
    if int(args.sampling_max_closure_pairs_per_edge) < 0:
        raise ValueError("sampling_max_closure_pairs_per_edge must be non-negative")
    if str(args.predictor_init_checkpoint) and not Path(args.predictor_init_checkpoint).is_file():
        raise FileNotFoundError(f"predictor init checkpoint not found: {args.predictor_init_checkpoint}")
    if str(args.validation_negative_bank) and not Path(args.validation_negative_bank).is_file():
        raise FileNotFoundError(f"validation negative bank not found: {args.validation_negative_bank}")
    if not 0.0 <= float(args.predictor_replay_ratio) <= 1.0:
        raise ValueError("predictor_replay_ratio must be in [0, 1]")
    if float(args.predictor_replay_ratio) > 0.0 and not Path(args.predictor_replay_negative_bank).is_file():
        raise FileNotFoundError(f"predictor replay negative bank not found: {args.predictor_replay_negative_bank}")
    protocol_replay_weights = _parse_weight_mapping(str(args.predictor_protocol_replay_weights))
    allowed_replay_protocols = {"sns", "mns", "cns", "mix"}
    unknown_replay_protocols = set(protocol_replay_weights) - allowed_replay_protocols
    if unknown_replay_protocols:
        raise ValueError(f"unknown predictor protocol replay sources: {sorted(unknown_replay_protocols)}")
    if any(float(weight) < 0.0 for weight in protocol_replay_weights.values()):
        raise ValueError("predictor protocol replay weights must be non-negative")
    if int(args.predictor_protocol_replay_banks) < 0:
        raise ValueError("predictor_protocol_replay_banks must be non-negative")
    if int(args.predictor_protocol_replay_warmup_epochs) < 0:
        raise ValueError("predictor_protocol_replay_warmup_epochs must be non-negative")
    if any(float(weight) > 0.0 for weight in protocol_replay_weights.values()) and int(args.predictor_protocol_replay_banks) < 1:
        raise ValueError("positive predictor protocol replay weights require at least one replay bank")


def _configure_reproducibility(seed: int, *, deterministic: bool) -> None:
    if deterministic:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    else:
        torch.use_deterministic_algorithms(False)
        torch.backends.cudnn.deterministic = False
    set_global_seed(seed)


def _config(args: argparse.Namespace) -> dict[str, object]:
    features = processed_feature_path(REPO_ROOT, args.dataset)
    data = {
        "source": "txt",
        "path": str(processed_edge_path(REPO_ROOT, args.dataset)),
        "separator": "whitespace",
        "split": {"train": 0.6, "val": 0.2, "test": 0.2, "strategy": str(args.split_strategy)},
    }
    if features.exists():
        data["features_path"] = str(features)
    return {
        "experiment": {"seed": args.seed, "_status_path": str(_output_root(args.dataset, args.method, args.mode, args.seed, args) / "status.jsonl")},
        "data": data,
        "model": {
            "type": args.method,
            "embedding_dim": args.embedding_dim,
            "hidden_dim": args.hidden_dim,
            "dropout": float(args.dropout),
            "message_passing_layers": int(args.message_passing_layers),
            "hnhn_alpha_e": float(args.hnhn_alpha_e),
            "hnhn_alpha_v": float(args.hnhn_alpha_v),
            "device": args.device,
        },
        "training": {
            "epochs": args.epochs,
            "max_epochs": args.epochs,
            "min_epochs": args.min_epochs,
            "learning_rate": float(args.learning_rate),
            "generator_learning_rate": float(args.generator_learning_rate),
            "weight_decay": float(args.weight_decay),
            "predictor_optimizer": str(args.predictor_optimizer),
            "predictor_init_checkpoint": str(args.predictor_init_checkpoint),
            "predictor_replay_negative_bank": str(args.predictor_replay_negative_bank),
            "predictor_replay_ratio": float(args.predictor_replay_ratio),
            "predictor_protocol_replay_weights": str(args.predictor_protocol_replay_weights),
            "predictor_protocol_replay_banks": int(args.predictor_protocol_replay_banks),
            "predictor_protocol_replay_warmup_epochs": int(args.predictor_protocol_replay_warmup_epochs),
            "predictor_protocol_replay_loss_mode": str(args.predictor_protocol_replay_loss_mode),
            "validation_negative_bank": str(args.validation_negative_bank),
            "test_negative_bank": str(args.test_negative_bank),
            "negatives_per_positive": max(1, int(args.negatives_per_positive)),
            "predictor_batch_size": int(args.predictor_batch_size),
            "use_validation_checkpoint": True,
            "validation_interval": args.validation_interval,
            "early_stopping_patience": args.early_stopping_patience,
            "early_stopping_min_delta": args.early_stopping_min_delta,
        },
        "sampling": {
            "max_attempts": int(args.sampling_max_attempts),
            "samplers": ["risk_controlled"],
            "future_positive_eval_max_sources": 0,
            "anchor_ratio": float(args.sampling_anchor_ratio),
            "nearest_positive_upper_bound": float(args.sampling_nearest_positive_upper_bound),
            "closure_risk_upper_bound": float(args.sampling_closure_risk_upper_bound),
            "max_closure_pairs_per_edge": int(args.sampling_max_closure_pairs_per_edge),
            "replacement_strategy": str(args.sampling_replacement_strategy),
            "model_aware_rerank": True,
            "rerank_candidate_multiplier": int(args.teacher_candidate_multiplier),
            "rerank_top_k": int(args.sampling_rerank_top_k),
            "rerank_selection_strategy": str(args.sampling_rerank_selection_strategy),
            "rerank_target_score": 0.5,
            "rerank_target_score_min": 0.5,
            "rerank_target_score_max": 0.5,
            "rerank_score_lower_bound": float(args.sampling_rerank_score_lower_bound),
            "rerank_score_upper_bound": float(args.sampling_rerank_score_upper_bound),
            "risk_aware_pool_multiplier": int(args.risk_aware_pool_multiplier),
            "risk_aware_nearest_weight": float(args.risk_aware_nearest_weight),
            "residual_safe_pool_multiplier": int(args.residual_safe_pool_multiplier),
            "residual_safe_structural_quantile": float(args.residual_safe_structural_quantile),
            "residual_safe_residual_weight": float(args.residual_safe_residual_weight),
        },
        "scans": _scans_metadata(args),
    }


def _train_scans(
    config: Mapping[str, object],
    dataset: HypergraphDataset,
    mode: str,
    rng: random.Random,
    *,
    checkpoint_dir: Path | None = None,
) -> tuple[BenchmarkBackbonePredictor, dict[str, float]]:
    training_started = time.perf_counter()
    print("phase model_build_start", flush=True)
    model = _build_benchmark_model(config, dataset)
    print(f"phase model_ready seconds={time.perf_counter() - training_started:.1f}", flush=True)
    training_config = config["training"]
    predictor_init_checkpoint = str(training_config.get("predictor_init_checkpoint", ""))
    predictor_initialized = bool(predictor_init_checkpoint)
    if predictor_initialized:
        _load_predictor_checkpoint(model, Path(predictor_init_checkpoint))
    replay_ratio = float(training_config.get("predictor_replay_ratio", 0.0))
    replay_bank_path = str(training_config.get("predictor_replay_negative_bank", ""))
    replay_negative_banks = (
        _load_replay_negative_bank(Path(replay_bank_path), dataset, int(config["experiment"]["seed"]))
        if replay_ratio > 0.0
        else []
    )
    protocol_replay_target_weights = _parse_weight_mapping(
        str(training_config.get("predictor_protocol_replay_weights", ""))
    )
    protocol_replay_banks = _build_training_protocol_replay_banks(
        config=config,
        dataset=dataset,
        protocol_weights=protocol_replay_target_weights,
        bank_count=int(training_config.get("predictor_protocol_replay_banks", 0)),
        seed=int(config["experiment"]["seed"]),
    )
    optimizer = build_predictor_optimizer(
        model.parameters(),
        name=str(training_config["predictor_optimizer"]),
        learning_rate=float(training_config["learning_rate"]),
        weight_decay=float(training_config["weight_decay"]),
    )
    criterion = nn.BCEWithLogitsLoss()
    min_size, max_size = _edge_size_range(dataset.train_edges)
    safe_sampler = build_sampler("risk_controlled", config["sampling"], min_size, max_size)
    sns_sampler = build_sampler("sns", config["sampling"], min_size, max_size)
    retrieval_sampler_cache: dict[str, NegativeSampler] = {"sns": sns_sampler}
    checkpoint_validation_batches = _build_protocol_validation_batches(
        config,
        dataset,
        _protocol_list(str(config["scans"].get("validation_protocols", "sns"))),
        int(config["experiment"]["seed"]),
    )

    generator: nn.Module | None = None
    generator_optimizer: torch.optim.Optimizer | None = None
    generator_ema: nn.Module | None = None
    strict_paired_ablation = _strict_paired_ablation(config)
    legacy_diffusion_rng_replay = _legacy_diffusion_rng_replay(config)
    experiment_seed = int(config["experiment"]["seed"])
    proposal_generator_name = str(config["scans"].get("proposal_generator", "diffusion"))
    if legacy_diffusion_rng_replay and proposal_generator_name == "diffusion":
        raise ValueError("legacy diffusion RNG replay is only valid for an MLP/VAE replacement")
    if legacy_diffusion_rng_replay and strict_paired_ablation:
        raise ValueError("legacy diffusion RNG replay and strict paired ablation are mutually exclusive")
    if _uses_diffusion_proposal(mode):
        with torch.no_grad():
            representation_dim = model.edge_representations(
                dataset.train_edges[:1],
                model.encode_all_nodes(),
            ).shape[1]
        if legacy_diffusion_rng_replay:
            # Reproduce the completed legacy full run's CPU RNG position after diffusion-generator
            # initialization, while keeping the replacement generator's parameter initialization private.
            reference_config = dict(config["scans"])
            reference_config["proposal_generator"] = "diffusion"
            reference_generator = _build_proposal_generator(
                representation_dim=int(representation_dim),
                hidden_dim=int(config["model"]["hidden_dim"]),
                config=reference_config,
            )
            del reference_generator
            with torch.random.fork_rng(devices=[]):
                generator = _build_proposal_generator(
                    representation_dim=int(representation_dim),
                    hidden_dim=int(config["model"]["hidden_dim"]),
                    config=config["scans"],
                ).to(next(model.parameters()).device)
        elif strict_paired_ablation:
            # Generator initialization must not advance the global torch RNG used later by predictor dropout.
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(_named_training_seed(experiment_seed, 0, "generator_init"))
                generator = _build_proposal_generator(
                    representation_dim=int(representation_dim),
                    hidden_dim=int(config["model"]["hidden_dim"]),
                    config=config["scans"],
                ).to(next(model.parameters()).device)
        else:
            generator = _build_proposal_generator(
                representation_dim=int(representation_dim),
                hidden_dim=int(config["model"]["hidden_dim"]),
                config=config["scans"],
            ).to(next(model.parameters()).device)
        generator_optimizer = torch.optim.AdamW(
            generator.parameters(),
            lr=float(training_config["generator_learning_rate"]),
            weight_decay=float(training_config["weight_decay"]),
        )
        generator_ema = _clone_generator_ema(generator, float(config["scans"].get("generator_ema_decay", 0.999)))

    initial_validation_metrics: dict[str, float] = {}
    if predictor_initialized:
        initial_validation_metrics = _evaluate_checkpoint_objective(
            model,
            checkpoint_validation_batches,
            objective=str(config["scans"].get("checkpoint_objective", "mean_auc_ap")),
            protocol_weights=_parse_weight_mapping(str(config["scans"].get("validation_protocol_weights", ""))),
            auc_weight=float(config["scans"].get("validation_auc_weight", 1.0)),
            ap_weight=float(config["scans"].get("validation_ap_weight", 1.0)),
        )
        print(
            "predictor_init "
            f"checkpoint={predictor_init_checkpoint} "
            f"val_auc={float(initial_validation_metrics['auc']):.6f} "
            f"val_ap={float(initial_validation_metrics['aupr']):.6f} "
            f"score={float(initial_validation_metrics['score']):.6f}",
            flush=True,
        )
    best_model_state = copy.deepcopy(model.state_dict())
    best_generator_state = copy.deepcopy(generator.state_dict()) if generator is not None else None
    best_generator_ema_state = copy.deepcopy(generator_ema.state_dict()) if generator_ema is not None else None
    best_epoch = 0
    best_val = {
        "val_auc": float(initial_validation_metrics.get("auc", 0.0)),
        "val_aupr": float(initial_validation_metrics.get("aupr", 0.0)),
        "val_score": float(initial_validation_metrics.get("score", float("-inf"))),
    }
    best_validation_metrics: dict[str, float] = dict(initial_validation_metrics)
    epoch_summaries: list[dict[str, float]] = []
    negatives_per_positive = int(training_config["negatives_per_positive"])
    if _uses_diffusion_proposal(mode) and int(config["scans"].get("generator_updates_per_epoch", 4)) < 1:
        raise ValueError("generator_updates_per_epoch must be >= 1")
    if _uses_diffusion_proposal(mode) and int(config["scans"].get("diffusion_steps", 4)) < 2:
        raise ValueError("diffusion_steps must be >= 2")
    if _uses_discrete_retrieval(mode) and mode == "scans_full":
        strategy = str(config["scans"].get("retrieval_selection_strategy", "top_score"))
        top_k = int(config["scans"].get("retrieval_top_k", 1))
        if strategy == "top_score" or top_k <= 1:
            print(
                "WARNING: scans_full is configured with hard retrieval "
                f"(retrieval_selection_strategy={strategy}, retrieval_top_k={top_k}); "
                "this does not execute top-K soft sampling.",
                flush=True,
            )

    total_epochs = int(training_config["max_epochs"])
    min_epochs = int(training_config["min_epochs"])
    validation_interval = int(training_config["validation_interval"])
    early_stopping_patience = int(training_config["early_stopping_patience"])
    early_stopping_min_delta = float(training_config["early_stopping_min_delta"])
    epochs_without_improvement = 0
    actual_epochs = 0
    stopped_early = False
    predictor_updates = 0
    generator_updates_total = 0
    generator_updates_per_epoch = int(config["scans"].get("generator_updates_per_epoch", 4))
    generator_ema_start_step = int(config["scans"].get("generator_ema_start_step", 0))
    minimum_checkpoint_epoch = 1
    if strict_paired_ablation and mode.startswith("scans"):
        ema_ready_epoch = math.ceil(generator_ema_start_step / max(1, generator_updates_per_epoch))
        lambda_ready_epoch = int(config["scans"].get("retrieval_lambda_warmup_epochs", 5)) + 1
        minimum_checkpoint_epoch = max(1, ema_ready_epoch, lambda_ready_epoch)
    if minimum_checkpoint_epoch > total_epochs:
        raise ValueError(
            f"minimum checkpoint epoch {minimum_checkpoint_epoch} exceeds max epochs {total_epochs}"
        )
    if minimum_checkpoint_epoch > 1:
        best_val = {"val_auc": 0.0, "val_aupr": 0.0, "val_score": float("-inf")}
        best_validation_metrics = {}
    best_checkpoint_path = checkpoint_dir / "best.pt" if checkpoint_dir is not None else None
    last_checkpoint_path = checkpoint_dir / "last.pt" if checkpoint_dir is not None else None
    if checkpoint_dir is not None:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
    if predictor_initialized and best_checkpoint_path is not None:
        _save_joint_checkpoint(
            best_checkpoint_path,
            predictor=model,
            generator=generator,
            generator_ema=generator_ema,
            predictor_optimizer=optimizer,
            generator_optimizer=generator_optimizer,
            epoch=0,
            best_epoch=0,
            validation_metrics=initial_validation_metrics,
            best_validation_metrics=initial_validation_metrics,
            epochs_without_improvement=0,
            config=config,
            training_rng=rng,
            kind="best_initial",
        )
    for epoch in range(total_epochs):
        epoch_started = time.perf_counter()
        candidate_rng = _epoch_training_rng(rng, experiment_seed, epoch, "candidate_pool", strict_paired_ablation)
        teacher_rng = _epoch_training_rng(rng, experiment_seed, epoch, "teacher", strict_paired_ablation)
        generator_rng = _epoch_training_rng(rng, experiment_seed, epoch, "generator_python", strict_paired_ablation)
        retrieval_rng = _epoch_training_rng(rng, experiment_seed, epoch, "retrieval", strict_paired_ablation)
        replay_rng = _epoch_training_rng(rng, experiment_seed, epoch, "replay", strict_paired_ablation)
        predictor_rng = _epoch_training_rng(rng, experiment_seed, epoch, "predictor_batches", strict_paired_ablation)
        generator_torch_rng = (
            _torch_generator_for_device(
                next(model.parameters()).device,
                _named_training_seed(experiment_seed, epoch, "generator_torch"),
            )
            if (strict_paired_ablation or legacy_diffusion_rng_replay) and generator is not None
            else None
        )
        model.train()
        if generator is not None:
            generator.train()
        teacher_candidate_batch = None
        teacher_candidate_block_size = None
        retrieval_candidate_batch = None
        retrieval_candidate_block_size = None
        candidate_sampling_started = time.perf_counter()
        if _uses_discrete_retrieval(mode):
            print(f"phase epoch {epoch + 1}/{total_epochs} candidate_sampling_start", flush=True)
            teacher_candidate_multiplier = max(1, int(config["sampling"].get("rerank_candidate_multiplier", 4)))
            retrieval_candidate_multiplier = max(1, int(config["scans"].get("retrieval_candidate_multiplier", 8)))
            if _uses_partitioned_candidate_pool(mode, config):
                teacher_candidate_block_size = negatives_per_positive * teacher_candidate_multiplier
                retrieval_candidate_block_size = negatives_per_positive * retrieval_candidate_multiplier
                effective_source_weights = _effective_pool_source_weights(config["scans"], epoch)
                quota_target = str(config["scans"].get("retrieval_pool_quota_target", "both"))
                effective_pool = _effective_candidate_pool(
                    mode,
                    config["scans"].get("retrieval_candidate_pool", "risk_controlled"),
                )
                if quota_target == "retrieval_only" and effective_source_weights and effective_pool != "risk_controlled":
                    teacher_candidate_batch = _with_candidate_labels(
                        safe_sampler.sample(
                            source_edges=dataset.train_edges,
                            num_nodes=dataset.num_nodes,
                            positive_edges=dataset.train_positive_edges,
                            rng=candidate_rng,
                            negatives_per_positive=teacher_candidate_block_size,
                        ),
                        "risk_controlled",
                    )
                    retrieval_candidate_batch, retrieval_candidate_block_size = _sample_retrieval_candidate_batch(
                        config=config,
                        mode=mode,
                        safe_sampler=safe_sampler,
                        source_edges=dataset.train_edges,
                        num_nodes=dataset.num_nodes,
                        positive_edges=dataset.train_positive_edges,
                        rng=candidate_rng,
                        negatives_per_positive=negatives_per_positive,
                        candidate_multiplier=retrieval_candidate_multiplier,
                        model=model,
                        sampler_cache=retrieval_sampler_cache,
                        source_weights_override=effective_source_weights,
                    )
                    retrieval_candidate_batch = _exclude_teacher_candidates_from_retrieval(
                        teacher_batch=teacher_candidate_batch,
                        retrieval_batch=retrieval_candidate_batch,
                        source_edges=dataset.train_edges,
                        teacher_block_size=teacher_candidate_block_size,
                        retrieval_block_size=retrieval_candidate_block_size,
                    )
                else:
                    candidate_pool_batch, candidate_pool_block_size = _sample_retrieval_candidate_batch(
                        config=config,
                        mode=mode,
                        safe_sampler=safe_sampler,
                        source_edges=dataset.train_edges,
                        num_nodes=dataset.num_nodes,
                        positive_edges=dataset.train_positive_edges,
                        rng=candidate_rng,
                        negatives_per_positive=negatives_per_positive,
                        candidate_multiplier=teacher_candidate_multiplier + retrieval_candidate_multiplier,
                        model=model,
                        sampler_cache=retrieval_sampler_cache,
                        candidate_partition_sizes=(
                            teacher_candidate_block_size,
                            retrieval_candidate_block_size,
                        ),
                        source_weights_override=effective_source_weights,
                    )
                    teacher_candidate_batch, retrieval_candidate_batch = _split_teacher_retrieval_candidate_batches(
                        candidate_batch=candidate_pool_batch,
                        source_edges=dataset.train_edges,
                        block_size=candidate_pool_block_size,
                        teacher_block_size=teacher_candidate_block_size,
                        retrieval_block_size=retrieval_candidate_block_size,
                    )
            else:
                retrieval_candidate_batch, retrieval_candidate_block_size = _sample_retrieval_candidate_batch(
                    config=config,
                    mode=mode,
                    safe_sampler=safe_sampler,
                    source_edges=dataset.train_edges,
                    num_nodes=dataset.num_nodes,
                    positive_edges=dataset.train_positive_edges,
                    rng=candidate_rng,
                    negatives_per_positive=negatives_per_positive,
                    candidate_multiplier=retrieval_candidate_multiplier,
                    model=model,
                    sampler_cache=retrieval_sampler_cache,
                    source_weights_override=_effective_pool_source_weights(config["scans"], epoch),
                )
            print(
                f"phase epoch {epoch + 1}/{total_epochs} candidate_sampling_done "
                f"seconds={time.perf_counter() - candidate_sampling_started:.1f}",
                flush=True,
            )
            should_trace_candidate_digest = strict_paired_ablation or (
                epoch == 0
                and mode in {
                    "scans_full",
                    "scans_no_diffusion_signal",
                    "scans_random_proposal_control",
                }
            )
            if should_trace_candidate_digest and retrieval_candidate_batch is not None:
                print(
                    f"paired_candidate_digest epoch={epoch + 1} "
                    f"sha256={_edge_sequence_digest(retrieval_candidate_batch.edges)} "
                    f"count={len(retrieval_candidate_batch.edges)}",
                    flush=True,
                )

        teacher_batch: NegativeSampleBatch | None = None
        if _uses_diffusion_proposal(mode):
            teacher_batch = _sample_teacher_negatives(
                config=config,
                mode=mode,
                safe_sampler=safe_sampler,
                sns_sampler=sns_sampler,
                model=model,
                source_edges=dataset.train_edges,
                num_nodes=dataset.num_nodes,
                positive_edges=dataset.train_positive_edges,
                rng=teacher_rng,
                negatives_per_positive=negatives_per_positive,
                candidate_batch=teacher_candidate_batch,
                block_size=teacher_candidate_block_size,
            )
        positive_edges = dataset.train_edges
        effective_lambda, lambda_progress = _effective_retrieval_lambda(
            mode,
            _configured_retrieval_lambda_target(config["scans"]),
            float(config["scans"].get("retrieval_lambda_start", 0.0)),
            int(config["scans"].get("retrieval_lambda_warmup_epochs", 5)),
            epoch,
        )
        if not 0.0 <= effective_lambda <= 1.0:
            raise RuntimeError(f"effective retrieval lambda out of range: {effective_lambda}")
        generated: torch.Tensor | None = None
        if generator is not None and generator_optimizer is not None and teacher_batch is not None:
            model.eval()
            with torch.no_grad():
                positive_representations = _edge_representations(model, positive_edges).detach()
            teacher_target_mode = _teacher_target_mode(mode)
            if teacher_target_mode == "positive_ddpm":
                teacher_representations = positive_representations.detach()
            elif teacher_target_mode == "undistilled_noise":
                teacher_representations = _undistilled_teacher(positive_representations.detach(), teacher_rng)
            else:
                with torch.no_grad():
                    teacher_representations = _edge_representations(model, teacher_batch.edges).detach()
            if teacher_target_mode == "positive_ddpm":
                teacher_conditions = positive_representations.detach()
            else:
                teacher_conditions = _teacher_condition_representations(
                    positive_representations=positive_representations.detach(),
                    source_edges=positive_edges,
                    teacher_batch=teacher_batch,
                ).detach()
            if teacher_representations.shape != teacher_conditions.shape:
                raise ValueError(
                    "generator condition and teacher batch sizes must match: "
                    f"teacher={tuple(teacher_representations.shape)} condition={tuple(teacher_conditions.shape)}"
                )
            positive_bank = _sample_risk_bank(
                positive_representations,
                int(config["scans"].get("risk_bank_size", 1024)),
                generator_rng,
            )
            generator_started = time.perf_counter()
            generator_metrics = _run_generator_updates(
                generator=generator,
                generator_optimizer=generator_optimizer,
                generator_ema=generator_ema,
                config=config,
                mode=mode,
                teacher_representations=teacher_representations.detach(),
                teacher_conditions=teacher_conditions.detach(),
                positive_bank=positive_bank.detach(),
                rng=generator_rng,
                torch_generator=generator_torch_rng,
                completed_updates_before=generator_updates_total,
            )
            generator_updates_total += int(generator_metrics["generator_updates_completed"])
            generator_metrics["generator_train_seconds"] = time.perf_counter() - generator_started
            proposal_model = _proposal_generator_for_sampling(
                generator,
                generator_ema,
                completed_updates=generator_updates_total,
                ema_start_step=generator_ema_start_step,
            )
            proposal_model.eval()
            proposal_rng = _torch_generator_for_device(
                positive_representations.device,
                int(config["experiment"]["seed"]) + 1000003 * (epoch + 1),
            )
            proposal_started = time.perf_counter()
            with torch.no_grad():
                proposal_output = _sample_generator(
                    proposal_model,
                    positive_representations.detach(),
                    torch_generator=proposal_rng,
                    return_details=True,
                )
                generated = _proposal_tensor(proposal_output).detach()
                proposal_sample_metrics = _proposal_details(proposal_output)
            proposal_seconds = time.perf_counter() - proposal_started
            generator.train()
            teacher_for_sources = _align_teacher_representations_to_sources(
                teacher_representations=teacher_representations,
                teacher_batch=teacher_batch,
                source_edges=positive_edges,
                fallback=positive_representations,
            )
            proposal_metrics = {
                **proposal_sample_metrics,
                **_proposal_diagnostics(generated, positive_representations, teacher_for_sources),
                "proposal_sample_seconds": proposal_seconds,
                "proposal_enabled": 1.0,
                "proposal_uses_ema": 1.0 if proposal_model is generator_ema else 0.0,
            }
        else:
            generator_metrics = {
                "generator_enabled": 0.0,
                "generator_updates_completed": 0.0,
                "generator_train_seconds": 0.0,
            }
            proposal_metrics = {
                "proposal_enabled": 0.0,
                "proposal_sample_seconds": 0.0,
                "proposal_uses_ema": 0.0,
            }
        retrieval_generated = generated
        if mode == "scans_random_proposal_control" and generated is not None:
            random_control_rng = _torch_generator_for_device(
                generated.device,
                int(config["experiment"]["seed"]) + 2000003 * (epoch + 1),
            )
            retrieval_generated = F.normalize(
                torch.randn(
                    generated.shape,
                    device=generated.device,
                    dtype=generated.dtype,
                    generator=random_control_rng,
                ),
                dim=1,
                eps=1e-8,
            )
            proposal_metrics["random_control_norm_mean"] = float(
                retrieval_generated.norm(dim=1).mean().detach().cpu().item()
            )
            proposal_metrics["random_control_learned_cosine_mean"] = float(
                (retrieval_generated * F.normalize(generated.detach(), dim=1, eps=1e-8))
                .sum(dim=1)
                .mean()
                .detach()
                .cpu()
                .item()
            )
        if _uses_discrete_retrieval(mode):
            retrieval_started = time.perf_counter()
            retrieval_batch = _sample_diffusion_to_discrete_negatives(
                config=config,
                mode=mode,
                safe_sampler=safe_sampler,
                model=model,
                generated=None if retrieval_generated is None else retrieval_generated.detach(),
                source_edges=dataset.train_edges,
                num_nodes=dataset.num_nodes,
                positive_edges=dataset.train_positive_edges,
                rng=retrieval_rng,
                negatives_per_positive=negatives_per_positive,
                candidate_batch=retrieval_candidate_batch,
                block_size=retrieval_candidate_block_size,
                effective_lambda=effective_lambda,
                teacher_batch=teacher_batch,
            )
            retrieval_seconds = time.perf_counter() - retrieval_started
            if replay_negative_banks:
                retrieval_batch = _mix_replay_negatives(
                    retrieval_batch,
                    replay_negative_banks[epoch % len(replay_negative_banks)],
                    source_edges=dataset.train_edges,
                    replay_ratio=replay_ratio,
                    rng=replay_rng,
                )
        else:
            if generated is None:
                raise RuntimeError(f"{mode} requires a continuous proposal when discrete retrieval is disabled")
            retrieval_batch = None
            retrieval_seconds = 0.0
        effective_protocol_replay_weights, protocol_replay_warmup_progress = _effective_protocol_replay_weights(
            protocol_replay_target_weights,
            epoch=epoch,
            warmup_epochs=int(training_config.get("predictor_protocol_replay_warmup_epochs", 0)),
        )
        active_protocol_replay_batches = {
            protocol: banks[epoch % len(banks)]
            for protocol, banks in protocol_replay_banks.items()
            if banks and effective_protocol_replay_weights.get(protocol, 0.0) > 0.0
        }
        (
            predictor_loss_value,
            positive_loss_value,
            negative_loss_value,
            epoch_predictor_updates,
            protocol_replay_metrics,
        ) = _run_predictor_minibatches(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            positive_edges=positive_edges,
            retrieval_batch=retrieval_batch,
            generated=generated,
            protocol_replay_batches=active_protocol_replay_batches,
            protocol_replay_weights=effective_protocol_replay_weights,
            protocol_replay_loss_mode=str(training_config.get("predictor_protocol_replay_loss_mode", "fixed_total")),
            batch_size=int(training_config["predictor_batch_size"]),
            pairwise_ranking=str(config["model"].get("type", "")) == "nhp",
            rng=predictor_rng,
        )
        predictor_updates += epoch_predictor_updates

        current_epoch = epoch + 1
        actual_epochs = current_epoch
        should_validate = current_epoch % validation_interval == 0 or current_epoch == total_epochs
        if not should_validate:
            raise RuntimeError("SCANS currently requires validation at the end of every completed epoch")
        val_metrics = _evaluate_checkpoint_objective(
            model,
            checkpoint_validation_batches,
            objective=str(config["scans"].get("checkpoint_objective", "mean_auc_ap")),
            protocol_weights=_parse_weight_mapping(str(config["scans"].get("validation_protocol_weights", ""))),
            auc_weight=float(config["scans"].get("validation_auc_weight", 1.0)),
            ap_weight=float(config["scans"].get("validation_ap_weight", 1.0)),
        )
        checkpoint_eligible = current_epoch >= minimum_checkpoint_epoch
        improved = checkpoint_eligible and val_metrics["score"] > best_val["val_score"] + early_stopping_min_delta
        if improved:
            best_model_state = copy.deepcopy(model.state_dict())
            best_generator_state = copy.deepcopy(generator.state_dict()) if generator is not None else None
            best_generator_ema_state = copy.deepcopy(generator_ema.state_dict()) if generator_ema is not None else None
            best_epoch = current_epoch
            best_val = {
                "val_auc": val_metrics["auc"],
                "val_aupr": val_metrics["aupr"],
                "val_score": val_metrics["score"],
            }
            best_validation_metrics = dict(val_metrics)
            epochs_without_improvement = 0
            if best_checkpoint_path is not None:
                _save_joint_checkpoint(
                    best_checkpoint_path,
                    predictor=model,
                    generator=generator,
                    generator_ema=generator_ema,
                    predictor_optimizer=optimizer,
                    generator_optimizer=generator_optimizer,
                    epoch=current_epoch,
                    best_epoch=best_epoch,
                    validation_metrics=val_metrics,
                    best_validation_metrics=best_validation_metrics,
                    epochs_without_improvement=epochs_without_improvement,
                    config=config,
                    training_rng=rng,
                    kind="best",
                )
        elif checkpoint_eligible:
            epochs_without_improvement += validation_interval
        else:
            epochs_without_improvement = 0

        epoch_summaries.append(
            {
                "epoch": float(current_epoch),
                "loss": float(predictor_loss_value + generator_metrics.get("generator_loss", 0.0)),
                "predictor_loss": float(predictor_loss_value),
                "predictor_updates": float(epoch_predictor_updates),
                "positive_loss": float(positive_loss_value),
                "negative_loss": float(negative_loss_value),
                "protocol_replay_warmup_progress": float(protocol_replay_warmup_progress),
                **protocol_replay_metrics,
                "boundary_loss": 0.0,
                "target_retrieval_lambda": float(config["scans"].get("retrieval_lambda", 0.8)),
                "effective_retrieval_lambda": float(effective_lambda),
                "retrieval_lambda_warmup_progress": float(lambda_progress),
                "checkpoint_eligible": 1.0 if checkpoint_eligible else 0.0,
                "minimum_checkpoint_epoch": float(minimum_checkpoint_epoch),
                "generator_ema_enabled": 1.0 if generator_ema is not None else 0.0,
                "retrieval_seconds": float(retrieval_seconds),
                **generator_metrics,
                **proposal_metrics,
                **({} if retrieval_batch is None else {f"retrieval_{key}": float(value) for key, value in retrieval_batch.metadata.items() if isinstance(value, (int, float))}),
            }
        )
        print(
            "epoch "
            f"{current_epoch}/{total_epochs} "
            f"loss={float(predictor_loss_value + generator_metrics.get('generator_loss', 0.0)):.6f} "
            f"updates={predictor_updates} "
            f"val_auc={float(val_metrics['auc']):.6f} "
            f"val_ap={float(val_metrics['aupr']):.6f} "
            f"score={float(val_metrics['score']):.6f} "
            f"best_epoch={best_epoch} "
            f"bad_epochs={epochs_without_improvement} "
            f"seconds={time.perf_counter() - epoch_started:.1f}",
            flush=True,
        )

        if last_checkpoint_path is not None:
            _save_joint_checkpoint(
                last_checkpoint_path,
                predictor=model,
                generator=generator,
                generator_ema=generator_ema,
                predictor_optimizer=optimizer,
                generator_optimizer=generator_optimizer,
                epoch=current_epoch,
                best_epoch=best_epoch,
                validation_metrics=val_metrics,
                best_validation_metrics=best_validation_metrics,
                epochs_without_improvement=epochs_without_improvement,
                config=config,
                training_rng=rng,
                kind="last",
            )

        if current_epoch >= min_epochs and epochs_without_improvement >= early_stopping_patience:
            stopped_early = True
            print(
                "early_stop "
                f"epoch={current_epoch} best_epoch={best_epoch} "
                f"best_score={best_val['val_score']:.6f} patience={early_stopping_patience} "
                f"min_delta={early_stopping_min_delta}",
                flush=True,
            )
            break

    model.load_state_dict(best_model_state)
    if generator is not None and best_generator_state is not None:
        generator.load_state_dict(best_generator_state)
    if generator_ema is not None and best_generator_ema_state is not None:
        generator_ema.load_state_dict(best_generator_ema_state)
    return model, {
        "best_epoch": float(best_epoch),
        "actual_epochs": float(actual_epochs),
        "predictor_updates": float(predictor_updates),
        "predictor_batch_size": float(training_config["predictor_batch_size"]),
        "max_epochs": float(total_epochs),
        "min_epochs": float(min_epochs),
        "validation_interval": float(validation_interval),
        "early_stopping_patience": float(early_stopping_patience),
        "early_stopping_min_delta": float(early_stopping_min_delta),
        "minimum_checkpoint_epoch": float(minimum_checkpoint_epoch),
        "epochs_without_improvement": float(epochs_without_improvement),
        "stopped_early": 1.0 if stopped_early else 0.0,
        "val_auc": float(best_val["val_auc"]),
        "val_aupr": float(best_val["val_aupr"]),
        "val_score": float(best_val["val_score"]),
        "predictor_initialized": 1.0 if predictor_initialized else 0.0,
        "predictor_replay_enabled": 1.0 if replay_negative_banks else 0.0,
        "predictor_replay_ratio": float(replay_ratio),
        "predictor_replay_bank_count": float(len(replay_negative_banks)),
        "predictor_protocol_replay_enabled": 1.0 if protocol_replay_banks else 0.0,
        "predictor_protocol_replay_protocol_count": float(len(protocol_replay_banks)),
        "predictor_protocol_replay_bank_count": float(sum(len(banks) for banks in protocol_replay_banks.values())),
        "initial_val_auc": float(initial_validation_metrics.get("auc", -1.0)),
        "initial_val_aupr": float(initial_validation_metrics.get("aupr", -1.0)),
        "initial_val_score": float(initial_validation_metrics.get("score", -1.0)),
        "improved_over_initial": (
            1.0
            if predictor_initialized
            and float(best_val["val_score"]) > float(initial_validation_metrics["score"]) + early_stopping_min_delta
            else 0.0
        ),
        **{f"best_{key}": float(value) for key, value in best_validation_metrics.items()},
        "scans_continuous_enabled": 1.0 if _uses_diffusion_proposal(mode) else 0.0,
        "scans_boundary_enabled": 0.0 if mode in {"sehp_diffusion", "scans_no_boundary"} or _uses_discrete_retrieval(mode) else 1.0,
        "scans_retrieval_boundary_enabled": 0.0,
        "scans_risk_enabled": 1.0 if _uses_generator_risk(mode) else 0.0,
        "scans_safe_distillation_enabled": 1.0 if _teacher_target_mode(mode) == "safe_teacher" else 0.0,
        "scans_teacher_target_mode": _teacher_target_mode(mode),
        "scans_discrete_retrieval_enabled": 1.0 if _uses_discrete_retrieval(mode) else 0.0,
        "scans_partitioned_candidate_pool_enabled": 1.0 if _uses_partitioned_candidate_pool(mode, config) else 0.0,
        **_average_epoch_summaries(epoch_summaries),
    }


def _run_predictor_minibatches(
    *,
    model: BenchmarkBackbonePredictor,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    positive_edges: list[Hyperedge],
    retrieval_batch: NegativeSampleBatch | None,
    generated: torch.Tensor | None,
    protocol_replay_batches: Mapping[str, NegativeSampleBatch],
    protocol_replay_weights: Mapping[str, float],
    protocol_replay_loss_mode: str,
    batch_size: int,
    pairwise_ranking: bool,
    rng: random.Random,
) -> tuple[float, float, float, int, dict[str, float]]:
    if retrieval_batch is None:
        if generated is None or generated.shape[0] != len(positive_edges):
            raise ValueError("continuous proposals must align one-to-one with positive edges")
        batches = (
            (positive_indices, positive_indices)
            for positive_indices in shuffled_index_batches(len(positive_edges), batch_size, rng)
        )
    else:
        batches = source_aligned_index_batches(
            positive_edges,
            retrieval_batch.edges,
            retrieval_batch.source_edges,
            batch_size,
            rng,
        )

    losses: list[float] = []
    positive_losses: list[float] = []
    negative_losses: list[float] = []
    replay_losses: dict[str, list[float]] = {protocol: [] for protocol in protocol_replay_batches}
    updates = 0
    for positive_indices, negative_indices in batches:
        model.train()
        optimizer.zero_grad(set_to_none=True)
        positive_minibatch = [positive_edges[index] for index in positive_indices]
        replay_minibatches: list[tuple[str, float, list[Hyperedge]]] = []
        for protocol, replay_batch in protocol_replay_batches.items():
            weight = max(0.0, float(protocol_replay_weights.get(protocol, 0.0)))
            if weight <= 0.0:
                continue
            if len(replay_batch.edges) != len(positive_edges) or replay_batch.source_edges != positive_edges:
                raise ValueError(f"{protocol} replay bank must align one-to-one with positive edges")
            replay_minibatches.append(
                (
                    protocol,
                    weight,
                    [replay_batch.edges[index] for index in positive_indices],
                )
            )

        if retrieval_batch is not None:
            negative_minibatch = [retrieval_batch.edges[index] for index in negative_indices]
            replay_edges = [edge for _protocol, _weight, edges in replay_minibatches for edge in edges]
            combined_logits = model.forward_edges(positive_minibatch + negative_minibatch + replay_edges)
            positive_logits = combined_logits[: len(positive_minibatch)]
            negative_start = len(positive_minibatch)
            negative_end = negative_start + len(negative_minibatch)
            negative_logits = combined_logits[negative_start:negative_end]
            replay_logits: dict[str, torch.Tensor] = {}
            cursor = negative_end
            for protocol, _weight, edges in replay_minibatches:
                replay_logits[protocol] = combined_logits[cursor : cursor + len(edges)]
                cursor += len(edges)
        else:
            positive_logits = model.forward_edges(positive_minibatch)
            generated_indices = torch.tensor(
                negative_indices,
                dtype=torch.long,
                device=generated.device,
            )
            negative_logits = model.logits_from_edge_representations(
                generated.index_select(0, generated_indices).detach()
            )
            replay_logits = {
                protocol: model.forward_edges(edges)
                for protocol, _weight, edges in replay_minibatches
            }

        positive_loss = criterion(positive_logits, torch.ones_like(positive_logits))
        negative_loss = criterion(negative_logits, torch.zeros_like(negative_logits))
        replay_loss_tensors = {
            protocol: criterion(logits, torch.zeros_like(logits))
            for protocol, logits in replay_logits.items()
        }
        replay_weight_sum = sum(
            weight
            for protocol, weight, _edges in replay_minibatches
            if protocol in replay_loss_tensors
        )
        weighted_replay_loss = sum(
            weight * replay_loss_tensors[protocol]
            for protocol, weight, _edges in replay_minibatches
            if protocol in replay_loss_tensors
        )
        if protocol_replay_loss_mode == "fixed_total":
            robust_negative_loss = (negative_loss + weighted_replay_loss) / (1.0 + replay_weight_sum)
            base_negative_component = negative_loss / (1.0 + replay_weight_sum)
        elif protocol_replay_loss_mode == "additive":
            robust_negative_loss = negative_loss + weighted_replay_loss
            base_negative_component = negative_loss
        else:
            raise ValueError(f"unknown protocol replay loss mode: {protocol_replay_loss_mode}")
        if pairwise_ranking:
            if negative_logits.numel() % positive_logits.numel() != 0:
                raise ValueError("pairwise ranking requires a fixed number of negatives per positive")
            negative_scores = torch.sigmoid(negative_logits).view(positive_logits.numel(), -1).mean(dim=1)
            predictor_loss = F.softplus(negative_scores - torch.sigmoid(positive_logits)).mean()
            if replay_weight_sum > 0.0:
                predictor_loss = predictor_loss + robust_negative_loss - base_negative_component
        else:
            predictor_loss = positive_loss + robust_negative_loss
        predictor_loss.backward()
        optimizer.step()
        losses.append(float(predictor_loss.detach().cpu().item()))
        positive_losses.append(float(positive_loss.detach().cpu().item()))
        negative_losses.append(float(negative_loss.detach().cpu().item()))
        for protocol, replay_loss in replay_loss_tensors.items():
            replay_losses[protocol].append(float(replay_loss.detach().cpu().item()))
        updates += 1
    if not losses:
        raise RuntimeError("predictor mini-batch iterator produced no updates")
    replay_metrics = {
        f"protocol_replay_{protocol}_loss": sum(values) / len(values)
        for protocol, values in replay_losses.items()
        if values
    }
    for protocol, weight in protocol_replay_weights.items():
        replay_metrics[f"protocol_replay_{protocol}_weight"] = float(weight)
    replay_metrics["protocol_replay_total_weight"] = float(
        sum(max(0.0, float(weight)) for weight in protocol_replay_weights.values())
    )
    return (
        sum(losses) / len(losses),
        sum(positive_losses) / len(positive_losses),
        sum(negative_losses) / len(negative_losses),
        updates,
        replay_metrics,
    )


def _save_joint_checkpoint(
    path: Path,
    *,
    predictor: nn.Module,
    generator: nn.Module | None,
    generator_ema: nn.Module | None,
    predictor_optimizer: torch.optim.Optimizer,
    generator_optimizer: torch.optim.Optimizer | None,
    epoch: int,
    best_epoch: int,
    validation_metrics: Mapping[str, float],
    best_validation_metrics: Mapping[str, float],
    epochs_without_improvement: int,
    config: Mapping[str, object],
    training_rng: random.Random,
    kind: str,
) -> None:
    payload: dict[str, Any] = {
        "checkpoint_format_version": 1,
        "kind": str(kind),
        "protocol_version": PROTOCOL_VERSION,
        "predictor": predictor.state_dict(),
        "generator": generator.state_dict() if generator is not None else None,
        "generator_ema": generator_ema.state_dict() if generator_ema is not None else None,
        "predictor_optimizer": predictor_optimizer.state_dict(),
        "generator_optimizer": generator_optimizer.state_dict() if generator_optimizer is not None else None,
        "epoch": int(epoch),
        "best_epoch": int(best_epoch),
        "validation_score": float(validation_metrics["score"]),
        "validation_metrics": {key: float(value) for key, value in validation_metrics.items()},
        "best_validation_score": float(best_validation_metrics.get("score", float("-inf"))),
        "best_validation_metrics": {key: float(value) for key, value in best_validation_metrics.items()},
        "epochs_without_improvement": int(epochs_without_improvement),
        "config": copy.deepcopy(config),
        "rng_state": {
            "python_global": random.getstate(),
            "python_training": training_rng.getstate(),
            "numpy": np.random.get_state(),
            "torch_cpu": torch.get_rng_state(),
            "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _uses_discrete_retrieval(mode: str) -> bool:
    return mode.startswith("scans") and mode != "scans_no_discrete_retrieval"


def _strict_paired_ablation(config: Mapping[str, object]) -> bool:
    return bool(config["scans"].get("strict_paired_ablation", False))


def _legacy_diffusion_rng_replay(config: Mapping[str, object]) -> bool:
    return bool(config["scans"].get("legacy_diffusion_rng_replay", False))


def _uses_partitioned_candidate_pool(mode: str, config: Mapping[str, object]) -> bool:
    if not _uses_discrete_retrieval(mode):
        return False
    return _uses_diffusion_proposal(mode) or (
        _strict_paired_ablation(config) and mode == "scans_no_diffusion_proposal"
    )


def _uses_generator_risk(mode: str) -> bool:
    return mode not in {"sehp_diffusion", "scans_no_risk", "scans_no_risk_control"}


def _teacher_target_mode(mode: str) -> str:
    if not _uses_diffusion_proposal(mode):
        return "none"
    if mode == "scans_no_distill":
        return "undistilled_noise"
    if mode == "scans_no_teacher_target":
        return "positive_ddpm"
    return "safe_teacher"


def _uses_diffusion_proposal(mode: str) -> bool:
    return mode != "scans_no_diffusion_proposal"


def _effective_candidate_pool(mode: str, configured_pool: object) -> str:
    if mode == "scans_no_risk_control":
        return "uncontrolled_union"
    return str(configured_pool)


def _effective_retrieval_selection(
    mode: str,
    configured_strategy: object,
    configured_top_k: object,
) -> tuple[str, int]:
    if mode == "scans_no_topk_soft":
        return "top_score", 1
    return str(configured_strategy), int(configured_top_k)


def _build_proposal_generator(
    representation_dim: int,
    hidden_dim: int,
    config: Mapping[str, object],
) -> nn.Module:
    proposal_generator = str(config.get("proposal_generator", "diffusion"))
    if proposal_generator == "mlp":
        return ResidualMLPProposalGenerator(
            representation_dim=representation_dim,
            hidden_dim=hidden_dim,
            residual_norm_clip=float(config.get("residual_norm_clip", 2.0)),
        )
    if proposal_generator == "mlp_direct":
        return DirectMLPProposalGenerator(
            representation_dim=representation_dim,
            hidden_dim=hidden_dim,
        )
    if proposal_generator == "vae":
        return UnconditionalVAEProposalGenerator(
            representation_dim=representation_dim,
            hidden_dim=hidden_dim,
            latent_dim=int(config.get("vae_latent_dim", max(1, representation_dim // 2))),
            kl_weight=float(config.get("vae_kl_weight", 0.01)),
        )
    if proposal_generator == "diffusion":
        return ConditionalResidualDDIMGenerator(
            representation_dim=representation_dim,
            hidden_dim=hidden_dim,
            steps=int(config["diffusion_steps"]),
            schedule=str(config.get("diffusion_schedule", "cosine")),
            min_alpha_bar=float(config.get("diffusion_min_alpha_bar", 1e-4)),
            cosine_s=float(config.get("diffusion_cosine_s", 0.008)),
            beta_start=float(config.get("diffusion_beta_start", 0.0001)),
            beta_end=float(config.get("diffusion_beta_end", 0.02)),
            residual_norm_clip=float(config.get("residual_norm_clip", 2.0)),
        )
    raise ValueError(f"unknown proposal generator: {proposal_generator}")


def _effective_pool_source_weights(
    retrieval_config: Mapping[str, object],
    epoch: int,
) -> dict[str, float]:
    target = _parse_weight_mapping(str(retrieval_config.get("retrieval_pool_source_weights", "")))
    start = _parse_weight_mapping(str(retrieval_config.get("retrieval_pool_source_weights_start", "")))
    warmup_epochs = int(retrieval_config.get("retrieval_pool_source_weights_warmup_epochs", 0))
    if not target or not start or warmup_epochs <= 0:
        return target
    progress = min(1.0, max(0.0, float(epoch) / float(warmup_epochs)))
    labels = sorted(set(target) | set(start))
    return {
        label: (1.0 - progress) * float(start.get(label, target.get(label, 1.0)))
        + progress * float(target.get(label, start.get(label, 1.0)))
        for label in labels
    }


def _with_candidate_labels(batch: NegativeSampleBatch, label: str) -> NegativeSampleBatch:
    return NegativeSampleBatch(
        edges=list(batch.edges),
        source_edges=list(batch.source_edges),
        metadata=dict(batch.metadata),
        candidate_labels=[label] * len(batch.edges),
    )


def _exclude_teacher_candidates_from_retrieval(
    *,
    teacher_batch: NegativeSampleBatch,
    retrieval_batch: NegativeSampleBatch,
    source_edges: list[Hyperedge],
    teacher_block_size: int,
    retrieval_block_size: int,
) -> NegativeSampleBatch:
    labels = (
        retrieval_batch.candidate_labels
        if len(retrieval_batch.candidate_labels) == len(retrieval_batch.edges)
        else ["unknown"] * len(retrieval_batch.edges)
    )
    output_edges: list[Hyperedge] = []
    output_sources: list[Hyperedge] = []
    output_labels: list[str] = []
    removed = 0
    padded = 0
    for source_index, source in enumerate(source_edges):
        teacher_start = source_index * teacher_block_size
        teacher_end = teacher_start + teacher_block_size
        teacher_edges = set(teacher_batch.edges[teacher_start:teacher_end])
        retrieval_start = source_index * retrieval_block_size
        retrieval_end = retrieval_start + retrieval_block_size
        items = [
            (edge, label)
            for edge, label in zip(
                retrieval_batch.edges[retrieval_start:retrieval_end],
                labels[retrieval_start:retrieval_end],
            )
            if edge not in teacher_edges
        ]
        removed += retrieval_block_size - len(items)
        if not items:
            raise RuntimeError(f"all retrieval candidates overlap the risk-controlled teacher pool for source {source}")
        base_items = list(items)
        while len(items) < retrieval_block_size:
            items.append(base_items[len(items) % len(base_items)])
            padded += 1
        items = items[:retrieval_block_size]
        output_edges.extend(edge for edge, _label in items)
        output_labels.extend(label for _edge, label in items)
        output_sources.extend([source] * retrieval_block_size)

    metadata = dict(retrieval_batch.metadata)
    metadata.update(
        {
            "candidate_pool_quota_target_retrieval_only_enabled": 1.0,
            "candidate_pool_teacher_risk_controlled_rate": 1.0,
            "candidate_pool_teacher_overlap_removed": float(removed),
            "candidate_pool_post_exclusion_padded": float(padded),
        }
    )
    total = max(1, len(output_labels))
    for label in ("risk_controlled", "sns", "mns", "cns", "mix", "unknown"):
        metadata[f"candidate_pool_{label}_rate"] = float(output_labels.count(label) / total)
    return NegativeSampleBatch(
        edges=output_edges,
        source_edges=output_sources,
        metadata=metadata,
        candidate_labels=output_labels,
    )


def _split_teacher_retrieval_candidate_batches(
    *,
    candidate_batch: NegativeSampleBatch,
    source_edges: list[Hyperedge],
    block_size: int,
    teacher_block_size: int,
    retrieval_block_size: int,
) -> tuple[NegativeSampleBatch, NegativeSampleBatch]:
    teacher_edges: list[Hyperedge] = []
    teacher_sources: list[Hyperedge] = []
    retrieval_edges: list[Hyperedge] = []
    retrieval_sources: list[Hyperedge] = []
    teacher_labels: list[str] = []
    retrieval_labels: list[str] = []
    deduplicated_sources = 0
    padded_teacher_candidates = 0
    padded_retrieval_candidates = 0
    for source_index, source in enumerate(source_edges):
        start = source_index * block_size
        end = start + block_size
        candidates = candidate_batch.edges[start:end]
        candidate_labels = (
            candidate_batch.candidate_labels[start:end]
            if len(candidate_batch.candidate_labels) == len(candidate_batch.edges)
            else ["unknown"] * len(candidates)
        )
        label_by_candidate = dict(zip(candidates, candidate_labels))
        teacher_candidates = candidates[:teacher_block_size]
        retrieval_candidates = candidates[teacher_block_size : teacher_block_size + retrieval_block_size]
        overlap = set(teacher_candidates) & set(retrieval_candidates)
        if overlap:
            deduplicated_sources += 1
            unique_candidates = list(dict.fromkeys(candidates))
            if len(unique_candidates) < 2:
                raise RuntimeError(
                    "cannot build disjoint teacher/retrieval candidate pools "
                    f"for source {source}: only {len(unique_candidates)} unique candidate(s)"
                )
            retrieval_unique_count = min(retrieval_block_size, max(1, len(unique_candidates) - 1))
            teacher_unique_count = min(teacher_block_size, len(unique_candidates) - retrieval_unique_count)
            teacher_unique_count = max(1, teacher_unique_count)
            teacher_candidates = unique_candidates[:teacher_unique_count]
            retrieval_candidates = unique_candidates[teacher_unique_count : teacher_unique_count + retrieval_unique_count]
            if set(teacher_candidates) & set(retrieval_candidates):
                raise RuntimeError(f"teacher and retrieval candidate pools overlap for source {source} after deduplication")
            teacher_pad = _repeat_to_length(teacher_candidates, teacher_block_size)
            retrieval_pad = _repeat_to_length(retrieval_candidates, retrieval_block_size)
            padded_teacher_candidates += max(0, len(teacher_pad) - len(teacher_candidates))
            padded_retrieval_candidates += max(0, len(retrieval_pad) - len(retrieval_candidates))
            teacher_candidates = teacher_pad
            retrieval_candidates = retrieval_pad
        teacher_edges.extend(teacher_candidates)
        teacher_sources.extend([source] * len(teacher_candidates))
        teacher_labels.extend(label_by_candidate.get(candidate, "unknown") for candidate in teacher_candidates)
        retrieval_edges.extend(retrieval_candidates)
        retrieval_sources.extend([source] * len(retrieval_candidates))
        retrieval_labels.extend(label_by_candidate.get(candidate, "unknown") for candidate in retrieval_candidates)

    metadata = dict(candidate_batch.metadata)
    metadata.update(
        {
            "candidate_pool_partitioned_enabled": 1.0,
            "candidate_pool_teacher_block_size": float(teacher_block_size),
            "candidate_pool_retrieval_block_size": float(retrieval_block_size),
            "candidate_pool_total_block_size": float(block_size),
            "candidate_pool_deduplicated_sources": float(deduplicated_sources),
            "candidate_pool_padded_teacher_candidates": float(padded_teacher_candidates),
            "candidate_pool_padded_retrieval_candidates": float(padded_retrieval_candidates),
        }
    )
    return (
        NegativeSampleBatch(
            edges=teacher_edges,
            source_edges=teacher_sources,
            metadata=metadata,
            candidate_labels=teacher_labels,
        ),
        NegativeSampleBatch(
            edges=retrieval_edges,
            source_edges=retrieval_sources,
            metadata=metadata,
            candidate_labels=retrieval_labels,
        ),
    )


def _repeat_to_length(candidates: list[Hyperedge], target_size: int) -> list[Hyperedge]:
    if target_size <= 0:
        return []
    if not candidates:
        raise RuntimeError("cannot pad an empty candidate partition")
    if len(candidates) >= target_size:
        return candidates[:target_size]
    padded = list(candidates)
    index = 0
    while len(padded) < target_size:
        padded.append(candidates[index % len(candidates)])
        index += 1
    return padded


def _retrieval_weights(
    mode: str,
    retrieval_config: Mapping[str, object],
    effective_lambda: float | None = None,
) -> dict[str, float]:
    retrieval_lambda = _bounded_retrieval_lambda(
        retrieval_config.get("retrieval_lambda", 0.8) if effective_lambda is None else effective_lambda
    )
    reference_lambda = retrieval_lambda
    if mode == "scans_no_diffusion_signal":
        return {
            "lambda": 0.0,
            "reference_lambda": reference_lambda,
            "diffusion": 0.0,
            "boundary": 0.0,
            # Do not silently reallocate the removed diffusion mass to the strong hardness fallback.
            "hardness": 1.0 - reference_lambda,
        }
    if mode in {"scans_no_diffusion", "scans_hard_only", "scans_no_diffusion_proposal"}:
        retrieval_lambda = 0.0
    elif mode == "scans_diffusion_only":
        retrieval_lambda = 1.0
    weights = {
        "lambda": retrieval_lambda,
        "reference_lambda": reference_lambda,
        "diffusion": retrieval_lambda,
        "boundary": 0.0,
        "hardness": 1.0 - retrieval_lambda,
    }
    return weights


def _effective_retrieval_lambda(mode: str, target_lambda: float, lambda_start: float, warmup_epochs: int, epoch_index: int) -> tuple[float, float]:
    if mode in {"scans_no_diffusion", "scans_hard_only", "scans_no_diffusion_proposal"}:
        return 0.0, 1.0
    if mode == "scans_diffusion_only":
        return 1.0, 1.0
    target = _bounded_retrieval_lambda(target_lambda)
    start = _bounded_retrieval_lambda(lambda_start)
    progress = min(1.0, max(0.0, float(epoch_index) / max(1.0, float(warmup_epochs))))
    return _bounded_retrieval_lambda(start + progress * (target - start)), progress


def _configured_retrieval_lambda_target(retrieval_config: Mapping[str, object]) -> float:
    return float(
        retrieval_config.get(
            "retrieval_lambda_target",
            retrieval_config.get("retrieval_lambda", 0.8),
        )
    )


def _bounded_retrieval_lambda(value: object) -> float:
    return max(0.0, min(1.0, float(value)))


def _classifier_logits_frozen(model: BenchmarkBackbonePredictor, representations: torch.Tensor) -> torch.Tensor:
    original = [parameter.requires_grad for parameter in model.classifier.parameters()]
    try:
        for parameter in model.classifier.parameters():
            parameter.requires_grad_(False)
        return model.logits_from_edge_representations(representations)
    finally:
        for parameter, requires_grad in zip(model.classifier.parameters(), original):
            parameter.requires_grad_(requires_grad)


def _boundary_loss(model: BenchmarkBackbonePredictor, generated: torch.Tensor, target: float) -> torch.Tensor:
    logits = _classifier_logits_frozen(model, generated)
    scores = torch.sigmoid(logits)
    return (scores - float(target)).square().mean()


def _positive_manifold_risk(generated: torch.Tensor, positive_bank: torch.Tensor, margin: float) -> tuple[torch.Tensor, dict[str, float]]:
    generated_norm = F.normalize(generated, dim=1)
    bank_norm = F.normalize(positive_bank, dim=1)
    similarities = generated_norm @ bank_norm.T
    nearest = similarities.max(dim=1).values
    violation = (nearest - float(margin)).clamp_min(0.0)
    return violation.square().mean(), {
        "positive_manifold_similarity_mean": float(nearest.detach().mean().cpu().item()),
        "positive_manifold_similarity_max": float(nearest.detach().max().cpu().item()),
        "positive_manifold_violation_rate": float((violation.detach() > 0).float().mean().cpu().item()),
    }


def _proposal_tensor(sample_output: torch.Tensor | tuple[torch.Tensor, Mapping[str, float]]) -> torch.Tensor:
    return sample_output[0] if isinstance(sample_output, tuple) else sample_output


def _proposal_details(sample_output: torch.Tensor | tuple[torch.Tensor, Mapping[str, float]]) -> dict[str, float]:
    return dict(sample_output[1]) if isinstance(sample_output, tuple) else {}


def _sample_generator(
    generator: nn.Module,
    condition: torch.Tensor,
    *,
    torch_generator: torch.Generator | None = None,
    return_details: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, Mapping[str, float]]:
    try:
        return generator.sample(condition, generator=torch_generator, return_details=return_details)  # type: ignore[call-arg]
    except TypeError:
        return generator.sample(condition, generator=torch_generator)  # type: ignore[call-arg]


def _clone_generator_ema(generator: nn.Module, decay: float) -> nn.Module | None:
    if float(decay) <= 0:
        return None
    ema = copy.deepcopy(generator)
    ema.eval()
    for parameter in ema.parameters():
        parameter.requires_grad_(False)
    return ema


def _update_generator_ema(ema: nn.Module | None, generator: nn.Module, decay: float) -> None:
    if ema is None or float(decay) <= 0:
        return
    with torch.no_grad():
        for ema_parameter, parameter in zip(ema.parameters(), generator.parameters()):
            ema_parameter.mul_(float(decay)).add_(parameter.detach(), alpha=1.0 - float(decay))
        for ema_buffer, buffer in zip(ema.buffers(), generator.buffers()):
            ema_buffer.copy_(buffer.detach())


def _copy_generator_to_ema(ema: nn.Module | None, generator: nn.Module) -> None:
    if ema is not None:
        ema.load_state_dict(generator.state_dict())


def _update_generator_ema_after_step(
    ema: nn.Module | None,
    generator: nn.Module,
    decay: float,
    *,
    completed_updates: int,
    ema_start_step: int,
) -> None:
    if ema is None:
        return
    if ema_start_step > 0 and completed_updates < ema_start_step:
        return
    if ema_start_step > 0 and completed_updates == ema_start_step:
        _copy_generator_to_ema(ema, generator)
        return
    _update_generator_ema(ema, generator, decay)


def _proposal_generator_for_sampling(
    generator: nn.Module,
    generator_ema: nn.Module | None,
    *,
    completed_updates: int,
    ema_start_step: int,
) -> nn.Module:
    if generator_ema is None:
        return generator
    if ema_start_step > 0 and completed_updates < ema_start_step:
        return generator
    return generator_ema


def _sample_risk_bank(positive_representations: torch.Tensor, risk_bank_size: int, rng: random.Random) -> torch.Tensor:
    positive_representations = positive_representations.detach()
    limit = int(risk_bank_size)
    if limit <= 0 or positive_representations.shape[0] <= limit:
        return positive_representations
    indices = rng.sample(range(positive_representations.shape[0]), limit)
    tensor_indices = torch.tensor(indices, device=positive_representations.device, dtype=torch.long)
    return positive_representations.index_select(0, tensor_indices).detach()


def _minibatch_indices(count: int, batch_size: int, device: torch.device, rng: random.Random) -> torch.Tensor:
    if count <= 0:
        raise ValueError("generator update requires at least one positive/teacher pair")
    size = min(count, max(1, int(batch_size)))
    if size >= count:
        indices = list(range(count))
    else:
        indices = rng.sample(range(count), size)
    return torch.tensor(indices, device=device, dtype=torch.long)


def _gradient_norm(parameters: object) -> float:
    total = 0.0
    for parameter in parameters:
        if parameter.grad is None:
            continue
        grad = parameter.grad.detach()
        total += float(grad.norm(2).cpu().item()) ** 2
    return math.sqrt(total)


def _consume_legacy_diffusion_training_rng(
    teacher: torch.Tensor,
    condition: torch.Tensor,
    diffusion_steps: int,
) -> None:
    """Advance the global device RNG exactly as one legacy diffusion update did."""
    if teacher.shape != condition.shape:
        raise ValueError(
            "legacy diffusion RNG replay requires matching teacher/condition shapes: "
            f"teacher={tuple(teacher.shape)} condition={tuple(condition.shape)}"
        )
    batch = int(teacher.shape[0])
    torch.randint(1, int(diffusion_steps) + 1, (batch,), device=teacher.device)
    torch.randn(teacher.shape, device=teacher.device, dtype=teacher.dtype)
    torch.randn(condition.shape, device=condition.device, dtype=condition.dtype)


def _run_generator_updates(
    *,
    generator: nn.Module,
    generator_optimizer: torch.optim.Optimizer,
    generator_ema: nn.Module | None,
    config: Mapping[str, object],
    mode: str,
    teacher_representations: torch.Tensor,
    teacher_conditions: torch.Tensor,
    positive_bank: torch.Tensor,
    rng: random.Random,
    torch_generator: torch.Generator | None = None,
    completed_updates_before: int = 0,
) -> dict[str, float]:
    rc_config = config["scans"]
    updates = int(rc_config.get("generator_updates_per_epoch", 4))
    if updates < 1:
        raise ValueError("generator_updates_per_epoch must be >= 1")
    batch_size = int(rc_config.get("generator_batch_size", 256))
    if teacher_representations.shape != teacher_conditions.shape:
        raise ValueError(
            "generator condition and teacher batch sizes must match: "
            f"teacher={tuple(teacher_representations.shape)} condition={tuple(teacher_conditions.shape)}"
        )
    if teacher_representations.shape[0] <= 0:
        raise ValueError("empty teacher batch for generator update")
    proposal_generator = str(rc_config.get("proposal_generator", "diffusion"))
    native_objective_only = (
        str(rc_config.get("proposal_generator_training", "shared")) == "native"
        and proposal_generator in {"mlp", "mlp_direct", "vae"}
    )
    legacy_rng_replay = bool(rc_config.get("legacy_diffusion_rng_replay", False))
    if legacy_rng_replay and proposal_generator == "diffusion":
        raise ValueError("legacy diffusion RNG replay must not be enabled for the diffusion generator")
    if legacy_rng_replay and torch_generator is None:
        raise ValueError("legacy diffusion RNG replay requires a private replacement-generator RNG")
    use_risk = _uses_generator_risk(mode) and not native_objective_only
    grad_clip = float(rc_config.get("generator_grad_clip", 5.0))
    ema_decay = float(rc_config.get("generator_ema_decay", 0.999))
    ema_start_step = int(rc_config.get("generator_ema_start_step", 0))
    update_metrics: list[dict[str, float]] = []
    for update_index in range(updates):
        indices = _minibatch_indices(teacher_representations.shape[0], batch_size, teacher_representations.device, rng)
        teacher_mb = teacher_representations.index_select(0, indices).detach()
        condition_mb = teacher_conditions.index_select(0, indices).detach()
        if legacy_rng_replay:
            _consume_legacy_diffusion_training_rng(
                teacher_mb,
                condition_mb,
                int(rc_config.get("diffusion_steps", 4)),
            )
        generator.train()
        generator_optimizer.zero_grad(set_to_none=True)
        diffusion_loss, diffusion_metrics = generator.training_loss(
            teacher_mb,
            condition_mb,
            generator=torch_generator,
        )
        generated_mb = _proposal_tensor(
            _sample_generator(generator, condition_mb, torch_generator=torch_generator)
        )
        risk_loss, risk_metrics = (
            _positive_manifold_risk(generated_mb, positive_bank, float(rc_config["risk_margin"]))
            if use_risk
            else (generated_mb.new_tensor(0.0), {})
        )
        diversity_loss = _diversity_loss(generated_mb)
        native_loss_weight = 1.0 if native_objective_only else float(rc_config["diffusion_weight"])
        generator_loss = native_loss_weight * diffusion_loss
        if use_risk:
            generator_loss = generator_loss + float(rc_config["risk_weight"]) * risk_loss
        if not native_objective_only:
            generator_loss = generator_loss + float(rc_config["diversity_weight"]) * diversity_loss
        generator_loss.backward()
        grad_norm_before_clip = _gradient_norm(generator.parameters())
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(generator.parameters(), grad_clip)
        generator_optimizer.step()
        completed_updates = int(completed_updates_before) + update_index + 1
        _update_generator_ema_after_step(
            generator_ema,
            generator,
            ema_decay,
            completed_updates=completed_updates,
            ema_start_step=ema_start_step,
        )
        metrics = {
            "generator_loss": float(generator_loss.detach().cpu().item()),
            "generator_risk_loss": float(risk_loss.detach().cpu().item()),
            "generator_diversity_loss": float(diversity_loss.detach().cpu().item()),
            "generator_native_objective_only": 1.0 if native_objective_only else 0.0,
            "generator_native_loss_weight": float(native_loss_weight),
            "legacy_diffusion_rng_replay": 1.0 if legacy_rng_replay else 0.0,
            "generator_grad_norm": float(grad_norm_before_clip),
            **diffusion_metrics,
            **risk_metrics,
        }
        update_metrics.append(metrics)
    averaged = _average_plain_metrics(update_metrics)
    for key in (
        "diffusion_denoise_mse",
        "residual_target_norm_mean",
        "residual_target_norm_std",
        "sampled_timestep_mean",
        "alpha_bar_sampled_mean",
        "alpha_bar_terminal",
        "positive_manifold_similarity_mean",
        "positive_manifold_similarity_max",
        "positive_manifold_violation_rate",
    ):
        averaged.setdefault(key, 0.0)
    averaged["generator_updates_completed"] = float(updates)
    averaged["generator_updates_total"] = float(int(completed_updates_before) + updates)
    averaged["generator_ema_ready"] = 1.0 if ema_start_step <= 0 or int(completed_updates_before) + updates >= ema_start_step else 0.0
    return averaged


def _average_plain_metrics(summaries: list[dict[str, float]]) -> dict[str, float]:
    if not summaries:
        return {}
    keys = sorted(set().union(*(summary.keys() for summary in summaries)))
    return {key: float(sum(summary.get(key, 0.0) for summary in summaries) / len(summaries)) for key in keys}


def _proposal_diagnostics(generated: torch.Tensor, positive_representations: torch.Tensor, teacher_representations: torch.Tensor) -> dict[str, float]:
    generated_unit = F.normalize(generated.detach(), dim=1, eps=1e-8)
    positive_unit = F.normalize(positive_representations.detach(), dim=1, eps=1e-8)
    teacher_unit = F.normalize(teacher_representations.detach(), dim=1, eps=1e-8)
    if generated_unit.shape != positive_unit.shape or generated_unit.shape != teacher_unit.shape:
        return {}
    residual = generated_unit - positive_unit
    return {
        "proposal_norm_mean": float(generated_unit.norm(dim=1).mean().cpu().item()),
        "proposal_positive_cosine_mean": float((generated_unit * positive_unit).sum(dim=1).mean().cpu().item()),
        "proposal_teacher_cosine_mean": float((generated_unit * teacher_unit).sum(dim=1).mean().cpu().item()),
        "proposal_residual_norm_mean": float(residual.norm(dim=1).mean().cpu().item()),
    }


def _align_teacher_representations_to_sources(
    *,
    teacher_representations: torch.Tensor,
    teacher_batch: NegativeSampleBatch,
    source_edges: list[Hyperedge],
    fallback: torch.Tensor,
) -> torch.Tensor:
    if teacher_representations.shape[0] == len(source_edges):
        return teacher_representations
    source_to_teacher: dict[Hyperedge, int] = {}
    for index, source in enumerate(teacher_batch.source_edges):
        source_to_teacher.setdefault(source, index)
    indices = [source_to_teacher.get(source) for source in source_edges]
    if all(index is not None for index in indices):
        tensor_indices = torch.tensor([int(index) for index in indices if index is not None], device=teacher_representations.device, dtype=torch.long)
        return teacher_representations.index_select(0, tensor_indices)
    if source_edges and teacher_representations.shape[0] % len(source_edges) == 0:
        repeat = teacher_representations.shape[0] // len(source_edges)
        tensor_indices = torch.arange(0, teacher_representations.shape[0], repeat, device=teacher_representations.device, dtype=torch.long)
        return teacher_representations.index_select(0, tensor_indices)
    return fallback


def _diversity_loss(generated: torch.Tensor) -> torch.Tensor:
    if generated.shape[0] < 2:
        return generated.new_tensor(0.0)
    normalized = F.normalize(generated, dim=1)
    similarity = normalized @ normalized.T
    eye = torch.eye(similarity.shape[0], device=similarity.device, dtype=torch.bool)
    return similarity.masked_select(~eye).square().mean()


def _undistilled_teacher(positive: torch.Tensor, rng: random.Random) -> torch.Tensor:
    scale = 0.75 + 0.25 * rng.random()
    return positive + scale * torch.randn_like(positive)


def _teacher_condition_representations(
    *,
    positive_representations: torch.Tensor,
    source_edges: list[Hyperedge],
    teacher_batch: NegativeSampleBatch,
) -> torch.Tensor:
    if len(teacher_batch.edges) == positive_representations.shape[0]:
        return positive_representations

    source_to_index = {edge: index for index, edge in enumerate(source_edges)}
    if len(teacher_batch.source_edges) == len(teacher_batch.edges):
        indices = [source_to_index.get(source) for source in teacher_batch.source_edges]
        if all(index is not None for index in indices):
            tensor_indices = torch.tensor(
                [int(index) for index in indices if index is not None],
                device=positive_representations.device,
                dtype=torch.long,
            )
            return positive_representations.index_select(0, tensor_indices)

    if source_edges and len(teacher_batch.edges) % len(source_edges) == 0:
        repeat = len(teacher_batch.edges) // len(source_edges)
        return positive_representations.repeat_interleave(repeat, dim=0)

    raise ValueError(
        "cannot align teacher negatives with positive conditions: "
        f"teacher_edges={len(teacher_batch.edges)} positives={positive_representations.shape[0]} "
        f"teacher_sources={len(teacher_batch.source_edges)}"
    )


def _sample_teacher_negatives(
    *,
    config: Mapping[str, object],
    mode: str,
    safe_sampler: NegativeSampler,
    sns_sampler: NegativeSampler,
    model: BenchmarkBackbonePredictor,
    source_edges: list[Hyperedge],
    num_nodes: int,
    positive_edges: set[Hyperedge],
    rng: random.Random,
    negatives_per_positive: int,
    candidate_batch: NegativeSampleBatch | None = None,
    block_size: int | None = None,
) -> NegativeSampleBatch:
    if mode == "sehp_diffusion":
        return sns_sampler.sample(source_edges, num_nodes, positive_edges, rng, negatives_per_positive)
    if candidate_batch is None or block_size is None:
        candidate_multiplier = max(1, int(config["sampling"].get("rerank_candidate_multiplier", 4)))
        candidate_batch, block_size = _sample_retrieval_candidate_batch(
            config=config,
            mode=mode,
            safe_sampler=safe_sampler,
            source_edges=source_edges,
            num_nodes=num_nodes,
            positive_edges=positive_edges,
            rng=rng,
            negatives_per_positive=negatives_per_positive,
            candidate_multiplier=candidate_multiplier,
            model=model,
        )
    selected_edges: list[Hyperedge] = []
    selected_sources: list[Hyperedge] = []
    teacher_mode = str(config["scans"].get("teacher_mode", "safe_hard"))
    candidate_scores = _score_candidate_hardness_in_chunks(model, candidate_batch.edges)
    if len(candidate_scores) != len(candidate_batch.edges):
        raise RuntimeError(
            "teacher candidate score count mismatch: "
            f"scores={len(candidate_scores)} candidates={len(candidate_batch.edges)}"
        )
    for source_index, source in enumerate(source_edges):
        start = source_index * block_size
        end = start + block_size
        candidates = candidate_batch.edges[start:end]
        if not candidates:
            continue
        scores = candidate_scores[start:end]
        order = sorted(range(len(candidates)), key=lambda index: scores[index], reverse=True)
        for index in order[:negatives_per_positive]:
            selected_edges.append(candidates[index])
            selected_sources.append(source)
    metadata = dict(candidate_batch.metadata)
    metadata["teacher_mode"] = 1.0 if teacher_mode != "safe_hard" else 0.0
    return NegativeSampleBatch(edges=selected_edges, source_edges=selected_sources, metadata=metadata)


def _sample_diffusion_to_discrete_negatives(
    *,
    config: Mapping[str, object],
    mode: str,
    safe_sampler: NegativeSampler,
    model: BenchmarkBackbonePredictor,
    generated: torch.Tensor | None,
    source_edges: list[Hyperedge],
    num_nodes: int,
    positive_edges: set[Hyperedge],
    rng: random.Random,
    negatives_per_positive: int,
    candidate_batch: NegativeSampleBatch | None = None,
    block_size: int | None = None,
    effective_lambda: float | None = None,
    teacher_batch: NegativeSampleBatch | None = None,
) -> NegativeSampleBatch:
    retrieval_config = config["scans"]
    candidate_multiplier = max(1, int(retrieval_config.get("retrieval_candidate_multiplier", 8)))
    if candidate_batch is None or block_size is None:
        candidate_batch, block_size = _sample_retrieval_candidate_batch(
            config=config,
            mode=mode,
            safe_sampler=safe_sampler,
            source_edges=source_edges,
            num_nodes=num_nodes,
            positive_edges=positive_edges,
            rng=rng,
            negatives_per_positive=negatives_per_positive,
            candidate_multiplier=candidate_multiplier,
            model=model,
        )
    selected_edges: list[Hyperedge] = []
    selected_sources: list[Hyperedge] = []
    selected_combined: list[float] = []
    selected_diffusion: list[float] = []
    selected_hardness: list[float] = []
    selected_labels: list[str] = []
    candidate_combined: list[float] = []
    candidate_diffusion: list[float] = []
    candidate_hardness: list[float] = []
    candidate_source_counts: dict[str, int] = {}
    selected_source_counts: dict[str, int] = {}
    selected_source_weights = _parse_weight_mapping(
        str(retrieval_config.get("retrieval_selected_source_weights", ""))
    )
    weights = _retrieval_weights(mode, retrieval_config, effective_lambda=effective_lambda)
    retrieval_lambda = weights["lambda"]
    reference_lambda = weights["reference_lambda"]
    diffusion_weight = weights["diffusion"]
    hardness_weight = weights["hardness"]
    selection_strategy, retrieval_top_k = _effective_retrieval_selection(
        mode,
        retrieval_config.get("retrieval_selection_strategy", "top_score"),
        retrieval_config.get("retrieval_top_k", 1),
    )
    retrieval_temperature = float(retrieval_config.get("retrieval_temperature", 1.0))
    normalization = str(retrieval_config.get("retrieval_normalization", "local_zscore"))
    per_sampler = negatives_per_positive * candidate_multiplier
    pool = _effective_candidate_pool(
        mode,
        retrieval_config.get("retrieval_candidate_pool", "risk_controlled"),
    )
    if diffusion_weight > 0.0:
        if generated is None:
            raise ValueError(f"{mode} has non-zero diffusion retrieval weight but no proposal")
        generated = F.normalize(generated, dim=1)
    source_chunk_size = _retrieval_source_chunk_size(block_size)
    for source_start in range(0, len(source_edges), source_chunk_size):
        source_end = min(len(source_edges), source_start + source_chunk_size)
        edge_start = source_start * block_size
        edge_end = source_end * block_size
        chunk_edges = candidate_batch.edges[edge_start:edge_end]
        chunk_labels = (
            candidate_batch.candidate_labels[edge_start:edge_end]
            if len(candidate_batch.candidate_labels) == len(candidate_batch.edges)
            else ["unknown"] * len(chunk_edges)
        )
        if not chunk_edges:
            continue
        chunk_representations, chunk_hardness = _score_retrieval_candidates_once(
            model=model,
            candidate_edges=chunk_edges,
        )
        chunk_representations = F.normalize(chunk_representations, dim=1)
        generated_chunk = (
            generated[source_start:source_end].to(chunk_representations.device)
            if generated is not None
            else None
        )

        for local_source_index, source in enumerate(source_edges[source_start:source_end]):
            local_start = local_source_index * block_size
            local_end = local_start + block_size
            candidates = chunk_edges[local_start:local_end]
            if not candidates:
                continue
            if teacher_batch is not None:
                teacher_for_source = {
                    edge for edge, teacher_source in zip(teacher_batch.edges, teacher_batch.source_edges) if teacher_source == source
                }
                overlap = set(candidates) & teacher_for_source
                if overlap:
                    raise RuntimeError(f"retrieval candidates contain teacher negative for source {source}: {len(overlap)} overlap")
            if diffusion_weight > 0.0 and generated_chunk is not None:
                with torch.no_grad():
                    diffusion_scores = (
                        chunk_representations[local_start:local_end]
                        @ generated_chunk[local_source_index].unsqueeze(1)
                    ).squeeze(1).detach().cpu().tolist()
            else:
                diffusion_scores = [0.0] * len(candidates)
            hardness_scores = chunk_hardness[local_start:local_end]
            source_candidate_labels = chunk_labels[local_start:local_end]
            pool_labels = (
                source_candidate_labels
                if any(label != "unknown" for label in source_candidate_labels)
                else _retrieval_pool_labels(pool, len(candidates), per_sampler)
            )
            for label in source_candidate_labels:
                candidate_source_counts[label] = candidate_source_counts.get(label, 0) + 1

            diffusion_norm = _normalize_scores(diffusion_scores, normalization, pool_labels)
            hardness_norm = _normalize_scores(hardness_scores, normalization, pool_labels)
            combined_scores = [
                diffusion_weight * diffusion_norm[index]
                + hardness_weight * hardness_norm[index]
                for index in range(len(candidates))
            ]
            order = sorted(range(len(candidates)), key=lambda index: combined_scores[index], reverse=True)
            candidate_combined.extend(combined_scores)
            candidate_diffusion.extend(diffusion_scores)
            candidate_hardness.extend(hardness_scores)
            if selected_source_weights and any(label != "unknown" for label in source_candidate_labels):
                selected_indices = _select_retrieval_indices_with_source_mix(
                    order=order,
                    scores=combined_scores,
                    candidate_labels=source_candidate_labels,
                    negatives_per_positive=negatives_per_positive,
                    strategy=selection_strategy,
                    top_k=retrieval_top_k,
                    temperature=retrieval_temperature,
                    source_weights=selected_source_weights,
                    source_counts=selected_source_counts,
                    rng=rng,
                )
            else:
                selected_indices = _select_retrieval_indices(
                    order=order,
                    scores=combined_scores,
                    negatives_per_positive=negatives_per_positive,
                    strategy=selection_strategy,
                    top_k=retrieval_top_k,
                    temperature=retrieval_temperature,
                    rng=rng,
                )
            for index in selected_indices:
                selected_edges.append(candidates[index])
                selected_sources.append(source)
                selected_combined.append(combined_scores[index])
                selected_diffusion.append(diffusion_scores[index])
                selected_hardness.append(hardness_scores[index])
                selected_label = source_candidate_labels[index]
                selected_labels.append(selected_label)
                if not selected_source_weights:
                    selected_source_counts[selected_label] = selected_source_counts.get(selected_label, 0) + 1

    metadata = dict(candidate_batch.metadata)
    metadata.update(
        {
            "discrete_retrieval_enabled": 1.0,
            "candidate_multiplier": float(candidate_multiplier),
            "retrieval_lambda": retrieval_lambda,
            "reference_retrieval_lambda": reference_lambda,
            "diffusion_weight": diffusion_weight,
            "boundary_weight": 0.0,
            "hardness_weight": hardness_weight,
            "retrieval_top_k": float(retrieval_top_k),
            "retrieval_temperature": retrieval_temperature,
            "retrieval_topk_sample_enabled": 1.0 if selection_strategy == "topk_sample" else 0.0,
            "retrieval_topk_multi_enabled": 1.0 if selection_strategy == "topk_multi" else 0.0,
            "retrieval_deterministic_top1_enabled": 1.0 if selection_strategy == "top_score" and retrieval_top_k == 1 else 0.0,
            "diffusion_proposal_enabled": 1.0 if generated is not None else 0.0,
            "learned_diffusion_signal_enabled": (
                0.0 if mode in {"scans_no_diffusion_signal", "scans_random_proposal_control"} else 1.0
            ),
            "random_proposal_control_enabled": 1.0 if mode == "scans_random_proposal_control" else 0.0,
            "risk_control_enabled": 0.0 if pool == "uncontrolled_union" else 1.0,
            "retrieval_normalization_none_enabled": 1.0 if normalization == "none" else 0.0,
            "retrieval_normalization_robust_zscore_enabled": 1.0 if normalization == "robust_zscore" else 0.0,
            "retrieval_normalization_local_rank_enabled": 1.0 if normalization == "local_rank" else 0.0,
            "retrieval_normalization_pool_zscore_enabled": 1.0 if normalization == "pool_zscore" else 0.0,
            "retrieval_normalization_pool_rank_enabled": 1.0 if normalization == "pool_rank" else 0.0,
            "selected_combined_score_mean": _mean(selected_combined),
            "candidate_combined_score_mean": _mean(candidate_combined),
            "selected_diffusion_similarity_mean": _mean(selected_diffusion),
            "candidate_diffusion_similarity_mean": _mean(candidate_diffusion),
            "selected_hardness_score_mean": _mean(selected_hardness),
            "candidate_hardness_score_mean": _mean(candidate_hardness),
            "selected_source_mix_control_enabled": 1.0 if selected_source_weights else 0.0,
        }
    )
    for label, weight in selected_source_weights.items():
        metadata[f"selected_source_requested_weight_{label}"] = float(weight)
    candidate_total = max(1, sum(candidate_source_counts.values()))
    selected_total = max(1, sum(selected_source_counts.values()))
    for label in sorted(set(candidate_source_counts) | set(selected_source_counts)):
        metadata[f"effective_candidate_source_{label}_rate"] = float(
            candidate_source_counts.get(label, 0) / candidate_total
        )
        metadata[f"selected_negative_source_{label}_rate"] = float(
            selected_source_counts.get(label, 0) / selected_total
        )
    return NegativeSampleBatch(
        edges=selected_edges,
        source_edges=selected_sources,
        metadata=metadata,
        candidate_labels=selected_labels,
    )


def _retrieval_source_chunk_size(block_size: int) -> int:
    # Keep candidate scoring batched enough to feed the GPU while avoiding DBLP/Recipe-sized
    # all-candidate tensors that can exceed available contiguous memory.
    target_candidates = 32768
    return max(1, min(512, target_candidates // max(1, int(block_size))))


def _score_retrieval_candidates_once(
    *,
    model: BenchmarkBackbonePredictor,
    candidate_edges: list[Hyperedge],
) -> tuple[torch.Tensor, list[float]]:
    if not candidate_edges:
        device = next(model.parameters()).device
        return torch.empty((0, 0), device=device), []

    was_training = model.training
    model.eval()
    with torch.no_grad():
        nodes = model.encode_all_nodes()
        candidate_representations = model.edge_representations(candidate_edges, nodes)
        candidate_probabilities = torch.sigmoid(
            model.logits_from_edge_representations(candidate_representations)
        )
        hardness_scores = candidate_probabilities.detach().cpu().tolist()
    if was_training:
        model.train()
    return candidate_representations.detach(), hardness_scores


def _sample_retrieval_candidate_batch(
    *,
    config: Mapping[str, object],
    mode: str,
    safe_sampler: NegativeSampler,
    source_edges: list[Hyperedge],
    num_nodes: int,
    positive_edges: set[Hyperedge],
    rng: random.Random,
    negatives_per_positive: int,
    candidate_multiplier: int,
    model: BenchmarkBackbonePredictor,
    sampler_cache: dict[str, NegativeSampler] | None = None,
    candidate_partition_sizes: tuple[int, int] | None = None,
    source_weights_override: Mapping[str, float] | None = None,
    _allow_chunking: bool = True,
) -> tuple[NegativeSampleBatch, int]:
    pool = _effective_candidate_pool(
        mode,
        config["scans"].get("retrieval_candidate_pool", "risk_controlled"),
    )
    target_block_size = negatives_per_positive * candidate_multiplier
    source_chunk_size = int(config["scans"].get("retrieval_sampling_source_chunk_size", 0))
    if _allow_chunking and source_chunk_size > 0 and len(source_edges) > source_chunk_size:
        chunk_batches: list[NegativeSampleBatch] = []
        chunk_count = math.ceil(len(source_edges) / source_chunk_size)
        report_every = max(1, chunk_count // 4)
        for chunk_index, start in enumerate(range(0, len(source_edges), source_chunk_size), start=1):
            chunk_batch, chunk_block_size = _sample_retrieval_candidate_batch(
                config=config,
                mode=mode,
                safe_sampler=safe_sampler,
                source_edges=source_edges[start : start + source_chunk_size],
                num_nodes=num_nodes,
                positive_edges=positive_edges,
                rng=rng,
                negatives_per_positive=negatives_per_positive,
                candidate_multiplier=candidate_multiplier,
                model=model,
                sampler_cache=sampler_cache,
                candidate_partition_sizes=candidate_partition_sizes,
                source_weights_override=source_weights_override,
                _allow_chunking=False,
            )
            if chunk_block_size != target_block_size:
                raise RuntimeError(
                    f"streamed candidate block mismatch: {chunk_block_size} versus {target_block_size}"
                )
            chunk_batches.append(chunk_batch)
            if chunk_index == 1 or chunk_index == chunk_count or chunk_index % report_every == 0:
                print(f"candidate_sampling chunk {chunk_index}/{chunk_count}", flush=True)
        metadata = _merge_streamed_candidate_metadata(chunk_batches)
        metadata["candidate_pool_streaming_enabled"] = 1.0
        metadata["candidate_pool_streaming_chunks"] = float(chunk_count)
        metadata["candidate_pool_streaming_source_chunk_size"] = float(source_chunk_size)
        return NegativeSampleBatch(
            edges=[edge for batch in chunk_batches for edge in batch.edges],
            source_edges=[edge for batch in chunk_batches for edge in batch.source_edges],
            metadata=metadata,
            candidate_labels=[label for batch in chunk_batches for label in batch.candidate_labels],
        ), target_block_size
    if pool == "risk_controlled":
        batch = safe_sampler.sample(
            source_edges=source_edges,
            num_nodes=num_nodes,
            positive_edges=positive_edges,
            rng=rng,
            negatives_per_positive=target_block_size,
        )
        return _with_candidate_labels(batch, "risk_controlled"), target_block_size

    min_size, max_size = _edge_size_range(source_edges)
    if pool == "uncontrolled_union":
        sampler_names = ["sns", "mns", "cns", "mix"]
    elif pool == "risk_sns":
        sampler_names = ["risk_controlled", "sns"]
    else:
        sampler_names = ["risk_controlled", "sns", "mns", "cns", "mix"]
    sampler_batches: list[tuple[str, NegativeSampleBatch]] = []
    for name in sampler_names:
        if name == "risk_controlled":
            sampler = safe_sampler
        elif sampler_cache is None:
            sampler = build_sampler(name, config["sampling"], min_size, max_size)
        else:
            sampler = sampler_cache.get(name)
            if sampler is None:
                sampler = build_sampler(name, config["sampling"], min_size, max_size)
                sampler_cache[name] = sampler
        sampler_batches.append(
            (
                name,
                sampler.sample(
                    source_edges=source_edges,
                    num_nodes=num_nodes,
                    positive_edges=positive_edges,
                    rng=rng,
                    negatives_per_positive=target_block_size,
                ),
            )
        )

    source_candidate_items: list[list[tuple[str, Hyperedge]]] = []
    raw_sizes: list[int] = []
    unique_sizes: list[int] = []
    for source_index, _source in enumerate(source_edges):
        candidate_items: list[tuple[str, Hyperedge]] = []
        seen: set[Hyperedge] = set()
        raw_size = 0
        for name, batch in sampler_batches:
            start = source_index * target_block_size
            end = start + target_block_size
            candidates = batch.edges[start:end]
            raw_size += len(candidates)
            for candidate in candidates:
                if candidate in seen:
                    continue
                seen.add(candidate)
                candidate_items.append((name, candidate))
        if not candidate_items:
            raise RuntimeError(f"candidate union is empty for source {source_edges[source_index]}")
        source_candidate_items.append(candidate_items)
        raw_sizes.append(raw_size)
        unique_sizes.append(len(candidate_items))

    hardness_scores: list[float] = []
    source_offsets: list[tuple[int, int]] = []
    if pool == "union_hard":
        flat_edges: list[Hyperedge] = []
        for candidate_items in source_candidate_items:
            start = len(flat_edges)
            flat_edges.extend(candidate for _, candidate in candidate_items)
            source_offsets.append((start, len(flat_edges)))
        hardness_scores = _score_candidate_hardness_in_chunks(model, flat_edges)

    edges: list[Hyperedge] = []
    sources: list[Hyperedge] = []
    candidate_labels: list[str] = []
    source_counts = {name: 0 for name in sampler_names}
    source_weights = (
        dict(source_weights_override)
        if source_weights_override is not None
        else _parse_weight_mapping(str(config["scans"].get("retrieval_pool_source_weights", "")))
    )
    if candidate_partition_sizes is not None and sum(candidate_partition_sizes) != target_block_size:
        raise ValueError(
            "candidate partition sizes must sum to the total candidate block size: "
            f"{candidate_partition_sizes} versus {target_block_size}"
        )
    partition_names = ("teacher", "retrieval")
    partition_source_counts = {
        partition_name: {name: 0 for name in sampler_names}
        for partition_name in partition_names
    }
    padded_candidates = 0
    for source_index, (source, candidate_items) in enumerate(zip(source_edges, source_candidate_items)):
        if source_weights:
            start, end = source_offsets[source_index] if pool == "union_hard" else (0, 0)
            source_scores = hardness_scores[start:end] if pool == "union_hard" else None
            if candidate_partition_sizes is not None:
                selected_partitions, partition_padding = _select_partitioned_source_weighted_candidates(
                    candidate_items=candidate_items,
                    hardness_scores=source_scores,
                    partition_sizes=candidate_partition_sizes,
                    source_weights=source_weights,
                    rng=rng,
                )
                selected_items = [item for partition in selected_partitions for item in partition]
                padded_candidates += partition_padding
                for partition_name, partition in zip(partition_names, selected_partitions):
                    for name, _candidate in partition:
                        partition_source_counts[partition_name][name] += 1
            else:
                selected_items = _select_source_weighted_candidates(
                    candidate_items=candidate_items,
                    hardness_scores=source_scores,
                    target_size=target_block_size,
                    source_weights=source_weights,
                    rng=rng,
                )
        elif pool == "union_hard":
            start, end = source_offsets[source_index]
            source_scores = hardness_scores[start:end]
            order = sorted(range(len(candidate_items)), key=lambda index: source_scores[index], reverse=True)
            selected_items = [candidate_items[index] for index in order[:target_block_size]]
        else:
            selected_items = list(candidate_items)
            rng.shuffle(selected_items)
            selected_items = selected_items[:target_block_size]

        if len(selected_items) < target_block_size:
            base_items = list(selected_items)
            if not base_items:
                raise RuntimeError(f"cannot pad an empty candidate union for source {source}")
            index = 0
            while len(selected_items) < target_block_size:
                selected_items.append(base_items[index % len(base_items)])
                index += 1
                padded_candidates += 1

        for name, candidate in selected_items:
            source_counts[name] += 1
            edges.append(candidate)
            sources.append(source)
            candidate_labels.append(name)

    total = max(1, len(edges))
    metadata = {
        "candidate_pool_union_enabled": 1.0,
        "candidate_pool_risk_sns_enabled": 1.0 if pool == "risk_sns" else 0.0,
        "candidate_pool_union_hard_enabled": 1.0 if pool == "union_hard" else 0.0,
        "candidate_pool_uncontrolled_union_enabled": 1.0 if pool == "uncontrolled_union" else 0.0,
        "candidate_pool_block_size": float(target_block_size),
        "candidate_pool_raw_size_mean": _mean([float(value) for value in raw_sizes]),
        "candidate_pool_unique_size_mean": _mean([float(value) for value in unique_sizes]),
        "candidate_pool_padded_candidates": float(padded_candidates),
    }
    for name, count in source_counts.items():
        metadata[f"candidate_pool_{name}_rate"] = float(count / total)
        metadata[f"candidate_pool_{name}_requested_weight"] = float(source_weights.get(name, 1.0))
    if source_weights and candidate_partition_sizes is not None:
        metadata["candidate_pool_independent_partition_quota_enabled"] = 1.0
        for partition_name, partition_size in zip(partition_names, candidate_partition_sizes):
            partition_total = max(1, len(source_edges) * partition_size)
            for name, count in partition_source_counts[partition_name].items():
                metadata[f"candidate_pool_{partition_name}_{name}_rate"] = float(count / partition_total)
    return NegativeSampleBatch(
        edges=edges,
        source_edges=sources,
        metadata=metadata,
        candidate_labels=candidate_labels,
    ), target_block_size


def _merge_streamed_candidate_metadata(batches: list[NegativeSampleBatch]) -> dict[str, float]:
    if not batches:
        return {}
    keys = {key for batch in batches for key in batch.metadata}
    total = max(1, sum(len(batch.edges) for batch in batches))
    merged: dict[str, float] = {}
    for key in keys:
        if key.endswith("padded_candidates"):
            merged[key] = float(sum(float(batch.metadata.get(key, 0.0)) for batch in batches))
        else:
            merged[key] = float(
                sum(float(batch.metadata.get(key, 0.0)) * len(batch.edges) for batch in batches) / total
            )
    return merged


def _select_source_weighted_candidates(
    *,
    candidate_items: list[tuple[str, Hyperedge]],
    hardness_scores: list[float] | None,
    target_size: int,
    source_weights: Mapping[str, float],
    rng: random.Random,
) -> list[tuple[str, Hyperedge]]:
    if target_size <= 0 or not candidate_items:
        return []
    grouped: dict[str, list[int]] = {}
    for index, (label, _) in enumerate(candidate_items):
        grouped.setdefault(label, []).append(index)
    labels = [label for label in sorted(grouped) if float(source_weights.get(label, 1.0)) > 0.0]
    if not labels:
        labels = sorted(grouped)
    weight_sum = sum(float(source_weights.get(label, 1.0)) for label in labels)
    weight_sum = max(weight_sum, 1e-12)
    exact = {
        label: target_size * float(source_weights.get(label, 1.0)) / weight_sum
        for label in labels
    }
    quotas = {label: int(exact[label]) for label in labels}
    remaining = target_size - sum(quotas.values())
    for label in sorted(labels, key=lambda item: (exact[item] - quotas[item], item), reverse=True)[:remaining]:
        quotas[label] += 1

    selected_indices: list[int] = []
    for label in labels:
        indices = list(grouped[label])
        if hardness_scores is not None:
            indices.sort(key=lambda index: hardness_scores[index], reverse=True)
        else:
            rng.shuffle(indices)
        selected_indices.extend(indices[: quotas[label]])

    selected_set = set(selected_indices)
    unused = [index for index in range(len(candidate_items)) if index not in selected_set]
    if hardness_scores is not None:
        unused.sort(key=lambda index: hardness_scores[index], reverse=True)
    else:
        rng.shuffle(unused)
    selected_indices.extend(unused[: max(0, target_size - len(selected_indices))])
    return [candidate_items[index] for index in selected_indices[:target_size]]


def _select_partitioned_source_weighted_candidates(
    *,
    candidate_items: list[tuple[str, Hyperedge]],
    hardness_scores: list[float] | None,
    partition_sizes: tuple[int, int],
    source_weights: Mapping[str, float],
    rng: random.Random,
) -> tuple[list[list[tuple[str, Hyperedge]]], int]:
    if any(size <= 0 for size in partition_sizes):
        raise ValueError(f"candidate partition sizes must be positive: {partition_sizes}")
    if len(candidate_items) < len(partition_sizes):
        raise RuntimeError(
            "cannot construct disjoint candidate partitions: "
            f"{len(candidate_items)} unique candidates for {len(partition_sizes)} partitions"
        )

    remaining = list(range(len(candidate_items)))
    partitions: list[list[tuple[str, Hyperedge]]] = []
    padded_candidates = 0
    for partition_index, target_size in enumerate(partition_sizes):
        partitions_left = len(partition_sizes) - partition_index - 1
        selectable_size = min(target_size, len(remaining) - partitions_left)
        available_items = [candidate_items[index] for index in remaining]
        available_scores = (
            [hardness_scores[index] for index in remaining]
            if hardness_scores is not None
            else None
        )
        selected = _select_source_weighted_candidates(
            candidate_items=available_items,
            hardness_scores=available_scores,
            target_size=selectable_size,
            source_weights=source_weights,
            rng=rng,
        )
        if not selected:
            raise RuntimeError(f"candidate partition {partition_index} is empty")

        selected_set = set(selected)
        remaining = [index for index in remaining if candidate_items[index] not in selected_set]
        unique_selected = list(selected)
        while len(selected) < target_size:
            selected.append(unique_selected[len(selected) % len(unique_selected)])
            padded_candidates += 1
        partitions.append(selected)

    partition_edge_sets = [set(candidate for _label, candidate in partition) for partition in partitions]
    if partition_edge_sets[0] & partition_edge_sets[1]:
        raise RuntimeError("teacher and retrieval source-weighted candidate partitions overlap")
    return partitions, padded_candidates


def _score_candidate_hardness_in_chunks(
    model: BenchmarkBackbonePredictor,
    candidate_edges: list[Hyperedge],
) -> list[float]:
    if not candidate_edges:
        return []
    was_training = model.training
    model.eval()
    scores: list[float] = []
    chunk_size = 32768
    with torch.no_grad():
        nodes = model.encode_all_nodes()
        for start in range(0, len(candidate_edges), chunk_size):
            representations = model.edge_representations(
                candidate_edges[start : start + chunk_size],
                nodes,
            )
            probabilities = torch.sigmoid(model.logits_from_edge_representations(representations))
            scores.extend(probabilities.detach().cpu().tolist())
    if was_training:
        model.train()
    return scores


def _standardize(values: list[float]) -> list[float]:
    if not values:
        return []
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    std = math.sqrt(variance)
    if std <= 1e-12:
        return [0.0 for _ in values]
    return [(value - mean) / std for value in values]


def _robust_standardize(values: list[float]) -> list[float]:
    if not values:
        return []
    ordered = sorted(values)
    median = ordered[len(ordered) // 2] if len(ordered) % 2 else 0.5 * (ordered[len(ordered) // 2 - 1] + ordered[len(ordered) // 2])
    deviations = sorted(abs(value - median) for value in values)
    mad = deviations[len(deviations) // 2] if len(deviations) % 2 else 0.5 * (deviations[len(deviations) // 2 - 1] + deviations[len(deviations) // 2])
    scale = 1.4826 * mad
    if scale <= 1e-12:
        return [0.0 for _ in values]
    return [(value - median) / scale for value in values]


def _rank_normalize(values: list[float]) -> list[float]:
    if not values:
        return []
    if len(values) == 1:
        return [0.0]
    order = sorted(range(len(values)), key=lambda index: values[index])
    normalized = [0.0 for _ in values]
    denominator = max(1, len(values) - 1)
    for rank, index in enumerate(order):
        normalized[index] = 2.0 * (rank / denominator) - 1.0
    return normalized


def _normalize_scores(values: list[float], mode: str, pool_labels: list[str] | None = None) -> list[float]:
    if mode == "none":
        return list(values)
    if mode == "robust_zscore":
        return _robust_standardize(values)
    if mode == "local_rank":
        return _rank_normalize(values)
    if mode in {"pool_zscore", "pool_rank"} and pool_labels and len(pool_labels) == len(values):
        normalized = [0.0 for _ in values]
        labels = sorted(set(pool_labels))
        for label in labels:
            indices = [index for index, item in enumerate(pool_labels) if item == label]
            subset = [values[index] for index in indices]
            subset_norm = _rank_normalize(subset) if mode == "pool_rank" else _standardize(subset)
            for local_index, global_index in enumerate(indices):
                normalized[global_index] = subset_norm[local_index]
        return normalized
    if mode == "pool_rank":
        return _rank_normalize(values)
    return _standardize(values)


def _retrieval_pool_labels(pool: str, candidate_count: int, per_sampler: int) -> list[str] | None:
    if pool != "union":
        return None
    labels: list[str] = []
    for name in ("risk_controlled", "sns", "mns", "cns", "mix"):
        labels.extend([name] * per_sampler)
    return labels[:candidate_count] if len(labels) >= candidate_count else None


def _select_retrieval_indices(
    *,
    order: list[int],
    scores: list[float],
    negatives_per_positive: int,
    strategy: str,
    top_k: int,
    temperature: float,
    rng: random.Random,
) -> list[int]:
    if negatives_per_positive <= 0 or not order:
        return []
    if strategy in {"top_score", "topk_multi"}:
        return order[:negatives_per_positive]

    pool_size = min(len(order), max(negatives_per_positive, int(top_k)))
    candidates = order[:pool_size]
    selected: list[int] = []
    remaining = list(candidates)
    temp = max(float(temperature), 1e-6)
    while remaining and len(selected) < negatives_per_positive:
        local_scores = [scores[index] / temp for index in remaining]
        max_score = max(local_scores)
        weights = [math.exp(max(-60.0, min(60.0, score - max_score))) for score in local_scores]
        total = sum(weights)
        if not math.isfinite(total) or total <= 0.0:
            selected.append(remaining.pop(0))
            continue
        threshold = rng.random() * total
        cumulative = 0.0
        chosen_position = len(remaining) - 1
        for position, weight in enumerate(weights):
            cumulative += weight
            if cumulative >= threshold:
                chosen_position = position
                break
        selected.append(remaining.pop(chosen_position))
    return selected


def _select_retrieval_indices_with_source_mix(
    *,
    order: list[int],
    scores: list[float],
    candidate_labels: list[str],
    negatives_per_positive: int,
    strategy: str,
    top_k: int,
    temperature: float,
    source_weights: Mapping[str, float],
    source_counts: dict[str, int],
    rng: random.Random,
) -> list[int]:
    positive_weights = {
        label: float(weight)
        for label, weight in source_weights.items()
        if float(weight) > 0.0
    }
    weight_sum = sum(positive_weights.values())
    if negatives_per_positive <= 0 or not order or weight_sum <= 0.0:
        return []

    selected: list[int] = []
    remaining = list(order)
    while remaining and len(selected) < negatives_per_positive:
        available_labels = {
            candidate_labels[index]
            for index in remaining
            if candidate_labels[index] in positive_weights
        }
        if available_labels:
            next_total = sum(source_counts.values()) + 1
            chosen_label = max(
                sorted(available_labels),
                key=lambda label: (
                    positive_weights[label] * next_total / weight_sum - source_counts.get(label, 0),
                    positive_weights[label],
                ),
            )
            eligible_order = [index for index in remaining if candidate_labels[index] == chosen_label]
        else:
            eligible_order = list(remaining)

        chosen = _select_retrieval_indices(
            order=eligible_order,
            scores=scores,
            negatives_per_positive=1,
            strategy=strategy,
            top_k=top_k,
            temperature=temperature,
            rng=rng,
        )
        if not chosen:
            break
        chosen_index = chosen[0]
        selected.append(chosen_index)
        remaining.remove(chosen_index)
        chosen_source = candidate_labels[chosen_index]
        source_counts[chosen_source] = source_counts.get(chosen_source, 0) + 1
    return selected


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def _weighted_mean(values: list[tuple[float, float]]) -> float:
    positive_weights = [(float(value), float(weight)) for value, weight in values if weight > 0.0]
    if not positive_weights:
        return 0.0
    total_weight = sum(weight for _, weight in positive_weights)
    if total_weight <= 0.0:
        return 0.0
    return float(sum(value * weight for value, weight in positive_weights) / total_weight)


def _parse_weight_mapping(value: str) -> dict[str, float]:
    weights: dict[str, float] = {}
    for item in str(value).split(","):
        if not item.strip():
            continue
        if ":" not in item:
            raise ValueError(f"invalid weight item {item!r}; expected name:value")
        name, raw_weight = item.split(":", 1)
        weights[name.strip()] = float(raw_weight)
    return weights


def _effective_protocol_replay_weights(
    target_weights: Mapping[str, float],
    *,
    epoch: int,
    warmup_epochs: int,
) -> tuple[dict[str, float], float]:
    if warmup_epochs <= 0:
        progress = 1.0
    else:
        progress = min(1.0, max(0.0, float(epoch + 1) / float(warmup_epochs)))
    return (
        {
            protocol: max(0.0, float(weight)) * progress
            for protocol, weight in target_weights.items()
            if float(weight) > 0.0
        },
        progress,
    )


def _build_training_protocol_replay_banks(
    *,
    config: Mapping[str, object],
    dataset: HypergraphDataset,
    protocol_weights: Mapping[str, float],
    bank_count: int,
    seed: int,
) -> dict[str, list[NegativeSampleBatch]]:
    active_protocols = sorted(
        protocol
        for protocol, weight in protocol_weights.items()
        if float(weight) > 0.0
    )
    if not active_protocols:
        return {}
    if bank_count < 1:
        raise ValueError("protocol replay requires at least one bank")

    min_size, max_size = _edge_size_range(dataset.train_edges)
    banks: dict[str, list[NegativeSampleBatch]] = {}
    for protocol in active_protocols:
        protocol_banks: list[NegativeSampleBatch] = []
        sampler = build_sampler(protocol, config["sampling"], min_size, max_size)
        for bank_index in range(bank_count):
            replay_seed = int(seed) + _stable_offset(f"training_protocol_replay:{protocol}:{bank_index}")
            batch = sampler.sample(
                source_edges=dataset.train_edges,
                num_nodes=dataset.num_nodes,
                positive_edges=dataset.train_positive_edges,
                rng=random.Random(replay_seed),
                negatives_per_positive=1,
            )
            if len(batch.edges) != len(dataset.train_edges) or batch.source_edges != dataset.train_edges:
                raise RuntimeError(f"{protocol} replay bank is not source-aligned")
            if any(edge in dataset.train_positive_edges for edge in batch.edges):
                raise RuntimeError(f"{protocol} replay bank contains a training positive")
            metadata = dict(batch.metadata)
            metadata.update(
                {
                    "protocol_replay_bank_index": float(bank_index),
                    "protocol_replay_seed": float(replay_seed),
                }
            )
            protocol_banks.append(
                NegativeSampleBatch(
                    edges=list(batch.edges),
                    source_edges=list(batch.source_edges),
                    metadata=metadata,
                    candidate_labels=[protocol for _ in batch.edges],
                )
            )
        banks[protocol] = protocol_banks
    return banks


def _edge_representations(model: BenchmarkBackbonePredictor, edges: list[Hyperedge]) -> torch.Tensor:
    nodes = model.encode_all_nodes()
    return model.edge_representations(edges, nodes)


def _build_benchmark_model(config: Mapping[str, object], dataset: HypergraphDataset) -> BenchmarkBackbonePredictor:
    model_config = config["model"]
    model = BenchmarkBackbonePredictor(
        backbone=str(model_config["type"]),
        num_nodes=dataset.num_nodes,
        train_edges=dataset.train_edges,
        embedding_dim=int(model_config["embedding_dim"]),
        hidden_dim=int(model_config["hidden_dim"]),
        dropout=float(model_config["dropout"]),
        node_features=dataset.node_features,
        message_passing_layers=int(model_config.get("message_passing_layers", 2)),
        hnhn_alpha_e=float(model_config.get("hnhn_alpha_e", 0.0)),
        hnhn_alpha_v=float(model_config.get("hnhn_alpha_v", 0.0)),
    )
    return model.to(_resolve_device(str(model_config.get("device", "cpu"))))


def _load_predictor_checkpoint(model: nn.Module, path: Path) -> None:
    device = next(model.parameters()).device
    try:
        payload = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location=device)
    if not isinstance(payload, Mapping) or "predictor" not in payload:
        raise ValueError(f"predictor init checkpoint has no 'predictor' state dict: {path}")
    state = payload["predictor"]
    if not isinstance(state, Mapping):
        raise TypeError(f"invalid predictor state dict in checkpoint: {path}")
    model.load_state_dict(state, strict=True)


def _edge_sequence_digest(edges: list[Hyperedge]) -> str:
    digest = hashlib.sha256()
    for edge in edges:
        digest.update(str(len(edge)).encode("ascii"))
        digest.update(b":")
        digest.update(",".join(str(int(node)) for node in edge).encode("ascii"))
        digest.update(b";")
    return digest.hexdigest()


def _load_replay_negative_bank(path: Path, dataset: HypergraphDataset, seed: int) -> list[list[Hyperedge]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("format") != "ahp_negative_bank_v1":
        raise ValueError(f"unsupported replay negative bank format: {path}")
    if int(payload.get("seed", -1)) != int(seed):
        raise ValueError(f"replay negative bank seed mismatch: expected {seed}, got {payload.get('seed')}")
    if int(payload.get("num_nodes", -1)) != int(dataset.num_nodes):
        raise ValueError("replay negative bank num_nodes mismatch")
    if int(payload.get("train_edge_count", -1)) != len(dataset.train_edges):
        raise ValueError("replay negative bank train-edge count mismatch")
    expected_digest = _edge_sequence_digest(dataset.train_edges)
    if str(payload.get("train_edge_digest", "")) != expected_digest:
        raise ValueError("replay negative bank train-edge alignment digest mismatch")
    banks: list[list[Hyperedge]] = []
    for raw_bank in payload.get("banks", []):
        raw_edges = raw_bank.get("edges", []) if isinstance(raw_bank, Mapping) else raw_bank
        edges = [tuple(int(node) for node in edge) for edge in raw_edges]
        if len(edges) != len(dataset.train_edges):
            raise ValueError("replay negative bank contains an incorrectly sized edge set")
        for source, edge in zip(dataset.train_edges, edges):
            if len(edge) != len(source):
                raise ValueError("replay negative edge cardinality is not source-aligned")
            if edge in dataset.train_positive_edges:
                raise ValueError("replay negative bank contains a training positive")
        banks.append(edges)
    if not banks:
        raise ValueError(f"replay negative bank contains no banks: {path}")
    return banks


def _mix_replay_negatives(
    retrieval_batch: NegativeSampleBatch,
    replay_edges: list[Hyperedge],
    *,
    source_edges: list[Hyperedge],
    replay_ratio: float,
    rng: random.Random,
) -> NegativeSampleBatch:
    if len(replay_edges) != len(source_edges):
        raise ValueError("replay negatives must align one-to-one with source edges")
    source_indices = {edge: index for index, edge in enumerate(source_edges)}
    if len(source_indices) != len(source_edges):
        raise ValueError("replay mixing requires unique source training edges")
    mixed_edges = list(retrieval_batch.edges)
    labels = (
        list(retrieval_batch.candidate_labels)
        if len(retrieval_batch.candidate_labels) == len(mixed_edges)
        else ["scans" for _ in mixed_edges]
    )
    replaced = 0
    for index, source in enumerate(retrieval_batch.source_edges):
        if rng.random() >= replay_ratio:
            continue
        source_index = source_indices.get(source)
        if source_index is None:
            raise ValueError("retrieval source edge is absent from replay alignment")
        mixed_edges[index] = replay_edges[source_index]
        labels[index] = "ahp_replay"
        replaced += 1
    metadata = dict(retrieval_batch.metadata)
    metadata["ahp_replay_requested_ratio"] = float(replay_ratio)
    metadata["ahp_replay_actual_ratio"] = replaced / max(1, len(mixed_edges))
    metadata["ahp_replay_count"] = float(replaced)
    return NegativeSampleBatch(
        edges=mixed_edges,
        source_edges=list(retrieval_batch.source_edges),
        metadata=metadata,
        candidate_labels=labels,
    )


def _resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"requested device {requested!r}, but CUDA is not available")
    return torch.device(requested)


def _torch_generator_for_device(device: torch.device, seed: int) -> torch.Generator:
    try:
        generator = torch.Generator(device=device)
    except (TypeError, RuntimeError):
        generator = torch.Generator()
    generator.manual_seed(int(seed))
    return generator


def _build_protocol_validation_batch(
    config: Mapping[str, object],
    dataset: HypergraphDataset,
    protocols: list[str],
    seed: int,
) -> dict[str, list[Hyperedge] | list[int]]:
    batches = _build_protocol_validation_batches(config, dataset, protocols, seed)
    edges: list[Hyperedge] = []
    labels: list[int] = []
    seen_positive = False
    for batch in batches.values():
        batch_edges = batch["edges"]  # type: ignore[assignment]
        batch_labels = batch["labels"]  # type: ignore[assignment]
        if not seen_positive:
            edges.extend(batch_edges)  # type: ignore[arg-type]
            labels.extend(batch_labels)  # type: ignore[arg-type]
            seen_positive = True
        else:
            for edge, label in zip(batch_edges, batch_labels):  # type: ignore[arg-type]
                if int(label) == 0:
                    edges.append(edge)
                    labels.append(0)
    return {"edges": edges, "labels": labels}


def _build_protocol_validation_batches(
    config: Mapping[str, object],
    dataset: HypergraphDataset,
    protocols: list[str],
    seed: int,
) -> dict[str, dict[str, list[Hyperedge] | list[int]]]:
    bank_path = str(config["training"].get("validation_negative_bank", ""))
    if bank_path:
        return _load_protocol_validation_bank(
            Path(bank_path),
            dataset,
            protocols,
            seed,
            int(config["training"]["negatives_per_positive"]),
        )
    min_size, max_size = _edge_size_range(dataset.train_edges)
    batches: dict[str, dict[str, list[Hyperedge] | list[int]]] = {}
    for protocol in protocols:
        eval_sampler = build_sampler(protocol, config["sampling"], min_size, max_size)
        negative_batch = eval_sampler.sample(
            source_edges=dataset.val_edges,
            num_nodes=dataset.num_nodes,
            positive_edges=dataset.positive_edges,
            rng=random.Random(int(seed) + _stable_offset(protocol)),
            negatives_per_positive=int(config["training"]["negatives_per_positive"]),
        )
        batches[protocol] = {
            "edges": dataset.val_edges + negative_batch.edges,
            "labels": [1] * len(dataset.val_edges) + [0] * len(negative_batch.edges),
        }
    return batches


def _load_protocol_validation_bank(
    path: Path,
    dataset: HypergraphDataset,
    protocols: list[str],
    seed: int,
    negatives_per_positive: int,
) -> dict[str, dict[str, list[Hyperedge] | list[int]]]:
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    if payload.get("format") != "recipe_protocol_bank_v1":
        raise ValueError(f"unsupported validation negative bank format: {path}")
    if int(payload.get("seed", -1)) != int(seed):
        raise ValueError(f"validation negative bank seed mismatch: expected {seed}, got {payload.get('seed')}")
    if int(payload.get("num_nodes", -1)) != int(dataset.num_nodes):
        raise ValueError("validation negative bank num_nodes mismatch")
    if int(payload.get("positive_count", -1)) != len(dataset.positive_edges):
        raise ValueError("validation negative bank positive-edge count mismatch")
    if int(payload.get("negatives_per_positive", -1)) != int(negatives_per_positive):
        raise ValueError("validation negative bank negatives-per-positive mismatch")
    source_edges = [tuple(int(node) for node in edge) for edge in payload.get("source_edges", [])]
    if source_edges != dataset.val_edges:
        raise ValueError("validation negative bank source edges do not match the current validation split")
    stored_protocols = payload.get("protocols", {})
    batches: dict[str, dict[str, list[Hyperedge] | list[int]]] = {}
    for protocol in protocols:
        if protocol not in stored_protocols:
            raise ValueError(f"validation negative bank is missing protocol {protocol}")
        negative_edges = [tuple(int(node) for node in edge) for edge in stored_protocols[protocol]["edges"]]
        if len(negative_edges) != len(dataset.val_edges) * int(negatives_per_positive):
            raise ValueError(f"validation negative bank has an invalid {protocol} edge count")
        expected_sizes = [len(source) for source in dataset.val_edges for _ in range(int(negatives_per_positive))]
        if any(len(edge) != size for edge, size in zip(negative_edges, expected_sizes)):
            raise ValueError(f"validation negative bank changes hyperedge cardinality for {protocol}")
        if any(edge in dataset.positive_edges for edge in negative_edges):
            raise ValueError(f"validation negative bank contains a known positive for {protocol}")
        batches[protocol] = {
            "edges": dataset.val_edges + negative_edges,
            "labels": [1] * len(dataset.val_edges) + [0] * len(negative_edges),
        }
    return batches


def _evaluate_checkpoint_objective(
    model: BenchmarkBackbonePredictor,
    validation_batches: dict[str, dict[str, list[Hyperedge] | list[int]]],
    *,
    objective: str,
    protocol_weights: Mapping[str, float],
    auc_weight: float,
    ap_weight: float,
) -> dict[str, float]:
    aucs: list[tuple[float, float]] = []
    auprs: list[tuple[float, float]] = []
    balanced_scores: list[tuple[float, float]] = []
    protocol_metrics: dict[str, float] = {}
    auc_weight = max(0.0, float(auc_weight))
    ap_weight = max(0.0, float(ap_weight))
    metric_weight_sum = max(1e-12, auc_weight + ap_weight)
    for protocol, batch in validation_batches.items():
        metrics = _evaluate_fixed_edges(
            model,
            batch["edges"],  # type: ignore[arg-type]
            batch["labels"],  # type: ignore[arg-type]
        )
        weight = max(0.0, float(protocol_weights.get(protocol, 1.0)))
        auc = float(metrics["auc"])
        aupr = float(metrics["aupr"])
        aucs.append((auc, weight))
        auprs.append((aupr, weight))
        balanced_scores.append((((auc_weight * auc) + (ap_weight * aupr)) / metric_weight_sum, weight))
        protocol_metrics[f"{protocol}_auc"] = auc
        protocol_metrics[f"{protocol}_ap"] = aupr

    mean_auc = _weighted_mean(aucs)
    mean_aupr = _weighted_mean(auprs)
    mean_score = _weighted_mean(balanced_scores)
    min_auc = min((value for value, _ in aucs), default=0.0)
    min_aupr = min((value for value, _ in auprs), default=0.0)
    min_score = min((value for value, _ in balanced_scores), default=0.0)
    if objective == "mean_auc":
        score = mean_auc
    elif objective == "mean_ap":
        score = mean_aupr
    elif objective == "min_auc_ap":
        score = min_score
    elif objective == "min_auc":
        score = min_auc
    elif objective == "min_ap":
        score = min_aupr
    else:
        score = mean_score
    return {
        "auc": mean_auc,
        "aupr": mean_aupr,
        "score": float(score),
        "min_auc": float(min_auc),
        "min_aupr": float(min_aupr),
        "min_score": float(min_score),
        **protocol_metrics,
    }


def _protocol_list(value: str) -> list[str]:
    protocols = [item.strip() for item in value.split(",") if item.strip()]
    allowed = {dataset_protocol for dataset in SEHP_DATASETS for dataset_protocol in dataset.evaluation_negative_sets}
    unknown = set(protocols) - allowed
    if unknown:
        raise ValueError(f"unknown validation protocols: {unknown}")
    return protocols or ["sns"]


def _evaluate_fixed_edges(model: BenchmarkBackbonePredictor, edges: list[Hyperedge], labels: list[int]) -> dict[str, float]:
    return binary_prediction_metrics(labels, model.predict_scores(edges))


def _evaluate_protocol_negative_sets(
    config: Mapping[str, object],
    dataset_key: str,
    dataset: HypergraphDataset,
    model: BenchmarkBackbonePredictor,
    seed: int,
    bank_path: str = "",
) -> dict[str, dict[str, float]]:
    protocols = list(dataset_by_key(dataset_key).evaluation_negative_sets)
    if bank_path:
        batches = load_protocol_batches(
            bank_path,
            dataset,
            protocols,
            seed=seed,
            source_edges=dataset.test_edges,
            negatives_per_positive=1,
            split="test",
        )
        return {
            name: {
                **binary_prediction_metrics(batch["labels"], model.predict_scores(batch["edges"])),
                **batch["metadata"],
            }
            for name, batch in batches.items()
        }
    min_size, max_size = _edge_size_range(dataset.train_edges)
    test = {}
    for name in protocols:
        test_sampler = build_sampler(name, config["sampling"], min_size, max_size)
        negative = test_sampler.sample(
            dataset.test_edges,
            dataset.num_nodes,
            dataset.positive_edges,
            random.Random(seed + _stable_offset(name)),
            1,
        )
        scores = model.predict_scores(dataset.test_edges + negative.edges)
        test[name] = {
            **binary_prediction_metrics([1] * len(dataset.test_edges) + [0] * len(negative.edges), scores),
            **negative.metadata,
        }
    return test


def _edge_size_range(edges: list[Hyperedge]) -> tuple[int, int]:
    sizes = [len(edge) for edge in edges]
    return min(sizes), max(sizes)


def _limit_dataset_for_probe(dataset: HypergraphDataset, args: argparse.Namespace) -> HypergraphDataset:
    train_edges = _limit_edges(dataset.train_edges, int(args.max_train_edges))
    val_edges = _limit_edges(dataset.val_edges, int(args.max_val_edges))
    test_edges = _limit_edges(dataset.test_edges, int(args.max_test_edges))
    if train_edges is dataset.train_edges and val_edges is dataset.val_edges and test_edges is dataset.test_edges:
        return dataset
    limited = HypergraphDataset(
        num_nodes=dataset.num_nodes,
        train_edges=train_edges,
        val_edges=val_edges,
        test_edges=test_edges,
        node_features=dataset.node_features,
    )
    limited.validate()
    return limited


def _limit_edges(edges: list[Hyperedge], limit: int) -> list[Hyperedge]:
    if limit <= 0 or len(edges) <= limit:
        return edges
    return list(edges[:limit])


def _completed(dataset: str, method: str, mode: str, seed: int, args: argparse.Namespace) -> bool:
    root = _output_root(dataset, method, mode, seed, args)
    for path in root.glob("*/metrics.json"):
        try:
            metrics = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if (
            metrics.get("protocol_version") == PROTOCOL_VERSION
            and int(metrics.get("training_epochs", 0)) == int(args.epochs)
            and metrics.get("scans_config") == _scans_metadata(args)
            and metrics.get("probe_limits") == _probe_limits(args)
            and bool(metrics.get("test_evaluation_skipped", False)) == bool(args.skip_test)
        ):
            return True
    return False


def _output_root(dataset: str, method: str, mode: str, seed: int, args: argparse.Namespace) -> Path:
    parts = [
        f"ed{int(args.embedding_dim)}",
        f"hd{int(args.hidden_dim)}",
        f"do{_compact_float(args.dropout)}",
        f"lr{_compact_float(args.learning_rate)}",
        f"opt{str(args.predictor_optimizer)}",
        f"sp{str(args.split_strategy)}",
        f"glr{_compact_float(args.generator_learning_rate)}",
        f"wd{_compact_float(args.weight_decay)}",
        f"pbs{int(args.predictor_batch_size)}",
        f"mp{int(args.message_passing_layers)}",
        f"he{_compact_float(args.hnhn_alpha_e)}",
        f"hv{_compact_float(args.hnhn_alpha_v)}",
        f"pfj{int(_uses_discrete_retrieval(args.mode))}",
        f"pg{_compact_proposal_generator(args.proposal_generator)}",
        f"pgt{_compact_token(args.proposal_generator_training)}",
        f"ldr{int(bool(args.legacy_diffusion_rng_replay))}",
        f"s{int(args.diffusion_steps)}",
        f"m{int(args.teacher_candidate_multiplier)}",
        f"r{int(args.retrieval_candidate_multiplier)}",
        f"rsc{int(args.retrieval_sampling_source_chunk_size)}",
        f"gn{_compact_float(args.sampling_nearest_positive_upper_bound)}",
        f"gc{_compact_float(args.sampling_closure_risk_upper_bound)}",
        f"gcp{int(args.sampling_max_closure_pairs_per_edge)}",
        f"l{_compact_float(args.retrieval_lambda)}",
        f"ls{_compact_float(args.retrieval_lambda_start)}",
        f"lw{int(args.retrieval_lambda_warmup_epochs)}",
        f"gu{int(args.generator_updates_per_epoch)}",
        f"gb{int(args.generator_batch_size)}",
        f"sel{_compact_retrieval_selection(args.retrieval_selection_strategy)}",
        f"k{int(args.retrieval_top_k)}",
        f"t{_compact_float(args.retrieval_temperature)}",
        f"rn{_compact_retrieval_normalization(args.retrieval_normalization)}",
        f"n{int(args.negatives_per_positive)}",
        f"vp{_compact_protocols(args.validation_protocols)}",
        f"co{_compact_checkpoint_objective(args.checkpoint_objective)}",
        f"vw{hashlib.sha1(str(args.validation_protocol_weights).encode('utf-8')).hexdigest()[:6]}",
        f"det{int(bool(args.deterministic))}",
        f"p{_compact_pool(args.retrieval_candidate_pool)}",
        f"sw{hashlib.sha1(str(args.retrieval_pool_source_weights).encode('utf-8')).hexdigest()[:6]}",
        f"ss{hashlib.sha1(str(args.retrieval_pool_source_weights_start).encode('utf-8')).hexdigest()[:6]}",
        f"qw{int(args.retrieval_pool_source_weights_warmup_epochs)}",
        f"qt{str(args.retrieval_pool_quota_target)}",
        f"fs{hashlib.sha1(str(args.retrieval_selected_source_weights).encode('utf-8')).hexdigest()[:6]}",
        f"prw{hashlib.sha1(str(args.predictor_protocol_replay_weights).encode('utf-8')).hexdigest()[:6]}",
        f"prb{int(args.predictor_protocol_replay_banks)}",
        f"pru{int(args.predictor_protocol_replay_warmup_epochs)}",
        f"tr{int(args.max_train_edges)}",
        f"va{int(args.max_val_edges)}",
        f"te{int(args.max_test_edges)}",
    ]
    if str(args.proposal_generator) == "diffusion":
        parts.extend(
            [
                f"ds{_compact_diffusion_schedule(args.diffusion_schedule)}",
                f"ab{_compact_float(args.diffusion_min_alpha_bar)}",
                f"rc{_compact_float(args.residual_norm_clip)}",
                f"ema{_compact_float(args.generator_ema_decay)}",
            ]
        )
    elif str(args.proposal_generator) in {"mlp", "mlp_direct"}:
        parts.extend(
            [
                f"rc{_compact_float(args.residual_norm_clip)}",
                f"ema{_compact_float(args.generator_ema_decay)}",
            ]
        )
    elif str(args.proposal_generator) == "vae":
        parts.extend([f"vl{int(args.vae_latent_dim)}", f"vk{_compact_float(args.vae_kl_weight)}"])
    full_suffix = "_".join(parts)
    config_digest = hashlib.sha1(full_suffix.encode("ascii")).hexdigest()[:12]
    suffix = "_".join(
        [
            f"ed{int(args.embedding_dim)}",
            f"hd{int(args.hidden_dim)}",
            f"do{_compact_float(args.dropout)}",
            f"lr{_compact_float(args.learning_rate)}",
            f"opt{str(args.predictor_optimizer)}",
            f"pbs{int(args.predictor_batch_size)}",
            f"mp{int(args.message_passing_layers)}",
            f"he{_compact_float(args.hnhn_alpha_e)}",
            f"hv{_compact_float(args.hnhn_alpha_v)}",
            f"pg{_compact_proposal_generator(args.proposal_generator)}",
            f"cfg{config_digest}",
        ]
    )
    path_mode = _compact_mode_for_path(mode)
    return REPO_ROOT / "results" / f"scans_{dataset}_{method}_{path_mode}_{suffix}_seed{seed}"


def _output_dir(args: argparse.Namespace) -> Path:
    return _output_root(args.dataset, args.method, args.mode, args.seed, args) / time.strftime("%Y%m%d_%H%M%S")


def _probe_limits(args: argparse.Namespace) -> dict[str, int]:
    return {
        "max_train_edges": int(args.max_train_edges),
        "max_val_edges": int(args.max_val_edges),
        "max_test_edges": int(args.max_test_edges),
    }


def _scans_metadata(args: argparse.Namespace) -> dict[str, object]:
    retrieval_lambda = _bounded_retrieval_lambda(args.retrieval_lambda)
    retrieval_weights = _retrieval_weights(
        str(args.mode),
        {"retrieval_lambda": retrieval_lambda},
        effective_lambda=retrieval_lambda,
    )
    proposal_enabled = _uses_diffusion_proposal(str(args.mode))
    effective_pool = _effective_candidate_pool(str(args.mode), args.retrieval_candidate_pool)
    effective_selection_strategy, effective_top_k = _effective_retrieval_selection(
        str(args.mode),
        args.retrieval_selection_strategy,
        args.retrieval_top_k,
    )
    return {
        "ablation_semantics_version": (
            STRICT_ABLATION_SEMANTICS_VERSION if args.strict_paired_ablation else ABLATION_SEMANTICS_VERSION
        ),
        "max_epochs": int(args.epochs),
        "min_epochs": int(args.min_epochs),
        "early_stopping_patience": int(args.early_stopping_patience),
        "early_stopping_min_delta": float(args.early_stopping_min_delta),
        "validation_interval": int(args.validation_interval),
        "embedding_dim": int(args.embedding_dim),
        "hidden_dim": int(args.hidden_dim),
        "dropout": float(args.dropout),
        "message_passing_layers": int(args.message_passing_layers),
        "predictor_batch_size": int(args.predictor_batch_size),
        "hnhn_alpha_e": float(args.hnhn_alpha_e),
        "hnhn_alpha_v": float(args.hnhn_alpha_v),
        "learning_rate": float(args.learning_rate),
        "predictor_optimizer": str(args.predictor_optimizer),
        "predictor_init_checkpoint": str(args.predictor_init_checkpoint),
        "predictor_replay_negative_bank": str(args.predictor_replay_negative_bank),
        "predictor_replay_ratio": float(args.predictor_replay_ratio),
        "predictor_protocol_replay_weights": str(args.predictor_protocol_replay_weights),
        "predictor_protocol_replay_banks": int(args.predictor_protocol_replay_banks),
        "predictor_protocol_replay_warmup_epochs": int(args.predictor_protocol_replay_warmup_epochs),
        "predictor_protocol_replay_loss_mode": str(args.predictor_protocol_replay_loss_mode),
        "predictor_protocol_replay_loss_normalization": (
            "fixed_total_negative_mass_v1"
            if str(args.predictor_protocol_replay_loss_mode) == "fixed_total"
            else "full_scans_plus_weighted_replay_v1"
        ),
        "validation_negative_seed_scheme": "experiment_seed_plus_stable_protocol_offset_v1",
        "split_strategy": str(args.split_strategy),
        "generator_learning_rate": float(args.generator_learning_rate),
        "weight_decay": float(args.weight_decay),
        "proposal_generator": str(args.proposal_generator),
        "proposal_generator_training": str(args.proposal_generator_training),
        "diffusion_proposal_enabled": proposal_enabled,
        "proposal_uses_positive_condition": bool(
            proposal_enabled and str(args.proposal_generator) != "vae"
        ),
        "vae_conditioning": "none" if str(args.proposal_generator) == "vae" else "not_applicable",
        "vae_latent_dim": int(args.vae_latent_dim),
        "vae_kl_weight": float(args.vae_kl_weight),
        "diffusion_steps": int(args.diffusion_steps),
        "diffusion_schedule": str(args.diffusion_schedule),
        "diffusion_min_alpha_bar": float(args.diffusion_min_alpha_bar),
        "diffusion_cosine_s": float(args.diffusion_cosine_s),
        "diffusion_beta_start": float(args.diffusion_beta_start),
        "diffusion_beta_end": float(args.diffusion_beta_end),
        "residual_norm_clip": float(args.residual_norm_clip),
        "generator_updates_per_epoch": int(args.generator_updates_per_epoch),
        "generator_batch_size": int(args.generator_batch_size),
        "generator_grad_clip": float(args.generator_grad_clip),
        "risk_bank_size": int(args.risk_bank_size),
        "generator_ema_decay": float(args.generator_ema_decay),
        "generator_ema_enabled": bool(proposal_enabled and float(args.generator_ema_decay) > 0),
        "generator_ema_start_step": int(args.generator_ema_start_step),
        "strict_paired_ablation": bool(args.strict_paired_ablation),
        "legacy_diffusion_rng_replay": bool(args.legacy_diffusion_rng_replay),
        "boundary_target": float(args.boundary_target),
        "risk_margin": float(args.risk_margin),
        "boundary_weight": float(args.boundary_weight),
        "risk_weight": float(args.risk_weight),
        "diversity_weight": float(args.diversity_weight),
        "diffusion_weight": float(args.diffusion_weight),
        "teacher_candidate_multiplier": int(args.teacher_candidate_multiplier),
        "retrieval_candidate_multiplier": int(args.retrieval_candidate_multiplier),
        "retrieval_sampling_source_chunk_size": int(args.retrieval_sampling_source_chunk_size),
        "partitioned_candidate_pool_for_scans": bool(
            _uses_discrete_retrieval(args.mode)
            and (proposal_enabled or (bool(args.strict_paired_ablation) and args.mode == "scans_no_diffusion_proposal"))
        ),
        "effective_teacher_candidate_multiplier": (
            int(args.teacher_candidate_multiplier)
            if proposal_enabled or (bool(args.strict_paired_ablation) and args.mode == "scans_no_diffusion_proposal")
            else 0
        ),
        "effective_retrieval_candidate_multiplier": int(args.retrieval_candidate_multiplier),
        "effective_total_candidate_multiplier": (
            (
                int(args.teacher_candidate_multiplier)
                if proposal_enabled or (bool(args.strict_paired_ablation) and args.mode == "scans_no_diffusion_proposal")
                else 0
            )
            + int(args.retrieval_candidate_multiplier)
            if _uses_discrete_retrieval(args.mode)
            else int(args.teacher_candidate_multiplier)
        ),
        "retrieval_lambda_requested": retrieval_lambda,
        "retrieval_lambda": retrieval_weights["lambda"],
        "retrieval_reference_lambda": retrieval_weights["reference_lambda"],
        "retrieval_lambda_target": retrieval_lambda,
        "retrieval_lambda_start": float(args.retrieval_lambda_start),
        "retrieval_lambda_warmup_epochs": int(args.retrieval_lambda_warmup_epochs),
        "retrieval_diffusion_weight": retrieval_weights["diffusion"],
        "retrieval_boundary_weight": 0.0,
        "retrieval_hardness_weight": retrieval_weights["hardness"],
        "learned_diffusion_signal_enabled": str(args.mode) not in {
            "scans_no_diffusion_signal",
            "scans_random_proposal_control",
        },
        "random_proposal_control_enabled": str(args.mode) == "scans_random_proposal_control",
        "retrieval_selection_strategy_requested": str(args.retrieval_selection_strategy),
        "retrieval_selection_strategy": effective_selection_strategy,
        "retrieval_top_k_requested": int(args.retrieval_top_k),
        "retrieval_top_k": effective_top_k,
        "retrieval_temperature": float(args.retrieval_temperature),
        "retrieval_normalization": str(args.retrieval_normalization),
        "negatives_per_positive": int(args.negatives_per_positive),
        "training_objective": (
            "nhp_pairwise_ranking"
            if str(args.method) == "nhp"
            else "binary_cross_entropy"
        ),
        "predictor_forward_batching": (
            "joint_positive_negative"
            if _uses_discrete_retrieval(args.mode)
            else "separate_or_generated"
        ),
        "validation_protocols": str(args.validation_protocols),
        "checkpoint_objective": str(args.checkpoint_objective),
        "validation_protocol_weights": str(args.validation_protocol_weights),
        "validation_auc_weight": float(args.validation_auc_weight),
        "validation_ap_weight": float(args.validation_ap_weight),
        "deterministic": bool(args.deterministic),
        "retrieval_candidate_pool_requested": str(args.retrieval_candidate_pool),
        "retrieval_candidate_pool": effective_pool,
        "retrieval_pool_source_weights": str(args.retrieval_pool_source_weights),
        "retrieval_pool_source_weights_start": str(args.retrieval_pool_source_weights_start),
        "retrieval_pool_source_weights_warmup_epochs": int(args.retrieval_pool_source_weights_warmup_epochs),
        "retrieval_pool_quota_target": str(args.retrieval_pool_quota_target),
        "retrieval_selected_source_weights": str(args.retrieval_selected_source_weights),
        "teacher_mode": str(args.teacher_mode),
        "sampling_max_attempts": int(args.sampling_max_attempts),
        "sampling_replacement_strategy": str(args.sampling_replacement_strategy),
        "sampling_anchor_ratio": float(args.sampling_anchor_ratio),
        "sampling_nearest_positive_upper_bound": float(args.sampling_nearest_positive_upper_bound),
        "sampling_closure_risk_upper_bound": float(args.sampling_closure_risk_upper_bound),
        "sampling_max_closure_pairs_per_edge": int(args.sampling_max_closure_pairs_per_edge),
        "sampling_rerank_top_k": int(args.sampling_rerank_top_k),
        "sampling_rerank_selection_strategy": str(args.sampling_rerank_selection_strategy),
        "sampling_rerank_score_lower_bound": float(args.sampling_rerank_score_lower_bound),
        "sampling_rerank_score_upper_bound": float(args.sampling_rerank_score_upper_bound),
        "risk_aware_pool_multiplier": int(args.risk_aware_pool_multiplier),
        "risk_aware_nearest_weight": float(args.risk_aware_nearest_weight),
        "residual_safe_pool_multiplier": int(args.residual_safe_pool_multiplier),
        "residual_safe_structural_quantile": float(args.residual_safe_structural_quantile),
        "residual_safe_residual_weight": float(args.residual_safe_residual_weight),
    }


def _run_parallel(jobs: list[tuple[str, str, str, int]], args: argparse.Namespace) -> None:
    log_dir = REPO_ROOT / "remote_logs" / "scans"
    log_dir.mkdir(parents=True, exist_ok=True)
    pending = list(jobs)
    running: list[tuple[tuple[str, str, str, int], subprocess.Popen[bytes], object]] = []
    failures = []
    devices = [device for device in args.devices.split(",") if device]
    launched = 0
    while pending or running:
        while pending and len(running) < args.max_parallel:
            dataset, method, mode, seed = pending.pop(0)
            worker_device = devices[launched % len(devices)] if devices else args.device
            launched += 1
            log = (log_dir / f"{dataset}_{method}_{mode}_seed{seed}.log").open("wb")
            command = [
                sys.executable,
                "scripts/run_scans_benchmark.py",
                "--worker",
                "--dataset", dataset,
                "--method", method,
                "--mode", mode,
                "--seed", str(seed),
                "--epochs", str(args.epochs),
                "--min-epochs", str(args.min_epochs),
                "--early-stopping-patience", str(args.early_stopping_patience),
                "--early-stopping-min-delta", str(args.early_stopping_min_delta),
                "--validation-interval", str(args.validation_interval),
                "--embedding-dim", str(args.embedding_dim),
                "--hidden-dim", str(args.hidden_dim),
                "--dropout", str(args.dropout),
                "--message-passing-layers", str(args.message_passing_layers),
                "--predictor-batch-size", str(args.predictor_batch_size),
                "--hnhn-alpha-e", str(args.hnhn_alpha_e),
                "--hnhn-alpha-v", str(args.hnhn_alpha_v),
                "--learning-rate", str(args.learning_rate),
                "--predictor-optimizer", str(args.predictor_optimizer),
                "--predictor-init-checkpoint", str(args.predictor_init_checkpoint),
                "--predictor-replay-negative-bank", str(args.predictor_replay_negative_bank),
                "--predictor-replay-ratio", str(args.predictor_replay_ratio),
                "--predictor-protocol-replay-weights", str(args.predictor_protocol_replay_weights),
                "--predictor-protocol-replay-banks", str(args.predictor_protocol_replay_banks),
                "--predictor-protocol-replay-warmup-epochs", str(args.predictor_protocol_replay_warmup_epochs),
                "--predictor-protocol-replay-loss-mode", str(args.predictor_protocol_replay_loss_mode),
                "--split-strategy", str(args.split_strategy),
                "--generator-learning-rate", str(args.generator_learning_rate),
                "--weight-decay", str(args.weight_decay),
                "--proposal-generator", str(args.proposal_generator),
                "--proposal-generator-training", str(args.proposal_generator_training),
                "--vae-latent-dim", str(args.vae_latent_dim),
                "--vae-kl-weight", str(args.vae_kl_weight),
                "--diffusion-steps", str(args.diffusion_steps),
                "--diffusion-schedule", str(args.diffusion_schedule),
                "--diffusion-min-alpha-bar", str(args.diffusion_min_alpha_bar),
                "--diffusion-cosine-s", str(args.diffusion_cosine_s),
                "--diffusion-beta-start", str(args.diffusion_beta_start),
                "--diffusion-beta-end", str(args.diffusion_beta_end),
                "--residual-norm-clip", str(args.residual_norm_clip),
                "--generator-updates-per-epoch", str(args.generator_updates_per_epoch),
                "--generator-batch-size", str(args.generator_batch_size),
                "--generator-grad-clip", str(args.generator_grad_clip),
                "--risk-bank-size", str(args.risk_bank_size),
                "--generator-ema-decay", str(args.generator_ema_decay),
                "--generator-ema-start-step", str(args.generator_ema_start_step),
                "--boundary-target", str(args.boundary_target),
                "--risk-margin", str(args.risk_margin),
                "--boundary-weight", str(args.boundary_weight),
                "--risk-weight", str(args.risk_weight),
                "--diversity-weight", str(args.diversity_weight),
                "--diffusion-weight", str(args.diffusion_weight),
                "--teacher-candidate-multiplier", str(args.teacher_candidate_multiplier),
                "--retrieval-candidate-multiplier", str(args.retrieval_candidate_multiplier),
                "--retrieval-sampling-source-chunk-size", str(args.retrieval_sampling_source_chunk_size),
                "--retrieval-lambda", str(args.retrieval_lambda),
                "--retrieval-lambda-warmup-epochs", str(args.retrieval_lambda_warmup_epochs),
                "--retrieval-lambda-start", str(args.retrieval_lambda_start),
                "--retrieval-selection-strategy", str(args.retrieval_selection_strategy),
                "--retrieval-top-k", str(args.retrieval_top_k),
                "--retrieval-temperature", str(args.retrieval_temperature),
                "--retrieval-normalization", str(args.retrieval_normalization),
                "--negatives-per-positive", str(args.negatives_per_positive),
                "--validation-protocols", str(args.validation_protocols),
                "--validation-negative-bank", str(args.validation_negative_bank),
                "--checkpoint-objective", str(args.checkpoint_objective),
                "--validation-protocol-weights", str(args.validation_protocol_weights),
                "--validation-auc-weight", str(args.validation_auc_weight),
                "--validation-ap-weight", str(args.validation_ap_weight),
                "--retrieval-candidate-pool", str(args.retrieval_candidate_pool),
                "--retrieval-pool-source-weights", str(args.retrieval_pool_source_weights),
                "--retrieval-pool-source-weights-start", str(args.retrieval_pool_source_weights_start),
                "--retrieval-pool-source-weights-warmup-epochs", str(args.retrieval_pool_source_weights_warmup_epochs),
                "--retrieval-pool-quota-target", str(args.retrieval_pool_quota_target),
                "--retrieval-selected-source-weights", str(args.retrieval_selected_source_weights),
                "--teacher-mode", str(args.teacher_mode),
                "--sampling-max-attempts", str(args.sampling_max_attempts),
                "--sampling-replacement-strategy", str(args.sampling_replacement_strategy),
                "--sampling-anchor-ratio", str(args.sampling_anchor_ratio),
                "--sampling-nearest-positive-upper-bound", str(args.sampling_nearest_positive_upper_bound),
                "--sampling-closure-risk-upper-bound", str(args.sampling_closure_risk_upper_bound),
                "--sampling-max-closure-pairs-per-edge", str(args.sampling_max_closure_pairs_per_edge),
                "--sampling-rerank-top-k", str(args.sampling_rerank_top_k),
                "--sampling-rerank-selection-strategy", str(args.sampling_rerank_selection_strategy),
                "--sampling-rerank-score-lower-bound", str(args.sampling_rerank_score_lower_bound),
                "--sampling-rerank-score-upper-bound", str(args.sampling_rerank_score_upper_bound),
                "--risk-aware-pool-multiplier", str(args.risk_aware_pool_multiplier),
                "--risk-aware-nearest-weight", str(args.risk_aware_nearest_weight),
                "--residual-safe-pool-multiplier", str(args.residual_safe_pool_multiplier),
                "--residual-safe-structural-quantile", str(args.residual_safe_structural_quantile),
                "--residual-safe-residual-weight", str(args.residual_safe_residual_weight),
                "--max-train-edges", str(args.max_train_edges),
                "--max-val-edges", str(args.max_val_edges),
                "--max-test-edges", str(args.max_test_edges),
                "--torch-threads", str(args.torch_threads),
                "--device", worker_device,
                "--deterministic" if args.deterministic else "--no-deterministic",
            ]
            if args.strict_paired_ablation:
                command.append("--strict-paired-ablation")
            if args.legacy_diffusion_rng_replay:
                command.append("--legacy-diffusion-rng-replay")
            environment = os.environ.copy()
            environment["PYTHONUNBUFFERED"] = "1"
            if args.deterministic:
                environment["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
                environment["PYTHONHASHSEED"] = str(seed)
            for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
                environment[key] = str(args.torch_threads)
            process = subprocess.Popen(command, cwd=REPO_ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT)
            running.append(((dataset, method, mode, seed), process, log))
            print(f"started {dataset} {method} {mode} seed={seed} device={worker_device} pid={process.pid}", flush=True)
        active = []
        for job, process, log in running:
            code = process.poll()
            if code is None:
                active.append((job, process, log))
            else:
                log.close()
                print(f"finished {job} rc={code}", flush=True)
                if code:
                    failures.append((job, int(code)))
        running = active
        if pending or running:
            time.sleep(5)
    if failures:
        raise RuntimeError(f"SCANS benchmark failures: {failures}")


def _average_epoch_summaries(summaries: list[dict[str, float]]) -> dict[str, float]:
    if not summaries:
        return {}
    keys = sorted(set().union(*(summary.keys() for summary in summaries)))
    return {f"scans_{key}": float(sum(summary.get(key, 0.0) for summary in summaries) / len(summaries)) for key in keys}


def _build_alpha_bar_schedule(
    *,
    steps: int,
    schedule: str,
    min_alpha_bar: float,
    cosine_s: float,
    beta_start: float,
    beta_end: float,
) -> torch.Tensor:
    steps = int(steps)
    if steps < 2:
        raise ValueError("diffusion_steps must be >= 2")
    min_alpha_bar = float(min_alpha_bar)
    if not 0.0 < min_alpha_bar <= 1.0:
        raise ValueError("diffusion_min_alpha_bar must be in (0, 1]")
    if schedule == "cosine":
        t = torch.arange(0, steps + 1, dtype=torch.float32)
        s = float(cosine_s)
        angles = ((t / float(steps)) + s) / (1.0 + s) * (math.pi / 2.0)
        values = torch.cos(angles).square()
        alpha_bar = values / values[0].clamp_min(1e-12)
        alpha_bar = alpha_bar.clamp(min=min_alpha_bar, max=1.0)
        alpha_bar[0] = 1.0
        alpha_bar[-1] = min_alpha_bar
        return alpha_bar
    if schedule == "linear_beta":
        betas = torch.linspace(float(beta_start), float(beta_end), steps, dtype=torch.float32).clamp(1e-8, 0.999)
        alpha_bar = torch.empty(steps + 1, dtype=torch.float32)
        alpha_bar[0] = 1.0
        alpha_bar[1:] = torch.cumprod(1.0 - betas, dim=0)
        alpha_bar = alpha_bar.clamp(min=min_alpha_bar, max=1.0)
        alpha_bar[0] = 1.0
        alpha_bar[-1] = min(float(alpha_bar[-1].item()), min_alpha_bar)
        return alpha_bar
    raise ValueError(f"unknown diffusion schedule: {schedule}")


def _run_self_check() -> None:
    _configure_reproducibility(123, deterministic=True)
    assert torch.are_deterministic_algorithms_enabled()
    alpha_bar = _build_alpha_bar_schedule(
        steps=4,
        schedule="cosine",
        min_alpha_bar=1e-4,
        cosine_s=0.008,
        beta_start=0.0001,
        beta_end=0.02,
    )
    assert alpha_bar.numel() == 5
    assert torch.isclose(alpha_bar[0], torch.tensor(1.0), atol=1e-7)
    assert torch.all(alpha_bar[:-1] >= alpha_bar[1:] - 1e-8)
    assert abs(float(alpha_bar[-1].item()) - 1e-4) < 1e-6

    dim, hidden, batch = 8, 16, 5
    generator = ConditionalResidualDDIMGenerator(
        representation_dim=dim,
        hidden_dim=hidden,
        steps=4,
        schedule="cosine",
        min_alpha_bar=1e-4,
        cosine_s=0.008,
        beta_start=0.0001,
        beta_end=0.02,
        residual_norm_clip=2.0,
    )
    condition = torch.randn(batch, dim)
    teacher = torch.randn(batch, dim)
    loss, metrics = generator.training_loss(teacher, condition)
    assert torch.isfinite(loss)
    loss.backward()
    grad_norm = _gradient_norm(generator.parameters())
    assert grad_norm > 0 and math.isfinite(grad_norm)
    assert "diffusion_denoise_mse" in metrics
    generator.zero_grad(set_to_none=True)

    seed_a = _torch_generator_for_device(torch.device("cpu"), 77)
    seed_b = _torch_generator_for_device(torch.device("cpu"), 77)
    proposal_a = generator.sample(condition, generator=seed_a)
    proposal_b = generator.sample(condition, generator=seed_b)
    assert isinstance(proposal_a, torch.Tensor) and isinstance(proposal_b, torch.Tensor)
    assert torch.allclose(proposal_a, proposal_b)
    proposal_c = generator.sample(condition, generator=_torch_generator_for_device(torch.device("cpu"), 78))
    assert isinstance(proposal_c, torch.Tensor)
    assert not torch.allclose(proposal_a, proposal_c)
    assert torch.allclose(proposal_a.norm(dim=1), torch.ones(batch), atol=1e-4, rtol=1e-4)
    clipped = generator._clip_residual_norm(torch.ones(batch, dim) * 10.0)  # noqa: SLF001
    assert float(clipped.norm(dim=1).max().item()) <= 2.0 + 1e-5

    predictor = nn.Linear(dim, dim)
    inputs = torch.randn(batch, dim)
    condition_from_predictor = predictor(inputs)
    teacher_from_predictor = predictor(inputs + 0.1)
    detach_loss, _ = generator.training_loss(teacher_from_predictor.detach(), condition_from_predictor.detach())
    detach_loss.backward()
    assert predictor.weight.grad is None

    mlp = ResidualMLPProposalGenerator(
        representation_dim=dim,
        hidden_dim=hidden,
        residual_norm_clip=2.0,
    )
    mlp_loss, mlp_metrics = mlp.training_loss(teacher, condition)
    assert torch.isfinite(mlp_loss) and "mlp_residual_mse" in mlp_metrics
    mlp_loss.backward()
    assert _gradient_norm(mlp.parameters()) > 0
    mlp_proposal = mlp.sample(condition)
    assert isinstance(mlp_proposal, torch.Tensor)
    assert torch.allclose(mlp_proposal.norm(dim=1), torch.ones(batch), atol=1e-4, rtol=1e-4)

    direct_mlp = DirectMLPProposalGenerator(representation_dim=dim, hidden_dim=hidden)
    direct_mlp_loss, direct_mlp_metrics = direct_mlp.training_loss(teacher, condition)
    assert torch.isfinite(direct_mlp_loss) and "mlp_direct_mse" in direct_mlp_metrics
    direct_mlp_loss.backward()
    assert _gradient_norm(direct_mlp.parameters()) > 0
    direct_mlp_proposal = direct_mlp.sample(condition)
    assert isinstance(direct_mlp_proposal, torch.Tensor)
    assert torch.allclose(direct_mlp_proposal.norm(dim=1), torch.ones(batch), atol=1e-4, rtol=1e-4)

    native_mlp = DirectMLPProposalGenerator(representation_dim=dim, hidden_dim=hidden)
    native_optimizer = torch.optim.AdamW(native_mlp.parameters(), lr=1e-3)
    native_metrics = _run_generator_updates(
        generator=native_mlp,
        generator_optimizer=native_optimizer,
        generator_ema=None,
        config={
            "scans": {
                "proposal_generator": "mlp_direct",
                "proposal_generator_training": "native",
                "generator_updates_per_epoch": 1,
                "generator_batch_size": batch,
                "generator_grad_clip": 5.0,
                "generator_ema_decay": 0.0,
                "generator_ema_start_step": 0,
                "diffusion_weight": 0.1,
                "risk_weight": 0.2,
                "diversity_weight": 0.02,
                "risk_margin": 0.2,
            }
        },
        mode="scans_full",
        teacher_representations=teacher,
        teacher_conditions=condition,
        positive_bank=teacher,
        rng=random.Random(314),
    )
    assert native_metrics["generator_native_objective_only"] == 1.0
    assert native_metrics["generator_native_loss_weight"] == 1.0
    assert native_metrics["generator_risk_loss"] == 0.0
    assert abs(native_metrics["generator_loss"] - native_metrics["mlp_direct_mse"]) < 1e-7

    vae = UnconditionalVAEProposalGenerator(
        representation_dim=dim,
        hidden_dim=hidden,
        latent_dim=4,
        kl_weight=0.01,
    )
    vae_loss_a, _ = vae.training_loss(
        teacher,
        condition,
        generator=_torch_generator_for_device(torch.device("cpu"), 901),
    )
    vae_loss_b, _ = vae.training_loss(
        teacher,
        condition + 100.0,
        generator=_torch_generator_for_device(torch.device("cpu"), 901),
    )
    assert torch.allclose(vae_loss_a, vae_loss_b)
    vae_proposal_a = vae.sample(
        condition,
        generator=_torch_generator_for_device(torch.device("cpu"), 902),
    )
    vae_proposal_b = vae.sample(
        condition - 100.0,
        generator=_torch_generator_for_device(torch.device("cpu"), 902),
    )
    assert isinstance(vae_proposal_a, torch.Tensor) and isinstance(vae_proposal_b, torch.Tensor)
    assert torch.allclose(vae_proposal_a, vae_proposal_b)
    assert torch.allclose(vae_proposal_a.norm(dim=1), torch.ones(batch), atol=1e-4, rtol=1e-4)
    diffusion = _build_proposal_generator(
        representation_dim=dim,
        hidden_dim=hidden,
        config={"proposal_generator": "diffusion", "diffusion_steps": 4},
    )
    assert isinstance(diffusion, ConditionalResidualDDIMGenerator)
    torch.manual_seed(1901)
    diffusion.training_loss(teacher, condition)
    diffusion.sample(condition)
    legacy_diffusion_rng_state = torch.random.get_rng_state().clone()
    torch.manual_seed(1901)
    _consume_legacy_diffusion_training_rng(teacher, condition, diffusion_steps=4)
    replayed_rng_state = torch.random.get_rng_state().clone()
    assert torch.equal(legacy_diffusion_rng_state, replayed_rng_state)
    _ = _build_proposal_generator(
        representation_dim=dim,
        hidden_dim=hidden,
        config={"proposal_generator": "mlp", "residual_norm_clip": 2.0},
    )
    direct_mlp_built = _build_proposal_generator(
        representation_dim=dim,
        hidden_dim=hidden,
        config={"proposal_generator": "mlp_direct"},
    )
    assert isinstance(direct_mlp_built, DirectMLPProposalGenerator)
    _ = _build_proposal_generator(
        representation_dim=dim,
        hidden_dim=hidden,
        config={"proposal_generator": "vae", "vae_latent_dim": 4, "vae_kl_weight": 0.01},
    )

    assert _effective_retrieval_lambda("scans_no_diffusion", 0.8, 0.0, 5, 3)[0] == 0.0
    assert _effective_retrieval_lambda("scans_no_diffusion_proposal", 0.8, 0.0, 5, 3)[0] == 0.0
    assert abs(_effective_retrieval_lambda("scans_no_diffusion_signal", 0.8, 0.0, 5, 5)[0] - 0.8) < 1e-12
    assert _effective_retrieval_lambda("scans_diffusion_only", 0.8, 0.0, 5, 3)[0] == 1.0
    assert _effective_retrieval_lambda("scans_full", 0.8, 0.0, 5, 0)[0] == 0.0
    assert abs(_effective_retrieval_lambda("scans_full", 0.8, 0.0, 5, 5)[0] - 0.8) < 1e-12
    assert _configured_retrieval_lambda_target(
        {"retrieval_lambda": 0.0, "retrieval_lambda_target": 0.8}
    ) == 0.8
    assert not _uses_diffusion_proposal("scans_no_diffusion_proposal")
    assert _uses_discrete_retrieval("scans_no_diffusion_proposal")
    assert not _uses_discrete_retrieval("scans_no_discrete_retrieval")
    assert not _uses_generator_risk("scans_no_risk_control")
    assert _uses_generator_risk("scans_full")
    assert _effective_candidate_pool("scans_no_risk_control", "risk_controlled") == "uncontrolled_union"
    assert _effective_retrieval_selection("scans_no_topk_soft", "topk_sample", 5) == ("top_score", 1)
    no_signal_weights = _retrieval_weights(
        "scans_no_diffusion_signal",
        {"retrieval_lambda": 0.8},
        effective_lambda=0.8,
    )
    assert no_signal_weights["diffusion"] == 0.0
    assert abs(no_signal_weights["hardness"] - 0.2) < 1e-12
    assert no_signal_weights["reference_lambda"] == 0.8
    strict_config = {"scans": {"strict_paired_ablation": True}}
    legacy_config = {"scans": {"strict_paired_ablation": False}}
    assert _uses_partitioned_candidate_pool("scans_no_diffusion_proposal", strict_config)
    assert not _uses_partitioned_candidate_pool("scans_no_diffusion_proposal", legacy_config)
    assert _named_training_seed(7, 3, "candidate_pool") == _named_training_seed(7, 3, "candidate_pool")
    assert _named_training_seed(7, 3, "candidate_pool") != _named_training_seed(7, 3, "retrieval")

    online = nn.Linear(dim, dim)
    ema = copy.deepcopy(online)
    with torch.no_grad():
        online.weight.add_(1.0)
    _update_generator_ema_after_step(ema, online, 0.999, completed_updates=3, ema_start_step=4)
    assert not torch.allclose(ema.weight, online.weight)
    _update_generator_ema_after_step(ema, online, 0.999, completed_updates=4, ema_start_step=4)
    assert torch.allclose(ema.weight, online.weight)
    assert _proposal_generator_for_sampling(online, ema, completed_updates=3, ema_start_step=4) is online
    assert _proposal_generator_for_sampling(online, ema, completed_updates=4, ema_start_step=4) is ema

    source = (1, 2, 3)
    candidate_batch = NegativeSampleBatch(
        edges=[(1, 4, 5), (2, 5, 6), (3, 6, 7)],
        source_edges=[source, source, source],
        metadata={},
    )
    teacher_pool, retrieval_pool = _split_teacher_retrieval_candidate_batches(
        candidate_batch=candidate_batch,
        source_edges=[source],
        block_size=3,
        teacher_block_size=1,
        retrieval_block_size=2,
    )
    assert not (set(teacher_pool.edges) & set(retrieval_pool.edges))

    quota_labels = ("risk_controlled", "sns", "mns", "cns", "mix")
    quota_candidates = [
        (label, (label_index + 1, item_index + 100, item_index + 1000))
        for label_index, label in enumerate(quota_labels)
        for item_index in range(32)
    ]
    quota_partitions, quota_padding = _select_partitioned_source_weighted_candidates(
        candidate_items=quota_candidates,
        hardness_scores=None,
        partition_sizes=(15, 15),
        source_weights={"risk_controlled": 0.5, "sns": 0.0, "mns": 1.0, "cns": 4.0, "mix": 2.0},
        rng=random.Random(123),
    )
    expected_quota_counts = {"risk_controlled": 1, "sns": 0, "mns": 2, "cns": 8, "mix": 4}
    for partition in quota_partitions:
        actual_quota_counts = {
            label: sum(1 for candidate_label, _candidate in partition if candidate_label == label)
            for label in quota_labels
        }
        assert actual_quota_counts == expected_quota_counts
    assert quota_padding == 0
    assert not (
        {candidate for _label, candidate in quota_partitions[0]}
        & {candidate for _label, candidate in quota_partitions[1]}
    )

    warmed_weights = _effective_pool_source_weights(
        {
            "retrieval_pool_source_weights": "risk_controlled:6,sns:2,mns:1,cns:1,mix:1",
            "retrieval_pool_source_weights_start": "risk_controlled:10,sns:2,mns:1,cns:0,mix:0",
            "retrieval_pool_source_weights_warmup_epochs": 10,
        },
        epoch=5,
    )
    assert warmed_weights == {"cns": 0.5, "mix": 0.5, "mns": 1.0, "risk_controlled": 8.0, "sns": 2.0}

    disjoint_teacher = NegativeSampleBatch(
        edges=[(1, 4, 5), (1, 6, 7)],
        source_edges=[source, source],
        candidate_labels=["risk_controlled", "risk_controlled"],
    )
    disjoint_retrieval = NegativeSampleBatch(
        edges=[(1, 4, 5), (2, 4, 6), (3, 4, 7)],
        source_edges=[source, source, source],
        candidate_labels=["cns", "mix", "mns"],
    )
    filtered_retrieval = _exclude_teacher_candidates_from_retrieval(
        teacher_batch=disjoint_teacher,
        retrieval_batch=disjoint_retrieval,
        source_edges=[source],
        teacher_block_size=2,
        retrieval_block_size=3,
    )
    assert not (set(disjoint_teacher.edges) & set(filtered_retrieval.edges))
    assert len(filtered_retrieval.edges) == 3
    assert filtered_retrieval.metadata["candidate_pool_quota_target_retrieval_only_enabled"] == 1.0

    controlled_counts: dict[str, int] = {}
    controlled_labels = ["risk_controlled", "sns", "mns", "cns", "mix"]
    for _ in range(100):
        controlled_selection = _select_retrieval_indices_with_source_mix(
            order=[0, 1, 2, 3, 4],
            scores=[0.1, 0.2, 0.3, 0.4, 0.5],
            candidate_labels=controlled_labels,
            negatives_per_positive=1,
            strategy="top_score",
            top_k=1,
            temperature=1.0,
            source_weights={"risk_controlled": 90, "sns": 4, "mns": 3, "cns": 2, "mix": 1},
            source_counts=controlled_counts,
            rng=random.Random(123),
        )
        assert len(controlled_selection) == 1
    assert sum(controlled_counts.values()) == 100
    assert controlled_counts["risk_controlled"] >= 89
    assert controlled_counts["cns"] + controlled_counts["mix"] <= 4

    replay_sources = [(0, 1), (2, 3)]
    replay_batch = NegativeSampleBatch(
        edges=[(0, 4), (2, 5)],
        source_edges=replay_sources,
        metadata={"existing": 1.0},
        candidate_labels=["risk_controlled", "risk_controlled"],
    )
    replay_mixed = _mix_replay_negatives(
        replay_batch,
        [(0, 6), (2, 7)],
        source_edges=replay_sources,
        replay_ratio=1.0,
        rng=random.Random(123),
    )
    assert replay_mixed.edges == [(0, 6), (2, 7)]
    assert replay_mixed.candidate_labels == ["ahp_replay", "ahp_replay"]
    assert replay_mixed.metadata["ahp_replay_actual_ratio"] == 1.0
    replay_weights, replay_progress = _effective_protocol_replay_weights(
        {"mns": 0.2, "cns": 0.4},
        epoch=1,
        warmup_epochs=4,
    )
    assert replay_progress == 0.5
    assert replay_weights == {"mns": 0.1, "cns": 0.2}
    full_replay_weights, full_replay_progress = _effective_protocol_replay_weights(
        {"mns": 0.2, "cns": 0.4},
        epoch=9,
        warmup_epochs=4,
    )
    assert full_replay_progress == 1.0
    assert full_replay_weights == {"mns": 0.2, "cns": 0.4}
    print("self-check passed", flush=True)


def _jsonable(config: Mapping[str, object]) -> dict[str, object]:
    return copy.deepcopy(dict(config))


def _compact_float(value: float) -> str:
    return f"{float(value):g}".replace("-", "m").replace(".", "p")


def _compact_token(value: object) -> str:
    text = str(value)
    return "".join(character if character.isalnum() else "_" for character in text).strip("_") or "x"


def _compact_objective(value: object) -> str:
    mapping = {
        "embedding_alignment": "ea",
        "gradient_alignment": "ga",
    }
    text = str(value)
    return mapping.get(text, _compact_token(text)[:8])


def _compact_mode_for_path(value: object) -> str:
    mapping = {
        "scans_no_teacher_target": "scans_ntgt",
        "scans_no_diffusion_proposal": "scans_nodp",
        "scans_no_diffusion_signal": "scans_nodsig",
        "scans_random_proposal_control": "scans_rndprop",
        "scans_no_risk_control": "scans_norisk",
        "scans_no_discrete_retrieval": "scans_nodisc",
        "scans_no_topk_soft": "scans_notopk",
    }
    text = str(value)
    return mapping.get(text, text)


def _compact_proposal_generator(value: object) -> str:
    mapping = {
        "diffusion": "diff",
        "mlp": "mlp",
        "mlp_direct": "mlpd",
        "vae": "vae",
    }
    text = str(value)
    return mapping.get(text, _compact_token(text)[:4])


def _compact_diffusion_schedule(value: object) -> str:
    mapping = {
        "cosine": "cos",
        "linear_beta": "lb",
    }
    text = str(value)
    return mapping.get(text, _compact_token(text)[:4])


def _compact_protocols(value: object) -> str:
    mapping = {"sns": "s", "mns": "m", "cns": "c", "mix": "x"}
    parts = [part.strip() for part in str(value).split(",") if part.strip()]
    return "".join(mapping.get(part, _compact_token(part)[:1]) for part in parts) or "s"


def _compact_pool(value: object) -> str:
    mapping = {
        "risk_controlled": "rc",
        "risk_sns": "rs",
        "union": "u",
        "union_hard": "uh",
        "uncontrolled_union": "uu",
    }
    text = str(value)
    return mapping.get(text, _compact_token(text)[:4])


def _compact_checkpoint_objective(value: object) -> str:
    mapping = {
        "mean_auc_ap": "maa",
        "mean_auc": "mauc",
        "mean_ap": "map",
        "min_auc_ap": "mina",
        "min_auc": "minu",
        "min_ap": "minp",
    }
    text = str(value)
    return mapping.get(text, _compact_token(text)[:5])


def _compact_replacement_strategy(value: object) -> str:
    mapping = {
        "random": "r",
        "neighborhood_mixed": "nm",
        "risk_aware_mixed": "ram",
        "residual_safe_mixed": "rsm",
        "diffusion_mixed": "dm",
        "calibrated_mixture": "cm",
    }
    text = str(value)
    return mapping.get(text, _compact_token(text)[:4])


def _compact_retrieval_selection(value: object) -> str:
    mapping = {
        "top_score": "ts",
        "topk_sample": "tks",
        "topk_multi": "tkm",
    }
    text = str(value)
    return mapping.get(text, _compact_token(text)[:4])


def _compact_retrieval_normalization(value: object) -> str:
    mapping = {
        "none": "n",
        "local_zscore": "lz",
        "robust_zscore": "rz",
        "local_rank": "lr",
        "pool_zscore": "pz",
        "pool_rank": "pr",
    }
    text = str(value)
    return mapping.get(text, _compact_token(text)[:4])


def _compact_teacher_mode(value: object) -> str:
    mapping = {
        "safe_hard": "sh",
    }
    text = str(value)
    return mapping.get(text, _compact_token(text)[:4])


def _stable_offset(name: str) -> int:
    return sum((index + 1) * ord(character) for index, character in enumerate(name))


def _named_training_seed(base_seed: int, epoch_index: int, stream_name: str) -> int:
    payload = f"scans-v3:{int(base_seed)}:{int(epoch_index)}:{stream_name}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") & ((1 << 63) - 1)


def _epoch_training_rng(
    legacy_rng: random.Random,
    base_seed: int,
    epoch_index: int,
    stream_name: str,
    strict: bool,
) -> random.Random:
    if not strict:
        return legacy_rng
    return random.Random(_named_training_seed(base_seed, epoch_index, stream_name))


if __name__ == "__main__":
    main()
