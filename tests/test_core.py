from __future__ import annotations

import random
import tempfile
import unittest
from pathlib import Path

import torch

from scans.data.file_loader import load_hyperedges_from_txt
from scans.data.hypergraph import HypergraphDataset
from scans.data.synthetic import generate_synthetic_hypergraph
from scans.calibration import calibrate_sampling_config
from scans.reporting import append_experiment_log
from scans.evaluation import future_positive_hit_metrics, negative_quality_metrics
from scans.models import BipartiteHyperedgePredictor
from scans.models import (
    BENCHMARK_BACKBONES,
    BenchmarkBackbonePredictor,
    DiscreteMembershipD3PM,
    LearnedCandidateRetriever,
    LearnedDualAnchor,
    MembershipDenoiser,
    PositiveSupportRiskEstimator,
    SparseIncidenceEncoder,
    StrongHyperedgePredictor,
    StrongEncoderPretrainer,
    primal_dual_update,
)
from scans.models.hyperedge_predictor import HyperedgePredictor
from scans.models.feature_encoder import DenseOrSparseLinear
from scans.risk import ClosureRiskIndex, CoWalkRiskIndex, DegreeCorrectedResidualRiskIndex, HittingRiskIndex
from scans.scoring import ModelAwareReranker, RerankConfig
from scans.samplers.anchored_safe_sampler import AnchoredSafeSampler, _PositiveEdgeIndex
from scans.samplers.anchored_replacement_sampler import AnchoredReplacementSampler
from scans.samplers.base import NegativeSampleBatch
from scans.samplers.base import jaccard
from scans.samplers.random_sampler import RandomSampler
from scans.samplers.registry import build_sampler
from scans.samplers.risk_controlled_sampler import RiskControlledSampler
from scans.samplers.size_matched_sampler import SizeMatchedSampler
from scans.training import run_mvp_experiment
from scans.training import FormalDiscreteGenerator, FormalGeneratorConfig
from scans.training.runner import (
    _build_adaptive_hardness_state,
    _build_negative_label_objective,
    _build_negative_loss_weighter,
    _build_proposal_mixture_state,
    _build_residual_safe_mixture_state,
    _effective_sampling_config,
    _loss_weights_for_batch,
    _training_loss,
    _update_adaptive_hardness_state,
    _update_proposal_mixture_state,
    _update_residual_safe_mixture_state,
)


class SyntheticDataTest(unittest.TestCase):
    def test_synthetic_dataset_is_valid(self) -> None:
        dataset = generate_synthetic_hypergraph(_synthetic_config(), seed=7)
        dataset.validate()
        self.assertGreater(len(dataset.train_edges), 0)
        self.assertGreater(len(dataset.val_edges), 0)
        self.assertGreater(len(dataset.test_edges), 0)


class FileLoaderTest(unittest.TestCase):
    def test_load_hyperedges_from_txt(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "edges.txt"
            path.write_text("0 1 2\n2 3\n4 5 6\n7 8\n8 9 10\n11 12\n", encoding="utf-8")
            dataset = load_hyperedges_from_txt(
                {
                    "path": str(path),
                    "separator": "whitespace",
                    "split": {"train": 0.6, "val": 0.2, "test": 0.2},
                },
                seed=1,
            )
            dataset.validate()
            self.assertEqual(dataset.num_nodes, 13)


class ModelTest(unittest.TestCase):
    def setUp(self) -> None:
        # Keep stochastic model tests reproducible and independent of execution order.
        random.seed(0)
        torch.manual_seed(0)

    def test_sparse_feature_projection_matches_dense_projection(self) -> None:
        dense = torch.tensor([[1.0, 0.0, 2.0], [0.0, 3.0, 0.0]])
        sparse = dense.to_sparse()
        projection = DenseOrSparseLinear(3, 4)
        self.assertTrue(torch.allclose(projection(dense), projection(sparse), atol=1e-6))

    def test_bipartite_predictor_supports_multilayer_fusion(self) -> None:
        train_edges = [(0, 1, 2), (2, 3, 4), (4, 5, 6)]
        model = BipartiteHyperedgePredictor(
            num_nodes=8,
            embedding_dim=6,
            hidden_dim=12,
            dropout=0.0,
            train_edges=train_edges,
            message_passing_layers=2,
            use_residual=True,
            use_layer_norm=True,
            representation_fusion_layers=2,
        )
        node_representations = model.encode_all_nodes()
        edge_representation = model.encode_edge_with_nodes(train_edges[0], node_representations)
        self.assertEqual(node_representations.shape, (8, 6))
        self.assertEqual(edge_representation.shape[0], 25)
        self.assertEqual(model.representation_fusion_layers, 2)

    def test_sparse_strong_encoder_and_pretraining_loss(self) -> None:
        train_edges = [(0, 1, 2), (2, 3, 4), (1, 4, 5)]
        encoder = SparseIncidenceEncoder(
            num_nodes=7,
            train_edges=train_edges,
            embedding_dim=8,
            hidden_dim=16,
            layers=2,
            dropout=0.0,
        )
        nodes, edges = encoder()
        self.assertEqual(nodes.shape, (7, 8))
        self.assertEqual(edges.shape, (3, 8))
        pretrainer = StrongEncoderPretrainer(encoder, projection_dim=8)
        loss, metrics = pretrainer.loss(mask_probability=0.25)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertIn("contrastive_loss", metrics)

    def test_strong_predictor_pretrains_and_scores_edges(self) -> None:
        train_edges = [(0, 1, 2), (2, 3, 4), (1, 4, 5)]
        model = StrongHyperedgePredictor(
            num_nodes=7,
            train_edges=train_edges,
            embedding_dim=8,
            hidden_dim=16,
            dropout=0.0,
            message_passing_layers=1,
            freeze_encoder=True,
        )
        summary = model.pretrain(
            epochs=1,
            learning_rate=0.001,
            weight_decay=0.0,
            mask_probability=0.2,
            contrastive_temperature=0.2,
        )
        logits = model.forward_edges(train_edges)
        self.assertEqual(logits.shape, (3,))
        self.assertTrue(torch.isfinite(logits).all())
        self.assertEqual(summary.epochs, 1)

    def test_learned_anchor_retrieval_d3pm_and_risk_modules(self) -> None:
        torch.manual_seed(3)
        batch_size = 2
        embedding_dim = 8
        edge_nodes = torch.randn(batch_size, 5, embedding_dim)
        padding = torch.tensor([[1, 1, 1, 1, 0], [1, 1, 1, 0, 0]], dtype=torch.bool)
        anchor = LearnedDualAnchor(embedding_dim, 16, max_edge_size=5)
        anchor_output = anchor(edge_nodes, padding, stochastic=True)
        self.assertTrue(torch.all(anchor_output.cardinality >= 1))
        self.assertTrue(torch.all(anchor_output.cardinality < padding.sum(dim=1)))

        all_nodes = torch.randn(12, embedding_dim)
        retriever = LearnedCandidateRetriever(embedding_dim)
        indices, scores = retriever(
            all_nodes,
            anchor_output.semantic_condition,
            edge_nodes[:, 0],
            top_k=6,
        )
        self.assertEqual(indices.shape, (batch_size, 6))
        self.assertEqual(scores.shape, (batch_size, 6))
        cached_indices, cached_scores = retriever(
            all_nodes,
            anchor_output.semantic_condition,
            edge_nodes[:, 0],
            top_k=6,
            projected_nodes=retriever.project_nodes(all_nodes),
        )
        self.assertTrue(torch.equal(indices, cached_indices))
        self.assertTrue(torch.allclose(scores, cached_scores))

        candidates = all_nodes[indices]
        x_0 = torch.zeros(batch_size, 6)
        x_0[:, :3] = 1
        anchor_mask = torch.zeros_like(x_0)
        anchor_mask[:, :1] = 1
        denoiser = MembershipDenoiser(embedding_dim, 24, max_steps=4)
        diffusion = DiscreteMembershipD3PM(denoiser, steps=4)
        self.assertTrue(torch.allclose(diffusion.transitions.sum(dim=-1), torch.ones_like(diffusion.transitions[..., 0])))
        self.assertTrue(
            torch.allclose(diffusion.cumulative_transitions.sum(dim=-1), torch.ones_like(diffusion.cumulative_transitions[..., 0]))
        )
        loss, metrics = diffusion.training_loss(
            x_0=x_0,
            anchor_mask=anchor_mask,
            candidate_embeddings=candidates,
            anchor_condition=anchor_output.semantic_condition,
            source_condition=edge_nodes[:, 0],
            target_hardness=torch.full((batch_size,), 0.5),
            risk_budget=torch.full((batch_size,), 0.2),
        )
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertIsNotNone(anchor.cardinality_head[-1].weight.grad)
        self.assertGreater(float(anchor.cardinality_head[-1].weight.grad.abs().sum()), 0.0)
        self.assertIn("d3pm_membership_accuracy", metrics)
        self.assertIn("d3pm_reverse_loss", metrics)
        sampled = diffusion.sample(
            anchor_mask=anchor_mask,
            candidate_embeddings=candidates,
            anchor_condition=anchor_output.semantic_condition,
            source_condition=edge_nodes[:, 0],
            target_hardness=torch.full((batch_size,), 0.5),
            risk_budget=torch.full((batch_size,), 0.2),
            target_cardinality=torch.full((batch_size,), 3),
        )
        self.assertTrue(torch.all(sampled[:, 0] == 1))
        self.assertTrue(torch.all(sampled.sum(dim=1) == 3))

        risk = PositiveSupportRiskEstimator(embedding_dim * 3, 16)
        risk_loss, _ = risk.loss(torch.randn(4, embedding_dim * 3), torch.randn(4, embedding_dim * 3))
        risk_loss.backward()
        updated = primal_dual_update(torch.tensor(0.0), torch.tensor([0.4, 0.6]), budget=0.3, learning_rate=0.1)
        self.assertGreater(float(updated), 0.0)

    def test_formal_discrete_generator_outputs_auditable_node_sets(self) -> None:
        train_edges = [(0, 1, 2), (2, 3, 4), (1, 4, 5), (0, 5, 6)]
        generator = FormalDiscreteGenerator(
            num_nodes=8,
            train_edges=train_edges,
            config=FormalGeneratorConfig(
                embedding_dim=8,
                hidden_dim=16,
                encoder_layers=1,
                encoder_epochs=1,
                generator_epochs=1,
                risk_epochs=1,
                candidate_size=8,
                diffusion_steps=2,
                batch_size=2,
                proposals_per_edge=2,
            ),
        )
        metrics = generator.fit(random.Random(5))
        negatives, audit = generator.generate(train_edges, set(train_edges), random.Random(7))
        self.assertEqual(len(negatives), len(train_edges))
        self.assertTrue(all(len(edge) == len(source) for edge, source in zip(negatives, train_edges)))
        self.assertTrue(all(edge not in set(train_edges) for edge in negatives))
        self.assertIn("pretrain_final_loss", metrics)
        self.assertIn("generated_risk_mean", audit)

    def test_historical_energy_ablation_outputs_auditable_node_sets(self) -> None:
        train_edges = [(0, 1, 2), (2, 3, 4), (1, 4, 5), (0, 5, 6)]
        generator = FormalDiscreteGenerator(
            num_nodes=8,
            train_edges=train_edges,
            config=FormalGeneratorConfig(
                embedding_dim=8,
                hidden_dim=16,
                encoder_layers=1,
                encoder_epochs=1,
                generator_epochs=1,
                risk_epochs=1,
                candidate_size=8,
                diffusion_steps=2,
                batch_size=2,
                diffusion_mode="historical_energy",
                proposals_per_edge=2,
            ),
        )
        generator.fit(random.Random(11))
        negatives, _ = generator.generate(train_edges, set(train_edges), random.Random(13))
        self.assertTrue(all(len(edge) == len(source) for edge, source in zip(negatives, train_edges)))
        self.assertTrue(all(edge not in set(train_edges) for edge in negatives))

    def test_sota_backbone_adapters_score_edges(self) -> None:
        train_edges = [(0, 1, 2), (2, 3, 4), (1, 4, 5), (0, 5, 6)]
        for backbone in BENCHMARK_BACKBONES:
            model = BenchmarkBackbonePredictor(
                backbone=backbone,
                num_nodes=8,
                train_edges=train_edges,
                embedding_dim=8,
                hidden_dim=16,
                dropout=0.0,
                message_passing_layers=1,
            )
            logits = model.forward_edges(train_edges)
            self.assertEqual(logits.shape, (len(train_edges),), backbone)
            self.assertTrue(torch.isfinite(logits).all(), backbone)

    def test_backbone_selection_candidates_share_scans_representation_contract(self) -> None:
        train_edges = [(0, 1, 2), (2, 3, 4), (1, 4, 5), (0, 5, 6)]
        candidates = ("hypergcn", "unigcnii", "hnhn", "allset_deepsets", "edhnn")
        for backbone in candidates:
            model = BenchmarkBackbonePredictor(
                backbone=backbone,
                num_nodes=8,
                train_edges=train_edges,
                embedding_dim=8,
                hidden_dim=16,
                dropout=0.0,
                message_passing_layers=2,
            )
            representations = model.edge_representations(train_edges)
            logits = model.logits_from_edge_representations(representations)
            expected_dim = 8 if backbone == "hnhn" else 33
            self.assertEqual(representations.shape, (len(train_edges), expected_dim), backbone)
            self.assertEqual(logits.shape, (len(train_edges),), backbone)
            self.assertTrue(torch.isfinite(representations).all(), backbone)
            self.assertTrue(torch.isfinite(logits).all(), backbone)
            logits.sum().backward()
            gradients = [parameter.grad for parameter in model.parameters() if parameter.requires_grad]
            self.assertTrue(any(gradient is not None for gradient in gradients), backbone)

    def test_formal_representation_cache_is_reused(self) -> None:
        train_edges = [(0, 1, 2), (2, 3, 4), (1, 4, 5), (0, 5, 6)]
        with tempfile.TemporaryDirectory() as tmp_dir:
            cache_path = str(Path(tmp_dir) / "representation.pt")
            config = FormalGeneratorConfig(
                embedding_dim=8,
                hidden_dim=16,
                encoder_layers=1,
                encoder_epochs=1,
                risk_epochs=1,
                candidate_size=8,
                diffusion_steps=2,
                batch_size=2,
                representation_cache_path=cache_path,
            )
            first = FormalDiscreteGenerator(8, train_edges, config)
            first_metrics = first.fit_representation(random.Random(17))
            second = FormalDiscreteGenerator(8, train_edges, config)
            second_metrics = second.fit_representation(random.Random(19))
            self.assertEqual(first_metrics["representation_cache_hit"], 0.0)
            self.assertEqual(second_metrics["representation_cache_hit"], 1.0)


class SamplerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.dataset = generate_synthetic_hypergraph(_synthetic_config(), seed=11)
        self.rng = random.Random(11)

    def test_random_sampler_outputs_valid_negatives(self) -> None:
        sampler = RandomSampler(min_edge_size=3, max_edge_size=6)
        batch = sampler.sample(self.dataset.train_edges[:10], self.dataset.num_nodes, self.dataset.positive_edges, self.rng, 1)
        self.assertEqual(len(batch.edges), 10)
        self.assertTrue(all(edge not in self.dataset.positive_edges for edge in batch.edges))

    def test_size_matched_sampler_preserves_source_size(self) -> None:
        sampler = SizeMatchedSampler()
        sources = self.dataset.train_edges[:10]
        batch = sampler.sample(sources, self.dataset.num_nodes, self.dataset.positive_edges, self.rng, 1)
        self.assertEqual([len(edge) for edge in batch.edges], [len(edge) for edge in sources])

    def test_official_sehp_sampler_names_output_valid_negatives(self) -> None:
        sources = self.dataset.train_edges[:8]
        for sampler_name in ("sns", "mns", "cns", "mix"):
            sampler = build_sampler(
                sampler_name,
                {"max_attempts": 100},
                min(len(edge) for edge in sources),
                max(len(edge) for edge in sources),
            )
            batch = sampler.sample(sources, self.dataset.num_nodes, self.dataset.positive_edges, self.rng, 1)
            self.assertEqual(len(batch.edges), len(sources))
            self.assertTrue(all(edge not in self.dataset.positive_edges for edge in batch.edges))

    def test_structural_sampler_index_cache_preserves_seeded_outputs(self) -> None:
        sources = self.dataset.train_edges[:12]
        for sampler_name in ("mns", "cns", "mix"):
            cached = build_sampler(sampler_name, {"max_attempts": 100}, 2, 8)
            cached.sample(sources, self.dataset.num_nodes, self.dataset.positive_edges, random.Random(17), 1)
            cached_batch = cached.sample(
                sources,
                self.dataset.num_nodes,
                self.dataset.positive_edges,
                random.Random(29),
                1,
            )
            fresh = build_sampler(sampler_name, {"max_attempts": 100}, 2, 8)
            fresh_batch = fresh.sample(
                sources,
                self.dataset.num_nodes,
                self.dataset.positive_edges,
                random.Random(29),
                1,
            )
            self.assertEqual(cached_batch.edges, fresh_batch.edges)

    def test_anchored_sampler_keeps_anchor_and_respects_jaccard_bound(self) -> None:
        sampler = AnchoredReplacementSampler(anchor_ratio=0.5, jaccard_upper_bound=0.75)
        sources = self.dataset.train_edges[:10]
        batch = sampler.sample(sources, self.dataset.num_nodes, self.dataset.positive_edges, self.rng, 1)
        for negative, source in zip(batch.edges, sources):
            self.assertEqual(len(negative), len(source))
            self.assertGreaterEqual(len(set(negative) & set(source)), 1)
            self.assertLessEqual(jaccard(negative, source), 0.75)
            self.assertNotIn(negative, self.dataset.positive_edges)

    def test_anchored_safe_sampler_respects_nearest_positive_bound_when_possible(self) -> None:
        sampler = AnchoredSafeSampler(
            anchor_ratio=0.25,
            jaccard_upper_bound=0.75,
            nearest_positive_upper_bound=0.5,
            max_attempts=200,
        )
        sources = self.dataset.train_edges[:10]
        batch = sampler.sample(sources, self.dataset.num_nodes, self.dataset.positive_edges, self.rng, 1)
        self.assertEqual(len(batch.edges), 10)
        for negative in batch.edges:
            nearest_positive = max(jaccard(negative, positive) for positive in self.dataset.positive_edges)
            self.assertLessEqual(nearest_positive, 0.5)

    def test_positive_edge_index_vectorized_path_matches_direct_jaccard(self) -> None:
        positive_edges = {(0, node, 1000 + node) for node in range(1, 129)}
        excluded_edge = (0, 64, 1064)
        candidate = (0, 64, 2000)
        index = _PositiveEdgeIndex(positive_edges)
        expected = max(
            jaccard(candidate, positive)
            for positive in positive_edges
            if positive != excluded_edge
        )
        self.assertGreaterEqual(sum(len(index.by_node[node]) for node in candidate if node in index.by_node), 64)
        self.assertEqual(
            index.nearest_similarity(candidate, excluded_edge=excluded_edge),
            expected,
        )

    def test_closure_risk_is_higher_for_seen_pairs(self) -> None:
        index = ClosureRiskIndex({(0, 1, 2), (0, 1, 3), (10, 11, 12)})
        self.assertGreater(index.risk((0, 1, 4)), index.risk((4, 5, 6)))

    def test_closure_risk_pair_cap_is_deterministic_and_bounded(self) -> None:
        large_edge = tuple(range(200))
        first = ClosureRiskIndex({large_edge}, max_pairs_per_edge=64)
        second = ClosureRiskIndex({large_edge}, max_pairs_per_edge=64)
        self.assertEqual(first.pair_counts, second.pair_counts)
        self.assertEqual(sum(first.pair_counts.values()), 64)
        self.assertEqual(first.risk(large_edge), second.risk(large_edge))

    def test_cowalk_risk_is_higher_for_shared_neighborhoods(self) -> None:
        index = CoWalkRiskIndex({(0, 1, 4), (1, 2, 4), (10, 11, 12)})
        self.assertGreater(index.risk((0, 2, 5)), index.risk((5, 6, 7)))

    def test_hitting_risk_is_higher_for_reachable_nodes(self) -> None:
        index = HittingRiskIndex({(0, 1, 4), (1, 2, 4), (10, 11, 12)})
        self.assertGreater(index.risk((0, 2, 5)), index.risk((5, 6, 7)))

    def test_degree_corrected_residual_ignores_degree_only_pairs(self) -> None:
        positive_edges = {
            (0, 1, 2),
            (0, 1, 3),
            (0, 1, 4),
            (0, 5, 6),
            (0, 7, 8),
            (1, 9, 10),
            (1, 11, 12),
        }
        index = DegreeCorrectedResidualRiskIndex(positive_edges)
        self.assertGreater(index.risk((0, 1, 13)), index.risk((0, 9, 13)))

    def test_risk_controlled_sampler_outputs_closure_limited_negatives(self) -> None:
        source = (0, 1, 2, 3)
        positive_edges = {source, (0, 1, 4), (10, 11, 12)}
        sampler = RiskControlledSampler(
            anchor_ratio=0.25,
            nearest_positive_upper_bound=1.0,
            closure_risk_upper_bound=0.7,
            use_cowalk_risk=True,
            cowalk_risk_upper_bound=1.0,
            use_residual_risk=True,
            residual_risk_upper_bound=10.0,
            max_attempts=300,
        )
        batch = sampler.sample([source] * 5, 20, positive_edges, self.rng, 1)
        closure_index = ClosureRiskIndex(positive_edges)
        self.assertEqual(len(batch.edges), 5)
        self.assertIn("risk_accept_rate", batch.metadata)
        self.assertIn("risk_fallback_rate", batch.metadata)
        self.assertIn("sampling_attempts_mean", batch.metadata)
        self.assertIn("nearest_constraint_pass_rate", batch.metadata)
        self.assertIn("closure_constraint_pass_rate", batch.metadata)
        self.assertIn("cowalk_constraint_pass_rate", batch.metadata)
        self.assertIn("residual_constraint_pass_rate", batch.metadata)
        self.assertIn("budget_constraint_pass_rate", batch.metadata)
        self.assertIn("combined_risk_mean", batch.metadata)
        self.assertIn("risk_cache_hit_rate", batch.metadata)
        self.assertIn("nearest_primary_rejection_rate", batch.metadata)
        self.assertIn("closure_primary_rejection_rate", batch.metadata)
        self.assertIn("cowalk_primary_rejection_rate", batch.metadata)
        self.assertIn("residual_primary_rejection_rate", batch.metadata)
        self.assertIn("budget_primary_rejection_rate", batch.metadata)
        for negative in batch.edges:
            self.assertLessEqual(closure_index.risk(negative), 0.7)

    def test_risk_budget_sampler_records_budget_diagnostics(self) -> None:
        source = (0, 1, 2, 3)
        positive_edges = {source, (0, 1, 4), (10, 11, 12)}
        sampler = RiskControlledSampler(
            anchor_ratio=0.25,
            nearest_positive_upper_bound=1.0,
            closure_risk_upper_bound=0.7,
            use_cowalk_risk=True,
            cowalk_risk_upper_bound=1.0,
            use_risk_budget=True,
            risk_budget=3.0,
            max_attempts=300,
        )
        batch = sampler.sample([source] * 5, 20, positive_edges, self.rng, 1)
        self.assertEqual(len(batch.edges), 5)
        self.assertGreaterEqual(batch.metadata["budget_constraint_pass_rate"], 0.0)
        self.assertGreaterEqual(batch.metadata["combined_risk_mean"], 0.0)

    def test_neighborhood_mixed_replacement_records_neighbor_usage(self) -> None:
        source = (0, 1, 2, 3)
        positive_edges = {
            source,
            (0, 4, 5),
            (1, 6, 7),
            (2, 8, 9),
            (10, 11, 12),
        }
        sampler = RiskControlledSampler(
            anchor_ratio=0.5,
            nearest_positive_upper_bound=1.0,
            closure_risk_upper_bound=10.0,
            replacement_strategy="neighborhood_mixed",
            neighbor_sample_probability=1.0,
            max_attempts=100,
        )
        batch = sampler.sample([source] * 5, 20, positive_edges, self.rng, 1)
        self.assertEqual(len(batch.edges), 5)
        self.assertIn("neighbor_replacement_rate", batch.metadata)
        self.assertIn("risk_cache_hit_rate", batch.metadata)
        self.assertGreater(batch.metadata["neighbor_replacement_rate"], 0.0)
        self.assertGreater(batch.metadata["candidate_pool_size_mean"], 0.0)

    def test_risk_aware_mixed_replacement_records_neighbor_usage(self) -> None:
        source = (0, 1, 2, 3)
        positive_edges = {
            source,
            (0, 4, 5),
            (1, 6, 7),
            (2, 8, 9),
            (10, 11, 12),
        }
        sampler = RiskControlledSampler(
            anchor_ratio=0.5,
            nearest_positive_upper_bound=1.0,
            closure_risk_upper_bound=10.0,
            replacement_strategy="risk_aware_mixed",
            neighbor_sample_probability=1.0,
            max_attempts=100,
        )
        batch = sampler.sample([source] * 5, 20, positive_edges, self.rng, 1)
        self.assertEqual(len(batch.edges), 5)
        self.assertIn("neighbor_replacement_rate", batch.metadata)
        self.assertGreater(batch.metadata["candidate_pool_size_mean"], 0.0)

    def test_residual_safe_mixed_replacement_records_usage(self) -> None:
        source = (0, 1, 2, 3)
        positive_edges = {
            source,
            (0, 4, 5),
            (1, 6, 7),
            (2, 8, 9),
            (0, 1, 10),
            (10, 11, 12),
        }
        sampler = RiskControlledSampler(
            anchor_ratio=0.5,
            nearest_positive_upper_bound=1.0,
            closure_risk_upper_bound=10.0,
            use_residual_risk=True,
            residual_risk_upper_bound=10.0,
            replacement_strategy="residual_safe_mixed",
            neighbor_sample_probability=1.0,
            residual_safe_pool_multiplier=3,
            max_attempts=100,
        )
        batch = sampler.sample([source] * 4, 20, positive_edges, self.rng, 1)
        self.assertEqual(len(batch.edges), 4)
        self.assertIn("residual_safe_replacement_rate", batch.metadata)
        self.assertGreater(batch.metadata["residual_safe_replacement_rate"], 0.0)
        self.assertGreater(batch.metadata["candidate_pool_size_mean"], 0.0)

    def test_diffusion_mixed_replacement_records_usage(self) -> None:
        source = (0, 1, 2, 3)
        positive_edges = {
            source,
            (0, 4, 5),
            (1, 6, 7),
            (2, 8, 9),
            (0, 1, 10),
            (10, 11, 12),
        }
        sampler = RiskControlledSampler(
            anchor_ratio=0.5,
            nearest_positive_upper_bound=1.0,
            closure_risk_upper_bound=10.0,
            use_residual_risk=True,
            residual_risk_upper_bound=10.0,
            replacement_strategy="diffusion_mixed",
            neighbor_sample_probability=1.0,
            diffusion_steps=3,
            diffusion_pool_multiplier=4,
            max_attempts=100,
        )
        batch = sampler.sample([source] * 4, 20, positive_edges, self.rng, 1)
        self.assertEqual(len(batch.edges), 4)
        self.assertIn("diffusion_replacement_rate", batch.metadata)
        self.assertGreater(batch.metadata["diffusion_replacement_rate"], 0.0)
        self.assertGreater(batch.metadata["candidate_pool_size_mean"], 0.0)

    def test_risk_aware_nearest_proxy_keeps_valid_output(self) -> None:
        source = (0, 1, 2, 3)
        positive_edges = {
            source,
            (0, 4, 5),
            (1, 6, 7),
            (2, 8, 9),
            (4, 6, 8),
        }
        sampler = RiskControlledSampler(
            anchor_ratio=0.5,
            nearest_positive_upper_bound=1.0,
            closure_risk_upper_bound=10.0,
            replacement_strategy="risk_aware_mixed",
            neighbor_sample_probability=1.0,
            risk_aware_nearest_weight=2.0,
            max_attempts=100,
        )
        batch = sampler.sample([source] * 4, 20, positive_edges, self.rng, 1)
        self.assertEqual(len(batch.edges), 4)
        self.assertGreater(batch.metadata["candidate_pool_size_mean"], 0.0)

    def test_calibrated_mixture_records_risk_aware_usage(self) -> None:
        source = (0, 1, 2, 3)
        positive_edges = {
            source,
            (0, 4, 5),
            (1, 6, 7),
            (2, 8, 9),
            (10, 11, 12),
        }
        sampler = RiskControlledSampler(
            anchor_ratio=0.5,
            nearest_positive_upper_bound=1.0,
            closure_risk_upper_bound=10.0,
            replacement_strategy="calibrated_mixture",
            neighbor_sample_probability=1.0,
            mixture_risk_aware_probability=1.0,
            max_attempts=100,
        )
        batch = sampler.sample([source] * 4, 20, positive_edges, self.rng, 1)
        self.assertEqual(len(batch.edges), 4)
        self.assertGreater(batch.metadata["risk_aware_replacement_rate"], 0.0)

    def test_boundary_guided_sampler_records_adversarial_diagnostics(self) -> None:
        source = (0, 1, 2, 3)
        positive_edges = {
            source,
            (0, 4, 5),
            (1, 4, 6),
            (2, 7, 8),
            (10, 11, 12),
        }
        sampler = RiskControlledSampler(
            anchor_ratio=0.5,
            nearest_positive_upper_bound=1.0,
            closure_risk_upper_bound=10.0,
            replacement_strategy="neighborhood_mixed",
            neighbor_sample_probability=1.0,
            boundary_guided_probe_count=4,
            boundary_guided_max_rounds=3,
            boundary_guided_min_feasible_candidates=2,
            max_attempts=100,
        )
        batch = sampler.sample_boundary_guided(
            source_edges=[source] * 3,
            num_nodes=20,
            positive_edges=positive_edges,
            scorer=_BoundaryToyScorer(),
            rng=self.rng,
            negatives_per_positive=1,
            target_score=0.5,
            score_lower_bound=0.3,
            score_upper_bound=0.7,
        )
        self.assertEqual(len(batch.edges), 3)
        self.assertEqual(batch.metadata["adversarial_proposal_enabled"], 1.0)
        self.assertGreater(batch.metadata["adversarial_probe_candidates_mean"], 0.0)
        self.assertIn("adversarial_feasible_candidates_mean", batch.metadata)
        self.assertIn("adversarial_selected_in_band_rate", batch.metadata)

    def test_boundary_guided_elite_search_records_feedback_usage(self) -> None:
        source = (0, 1, 2, 3)
        positive_edges = {
            source,
            (0, 1, 4, 5),
            (0, 2, 6, 7),
            (8, 9, 10, 11),
        }
        sampler = RiskControlledSampler(
            anchor_ratio=0.5,
            nearest_positive_upper_bound=1.0,
            closure_risk_upper_bound=10.0,
            replacement_strategy="neighborhood_mixed",
            neighbor_sample_probability=1.0,
            boundary_guided_probe_count=4,
            boundary_guided_max_rounds=3,
            boundary_guided_min_feasible_candidates=20,
            boundary_guided_elite_enabled=True,
            boundary_guided_elite_mix_probability=1.0,
        )
        scorer = _StaticScorer([0.1, 0.45, 0.7, 0.35] * 4)
        batch = sampler.sample_boundary_guided(
            source_edges=[source],
            num_nodes=16,
            positive_edges=positive_edges,
            scorer=scorer,
            rng=random.Random(31),
            negatives_per_positive=1,
            target_score=0.5,
            target_score_min=0.5,
            target_score_max=0.5,
            score_lower_bound=0.3,
            score_upper_bound=0.7,
        )
        self.assertEqual(batch.metadata["adversarial_elite_enabled"], 1.0)
        self.assertGreater(batch.metadata["adversarial_elite_updates_mean"], 0.0)
        self.assertGreater(batch.metadata["adversarial_elite_replacement_rate"], 0.0)

    def test_primal_dual_boundary_selection_records_constraint_metrics(self) -> None:
        source = (0, 1, 2, 3)
        positive_edges = {
            source,
            (0, 1, 4, 5),
            (0, 2, 6, 7),
            (8, 9, 10, 11),
        }
        sampler = RiskControlledSampler(
            anchor_ratio=0.5,
            nearest_positive_upper_bound=0.30,
            closure_risk_upper_bound=0.01,
            use_risk_budget=True,
            risk_budget=2.0,
            use_budget_safety_caps=True,
            safety_nearest_positive_upper_bound=1.0,
            safety_closure_risk_upper_bound=10.0,
            replacement_strategy="neighborhood_mixed",
            neighbor_sample_probability=1.0,
            boundary_guided_probe_count=4,
            boundary_guided_max_rounds=3,
            boundary_guided_min_feasible_candidates=2,
            primal_dual_enabled=True,
            primal_dual_learning_rate=0.2,
            primal_dual_max_lambda=5.0,
        )
        batch = sampler.sample_boundary_guided(
            source_edges=[source] * 2,
            num_nodes=16,
            positive_edges=positive_edges,
            scorer=_BoundaryToyScorer(),
            rng=random.Random(41),
            negatives_per_positive=1,
            target_score=0.5,
            score_lower_bound=0.3,
            score_upper_bound=0.7,
        )
        self.assertEqual(len(batch.edges), 2)
        self.assertEqual(batch.metadata["primal_dual_enabled"], 1.0)
        self.assertIn("primal_dual_objective_mean", batch.metadata)
        self.assertIn("primal_dual_risk_penalty_mean", batch.metadata)
        self.assertGreaterEqual(batch.metadata["primal_dual_lambda_nearest_mean"], 0.0)

    def test_registry_passes_full_positive_support_risk_config(self) -> None:
        sampler = build_sampler(
            "risk_controlled",
            {
                "use_hitting_risk": True,
                "hitting_risk_upper_bound": 0.11,
                "use_residual_risk": True,
                "residual_risk_upper_bound": 0.22,
                "hitting_risk_weight": 3.0,
                "residual_risk_weight": 4.0,
                "mixture_residual_safe_probability": 0.25,
                "residual_safe_pool_multiplier": 7,
                "primal_dual_enabled": True,
                "primal_dual_learning_rate": 0.12,
                "primal_dual_max_lambda": 3.5,
                "primal_dual_hardness_scale": 0.8,
                "adversarial_proposal": {"elite_enabled": True},
            },
            min_edge_size=3,
            max_edge_size=5,
        )
        self.assertIsInstance(sampler, RiskControlledSampler)
        assert isinstance(sampler, RiskControlledSampler)
        self.assertTrue(sampler.use_hitting_risk)
        self.assertTrue(sampler.use_residual_risk)
        self.assertAlmostEqual(sampler.hitting_risk_upper_bound, 0.11)
        self.assertAlmostEqual(sampler.residual_risk_upper_bound, 0.22)
        self.assertAlmostEqual(sampler.hitting_risk_weight, 3.0)
        self.assertAlmostEqual(sampler.residual_risk_weight, 4.0)
        self.assertAlmostEqual(sampler.mixture_residual_safe_probability, 0.25)
        self.assertEqual(sampler.residual_safe_pool_multiplier, 7)
        self.assertTrue(sampler.primal_dual_enabled)
        self.assertAlmostEqual(sampler.primal_dual_learning_rate, 0.12)
        self.assertAlmostEqual(sampler.primal_dual_max_lambda, 3.5)
        self.assertAlmostEqual(sampler.primal_dual_hardness_scale, 0.8)
        self.assertTrue(sampler.boundary_guided_elite_enabled)

    def test_budget_safety_caps_keep_nearest_positive_limited(self) -> None:
        source = (0, 1, 2, 3)
        positive_edges = {source, (0, 1, 4), (10, 11, 12)}
        sampler = RiskControlledSampler(
            anchor_ratio=0.25,
            nearest_positive_upper_bound=1.0,
            closure_risk_upper_bound=1.0,
            use_cowalk_risk=True,
            cowalk_risk_upper_bound=1.0,
            use_risk_budget=True,
            risk_budget=3.0,
            use_budget_safety_caps=True,
            safety_nearest_positive_upper_bound=0.5,
            safety_closure_risk_upper_bound=1.0,
            safety_cowalk_risk_upper_bound=1.0,
            max_attempts=300,
        )
        batch = sampler.sample([source] * 5, 20, positive_edges, self.rng, 1)
        self.assertEqual(len(batch.edges), 5)
        for negative in batch.edges:
            nearest_other = max(
                jaccard(negative, positive)
                for positive in positive_edges
                if positive != source
            )
            self.assertLessEqual(nearest_other, 0.5)

    def test_adaptive_risk_calibration_updates_thresholds(self) -> None:
        sampling_config = {
            "anchor_ratio": 0.25,
            "nearest_positive_upper_bound": 1.0,
            "closure_risk_upper_bound": 1.0,
            "use_cowalk_risk": True,
            "cowalk_risk_upper_bound": 1.0,
            "use_risk_budget": True,
            "risk_budget": 10.0,
            "nearest_risk_weight": 4.0,
            "closure_risk_weight": 1.0,
            "cowalk_risk_weight": 1.0,
            "max_cowalk_neighbors": 32,
            "adaptive_risk": {
                "enabled": True,
                "max_source_edges": 12,
                "candidates_per_edge": 3,
                "scale_quantile": 0.5,
                "budget_quantile": 0.2,
            },
        }
        calibrated_config, metrics = calibrate_sampling_config(
            sampling_config=sampling_config,
            train_edges=self.dataset.train_edges,
            num_nodes=self.dataset.num_nodes,
            positive_edges=self.dataset.train_positive_edges,
            seed=13,
        )
        self.assertGreater(metrics["adaptive_calibration_candidates"], 0.0)
        self.assertGreater(calibrated_config["nearest_positive_upper_bound"], 0.0)
        self.assertGreater(calibrated_config["closure_risk_upper_bound"], 0.0)
        self.assertGreater(calibrated_config["cowalk_risk_upper_bound"], 0.0)
        self.assertGreater(calibrated_config["risk_budget"], 0.0)

    def test_adaptive_risk_can_set_order_statistic_safety_caps(self) -> None:
        sampling_config = {
            "anchor_ratio": 0.25,
            "nearest_positive_upper_bound": 1.0,
            "closure_risk_upper_bound": 1.0,
            "use_cowalk_risk": True,
            "cowalk_risk_upper_bound": 1.0,
            "use_risk_budget": True,
            "risk_budget": 10.0,
            "nearest_risk_weight": 4.0,
            "closure_risk_weight": 1.0,
            "cowalk_risk_weight": 1.0,
            "max_cowalk_neighbors": 32,
            "use_budget_safety_caps": False,
            "adaptive_risk": {
                "enabled": True,
                "max_source_edges": 12,
                "candidates_per_edge": 3,
                "scale_quantile": 0.35,
                "budget_quantile": 0.35,
                "nonzero_scale_quantile": 0.35,
                "nonzero_budget_quantile": 0.35,
                "quantile_method": "order_statistic",
                "set_safety_caps": True,
                "safety_cap_multiplier": 1.0,
            },
        }
        calibrated_config, metrics = calibrate_sampling_config(
            sampling_config=sampling_config,
            train_edges=self.dataset.train_edges,
            num_nodes=self.dataset.num_nodes,
            positive_edges=self.dataset.train_positive_edges,
            seed=17,
        )
        self.assertTrue(calibrated_config["use_budget_safety_caps"])
        self.assertEqual(metrics["adaptive_order_statistic_quantile"], 1.0)
        self.assertEqual(
            calibrated_config["safety_nearest_positive_upper_bound"],
            calibrated_config["nearest_positive_upper_bound"],
        )
        self.assertEqual(
            calibrated_config["safety_closure_risk_upper_bound"],
            calibrated_config["closure_risk_upper_bound"],
        )
        self.assertEqual(
            calibrated_config["safety_cowalk_risk_upper_bound"],
            calibrated_config["cowalk_risk_upper_bound"],
        )

    def test_adaptive_risk_can_calibrate_hitting_risk(self) -> None:
        sampling_config = {
            "anchor_ratio": 0.25,
            "nearest_positive_upper_bound": 1.0,
            "closure_risk_upper_bound": 1.0,
            "use_cowalk_risk": True,
            "cowalk_risk_upper_bound": 1.0,
            "use_hitting_risk": True,
            "hitting_risk_upper_bound": 1.0,
            "use_risk_budget": True,
            "risk_budget": 10.0,
            "nearest_risk_weight": 4.0,
            "closure_risk_weight": 1.0,
            "cowalk_risk_weight": 1.0,
            "hitting_risk_weight": 1.0,
            "adaptive_risk": {
                "enabled": True,
                "max_source_edges": 12,
                "candidates_per_edge": 3,
                "scale_quantile": 0.5,
                "budget_quantile": 0.2,
                "set_safety_caps": True,
            },
        }
        calibrated_config, metrics = calibrate_sampling_config(
            sampling_config=sampling_config,
            train_edges=self.dataset.train_edges,
            num_nodes=self.dataset.num_nodes,
            positive_edges=self.dataset.train_positive_edges,
            seed=19,
        )
        self.assertGreater(metrics["adaptive_hitting_risk_upper_bound"], 0.0)
        self.assertGreater(calibrated_config["hitting_risk_upper_bound"], 0.0)
        self.assertEqual(
            calibrated_config["safety_hitting_risk_upper_bound"],
            calibrated_config["hitting_risk_upper_bound"],
        )

    def test_adaptive_risk_can_calibrate_residual_risk(self) -> None:
        sampling_config = {
            "anchor_ratio": 0.25,
            "nearest_positive_upper_bound": 1.0,
            "closure_risk_upper_bound": 1.0,
            "use_residual_risk": True,
            "residual_risk_upper_bound": 1.0,
            "use_risk_budget": True,
            "risk_budget": 10.0,
            "nearest_risk_weight": 4.0,
            "closure_risk_weight": 1.0,
            "residual_risk_weight": 1.0,
            "adaptive_risk": {
                "enabled": True,
                "max_source_edges": 12,
                "candidates_per_edge": 3,
                "scale_quantile": 0.5,
                "budget_quantile": 0.2,
                "set_safety_caps": True,
            },
        }
        calibrated_config, metrics = calibrate_sampling_config(
            sampling_config=sampling_config,
            train_edges=self.dataset.train_edges,
            num_nodes=self.dataset.num_nodes,
            positive_edges=self.dataset.train_positive_edges,
            seed=23,
        )
        self.assertGreater(metrics["adaptive_residual_risk_upper_bound"], 0.0)
        self.assertGreater(calibrated_config["residual_risk_upper_bound"], 0.0)
        self.assertEqual(
            calibrated_config["safety_residual_risk_upper_bound"],
            calibrated_config["residual_risk_upper_bound"],
        )

    def test_proposal_gating_resolves_auto_strategy(self) -> None:
        positive_edges = {
            (0, 1, 2, 3),
            (0, 4, 5),
            (1, 6, 7),
            (2, 8, 9),
            (10, 11, 12),
        }
        sampling_config = {
            "max_attempts": 50,
            "anchor_ratio": 0.5,
            "nearest_positive_upper_bound": 1.0,
            "closure_risk_upper_bound": 10.0,
            "replacement_strategy": "auto_risk_aware",
            "neighbor_sample_probability": 1.0,
            "proposal_gating": {
                "enabled": True,
                "max_source_edges": 3,
                "nearest_activation_rate": 0.0,
                "fallback_activation_rate": 0.0,
                "max_fallback_increase": 1.0,
                "risk_reduction_margin": 0.0,
            },
        }
        calibrated_config, metrics = calibrate_sampling_config(
            sampling_config=sampling_config,
            train_edges=list(positive_edges),
            num_nodes=20,
            positive_edges=positive_edges,
            seed=42,
        )
        self.assertIn(calibrated_config["replacement_strategy"], {"neighborhood_mixed", "risk_aware_mixed"})
        self.assertEqual(metrics["proposal_gating_enabled"], 1.0)
        self.assertIn("proposal_gating_selected_risk_aware", metrics)

    def test_calibrated_mixture_sets_probability_from_training_pilot(self) -> None:
        positive_edges = {
            (0, 1, 2, 3),
            (0, 4, 5),
            (1, 6, 7),
            (2, 8, 9),
            (10, 11, 12),
        }
        sampling_config = {
            "max_attempts": 50,
            "anchor_ratio": 0.5,
            "nearest_positive_upper_bound": 1.0,
            "closure_risk_upper_bound": 10.0,
            "replacement_strategy": "calibrated_mixture",
            "neighbor_sample_probability": 1.0,
            "proposal_mixture": {
                "max_source_edges": 3,
                "fallback_low": 0.0,
                "fallback_high": 1.0,
                "max_risk_aware_probability": 0.5,
            },
        }
        calibrated_config, metrics = calibrate_sampling_config(
            sampling_config=sampling_config,
            train_edges=list(positive_edges),
            num_nodes=20,
            positive_edges=positive_edges,
            seed=42,
        )
        self.assertEqual(calibrated_config["replacement_strategy"], "calibrated_mixture")
        self.assertIn("mixture_risk_aware_probability", calibrated_config)
        self.assertGreaterEqual(calibrated_config["mixture_risk_aware_probability"], 0.0)
        self.assertLessEqual(calibrated_config["mixture_risk_aware_probability"], 0.5)
        self.assertEqual(metrics["proposal_mixture_enabled"], 1.0)


class MetricsTest(unittest.TestCase):
    def test_negative_quality_metrics_records_hard_negative_ratio(self) -> None:
        metrics = negative_quality_metrics(
            negative_edges=[(0, 1), (2, 3), (4, 5)],
            source_edges=[(0, 2), (2, 4), (4, 6)],
            positive_edges={(0, 2), (2, 4), (4, 6)},
            hardness_scores=[0.1, 0.4, 0.8],
            hard_negative_lower_bound=0.3,
            hard_negative_upper_bound=0.7,
        )
        self.assertEqual(metrics["hard_negative_ratio"], 1 / 3)
        self.assertEqual(metrics["hard_negative_lower_bound"], 0.3)
        self.assertEqual(metrics["hard_negative_upper_bound"], 0.7)

    def test_future_positive_hit_metrics_counts_future_hits(self) -> None:
        metrics = future_positive_hit_metrics(
            negative_edges=[(0, 1), (2, 3), (4, 5)],
            future_positive_edges={(2, 3), (6, 7)},
            source_count=3,
        )
        self.assertEqual(metrics["future_positive_hits"], 1.0)
        self.assertEqual(metrics["future_positive_hit_rate"], 1 / 3)
        self.assertEqual(metrics["future_positive_eval_samples"], 3.0)


class RunnerTest(unittest.TestCase):
    def test_validation_checkpoint_adds_training_summary(self) -> None:
        config = {
            "experiment": {"seed": 3},
            "data": _synthetic_config(),
            "model": {"type": "mean", "embedding_dim": 8, "hidden_dim": 16, "dropout": 0.0},
            "training": {
                "epochs": 2,
                "learning_rate": 0.003,
                "weight_decay": 0.0,
                "negatives_per_positive": 1,
                "use_validation_checkpoint": True,
                "validation_interval": 1,
            },
            "sampling": {
                "max_attempts": 100,
                "samplers": ["random"],
            },
        }
        metrics = run_mvp_experiment(config)
        random_metrics = metrics["samplers"]["random"]
        self.assertIn("best_epoch", random_metrics)
        self.assertIn("val_aupr", random_metrics)
        self.assertIn("hard_negative_ratio", random_metrics)
        self.assertIn("future_positive_hit_rate", random_metrics)

    def test_model_aware_rerank_sampler_runs(self) -> None:
        config = {
            "experiment": {"seed": 5},
            "data": _synthetic_config(),
            "model": {"type": "mean", "embedding_dim": 8, "hidden_dim": 16, "dropout": 0.0},
            "training": {
                "epochs": 1,
                "learning_rate": 0.003,
                "weight_decay": 0.0,
                "negatives_per_positive": 1,
                "use_validation_checkpoint": False,
            },
            "sampling": {
                "max_attempts": 100,
                "anchor_ratio": 0.25,
                "nearest_positive_upper_bound": 1.0,
                "closure_risk_upper_bound": 1.0,
                "use_risk_budget": True,
                "risk_budget": 4.0,
                "rerank_candidate_multiplier": 2,
                "rerank_selection_strategy": "topk_sample",
                "rerank_top_k": 2,
                "samplers": ["model_aware_risk_controlled"],
            },
        }
        metrics = run_mvp_experiment(config)
        rerank_metrics = metrics["samplers"]["model_aware_risk_controlled"]
        self.assertIn("rerank_candidate_multiplier", rerank_metrics)
        self.assertIn("rerank_top_k", rerank_metrics)
        self.assertIn("rerank_selected_score_mean", rerank_metrics)

    def test_fast_safe_sampler_outputs_non_positive_edges(self) -> None:
        dataset = generate_synthetic_hypergraph(_synthetic_config(), seed=11)
        sampler = build_sampler(
            "fast_safe",
            {
                "max_attempts": 100,
                "fast_safe_anchor_ratio": 0.5,
                "fast_safe_source_jaccard_upper_bound": 0.85,
            },
            min_edge_size=2,
            max_edge_size=3,
        )
        batch = sampler.sample(
            source_edges=dataset.train_edges[:4],
            num_nodes=dataset.num_nodes,
            positive_edges=dataset.positive_edges,
            rng=random.Random(11),
            negatives_per_positive=2,
        )
        self.assertEqual(len(batch.edges), 8)
        self.assertTrue(all(edge not in dataset.positive_edges for edge in batch.edges))
        self.assertIn("fast_safe_source_jaccard_mean", batch.metadata)
        self.assertLessEqual(batch.metadata["fast_safe_source_jaccard_mean"], 0.85)
    def test_bipartite_backbone_runs(self) -> None:
        config = {
            "experiment": {"seed": 8},
            "data": _synthetic_config(),
            "model": {
                "type": "bipartite",
                "embedding_dim": 8,
                "hidden_dim": 16,
                "dropout": 0.0,
                "message_passing_layers": 1,
                "use_residual": True,
                "use_layer_norm": True,
            },
            "training": {
                "epochs": 1,
                "learning_rate": 0.003,
                "weight_decay": 0.0,
                "negatives_per_positive": 1,
                "use_validation_checkpoint": False,
            },
            "sampling": {
                "max_attempts": 100,
                "samplers": ["random"],
            },
        }
        metrics = run_mvp_experiment(config)
        random_metrics = metrics["samplers"]["random"]
        self.assertIn("auc", random_metrics)
        self.assertIn("aupr", random_metrics)

    def test_curriculum_sampling_config_switches_rerank_settings(self) -> None:
        sampling_config = {
            "rerank_candidate_multiplier": 8,
            "rerank_selection_strategy": "top_score",
            "rerank_target_score": 0.5,
            "rerank_target_score_min": 0.5,
            "rerank_target_score_max": 0.5,
            "rerank_top_k": 1,
            "curriculum": {
                "enabled": True,
                "warmup_epochs": 2,
                "warmup_candidate_multiplier": 2,
                "warmup_selection_strategy": "topk_sample",
                "warmup_top_k": 2,
                "warmup_target_score_min": 0.2,
                "warmup_target_score_max": 0.5,
                "target_candidate_multiplier": 8,
                "target_selection_strategy": "top_score",
                "target_top_k": 1,
                "target_target_score_min": 0.3,
                "target_target_score_max": 0.7,
            },
        }
        warmup_config = _effective_sampling_config(sampling_config, epoch=1)
        target_config = _effective_sampling_config(sampling_config, epoch=3)
        self.assertEqual(warmup_config["rerank_candidate_multiplier"], 2)
        self.assertEqual(warmup_config["rerank_selection_strategy"], "topk_sample")
        self.assertEqual(warmup_config["rerank_target_score_min"], 0.2)
        self.assertEqual(warmup_config["rerank_target_score_max"], 0.5)
        self.assertEqual(warmup_config["curriculum_phase"], "warmup")
        self.assertEqual(target_config["rerank_candidate_multiplier"], 8)
        self.assertEqual(target_config["rerank_selection_strategy"], "top_score")
        self.assertEqual(target_config["rerank_target_score_min"], 0.3)
        self.assertEqual(target_config["rerank_target_score_max"], 0.7)
        self.assertEqual(target_config["curriculum_phase"], "target")

    def test_adaptive_hardness_overrides_curriculum_target(self) -> None:
        sampling_config = {
            "rerank_target_score": 0.5,
            "rerank_score_lower_bound": 0.3,
            "rerank_score_upper_bound": 0.7,
            "curriculum": {
                "enabled": True,
                "warmup_epochs": 0,
                "target_target_score_min": 0.3,
                "target_target_score_max": 0.7,
            },
            "adaptive_hardness": {
                "enabled": True,
                "initial_center": 0.4,
                "target_width": 0.1,
                "min_center": 0.3,
                "max_center": 0.5,
            },
        }
        state = _build_adaptive_hardness_state(sampling_config)
        self.assertIsNotNone(state)
        effective_config = _effective_sampling_config(sampling_config, epoch=1, adaptive_hardness_state=state)
        self.assertEqual(effective_config["rerank_target_score"], 0.4)
        self.assertAlmostEqual(effective_config["rerank_target_score_min"], 0.35)
        self.assertAlmostEqual(effective_config["rerank_target_score_max"], 0.45)

    def test_adaptive_hardness_controller_updates_from_constraints(self) -> None:
        sampling_config = {
            "adaptive_hardness": {
                "enabled": True,
                "initial_center": 0.4,
                "target_width": 0.1,
                "min_center": 0.3,
                "max_center": 0.5,
                "step_size": 0.05,
                "min_val_delta": -0.001,
                "max_risk_fallback_rate": 0.2,
                "min_in_band_rate": 0.15,
            }
        }
        state = _build_adaptive_hardness_state(sampling_config)
        self.assertIsNotNone(state)
        _update_adaptive_hardness_state(
            state,
            val_aupr=0.70,
            negative_metadata={"risk_fallback_rate": 0.0, "rerank_selected_in_band_rate": 0.20},
        )
        _update_adaptive_hardness_state(
            state,
            val_aupr=0.701,
            negative_metadata={"risk_fallback_rate": 0.0, "rerank_selected_in_band_rate": 0.20},
        )
        self.assertAlmostEqual(state.center, 0.45)
        self.assertEqual(state.increase_updates, 1)

        _update_adaptive_hardness_state(
            state,
            val_aupr=0.69,
            negative_metadata={"risk_fallback_rate": 0.5, "rerank_selected_in_band_rate": 0.10},
        )
        self.assertAlmostEqual(state.center, 0.4)
        self.assertEqual(state.decrease_updates, 1)

    def test_proposal_mixture_controller_updates_from_constraints(self) -> None:
        sampling_config = {
            "replacement_strategy": "calibrated_mixture",
            "mixture_risk_aware_probability": 0.2,
            "proposal_controller": {
                "enabled": True,
                "min_probability": 0.0,
                "max_probability": 1.0,
                "step_size": 0.5,
                "target_in_band_rate": 0.4,
                "max_risk_fallback_rate": 0.3,
                "max_sampling_attempts_mean": 5.0,
                "min_candidate_pool_size": 1.0,
            },
        }
        state = _build_proposal_mixture_state(sampling_config)
        self.assertIsNotNone(state)
        assert state is not None
        _update_proposal_mixture_state(
            state,
            {
                "rerank_selected_in_band_rate": 0.8,
                "risk_fallback_rate": 0.6,
                "sampling_attempts_mean": 3.0,
                "candidate_pool_size_mean": 2.0,
            },
        )
        self.assertGreater(state.probability, 0.2)
        increased_probability = state.probability
        _update_proposal_mixture_state(
            state,
            {
                "rerank_selected_in_band_rate": 0.1,
                "risk_fallback_rate": 0.1,
                "sampling_attempts_mean": 8.0,
                "candidate_pool_size_mean": 0.5,
            },
        )
        self.assertLess(state.probability, increased_probability)
        self.assertEqual(state.increase_updates, 1)
        self.assertEqual(state.decrease_updates, 1)

    def test_residual_safe_controller_updates_from_hardness_and_risk(self) -> None:
        sampling_config = {
            "replacement_strategy": "calibrated_mixture",
            "mixture_residual_safe_probability": 0.2,
            "residual_safe_controller": {
                "enabled": True,
                "min_probability": 0.0,
                "max_probability": 0.6,
                "step_size": 0.5,
                "target_in_band_rate": 0.4,
                "max_risk_fallback_rate": 0.3,
                "max_sampling_attempts_mean": 5.0,
                "min_candidate_pool_size": 1.0,
                "min_val_delta": -0.01,
            },
        }
        state = _build_residual_safe_mixture_state(sampling_config)
        self.assertIsNotNone(state)
        assert state is not None
        _update_residual_safe_mixture_state(
            state,
            {
                "rerank_selected_in_band_rate": 0.1,
                "risk_fallback_rate": 0.1,
                "sampling_attempts_mean": 3.0,
                "candidate_pool_size_mean": 2.0,
            },
            val_aupr=0.70,
        )
        self.assertGreater(state.probability, 0.2)
        increased_probability = state.probability
        _update_residual_safe_mixture_state(
            state,
            {
                "rerank_selected_in_band_rate": 0.1,
                "risk_fallback_rate": 0.8,
                "sampling_attempts_mean": 8.0,
                "candidate_pool_size_mean": 0.5,
            },
            val_aupr=0.65,
        )
        self.assertLess(state.probability, increased_probability)
        self.assertEqual(state.increase_updates, 1)
        self.assertEqual(state.decrease_updates, 1)

    def test_weighted_negative_loss_downweights_high_risk_edges(self) -> None:
        dataset = HypergraphDataset(
            num_nodes=12,
            train_edges=[(0, 1, 2), (0, 1, 3), (6, 7, 8)],
            val_edges=[(3, 4, 5)],
            test_edges=[(8, 9, 10)],
        )
        config = {
            "training": {
                "weighted_negative_loss": {
                    "enabled": True,
                    "beta": 1.0,
                    "min_weight": 0.01,
                }
            },
            "sampling": {
                "risk_budget": 2.0,
                "nearest_positive_upper_bound": 0.25,
                "closure_risk_upper_bound": 10.0,
                "use_cowalk_risk": False,
                "use_hitting_risk": False,
                "use_residual_risk": False,
                "nearest_risk_weight": 4.0,
                "closure_risk_weight": 1.0,
            },
        }
        weighter = _build_negative_loss_weighter(config, dataset)
        self.assertIsNotNone(weighter)
        assert weighter is not None
        source_edge = dataset.train_edges[2]
        risky_edge = dataset.train_edges[0]
        low_risk_edge = (9, 10, 11)
        batch = NegativeSampleBatch(
            edges=[risky_edge, low_risk_edge],
            source_edges=[source_edge, source_edge],
        )
        weights = _loss_weights_for_batch(
            positive_count=1,
            negative_batch=batch,
            negative_loss_weighter=weighter,
        )
        self.assertEqual(float(weights[0]), 1.0)
        self.assertLess(float(weights[1]), float(weights[2]))

    def test_gce_negative_label_loss_is_bounded_for_noisy_negative(self) -> None:
        config = {
            "training": {
                "negative_label_loss": {
                    "type": "gce",
                    "q": 0.7,
                }
            },
            "sampling": {},
        }
        objective = _build_negative_label_objective(config)
        criterion = torch.nn.BCEWithLogitsLoss(reduction="none")
        labels = torch.tensor([1.0, 0.0])
        weights = torch.ones(2)
        logits = torch.tensor([0.0, 8.0])
        gce_loss = _training_loss(
            logits=logits,
            labels=labels,
            positive_count=1,
            loss_weights=weights,
            criterion=criterion,
            objective=objective,
        )
        bce_objective = _build_negative_label_objective({"training": {}, "sampling": {}})
        bce_loss = _training_loss(
            logits=logits,
            labels=labels,
            positive_count=1,
            loss_weights=weights,
            criterion=criterion,
            objective=bce_objective,
        )
        self.assertLess(float(gce_loss), float(bce_loss))
        self.assertEqual(objective.summary()["negative_label_loss_gce_enabled"], 1.0)

    def test_nnpu_negative_label_loss_uses_risk_calibrated_prior(self) -> None:
        config = {
            "training": {
                "negative_label_loss": {
                    "type": "nnpu",
                    "positive_prior_max": 0.8,
                }
            },
            "sampling": {},
        }
        objective = _build_negative_label_objective(config)
        criterion = torch.nn.BCEWithLogitsLoss(reduction="none")
        labels = torch.tensor([1.0, 0.0, 0.0])
        weights = torch.tensor([1.0, 0.25, 0.75])
        logits = torch.tensor([2.0, -2.0, -1.0])
        loss = _training_loss(
            logits=logits,
            labels=labels,
            positive_count=1,
            loss_weights=weights,
            criterion=criterion,
            objective=objective,
        )
        positive_only = criterion(logits[:1], torch.ones(1)).mean()
        self.assertGreaterEqual(float(loss), float(positive_only))
        self.assertAlmostEqual(objective.summary()["negative_label_loss_positive_prior"], 0.5, places=6)
        self.assertEqual(objective.summary()["negative_label_loss_nnpu_enabled"], 1.0)

    def test_runner_records_adaptive_risk_calibration(self) -> None:
        config = {
            "experiment": {"seed": 6},
            "data": _synthetic_config(),
            "model": {"type": "mean", "embedding_dim": 8, "hidden_dim": 16, "dropout": 0.0},
            "training": {
                "epochs": 1,
                "learning_rate": 0.003,
                "weight_decay": 0.0,
                "negatives_per_positive": 1,
                "use_validation_checkpoint": False,
            },
            "sampling": {
                "max_attempts": 100,
                "anchor_ratio": 0.25,
                "nearest_positive_upper_bound": 1.0,
                "closure_risk_upper_bound": 1.0,
                "use_risk_budget": True,
                "risk_budget": 4.0,
                "adaptive_risk": {
                    "enabled": True,
                    "max_source_edges": 10,
                    "candidates_per_edge": 2,
                    "scale_quantile": 0.5,
                    "budget_quantile": 0.2,
                },
                "samplers": ["risk_controlled"],
            },
        }
        metrics = run_mvp_experiment(config)
        sampler_metrics = metrics["samplers"]["risk_controlled"]
        self.assertIn("adaptive_calibration_candidates", sampler_metrics)
        self.assertIn("adaptive_risk_budget", sampler_metrics)


class ScoringTest(unittest.TestCase):
    def test_model_aware_reranker_topk_sample_keeps_group_shape(self) -> None:
        batch = NegativeSampleBatch(
            edges=[(0, 1), (0, 2), (3, 4), (3, 5)],
            source_edges=[(0, 9), (0, 9), (3, 9), (3, 9)],
        )
        scorer = _StaticScorer([0.1, 0.9, 0.8, 0.2])
        reranker = ModelAwareReranker(RerankConfig(candidate_multiplier=2, selection_strategy="topk_sample", top_k=2))
        selected = reranker.select(batch, scorer, num_sources=2, negatives_per_positive=1, rng=random.Random(3))
        self.assertEqual(len(selected.edges), 2)
        self.assertEqual(len(selected.source_edges), 2)
        self.assertIn("rerank_selected_score_mean", selected.metadata)

    def test_model_aware_reranker_boundary_closest_prefers_target_band(self) -> None:
        batch = NegativeSampleBatch(
            edges=[(0, 1), (0, 2), (0, 3), (4, 5), (4, 6), (4, 7)],
            source_edges=[(0, 9), (0, 9), (0, 9), (4, 9), (4, 9), (4, 9)],
        )
        scorer = _StaticScorer([0.95, 0.52, 0.25, 0.1, 0.66, 0.48])
        reranker = ModelAwareReranker(
            RerankConfig(
                candidate_multiplier=3,
                selection_strategy="boundary_closest",
                target_score=0.5,
                score_lower_bound=0.3,
                score_upper_bound=0.7,
            )
        )
        selected = reranker.select(batch, scorer, num_sources=2, negatives_per_positive=1, rng=random.Random(3))
        self.assertEqual(selected.edges, [(0, 2), (4, 7)])
        self.assertEqual(selected.metadata["rerank_selected_in_band_rate"], 1.0)
        self.assertEqual(selected.metadata["rerank_selection_strategy"], "boundary_closest")

    def test_model_aware_reranker_records_target_range(self) -> None:
        batch = NegativeSampleBatch(
            edges=[(0, 1), (0, 2), (0, 3)],
            source_edges=[(0, 9), (0, 9), (0, 9)],
        )
        scorer = _StaticScorer([0.31, 0.49, 0.69])
        reranker = ModelAwareReranker(
            RerankConfig(
                candidate_multiplier=3,
                selection_strategy="boundary_closest",
                target_score=0.5,
                target_score_min=0.3,
                target_score_max=0.7,
                score_lower_bound=0.3,
                score_upper_bound=0.7,
            )
        )
        selected = reranker.select(batch, scorer, num_sources=1, negatives_per_positive=1, rng=random.Random(9))
        self.assertEqual(len(selected.edges), 1)
        self.assertEqual(selected.metadata["rerank_target_score_min"], 0.3)
        self.assertEqual(selected.metadata["rerank_target_score_max"], 0.7)
        self.assertGreaterEqual(selected.metadata["rerank_sampled_target_score_mean"], 0.3)
        self.assertLessEqual(selected.metadata["rerank_sampled_target_score_mean"], 0.7)

    def test_embedding_boundary_reranker_prefers_embedding_anchor_when_scores_tie(self) -> None:
        batch = NegativeSampleBatch(
            edges=[(0, 1), (0, 2)],
            source_edges=[(0, 9), (0, 9)],
        )
        scorer = _EmbeddingScorer(
            scores=[0.5, 0.5],
            embeddings={
                (0, 1): [1.0, 0.0],
                (0, 2): [0.0, 1.0],
                (0, 9): [0.0, 1.0],
            },
        )
        reranker = ModelAwareReranker(
            RerankConfig(
                candidate_multiplier=2,
                selection_strategy="embedding_boundary_closest",
                target_score=0.5,
                score_lower_bound=0.3,
                score_upper_bound=0.7,
                embedding_similarity_weight=0.1,
            )
        )
        selected = reranker.select(batch, scorer, num_sources=1, negatives_per_positive=1, rng=random.Random(4))
        self.assertEqual(selected.edges, [(0, 2)])
        self.assertIn("embedding_source_similarity_mean", selected.metadata)
        self.assertGreater(selected.metadata["embedding_source_similarity_mean"], 0.9)



class ReportingTest(unittest.TestCase):
    def test_append_experiment_log_writes_markdown_entry(self) -> None:
        metrics = {
            "dataset": {
                "source": "synthetic",
                "num_nodes": 10,
                "train_edges": 6,
                "val_edges": 2,
                "test_edges": 2,
            },
            "samplers": {
                "random": {
                    "auc": 0.5,
                    "aupr": 0.5,
                    "hardness_mean": 0.4,
                    "jaccard_mean": 0.1,
                    "nearest_positive_similarity_mean": 0.2,
                }
            },
        }
        with tempfile.TemporaryDirectory() as tmp_dir:
            log_path = Path(tmp_dir) / "experiment_log.md"
            append_experiment_log(log_path, "mvp_synthetic", "config.yaml", "results/run", metrics)
            text = log_path.read_text(encoding="utf-8")
            self.assertIn("mvp_synthetic", text)
            self.assertIn("配置文件", text)
            self.assertIn("负采样器", text)
            self.assertIn("本次执行操作", text)
            self.assertIn("得到的结果", text)
            self.assertIn("结合研究内容的解释", text)
            self.assertIn("| random |", text)


def _synthetic_config() -> dict[str, object]:
    return {
        "source": "synthetic",
        "num_nodes": 40,
        "num_hyperedges": 80,
        "num_communities": 4,
        "min_edge_size": 3,
        "max_edge_size": 6,
        "intra_community_probability": 0.8,
        "split": {"train": 0.7, "val": 0.1, "test": 0.2},
    }


class _StaticScorer:
    def __init__(self, scores: list[float]) -> None:
        self.scores = scores

    def predict_scores(self, edges: list[tuple[int, ...]]) -> list[float]:
        return self.scores[: len(edges)]


class _EmbeddingScorer:
    def __init__(self, scores: list[float], embeddings: dict[tuple[int, ...], list[float]]) -> None:
        self.scores = scores
        self.embeddings = embeddings

    def predict_scores(self, edges: list[tuple[int, ...]]) -> list[float]:
        return self.scores[: len(edges)]

    def encode_edges(self, edges: list[tuple[int, ...]]) -> list[list[float]]:
        return [self.embeddings[edge] for edge in edges]


class _BoundaryToyScorer:
    def predict_scores(self, edges: list[tuple[int, ...]]) -> list[float]:
        return [0.5 if 4 in edge else 0.1 for edge in edges]


if __name__ == "__main__":
    unittest.main()
