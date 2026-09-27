from __future__ import annotations

import random
import unittest

import torch
from torch import nn

from scripts.run_scans_benchmark import _sample_teacher_negatives
from scans.data.hypergraph import Hyperedge
from scans.samplers.anchored_safe_sampler import _PositiveEdgeIndex
from scans.samplers.base import NegativeSampleBatch, jaccard
from scans.samplers.clique_sampler import CliqueNegativeSampler
from scans.samplers.motif_sampler import MotifNegativeSampler


class RuntimeOptimizationTests(unittest.TestCase):
    def test_positive_edge_index_matches_brute_force_jaccard(self) -> None:
        positives = {
            (0, 1, 2),
            (1, 3),
            (0, 2, 4, 6),
            (4, 5, 6),
        }
        index = _PositiveEdgeIndex(positives)
        queries = [(0, 1, 4), (2, 3, 5), (7, 8), (0, 2, 4, 6)]
        exclusions: list[Hyperedge | None] = [None, (0, 1, 2), (7, 8)]

        for query in queries:
            for excluded in exclusions:
                expected = max(
                    (jaccard(query, positive) for positive in positives if positive != excluded),
                    default=0.0,
                )
                self.assertAlmostEqual(index.nearest_similarity(query, excluded_edge=excluded), expected)

    def test_cns_reuses_exact_source_candidate_pool_without_consuming_rng(self) -> None:
        positives = {
            (0, 1, 2),
            (0, 1, 3),
            (0, 2, 4),
            (1, 2, 5),
            (0, 1, 6),
            (0, 2, 7),
            (1, 2, 8),
        }
        sources = [(0, 1, 2), (0, 1, 3)]
        cached = CliqueNegativeSampler(max_attempts=100)
        adjacency = cached._index(positives)
        original_neighbors = adjacency.neighbors
        neighbor_calls = 0

        def counted_neighbors(node: int) -> set[int]:
            nonlocal neighbor_calls
            neighbor_calls += 1
            return original_neighbors(node)

        adjacency.neighbors = counted_neighbors  # type: ignore[method-assign]
        first_rng = random.Random(29)
        first = cached.sample(sources, 12, positives, first_rng, 4)
        first_state = first_rng.getstate()
        calls_after_first = neighbor_calls
        second_rng = random.Random(29)
        second = cached.sample(sources, 12, positives, second_rng, 4)

        fresh_positives = set(positives)
        fresh = CliqueNegativeSampler(max_attempts=100)
        fresh_rng = random.Random(29)
        expected = fresh.sample(sources, 12, fresh_positives, fresh_rng, 4)

        self.assertEqual(first.edges, expected.edges)
        self.assertEqual(second.edges, expected.edges)
        self.assertEqual(first_state, fresh_rng.getstate())
        self.assertEqual(second_rng.getstate(), fresh_rng.getstate())
        self.assertEqual(calls_after_first, sum(map(len, sources)))
        self.assertEqual(neighbor_calls, calls_after_first)

    def test_cns_pivot_pool_matches_reference_order_exactly(self) -> None:
        rng = random.Random(193)
        positives: set[Hyperedge] = set()
        for _ in range(120):
            size = rng.randint(2, 9)
            positives.add(tuple(sorted(rng.sample(range(36), size))))
        sources = [edge for edge in sorted(positives) if len(edge) >= 4]
        sampler = CliqueNegativeSampler(max_attempts=100)
        adjacency = sampler._index(positives)
        for source in sources:
            expected = sampler._candidate_pool_reference(source, adjacency)
            actual = sampler._candidate_pool_from_pivots(source, adjacency)
            self.assertEqual(actual, expected, source)

    def test_mns_reference_preserves_candidates_and_rng(self) -> None:
        positives = {
            (0, 1, 2, 3, 4, 5),
            (5, 6, 7),
            (8, 9, 10),
            (10, 11, 12),
        }
        sampler = MotifNegativeSampler(max_attempts=20)
        adjacency, starts = sampler._index(positives)

        def reference_sample_one(size: int, rng: random.Random) -> Hyperedge | None:
            for _ in range(sampler.max_attempts):
                selected = {rng.choice(starts)}
                frontier = set(adjacency.neighbors(next(iter(selected))))
                frontier_list = list(frontier)
                while len(selected) < size and frontier_list:
                    index = rng.randrange(len(frontier_list))
                    node = frontier_list[index]
                    frontier_list[index] = frontier_list[-1]
                    frontier_list.pop()
                    frontier.remove(node)
                    selected.add(node)
                    for candidate in adjacency.neighbors(node):
                        if candidate not in selected and candidate not in frontier:
                            frontier.add(candidate)
                            frontier_list.append(candidate)
                if len(selected) == size:
                    edge = tuple(sorted(selected))
                    if edge not in positives:
                        return edge
            return None

        expected_rng = random.Random(71)
        actual_rng = random.Random(71)
        expected = [reference_sample_one(size, expected_rng) for size in (3, 5, 2, 4, 3)]
        actual = [
            sampler._sample_one(size, adjacency, starts, positives, actual_rng)
            for size in (3, 5, 2, 4, 3)
        ]
        self.assertEqual(actual, expected)
        self.assertEqual(actual_rng.getstate(), expected_rng.getstate())

    def test_mns_native_batch_preserves_candidates_and_rng(self) -> None:
        positives = {
            (0, 1, 2, 3, 4, 5),
            (0, 6, 7, 8, 9),
            (3, 10, 11, 12),
            (5, 13, 14, 15),
        }
        sources = [(0, 1, 2, 3), (5, 6, 7), (10, 11)]
        expected_sampler = MotifNegativeSampler(max_attempts=100)
        adjacency, starts = expected_sampler._index(positives)
        expected_rng = random.Random(207)
        expected: list[Hyperedge] = []
        for source in sources:
            for _ in range(4):
                edge = expected_sampler._sample_one(
                    len(source), adjacency, starts, positives, expected_rng
                )
                self.assertIsNotNone(edge)
                expected.append(edge)  # type: ignore[arg-type]

        actual_rng = random.Random(207)
        actual = MotifNegativeSampler(max_attempts=100).sample(
            sources,
            num_nodes=16,
            positive_edges=positives,
            rng=actual_rng,
            negatives_per_positive=4,
        )
        self.assertEqual(actual.edges, expected)
        self.assertEqual(actual_rng.getstate(), expected_rng.getstate())
        self.assertEqual(actual.metadata["mns_fallback_rate"], 0.0)

class _CountingEdgeModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.classifier = nn.Identity()
        self.encode_calls = 0

    def encode_all_nodes(self) -> torch.Tensor:
        self.encode_calls += 1
        return torch.arange(12, dtype=torch.float32).unsqueeze(1)

    def _encode_edge_batch(self, edges: list[Hyperedge], nodes: torch.Tensor) -> torch.Tensor:
        return torch.tensor(
            [[sum(edge) / len(edge)] for edge in edges],
            dtype=nodes.dtype,
            device=nodes.device,
        )

    def edge_representations(
        self,
        edges: list[Hyperedge],
        nodes: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self._encode_edge_batch(edges, self.encode_all_nodes() if nodes is None else nodes)

    def logits_from_edge_representations(self, representations: torch.Tensor) -> torch.Tensor:
        return self.classifier(representations).squeeze(-1)

    def predict_scores(self, edges: list[Hyperedge]) -> list[float]:
        raise AssertionError("teacher candidates must be scored in one batched pass")


class TeacherBatchingTests(unittest.TestCase):
    def test_teacher_candidates_share_one_node_encoding(self) -> None:
        sources = [(0, 1), (2, 3)]
        candidates = [(0, 2), (8, 9), (1, 3), (9, 10)]
        candidate_batch = NegativeSampleBatch(
            edges=candidates,
            source_edges=[sources[0], sources[0], sources[1], sources[1]],
        )
        model = _CountingEdgeModel()

        selected = _sample_teacher_negatives(
            config={"sampling": {}, "scans": {"teacher_mode": "safe_hard"}},
            mode="scans_full",
            safe_sampler=None,  # type: ignore[arg-type]
            sns_sampler=None,  # type: ignore[arg-type]
            model=model,  # type: ignore[arg-type]
            source_edges=sources,
            num_nodes=12,
            positive_edges=set(sources),
            rng=random.Random(1),
            negatives_per_positive=1,
            candidate_batch=candidate_batch,
            block_size=2,
        )

        self.assertEqual(selected.edges, [(8, 9), (9, 10)])
        self.assertEqual(selected.source_edges, sources)
        self.assertEqual(model.encode_calls, 1)


if __name__ == "__main__":
    unittest.main()
