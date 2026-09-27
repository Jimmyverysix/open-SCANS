from __future__ import annotations

import random

from scans.data.hypergraph import Hyperedge
from scans.samplers.base import NegativeSampleBatch
from scans.samplers.clique_sampler import CliqueNegativeSampler
from scans.samplers.motif_sampler import MotifNegativeSampler
from scans.samplers.size_matched_sampler import SizeMatchedSampler
from scans.samplers.two_section_index import LazyTwoSectionIndex


class MixedOfficialSampler:
    name = "mix"

    def __init__(self, max_attempts: int = 100) -> None:
        self.samplers = (
            SizeMatchedSampler(max_attempts=max_attempts),
            MotifNegativeSampler(max_attempts=max_attempts),
            CliqueNegativeSampler(max_attempts=max_attempts),
        )

    def sample(
        self,
        source_edges: list[Hyperedge],
        num_nodes: int,
        positive_edges: set[Hyperedge],
        rng: random.Random,
        negatives_per_positive: int,
    ) -> NegativeSampleBatch:
        edges: list[Hyperedge] = []
        sources: list[Hyperedge] = []
        counts = [0, 0, 0]
        size_matched, motif, clique = self.samplers
        motif_index: tuple[LazyTwoSectionIndex, list[int]] | None = None
        clique_adjacency: LazyTwoSectionIndex | None = None
        for source in source_edges:
            for _ in range(negatives_per_positive):
                sampler_index = rng.randrange(len(self.samplers))
                if sampler_index == 0:
                    negative = size_matched._sample_one(len(source), num_nodes, positive_edges, rng)
                elif sampler_index == 1:
                    if motif_index is None:
                        motif_index = motif._index(positive_edges)
                    adjacency, starts = motif_index
                    negative = motif._sample_one(len(source), adjacency, starts, positive_edges, rng)
                    if negative is None:
                        negative = size_matched._sample_one(len(source), num_nodes, positive_edges, rng)
                else:
                    if clique_adjacency is None:
                        clique_adjacency = clique._index(positive_edges)
                    negative = clique._sample_one(source, clique_adjacency, positive_edges, rng)
                    if negative is None:
                        negative = size_matched._sample_one(len(source), num_nodes, positive_edges, rng)
                edges.append(negative)
                sources.append(source)
                counts[sampler_index] += 1
        total = max(1, len(edges))
        return NegativeSampleBatch(
            edges,
            sources,
            {
                "mix_sns_rate": counts[0] / total,
                "mix_mns_rate": counts[1] / total,
                "mix_cns_rate": counts[2] / total,
            },
        )
