from __future__ import annotations

import random

from scans.data.hypergraph import Hyperedge
from scans.samplers.base import NegativeSampleBatch


class HalfCorruptionSampler:
    """NHP negative sampler that retains half of each positive hyperedge."""

    name = "half_corruption"

    def __init__(self, max_attempts: int = 100) -> None:
        self.max_attempts = int(max_attempts)

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
        retained_fractions: list[float] = []
        for source in source_edges:
            keep_count = (len(source) + 1) // 2
            replace_count = len(source) - keep_count
            source_set = set(source)
            if num_nodes - len(source_set) < replace_count:
                raise RuntimeError(
                    f"cannot half-corrupt a size-{len(source)} edge with "
                    f"{num_nodes} total nodes"
                )
            for _ in range(negatives_per_positive):
                candidate = None
                for _ in range(self.max_attempts):
                    retained = rng.sample(source, keep_count)
                    replacements: set[int] = set()
                    while len(replacements) < replace_count:
                        node = rng.randrange(num_nodes)
                        if node not in source_set:
                            replacements.add(node)
                    proposal = tuple(sorted((*retained, *replacements)))
                    if proposal not in positive_edges:
                        candidate = proposal
                        break
                if candidate is None:
                    raise RuntimeError(
                        f"failed to half-corrupt edge {source} after "
                        f"{self.max_attempts} attempts"
                    )
                edges.append(candidate)
                sources.append(source)
                retained_fractions.append(keep_count / len(source))
        mean_retained = (
            sum(retained_fractions) / len(retained_fractions)
            if retained_fractions
            else 0.0
        )
        return NegativeSampleBatch(
            edges,
            sources,
            {
                "half_corruption_retained_fraction": mean_retained,
                "half_corruption_rate": 1.0,
            },
        )
