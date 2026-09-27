from __future__ import annotations

try:
    from scans.samplers._frontier_accel import (
        append_unseen_neighbors,
        sample_motif_batch,
        sample_motif_one,
    )
except ImportError:
    def append_unseen_neighbors(
        neighbors: set[int],
        selected: set[int],
        frontier: set[int],
        frontier_list: list[int],
    ) -> int:
        added = 0
        for candidate in neighbors:
            if candidate not in selected and candidate not in frontier:
                frontier.add(candidate)
                frontier_list.append(candidate)
                added += 1
        return added

    def sample_motif_one(
        size: int,
        adjacency: object,
        starts: list[int],
        positive_edges: set[tuple[int, ...]],
        rng: object,
        max_attempts: int,
    ) -> tuple[int, ...] | None:
        for _ in range(max_attempts):
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
                append_unseen_neighbors(adjacency.neighbors(node), selected, frontier, frontier_list)
            if len(selected) != size:
                continue
            edge = tuple(sorted(selected))
            if edge not in positive_edges:
                return edge
        return None

    def sample_motif_batch(
        sizes: list[int],
        adjacency: object,
        starts: list[int],
        positive_edges: set[tuple[int, ...]],
        rng: object,
        max_attempts: int,
    ) -> object:
        return NotImplemented

__all__ = ["append_unseen_neighbors", "sample_motif_batch", "sample_motif_one"]
