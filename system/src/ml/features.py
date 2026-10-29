"""Leakage-resistant structural features for candidate edges.

ML_ENGINE.md R1.2 requires a feature dataset built *"from a fixed graph
snapshot"*, with *"future data, outcome-derived fields, and features that
encode the target edge"* excluded.

This module only ever reads a :class:`~ml.snapshot.GraphSnapshot`, which is
what enforces the third rule. A snapshot built with ``withheld_edge_ids``
already has the withheld edge removed from its adjacency, so
:meth:`GraphSnapshot.common_neighbours` cannot count the answer. Nothing here
touches ``confidence``, ``contradicting_evidence`` or hypothesis factors — all
of those are outcome-derived (they encode how much the system already believes
the edge) and are excluded by design.

The features are deliberately plain structural statistics. R1.3 requires a
*"degree/common-neighbor or similar baseline"* to exist before any neural model
is considered, and the same vector serves both: whatever a GNN is measured
against must be a real, inspectable signal, not a strawman.
"""

from __future__ import annotations

import math
from typing import Any, Sequence

from .contracts import FEATURE_SCHEMA_VERSION, FeatureVector
from .snapshot import GraphSnapshot

__all__ = [
    "EDGE_STRUCTURAL_FEATURES",
    "NODE_STRUCTURAL_FEATURES",
    "pair_features",
    "structural_features_for",
]

#: Features computed for a candidate *pair* (both endpoints in the snapshot).
EDGE_STRUCTURAL_FEATURES: tuple[str, ...] = (
    "degree_product",
    "degree_min",
    "degree_sum",
    "common_neighbours",
    "jaccard",
    "adamic_adar",
    "resource_allocation",
    "shortest_path_length",
)

#: Features computed for a single node, used for subgroup/error analysis.
NODE_STRUCTURAL_FEATURES: tuple[str, ...] = (
    "degree",
    "clustering_coefficient",
)


def _ratio(numerator: float, denominator: float) -> float:
    if denominator <= 0.0:
        return 0.0
    return numerator / denominator


def _clustering(snapshot: GraphSnapshot, node: str) -> float:
    """Local clustering coefficient; 0.0 when the node has < 2 neighbours."""
    neighbours = snapshot.neighbours(node)
    size = len(neighbours)
    if size < 2:
        return 0.0
    links = 0
    ordered = sorted(neighbours)
    for index, left in enumerate(ordered):
        for right in ordered[index + 1 :]:
            if snapshot.are_linked(left, right):
                links += 1
    possible = size * (size - 1) / 2
    return _ratio(links, possible)


def _shortest_path(snapshot: GraphSnapshot, left: str, right: str) -> float:
    """Hop distance between two nodes; returns 0.0 when unreachable.

    Unreachable is encoded as 0.0 rather than infinity so the feature vector
    stays finite, and ``is_connected`` (below) records which case occurred so
    a consumer never has to guess.
    """
    if left == right:
        return 0.0
    if snapshot.are_linked(left, right):
        return 1.0
    frontier = [left]
    visited = {left}
    depth = 0
    limit = max(8, snapshot.n_nodes)
    while frontier and depth < limit:
        depth += 1
        following: list[str] = []
        for node in frontier:
            for neighbour in snapshot.neighbours(node):
                if neighbour in visited:
                    continue
                if neighbour == right:
                    return float(depth)
                visited.add(neighbour)
                following.append(neighbour)
        frontier = following
    return 0.0


def pair_features(snapshot: GraphSnapshot, left: str, right: str) -> FeatureVector:
    """Structural features for the candidate pair ``(left, right)``.

    Both endpoints must already exist in the snapshot. If either is missing
    the features are undefined — silently returning zeros would make an unseen
    node look like an isolated one, which is a measurement, not a default.
    """
    if left not in snapshot.adjacency:
        raise KeyError(f"{left!r} is not in snapshot {snapshot.hash}")
    if right not in snapshot.adjacency:
        raise KeyError(f"{right!r} is not in snapshot {snapshot.hash}")

    left_deg = snapshot.degree(left)
    right_deg = snapshot.degree(right)
    common = snapshot.common_neighbours(left, right)

    adamic_adar = 0.0
    resource_allocation = 0.0
    for node in common:
        degree = snapshot.degree(node)
        if degree > 1:
            adamic_adar += 1.0 / math.log(degree)
            resource_allocation += 1.0 / degree

    union = len(snapshot.neighbours(left) | snapshot.neighbours(right))
    jaccard = _ratio(len(common), union)

    values: tuple[tuple[str, float], ...] = (
        ("degree_product", float(left_deg * right_deg)),
        ("degree_min", float(min(left_deg, right_deg))),
        ("degree_sum", float(left_deg + right_deg)),
        ("common_neighbours", float(len(common))),
        ("jaccard", float(jaccard)),
        ("adamic_adar", float(adamic_adar)),
        ("resource_allocation", float(resource_allocation)),
        ("shortest_path_length", _shortest_path(snapshot, left, right)),
    )

    return FeatureVector(
        schema_version=FEATURE_SCHEMA_VERSION,
        snapshot_hash=snapshot.hash,
        subject=(left, right),
        values=values,
        source_record_ids=snapshot.edge_ids,
    )


def structural_features_for(
    snapshot: GraphSnapshot, node: str
) -> dict[str, float]:
    """Single-node structural features, used for subgroup error analysis."""
    if node not in snapshot.adjacency:
        raise KeyError(f"{node!r} is not in snapshot {snapshot.hash}")
    return {
        "degree": float(snapshot.degree(node)),
        "clustering_coefficient": _clustering(snapshot, node),
    }


def vectors_to_matrix(
    vectors: Sequence[FeatureVector], names: Sequence[str] = EDGE_STRUCTURAL_FEATURES
) -> list[list[float]]:
    """Order feature vectors into a dense matrix with a fixed column order.

    The column order is declared, not derived from whichever vector arrives
    first — otherwise two runs could silently transpose their own results.
    """
    matrix: list[list[float]] = []
    for vector in vectors:
        lookup = vector.as_dict()
        missing = [name for name in names if name not in lookup]
        if missing:
            raise ValueError(
                f"feature vector {vector.subject} is missing columns {missing}"
            )
        matrix.append([float(lookup[name]) for name in names])
    return matrix


def normalise(matrix: list[list[float]]) -> list[list[float]]:
    """Min-max scale each column using its own observed range.

    A constant column scales to 0.0, which is the correct answer: a feature
    with no variance carries no information, and dividing by zero would inject
    NaN into every row.
    """
    if not matrix:
        return []
    width = len(matrix[0])
    if any(len(row) != width for row in matrix):
        raise ValueError("matrix rows must all have the same width")
    columns = [[row[index] for row in matrix] for index in range(width)]
    lower = [min(column) for column in columns]
    upper = [max(column) for column in columns]
    out: list[list[float]] = []
    for row in matrix:
        scaled: list[float] = []
        for index, value in enumerate(row):
            span = upper[index] - lower[index]
            scaled.append(0.0 if span <= 0.0 else (value - lower[index]) / span)
        out.append(scaled)
    return out


def feature_matrix_report(vectors: Sequence[FeatureVector]) -> dict[str, Any]:
    """Describe a feature dataset without leaking its labels."""
    snapshots = {vector.snapshot_hash for vector in vectors}
    return {
        "rows": len(vectors),
        "columns": list(EDGE_STRUCTURAL_FEATURES),
        "snapshot_hashes": sorted(snapshots),
        "single_snapshot": len(snapshots) == 1,
        "schema_version": FEATURE_SCHEMA_VERSION,
    }
