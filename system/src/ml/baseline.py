"""Deterministic link-prediction baselines.

ML_ENGINE.md R1.3: *"Establish a degree/common-neighbor or similar baseline;
report AUPRC, calibration, coverage, and performance by entity/source subgroup.
Use case-level splits and confidence intervals. AUC alone is not an acceptance
criterion."*

These scorers have no parameters, no training step and no randomness. That is
the point: they are the reference a learned model has to beat, and a reference
that can drift is worthless. The same code path scores the baseline and every
future candidate model's inputs, so a comparison is always apples to apples.

**Normalisation is structural, never fitted.** Raw counts are rescaled to
``[0, 1]`` by dividing by the maximum raw value attainable anywhere in the
snapshot. That maximum depends only on the graph's own structure — it is not
derived from labels — so no information about train or test outcomes leaks
into the score's scale. Fitting a scaler on the evaluation set would have been
the easy way to get impressive numbers; it would also have been leakage.

Each maximum is cached by snapshot hash, so a second call on the same graph
reuses the identical constant rather than recomputing a subtly different one.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

from .snapshot import GraphSnapshot

__all__ = [
    "BASELINE_SCORERS",
    "DEFAULT_BASELINE",
    "Scorer",
    "available_scorers",
    "describe",
    "score",
    "score_many",
]


def _ratio(numerator: float, denominator: float) -> float:
    if denominator <= 0.0:
        return 0.0
    return numerator / denominator


def _common_neighbours(snapshot: GraphSnapshot, left: str, right: str) -> float:
    return float(len(snapshot.common_neighbours(left, right)))


def _jaccard(snapshot: GraphSnapshot, left: str, right: str) -> float:
    neighbours_left = snapshot.neighbours(left)
    neighbours_right = snapshot.neighbours(right)
    union = neighbours_left | neighbours_right
    if not union:
        return 0.0
    return len(neighbours_left & neighbours_right) / len(union)


def _adamic_adar(snapshot: GraphSnapshot, left: str, right: str) -> float:
    total = 0.0
    for node in snapshot.common_neighbours(left, right):
        degree = snapshot.degree(node)
        if degree > 1:
            total += 1.0 / math.log(degree)
    return total


def _resource_allocation(snapshot: GraphSnapshot, left: str, right: str) -> float:
    total = 0.0
    for node in snapshot.common_neighbours(left, right):
        degree = snapshot.degree(node)
        if degree > 0:
            total += 1.0 / degree
    return total


def _degree_product(snapshot: GraphSnapshot, left: str, right: str) -> float:
    return float(snapshot.degree(left) * snapshot.degree(right))


def _degree_sum(snapshot: GraphSnapshot, left: str, right: str) -> float:
    return float(snapshot.degree(left) + snapshot.degree(right))


@dataclass(frozen=True)
class Scorer:
    """A named structural scorer plus the invariant it relies on."""

    name: str
    raw: Callable[[GraphSnapshot, str, str], float]
    summary: str


BASELINE_SCORERS: dict[str, Scorer] = {
    scorer.name: scorer
    for scorer in (
        Scorer(
            name="common_neighbours",
            raw=_common_neighbours,
            summary="count of shared neighbours; the primary R1.3 baseline",
        ),
        Scorer(
            name="jaccard",
            raw=_jaccard,
            summary="shared neighbours over shared neighbourhood union",
        ),
        Scorer(
            name="adamic_adar",
            raw=_adamic_adar,
            summary="shared neighbours weighted by inverse log degree",
        ),
        Scorer(
            name="resource_allocation",
            raw=_resource_allocation,
            summary="shared neighbours weighted by inverse degree",
        ),
        Scorer(
            name="degree_product",
            raw=_degree_product,
            summary="preferential attachment; ignores neighbourhood overlap",
        ),
        Scorer(
            name="degree_sum",
            raw=_degree_sum,
            summary="endpoint degree sum; the weakest structural signal",
        ),
    )
}

#: The reference model every learned model is measured against (R1.3).
DEFAULT_BASELINE: str = "common_neighbours"


def available_scorers() -> tuple[str, ...]:
    return tuple(sorted(BASELINE_SCORERS))


def describe() -> dict[str, str]:
    return {name: scorer.summary for name, scorer in sorted(BASELINE_SCORERS.items())}


def _pairs(snapshot: GraphSnapshot) -> list[tuple[str, str]]:
    nodes = snapshot.nodes
    return [
        (nodes[i], nodes[j])
        for i in range(len(nodes))
        for j in range(i + 1, len(nodes))
    ]


#: Per-snapshot score ceilings, keyed by snapshot hash. Two snapshots with the
#: same hash have identical structure by construction, so sharing a ceiling is
#: correct; the snapshot object itself is deliberately not a cache key because
#: it holds unhashable mappings.
_MAXIMA_CACHE: dict[str, dict[str, float]] = {}


def _maxima(snapshot: GraphSnapshot) -> dict[str, float]:
    """Largest raw score each scorer attains anywhere in this snapshot.

    Computed over every node pair, without reference to any label. Used as the
    structural denominator that rescales raw counts into ``[0, 1]``.
    """
    cached = _MAXIMA_CACHE.get(snapshot.hash)
    if cached is not None:
        return cached
    maxima = {name: 0.0 for name in BASELINE_SCORERS}
    for left, right in _pairs(snapshot):
        for name, scorer in BASELINE_SCORERS.items():
            value = scorer.raw(snapshot, left, right)
            if value > maxima[name]:
                maxima[name] = value
    if not any(maxima.values()):
        maxima = {name: 1.0 for name in BASELINE_SCORERS}
    _MAXIMA_CACHE[snapshot.hash] = maxima
    return maxima


def score(
    snapshot: GraphSnapshot, left: str, right: str, *, scorer: str = DEFAULT_BASELINE
) -> float:
    """Structural score in ``[0, 1]`` for one candidate pair.

    Raises ``KeyError`` if an endpoint is absent from the snapshot rather than
    returning 0.0: "unseen node" and "disconnected node" are different facts
    and must not collapse into the same number.
    """
    if scorer not in BASELINE_SCORERS:
        raise KeyError(
            f"unknown scorer {scorer!r}; available: {available_scorers()}"
        )
    if left not in snapshot.adjacency:
        raise KeyError(f"{left!r} is not in snapshot {snapshot.hash}")
    if right not in snapshot.adjacency:
        raise KeyError(f"{right!r} is not in snapshot {snapshot.hash}")
    raw_value = BASELINE_SCORERS[scorer].raw(snapshot, left, right)
    ceiling = _maxima(snapshot)[scorer]
    if ceiling <= 0.0:
        return 0.0
    normalised = raw_value / ceiling
    # Clamping absorbs float drift at the ceiling; it cannot manufacture a
    # score above what the graph structurally supports.
    return max(0.0, min(1.0, normalised))


def score_many(
    snapshot: GraphSnapshot,
    pairs: list[tuple[str, str]],
    *,
    scorer: str = DEFAULT_BASELINE,
) -> list[float]:
    """Score a batch of pairs, skipping any endpoint missing from the snapshot.

    Skipped pairs are reported by :func:`coverage`, not silently scored 0.0 —
    otherwise "no features available" would read as "structurally unlikely".
    """
    return [
        score(snapshot, left, right, scorer=scorer)
        for left, right in pairs
        if left in snapshot.adjacency and right in snapshot.adjacency
    ]


def coverage(
    snapshot: GraphSnapshot, pairs: list[tuple[str, str]]
) -> dict[str, float | int]:
    """How many candidate pairs are actually scoreable against this snapshot."""
    scorable = sum(
        1
        for left, right in pairs
        if left in snapshot.adjacency and right in snapshot.adjacency
    )
    total = len(pairs)
    return {
        "candidates": total,
        "scorable": scorable,
        "unscorable": total - scorable,
        "coverage": _ratio(scorable, total),
    }
