"""Label inventory — the feasibility audit that gates every ML milestone.

ML_ENGINE.md Roadmap 0.3 and R1.1 make label availability a precondition, not
an afterthought:

    *"Establish a fixed, case-separated evaluation set and simple baselines
    before selecting libraries or training models."*

    *"Inventory candidate training labels, their provenance, coverage, and
    permissible use. Do not label an unobserved edge as a negative merely
    because it is absent. Use time-aware edge holdout and hard negatives only
    where absence is known."*

Two rules are enforced structurally here.

**Negatives must come from a recorded assertion of absence.** A pair that is
simply missing from ``graph_edges.json`` is not a negative — it is merely
unobserved, and treating it as one would label a potentially real relationship
as false (R1.1 explicitly forbids this). The only negative source in this
system is ``missing_edges.json``, where Stage 5 scanned the source corpus and
recorded that an *expected* relation was absent. That is an assertion of
absence with provenance, not an inference from silence.

**Positives must be observations.** Every candidate positive is checked for
``epistemic_status == "observation"``; a predicted or inferred edge can never
be a label.

Timestamps drive the time-aware holdout. The snapshot's ``created_at`` is
*not* usable for this: in this corpus it carries the run clock (three distinct
values across 121 edges), so ordering by it would silently collapse to a random
split. ``temporal_info.timestamp`` is the only field that carries event time.

The result of this module is a verdict, and the verdict can be "insufficient".
That is the intended behaviour: R1.4 permits a neural model to be attempted
*"only if sufficient independent labels exist"*, and this inventory is where
that is decided — before any model is written.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

__all__ = [
    "LabelInventory",
    "LabelledPair",
    "LabelSource",
    "MIN_POSITIVES_FOR_TIME_SPLIT",
    "NEGATIVE_SOURCE_MISSING_EDGES",
    "POSITIVE_SOURCE_OBSERVED_EDGES",
    "TRAIN_FRACTION",
    "candidate_pairs",
    "inventory_labels",
    "parse_timestamp",
]

#: Predeclared before any result is seen. Changing these after seeing metrics
#: would make the evaluation meaningless, so they live here and are reported
#: verbatim in the inventory.
TRAIN_FRACTION: float = 0.70
MIN_POSITIVES_FOR_TIME_SPLIT: int = 2

POSITIVE_SOURCE_OBSERVED_EDGES: str = "graph_edges.json (epistemic_status=observation)"
NEGATIVE_SOURCE_MISSING_EDGES: str = "missing_edges.json (Stage 5 source scan)"


@dataclass(frozen=True)
class LabelledPair:
    """One candidate edge with a known label and its provenance."""

    source_id: str
    target_id: str
    label: int
    provenance: str
    record_id: str
    timestamp: str = ""

    @property
    def key(self) -> tuple[str, str]:
        return tuple(sorted((self.source_id, self.target_id)))  # type: ignore[return-value]


@dataclass(frozen=True)
class LabelSource:
    """Where a family of labels came from and what it may be used for."""

    name: str
    kind: str
    count: int
    provenance: str
    permissible_use: str
    notes: str = ""


@dataclass(frozen=True)
class LabelInventory:
    """The audit verdict: what labels exist, and whether they can support a gate."""

    sources: tuple[LabelSource, ...]
    positives: tuple[LabelledPair, ...]
    negatives: tuple[LabelledPair, ...]
    split: tuple[tuple[str, ...], tuple[str, ...]]
    blockers: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    parameters: Mapping[str, Any] = field(default_factory=dict)

    @property
    def n_positives(self) -> int:
        return len(self.positives)

    @property
    def n_negatives(self) -> int:
        return len(self.negatives)

    @property
    def n_timestamped_positives(self) -> int:
        return sum(1 for item in self.positives if item.timestamp)

    @property
    def timestamp_coverage(self) -> float:
        """Fraction of positives that carry usable event time."""
        if not self.positives:
            return 0.0
        return self.n_timestamped_positives / len(self.positives)

    @property
    def negative_ratio(self) -> float:
        """Known-absent negatives per positive. Not a tuning knob — a coverage fact."""
        if not self.positives:
            return 0.0
        return self.n_negatives / len(self.positives)

    @property
    def gate_ready(self) -> bool:
        """True only when the inventory can support a predeclared acceptance gate."""
        return not self.blockers

    @property
    def train_edge_ids(self) -> tuple[str, ...]:
        return self.split[0]

    @property
    def test_edge_ids(self) -> tuple[str, ...]:
        return self.split[1]

    @property
    def verdict(self) -> str:
        if self.gate_ready:
            return "sufficient"
        return "insufficient"

    def to_dict(self) -> dict[str, Any]:
        return {
            "verdict": self.verdict,
            "gate_ready": self.gate_ready,
            "blockers": list(self.blockers),
            "warnings": list(self.warnings),
            "counts": {
                "positives": self.n_positives,
                "negatives": self.n_negatives,
                "timestamped_positives": self.n_timestamped_positives,
                "train": len(self.train_edge_ids),
                "test": len(self.test_edge_ids),
            },
            "coverage": {
                "timestamp_coverage": round(self.timestamp_coverage, 6),
                "negative_ratio": round(self.negative_ratio, 6),
            },
            "parameters": dict(self.parameters),
            "sources": [
                {
                    "name": s.name,
                    "kind": s.kind,
                    "count": s.count,
                    "provenance": s.provenance,
                    "permissible_use": s.permissible_use,
                    "notes": s.notes,
                }
                for s in self.sources
            ],
        }


def parse_timestamp(raw: object) -> datetime | None:
    """Parse an event timestamp, or return None when it is unusable.

    Accepts ``YYYY-MM-DD`` and ``YYYY-MM-DDTHH:MM:SS``. Anything else —
    including the ``precision: unknown`` records in this corpus — returns
    ``None`` rather than being coerced, because a guessed time would silently
    corrupt the time-aware split.
    """
    if raw is None:
        return None
    text = str(raw).strip()
    if not text:
        return None
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        pass
    for pattern in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
        try:
            return datetime.strptime(text, pattern)
        except ValueError:
            continue
    return None


def _load(output_dir: Path, name: str) -> Any:
    path = output_dir / name
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _as_records(payload: Any) -> list[dict[str, Any]]:
    """Accept either a list of records or a dict keyed by record id."""
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict):
        return [value for value in payload.values() if isinstance(value, dict)]
    return []


def _pair(source_id: str, target_id: str) -> tuple[str, str]:
    return tuple(sorted((source_id, target_id)))  # type: ignore[return-value]


def _time_aware_split(
    positives: Sequence[LabelledPair],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Split timestamped positives by event time, earliest -> train.

    Edges without a timestamp are excluded entirely: assigning them by index
    would be a random split wearing a temporal costume. Ties are broken by
    record id so the split is reproducible.
    """
    timed = [item for item in positives if item.timestamp]
    timed.sort(key=lambda item: (item.timestamp, item.record_id))
    if len(timed) < MIN_POSITIVES_FOR_TIME_SPLIT:
        return (), ()
    cut = max(1, int(len(timed) * TRAIN_FRACTION))
    cut = min(cut, len(timed) - 1)
    train = tuple(item.record_id for item in timed[:cut])
    test = tuple(item.record_id for item in timed[cut:])
    return train, test


def inventory_labels(output_dir: str | Path) -> LabelInventory:
    """Audit every available label source in a pipeline output directory.

    Returns a :class:`LabelInventory` whose ``blockers`` explain, in priority
    order, exactly why the labels do or do not support an acceptance gate.
    """
    root = Path(output_dir)
    raw_edges = _as_records(_load(root, "graph_edges.json"))
    raw_missing = _as_records(_load(root, "missing_edges.json"))
    raw_contradictions = _as_records(_load(root, "contradictions.json"))

    positives: list[LabelledPair] = []
    non_observed = 0
    blank = 0
    for edge in raw_edges:
        record_id = str(edge.get("id", ""))
        source_id = str(edge.get("source_id", ""))
        target_id = str(edge.get("target_id", ""))
        if not record_id or not source_id or not target_id:
            blank += 1
            continue
        if str(edge.get("epistemic_status", "")) != "observation":
            non_observed += 1
            continue
        temporal = edge.get("temporal_info") or {}
        stamp = parse_timestamp(temporal.get("timestamp")) if isinstance(
            temporal, Mapping
        ) else None
        positives.append(
            LabelledPair(
                source_id=source_id,
                target_id=target_id,
                label=1,
                provenance=POSITIVE_SOURCE_OBSERVED_EDGES,
                record_id=record_id,
                timestamp=stamp.isoformat() if stamp else "",
            )
        )

    negatives: list[LabelledPair] = []
    missing_seen: set[tuple[str, str]] = set()
    for record in raw_missing:
        source_id = str(record.get("source_id", ""))
        target_id = str(record.get("target_id", ""))
        if not source_id or not target_id:
            continue
        key = _pair(source_id, target_id)
        if key in missing_seen:
            continue
        missing_seen.add(key)
        negatives.append(
            LabelledPair(
                source_id=source_id,
                target_id=target_id,
                label=0,
                provenance=NEGATIVE_SOURCE_MISSING_EDGES,
                record_id=f"MISSING_{key[0]}_{key[1]}_{record.get('expected_relation', '')}",
            )
        )

    observed_keys = {_pair(item.source_id, item.target_id) for item in positives}
    negatives = [item for item in negatives if item.key not in observed_keys]

    train, test = _time_aware_split(positives)

    blockers: list[str] = []
    if not positives:
        blockers.append("no observed positive edges available as labels")
    if not negatives:
        blockers.append(
            "no known-absent negatives: only missing_edges.json asserts "
            "absence in this system, and it is empty. Unobserved pairs are "
            "deliberately NOT labelled negative (Roadmap R1.1), so no "
            "precision-recall evaluation is possible"
        )
    if len(positives) and not train:
        blockers.append(
            f"time-aware split needs at least {MIN_POSITIVES_FOR_TIME_SPLIT} "
            f"positives carrying a usable temporal_info.timestamp; "
            f"{sum(1 for p in positives if p.timestamp)} do"
        )
    if train and test:
        test_positives = set(test)
        test_known_negatives = {
            item.key for item in negatives if item.key not in test_positives
        }
        if not (test_positives and test_known_negatives):
            blockers.append(
                "test split contains only one class; a single-class test set "
                "cannot rank anything"
            )

    warnings: list[str] = []
    if blank:
        warnings.append(f"{blank} edges skipped: blank id or endpoint")
    if non_observed:
        warnings.append(
            f"{non_observed} edges excluded: epistemic_status != observation "
            "(predicted edges may never be labels)"
        )
    if positives and not negatives and not any(
        "known-absent negatives" in item for item in blockers
    ):
        warnings.append("positive labels exist but no negatives are recorded")
    overlap = observed_keys & missing_seen
    if overlap:
        warnings.append(
            f"{len(overlap)} pairs appear both observed and in missing_edges; "
            "observed wins and the negative is dropped"
        )
    timestamped = sum(1 for item in positives if item.timestamp)
    if positives and timestamped < len(positives):
        warnings.append(
            f"{timestamped}/{len(positives)} positives carry event time; "
            "the remainder are excluded from the time-aware split"
        )

    sources = [
        LabelSource(
            name="observed_edges",
            kind="positive",
            count=len(positives),
            provenance="graph_edges.json, epistemic_status == 'observation'",
            permissible_use=(
                "train/test labels for link prediction; every entry is an "
                "observed record, so using it as a positive never upgrades an "
                "inference into evidence"
            ),
            notes=f"{len(raw_edges)} raw edge records scanned",
        ),
        LabelSource(
            name="missing_edges",
            kind="negative",
            count=len(negatives),
            provenance=(
                "missing_edges.json: Stage 5 scanned the source corpus and "
                "recorded that an expected relation was absent"
            ),
            permissible_use=(
                "hard negatives ONLY. Absence here is asserted from a source "
                "scan, not inferred from a pair being absent from the graph"
            ),
            notes="this is the system's sole known-absent negative source",
        ),
        LabelSource(
            name="contradictions",
            kind="negative_candidate",
            count=0,
            provenance="contradictions.json",
            permissible_use="unused",
            notes=(
                f"{len(raw_contradictions)} contradictions present but Stage 8 "
                "resolved none, so no disproven relation is available as a "
                "negative label"
            ),
        ),
    ]

    return LabelInventory(
        sources=tuple(sources),
        positives=tuple(positives),
        negatives=tuple(negatives),
        split=(train, test),
        blockers=tuple(blockers),
        warnings=tuple(warnings),
        parameters={
            "train_fraction": TRAIN_FRACTION,
            "min_positives_for_time_split": MIN_POSITIVES_FOR_TIME_SPLIT,
            "negative_policy": (
                "absence must be recorded by a source scan; unobserved pairs "
                "are never negative"
            ),
            "positive_policy": "epistemic_status == 'observation'",
        },
    )


def candidate_pairs(
    inventory: LabelInventory,
    snapshot_nodes: Iterable[str],
    *,
    include_unknown_pairs: bool = True,
) -> list[tuple[str, str]]:
    """Enumerate evaluation candidates.

    When ``include_unknown_pairs`` is False only labelled pairs are returned,
    which is the only mode in which a score can be interpreted. Unknown pairs
    are included solely as *unscored* context for ranking-coverage reports and
    are never assigned a label.
    """
    nodes = sorted(set(snapshot_nodes))
    labelled = [item.key for item in inventory.positives] + [
        item.key for item in inventory.negatives
    ]
    if not include_unknown_pairs:
        return sorted(set(labelled))
    seen = set(labelled)
    pairs: list[tuple[str, str]] = list(seen)
    for index, left in enumerate(nodes):
        for right in nodes[index + 1 :]:
            key = _pair(left, right)
            if key not in seen:
                seen.add(key)
                pairs.append(key)
    return sorted(pairs)
