"""Fixed graph snapshots — the leakage control for every feature.

ML_ENGINE.md R1.2: *"Build a leakage-resistant feature dataset from a fixed
graph snapshot. Exclude future data, outcome-derived fields, and features that
encode the target edge."*

A snapshot is an immutable view of the graph with a content hash. Two things
follow from making the view explicit rather than reading the live store:

  * **Reproducibility.** Every feature vector and prediction carries the
    snapshot hash it was computed from (``Provenance.input_snapshot``), so a
    number can always be traced back to the exact graph that produced it.
  * **Holdout correctness.** :func:`build_snapshot` accepts
    ``withheld_edge_ids``. Held-out edges are removed *before* the adjacency,
    degrees and neighborhoods are computed, so no feature for a withheld pair
    can see the edge it is meant to predict. Reading the live store instead
    would let common-neighbour counts include the answer.

The snapshot contains only structure that exists at extraction time — endpoint
ids and relationship type. Confidence, contradiction counts and hypothesis
factors are deliberately *not* in the structural view used by the baseline: a
feature derived from ``confidence`` would encode how much the system already
believes the edge, which is an outcome-derived field under R1.2.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

__all__ = [
    "GraphSnapshot",
    "build_snapshot",
    "load_snapshot",
    "snapshot_hash",
]

NODE_ID_KEYS: tuple[str, ...] = ("source_id", "target_id")


def _canonical(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)


def snapshot_hash(payload: Any) -> str:
    """Stable content hash of a snapshot payload (sha256, first 16 hex chars)."""
    return hashlib.sha256(_canonical(payload).encode("utf-8")).hexdigest()[:16]


@dataclass(frozen=True)
class GraphSnapshot:
    """An immutable, hashable view of the graph at one point in time."""

    hash: str
    run_id: str
    edge_ids: tuple[str, ...]
    nodes: tuple[str, ...]
    adjacency: Mapping[str, frozenset[str]]
    edge_relations: Mapping[str, str]
    withheld_edge_ids: tuple[str, ...] = ()

    @property
    def n_edges(self) -> int:
        return len(self.edge_ids)

    @property
    def n_nodes(self) -> int:
        return len(self.nodes)

    @property
    def has_withheld(self) -> bool:
        return bool(self.withheld_edge_ids)

    def degree(self, node: str) -> int:
        return len(self.adjacency.get(node, frozenset()))

    def neighbours(self, node: str) -> frozenset[str]:
        return self.adjacency.get(node, frozenset())

    def common_neighbours(self, left: str, right: str) -> set[str]:
        return set(self.neighbours(left)) & set(self.neighbours(right))

    def are_linked(self, left: str, right: str) -> bool:
        return right in self.neighbours(left)

    def is_observed(self, edge_id: str) -> bool:
        return edge_id in set(self.edge_ids)

    def summary(self) -> dict[str, Any]:
        return {
            "hash": self.hash,
            "run_id": self.run_id,
            "nodes": self.n_nodes,
            "edges": self.n_edges,
            "withheld_edges": len(self.withheld_edge_ids),
        }


def build_snapshot(
    edges: Sequence[Mapping[str, Any]],
    *,
    run_id: str = "",
    withheld_edge_ids: Iterable[str] = (),
) -> GraphSnapshot:
    """Build a deterministic snapshot, optionally with edges held out.

    Args:
        edges: Edge records with ``id``, ``source_id`` and ``target_id``.
        run_id: Owning run, carried for provenance.
        withheld_edge_ids: Edges to exclude *before* adjacency is built. This
            is the only correct way to evaluate a link predictor: features for
            a withheld pair must not be able to see the withheld edge.

    Returns:
        An immutable snapshot whose ``hash`` changes if and only if the
        retained structure changes. Withheld ids are recorded so the split is
        auditable from the artifact alone.
    """
    held = frozenset(withheld_edge_ids)
    if held and any(not edge_id for edge_id in held):
        raise ValueError("withheld_edge_ids must not contain blanks")

    retained: list[dict[str, str]] = []
    for edge in edges:
        edge_id = str(edge.get("id", ""))
        if not edge_id:
            raise ValueError("every edge needs a non-empty id for holdout accounting")
        if edge_id in held:
            continue
        try:
            source = str(edge[NODE_ID_KEYS[0]])
            target = str(edge[NODE_ID_KEYS[1]])
        except KeyError as exc:
            raise ValueError(
                f"edge {edge_id} is missing {exc.args[0]!r}; structural "
                "snapshots require both endpoints"
            ) from None
        if not source or not target:
            raise ValueError(f"edge {edge_id} has a blank endpoint")
        retained.append(
            {
                "id": edge_id,
                "source_id": source,
                "target_id": target,
                "relationship_type": str(edge.get("relationship_type", "")),
            }
        )

    retained.sort(key=lambda item: item["id"])
    digest = snapshot_hash(
        {
            "run_id": run_id,
            "withheld": sorted(held),
            "edges": retained,
        }
    )

    adjacency: dict[str, set[str]] = {}
    relations: dict[str, str] = {}
    nodes: set[str] = set()
    for item in retained:
        left, right = item["source_id"], item["target_id"]
        nodes.update((left, right))
        adjacency.setdefault(left, set()).add(right)
        adjacency.setdefault(right, set()).add(left)
        relations[item["id"]] = item["relationship_type"]

    return GraphSnapshot(
        hash=digest,
        run_id=run_id,
        edge_ids=tuple(item["id"] for item in retained),
        nodes=tuple(sorted(nodes)),
        adjacency={k: frozenset(v) for k, v in sorted(adjacency.items())},
        edge_relations=dict(sorted(relations.items())),
        withheld_edge_ids=tuple(sorted(held)),
    )


def load_snapshot(
    output_dir: str | Path,
    *,
    run_id: str = "",
    withheld_edge_ids: Iterable[str] = (),
) -> GraphSnapshot:
    """Load ``graph_edges.json`` from a pipeline output directory."""
    path = Path(output_dir) / "graph_edges.json"
    with path.open(encoding="utf-8") as handle:
        edges = json.load(handle)
    if not isinstance(edges, list):
        raise ValueError(f"{path} must contain a list of edge records")
    return build_snapshot(edges, run_id=run_id, withheld_edge_ids=withheld_edge_ids)
