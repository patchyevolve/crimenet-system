"""Stage 9: Gap Detection.

Stage 9 detects and describes **missing evidence requirements**. It never
invents evidence and never resolves the underlying uncertainty — the
uncertainty stays open and is handed to Stage 10.

Identity and deduplication
--------------------------

A gap is one missing evidence requirement, not an occurrence of that
requirement inside a particular hypothesis::

    identity = (kind, subject, requirement)
    id       = generate_id("GAP", f"{kind}:{subject}:{requirement}")

  * kind          missing_relation | unresolved_attribute
  * subject       sorted resolved entity ids (a pair for relations)
  * requirement   expected_relation for relations, attribute for attributes

The hypothesis is **not** part of identity. That single decision is the
deduplication rule: Stage 7 repeats each missing edge across up to seven
hypotheses (38 occurrences for 6 distinct edges here), and those collapse to
one gap each. Hypotheses are reattached afterwards through the multi-valued
`affects_hypothesis`, with `reach` reporting its size.

The two producers cannot collide. Four entities appear in both stores, but a
missing ``CALLED`` relation between A and B and an authoritative ``name``
record for A are different requirements: the ``kind`` differs and the subject
cardinality differs (pair versus entity set). Two contradictions that resolve
to the same entity set with the same attribute *do* collapse, because they are
the same requirement.

Gaps are **not** deduplicated on ``data_source``: distinct requirements that
happen to be satisfied from the same place to look remain distinct.

Out of scope for gap production
-------------------------------

  * ``resolvable: false``  -- nothing can be gathered; this is a limitation
    for Stage 10, exactly as Stage 8 routed it.
  * ``resolved: true``     -- the requirement no longer exists.
  * ``provenance_available: false`` -- input failure, not a gap.

Metrics
-------

``reach`` and ``impact_on_hypothesis`` are measured. ``discrimination`` and
``information_gain`` are **absent from this contract**, not zeroed: both need a
posterior over competing hypotheses, no probabilistic model exists in this
pipeline, and a placeholder 0.0 would be read downstream as a measurement.

``impact_on_hypothesis`` is the conditional confidence delta on the citing
primary hypothesis — recomputed, never a band label. For attribute gaps it is
Stage 8's own recomputation, reused rather than re-derived (Stage 8 is frozen).
For relation gaps the predicted ``missing_edges.confidence`` is injected as the
score the missing record would carry, anchored on the hypothesis's reported
score so ``stored + impact == recomputed`` holds exactly.

It may be negative: ``corroboration`` is a mean and the predicted confidence
sits below this corpus's family means, so a gathered record would dilute
rather than strengthen. A reported ``0.0`` is likewise a measurement, not a
placeholder — it means the delta is genuinely zero because the hypothesis has
no citing primary, or because the missing edge touches the community without
sitting inside it (Stage 7's ``_missing_for`` fallback), so it could never
become internal supporting evidence.

``feasible`` and ``data_source`` are derived structurally, with no word list:

  * attribute gap -- corpus source classes minus the classes that already
    assert the key (the complement of ``standing_source_classes``);
  * relation gap  -- source classes observed carrying ``expected_relation``
    elsewhere in the graph, falling back to classes observed carrying any edge
    incident to either subject when the expected relation is a derived
    predicate with no carriers of its own (e.g. SHARED_ASSOCIATE).

Ordering: Stage 9 runs after Stage 8 and before Stage 11.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from ..hypothesis.engine import confidence_score, evidence_independence
from ..models.schema import generate_id

KIND_MISSING_RELATION = "missing_relation"
KIND_UNRESOLVED_ATTRIBUTE = "unresolved_attribute"
GAP_KINDS: Tuple[str, ...] = (KIND_MISSING_RELATION, KIND_UNRESOLVED_ATTRIBUTE)

CANONICAL_FIELDS: Tuple[str, ...] = (
    "id",
    "kind",
    "subject",
    "requirement",
    "description",
    "affects_hypothesis",
    "feasible",
    "suggested_action",
    "data_source",
    "reach",
    "impact_on_hypothesis",
    "run_id",
)


def _read(output_dir: Path, name: str, default):
    path = output_dir / name
    if not path.exists():
        return default
    try:
        with open(path) as fh:
            return json.load(fh)
    except (json.JSONDecodeError, OSError):
        return default


def load_inputs(output_dir: str | Path) -> Dict[str, Any]:
    """Read the Stage 1/3/5/6/7/8 outputs Stage 9 reasons over."""
    out = Path(output_dir)
    hypotheses = _read(out, "hypotheses.json", [])
    contradictions = _read(out, "contradictions.json", {})
    missing_edges = _read(out, "missing_edges.json", [])
    graph_edges = _read(out, "graph_edges.json", [])
    communities = _read(out, "community_assignments.json", [])
    summary = _read(out, "extraction_summary.json", {})
    resolved = _read(out, "resolved_entities.json", {})
    id_map = _read(out, "id_map.json", {})

    if isinstance(contradictions, dict):
        contradictions = list(contradictions.values())
    if not isinstance(contradictions, list):
        contradictions = []
    if not isinstance(hypotheses, list):
        hypotheses = []
    if not isinstance(missing_edges, list):
        missing_edges = []
    if not isinstance(graph_edges, list):
        graph_edges = []
    if not isinstance(communities, list):
        communities = []

    ingestion = summary.get("ingestion_summary", {}) if isinstance(summary, dict) else {}
    class_of = {str(f.get("name", "")): str(f.get("source_type") or "unknown")
                for f in (ingestion.get("files", []) or []) if isinstance(f, dict)}
    corpus_classes = sorted({c for c in class_of.values() if c and c != "unknown"})

    return {
        "hypotheses": hypotheses,
        "contradictions": [c for c in contradictions if isinstance(c, dict)],
        "missing_edges": [m for m in missing_edges if isinstance(m, dict)],
        "graph_edges": [e for e in graph_edges if isinstance(e, dict)],
        "communities": communities,
        "resolved_entities": resolved if isinstance(resolved, dict) else {},
        "id_map": {str(k): str(v) for k, v in (id_map or {}).items()}
        if isinstance(id_map, dict) else {},
        "class_of": class_of,
        "corpus_classes": corpus_classes,
    }


# ── structural helpers ────────────────────────────────────────────────

def _resolve(entity_id: str, id_map: Dict[str, str]) -> str:
    if not entity_id:
        return ""
    return id_map.get(entity_id, entity_id)


def _label(entity_id: str, resolved_entities: dict) -> str:
    record = resolved_entities.get(entity_id)
    if isinstance(record, dict) and record.get("canonical_name"):
        return str(record["canonical_name"])
    return entity_id or "an unidentified entity"


def _missing_key(edge: dict) -> str:
    return (f"{edge.get('source_id', '')}->{edge.get('target_id', '')}:"
            f"{edge.get('expected_relation', '')}")


def gap_id(kind: str, subject: Sequence[str], requirement: str) -> str:
    """Stable across runs: a gap is world state, like a CONTRADICTION id."""
    return generate_id("GAP", f"{kind}:{':'.join(subject)}:{requirement}")


def _subjects_of(contradiction: dict, id_map: Dict[str, str]) -> List[str]:
    return sorted({r for r in (_resolve(x, id_map)
                               for x in (contradiction.get("entity_ids") or []))
                   if r})


def _carriers_and_incident(graph_edges: Sequence[dict],
                           class_of: Dict[str, str]) -> Tuple[Dict[str, Set[str]],
                                                             Dict[str, Set[str]]]:
    carriers: Dict[str, Set[str]] = {}
    incident: Dict[str, Set[str]] = {}
    for edge in graph_edges:
        classes = {class_of[f] for f in (edge.get("provenance_chain") or [])
                   if f in class_of}
        relation = str(edge.get("relationship_type") or "")
        if relation:
            carriers.setdefault(relation, set()).update(classes)
        for endpoint in (edge.get("source_id"), edge.get("target_id")):
            if endpoint:
                incident.setdefault(str(endpoint), set()).update(classes)
    return carriers, incident


def relation_data_source(requirement: str,
                         subject: Sequence[str],
                         carriers: Dict[str, Set[str]],
                         incident: Dict[str, Set[str]]) -> List[str]:
    observed = sorted(carriers.get(requirement, set()))
    if observed:
        return observed
    fallback: Set[str] = set()
    for entity_id in subject:
        fallback |= incident.get(entity_id, set())
    return sorted(fallback)


def attribute_data_source(standing_classes: Sequence[str],
                          corpus_classes: Sequence[str]) -> List[str]:
    standing = set(standing_classes)
    return [c for c in corpus_classes if c not in standing]


# ── impact ────────────────────────────────────────────────────────────

def _factor(confidence: dict, factor_type: str) -> float:
    for factor in (confidence.get("factors") or []):
        if factor.get("factor_type") == factor_type:
            return float(factor.get("value") or 0.0)
    return 0.0


def relation_impact(primary: Optional[dict],
                    record: dict,
                    edges_by_id: Dict[str, dict],
                    members: Set[str]) -> float:
    """Confidence delta if the missing relation were present.

    `record` must carry **resolved** endpoint ids and `members` the resolved
    node ids of the primary's community — the membership test that decides
    whether the record could ever become internal supporting evidence is
    meaningless otherwise, and fails closed to a delta of 0.0.

    The delta is anchored on the hypothesis's own reported score rather than
    on a recomputed baseline, so the invariant a downstream consumer can rely
    on is exact: `confidence.score + impact_on_hypothesis == recomputed score
    with the missing record present`.

    The value may be **negative**. It is the pipeline's own confidence model
    speaking: `corroboration` is a mean, and `missing_edges.confidence` for
    this corpus sits below the families' means, so a predicted record dilutes
    rather than strengthens. Clamping that to zero would replace a
    measurement with a nicer-looking number.
    """
    if primary is None:
        return 0.0
    endpoints = {str(record.get("source_id") or ""),
                 str(record.get("target_id") or "")}
    if not endpoints <= members:
        return 0.0

    confidence = primary.get("confidence") or {}
    corroboration = _factor(confidence, "corroboration")
    contradiction_absence = _factor(confidence, "contradiction_absence")
    stored = float(confidence.get("score") or 0.0)
    # `count` is Stage 7's own n for the corroboration mean; `supporting` is
    # the resolved subset used to rebuild the independence mention list.
    count = len(primary.get("supporting") or [])
    supporting = [edges_by_id[e] for e in (primary.get("supporting") or [])
                  if e in edges_by_id]

    predicted = float(record.get("confidence") or 0.0)
    injected = {"provenance_chain": list(record.get("source_files") or [])}
    independence_new = evidence_independence(supporting + [injected])
    corroboration_new = ((corroboration * count + predicted) / (count + 1)
                         if count else predicted)

    new = confidence_score(corroboration_new, independence_new,
                           contradiction_absence)
    return round(new - stored, 6)


# ── gap construction ──────────────────────────────────────────────────

def _scope_by_missing(hypotheses: Sequence[dict]) -> Dict[str, List[str]]:
    scoped: Dict[str, List[str]] = {}
    for hypothesis in hypotheses:
        hypothesis_id = hypothesis.get("id", "")
        for key in (hypothesis.get("missing") or []):
            scoped.setdefault(str(key), []).append(hypothesis_id)
    return scoped


def _scope_by_contradiction(hypotheses: Sequence[dict]) -> Dict[str, List[str]]:
    scoped: Dict[str, List[str]] = {}
    for hypothesis in hypotheses:
        hypothesis_id = hypothesis.get("id", "")
        for contradiction_id in (hypothesis.get("contradicting") or []):
            scoped.setdefault(str(contradiction_id), []).append(hypothesis_id)
    return scoped


def _finish(gap: Dict[str, Any]) -> Dict[str, Any]:
    gap["reach"] = len(gap["affects_hypothesis"])
    missing = [f for f in CANONICAL_FIELDS if f not in gap]
    if missing:
        raise ValueError(f"gap {gap.get('id')!r} missing fields {missing}")
    return gap


def _attribute_identity(contradiction: dict,
                        id_map: Dict[str, str]
                        ) -> Optional[Tuple[str, Tuple[str, ...], str]]:
    """Gap identity for a contradiction, or None when it yields no gap.

    Order matters: a record the pipeline already settled carries no
    outstanding requirement, a record whose provenance never loaded is an
    input failure rather than a gap, and an unresolvable one is Stage 8's
    limitation to route onward — none of the three is a gap.
    """
    evidence = contradiction.get("resolution_evidence") or {}
    if contradiction.get("resolved"):
        return None
    if evidence.get("provenance_available") is False:
        return None
    if not contradiction.get("resolvable"):
        return None
    subject = _subjects_of(contradiction, id_map)
    if not subject:
        return None
    requirement = str(contradiction.get("attribute") or "")
    if not requirement:
        return None
    return (KIND_UNRESOLVED_ATTRIBUTE, tuple(subject), requirement)


def _relation_identity(record: dict,
                       id_map: Dict[str, str]
                       ) -> Optional[Tuple[str, Tuple[str, ...], str]]:
    subject = sorted({e for e in (_resolve(record.get("source_id", ""), id_map),
                                  _resolve(record.get("target_id", ""), id_map))
                      if e})
    requirement = str(record.get("expected_relation") or "")
    # A malformed record — no relation, or both endpoints collapsing onto a
    # single entity — describes no evidence requirement, so it is not one.
    if not requirement or len(subject) < 2:
        return None
    return (KIND_MISSING_RELATION, tuple(subject), requirement)


def _attribute_gap(identity: Tuple[str, Tuple[str, ...], str],
                   group: List[dict],
                   data: Dict[str, Any],
                   scope: Dict[str, List[str]],
                   run_id: str) -> Dict[str, Any]:
    kind, subject, requirement = identity
    evidences = [c.get("resolution_evidence") or {} for c in group]
    standing = sorted({v for e in evidences
                       for v in (e.get("standing_values") or [])})
    sources = sorted({s for e in evidences
                      for s in (e.get("standing_sources") or [])})
    standing_classes = sorted({c for e in evidences
                               for c in (e.get("standing_source_classes") or [])})
    data_source = attribute_data_source(standing_classes, data["corpus_classes"])
    labels = [_label(e, data["resolved_entities"]) for e in subject]
    label = ", ".join(dict.fromkeys(labels))

    affected: Set[str] = set()
    for contradiction in group:
        affected |= set(scope.get(str(contradiction.get("id", "")), []))
        if contradiction.get("hypothesis_id"):
            affected.add(str(contradiction["hypothesis_id"]))

    return _finish({
        "id": gap_id(kind, subject, requirement),
        "kind": kind,
        "subject": list(subject),
        "requirement": requirement,
        "description": (
            f"Attribute '{requirement}' of {label} carries "
            f"{len(standing)} disputed value(s) asserted by {len(sources)} "
            f"source(s) in class(es) {standing_classes or 'unknown'}. "
            f"Source class(es) never consulted for this key: "
            f"{data_source or 'none'}."
        ),
        "affects_hypothesis": sorted(affected),
        "feasible": bool(data_source),
        "suggested_action": (
            f"Acquire a {requirement} record for {label} from source class in "
            f"{{{', '.join(data_source)}}}."
        ),
        "data_source": data_source,
        # Stage 8 already recomputed this; reuse rather than re-derive.
        # Where several contradictions form one requirement, the largest
        # conditional effect is the one worth reporting.
        "impact_on_hypothesis": max(
            float(c.get("impact_on_hypothesis") or 0.0) for c in group),
        "run_id": run_id,
    })


def _relation_gap(identity: Tuple[str, Tuple[str, ...], str],
                  group: List[dict],
                  data: Dict[str, Any],
                  scope: Dict[str, List[str]],
                  primaries: Dict[str, dict],
                  members_by_community: Dict[str, Set[str]],
                  edges_by_id: Dict[str, dict],
                  carriers: Dict[str, Set[str]],
                  incident: Dict[str, Set[str]],
                  run_id: str) -> Dict[str, Any]:
    kind, subject, requirement = identity
    subject = list(subject)
    data_source = relation_data_source(requirement, subject, carriers, incident)
    labels = [_label(e, data["resolved_entities"]) for e in subject]
    label = " and ".join(labels)
    representative = min(group, key=_missing_key)

    affected: Set[str] = set()
    impacts: List[float] = []
    for record in group:
        record_scope = scope.get(_missing_key(record), [])
        affected |= set(record_scope)
        # Normalize to resolved endpoints before the community membership
        # test; the raw ids in missing_edges.json are not guaranteed to be
        # resolved and would otherwise fail the test silently.
        resolved = {**record,
                    "source_id": _resolve(record.get("source_id", ""),
                                          data["id_map"]),
                    "target_id": _resolve(record.get("target_id", ""),
                                          data["id_map"])}
        for hypothesis_id in record_scope:
            primary = primaries.get(hypothesis_id)
            if primary is None:
                continue
            members = members_by_community.get(
                str(primary.get("community_id", "")), set())
            impacts.append(relation_impact(primary, resolved, edges_by_id,
                                           members))

    return _finish({
        "id": gap_id(kind, subject, requirement),
        "kind": kind,
        "subject": subject,
        "requirement": requirement,
        "description": (
            f"Expected relation {requirement} between {label} is absent from "
            f"the evidence graph (predicted confidence "
            f"{float(representative.get('confidence') or 0.0):.4f}). "
            f"Candidate source class(es): {data_source or 'none'}."
        ),
        "affects_hypothesis": sorted(affected),
        "feasible": bool(data_source),
        "suggested_action": (
            f"Acquire a {requirement} record for {label} from source class in "
            f"{{{', '.join(data_source)}}}."
        ),
        "data_source": data_source,
        "impact_on_hypothesis": max(impacts, default=0.0),
        "run_id": run_id,
    })


def _group(items: Sequence[Any],
           identity_of) -> Dict[Tuple[str, Tuple[str, ...], str], List[Any]]:
    """Bucket records by gap identity, preserving insertion order."""
    grouped: Dict[Tuple[str, Tuple[str, ...], str], List[Any]] = {}
    for item in items:
        identity = identity_of(item)
        if identity is None:
            continue
        grouped.setdefault(identity, []).append(item)
    return grouped


def build_gaps(data: Dict[str, Any], run_id: str) -> List[dict]:
    """One gap per missing evidence requirement, deduplicated by identity.

    Grouping happens *before* construction, so two records describing the
    same requirement collapse into one gap whose `affects_hypothesis` is the
    union of everything that cited either of them.
    """
    hypotheses = data["hypotheses"]
    missing_scope = _scope_by_missing(hypotheses)
    contradiction_scope = _scope_by_contradiction(hypotheses)
    primaries = {h.get("id", ""): h for h in hypotheses
                 if h.get("generation_basis") == "community"}
    members_by_community = {
        str(c.get("community_id", "")): {m for m in (c.get("node_ids") or []) if m}
        for c in data["communities"]
    }
    edges_by_id = {e.get("id", ""): e for e in data["graph_edges"] if e.get("id")}
    carriers, incident = _carriers_and_incident(data["graph_edges"],
                                                data["class_of"])

    gaps: List[dict] = []
    attribute_groups = _group(
        data["contradictions"],
        lambda c: _attribute_identity(c, data["id_map"]))
    for identity, group in attribute_groups.items():
        gaps.append(_attribute_gap(identity, group, data,
                                   contradiction_scope, run_id))

    relation_groups = _group(
        data["missing_edges"],
        lambda m: _relation_identity(m, data["id_map"]))
    for identity, group in relation_groups.items():
        gaps.append(_relation_gap(identity, group, data, missing_scope,
                                  primaries, members_by_community, edges_by_id,
                                  carriers, incident, run_id))

    gaps.sort(key=lambda g: (g["kind"], g["subject"], g["requirement"], g["id"]))
    return gaps


def _summarise(gaps: List[dict], run_id: str) -> Dict[str, Any]:
    kinds: Dict[str, int] = {}
    for gap in gaps:
        kinds[gap["kind"]] = kinds.get(gap["kind"], 0) + 1
    covered = {h for gap in gaps for h in gap["affects_hypothesis"]}
    return {
        "run_id": run_id,
        "gaps": len(gaps),
        "by_kind": dict(sorted(kinds.items())),
        "feasible": sum(1 for g in gaps if g["feasible"]),
        "infeasible": sum(1 for g in gaps if not g["feasible"]),
        "unscoped": sum(1 for g in gaps if not g["affects_hypothesis"]),
        "hypotheses_covered": len(covered),
        "max_impact": max((g["impact_on_hypothesis"] for g in gaps), default=0.0),
        "output": "evidence_gaps.json",
    }


def generate(output_dir: str | Path, run_id: str = "") -> Dict[str, Any]:
    """Stage 9: write evidence_gaps.json and return a stage summary."""
    output_dir = Path(output_dir)
    data = load_inputs(output_dir)
    gaps = build_gaps(data, run_id)

    path = output_dir / "evidence_gaps.json"
    with open(path, "w") as fh:
        json.dump(gaps, fh, indent=2, default=str)

    return _summarise(gaps, run_id)
