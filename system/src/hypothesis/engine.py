"""Stage 7: Hypothesis Reasoner — competing explanations for graph structure.

WHAT THIS STAGE IS
    STAGE_REASONERS.md §Stage 7: "What could explain all this evidence?
    What are the alternatives?" A hypothesis is an *inference*, never an
    observation — see the pipeline invariant: evidence may raise a
    hypothesis' confidence but can never upgrade an inference into an
    observation. Every confidence here is therefore semantic_type
    "hypothesis".

CANONICAL VOCABULARY (reconciled — the three design docs disagreed)
    OUTPUTS.md §Stage 7, OUTPUTS.md §Falsifiable Hypothesis Model and
    STAGE_REASONERS.md §Stage 7 each specify a slightly different shape.
    The canonical field set is the union, with synonyms collapsed so
    Stages 7-10 share one vocabulary:

        conflicting name      canonical decision
        -----------------     ------------------
        type/hypothesis_type  -> type
        supporting/           -> supporting  (+ contradicting, kept
        supporting_evidence       because STAGE_REASONERS evidence
                                  evaluation requires both)
        alternatives: ID[]/   -> alternatives: ID[]  (flat references,
        alternatives: Hypothesis[]                        no nesting)
        null_hypothesis: ID/  -> null_hypothesis: ID (same reason)

    Synergous duplicates are deliberately NOT emitted: there is no
    supporting_evidence, no hypothesis_type, no embedded Hypothesis
    objects. See CANONICAL_FIELDS.

CONFIDENCE
    confidence is a ConfidenceSchema (score + basis + factors), matching
    how graph_edges.json already emits confidence, and satisfying
    OUTPUTS.md:121 "A bare number is not enough." Stages 8/10 read
    confidence.score.

GENERATION UNIT
    One primary hypothesis per community (community_assignments.json),
    recorded in generation_basis="community" + community_id so Stages 8-10
    never have to infer why a hypothesis exists. Communities are a disjoint
    partition, so they cannot produce nested/overlapping primaries; the 4
    candidate components are consumed only as evidence (anomaly signals),
    never as competing units.

ALTERNATIVES (STAGE_REASONERS RULE 1-5)
    RULE 1 every primary gets alternatives at all five levels.
    RULE 2 source-error alternatives are generated whenever any structural
          basis exists (Stage 1 adversarial flags + edge adversarial_score).
    RULE 3 every alternative carries confidence/supporting/contradicting/
          missing/falsifiers.
    RULE 4 the null hypothesis is always present and referenced by
          null_hypothesis on every hypothesis.
    RULE 5 alternatives must be genuinely different: the entity-level
          alternative deliberately does NOT assume the most-connected
          member is involved, and the source-error alternative questions
          the evidence itself.

ANTI-HARDCODING
    There is no crime taxonomy here. OUTPUTS.md §Stage 7 lists example
    values ("fraud_ring", "drug_trafficking") that can only be produced by
    a hand-tuned keyword mapping or an LLM; neither is available (LLM is
    disabled in this pipeline). `type` therefore uses the structural
    five-level enum from the Falsifiable Hypothesis Model, and the concrete
    structural composition of each community is written into `description`
    and `reasoning` instead. No detection threshold is introduced by this
    stage: scores are reported, never used to gate generation.
"""

from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from ..models.schema import ConfidenceFactor, ConfidenceSchema, generate_id

# ── Canonical contract ────────────────────────────────────────────────

CANONICAL_FIELDS = (
    "id",
    "type",
    "description",
    "confidence",
    "supporting",
    "contradicting",
    "missing",
    "alternatives",
    "null_hypothesis",
    "falsifiers",
    "sensitivity",
    "evidence_independence",
    "last_updated",
    "reasoning",
    "pattern_matches",
    "generation_basis",
    "community_id",
)

# Fields that exist in one design doc but are synonyms of a canonical
# field. Emitted deliberately-absent so no consumer can read two names
# for the same thing.
FORBIDDEN_SYNONYMS = (
    "supporting_evidence",
    "hypothesis_type",
)

HYPOTHESIS_LEVELS = (
    "entity",        # who is involved?
    "event",         # what happened?
    "relationship",  # how are entities connected?
    "causal",        # why?
    "source_error",  # is the evidence itself wrong?
)

# Levels a PRIMARY may take, in precedence order (also the tie-break
# order used by derive_primary_type). source_error and causal are
# challenge levels: always generated as alternatives, never as a primary.
PRIMARY_CANDIDATE_LEVELS = ("entity", "event", "relationship")

NULL_HYPOTHESIS_ID = "HYP_NULL"

# Decomposition weights for confidence.score. These are additive
# coefficients that make score == sum(value * weight) — they are NOT
# decision thresholds: nothing in this stage branches on a score.
W_CORROBORATION = 0.4   # mean confidence of supporting edges
W_INDEPENDENCE = 0.3    # source-file diversity of supporting evidence
W_CONTRADICTION = 0.3   # absence of unresolved contradiction among members


# ── Input loading ─────────────────────────────────────────────────────

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
    """Read the Stage 5/6 outputs Stage 7 reasons over."""
    out = Path(output_dir)
    communities = _read(out, "community_assignments.json", [])
    edges = _read(out, "graph_edges.json", [])
    contradictions = _read(out, "contradictions.json", {})
    missing_edges = _read(out, "missing_edges.json", [])
    signals = _read(out, "anomaly_signals.json", [])
    entities = _read(out, "resolved_entities.json", {})
    events = _read(out, "timeline_events.json", [])
    summary = _read(out, "extraction_summary.json", {})
    id_map = _read(out, "id_map.json", {})

    if isinstance(entities, dict):
        entities = list(entities.values())
    if isinstance(contradictions, dict):
        contradictions = list(contradictions.values())
    if not isinstance(events, list):
        events = []
    if not isinstance(id_map, dict):
        id_map = {}
    # Stage 2 emits raw extraction ids (LOC_/PERSON_/DATE_/POST_...); the
    # graph and communities use RES_ ids. Without this translation every
    # downstream membership test silently matches nothing.
    id_map = {str(k): str(v) for k, v in id_map.items()}

    adversarial_files = set(
        summary.get("ingestion_summary", {})
              .get("adversarial", {})
              .get("suspicious_files", [])
    )
    return {
        "communities": communities if isinstance(communities, list) else [],
        "edges": edges if isinstance(edges, list) else [],
        "contradictions": contradictions,
        "missing_edges": missing_edges if isinstance(missing_edges, list) else [],
        "signals": signals if isinstance(signals, list) else [],
        "entities": entities,
        "events": events,
        "id_map": id_map,
        "adversarial_files": adversarial_files,
    }


# ── Structural primitives ─────────────────────────────────────────────

def _edge_score(edge: dict) -> float:
    conf = edge.get("confidence")
    if isinstance(conf, dict):
        return float(conf.get("score", 0.0) or 0.0)
    if isinstance(conf, (int, float)):
        return float(conf)
    return 0.0


def _prov(edge: dict) -> List[str]:
    """Provenance chain as a clean list of file names."""
    chain = edge.get("provenance_chain") or []
    if isinstance(chain, str):
        return [chain] if chain else []
    return [str(x) for x in chain if x]


def _resolve(entity_id: str, id_map: Dict[str, str]) -> str:
    """Raw extraction id -> graph/resolved id (identity map when absent)."""
    if not entity_id:
        return ""
    return id_map.get(entity_id, entity_id)


def evidence_independence(edges: Sequence[dict]) -> float:
    """How independent are the sources behind a set of edges.

    distinct source files / total provenance mentions. 1.0 means every
    mention comes from a different file (no source is reused), low values
    mean the whole evidential basis rests on a handful of files.
    """
    mentions: List[str] = []
    for e in edges:
        mentions.extend(_prov(e))
    if not mentions:
        return 0.0
    return round(len(set(mentions)) / len(mentions), 6)


def _ratio(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return max(0.0, min(1.0, numerator / denominator))


def confidence_score(mean_corroboration: float,
                    independence: float,
                    contradiction_absence: float) -> float:
    """score == sum(value * weight) over the three reported factors."""
    score = (W_CORROBORATION * mean_corroboration
             + W_INDEPENDENCE * independence
             + W_CONTRADICTION * contradiction_absence)
    return round(max(0.0, min(1.0, score)), 6)


def build_confidence(*, basis: List[str],
                    factors: List[ConfidenceFactor],
                    supporting_count: int, contradicting_count: int,
                    unknown_count: int, independence: float,
                    run_id: str) -> dict:
    # Score is DERIVED from the reported factors, never passed in, so
    # score == sum(value * weight) holds by construction instead of by hope.
    score = round(max(0.0, min(1.0,
                               sum(f.value * f.weight for f in factors))), 6)
    conf = ConfidenceSchema(
        score=score,
        basis=basis,
        supporting_count=supporting_count,
        contradicting_count=contradicting_count,
        unknown_count=unknown_count,
        source_reliability=independence,
        derivation_depth=1,
        is_independent=independence >= 1.0,
        semantic_type="hypothesis",
        run_id=run_id,
        factors=factors,
    )
    return conf.to_dict()


def _factors(*, corroboration: float, independence: float,
            contradiction_absence: float) -> List[ConfidenceFactor]:
    return [
        ConfidenceFactor(
            factor_type="corroboration",
            value=round(corroboration, 6),
            weight=W_CORROBORATION,
            description=f"Mean confidence of supporting evidence: {corroboration:.3f}",
        ),
        ConfidenceFactor(
            factor_type="evidence_independence",
            value=round(independence, 6),
            weight=W_INDEPENDENCE,
            description=f"Source-file diversity of supporting evidence: {independence:.3f}",
        ),
        ConfidenceFactor(
            factor_type="contradiction_absence",
            value=round(contradiction_absence, 6),
            weight=W_CONTRADICTION,
            description=(
                f"Share of members without unresolved contradiction: "
                f"{contradiction_absence:.3f}"
            ),
        ),
    ]


# ── Falsifiers ────────────────────────────────────────────────────────

def _falsifier(fid: str, description: str, required_evidence: str,
              findability: str, impact: str, status: str = "not_searched") -> dict:
    return {
        "id": fid,
        "description": description,
        "required_evidence": required_evidence,
        "findability": findability,
        "impact": impact,
        "status": status,
    }


def build_falsifiers(*, family_key: str, members: set, edges: Sequence[dict],
                    contradictions: Sequence[dict], missing: Sequence[dict],
                    entities_by_id: Dict[str, dict], id_map: Dict[str, str],
                    adversarial_files: set) -> List[dict]:
    """Falsifiers grounded in gaps the run actually measured.

    status is always "not_searched": this pipeline performs no
    falsification search, so claiming searched_not_found/found would be a
    false assertion. findability distinguishes evidence we already hold
    ("likely" — the corpus bears on it) from evidence that would have to
    be collected ("possible").
    """
    out: List[dict] = []

    # 1. Identity uncertainty — an unresolved merge would invalidate the
    #    membership claim the whole hypothesis rests on.
    uncertain = sorted(
        eid for eid in members
        if entities_by_id.get(eid, {}).get("merge_type") == "review"
    )
    for eid in uncertain[:3]:
        name = entities_by_id.get(eid, {}).get("canonical_name", eid)
        out.append(_falsifier(
            generate_id("FALS", f"{family_key}:identity:{eid}"),
            f"Show that {name} is an entity-resolution artifact rather than a real participant",
            "Independent confirmation (or rejection) of the identity of " + name,
            "likely", "reject_hypothesis",
        ))

    # 2. Fabricated provenance — a Stage 1 flagged file backing the
    #    supporting edges would remove that support entirely.
    flagged = sorted({
        f for e in edges for f in _prov(e) if f in adversarial_files
    })
    for fname in flagged[:3]:
        out.append(_falsifier(
            generate_id("FALS", f"{family_key}:adversarial:{fname}"),
            f"Show that {fname} is fabricated or misattributed",
            "Authentication of the source file " + fname,
            "likely", "reject_hypothesis",
        ))

    # 3. Absent relation — an expected-but-absent link would weaken the
    #    structural claim. Requires new collection.
    for m in missing[:3]:
        src = m.get("source_id", "")
        tgt = m.get("target_id", "")
        rel = m.get("expected_relation", "")
        src_name = entities_by_id.get(src, {}).get("canonical_name", src)
        tgt_name = entities_by_id.get(tgt, {}).get("canonical_name", tgt)
        out.append(_falsifier(
            generate_id("FALS", f"{family_key}:missing:{src}:{tgt}"),
            f"Establish the expected {rel} relation between {src_name} and "
            f"{tgt_name} directly",
            f"Direct evidence of {rel} between {src_name} and {tgt_name}",
            "possible", "weaken_hypothesis",
        ))

    # 4. Contradiction resolution — an unresolved identity contradiction
    #    among members undermines the grouping. entity_ids are raw
    #    extraction ids, so they must be resolved before comparison.
    for c in contradictions[:3]:
        resolved_ids = {_resolve(x, id_map) for x in (c.get("entity_ids") or [])}
        if resolved_ids & members:
            out.append(_falsifier(
                generate_id("FALS", f"{family_key}:contradiction:{c.get('id','')}"),
                f"Resolve the unresolved {c.get('type','')} contradiction on {c.get('attribute','')}",
                f"Reconciliation of the conflicting values recorded for {c.get('attribute','')}",
                "possible", "require_revision",
            ))

    # 5. Every hypothesis must carry at least one falsifier (RULE 3).
    #    When none of the measured gaps above applies, fall back to a test
    #    that is true of any hypothesis by construction.
    if not out:
        if edges:
            out.append(_falsifier(
                generate_id("FALS", f"{family_key}:verification"),
                "Verify any one supporting link against the record it cites; if "
                "that record does not contain the claimed relation, the support is void",
                "The source records named in this hypothesis's provenance",
                "likely" if any(_prov(e) for e in edges) else "possible",
                "reject_hypothesis",
            ))
        else:
            out.append(_falsifier(
                generate_id("FALS", f"{family_key}:decisive_record"),
                "Recover a record that positively establishes the primary reading "
                "over this alternative",
                "An observation bearing directly on this reading, absent from the "
                "current corpus",
                "possible",
                "reject_hypothesis",
            ))
    return out


# ── Sensitivity ───────────────────────────────────────────────────────

def build_sensitivity(*, members: set, edges: Sequence[dict],
                     entities_by_id: Dict[str, dict],
                     contradiction_absence: float,
                     independence: float) -> dict:
    """What if a key assumption is wrong?

    `if_false` is a real recomputation, not a guess: the score is
    recalculated with the single most heavily relied-upon source file
    removed from the evidential basis.
    """
    file_counts: Counter = Counter()
    for e in edges:
        for f in set(_prov(e)):
            file_counts[f] += 1

    assumptions: List[dict] = [{
        "assumption": f"All {len(edges)} supporting links are correctly attributed "
                      f"to this group",
        "if_false": "Confidence would fall if any supporting link were retracted",
        "testability": "testable",
    }]

    if file_counts:
        dominant, dominant_n = file_counts.most_common(1)[0]
        surviving = [e for e in edges if dominant not in set(_prov(e))]
        surviving_indep = evidence_independence(surviving)
        surviving_mean = (
            sum(_edge_score(e) for e in surviving) / len(surviving)
            if surviving else 0.0
        )
        alt = confidence_score(surviving_mean, surviving_indep, contradiction_absence)
        assumptions.append({
            "assumption": f"The run does not rest on a single source file ({dominant})",
            "if_false": f"Confidence would drop to {alt:.3f} "
                        f"({dominant_n} of {len(edges)} links depend on it)",
            "testability": "testable",
        })

    assumptions.append({
        "assumption": "Every member is a correctly resolved identity",
        "if_false": "Membership of the group would change, invalidating the grouping",
        "testability": "testable",
    })

    uncertain = [
        eid for eid in sorted(members)
        if entities_by_id.get(eid, {}).get("merge_type") == "review"
    ]
    identity_sensitivity: Optional[dict] = None
    if uncertain:
        eid = uncertain[0]
        ent = entities_by_id.get(eid, {})
        identity_sensitivity = {
            "entity_id": eid,
            "current_resolution": ent.get("canonical_name", eid),
            "alternative_resolutions": [],
            "basis": "Entity is flagged merge_type=review; no alternative resolution "
                     "is currently proposed, so the impact of splitting is unquantified.",
        }

    return {
        "key_assumptions": assumptions,
        "identity_sensitivity": identity_sensitivity,
    }


# ── Primary hypothesis ────────────────────────────────────────────────

def derive_primary_type(*, internal_edges: Sequence[dict],
                       disputed_members: set,
                       event_anchored_members: set) -> tuple:
    """Level at which the primary claim is most sharply characterised.

    Every candidate is measured over the SAME denominator — the
    community's internal links — and assigned exclusively under a
    documented precedence (identity is upstream of temporality, which is
    upstream of structure). That keeps the comparison apples-to-apples;
    the previous shape compared a count over N links against counts over
    much smaller populations, which made `relationship` win by
    construction rather than by evidence.

    `source_error` and `causal` are challenge levels: they are always
    generated as alternatives (RULE 1 / RULE 2) but never as the primary,
    because the primary is the substantive claim the evidence is offered
    FOR, not the objection to it.
    """
    def _disputed(edge: dict) -> bool:
        return (edge.get("source_id") in disputed_members
                or edge.get("target_id") in disputed_members)

    def _anchored(edge: dict) -> bool:
        return (edge.get("source_id") in event_anchored_members
                or edge.get("target_id") in event_anchored_members)

    entity_n = sum(1 for e in internal_edges if _disputed(e))
    rest = [e for e in internal_edges if not _disputed(e)]
    event_n = sum(1 for e in rest if _anchored(e))
    relationship_n = len(internal_edges) - entity_n - event_n

    shares = {"entity": entity_n, "event": event_n,
              "relationship": relationship_n}
    best = max(shares.values())
    for level in PRIMARY_CANDIDATE_LEVELS:      # tie-break == precedence
        if shares[level] == best:
            return level, shares
    return "relationship", shares


def _base_hypothesis(*, hid: str, htype: str, description: str, reasoning: str,
                    confidence: dict, supporting: List[str],
                    contradicting: List[str], missing: List[str],
                    alternatives: List[str], falsifiers: List[dict],
                    sensitivity: dict, independence: float,
                    pattern_matches: List[str], run_id: str,
                    community_id: str, generation_basis: str) -> dict:
    """Assemble a hypothesis with exactly the canonical field set."""
    hyp = {
        "id": hid,
        "type": htype,
        "description": description,
        "confidence": confidence,
        "supporting": supporting,
        "contradicting": contradicting,
        "missing": missing,
        "alternatives": alternatives,
        "null_hypothesis": NULL_HYPOTHESIS_ID,
        "falsifiers": falsifiers,
        "sensitivity": sensitivity,
        "evidence_independence": independence,
        "last_updated": datetime.now(timezone.utc).isoformat(),
        "reasoning": reasoning,
        "pattern_matches": pattern_matches,
        "generation_basis": generation_basis,
        "community_id": community_id,
    }
    for forbidden in FORBIDDEN_SYNONYMS:
        hyp.pop(forbidden, None)
    return hyp


def _missing_for(missing_edges: Sequence[dict], members: set) -> List[dict]:
    """Expected-but-absent relations. Fully inside the group first; if the
    group has none of its own, fall back to relations that touch it."""
    inside = [m for m in missing_edges
              if {m.get("source_id"), m.get("target_id")} <= members]
    if inside:
        return inside
    return [m for m in missing_edges
            if {m.get("source_id"), m.get("target_id")} & members]


def build_hypotheses(data: Dict[str, Any], run_id: str) -> List[dict]:
    """Build one family (primary + five-level alternatives) per community,
    plus the always-present null hypothesis (RULE 4)."""
    communities = data["communities"]
    all_edges = data["edges"]
    contradictions = data["contradictions"]
    missing_edges = data["missing_edges"]
    signals = data["signals"]
    entities = data["entities"]
    events = data["events"]
    id_map = data["id_map"]
    adversarial_files = data["adversarial_files"]

    entities_by_id = {e.get("id"): e for e in entities if e.get("id")}
    by_edge_id = {e.get("id"): e for e in all_edges if e.get("id")}

    out: List[dict] = []
    primary_ids: List[str] = []

    for comm in communities:
        community_id = comm.get("community_id", "")
        members = {m for m in comm.get("node_ids", []) if m}
        if not members:
            continue

        internal = [e for e in all_edges
                    if e.get("source_id") in members and e.get("target_id") in members]
        internal_ids = sorted({e["id"] for e in internal if e.get("id")})

        # Membership tests run on RES_ ids: contradictions carry raw
        # extraction ids, so they must be translated first.
        contra_on = [
            c for c in contradictions
            if not c.get("resolved")
            and {_resolve(x, id_map) for x in (c.get("entity_ids") or [])} & members
        ]
        contra_ids = sorted({c["id"] for c in contra_on if c.get("id")})

        missing = _missing_for(missing_edges, members)
        missing_keys = sorted({
            f"{m.get('source_id','')}->{m.get('target_id','')}:"
            f"{m.get('expected_relation','')}" for m in missing
        })

        internal_edge_ids = set(internal_ids)
        sig_on = []
        for s in signals:
            sid = _resolve(s.get("entity_id", ""), id_map)
            if sid in members:
                sig_on.append(s)
            elif sid.startswith("EDGE_") and sid in internal_edge_ids:
                # edge-scoped signals belong to whichever community
                # contains that edge
                sig_on.append(s)
        sig_ids = sorted({s["signal_id"] for s in sig_on if s.get("signal_id")})

        events_on = [ev for ev in events
                     if _resolve(ev.get("entity_id", ""), id_map) in members]
        event_ids = sorted({ev["id"] for ev in events_on if ev.get("id")})

        adversarial_internal = [e for e in internal
                                if any(f in adversarial_files for f in _prov(e))]

        independence = evidence_independence(internal)
        mean_corr = (sum(_edge_score(e) for e in internal) / len(internal)
                     if internal else 0.0)
        contradiction_absence = 1.0 - _ratio(len(contra_on), len(members))

        # Identity is disputed when the entity was left for review at
        # resolution, or when an unresolved identity contradiction names it.
        disputed_members = {
            n for n in members
            if (entities_by_id.get(n, {}) or {}).get("merge_type") == "review"
        } | {
            r for c in contra_on
            for r in (_resolve(x, id_map) for x in (c.get("entity_ids") or []))
            if r in members
        }
        event_anchored_members = {
            _resolve(ev.get("entity_id", ""), id_map) for ev in events_on
        }

        primary_type, level_shares = derive_primary_type(
            internal_edges=internal,
            disputed_members=disputed_members,
            event_anchored_members=event_anchored_members,
        )

        primary_id = generate_id("HYP", f"{run_id}:{community_id}:primary")
        primary_ids.append(primary_id)

        degree: Counter = Counter()
        for e in internal:
            degree[e.get("source_id")] += 1
            degree[e.get("target_id")] += 1
        ranked = [n for n, _ in degree.most_common()]

        unknowns = [
            "causal motive is not represented anywhere in the ingested evidence",
            "legitimate-versus-coordinated interpretation is not observable from "
            "these records",
        ]
        if not events_on:
            unknowns.append("no temporal event evidence is attached to this group")

        provenance_files = {f for e in internal for f in _prov(e)}
        if provenance_files:
            independence_basis = (
                f"independence {independence:.3f}: {len(provenance_files)} distinct "
                f"files behind {sum(len(_prov(e)) for e in internal)} provenance "
                f"mentions"
            )
        else:
            independence_basis = (
                "no source provenance is recorded on these links (they are derived "
                "from shared attributes), so source independence cannot be measured "
                "and is reported as 0.000"
            )
        factors = _factors(corroboration=mean_corr, independence=independence,
                          contradiction_absence=contradiction_absence)
        confidence = build_confidence(
            basis=[
                f"{len(internal)} supporting links, mean confidence {mean_corr:.3f}",
                independence_basis,
                f"{len(contra_on)} unresolved contradictions touching "
                f"{_ratio(len(contra_on), len(members)):.3f} of members",
                "score = 0.4*corroboration + 0.3*independence + "
                "0.3*contradiction_absence",
            ],
            factors=factors,
            supporting_count=len(internal_ids),
            contradicting_count=len(contra_ids),
            unknown_count=len(unknowns),
            independence=independence,
            run_id=run_id,
        )

        node_types = Counter(
            (entities_by_id.get(n, {}) or {}).get("entity_type", "unknown")
            for n in members
        )
        rel_types = Counter(e.get("relationship_type", "unknown") for e in internal)

        if primary_type == "entity":
            level_clause = (
                f"{level_shares['entity']} of those links touch a member whose "
                f"identity is unresolved ({len(disputed_members)} of "
                f"{len(members)} members are held for review or named by an "
                f"unresolved identity contradiction), so the claim turns on who "
                f"is involved."
            )
            level_reason = (
                f"Level selected as 'entity': identity is upstream of structure — "
                f"if membership is wrong the grouping itself is wrong — and "
                f"{level_shares['entity']}/{len(internal)} links touch a disputed "
                f"member, the largest exclusive share "
                f"(event {level_shares['event']}, relationship "
                f"{level_shares['relationship']})."
            )
        elif primary_type == "event":
            level_clause = (
                f"{level_shares['event']} of those links touch a member that "
                f"carries a recorded timeline event ({len(event_anchored_members)} "
                f"of {len(members)} members), so the claim turns on what happened."
            )
            level_reason = (
                f"Level selected as 'event': {level_shares['event']}/{len(internal)} "
                f"links are anchored in dated events, the largest exclusive share "
                f"(entity {level_shares['entity']}, relationship "
                f"{level_shares['relationship']})."
            )
        else:
            level_clause = (
                f"{level_shares['relationship']} of those links are neither "
                f"identity-disputed nor event-anchored, so the claim turns on how "
                f"the members are connected."
            )
            level_reason = (
                f"Level selected as 'relationship': {level_shares['relationship']}/"
                f"{len(internal)} links carry no identity dispute "
                f"(entity {level_shares['entity']}) and no temporal anchor "
                f"(event {level_shares['event']}), the largest exclusive share."
            )

        primary = _base_hypothesis(
            hid=primary_id,
            htype=primary_type,
            description=(
                f"{len(members)} entities form a connected group (community "
                f"{community_id}) whose internal structure is carried by "
                f"{len(internal)} links of types: "
                f"{', '.join(f'{t} x{n}' for t, n in rel_types.most_common(3)) or 'none'}; "
                f"{level_clause}"
            ),
            reasoning=(
                f"Group of {len(members)} nodes (dominant node types: "
                f"{', '.join(f'{t} x{n}' for t, n in node_types.most_common(3))}) with "
                f"{len(internal)} internal links at mean confidence {mean_corr:.3f}. "
                f"{len(provenance_files)} distinct source files back the links "
                f"(independence {independence:.3f}); {len(contra_on)} unresolved "
                f"contradictions and {len(missing)} expected-but-absent relations "
                f"qualify the claim. {level_reason} This is an inference over "
                f"observations; evidence raises its confidence but never promotes "
                f"it to an observation."
            ),
            confidence=confidence,
            supporting=internal_ids,
            contradicting=contra_ids,
            missing=missing_keys,
            alternatives=[],          # filled after the family is built
            falsifiers=build_falsifiers(
                family_key=primary_id, members=members, edges=internal,
                contradictions=contra_on, missing=missing,
                entities_by_id=entities_by_id, id_map=id_map,
                adversarial_files=adversarial_files,
            ),
            sensitivity=build_sensitivity(
                members=members, edges=internal, entities_by_id=entities_by_id,
                contradiction_absence=contradiction_absence,
                independence=independence,
            ),
            independence=independence,
            pattern_matches=sig_ids,
            run_id=run_id,
            community_id=community_id,
            generation_basis="community",
        )

        # ── five-level alternatives (RULE 1) ──────────────────────────
        specs: List[tuple] = []

        # LEVEL 1 — entity. RULE 5: must not assume the top member is involved.
        if len(ranked) >= 2:
            focus = ranked[1]
            focus_edges = [e for e in internal
                           if focus in (e.get("source_id"), e.get("target_id"))]
            focus_ids = sorted({e["id"] for e in focus_edges if e.get("id")})
            focus_name = entities_by_id.get(focus, {}).get("canonical_name", focus)
            top_name = entities_by_id.get(ranked[0], {}).get("canonical_name", ranked[0])
            specs.append((
                "entity",
                f"Coordination is attributable to {focus_name} rather than the "
                f"most-connected member {top_name}.",
                f"Second-ranked member {focus_name} carries {len(focus_ids)} of "
                f"{len(internal)} internal links; this alternative deliberately "
                f"excludes the top-ranked member.",
                focus_ids,
                [m for m in missing if focus in (m.get("source_id"), m.get("target_id"))],
            ))
        else:
            specs.append((
                "entity",
                "Coordination is attributable to a party not present in the ingested "
                "records.",
                "The group has fewer than two connected members, so no in-group "
                "alternative actor exists; an unobserved actor is the only distinct "
                "entity-level account.",
                [],
                list(missing),
            ))

        # LEVEL 2 — event: separate episodes, not one pattern. Events are
        # matched through id_map because Stage 3 attaches them to raw
        # extraction ids, not to the RES_ ids communities are built from.
        specs.append((
            "event",
            "The group's links are the residue of several separate episodes rather "
            "than one coordinated pattern.",
            (f"{len(event_ids)} timeline events resolve to members of this group, "
             f"spanning {len({e.get('event_type') for e in events_on})} event types; "
             f"if these are distinct episodes the group is an artifact of "
             f"co-occurrence rather than one pattern."
             if event_ids else
             "No timeline event resolves to any member of this group, so the "
             "episode-reading has no positive support here. The level is still "
             "generated per RULE 1, reported as unsupported rather than assumed."),
            event_ids,
            list(missing),
        ))

        # LEVEL 3 — relationship: the ties are ordinary association rather
        # than the coordinated relation the primary asserts. edge_type is
        # Stage 5's own classification (classify_semantic_edge), not one
        # invented here; "associational" is its social/bonding category.
        assoc_ids = sorted({e["id"] for e in internal
                            if e.get("edge_type") == "associational" and e.get("id")})
        specs.append((
            "relationship",
            "The ties are ordinary association between the members rather than the "
            "coordinated relationship the group appears to show.",
            f"{len(assoc_ids)} of {len(internal)} internal links are classified by "
            f"the graph builder as 'associational' (its social/bonding category: "
            f"calls, visits, association, ownership and location ties), so on that "
            f"reading the group is an ordinary circle of contact rather than a "
            f"coordinated network.",
            assoc_ids,
            list(missing),
        ))

        # LEVEL 4 — causal: no motive evidence exists in this corpus.
        specs.append((
            "causal",
            "The group is coordinated, but by a motive not represented in the ingested "
            "evidence (which would be non-criminal or benign).",
            "No evidence type in this corpus records motive, so nothing here can "
            "support or refute a motive claim either way.",
            [],
            list(missing),
        ))

        # LEVEL 5 — source-error: the evidence itself may be wrong.
        src_ids = sorted({e["id"] for e in adversarial_internal if e.get("id")})
        specs.append((
            "source_error",
            "The supporting evidence is fabricated or misattributed, so the group does "
            "not exist as described.",
            (f"{len(src_ids)} internal links draw on files Stage 1 flagged as "
             f"adversarial ({', '.join(sorted(adversarial_files)) or 'none'}); if those "
             f"files are fabricated the supporting basis collapses."
             if src_ids else
             f"No internal link draws on a Stage 1 flagged file "
             f"({', '.join(sorted(adversarial_files)) or 'none'}). The level is still "
             f"generated per RULE 1, with empty support rather than assumed support."),
            src_ids,
            list(missing),
        ))

        alts: List[dict] = []
        for level, desc, reason, sup_ids, sup_missing in specs:
            alt_id = generate_id("HYP", f"{run_id}:{community_id}:alt:{level}")
            alt_edges = [by_edge_id[i] for i in sup_ids if i in by_edge_id]
            alt_indep = evidence_independence(alt_edges)
            alt_corr = (sum(_edge_score(e) for e in alt_edges) / len(alt_edges)
                        if alt_edges else 0.0)
            # The factor measures the GROUP's contradiction load, not the
            # level's, so every alternative reports the same value the
            # primary does — otherwise scores would not be comparable.
            alt_contra_abs = contradiction_absence
            alt_unknowns = list(unknowns)
            if not sup_ids:
                alt_unknowns.append(
                    f"no structural evidence is available at the {level} level"
                )
                # Never let the claim read as established when nothing
                # supports it.
                desc = (f"{desc} Nothing in this community's evidence supports "
                        f"that reading today; the level is present because RULE 1 "
                        f"requires all five, not because it is evidenced.")
            alt_conf = build_confidence(
                basis=[
                    f"{len(sup_ids)} supporting items at the {level} level",
                    f"mean supporting confidence {alt_corr:.3f}",
                    (f"independence {alt_indep:.3f}: "
                     f"{len({f for e in alt_edges for f in _prov(e)})} distinct "
                     f"files behind {sum(len(_prov(e)) for e in alt_edges)} mentions"
                     if alt_edges and any(_prov(e) for e in alt_edges) else
                     "source independence not measurable (no supporting items)"),
                    "score = 0.4*corroboration + 0.3*independence + "
                    "0.3*contradiction_absence",
                ] + ([] if sup_ids else ["no supporting evidence exists at this level"]),
                factors=_factors(corroboration=alt_corr, independence=alt_indep,
                               contradiction_absence=alt_contra_abs),
                supporting_count=len(sup_ids),
                contradicting_count=len(contra_ids),
                unknown_count=len(alt_unknowns),
                independence=alt_indep,
                run_id=run_id,
            )
            alts.append(_base_hypothesis(
                hid=alt_id, htype=level, description=desc, reasoning=reason,
                confidence=alt_conf,
                supporting=sorted(set(sup_ids)),
                contradicting=contra_ids,
                missing=sorted({
                    f"{m.get('source_id','')}->{m.get('target_id','')}:"
                    f"{m.get('expected_relation','')}" for m in sup_missing
                }),
                alternatives=[],      # filled just below
                falsifiers=build_falsifiers(
                    family_key=alt_id, members=members, edges=alt_edges,
                    contradictions=contra_on, missing=sup_missing,
                    entities_by_id=entities_by_id, id_map=id_map,
                    adversarial_files=adversarial_files,
                ),
                sensitivity=build_sensitivity(
                    members=members, edges=alt_edges, entities_by_id=entities_by_id,
                    contradiction_absence=alt_contra_abs, independence=alt_indep,
                ),
                independence=alt_indep,
                pattern_matches=sig_ids,
                run_id=run_id,
                community_id=community_id,
                generation_basis="community_alternative",
            ))

        alt_ids = [a["id"] for a in alts]
        primary["alternatives"] = alt_ids
        for i, alt in enumerate(alts):
            alt["alternatives"] = [primary["id"]] + [a["id"] for j, a in enumerate(alts)
                                                     if j != i]

        out.append(primary)
        out.extend(alts)

    out.append(_null_hypothesis(run_id=run_id, all_edges=all_edges,
                               missing_keys=sorted({
                                   f"{m.get('source_id','')}->{m.get('target_id','')}:"
                                   f"{m.get('expected_relation','')}"
                                   for m in missing_edges
                               }),
                               sig_ids=sorted({s["signal_id"] for s in signals
                                               if s.get("signal_id")}),
                               primary_ids=primary_ids))
    return out


def _null_hypothesis(*, run_id: str, all_edges: Sequence[dict],
                    missing_keys: List[str], sig_ids: List[str],
                    primary_ids: List[str]) -> dict:
    """RULE 4: 'nothing criminal occurred' is always a competing hypothesis.

    Its confidence is built from the same measurements as the others but
    weighted the other way: it rises as source independence and
    corroboration fall (concentrated, weakly-corroborated structure is
    exactly what shared-evidence co-mention would produce).
    """
    independence = evidence_independence(all_edges)
    mean_corr = (sum(_edge_score(e) for e in all_edges) / len(all_edges)
                 if all_edges else 0.0)

    shared_concentration = round(1.0 - independence, 6)
    absent_corroboration = round(1.0 - mean_corr, 6)
    factors = [
        ConfidenceFactor(
            factor_type="shared_source_concentration",
            value=shared_concentration,
            weight=0.7,
            description=(f"Supporting structure is concentrated in few source files "
                         f"(independence {independence:.3f})"),
        ),
        ConfidenceFactor(
            factor_type="absence_of_corroboration",
            value=absent_corroboration,
            weight=0.3,
            description=f"Mean link confidence {mean_corr:.3f} across the structure",
        ),
    ]
    confidence = build_confidence(
        basis=[
            "null confidence = 0.7*(1 - independence) + 0.3*(1 - mean_confidence)",
            f"independence {independence:.3f} over {len(all_edges)} links",
            f"mean link confidence {mean_corr:.3f}",
        ],
        factors=factors,
        supporting_count=0,
        contradicting_count=0,
        unknown_count=2,
        independence=independence,
        run_id=run_id,
    )

    falsifiers: List[dict] = []
    for k in missing_keys[:5]:
        rel = k.rsplit(":", 1)[-1] if ":" in k else k
        falsifiers.append(_falsifier(
            generate_id("FALS", f"{NULL_HYPOTHESIS_ID}:coordination:{k}"),
            f"Establish the expected {rel} relation directly — a confirmed link "
            f"would defeat the claim that the structure is incidental",
            f"Direct evidence of {rel} between the named parties",
            "possible", "reject_hypothesis",
        ))
    if all_edges:
        falsifiers.append(_falsifier(
            generate_id("FALS", f"{NULL_HYPOTHESIS_ID}:corroboration"),
            "Corroborate the observed links from a source independent of the "
            "ingested files",
            "An independent record of the interactions the graph asserts",
            "possible", "reject_hypothesis",
        ))
    if not falsifiers:
        # Degenerate corpus: no missing relations and no links at all.
        falsifiers.append(_falsifier(
            generate_id("FALS", f"{NULL_HYPOTHESIS_ID}:decisive_record"),
            "Recover a record that positively establishes coordinated activity "
            "among the group",
            "An observation of coordination, absent from the current corpus",
            "possible", "reject_hypothesis",
        ))

    return _base_hypothesis(
        hid=NULL_HYPOTHESIS_ID,
        htype="event",
        description=("No coordinated activity occurred; the observed structure is "
                     "explained by shared-evidence co-mention alone."),
        reasoning=(
            f"The null claims the {len(all_edges)}-link structure arises without "
            f"coordination. Its confidence rises as source independence falls "
            f"({independence:.3f}) and as link corroboration falls "
            f"({mean_corr:.3f}), because concentrated, weakly corroborated structure "
            f"is precisely what co-mention produces. It must always remain a "
            f"competing hypothesis (RULE 4). Nothing in this corpus rules it "
            f"out — every observation here is also explainable as shared-source "
            f"co-mention — so it reports no contradicting evidence rather than "
            f"pretending the question has been settled."
        ),
        confidence=confidence,
        # Nothing in this corpus rules the null out: every observation here
        # is also explainable as shared-source co-mention, which is exactly
        # why RULE 4 keeps the null permanently on the table.
        supporting=[],
        contradicting=[],
        missing=missing_keys,
        alternatives=list(primary_ids),
        falsifiers=falsifiers,
        sensitivity={
            "key_assumptions": [{
                "assumption": "shared-evidence co-mention fully explains the structure",
                "if_false": "substantive hypotheses gain the confidence the null loses",
                "testability": "testable",
            }],
            "identity_sensitivity": None,
        },
        independence=independence,
        pattern_matches=sig_ids,
        run_id=run_id,
        community_id="",
        generation_basis="null_hypothesis",
    )


# ── Stage entry point ─────────────────────────────────────────────────

def generate(output_dir: str | Path, run_id: str = "") -> Dict[str, Any]:
    """Stage 7: build hypotheses.json and return a stage summary."""
    output_dir = Path(output_dir)
    data = load_inputs(output_dir)
    hypotheses = build_hypotheses(data, run_id)

    path = output_dir / "hypotheses.json"
    with open(path, "w") as fh:
        json.dump(hypotheses, fh, indent=2, default=str)

    primaries = [h for h in hypotheses if h["generation_basis"] == "community"]
    alternatives = [h for h in hypotheses if h["generation_basis"] == "community_alternative"]
    level_counts = Counter(h["type"] for h in hypotheses)
    return {
        "run_id": run_id,
        "hypotheses": len(hypotheses),
        "primary_hypotheses": len(primaries),
        "alternative_hypotheses": len(alternatives),
        "null_hypothesis": 1 if any(h["id"] == NULL_HYPOTHESIS_ID
                                    for h in hypotheses) else 0,
        "communities_seen": len(data["communities"]),
        "type_counts": dict(sorted(level_counts.items())),
        "output": "hypotheses.json",
    }
