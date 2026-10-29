"""Stage 8: Contradiction Adjudication.

Stage 3 detects contradictions. Stage 8 decides what they *are* before any
consumer acts on them, and routes the ones it cannot decide to Stage 9.

Semantic boundary
-----------------

RESOLVED means: after applying admissible evidence, exactly one asserted value
remains standing for the disputed key, and the survivor plus the evidence that
eliminated the others are written to the record. Three conditions, all required:

  1. single survivor   -- |standing values| == 1, not "fewest" and not
                          "best supported";
  2. externally decided-- every elimination was caused by evidence computed
                          without reference to this contradiction;
  3. recorded          -- resolved/resolution_note/resolved_by/resolution_evidence.

Not resolution (explicit negatives): low severity, low impact on a hypothesis,
Stage 3 having merged the entities (the merge created the conflict), a
hypothesis absorbing the conflict into its contradiction_absence factor, or
`resolvable: false` (the opposite verdict, a different field).

ADMISSIBLE EVIDENCE is a closed whitelist. Every entry must be:

  * independent  -- produced by another stage without seeing this record;
  * discriminating -- actually bears on the disputed key;
  * traceable    -- carries a stable id Stage 10 can audit.

  A1 source_disqualification -- a value whose *every* asserting source is Stage 1
      `is_suspicious` loses standing (one clean source keeps it standing).
  A2 provenance_absence      -- a value with no source record loses standing.

Rejected: the per-source-type reliability matrix (a prior, not evidence about
this conflict), name similarity, temporal leeway, amount tolerance (tuned
thresholds), corroboration counts (plurality is not truth and cannot yield a
single survivor), merge_confidence and hypothesis confidence (both circular),
severity labels (a summary cannot settle what it summarizes), LLM verdicts
(untracked and disabled here).

Corroboration and reliability are therefore admitted for *ranking*
(resolution_suggestion) but never for survivorship.

RESOLVABILITY: `resolvable` is true iff the surviving values are asserted by
more than one source class. Records of one class share one failure mode, so
more of them can only outnumber, never decide; a different record class has an
independent relationship to the key. When one class contradicts itself the
conflict is reported as irreducible and Stage 10 must carry it as a limitation.

IMPACT: no band names. `impact_on_hypothesis` is the confidence the affected
primary hypothesis would gain if this contradiction were resolved, recomputed
from Stage 7's own contradiction_absence weight rather than classified.

ORDERING: Stage 8 runs after Stage 7, so hypotheses.json in run N reflects the
contradiction state entering run N. The post-resolution projection travels
inside this stage's own output; Stage 8 never rewrites Stage 7's file.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

from ..hypothesis.engine import W_CONTRADICTION
from ..models.schema import generate_id

STAGE8_FIELDS: Tuple[str, ...] = (
    "description",
    "hypothesis_id",
    "impact_on_hypothesis",
    "resolvable",
    "resolution_evidence",
    "resolved_by",
    "resolution_suggestion",
)

RULE_SOURCE_DISQUALIFICATION = "source_disqualification"
RULE_PROVENANCE_ABSENCE = "provenance_absence"

UNKNOWN_SOURCE_CLASS = "unknown"


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
    """Read the Stage 2/3/5/6/7 outputs Stage 8 adjudicates over."""
    out = Path(output_dir)
    contradictions = _read(out, "contradictions.json", {})
    hypotheses = _read(out, "hypotheses.json", [])
    summary = _read(out, "extraction_summary.json", {})
    extraction = _read(out, "extraction_output.json", {})
    communities = _read(out, "community_assignments.json", [])
    resolved = _read(out, "resolved_entities.json", {})
    id_map = _read(out, "id_map.json", {})

    if isinstance(contradictions, list):
        contradictions = {str(c.get("id", "")): c for c in contradictions
                          if isinstance(c, dict) and c.get("id")}
    elif not isinstance(contradictions, dict):
        contradictions = {}
    contradictions = {str(k): v for k, v in contradictions.items()
                      if isinstance(v, dict)}

    entities = extraction.get("entities", []) if isinstance(extraction, dict) else []
    if not isinstance(entities, list):
        entities = []
    entities_by_id = {e.get("id"): e for e in entities if isinstance(e, dict)
                      and e.get("id")}

    ingestion = summary.get("ingestion_summary", {}) if isinstance(summary, dict) else {}
    adversarial = ingestion.get("adversarial", {}) or {}
    adversarial_files = set(adversarial.get("suspicious_files", []) or [])

    source_types: Dict[str, str] = {}
    for entry in (ingestion.get("files", []) or []):
        if isinstance(entry, dict) and entry.get("name"):
            source_types[str(entry["name"])] = str(
                entry.get("source_type") or UNKNOWN_SOURCE_CLASS)

    return {
        "contradictions": contradictions,
        "hypotheses": hypotheses if isinstance(hypotheses, list) else [],
        "communities": communities if isinstance(communities, list) else [],
        "entities_by_id": entities_by_id,
        "resolved_entities": resolved if isinstance(resolved, dict) else {},
        "id_map": {str(k): str(v) for k, v in (id_map or {}).items()}
        if isinstance(id_map, dict) else {},
        "adversarial_files": adversarial_files,
        "source_types": source_types,
    }


# ── Provenance attribution ────────────────────────────────────────────

def _resolve(entity_id: str, id_map: Dict[str, str]) -> str:
    if not entity_id:
        return ""
    return id_map.get(entity_id, entity_id)


def attribution_available(contradiction: dict, entities_by_id: dict) -> bool:
    """True when the extraction records Stage 8 needs are actually present.

    Without this guard a missing or malformed extraction_output.json would
    look like 'every value has no provenance' and A2 would erase the record
    wholesale instead of reporting an input failure.
    """
    ids = contradiction.get("entity_ids") or []
    return bool(ids) and any(eid in entities_by_id for eid in ids)


def value_sources(contradiction: dict, entities_by_id: dict) -> Dict[str, List[str]]:
    """Map each disputed value to the source files that assert it.

    Stage 3 built `values` from the same entity group this walks, so a value
    with no matching entity genuinely has no provenance record behind it.
    """
    attribute = str(contradiction.get("attribute") or "name")
    out: Dict[str, set] = {str(v): set() for v in (contradiction.get("values") or [])}

    for raw_id in (contradiction.get("entity_ids") or []):
        record = entities_by_id.get(raw_id)
        if not isinstance(record, dict):
            continue
        if attribute == "name":
            value = record.get("name", "")
        else:
            attributes = record.get("attributes") or {}
            value = attributes.get(attribute, "")
            if value in (None, ""):
                value = record.get("name", "")
        if value in (None, ""):
            continue
        value = str(value)
        if value not in out:
            continue
        source = record.get("source_id") or ""
        if not source:
            source = (record.get("source") or {}).get("file_name", "") or ""
        if source:
            out[value].add(str(source))

    return {k: sorted(v) for k, v in out.items()}


# ── Adjudication ──────────────────────────────────────────────────────

def _ratio(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return max(0.0, min(1.0, numerator / denominator))


def adjudicate(contradiction: dict,
               sources: Dict[str, List[str]],
               adversarial_files: set,
               source_types: Dict[str, str],
               can_attribute: bool) -> Dict[str, Any]:
    """Apply the admissible whitelist and return the Stage 8 verdict."""
    values = [str(v) for v in (contradiction.get("values") or [])]
    attribute = str(contradiction.get("attribute") or "name")
    adversarial = set(adversarial_files)

    eliminated: List[dict] = []
    standing: List[str] = []

    if can_attribute:
        for value in values:
            asserting = set(sources.get(value, []))
            if not asserting:
                eliminated.append({"value": value,
                                   "rule": RULE_PROVENANCE_ABSENCE,
                                   "evidence": []})
            elif asserting <= adversarial:
                eliminated.append({"value": value,
                                   "rule": RULE_SOURCE_DISQUALIFICATION,
                                   "evidence": sorted(asserting)})
            else:
                standing.append(value)
    else:
        standing = list(values)

    resolved = len(standing) == 1 and bool(eliminated)

    standing_sources = {s for value in standing
                        for s in sources.get(value, [])}
    standing_classes = {source_types.get(s, UNKNOWN_SOURCE_CLASS)
                        for s in standing_sources}

    if resolved:
        survivor = standing[0]
        rules = sorted({e["rule"] for e in eliminated})
        detail = "; ".join(f"'{e['value']}' eliminated by {e['rule']}"
                           for e in eliminated)
        return {
            "resolved": True,
            "resolution_note": (
                f"Exactly one value retains admissible provenance: '{survivor}' "
                f"asserted by {', '.join(sources.get(survivor, []))}. "
                f"{len(eliminated)} competing value(s) removed: {detail}."
            ),
            "resolved_by": "|".join(rules),
            "resolvable": False,
            "resolution_suggestion": "",
            "resolution_evidence": {
                "admissible_rules": rules,
                "standing_values": standing,
                "eliminated_values": eliminated,
                "standing_sources": sorted(standing_sources),
                "standing_source_classes": sorted(standing_classes),
                "provenance_available": can_attribute,
            },
        }

    if not can_attribute:
        suggestion = (
            "Adjudication skipped: extraction provenance for this record's "
            "entities is unavailable, so no value can be attributed to a source."
        )
    elif not standing:
        suggestion = (
            f"No admissible value remains for '{attribute}'. Re-acquire "
            f"{attribute} records for this entity before any consumer treats it "
            f"as known."
        )
    elif len(standing_classes) >= 2:
        suggestion = (
            f"Obtain a record stating '{attribute}' for this entity from a "
            f"source class outside {{{', '.join(sorted(standing_classes))}}}. "
            f"Surviving values {{{', '.join(standing)}}} are asserted by "
            f"{{{', '.join(sorted(standing_sources))}}}; the conflict spans "
            f"record classes, so an independent class can decide it."
        )
    else:
        only = ", ".join(sorted(standing_classes))
        suggestion = (
            f"Surviving values {{{', '.join(standing)}}} all originate from "
            f"source class '{only}', which contradicts itself. Further records "
            f"of that class can only outnumber, never decide; settling "
            f"'{attribute}' requires evidence from outside this corpus."
        )

    return {
        "resolved": False,
        "resolution_note": "",
        "resolved_by": "",
        "resolvable": bool(standing) and len(standing_classes) >= 2,
        "resolution_suggestion": suggestion,
        "resolution_evidence": {
            "admissible_rules": sorted({e["rule"] for e in eliminated}),
            "standing_values": standing,
            "eliminated_values": eliminated,
            "standing_sources": sorted(standing_sources),
            "standing_source_classes": sorted(standing_classes),
            "provenance_available": can_attribute,
        },
    }


# ── Hypothesis linkage and impact ─────────────────────────────────────

def community_baselines(data: Dict[str, Any]) -> Dict[str, Tuple[int, int]]:
    """community_id -> (unresolved contradictions touching it, member count).

    Computed once from the pre-adjudication state so every impact figure on
    this run shares one baseline instead of drifting as records are updated.
    """
    contradictions = list(data["contradictions"].values())
    id_map = data["id_map"]
    baselines: Dict[str, Tuple[int, int]] = {}
    for comm in data["communities"]:
        community_id = comm.get("community_id", "")
        members = {m for m in comm.get("node_ids", []) if m}
        if not community_id or not members:
            continue
        touching = 0
        for c in contradictions:
            if c.get("resolved"):
                continue
            resolved_ids = {_resolve(x, id_map)
                            for x in (c.get("entity_ids") or [])}
            if resolved_ids & members:
                touching += 1
        baselines[community_id] = (touching, len(members))
    return baselines


def link_hypothesis(contradiction: dict,
                    data: Dict[str, Any],
                    baselines: Dict[str, Tuple[int, int]],
                    run_id: str) -> Dict[str, Any]:
    """Attach the affected primary hypothesis and its confidence gain."""
    id_map = data["id_map"]
    resolved_ids = {_resolve(x, id_map)
                    for x in (contradiction.get("entity_ids") or [])}

    best_community = ""
    best_overlap = 0
    for comm in data["communities"]:
        community_id = comm.get("community_id", "")
        members = {m for m in comm.get("node_ids", []) if m}
        overlap = len(resolved_ids & members)
        if not community_id or overlap <= 0:
            continue
        if overlap > best_overlap or (overlap == best_overlap
                                      and (not best_community
                                           or community_id < best_community)):
            best_overlap = overlap
            best_community = community_id

    if not best_community:
        return {"hypothesis_id": "", "impact_on_hypothesis": 0.0}

    primary = next(
        (h for h in data["hypotheses"]
         if h.get("generation_basis") == "community"
         and h.get("community_id") == best_community),
        None,
    )
    hypothesis_id = (primary or {}).get("id") or generate_id(
        "HYP", f"{run_id}:{best_community}:primary")

    touching, members = baselines.get(best_community, (0, 0))
    if touching <= 0 or members <= 0:
        return {"hypothesis_id": hypothesis_id, "impact_on_hypothesis": 0.0}

    impact = round(W_CONTRADICTION * _ratio(1, members), 6)
    return {"hypothesis_id": hypothesis_id, "impact_on_hypothesis": impact}


# ── Description ───────────────────────────────────────────────────────

def _describe(contradiction: dict, verdict: dict,
              data: Dict[str, Any]) -> str:
    """Derive a human-readable description from the record itself."""
    attribute = str(contradiction.get("attribute") or "name")
    entity_label = ""
    for raw_id in (contradiction.get("entity_ids") or []):
        canonical = _resolve(raw_id, data["id_map"])
        record = data["resolved_entities"].get(canonical)
        if isinstance(record, dict) and record.get("canonical_name"):
            entity_label = str(record["canonical_name"])
            break
    if not entity_label:
        ids = contradiction.get("entity_ids") or []
        entity_label = ids[0] if ids else "an unidentified entity"

    values = verdict["resolution_evidence"]["standing_values"]
    eliminated = verdict["resolution_evidence"]["eliminated_values"]
    text = (f"{len(values)} source-backed value(s) for '{attribute}' of "
            f"{entity_label} remain in dispute: "
            f"{', '.join(repr(v) for v in values)}.")
    if eliminated:
        text += (" " + f"{len(eliminated)} value(s) eliminated by admissible "
                 f"evidence: "
                 + "; ".join(f"{e['value']} ({e['rule']})"
                             for e in eliminated) + ".")
    if verdict["resolved"]:
        text += f" Resolved: {verdict['resolution_note']}"
    return text


# ── Entry point ───────────────────────────────────────────────────────

def adjudicate_all(data: Dict[str, Any], run_id: str) -> Dict[str, dict]:
    baselines = community_baselines(data)
    updated: Dict[str, dict] = {}
    for contradiction_id, contradiction in data["contradictions"].items():
        record = dict(contradiction)
        can_attribute = attribution_available(record, data["entities_by_id"])
        sources = value_sources(record, data["entities_by_id"])
        verdict = adjudicate(record, sources, data["adversarial_files"],
                             data["source_types"], can_attribute)
        record.update(verdict)
        record.update(link_hypothesis(record, data, baselines, run_id))
        record["description"] = _describe(record, verdict, data)
        updated[contradiction_id] = record
    return updated


def _summarise(updated: Dict[str, dict], run_id: str) -> Dict[str, Any]:
    rules: Dict[str, int] = {}
    eliminated = 0
    linked = 0
    for record in updated.values():
        evidence = record.get("resolution_evidence") or {}
        for rule in (evidence.get("admissible_rules") or []):
            rules[rule] = rules.get(rule, 0) + 1
        eliminated += len(evidence.get("eliminated_values") or [])
        if record.get("hypothesis_id"):
            linked += 1
    return {
        "run_id": run_id,
        "contradictions": len(updated),
        "resolved": sum(1 for r in updated.values() if r.get("resolved")),
        "unresolved": sum(1 for r in updated.values() if not r.get("resolved")),
        "resolvable": sum(1 for r in updated.values()
                          if r.get("resolvable")),
        "irreducible": sum(1 for r in updated.values()
                           if not r.get("resolved") and not r.get("resolvable")),
        "eliminated_values": eliminated,
        "rules_fired": dict(sorted(rules.items())),
        "linked_hypotheses": linked,
        "output": "contradictions.json",
    }


def generate(output_dir: str | Path, run_id: str = "") -> Dict[str, Any]:
    """Stage 8: adjudicate contradictions.json and return a stage summary."""
    output_dir = Path(output_dir)
    data = load_inputs(output_dir)
    updated = adjudicate_all(data, run_id)

    path = output_dir / "contradictions.json"
    with open(path, "w") as fh:
        json.dump(updated, fh, indent=2, default=str)

    return _summarise(updated, run_id)
