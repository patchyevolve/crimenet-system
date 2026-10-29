"""Stage 10: bounded review of pipeline outputs.

The Critic can flag records for human review. It never edits analytical stores,
creates evidence, decides guilt, or treats absent data as a negative finding.
"""

import json
from pathlib import Path

from ..ai.caller import AICaller


def _load(path: Path, default):
    if not path.exists():
        return default
    try:
        with path.open(encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return default


def _records(value) -> list:
    if isinstance(value, dict):
        return [dict(item, id=item.get("id", key)) if isinstance(item, dict) else item
                for key, item in value.items()]
    return value if isinstance(value, list) else []


def _context(out: Path) -> tuple[dict, set[str], set[str]]:
    hypotheses = _records(_load(out / "hypotheses.json", []))
    contradictions = _records(_load(out / "contradictions.json", {}))
    gaps = _records(_load(out / "evidence_gaps.json", []))
    edges = _records(_load(out / "graph_edges.json", []))
    entities = _records(_load(out / "resolved_entities.json", {}))

    all_evidence_ids = {
        str(evidence_id)
        for edge in edges if isinstance(edge, dict)
        for evidence_id in edge.get("supporting_evidence", [])
        if evidence_id
    }
    all_evidence_ids.update(
        str(evidence_id)
        for row in contradictions if isinstance(row, dict)
        for key in ("evidence_ids", "supporting_evidence", "contradicting_evidence")
        for evidence_id in row.get(key, []) if evidence_id
    )
    # Keep the model prompt within small-model context limits. Highest-scored
    # hypotheses/gaps are included first; IDs are whitelisted only if they are
    # actually present in this bounded prompt.
    def slim(rows, fields):
        return [{k: row[k] for k in fields if k in row} for row in rows[:60]
                if isinstance(row, dict)]

    def confidence_score(row):
        value = row.get("confidence", 0)
        if isinstance(value, dict):
            value = value.get("score", 0)
        return value if isinstance(value, (int, float)) else 0

    hypotheses.sort(key=confidence_score, reverse=True)
    gaps.sort(key=lambda row: row.get("impact_on_hypothesis", 0)
              if isinstance(row, dict) and isinstance(row.get("impact_on_hypothesis"), (int, float)) else 0,
              reverse=True)
    hypotheses = hypotheses[:15]
    contradictions = contradictions[:15]
    gaps = gaps[:15]
    edges = edges[:20]
    entities = entities[:40]

    model_hypotheses = []
    for row in hypotheses:
        if not isinstance(row, dict):
            continue
        confidence = row.get("confidence", 0)
        if isinstance(confidence, dict):
            confidence = confidence.get("score", 0)
        model_hypotheses.append({
            "id": row.get("id", ""),
            "type": row.get("type", ""),
            "description": str(row.get("description", ""))[:240],
            "confidence_score": confidence,
            "supporting_ids": [str(x) for x in row.get("supporting", [])[:5]],
            "contradicting_ids": [str(x) for x in row.get("contradicting", [])[:5]],
        })
    payload = {
        "hypotheses": model_hypotheses[:15],
        "contradictions": slim(contradictions, ["id", "type", "description", "status",
                                                 "evidence_ids", "supporting_evidence",
                                                 "contradicting_evidence"]),
        "gaps": slim(gaps, ["id", "kind", "subject", "requirement", "affects_hypothesis",
                            "impact_on_hypothesis", "reach"]),
        "graph_edges": slim(edges, ["id", "source_id", "target_id", "confidence",
                                     "supporting_evidence", "epistemic_status",
                                     "derivation_depth"]),
        "resolved_entity_ids": sorted(str(row["id"]) for row in entities
                                      if isinstance(row, dict) and row.get("id"))[:100],
        "known_evidence_ids": sorted(all_evidence_ids)[:80],
    }
    target_ids = {
        str(row["id"])
        for key in ("hypotheses", "contradictions", "gaps", "graph_edges")
        for row in payload[key] if row.get("id")
    } | set(payload["resolved_entity_ids"])
    evidence_ids = set(payload["known_evidence_ids"])
    evidence_ids.update(str(ref) for edge in payload["graph_edges"]
                        for ref in edge.get("supporting_evidence", []) if ref)
    return payload, target_ids, evidence_ids


def _heuristic_checks(payload: dict) -> list[dict]:
    checks = []
    for hypothesis in payload["hypotheses"]:
        confidence = hypothesis.get("confidence_score")
        if not isinstance(confidence, (int, float)) or not 0 <= confidence <= 1:
            checks.append({"code": "invalid_hypothesis_confidence", "target_id": hypothesis["id"]})
        if not hypothesis.get("supporting_ids") and not hypothesis.get("contradicting_ids"):
            checks.append({"code": "hypothesis_without_cited_support", "target_id": hypothesis["id"]})
    for gap in payload["gaps"]:
        refs = gap.get("affects_hypothesis", [])
        if any(ref not in {h["id"] for h in payload["hypotheses"]} for ref in refs):
            checks.append({"code": "gap_references_unknown_hypothesis", "target_id": gap["id"]})
    return checks


def generate(output_dir: str, run_id: str = "", ai: AICaller | None = None) -> dict:
    """Create ``critic_review.json``; all model-suggested references are validated."""
    out = Path(output_dir)
    payload, target_ids, evidence_ids = _context(out)
    review = {"findings": []}
    llm_status = "disabled"
    provider = model = None
    llm_error = None

    if ai:
        result = ai.critic_review(
            text=json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
            source_file="pipeline_analytical_outputs",
            max_tokens=1800,
        )
        llm_status = result.get("status", "failed")
        provider, model = result.get("provider"), result.get("model")
        llm_error = str(result.get("error") or "")[:400] or None
        parsed = result.get("parsed")
        if llm_status == "success" and isinstance(parsed, dict):
            for finding in parsed.get("findings", []):
                if not isinstance(finding, dict):
                    continue
                targets = [str(x) for x in finding.get("target_ids", []) if str(x) in target_ids]
                cited = [str(x) for x in finding.get("evidence_ids", []) if str(x) in evidence_ids]
                if not targets or not finding.get("finding"):
                    continue
                review["findings"].append({
                    "severity": finding.get("severity") if finding.get("severity") in {"info", "review", "high"} else "review",
                    "category": str(finding.get("category", "other"))[:60],
                    "target_ids": targets,
                    "evidence_ids": cited,
                    "finding": str(finding["finding"])[:1000],
                    "source": "llm_review",
                })
        elif llm_status == "success":
            llm_status = "invalid_response"

    result = {
        "run_id": run_id,
        "mode": "llm_assisted" if llm_status == "success" else "structural_checks_only",
        "llm_status": llm_status,
        "provider": provider,
        "model": model,
        "llm_error": llm_error,
        "findings": review["findings"],
        "structural_checks": _heuristic_checks(payload),
        "target_count": len(target_ids),
        "evidence_reference_count": len(evidence_ids),
        "changes_applied": False,
        "human_review_required": True,
    }
    with (out / "critic_review.json").open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    return {"mode": result["mode"], "llm_status": llm_status,
            "findings": len(result["findings"]),
            "structural_checks": len(result["structural_checks"])}
