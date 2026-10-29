"""Stage 12: summarize a declared set of cases without pooling their evidence."""

import json
from pathlib import Path


def _read(path: Path, default):
    try:
        with path.open(encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return default


def generate(output_dir: str, case_results: dict) -> dict:
    root = Path(output_dir)
    index_dir = root / "global_index"
    globals_data = _read(index_dir / "global_entities.json", {})
    links = _read(index_dir / "global_entity_links.json", [])
    alerts = _read(index_dir / "cross_case_alerts.json", [])

    cases = {}
    for case_id, entry in case_results.items():
        summary = entry.get("summary", {})
        case_path = Path(entry["output_dir"])
        local_links = _read(case_path / "global_entity_links.json", [])
        cases[case_id] = {
            "entities": summary.get("total_entities", 0),
            "relations": summary.get("total_relations", 0),
            "global_links": len(local_links),
        }

    cases_by_global = {}
    for link in links:
        gid = link.get("global_entity_id")
        cid = link.get("case_id")
        if gid and cid:
            cases_by_global.setdefault(gid, set()).add(cid)
    shared = [
        {"global_entity_id": gid,
         "canonical_name": globals_data.get(gid, {}).get("canonical_name", ""),
         "case_ids": sorted(case_ids)}
        for gid, case_ids in sorted(cases_by_global.items()) if len(case_ids) > 1
    ]

    result = {
        "stage": 12,
        "scope_type": "declared_case_set",
        "case_count": len(cases),
        "case_metrics": cases,
        "case_metric_totals": {
            "entities": sum(x["entities"] for x in cases.values()),
            "relations": sum(x["relations"] for x in cases.values()),
        },
        "shared_global_entities": shared,
        "cross_case_alerts": alerts,
        "shared_global_entity_count": len(shared),
        "cross_case_alert_count": len(alerts),
        "evidence_pooled": False,
        "risk_scores_combined": False,
    }
    with (root / "scoped_analytics.json").open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    return result
