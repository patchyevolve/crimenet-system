"""Baseline evaluation — the gate R1 requires before any neural model.

ML_ENGINE.md R1.3: *"report AUPRC, calibration, coverage, and performance by
entity/source subgroup. Use case-level splits and confidence intervals. AUC
alone is not an acceptance criterion."*

The sequence here is fixed and is the reason leakage is structurally
impossible rather than merely avoided:

1. :func:`~ml.labels.inventory_labels` decides, before anything is scored,
   whether labels exist at all. If they do not, the run reports blockers and
   stops. It does not invent negatives to make a curve possible.
2. The **test** edges are withheld *first*, then the snapshot is built. Every
   feature and every score is therefore computed on a graph that does not
   contain the edge being predicted. Scoring against the full graph would let
   common-neighbour counts read the answer directly.
3. Positives are the held-out observed edges; negatives are only the pairs
   whose absence Stage 5 recorded. Pairs that are merely unobserved are never
   scored as negatives.
4. Metrics are reported per scorer with a bootstrap confidence interval and a
   subgroup breakdown, so a single favourable number cannot stand in for the
   whole picture.

The result is a verdict, and "insufficient labels" is a legitimate outcome.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .baseline import DEFAULT_BASELINE, describe as describe_scorers, score
from .contracts import (
    EPISTEMIC_PREDICTION,
    PREDICTION_SCHEMA_VERSION,
    LinkPrediction,
    Provenance,
)
from .labels import LabelInventory, inventory_labels
from .metrics import (
    bootstrap_metric,
    average_precision,
    score_classification_report,
    subgroup_reports,
)
from .snapshot import load_snapshot

__all__ = [
    "evaluate_baseline",
    "scored_candidates",
]

_MODEL_NAME = "structural_link_baseline"
_MODEL_VERSION = "1.0.0"
_BOOTSTRAP_SAMPLES = 500
_BOOTSTRAP_SEED = 20260925


def _relation_of(record: Mapping[str, Any], *, positive: bool) -> str:
    field = "relationship_type" if positive else "expected_relation"
    return str(record.get(field) or "unknown")


def _load_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict):
        return [value for value in payload.values() if isinstance(value, dict)]
    return []


def scored_candidates(
    output_dir: str | Path,
    inventory: LabelInventory,
    *,
    run_id: str,
    scorers: Sequence[str],
) -> dict[str, Any]:
    """Score held-out positives and known-absent negatives on a withheld graph."""
    root = Path(output_dir)
    positives_by_id = {item.record_id: item for item in inventory.positives}
    negatives = list(inventory.negatives)

    test_ids = inventory.test_edge_ids
    held_out = [positives_by_id[eid] for eid in test_ids if eid in positives_by_id]

    # Withhold the test edges BEFORE the snapshot exists. This is the whole
    # leakage defence: nothing downstream can see an edge it is scored against.
    snapshot = load_snapshot(
        root, run_id=run_id, withheld_edge_ids=[item.record_id for item in held_out]
    )

    edge_relations: dict[str, str] = {}
    for record in _load_records(root / "graph_edges.json"):
        edge_relations[str(record.get("id", ""))] = _relation_of(record, positive=True)

    candidates: list[dict[str, Any]] = []
    for item in held_out:
        candidates.append(
            {
                "label": 1,
                "left": item.source_id,
                "right": item.target_id,
                "record_id": item.record_id,
                "group": edge_relations.get(item.record_id, "unknown"),
                "withheld": True,
            }
        )
    for item in negatives:
        candidates.append(
            {
                "label": 0,
                "left": item.source_id,
                "right": item.target_id,
                "record_id": item.record_id,
                "group": "missing_expected_relation",
                "withheld": False,
            }
        )

    scorable = [
        c for c in candidates if c["left"] in snapshot.adjacency and c["right"] in snapshot.adjacency
    ]
    unscorable = [c for c in candidates if c not in scorable]

    per_scorer: dict[str, Any] = {}
    predictions: list[LinkPrediction] = []
    provenance = Provenance(
        schema_version=PREDICTION_SCHEMA_VERSION,
        model_name=_MODEL_NAME,
        model_version=_MODEL_VERSION,
        run_id=run_id,
        input_snapshot=snapshot.hash,
        supporting_record_ids=tuple(sorted({c["record_id"] for c in scorable})),
        created_at=datetime.now(timezone.utc).isoformat(),
    )

    for name in scorers:
        labels = [int(c["label"]) for c in scorable]
        scores = [score(snapshot, c["left"], c["right"], scorer=name) for c in scorable]
        if name == DEFAULT_BASELINE:
            for candidate, value in zip(scorable, scores):
                predictions.append(
                    LinkPrediction(
                        provenance=provenance,
                        source_id=candidate["left"],
                        target_id=candidate["right"],
                        score=value,
                        calibration_status="not_measured",
                        epistemic_status=EPISTEMIC_PREDICTION,
                    )
                )
        report: dict[str, Any] = score_classification_report(labels, scores)
        try:
            interval = bootstrap_metric(
                labels,
                scores,
                average_precision,
                samples=_BOOTSTRAP_SAMPLES,
                seed=_BOOTSTRAP_SEED,
            )
            report["average_precision_ci95"] = [round(interval[0], 6), round(interval[1], 6)]
        except ValueError as exc:
            report["average_precision_ci95"] = None
            report["ci_note"] = str(exc)
        report["subgroups"] = subgroup_reports(
            labels, scores, [str(c["group"]) for c in scorable]
        )
        per_scorer[name] = report

    return {
        "snapshot": snapshot.summary(),
        "withheld_edge_ids": [item.record_id for item in held_out],
        "candidates": len(candidates),
        "scorable": len(scorable),
        "unscorable": len(unscorable),
        "unscorable_examples": [
            {"record_id": c["record_id"], "left": c["left"], "right": c["right"]}
            for c in unscorable[:10]
        ],
        "scorers": per_scorer,
        "predictions": [p.to_dict() for p in predictions],
        "default_scorer": DEFAULT_BASELINE,
    }


def _acceptance_findings(
    metrics: Mapping[str, Any], *, default_scorer: str
) -> list[dict[str, Any]]:
    """Decide whether the numbers could support an acceptance gate.

    The test is derived from the measurements themselves, not from a
    predeclared count: **can the baseline be distinguished from a random
    ranking?** That is answered by whether the bootstrap confidence interval
    on AUPRC sits entirely above the no-skill reference. If the no-skill rate
    falls inside the interval, the ranking carries no demonstrated signal
    however high the point estimate looks — which is exactly the trap a bare
    AUPRC of 0.95 sets when 81% of the sample is already positive.

    A finding is raised rather than silently failing the gate, so the report
    explains *why* the numbers do not license a decision.
    """
    findings: list[dict[str, Any]] = []
    primary = metrics.get(default_scorer)
    if not primary:
        return findings

    no_skill = float(primary["no_skill_ap"])
    interval = primary.get("average_precision_ci95")
    ap = float(primary["average_precision"])
    auc = float(primary["auc_roc"])

    if interval is None:
        findings.append(
            {
                "code": "no_confidence_interval",
                "detail": primary.get(
                    "ci_note", "AUPRC confidence interval could not be computed"
                ),
            }
        )
    else:
        low, high = float(interval[0]), float(interval[1])
        if low <= no_skill <= high:
            findings.append(
                {
                    "code": "indistinguishable_from_random",
                    "detail": (
                        f"bootstrap CI95 on AUPRC is [{low:.4f}, {high:.4f}] and "
                        f"contains the no-skill reference {no_skill:.4f}; the "
                        "baseline cannot be distinguished from a random ranking"
                    ),
                    "average_precision": round(ap, 6),
                    "ci95": [round(low, 6), round(high, 6)],
                    "no_skill": round(no_skill, 6),
                }
            )

    if auc < 0.5:
        findings.append(
            {
                "code": "auc_below_random",
                "detail": (
                    f"AUC-ROC {auc:.4f} is below 0.5: the baseline ranks a "
                    "random positive below a random negative more often than "
                    "not"
                ),
                "auc_roc": round(auc, 6),
            }
        )

    negatives = int(primary["negatives"])
    if negatives:
        findings.append(
            {
                "code": "negative_label_count",
                "detail": (
                    f"{negatives} known-absent negatives support this curve; "
                    "each one carries "
                    f"{1.0 / negatives:.3f} of the ranking signal. This is a "
                    "coverage fact, not a threshold: no operational claim "
                    "should rest on it"
                ),
                "negatives": negatives,
            }
        )
    return findings


def evaluate_baseline(
    output_dir: str | Path,
    *,
    run_id: str = "offline",
    scorers: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Run the full R1.3 evaluation and return the reviewable report.

    Always returns a report. When labels are insufficient the report carries
    ``gate.blockers`` and no metrics, because computing AUPRC without negatives
    would require inventing them.
    """
    inventory = inventory_labels(output_dir)
    names = tuple(scorers) if scorers else (DEFAULT_BASELINE,)

    report: dict[str, Any] = {
        "schema_version": PREDICTION_SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "run_id": run_id,
        "label_inventory": inventory.to_dict(),
        "gate": {
            "computable": inventory.gate_ready,
            "acceptance": "not_computable",
            "blockers": list(inventory.blockers),
            "warnings": list(inventory.warnings),
            "decisive_findings": [],
        },
        "scorer_descriptions": describe_scorers(),
        "scorers_requested": list(names),
        "leakage_controls": {
            "test_edges_withheld_before_snapshot": True,
            "outcome_fields_excluded": [
                "confidence",
                "contradicting_evidence",
                "hypothesis factors",
            ],
            "negative_policy": inventory.parameters.get("negative_policy"),
            "split": "temporal (temporal_info.timestamp), earliest -> train",
        },
    }

    if not inventory.gate_ready:
        report["status"] = "not_evaluated"
        report["metrics"] = None
        report["note"] = (
            "Baseline metrics were not computed: see gate.blockers. Computing "
            "them anyway would require labelling unobserved pairs as "
            "negatives, which Roadmap R1.1 forbids."
        )
        return report

    evaluation = scored_candidates(
        Path(output_dir), inventory, run_id=run_id, scorers=names
    )
    findings = _acceptance_findings(
        evaluation["scorers"], default_scorer=evaluation["default_scorer"]
    )
    report["status"] = "evaluated"
    report["gate"]["acceptance"] = (
        "capable" if not findings else "not_capable"
    )
    report["gate"]["decisive_findings"] = findings
    report["gate"]["meaning_of_acceptance"] = (
        "'capable' means the measurements could support the R1.3 exit gate "
        "(reproducible held-out improvement over baseline). It is not itself "
        "a pass."
    )
    report["snapshot"] = evaluation["snapshot"]
    report["withheld_edge_ids"] = evaluation["withheld_edge_ids"]
    report["candidates"] = evaluation["candidates"]
    report["scorable"] = evaluation["scorable"]
    report["unscorable"] = evaluation["unscorable"]
    report["unscorable_examples"] = evaluation["unscorable_examples"]
    report["metrics"] = evaluation["scorers"]
    report["predictions"] = evaluation["predictions"]
    report["default_scorer"] = evaluation["default_scorer"]
    report["roadmap_conclusion"] = {
        "gnn_justified": False,
        "reason": (
            "R1.4 permits a neural model only after sufficient independent "
            "labels exist and the baseline leaves a material, reproducible "
            "gap. See gate.decisive_findings and gate.blockers."
        ),
        "baseline_retained": True,
    }
    if not findings:
        report["roadmap_conclusion"]["gnn_justified"] = "pending_baseline_gap_test"
    return report
