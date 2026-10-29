"""Anomaly signal detection — finds unusual patterns in the graph data."""

from typing import List, Dict
from collections import defaultdict, Counter
import numpy as np
import networkx as nx

from .models import AnomalySignal
from ..models.schema import generate_id


def compute_anomaly_signals(
    G: nx.Graph,
    centrality_scores: List[dict],
    temporal_infos: List[dict],
    contradictions: dict,
    temporal_contradictions: List[dict],
    coverage_intervals: List[dict],
    adversarial_scores: List[dict],
    run_id: str = "",
) -> List[AnomalySignal]:
    """Detect six types of anomaly signals."""
    signals = []

    # Build lookup structures
    node_types = {n: G.nodes[n].get("node_type", "unknown") for n in G.nodes}
    node_conf = {n: G.nodes[n].get("effective_confidence", 0.5) for n in G.nodes}
    node_provenance = {n: G.nodes[n].get("provenance_chain", []) for n in G.nodes}

    # Temporal count per entity
    temporal_by_entity = defaultdict(int)
    for t in temporal_infos:
        temporal_by_entity[t.get("entity_id", "")] += 1

    # Contradiction count per entity
    contra_by_entity = defaultdict(int)
    for con_id, con in contradictions.items():
        for eid in con.get("entity_ids", []):
            contra_by_entity[eid] += 1

    # Temporal contradiction count per entity
    temp_contra_by_entity = defaultdict(int)
    for tc in temporal_contradictions:
        eid = tc.get("entity_id", "")
        temp_contra_by_entity[eid] += 1

    degrees = dict(G.degree())
    person_nodes = [n for n in G.nodes if node_types.get(n) == "person"]

    # ─── 1. Degree Outlier ───
    if len(person_nodes) >= 5:
        person_degrees = [degrees.get(n, 0) for n in person_nodes]
        mean_deg = np.mean(person_degrees)
        std_deg = np.std(person_degrees)
        if std_deg > 0:
            for n in person_nodes:
                d = degrees.get(n, 0)
                z = (d - mean_deg) / std_deg
                if z > 2.0:
                    severity = "high" if z > 3.0 else ("medium" if z > 2.5 else "low")
                    signals.append(AnomalySignal(
                        signal_id=generate_id("SIG", f"degree_{n}"),
                        signal_type="degree_outlier",
                        severity=severity,
                        entity_id=n,
                        entity_type=node_types.get(n, "unknown"),
                        description=f"Degree {d} exceeds mean+{z:.1f}σ (mean={mean_deg:.1f}, std={std_deg:.1f})",
                        evidence_count=d,
                        score=round(z / 5.0, 4),
                        run_id=run_id,
                    ))

    # ─── 2. Confidence Gap ───
    # Flag ONLY high-confidence nodes that are completely isolated (degree=0)
    # despite having provenance — not the tautological conf>0.8 && d<=1 case
    # (which fired on nearly every well-evidenced leaf).
    for n in G.nodes:
        conf = node_conf.get(n, 0.5)
        d = degrees.get(n, 0)
        prov = node_provenance.get(n, [])
        if conf > 0.75 and d == 0 and len(prov) >= 1:
            signals.append(AnomalySignal(
                signal_id=generate_id("SIG", f"conf_gap_{n}"),
                signal_type="confidence_gap",
                severity="medium",
                entity_id=n,
                entity_type=node_types.get(n, "unknown"),
                description=f"High confidence ({conf:.2f}) but zero connections — possibly missing evidence",
                evidence_count=len(prov),
                score=round(conf * 0.7, 4),
                run_id=run_id,
            ))

    # ─── 3. Contradiction Cluster ───
    # entity_ids already remapped to graph node IDs via id_map in engine.py
    for eid, count in contra_by_entity.items():
        if count >= 3:
            signals.append(AnomalySignal(
                signal_id=generate_id("SIG", f"contra_{eid}"),
                signal_type="contradiction_cluster",
                severity="high" if count >= 5 else "medium",
                entity_id=eid,
                entity_type=node_types.get(eid, "unknown"),
                description=f"Entity involved in {count} identity contradictions — high data conflict",
                evidence_count=count,
                score=round(min(1.0, count / 10.0), 4),
                run_id=run_id,
            ))

    # Also flag unresolved adversarial-score edges as anomaly context
    adv_flagged = [a for a in adversarial_scores if isinstance(a, dict) and a.get("is_suspicious")]
    if adv_flagged:
        # Aggregate by edge endpoints when possible
        for a in adv_flagged[:20]:
            eid = a.get("edge_id", "")
            if not eid:
                continue
            signals.append(AnomalySignal(
                signal_id=generate_id("SIG", f"adv_{eid}"),
                signal_type="adversarial_edge",
                severity="medium" if a.get("overall_score", 0) > 0.4 else "low",
                entity_id=eid,
                entity_type="edge",
                description=f"Adversarial edge score {a.get('overall_score', 0):.2f}: {'; '.join(a.get('flags', [])[:3])}",
                evidence_count=len(a.get("flags", [])),
                score=round(float(a.get("overall_score", 0)), 4),
                run_id=run_id,
            ))

    # ─── 4. Isolated High-Value ───
    for n in G.nodes:
        d = degrees.get(n, 0)
        conf = node_conf.get(n, 0.5)
        prov = node_provenance.get(n, [])
        if d == 0 and conf > 0.7 and len(prov) >= 2:
            signals.append(AnomalySignal(
                signal_id=generate_id("SIG", f"isolated_{n}"),
                signal_type="isolated_high_value",
                severity="medium",
                entity_id=n,
                entity_type=node_types.get(n, "unknown"),
                description=f"Isolated node (degree=0) with high confidence ({conf:.2f}) and {len(prov)} sources — connections may be missing",
                evidence_count=len(prov),
                score=round(conf * 0.8, 4),
                run_id=run_id,
            ))

    # ─── 5. Coverage Gap ───
    for ci in coverage_intervals:
        ratio = ci.get("coverage_ratio", 1.0)
        status = ci.get("status", "")
        if ratio < 0.5 or status == "sparse":
            severity = "high" if ratio < 0.2 else ("medium" if ratio < 0.5 else "low")
            signals.append(AnomalySignal(
                signal_id=generate_id("SIG", f"coverage_{ci.get('entity_id', '')}"),
                signal_type="coverage_gap",
                severity=severity,
                entity_id=ci.get("entity_id", ""),
                entity_type=node_types.get(ci.get("entity_id", ""), "unknown"),
                description=f"Coverage ratio {ratio:.2f} (status={status}) — sparse temporal data",
                evidence_count=0,
                score=round(1.0 - ratio, 4),
                run_id=run_id,
            ))

    # ─── 6. Temporal Burst ───
    entity_temps = defaultdict(list)
    for t in temporal_infos:
        eid = t.get("entity_id", "")
        start = t.get("start_time", "")
        if start:
            entity_temps[eid].append(start)

    for eid, times in entity_temps.items():
        if len(times) < 3:
            continue
        sorted_times = sorted(times)
        # Check if many events are close together (simplified: count duplicates/near-duplicates)
        unique_times = set(sorted_times)
        if len(unique_times) < len(sorted_times) * 0.5:
            signals.append(AnomalySignal(
                signal_id=generate_id("SIG", f"burst_{eid}"),
                signal_type="temporal_burst",
                severity="medium",
                entity_id=eid,
                entity_type=node_types.get(eid, "unknown"),
                description=f"Temporal burst: {len(sorted_times)} events with only {len(unique_times)} unique timestamps",
                evidence_count=len(sorted_times),
                score=round(1.0 - len(unique_times) / len(sorted_times), 4),
                run_id=run_id,
            ))

    signals.sort(key=lambda s: s.score, reverse=True)
    return signals
