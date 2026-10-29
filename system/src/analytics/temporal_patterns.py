"""Temporal pattern detection — finds time-based patterns in entity activity."""

from typing import List
from collections import defaultdict
from datetime import datetime
import numpy as np
import networkx as nx

from .models import TemporalPattern
from ..models.schema import generate_id


def _parse_time(time_str: str) -> float:
    """Parse ISO timestamp to epoch seconds. Returns 0.0 on failure."""
    if not time_str:
        return 0.0
    try:
        # Handle various ISO formats
        for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%d", "%Y-%m-%dT%H:%M:%S.%f"):
            try:
                return datetime.strptime(time_str[:19], fmt).timestamp()
            except ValueError:
                continue
        return 0.0
    except Exception:
        return 0.0


def compute_temporal_patterns(
    G: nx.Graph,
    temporal_infos: List[dict],
    run_id: str = "",
) -> List[TemporalPattern]:
    """Detect temporal patterns: activity bursts, regularity, anomalies.

    Patterns are grounded in actual data constraints:
    - Activity Burst: >= 3 events with inter-event time < 1 hour
    - Regularity: >= 4 events with consistent spacing (CV < 0.3)
    - Temporal Anomaly: timestamps far from entity's mean
    """
    if not temporal_infos:
        return []

    node_types = {n: G.nodes[n].get("node_type", "unknown") for n in G.nodes}

    # Group temporal infos by entity
    entity_events = defaultdict(list)
    for t in temporal_infos:
        eid = t.get("entity_id", "")
        start = t.get("start_time", "")
        if eid and start:
            epoch = _parse_time(start)
            if epoch > 0:
                entity_events[eid].append({
                    "epoch": epoch,
                    "start": start,
                    "event_type": t.get("event_type", ""),
                    "confidence": t.get("confidence", 0.5),
                })

    patterns = []

    for eid, events in entity_events.items():
        if len(events) < 2:
            continue

        events.sort(key=lambda e: e["epoch"])
        epochs = np.array([e["epoch"] for e in events])

        # Inter-event times
        diffs = np.diff(epochs)
        if len(diffs) == 0:
            continue

        # ─── Activity Burst ───
        if len(events) >= 3:
            burst_threshold = 3600  # 1 hour
            burst_count = sum(1 for d in diffs if d < burst_threshold)
            if burst_count > 0:
                patterns.append(TemporalPattern(
                    pattern_id=generate_id("TPAT", f"burst_{eid}"),
                    pattern_type="activity_burst",
                    entity_id=eid,
                    entity_type=node_types.get(eid, "unknown"),
                    description=f"{burst_count} inter-event gap(s) under 1 hour across {len(events)} events",
                    event_count=len(events),
                    time_span_seconds=round(float(epochs[-1] - epochs[0]), 2),
                    severity="medium" if burst_count >= 2 else "low",
                    run_id=run_id,
                ))

        # ─── Regularity ───
        if len(events) >= 4 and np.std(diffs) > 0:
            mean_diff = np.mean(diffs)
            cv = np.std(diffs) / mean_diff if mean_diff > 0 else 999
            if cv < 0.3:
                patterns.append(TemporalPattern(
                    pattern_id=generate_id("TPAT", f"regular_{eid}"),
                    pattern_type="regularity",
                    entity_id=eid,
                    entity_type=node_types.get(eid, "unknown"),
                    description=f"Regular inter-event spacing: mean={mean_diff:.0f}s, CV={cv:.2f} (regular pattern)",
                    event_count=len(events),
                    time_span_seconds=round(float(epochs[-1] - epochs[0]), 2),
                    severity="low",
                    run_id=run_id,
                ))

        # ─── Temporal Anomaly ───
        if len(events) >= 3:
            mean_epoch = np.mean(epochs)
            std_epoch = np.std(epochs)
            if std_epoch > 0:
                for e in events:
                    z = abs(e["epoch"] - mean_epoch) / std_epoch
                    if z > 2.0:
                        patterns.append(TemporalPattern(
                            pattern_id=generate_id("TPAT", f"anomaly_{eid}_{e['start'][:10]}"),
                            pattern_type="temporal_anomaly",
                            entity_id=eid,
                            entity_type=node_types.get(eid, "unknown"),
                            description=f"Event at {e['start'][:10]} is {z:.1f}σ from entity's mean timestamp",
                            event_count=len(events),
                            time_span_seconds=round(float(epochs[-1] - epochs[0]), 2),
                            severity="medium" if z > 2.5 else "low",
                            run_id=run_id,
                        ))
                        break  # One anomaly per entity

    patterns.sort(key=lambda p: p.event_count, reverse=True)
    return patterns
