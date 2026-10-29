"""Derived connection detection — Stage 6 hidden-connection analysis.

Finds expected-but-absent edges that Stage 5 did not materialize:
1. Person-person co-location: two persons share a location node
2. Temporal co-occurrence: two persons active in overlapping time windows
3. Shared-associate gaps: already partially computed in Stage 5;
   re-validated here against the final graph.
4. Multi-hop path findings (suspect → victim aggregation).

Outputs are DERIVED (epistemic_status=inference) — never observations.
"""

from typing import List, Dict, Set, Tuple
from collections import defaultdict
from datetime import datetime
import networkx as nx

from ..models.schema import generate_id


def _parse_time(time_str: str) -> float:
    if not time_str:
        return 0.0
    try:
        for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%d", "%Y-%m-%dT%H:%M:%S.%f"):
            try:
                return datetime.strptime(str(time_str)[:19], fmt).timestamp()
            except ValueError:
                continue
        return 0.0
    except Exception:
        return 0.0


def compute_derived_connections(
    G: nx.Graph,
    nodes_data: List[dict],
    edges_data: List[dict],
    temporal_infos: List[dict],
    spatial_infos: List[dict],
    missing_edges: List[dict],
    multi_hop_paths: List[dict],
    run_id: str = "",
) -> List[dict]:
    """Return list of derived connection dicts (inference, not observation)."""
    derived: List[dict] = []

    person_ids = {
        n["id"] for n in nodes_data
        if n.get("node_type") == "person"
    }
    if not person_ids:
        return derived

    # Index: person → location nodes they touch (via any edge)
    person_locations: Dict[str, Set[str]] = defaultdict(set)
    for e in edges_data:
        s, t = e.get("source_id", ""), e.get("target_id", "")
        for a, b in ((s, t), (t, s)):
            if a in person_ids and b in person_ids:
                continue
            if a in person_ids:
                # target may be location-ish
                tn = next((n for n in nodes_data if n["id"] == b), None)
                if tn and tn.get("node_type") in ("location", "organization"):
                    person_locations[a].add(b)

    # Also use spatial_infos: person → locationName
    locname_people: Dict[str, Set[str]] = defaultdict(set)
    for s in spatial_infos:
        eid = s.get("entity_id", "")
        lname = s.get("location_name") or s.get("locationName") or ""
        if eid in person_ids and lname:
            locname_people[lname.lower()].add(eid)

    # ── 1. Person-person co-location via shared location node ──
    loc_people: Dict[str, Set[str]] = defaultdict(set)
    for pid, locs in person_locations.items():
        for loc in locs:
            loc_people[loc].add(pid)
    for lname, peeps in locname_people.items():
        if len(peeps) >= 2:
            loc_people[f"name:{lname}"].update(peeps)

    seen_pairs: Set[frozenset] = set()
    # Existing direct person-person edges
    existing: Set[frozenset] = set()
    for e in edges_data:
        s, t = e.get("source_id", ""), e.get("target_id", "")
        if s in person_ids and t in person_ids and s != t:
            existing.add(frozenset((s, t)))

    for loc_key, peeps in loc_people.items():
        plist = sorted(peeps)
        for i, a in enumerate(plist):
            for b in plist[i + 1:]:
                pair = frozenset((a, b))
                if pair in existing or pair in seen_pairs:
                    continue
                seen_pairs.add(pair)
                derived.append({
                    "id": generate_id("DCOL", f"{a}_{b}_{loc_key}"),
                    "connection_type": "person_colocation",
                    "source_id": a,
                    "target_id": b,
                    "shared_context": str(loc_key),
                    "confidence": 0.55,
                    "epistemic_status": "inference",
                    "basis": "persons observed at same location",
                    "run_id": run_id,
                })

    # ── 2. Temporal co-occurrence: overlapping active windows ──
    # Build per-person [min_epoch, max_epoch] from temporal_infos
    person_windows: Dict[str, List[float]] = {}
    for t in temporal_infos:
        eid = t.get("entity_id", "")
        if eid not in person_ids:
            continue
        ep = _parse_time(t.get("start_time", ""))
        if ep <= 0:
            continue
        if eid not in person_windows:
            person_windows[eid] = [ep, ep]
        else:
            person_windows[eid][0] = min(person_windows[eid][0], ep)
            person_windows[eid][1] = max(person_windows[eid][1], ep)

    pids = sorted(person_windows.keys())
    # Overlap window of 48 hours counts as co-occurrence signal
    OVERLAP = 48 * 3600
    for i, a in enumerate(pids):
        a0, a1 = person_windows[a]
        for b in pids[i + 1:]:
            pair = frozenset((a, b))
            if pair in existing:
                continue
            b0, b1 = person_windows[b]
            # Overlap?
            latest_start = max(a0, b0)
            earliest_end = min(a1, b1)
            if earliest_end - latest_start >= -OVERLAP:
                # only add if not already derived colocation for same pair
                if any(
                    d["connection_type"] == "person_colocation"
                    and frozenset((d["source_id"], d["target_id"])) == pair
                    for d in derived
                ):
                    continue
                derived.append({
                    "id": generate_id("DTMP", f"{a}_{b}"),
                    "connection_type": "temporal_cooccurrence",
                    "source_id": a,
                    "target_id": b,
                    "shared_context": "overlapping_activity_window",
                    "confidence": 0.50,
                    "epistemic_status": "inference",
                    "basis": "activity windows overlap within 48h",
                    "run_id": run_id,
                })

    # ── 3. Append Stage 5 missing edges / multi-hop as derived findings ──
    for m in missing_edges:
        if not isinstance(m, dict):
            continue
        sid, tid = m.get("source_id", ""), m.get("target_id", "")
        if not sid or not tid:
            continue
        pair = frozenset((sid, tid))
        if pair in existing:
            continue
        if any(
            frozenset((d["source_id"], d["target_id"])) == pair
            for d in derived
        ):
            continue
        derived.append({
            "id": generate_id("DMIS", f"{sid}_{tid}_{m.get('expected_relation', '')}"),
            "connection_type": m.get("expected_relation", "missing_edge"),
            "source_id": sid,
            "target_id": tid,
            "shared_context": "shared_associate_or_expected_link",
            "confidence": float(m.get("confidence", 0.4)),
            "epistemic_status": "inference",
            "basis": "expected connection absent from graph",
            "source_files": m.get("source_files", []),
            "run_id": run_id,
        })

    for mh in multi_hop_paths:
        if not isinstance(mh, dict):
            continue
        sid, tid = mh.get("source_id", ""), mh.get("target_id", "")
        if not sid or not tid:
            continue
        pair = frozenset((sid, tid))
        if pair in existing:
            continue
        derived.append({
            "id": generate_id("DMH", f"{sid}_{tid}"),
            "connection_type": "multi_hop_path",
            "source_id": sid,
            "target_id": tid,
            "shared_context": f"path={'→'.join(mh.get('path', []))}",
            "confidence": float(mh.get("path_confidence", 0.4)),
            "epistemic_status": "inference",
            "basis": f"{mh.get('hops', 0)}-hop path via intermediate entities",
            "relationship_types": mh.get("relationship_types", []),
            "run_id": run_id,
        })

    # Deduplicate by (source, target) keeping highest confidence
    best: Dict[Tuple[str, str], dict] = {}
    for d in derived:
        key = tuple(sorted((d["source_id"], d["target_id"])))
        if key not in best or d["confidence"] > best[key]["confidence"]:
            best[key] = d
    result = sorted(best.values(), key=lambda x: -x["confidence"])
    return result
