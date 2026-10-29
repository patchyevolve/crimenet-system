"""
Temporal Engine — orchestrates temporal and spatial enrichment.
Includes Allen's Interval Algebra for temporal relations and event clustering.
"""

import json
import time
from typing import List, Dict, Optional, Set, Tuple
from pathlib import Path
from datetime import datetime, timedelta, timezone

from ..models.schema import ExtractedEntity, ExtractedRelation, generate_id
from ..resolution.merger import ResolvedEntity
from .schema import (
    TemporalInfo, SpatialInfo, CoverageInterval, TimelineEvent,
    AllenRelation, CoverageStatus, EventCluster,
    create_temporal_info, create_spatial_info, create_coverage_interval,
    create_timeline_event,
)
from .timestamp import normalize_timestamp, get_time_range, parse_timestamp, detect_precision
from .spatial import normalize_location, get_coordinates, assess_spatial_precision


def _to_utc(value: str) -> datetime:
    """Coerce an ISO 8601 string to an aware UTC datetime.

    Naive strings are assumed UTC; aware strings are converted. Mixing the
    two is what used to raise TypeError in ``allen_relation``.
    """
    dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _to_utc_lenient(value) -> datetime:
    """``_to_utc`` with the legacy truncated-ISO fallback.

    Raw CDR/CCTV timestamps can carry decorations after the seconds
    ("...14:30:00 IST"); the previous implementation truncated to the first
    19 characters before parsing, so keep that as a fallback — otherwise
    those pairs would silently stop being compared.
    """
    try:
        return _to_utc(value)
    except (ValueError, TypeError):
        return _to_utc(str(value)[:19])


def allen_relation(start_a: str, end_a: str, start_b: str, end_b: str) -> AllenRelation:
    """
    Compute Allen's Interval Algebra relation between two intervals.
    All times should be ISO 8601 strings.

    Naive and offset-aware inputs are normalized to UTC first, so mixed
    comparisons never raise TypeError; unparsable input degrades to EQUALS.
    """
    try:
        a_start = _to_utc(start_a)
        a_end = _to_utc(end_a)
        b_start = _to_utc(start_b)
        b_end = _to_utc(end_b)

        if a_end < b_start:
            return AllenRelation.BEFORE
        elif a_start > b_end:
            return AllenRelation.AFTER
        elif a_end == b_start:
            return AllenRelation.MEETS
        elif a_start == b_end:
            return AllenRelation.MET_BY
        elif a_start < b_start and a_end > b_start and a_end < b_end:
            return AllenRelation.OVERLAPS
        elif a_start > b_start and a_start < b_end and a_end > b_end:
            return AllenRelation.OVERLAPPED_BY
        elif a_start >= b_start and a_end <= b_end:
            return AllenRelation.DURING if a_start > b_start or a_end < b_end else AllenRelation.EQUALS
        elif a_start <= b_start and a_end >= b_end:
            return AllenRelation.CONTAINS if a_start < b_start or a_end > b_end else AllenRelation.EQUALS
        elif a_start == b_start and a_end < b_end:
            return AllenRelation.STARTS
        elif a_start == b_start and a_end > b_end:
            return AllenRelation.STARTED_BY
        elif a_end == b_end and a_start > b_start:
            return AllenRelation.FINISHES
        elif a_end == b_end and a_start < b_start:
            return AllenRelation.FINISHED_BY
        else:
            return AllenRelation.EQUALS
    except (ValueError, TypeError):
        return AllenRelation.EQUALS  # Cannot determine — treat as same time


def compute_temporal_relations(events: List[TimelineEvent]) -> List[dict]:
    """
    Compute Allen's Interval Algebra relations between all pairs of events.
    Returns list of {event_a, event_b, relation, confidence}.
    """
    relations = []

    for i, ev_a in enumerate(events):
        for ev_b in events[i+1:]:
            # Skip if different entities (unless comparing across entities is desired)
            if ev_a.entity_id != ev_b.entity_id:
                continue

            # Get time ranges
            start_a = ev_a.timestamp
            end_a = ev_a.timestamp  # Point events
            if ev_a.duration_seconds:
                try:
                    dt = datetime.fromisoformat(start_a.replace("Z", "+00:00"))
                    end_a = (dt + timedelta(seconds=ev_a.duration_seconds)).isoformat()
                except (ValueError, TypeError):
                    pass

            start_b = ev_b.timestamp
            end_b = ev_b.timestamp
            if ev_b.duration_seconds:
                try:
                    dt = datetime.fromisoformat(start_b.replace("Z", "+00:00"))
                    end_b = (dt + timedelta(seconds=ev_b.duration_seconds)).isoformat()
                except (ValueError, TypeError):
                    pass

            relation = allen_relation(start_a, end_a, start_b, end_b)

            # Confidence based on precision of both events
            conf_a = ev_a.confidence
            conf_b = ev_b.confidence
            combined_conf = (conf_a + conf_b) / 2

            relations.append({
                "event_a": ev_a.id,
                "event_b": ev_b.id,
                "relation": relation.value,
                "confidence": combined_conf,
            })

    return relations


def cluster_events(events: List[TimelineEvent], max_gap_seconds: float = 600) -> List[EventCluster]:
    """
    Cluster events that are part of the same real-world event.
    Events within max_gap_seconds of each other and same type are clustered.
    Default: 10 minutes (600 seconds).
    """
    if not events:
        return []

    # Sort by UTC instant, not by string. A plain string sort puts
    # "…T14:30:00+05:30" (= 09:00 UTC) after "…T09:00:30", which reverses
    # neighbouring pairs and yields a negative time_span on mixed
    # naive/aware input. Unparsable timestamps keep their relative order at
    # the end instead of crashing the sort.
    def _sort_key(ev: TimelineEvent):
        try:
            return (0, _to_utc(ev.timestamp))
        except (ValueError, TypeError):
            return (1, datetime.max.replace(tzinfo=timezone.utc))

    sorted_events = sorted(events, key=_sort_key)

    clusters = []
    current_cluster_events = [sorted_events[0]]
    current_cluster_type = sorted_events[0].event_type

    for ev in sorted_events[1:]:
        # Check if this event belongs to the current cluster
        try:
            # _to_utc: mixed naive/aware inputs would otherwise raise
            # TypeError on subtraction, degrading gap to float('inf').
            last_time = _to_utc(current_cluster_events[-1].timestamp)
            this_time = _to_utc(ev.timestamp)
            gap = abs((this_time - last_time).total_seconds())
        except (ValueError, TypeError):
            gap = float('inf')

        if gap <= max_gap_seconds and ev.event_type == current_cluster_type:
            current_cluster_events.append(ev)
        else:
            # Finalize current cluster
            if len(current_cluster_events) > 1:
                cluster_id = generate_id("CLUSTER", f"{current_cluster_events[0].id}_{current_cluster_events[-1].id}")
                try:
                    start_dt = _to_utc(current_cluster_events[0].timestamp)
                    end_dt = _to_utc(current_cluster_events[-1].timestamp)
                    time_span = (end_dt - start_dt).total_seconds()
                except (ValueError, TypeError):
                    time_span = 0

                avg_conf = sum(e.confidence for e in current_cluster_events) / len(current_cluster_events)
                clusters.append(EventCluster(
                    cluster_id=cluster_id,
                    events=[e.id for e in current_cluster_events],
                    event_type=current_cluster_type,
                    time_span_seconds=time_span,
                    confidence=avg_conf,
                ))

            # Start new cluster
            current_cluster_events = [ev]
            current_cluster_type = ev.event_type

    # Finalize last cluster
    if len(current_cluster_events) > 1:
        cluster_id = generate_id("CLUSTER", f"{current_cluster_events[0].id}_{current_cluster_events[-1].id}")
        try:
            start_dt = _to_utc(current_cluster_events[0].timestamp)
            end_dt = _to_utc(current_cluster_events[-1].timestamp)
            time_span = (end_dt - start_dt).total_seconds()
        except (ValueError, TypeError):
            time_span = 0

        avg_conf = sum(e.confidence for e in current_cluster_events) / len(current_cluster_events)
        clusters.append(EventCluster(
            cluster_id=cluster_id,
            events=[e.id for e in current_cluster_events],
            event_type=current_cluster_type,
            time_span_seconds=time_span,
            confidence=avg_conf,
        ))

    return clusters


class TemporalEngine:
    """Enriches entities and relations with temporal and spatial context."""

    def __init__(self):
        pass

    def enrich(
        self,
        entities: List[dict],
        relations: List[dict],
        output_dir: str = "output",
        run_id: str = "",
    ) -> dict:
        """
        Run temporal/spatial enrichment.
        Returns summary dict.
        """
        start_time = time.time()

        print(f"\n[TEMPORAL] Starting with {len(entities)} entities, {len(relations)} relations")

        # Collect temporal and spatial info
        temporal_infos: List[TemporalInfo] = []
        spatial_infos: List[SpatialInfo] = []
        coverage_intervals: List[CoverageInterval] = []
        timeline_events: List[TimelineEvent] = []

        # Track seen timestamps to avoid duplicates
        seen_temporal: Set[str] = set()
        seen_spatial: Set[str] = set()

        # Source type to base confidence mapping
        source_confidence = {
            "cdr": 0.95,
            "bank": 0.98,
            "device": 0.99,
            "cctv": 0.85,
            "social": 0.80,
            "text": 0.75,
            "report": 0.80,
            "tabular": 0.90,
            "generic": 0.70,
        }

        # Process entities — accepts either raw ExtractedEntity dicts (the
        # pre-resolution shape) or canonical ResolvedEntity dicts, so Stage 4
        # can bind timestamps and coordinates to RES_ ids.
        for entity in entities:
            eid = entity.get("id", "")
            name = entity.get("canonical_name", "") or entity.get("name", "")
            attrs = entity.get("attributes", {}) or {}
            source = entity.get("source", {}) or {}
            source_file = (
                source.get("file_name", "")
                or (entity.get("provenance_chain") or [""])[0]
                or (attrs.get("observed_in") or [""])[0]
            )
            source_type = source.get("source_type", "generic")
            base_confidence = source_confidence.get(source_type, 0.70)

            # Extract timestamps from entity attributes
            for key, value in attrs.items():
                if not value or not isinstance(value, str):
                    continue

                # Skip non-temporal attributes
                key_lower = key.lower()
                if any(skip in key_lower for skip in ['tower_id', 'id', 'name', 'type', 'status', 'notes', 'description', 'caption']):
                    continue

                # Check if value looks like a timestamp
                ts = normalize_timestamp(value)
                if ts:
                    event_type = self._infer_event_type(key, entity.get("entity_type", ""), attrs)
                    precision = detect_precision(value)
                    # Confidence varies by precision and source
                    precision_mult = {"exact": 1.0, "approximate": 0.85, "range": 0.75, "unknown": 0.5}
                    confidence = base_confidence * precision_mult.get(
                        precision.value if hasattr(precision, 'value') else str(precision), 0.5
                    )
                    temporal_key = f"{eid}_{event_type}_{ts}"
                    if temporal_key not in seen_temporal:
                        seen_temporal.add(temporal_key)
                        temporal_infos.append(create_temporal_info(
                            entity_id=eid,
                            event_type=event_type,
                            start_time=ts,
                            precision=precision.value if hasattr(precision, 'value') else precision,
                            source_id=source_file,
                            confidence=round(confidence, 2),
                            run_id=run_id,
                        ))

                        # Add to timeline
                        timeline_events.append(create_timeline_event(
                            entity_id=eid,
                            event_type=event_type,
                            timestamp=ts,
                            source_id=source_file,
                            description=f"{event_type} from {source_file or name}",
                            confidence=round(confidence, 2),
                            run_id=run_id,
                        ))

                # Check if value looks like a location
                if self._is_location(key, value):
                    spatial_key = f"{eid}_{value}"
                    if spatial_key not in seen_spatial:
                        seen_spatial.add(spatial_key)
                        normalized = normalize_location(value)
                        spatial_infos.append(create_spatial_info(
                            entity_id=eid,
                            event_type="location",
                            latitude=normalized.get("latitude"),
                            longitude=normalized.get("longitude"),
                            radius_km=normalized.get("radius_km"),
                            address=normalized.get("raw", value),
                            city=normalized.get("city", ""),
                            state=normalized.get("state", ""),
                            precision=normalized.get("precision", "unknown"),
                            source_id=source_file,
                            confidence=round(base_confidence, 2),
                            run_id=run_id,
                        ))

            # Extract temporal from phone call timestamps
            if entity.get("entity_type") == "EVENT":
                # Check for call duration, timestamps
                call_time = attrs.get("timestamp", "") or attrs.get("time", "")
                if call_time:
                    ts = normalize_timestamp(call_time)
                    if ts:
                        temporal_key = f"{eid}_call_{ts}"
                        if temporal_key not in seen_temporal:
                            seen_temporal.add(temporal_key)
                            temporal_infos.append(create_temporal_info(
                                entity_id=eid,
                                event_type="call",
                                start_time=ts,
                                source_id=source_file,
                                confidence=round(base_confidence, 2),
                                run_id=run_id,
                            ))

                    # Duration
                    duration = attrs.get("duration", "")
                    if duration:
                        duration_secs = self._parse_duration(duration)
                        if duration_secs is not None:
                            timeline_events.append(create_timeline_event(
                                entity_id=eid,
                                event_type="call",
                                timestamp=ts or datetime.now().isoformat(),
                                duration_seconds=duration_secs,
                                source_id=source_file,
                                description=f"Call duration: {duration}",
                                confidence=round(base_confidence, 2),
                                run_id=run_id,
                            ))

        # Process relations
        for relation in relations:
            rid = relation.get("id", "")
            source_entity_id = relation.get("source_entity_id", "")
            target_id = relation.get("target_entity_id", "")
            attrs = relation.get("attributes", {})
            rel_type = relation.get("relation_type", "")
            rel_source = relation.get("source", {})
            rel_source_file = rel_source.get("file_name", "")
            rel_source_type = rel_source.get("source_type", "generic")
            rel_confidence = source_confidence.get(rel_source_type, 0.70)

            # Extract timestamps from relation attributes
            for key, value in attrs.items():
                if not value or not isinstance(value, str):
                    continue

                # Skip non-temporal attributes
                key_lower = key.lower()
                if any(skip in key_lower for skip in ['tower_id', 'id', 'name', 'type', 'status', 'notes', 'description', 'caption']):
                    continue

                ts = normalize_timestamp(value)
                if ts:
                    event_type = f"relation_{rel_type.lower()}"
                    precision = detect_precision(value)
                    precision_mult = {"exact": 1.0, "approximate": 0.85, "range": 0.75, "unknown": 0.5}
                    confidence = rel_confidence * precision_mult.get(
                        precision.value if hasattr(precision, 'value') else str(precision), 0.5
                    )
                    temporal_key = f"{rid}_{event_type}_{ts}"
                    if temporal_key not in seen_temporal:
                        seen_temporal.add(temporal_key)
                        temporal_infos.append(create_temporal_info(
                            entity_id=source_entity_id,
                            event_type=event_type,
                            start_time=ts,
                            precision=precision.value if hasattr(precision, 'value') else precision,
                            source_id=rel_source_file,
                            confidence=round(confidence, 2),
                            run_id=run_id,
                        ))

        # Create coverage intervals
        print("[TEMPORAL] Computing coverage intervals...")
        coverage_intervals = self._compute_coverage(entities, relations, run_id)

        # Compute Allen's Interval Algebra temporal relations
        print("[TEMPORAL] Computing temporal relations (Allen's Interval Algebra)...")
        temporal_relations = compute_temporal_relations(timeline_events)

        # Cluster events (events within 10 min of same type = same event)
        print("[TEMPORAL] Clustering events...")
        event_clusters = cluster_events(timeline_events, max_gap_seconds=600)

        # Detect temporal contradictions (same entity at different locations same time)
        print("[TEMPORAL] Detecting temporal contradictions...")
        temporal_contradictions = self._detect_temporal_contradictions(
            entities, relations, spatial_infos, temporal_infos, run_id
        )

        elapsed = time.time() - start_time

        # Build summary
        summary = {
            "run_id": run_id,
            "timestamp": datetime.now().isoformat(),
            "input_entities": len(entities),
            "input_relations": len(relations),
            "temporal_infos": len(temporal_infos),
            "spatial_infos": len(spatial_infos),
            "coverage_intervals": len(coverage_intervals),
            "timeline_events": len(timeline_events),
            "temporal_relations": len(temporal_relations),
            "event_clusters": len(event_clusters),
            "temporal_contradictions": len(temporal_contradictions),
            "processing_time_seconds": round(elapsed, 2),
        }

        # Save outputs
        self._save_outputs(
            temporal_infos, spatial_infos, coverage_intervals,
            timeline_events, temporal_relations, event_clusters,
            summary, output_dir,
        )

        # Save temporal contradictions
        if temporal_contradictions:
            contradictions_path = Path(output_dir) / "temporal_contradictions.json"
            with open(contradictions_path, "w") as f:
                json.dump(temporal_contradictions, f, indent=2, default=str)

        print(f"\n[TEMPORAL] Complete in {elapsed:.2f}s")
        print(f"  -> {len(temporal_infos)} temporal infos")
        print(f"  -> {len(spatial_infos)} spatial infos")
        print(f"  -> {len(coverage_intervals)} coverage intervals")
        print(f"  -> {len(timeline_events)} timeline events")
        print(f"  -> {len(temporal_contradictions)} temporal contradictions")

        return summary

    def _infer_event_type(self, attr_key: str, entity_type: str, entity_attrs: dict = None) -> str:
        """Infer event type from attribute key and entity type."""
        key_lower = attr_key.lower()

        # Direct attribute name matches
        if "arrest" in key_lower:
            return "arrest"
        elif "incident" in key_lower:
            return "incident"
        elif "call" in key_lower:
            return "call"
        elif "transaction" in key_lower or "transfer" in key_lower:
            return "transaction"
        elif "payment" in key_lower:
            return "payment"
        elif "sighting" in key_lower or "spotted" in key_lower:
            return "sighting"
        elif "seizure" in key_lower or "recovered" in key_lower:
            return "seizure"
        elif "fir" in key_lower or "complaint" in key_lower:
            return "case_filing"
        elif "death" in key_lower:
            return "death"
        elif "birth" in key_lower:
            return "birth"

        # Entity type based inference
        if entity_type == "EVENT":
            # Check if this is a social media post
            if entity_attrs and (entity_attrs.get("social_type") == "post" or entity_attrs.get("post_id")):
                return "social_post"
            return "incident"
        elif entity_type == "DATE":
            # DATE entities are usually incident or case dates
            return "incident_date"
        elif entity_type == "PERSON":
            if "date" in key_lower or "time" in key_lower:
                return "person_event"
        elif entity_type == "PHONE":
            if "date" in key_lower or "time" in key_lower:
                return "communication_event"
        elif entity_type == "ACCOUNT":
            if "date" in key_lower or "time" in key_lower:
                return "financial_event"
        elif entity_type == "DEVICE":
            if "extraction" in key_lower:
                return "device_extraction"
            elif "seizure" in key_lower:
                return "device_seizure"
            return "device_event"
        elif entity_type == "ORGANIZATION":
            return "organization_event"

        # Generic date/time attributes
        if "date" in key_lower or "time" in key_lower:
            return "generic_temporal"

        return "event"

    def _is_location(self, key: str, value: str) -> bool:
        """Check if an attribute value looks like a location."""
        key_lower = key.lower()
        location_keys = ["location", "address", "place", "area", "city", "district", "state"]
        if any(lk in key_lower for lk in location_keys):
            return True
        # Check if value contains location-like patterns
        if re.search(r'(?:near|area|road|street|hotel|colony|nagar|mandir|masjid)', str(value), re.IGNORECASE):
            return True
        return False

    def _parse_duration(self, duration_str: str) -> Optional[float]:
        """Parse duration string to seconds."""
        import re
        duration_str = str(duration_str).strip()

        # "01:23" = 1 min 23 sec
        match = re.match(r'(\d+):(\d+)', duration_str)
        if match:
            mins = int(match.group(1))
            secs = int(match.group(2))
            return mins * 60 + secs

        # "90s" or "90 sec"
        match = re.match(r'(\d+)\s*s', duration_str)
        if match:
            return float(match.group(1))

        # "5m" or "5 min"
        match = re.match(r'(\d+)\s*m', duration_str)
        if match:
            return float(match.group(1)) * 60

        # Just a number (assume seconds)
        try:
            return float(duration_str)
        except ValueError:
            return None

    def _detect_temporal_contradictions(
        self,
        entities: List[dict],
        relations: List[dict],
        spatial_infos: List[SpatialInfo],
        temporal_infos: List[TemporalInfo],
        run_id: str = "",
    ) -> List[dict]:
        """
        Detect temporal contradictions:
        1. Same entity at different locations at the same time (via CDR tower data)
        2. Same entity with conflicting temporal attributes
        3. Self-calls (entity calling itself)
        4. Impossible time sequences
        
        Returns list of contradiction dicts.
        """
        contradictions = []
        
        # Build entity -> locations mapping from spatial_infos
        entity_locations: Dict[str, List[dict]] = {}
        for spatial in spatial_infos:
            eid = spatial.entity_id
            if eid not in entity_locations:
                entity_locations[eid] = []
            entity_locations[eid].append({
                "location": spatial.address or "unknown",
                "latitude": spatial.latitude,
                "longitude": spatial.longitude,
                "source": spatial.source_id,
                "confidence": spatial.confidence,
            })
        
        # Build entity -> timestamps mapping from temporal_infos
        entity_timestamps: Dict[str, List[dict]] = {}
        for temporal in temporal_infos:
            eid = temporal.entity_id
            if eid not in entity_timestamps:
                entity_timestamps[eid] = []
            entity_timestamps[eid].append({
                "timestamp": temporal.start_time,
                "event_type": temporal.event_type,
                "source": temporal.source_id,
                "confidence": temporal.confidence,
            })
        
        # Also build from CALLED relations with location attributes
        # This is the primary source for CDR-based temporal contradictions
        person_call_locations: Dict[str, List[dict]] = {}
        for relation in relations:
            if relation.get("relation_type") == "CALLED":
                attrs = relation.get("attributes", {})
                # Check for location or tower_location attribute
                tower = attrs.get("location", "") or attrs.get("tower_location", "")
                timestamp = attrs.get("timestamp", "") or relation.get("timestamp", "")
                source_file = relation.get("source_file", "") or relation.get("source", {}).get("file_name", "")
                caller_id = relation.get("source_entity_id", "")
                
                if tower and caller_id:
                    if caller_id not in person_call_locations:
                        person_call_locations[caller_id] = []
                    person_call_locations[caller_id].append({
                        "location": tower,
                        "timestamp": timestamp,
                        "source": source_file,
                    })
        
        # Check for contradictions: same person at different locations at same time
        # Method 1: Check via spatial_infos ( LOCATION entities)
        for eid in entity_locations:
            if eid not in entity_timestamps:
                continue
            
            locations = entity_locations[eid]
            timestamps = entity_timestamps[eid]
            
            for i, ts1 in enumerate(timestamps):
                for ts2 in timestamps[i+1:]:
                    try:
                        t1 = _to_utc(ts1["timestamp"])
                        t2 = _to_utc(ts2["timestamp"])
                        time_diff = abs((t1 - t2).total_seconds())
                        
                        if time_diff <= 300:  # 5 minutes
                            loc1 = ts1.get("source", "")
                            loc2 = ts2.get("source", "")
                            
                            if loc1 != loc2:
                                loc1_info = None
                                loc2_info = None
                                for loc in locations:
                                    if loc["source"] == loc1:
                                        loc1_info = loc
                                    if loc["source"] == loc2:
                                        loc2_info = loc
                                
                                if loc1_info and loc2_info:
                                    loc1_name = loc1_info.get("location", "")
                                    loc2_name = loc2_info.get("location", "")
                                    
                                    if loc1_name and loc2_name and loc1_name != loc2_name:
                                        contradiction = {
                                            "id": generate_id("TEMPORAL_CONTRADICTION", f"{eid}_{ts1['timestamp']}_{ts2['timestamp']}_{loc1_name}_{loc2_name}"),
                                            "type": "temporal_location",
                                            "entity_id": eid,
                                            "timestamp1": ts1["timestamp"],
                                            "timestamp2": ts2["timestamp"],
                                            "time_diff_seconds": time_diff,
                                            "location1": loc1_name,
                                            "location2": loc2_name,
                                            "source1": loc1,
                                            "source2": loc2,
                                            "severity": "high" if time_diff < 60 else "medium",
                                            "confidence": min(ts1.get("confidence", 0.5), ts2.get("confidence", 0.5)),
                                            "run_id": run_id,
                                        }
                                        contradictions.append(contradiction)
                    except (ValueError, TypeError):
                        continue
        
        # Method 2: Check via CALL relations with tower_location
        # This catches CDR-based contradictions where person is at different towers same time
        for caller_id, call_locs in person_call_locations.items():
            if len(call_locs) < 2:
                continue
            
            # Group calls by time window (within 5 minutes)
            for i, call1 in enumerate(call_locs):
                for call2 in call_locs[i+1:]:
                    try:
                        # Parse timestamps if available
                        ts1_str = call1.get("timestamp", "")
                        ts2_str = call2.get("timestamp", "")
                        
                        if not ts1_str or not ts2_str:
                            continue
                        
                        # Parse both timestamps to aware UTC so mixed
                        # naive/aware values subtract cleanly (a TypeError
                        # here used to discard the pair).
                        t1 = _to_utc_lenient(ts1_str)
                        t2 = _to_utc_lenient(ts2_str)

                        if t1 and t2:
                            time_diff = abs((t1 - t2).total_seconds())
                            loc1 = call1.get("location", "")
                            loc2 = call2.get("location", "")
                            
                            # Same time, different location = contradiction
                            if time_diff <= 300 and loc1 and loc2 and loc1 != loc2:
                                contradiction = {
                                    "id": generate_id("TEMPORAL_CONTRADICTION", f"call_{caller_id}_{ts1_str}_{ts2_str}_{loc1}_{loc2}"),
                                    "type": "temporal_location",
                                    "entity_id": caller_id,
                                    "timestamp1": ts1_str,
                                    "timestamp2": ts2_str,
                                    "time_diff_seconds": time_diff,
                                    "location1": loc1,
                                    "location2": loc2,
                                    "source1": call1.get("source", ""),
                                    "source2": call2.get("source", ""),
                                    "severity": "high" if time_diff < 60 else "medium",
                                    "confidence": 0.8,
                                    "run_id": run_id,
                                }
                                contradictions.append(contradiction)
                    except (ValueError, TypeError):
                        continue
        
        # Check for self-calls (entity calling itself)
        for relation in relations:
            if relation.get("relation_type") == "CALLED":
                source = relation.get("source_entity_id", "")
                target = relation.get("target_entity_id", "")
                if source and target and source == target:
                    contradiction = {
                        "id": generate_id("TEMPORAL_CONTRADICTION", f"self_call_{source}_{relation.get('id', '')}"),
                        "type": "self_call",
                        "entity_id": source,
                        "relation_id": relation.get("id", ""),
                        "severity": "medium",
                        "confidence": relation.get("confidence", 0.5),
                        "run_id": run_id,
                    }
                    contradictions.append(contradiction)

        # Dedup by ID — duplicate timestamp entries in temporal_infos produce
        # identical contradictions from different source files (same entity,
        # same timestamps, same location pair). Keep the first occurrence.
        seen_ids = set()
        deduped = []
        for c in contradictions:
            cid = c.get("id", "")
            if cid in seen_ids:
                continue
            seen_ids.add(cid)
            deduped.append(c)
        return deduped

    def _compute_coverage(
        self,
        entities: List[dict],
        relations: List[dict],
        run_id: str = "",
    ) -> List[CoverageInterval]:
        """Compute coverage intervals for each entity."""
        from datetime import datetime, timedelta
        intervals = []

        # Group by entity
        entity_events: Dict[str, List[str]] = {}
        for entity in entities:
            eid = entity.get("id", "")
            attrs = entity.get("attributes", {})
            dates = []
            for key, value in attrs.items():
                if not value or not isinstance(value, str):
                    continue
                ts = normalize_timestamp(value)
                if ts:
                    dates.append(ts[:10])  # Just the date part
            if dates:
                entity_events[eid] = sorted(set(dates))

        # Create coverage intervals with actual ratio computation
        for eid, dates in entity_events.items():
            if dates:
                start = min(dates)
                end = max(dates)

                # Compute coverage ratio: what fraction of the time range has data
                try:
                    d_start = datetime.strptime(start, "%Y-%m-%d")
                    d_end = datetime.strptime(end, "%Y-%m-%d")
                    total_days = (d_end - d_start).days + 1
                    if total_days <= 1:
                        # Single point = full coverage
                        coverage_ratio = 1.0
                        status = "observed"
                    else:
                        # Count days with data vs total days in range
                        unique_dates = set(dates)
                        coverage_ratio = len(unique_dates) / total_days
                        if coverage_ratio >= 0.7:
                            status = "observed"
                        elif coverage_ratio >= 0.3:
                            status = "partial"
                        else:
                            status = "sparse"
                except ValueError:
                    coverage_ratio = 1.0
                    status = "observed"

                # Determine source_type from entity
                entity = next((e for e in entities if e.get("id") == eid), {})
                entity_type = entity.get("entity_type", "")
                if entity_type == "PHONE":
                    source_type = "cdr"
                elif entity_type == "ACCOUNT":
                    source_type = "bank"
                elif entity_type == "DEVICE":
                    source_type = "device"
                elif entity_type == "LOCATION":
                    source_type = "cctv"
                elif entity_type == "DATE":
                    source_type = "report"
                else:
                    source_type = "mixed"

                intervals.append(create_coverage_interval(
                    entity_id=eid,
                    source_type=source_type,
                    start=start,
                    end=end,
                    status=status,
                    coverage_ratio=round(coverage_ratio, 2),
                    run_id=run_id,
                ))

        return intervals

    def _save_outputs(
        self,
        temporal_infos: List[TemporalInfo],
        spatial_infos: List[SpatialInfo],
        coverage_intervals: List[CoverageInterval],
        timeline_events: List[TimelineEvent],
        temporal_relations: List[dict],
        event_clusters: List[EventCluster],
        summary: dict,
        output_dir: str,
    ):
        """Save temporal outputs to JSON files."""
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        # Temporal infos — deduplicate by ID
        seen_temporal_ids = set()
        deduped_temporal = []
        for t in temporal_infos:
            if t.id not in seen_temporal_ids:
                seen_temporal_ids.add(t.id)
                deduped_temporal.append(t)
        with open(out / "temporal_infos.json", "w") as f:
            json.dump([t.to_dict() for t in deduped_temporal], f, indent=2)

        # Spatial infos — save as list to preserve all entries
        with open(out / "spatial_infos.json", "w") as f:
            json.dump([s.to_dict() for s in spatial_infos], f, indent=2)

        # Coverage intervals
        with open(out / "coverage_intervals.json", "w") as f:
            json.dump([c.to_dict() for c in coverage_intervals], f, indent=2)

        # Timeline events — save as list to preserve all entries
        with open(out / "timeline_events.json", "w") as f:
            json.dump([t.to_dict() for t in timeline_events], f, indent=2)

        # Temporal relations (Allen's Interval Algebra)
        with open(out / "temporal_relations.json", "w") as f:
            json.dump(temporal_relations, f, indent=2)

        # Event clusters
        with open(out / "event_clusters.json", "w") as f:
            json.dump([c.to_dict() for c in event_clusters], f, indent=2)

        # Summary
        with open(out / "temporal_log.json", "w") as f:
            json.dump(summary, f, indent=2)

        print(f"  -> Saved to {output_dir}/temporal_infos.json")
        print(f"  -> Saved to {output_dir}/spatial_infos.json")
        print(f"  -> Saved to {output_dir}/coverage_intervals.json")
        print(f"  -> Saved to {output_dir}/timeline_events.json")
        print(f"  -> Saved to {output_dir}/temporal_relations.json")
        print(f"  -> Saved to {output_dir}/event_clusters.json")


import re
