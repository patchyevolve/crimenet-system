"""
Temporal Stage Models — TemporalInfo, SpatialInfo, CoverageInterval.
Includes Allen's Interval Algebra for temporal relations and event clustering.
"""

from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict
from datetime import datetime
from enum import Enum
from ..models.schema import generate_id


class TemporalPrecision(str, Enum):
    EXACT = "exact"
    APPROXIMATE = "approximate"
    RANGE = "range"
    UNKNOWN = "unknown"


class SpatialPrecision(str, Enum):
    EXACT = "exact"       # GPS coordinates
    TOWER = "tower"       # Cell tower (1-3km)
    CITY = "city"         # City-level
    VAGUE = "vague"       # "near X", "area of Y"
    UNKNOWN = "unknown"


class AllenRelation(str, Enum):
    """
    Allen's Interval Algebra — 13 temporal relations between intervals.
    Reference: Allen, J.F. (1983) "Maintaining knowledge about temporal intervals"
    """
    BEFORE = "before"           # A ends before B starts
    AFTER = "after"             # A starts after B ends
    MEETS = "meets"             # A ends when B starts
    MET_BY = "met_by"           # A starts when B ends
    OVERLAPS = "overlaps"       # A starts before B, ends during B
    OVERLAPPED_BY = "overlapped_by"  # A starts during B, ends after B
    DURING = "during"           # A is entirely within B
    CONTAINS = "contains"       # B is entirely within A
    STARTS = "starts"           # A and B start together, A ends first
    STARTED_BY = "started_by"   # A and B start together, B ends first
    FINISHES = "finishes"       # A and B end together, A starts later
    FINISHED_BY = "finished_by" # A and B end together, B starts later
    EQUALS = "equals"           # A and B are identical intervals


class CoverageStatus(str, Enum):
    """Coverage vocabulary — per docs"""
    OBSERVED = "observed"       # Data exists for this period
    SPARSE = "sparse"           # Some data, but gaps exist
    GAP = "gap"                 # No data for this period
    INFERRED = "inferred"       # Data inferred from other sources
    EXTRAPOLATED = "extrapolated"  # Data extrapolated beyond observed range


class EventCluster:
    """Groups related events that are part of the same real-world event."""
    def __init__(self, cluster_id: str, events: List[str], event_type: str,
                 time_span_seconds: float, confidence: float = 0.8):
        self.cluster_id = cluster_id
        self.events = events  # List of TimelineEvent IDs
        self.event_type = event_type
        self.time_span_seconds = time_span_seconds
        self.confidence = confidence

    def to_dict(self) -> dict:
        return {
            "cluster_id": self.cluster_id,
            "events": self.events,
            "event_type": self.event_type,
            "time_span_seconds": self.time_span_seconds,
            "confidence": self.confidence,
        }


@dataclass
class TemporalInfo:
    """Temporal information for an entity or event."""
    id: str
    entity_id: str
    event_type: str            # "call", "transaction", "sighting", etc.
    start_time: str            # ISO 8601
    end_time: str              # ISO 8601 (same as start for point events)
    precision: str             # "exact", "approximate", "range", "unknown"
    source_id: str             # RawEvidence ID
    confidence: float = 0.8
    run_id: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class SpatialInfo:
    """Spatial information for an entity or event."""
    id: str
    entity_id: str
    event_type: str            # "location", "sighting", "tower_connect"
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    radius_km: Optional[float] = None
    address: str = ""
    city: str = ""
    state: str = ""
    precision: str = "unknown"  # "exact", "tower", "city", "vague", "unknown"
    source_id: str = ""
    confidence: float = 0.8
    run_id: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class CoverageInterval:
    """Time interval covered by a data source."""
    entity_id: str
    source_type: str           # "cdr", "bank", "social", etc.
    start: str                 # ISO date
    end: str                   # ISO date
    status: str = "observed"   # "observed", "gap", "inferred"
    coverage_ratio: float = 0.0
    run_id: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class TimelineEvent:
    """Single event on the unified timeline."""
    id: str
    entity_id: str
    event_type: str
    timestamp: str
    duration_seconds: Optional[float] = None
    source_id: str = ""
    description: str = ""
    confidence: float = 0.8
    run_id: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def create_temporal_info(
    entity_id: str,
    event_type: str,
    start_time: str,
    end_time: str = None,
    precision: str = "unknown",
    source_id: str = "",
    confidence: float = 0.8,
    run_id: str = "",
) -> TemporalInfo:
    """Create a TemporalInfo with auto-generated ID."""
    return TemporalInfo(
        id=generate_id("TEMP", f"{entity_id}_{event_type}_{start_time}"),
        entity_id=entity_id,
        event_type=event_type,
        start_time=start_time,
        end_time=end_time or start_time,
        precision=precision,
        source_id=source_id,
        confidence=confidence,
        run_id=run_id,
    )


def create_spatial_info(
    entity_id: str,
    event_type: str,
    latitude: float = None,
    longitude: float = None,
    radius_km: float = None,
    address: str = "",
    city: str = "",
    state: str = "",
    precision: str = "unknown",
    source_id: str = "",
    confidence: float = 0.8,
    run_id: str = "",
) -> SpatialInfo:
    """Create a SpatialInfo with auto-generated ID."""
    return SpatialInfo(
        id=generate_id("SPAT", f"{entity_id}_{event_type}_{address or ''}_{city or ''}"),
        entity_id=entity_id,
        event_type=event_type,
        latitude=latitude,
        longitude=longitude,
        radius_km=radius_km,
        address=address,
        city=city,
        state=state,
        precision=precision,
        source_id=source_id,
        confidence=confidence,
        run_id=run_id,
    )


def create_coverage_interval(
    entity_id: str,
    source_type: str,
    start: str,
    end: str,
    status: str = "observed",
    coverage_ratio: float = 0.0,
    run_id: str = "",
) -> CoverageInterval:
    """Create a CoverageInterval."""
    return CoverageInterval(
        entity_id=entity_id,
        source_type=source_type,
        start=start,
        end=end,
        status=status,
        coverage_ratio=coverage_ratio,
        run_id=run_id,
    )


def create_timeline_event(
    entity_id: str,
    event_type: str,
    timestamp: str,
    duration_seconds: float = None,
    source_id: str = "",
    description: str = "",
    confidence: float = 0.8,
    run_id: str = "",
) -> TimelineEvent:
    """Create a TimelineEvent with auto-generated ID."""
    return TimelineEvent(
        id=generate_id("TLEVT", f"{entity_id}_{event_type}_{timestamp}"),
        entity_id=entity_id,
        event_type=event_type,
        timestamp=timestamp,
        duration_seconds=duration_seconds,
        source_id=source_id,
        description=description,
        confidence=confidence,
        run_id=run_id,
    )
