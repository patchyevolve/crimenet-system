"""Geospatial Crime Zone Risk Mapping — H3 hex-based risk scoring.

Converts location entities to hexagonal grid cells and scores each cell
by evidence density, suspect activity, and temporal recency.

Uses geopy Nominatim for real geocoding with a persistent local cache.
"""

import json
import re
import time
from typing import List, Dict, Optional, Tuple, Iterable
from collections import defaultdict
from pathlib import Path
from dataclasses import dataclass, field


# Local cache for geocoding results (avoids repeated API calls)
_GEOCODE_CACHE: Dict[str, Optional[Tuple[float, float]]] = {}

# Persistent disk cache
_GEOCODE_CACHE_PATH = Path.home() / ".cache" / "criminal_network" / "geocode_cache.json"
_GEOCODE_CACHE_LOADED = False

# Shared geolocator (created once)
_GEOLocator = None

# H3 resolution 9 = ~0.1 km² per hex cell
H3_RESOLUTION = 9

# Risk band thresholds
RISK_BANDS = {
    "GREEN": (0.0, 0.3),
    "AMBER": (0.3, 0.7),
    "RED": (0.7, 1.0),
}

# Nominatim user agent (required by usage policy)
_NOMINATIM_USER_AGENT = "criminal_network_analysis_v1"

# Noise suffixes to strip from location names (cameras, towers, gates)
_NOISE_SUFFIX_RE = re.compile(
    r"\s+(?:camera|tower|gate|cctv|cam)\s*\d*\s*$",
    re.IGNORECASE,
)

# Names too short / non-geographic / possessive fragments to skip
_MIN_NAME_LEN = 4
_NON_GEO_TOKENS = {
    "hai", "flat", "room", "house", "road", "street", "near",
    "the", "and", "with", "from", "this", "that",
}


@dataclass
class ZoneScore:
    """A geospatial zone risk score."""
    hex_id: str
    latitude: float
    longitude: float
    location_names: List[str]
    evidence_count: int
    suspect_count: int
    risk_score: float
    risk_band: str
    case_ids: List[str] = field(default_factory=list)
    run_id: str = ""

    def to_dict(self):
        return {
            "hex_id": self.hex_id,
            "latitude": self.latitude,
            "longitude": self.longitude,
            "location_names": self.location_names,
            "evidence_count": self.evidence_count,
            "suspect_count": self.suspect_count,
            "risk_score": self.risk_score,
            "risk_band": self.risk_band,
            "case_ids": self.case_ids,
            "run_id": self.run_id,
        }


def _load_disk_cache():
    global _GEOCODE_CACHE_LOADED
    if _GEOCODE_CACHE_LOADED:
        return
    _GEOCODE_CACHE_LOADED = True
    try:
        if _GEOCODE_CACHE_PATH.exists():
            with open(_GEOCODE_CACHE_PATH) as f:
                raw = json.load(f)
            for k, v in raw.items():
                _GEOCODE_CACHE[k] = tuple(v) if v else None
    except Exception:
        pass


def _save_disk_cache():
    try:
        _GEOCODE_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        serializable = {k: list(v) if v else None for k, v in _GEOCODE_CACHE.items()}
        with open(_GEOCODE_CACHE_PATH, "w") as f:
            json.dump(serializable, f, indent=2)
    except Exception:
        pass


def _get_geolocator():
    global _GEOLocator
    if _GEOLocator is None:
        from geopy.geocoders import Nominatim
        _GEOLocator = Nominatim(user_agent=_NOMINATIM_USER_AGENT, timeout=5)
    return _GEOLocator


def _geocode_nominatim(location_name: str) -> Optional[Tuple[float, float]]:
    """Geocode a single string via Nominatim. Returns (lat, lng) or None."""
    try:
        g = _get_geolocator()
        loc = g.geocode(location_name, exactly_one=True, language="en")
        if loc:
            return (loc.latitude, loc.longitude)
        return None
    except Exception:
        return None


def _clean_location_name(raw: str) -> Optional[str]:
    """Strip camera/tower noise; return None if name is not geocodable."""
    if not raw or not raw.strip():
        return None
    name = raw.strip()
    # Strip trailing noise (Camera 5, Tower 2, Gate, etc.)
    name = _NOISE_SUFFIX_RE.sub("", name).strip()
    # Strip trailing/leading punctuation
    name = name.strip(" ,.-")
    if len(name) < _MIN_NAME_LEN:
        return None
    # Skip possessive fragments like "Meena Devi's"
    if name.endswith("'s") or name.endswith("’s"):
        return None
    # Skip pure non-geographic short tokens
    if name.lower() in _NON_GEO_TOKENS:
        return None
    # Skip short Devanagari fragments (likely narrative, not place names)
    if re.fullmatch(r"[ऀ-ॿ\s\d]+", name):
        # Skip if contains digits (e.g., "फ्लैट 5ए") — flat numbers aren't geocodable
        if re.search(r"\d", name):
            return None
        # Skip very short fragments
        if len(name) < 8:
            return None
    # Skip Hindi phrases that are clearly narrative fragments
    hindi_noise = ("से", "को", "का", "की", "और", "है", "हैं", "मैंने", "रहता", "हो", "उनमें", "खाते", "हमारे", "रोज")
    if re.fullmatch(r"[ऀ-ॿ\s\d]+", name) and any(w in name for w in hindi_noise):
        return None
    return name


def _candidate_names(entity: dict) -> List[str]:
    """Extract candidate geocodable names from a location entity.

    Prefers structured attributes (location_name list) over the raw entity name.
    Deduplicates after cleaning.
    """
    candidates: List[str] = []

    attrs = entity.get("attributes") or {}

    # location_name may be a list or a string
    ln = attrs.get("location_name")
    if isinstance(ln, list):
        candidates.extend(ln)
    elif isinstance(ln, str):
        candidates.append(ln)

    # Raw entity name as fallback
    raw = entity.get("name", "")
    if raw:
        candidates.append(raw)

    # Clean + dedupe while preserving order
    seen = set()
    cleaned: List[str] = []
    for c in candidates:
        cc = _clean_location_name(str(c))
        if cc and cc.lower() not in seen:
            seen.add(cc.lower())
            cleaned.append(cc)
    return cleaned


def _build_variants(name: str) -> List[str]:
    """Build geocoding query variants: full name, progressive token stripping, contexts.

    Context-suffixed variants are tried BEFORE bare names for multi-token names
    to reduce ambiguity (e.g., "Nehru Nagar" resolving to Mumbai instead of Delhi).
    """
    # Progressive stripping of trailing qualifier tokens (Part II, Phase 2, Sector 12, etc.)
    bases: List[str] = [name]
    tokens = name.split()
    qualifier_re = re.compile(
        r"^(?:part|phase|sector|block|wing|tower|gate|road|rd|st|street|colony|extension|ext)$",
        re.IGNORECASE,
    )
    while len(tokens) > 1:
        last = tokens[-1]
        if qualifier_re.match(last) or re.fullmatch(r"[0-9]+|[ivxlcdm]+", last, re.IGNORECASE):
            tokens = tokens[:-1]
            stripped = " ".join(tokens)
            if stripped not in bases:
                bases.append(stripped)
        else:
            break

    variants: List[str] = []
    if "," in name:
        # Already has context — try as-is first, then stripped bases
        variants.append(name)
        for b in bases[1:]:
            variants.append(b)
    else:
        # Prefer Delhi/India context for bare names to avoid ambiguous national hits
        for b in bases:
            variants.append(f"{b}, Delhi, India")
        for b in bases:
            variants.append(f"{b}, India")
        for b in bases:
            if b not in variants:
                variants.append(b)

    # Dedupe preserving order
    seen = set()
    unique: List[str] = []
    for v in variants:
        if v not in seen:
            seen.add(v)
            unique.append(v)
    return unique


def _get_coords(name: str) -> Optional[Tuple[float, float]]:
    """Get coordinates for a (already cleaned) location name.

    1. Check cache (memory + disk)
    2. Try Nominatim with progressive variants (strip Part II / Phase 2 / etc.)
    3. Cache result (even None) to avoid repeat lookups
    """
    if not name or not name.strip():
        return None

    _load_disk_cache()
    key = name.lower().strip()

    if key in _GEOCODE_CACHE:
        return _GEOCODE_CACHE[key]

    variants = _build_variants(name)

    result: Optional[Tuple[float, float]] = None
    for attempt in variants:
        result = _geocode_nominatim(attempt)
        # Rate limit: 1 request per second (Nominatim policy)
        time.sleep(1.0)
        if result:
            break

    _GEOCODE_CACHE[key] = result
    _save_disk_cache()
    return result


def _get_coords_for_entity(entity: dict) -> Optional[Tuple[float, float]]:
    """Try candidate names for an entity until one geocodes."""
    for cand in _candidate_names(entity):
        coords = _get_coords(cand)
        if coords:
            return coords
    return None


def _latlng_to_h3(lat: float, lng: float) -> str:
    """Convert lat/lng to H3 hex index. Fallback to simple grid if h3 not installed."""
    try:
        import h3
        if hasattr(h3, "latlng_to_cell"):
            return str(h3.latlng_to_cell(lat, lng, H3_RESOLUTION))
        if hasattr(h3, "geo_to_h3"):
            return str(h3.geo_to_h3(lat, lng, H3_RESOLUTION))
        return f"GRID_{round(lat, 3)}_{round(lng, 3)}"
    except Exception:  # noqa: BLE001 - any h3 failure degrades to the grid
        return f"GRID_{round(lat, 3)}_{round(lng, 3)}"


def compute_zone_scores(
    location_entities: List[dict],
    graph_nodes: List[dict],
    contradictions: dict,
    temporal_infos: List[dict],
    run_id: str = "",
) -> List[ZoneScore]:
    """Compute geospatial zone risk scores.

    1. Geocode location entities (cleaned names) via Nominatim
    2. Convert to H3 hex cells
    3. Score each cell by evidence density + suspect activity
    4. Assign risk bands (GREEN/AMBER/RED)
    """
    hex_data = defaultdict(lambda: {
        "lat": 0.0, "lng": 0.0,
        "location_names": set(),
        "evidence_count": 0,
        "suspect_count": 0,
        "case_ids": set(),
    })

    geocoded_count = 0
    failed_count = 0

    # Process location entities
    for entity in location_entities:
        coords = _get_coords_for_entity(entity)
        if not coords:
            failed_count += 1
            continue

        geocoded_count += 1
        lat, lng = coords
        hex_id = _latlng_to_h3(lat, lng)

        # Prefer cleaned display name
        display = _clean_location_name(entity.get("name", "")) or entity.get("name", "")
        hex_data[hex_id]["lat"] = lat
        hex_data[hex_id]["lng"] = lng
        hex_data[hex_id]["location_names"].add(display)
        hex_data[hex_id]["evidence_count"] += 1

    # Process person nodes for suspect activity via structured location attributes only
    _PERSON_LOC_ATTRS = ("location_name", "tower_location", "address", "location")
    for node in graph_nodes:
        if node.get("node_type") != "person":
            continue
        attrs = node.get("attributes") or {}
        person_name = node.get("name", "")

        for key in _PERSON_LOC_ATTRS:
            attr_val = attrs.get(key)
            if attr_val is None:
                continue
            values = attr_val if isinstance(attr_val, list) else [attr_val]
            found = False
            for v in values:
                cleaned = _clean_location_name(str(v))
                if not cleaned:
                    continue
                coords = _get_coords(cleaned)
                if coords:
                    lat, lng = coords
                    hex_id = _latlng_to_h3(lat, lng)
                    hex_data[hex_id]["lat"] = lat
                    hex_data[hex_id]["lng"] = lng
                    hex_data[hex_id]["suspect_count"] += 1
                    if person_name:
                        hex_data[hex_id]["location_names"].add(person_name)
                    found = True
                    break  # one location per person is enough
            if found:
                break

    print(f"[GEOSPATIAL] Geocoded {geocoded_count} locations, {failed_count} not found")

    # Compute risk scores
    max_evidence = max((h["evidence_count"] for h in hex_data.values()), default=1) or 1
    max_suspects = max((h["suspect_count"] for h in hex_data.values()), default=1) or 1

    results = []
    for hex_id, data in hex_data.items():
        evidence_factor = data["evidence_count"] / max_evidence
        suspect_factor = data["suspect_count"] / max_suspects
        risk_score = round(min(1.0, evidence_factor * 0.6 + suspect_factor * 0.4), 4)

        risk_band = "GREEN"
        for band, (low, high) in RISK_BANDS.items():
            if low <= risk_score < high or (band == "RED" and risk_score >= high):
                risk_band = band
                break

        results.append(ZoneScore(
            hex_id=hex_id,
            latitude=data["lat"],
            longitude=data["lng"],
            location_names=sorted(data["location_names"]),
            evidence_count=data["evidence_count"],
            suspect_count=data["suspect_count"],
            risk_score=risk_score,
            risk_band=risk_band,
            run_id=run_id,
        ))

    results.sort(key=lambda z: z.risk_score, reverse=True)
    return results
