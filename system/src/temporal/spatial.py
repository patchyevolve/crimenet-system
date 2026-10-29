"""
Spatial Normalization — geocoding, location parsing, precision assessment.
"""

import re
from typing import Optional, Tuple, Dict, List
from .schema import SpatialPrecision


# Known locations in India (expanded for demo data)
KNOWN_LOCATIONS: Dict[str, Tuple[float, float]] = {
    # Madhya Pradesh
    "guna": (24.6355, 77.3116),
    "guna district": (24.6355, 77.3116),
    "guna city": (24.6355, 77.3116),
    "shivpuri": (25.4229, 77.6575),
    "rajendra nagar": (24.6355, 77.3116),
    "indore": (22.7196, 75.8577),
    "bhopal": (23.2599, 77.4126),
    "khandwa": (21.8258, 76.3515),
    "ujjain": (23.1765, 75.7885),
    "satna": (24.3524, 80.8332),
    "jabalpur": (23.1815, 79.9864),
    "gwalior": (26.2183, 78.1828),
    "sagar": (23.8388, 78.7378),
    "rewa": (24.5330, 81.3015),
    "damoh": (23.8368, 79.4703),
    "chhatarpur": (24.9175, 79.5884),
    "panna": (24.7221, 80.1856),
    "satna": (24.3524, 80.8332),

    # Delhi NCR
    "delhi": (28.7041, 77.1025),
    "ncr": (28.7041, 77.1025),
    "noida": (28.5355, 77.3910),
    "gurgaon": (28.4595, 77.0266),
    "faridabad": (28.4089, 77.3178),
    "ghaziabad": (28.6692, 77.4538),
    "karol bagh": (28.6514, 77.1894),
    "connaught place": (28.6315, 77.2167),
    "lajpat nagar": (28.5700, 77.2430),
    "rajouri garden": (28.6486, 77.1219),
    "rohini": (28.7495, 77.0654),
    "dwarka": (28.5921, 77.0460),
    "rajendra nagar, delhi": (28.6415, 77.1792),
    "old rajendra nagar": (28.6415, 77.1792),
    "new rajendra nagar": (28.6385, 77.1812),

    # Maharashtra
    "mumbai": (19.0760, 72.8777),
    "pune": (18.5204, 73.8567),
    "nagpur": (21.1458, 79.0882),
    "thane": (19.2183, 72.9781),
    "navi mumbai": (19.0330, 73.0297),

    # Karnataka
    "bangalore": (12.9716, 77.5946),
    "bengaluru": (12.9716, 77.5946),
    "mysore": (12.2958, 76.6394),
    "hubli": (15.3647, 75.1240),
    "mangalore": (12.9141, 74.8560),

    # Tamil Nadu
    "chennai": (13.0827, 80.2707),
    "coimbatore": (11.0168, 76.9558),
    "madurai": (9.9252, 78.1198),

    # Andhra Pradesh / Telangana
    "hyderabad": (17.3850, 78.4867),

    # Gujarat
    "ahmedabad": (23.0225, 72.5714),
    "surat": (21.1702, 72.8311),
    "vadodara": (22.3072, 73.1812),

    # West Bengal
    "kolkata": (22.5726, 88.3639),

    # Kerala
    "kochi": (9.9312, 76.2673),
    "trivandrum": (8.5241, 76.9366),

    # Odisha
    "bhubaneswar": (20.2961, 85.8245),

    # Punjab
    "chandigarh": (30.7333, 76.7794),

    # Bihar
    "patna": (25.6093, 85.1376),

    # Jharkhand
    "ranchi": (23.3441, 85.3096),

    # Chhattisgarh
    "raipur": (21.2514, 81.6296),

    # Goa
    "goa": (15.2993, 74.1240),
    "panaji": (15.4909, 73.8278),
    "mapusa": (15.5953, 73.8148),
    "calangute": (15.5449, 73.7551),

    # UAE
    "dubai": (25.2048, 55.2708),
    "dubai, uae": (25.2048, 55.2708),
    "abu dhabi": (24.4539, 54.3773),
    "sharjah": (25.3573, 55.3908),

    # Rajasthan
    "jaipur": (26.9124, 75.7873),
    "jodhpur": (26.2389, 73.0243),
    "udaipur": (24.5854, 73.7125),
    "kota": (25.1825, 75.8573),
    "ajmer": (26.4499, 74.6399),

    # Uttar Pradesh
    "lucknow": (26.8467, 80.9462),
    "agra": (27.1767, 78.0081),
    "varanasi": (25.3176, 82.9739),
    "prayagraj": (25.4358, 81.8463),
    "meerut": (28.9845, 77.7066),

    # Cell tower areas (approximate)
    "cell_456": (24.6355, 77.3116),
    "tower_789": (24.6355, 77.3116),

    # Villages
    "dhaneli": (24.6355, 77.3116),
    "village dhaneli": (24.6355, 77.3116),
}

# Tower ID to city/state mapping — city must exist in KNOWN_LOCATIONS
TOWER_CITY_MAP: Dict[str, Tuple[str, str]] = {
    "DEL": ("Delhi", "Delhi"),
    "UP": ("Noida", "Uttar Pradesh"),
    "MH": ("Mumbai", "Maharashtra"),
    "KA": ("Bangalore", "Karnataka"),
    "TN": ("Chennai", "Tamil Nadu"),
    "GJ": ("Ahmedabad", "Gujarat"),
    "RJ": ("Jaipur", "Rajasthan"),
    "MP": ("Bhopal", "Madhya Pradesh"),
    "WB": ("Kolkata", "West Bengal"),
    "AP": ("Hyderabad", "Andhra Pradesh"),
    "TS": ("Hyderabad", "Telangana"),
    "KL": ("Kochi", "Kerala"),
    "OR": ("Bhubaneswar", "Odisha"),
    "PB": ("Chandigarh", "Punjab"),
    "HR": ("Gurgaon", "Haryana"),
    "BR": ("Patna", "Bihar"),
    "JH": ("Ranchi", "Jharkhand"),
    "CG": ("Raipur", "Chhattisgarh"),
    "GA": ("Panaji", "Goa"),
}

# Location patterns
LOCATION_PATTERNS = [
    # "near Hotel Taj, Guna"
    r'near\s+(.+?)(?:,\s*(.+))?$',
    # "area of Delhi"
    r'area\s+of\s+(.+)',
    # "Delhi NCR"
    r'(.+?)\s+(?:NCR|metro|region)',
    # "Hotel Taj, Guna"
    r'(.+?),\s*(.+)',
    # Just a city name
    r'^([A-Za-z\s]+)$',
]


def parse_location_string(location: str) -> Dict[str, str]:
    """
    Parse a location string into components.
    Returns dict with city, state, address, etc.
    """
    if not location:
        return {}

    location = location.strip()
    result = {"raw": location}

    # Try known locations first
    location_lower = location.lower()
    if location_lower in KNOWN_LOCATIONS:
        lat, lon = KNOWN_LOCATIONS[location_lower]
        result["latitude"] = lat
        result["longitude"] = lon
        result["precision"] = SpatialPrecision.EXACT
        # Determine city/state
        if "guna" in location_lower:
            result["city"] = "Guna"
            result["state"] = "Madhya Pradesh"
        elif "delhi" in location_lower or "ncr" in location_lower:
            result["city"] = "Delhi"
            result["state"] = "Delhi"
        elif "indore" in location_lower:
            result["city"] = "Indore"
            result["state"] = "Madhya Pradesh"
        return result

    # Tower ID pattern: DEL_TWR_042
    tower_match = re.match(r'([A-Z]{2,3})_TWR_(\d+)', location, re.IGNORECASE)
    if tower_match:
        prefix = tower_match.group(1).upper()
        tower_num = tower_match.group(2)
        if prefix in TOWER_CITY_MAP:
            city, state = TOWER_CITY_MAP[prefix]
            result["address"] = f"Cell Tower {location}"
            result["city"] = city
            result["state"] = state
            result["precision"] = SpatialPrecision.TOWER
            coords = get_coordinates(location)
            if coords:
                result["latitude"] = coords[0]
                result["longitude"] = coords[1]
            return result

    # Parse pattern-based locations
    for pattern in LOCATION_PATTERNS:
        match = re.match(pattern, location, re.IGNORECASE)
        if match:
            groups = match.groups()
            if len(groups) >= 1:
                result["address"] = groups[0]
            if len(groups) >= 2 and groups[1]:
                result["city"] = groups[1]
            break

    # Assess precision
    if re.search(r'\d+\.\d+,\s*\d+\.\d+', location):
        # GPS coordinates
        coords = re.search(r'(\d+\.\d+),\s*(\d+\.\d+)', location)
        if coords:
            result["latitude"] = float(coords.group(1))
            result["longitude"] = float(coords.group(2))
            result["precision"] = SpatialPrecision.EXACT
    elif any(kw in location_lower for kw in ["near", "area", "vicinity", "around"]):
        result["precision"] = SpatialPrecision.VAGUE
    elif re.search(r'[A-Z]{2,3}_TWR_', location, re.IGNORECASE) or "cell" in location_lower or "tower" in location_lower:
        result["precision"] = SpatialPrecision.TOWER
    elif result.get("city"):
        result["precision"] = SpatialPrecision.CITY
    else:
        result["precision"] = SpatialPrecision.UNKNOWN

    return result


def get_coordinates(location: str) -> Optional[Tuple[float, float]]:
    """
    Get coordinates for a location.
    Returns (latitude, longitude) or None.
    """
    location_lower = location.lower().strip()

    # Check known locations (exact match)
    if location_lower in KNOWN_LOCATIONS:
        return KNOWN_LOCATIONS[location_lower]

    # Check partial match — e.g. "Karol Bagh, Delhi" -> try "delhi"
    parts = [p.strip().lower() for p in location.split(",")]
    for part in parts:
        if part in KNOWN_LOCATIONS:
            return KNOWN_LOCATIONS[part]

    # Try tower ID pattern: "DEL_TWR_042" -> "DEL" -> Delhi
    tower_match = re.match(r'([A-Z]{2,3})_TWR_(\d+)', location, re.IGNORECASE)
    if tower_match:
        prefix = tower_match.group(1).upper()
        tower_num = int(tower_match.group(2))
        if prefix in TOWER_CITY_MAP:
            city, _ = TOWER_CITY_MAP[prefix]
            base_coords = KNOWN_LOCATIONS.get(city.lower())
            if base_coords:
                # Deterministic pseudo-offset within ~3-5km of city center
                offset_lat = ((tower_num * 17) % 60 - 30) * 0.001
                offset_lng = ((tower_num * 31) % 60 - 30) * 0.001
                return (round(base_coords[0] + offset_lat, 4), round(base_coords[1] + offset_lng, 4))

    # Try GPS coordinates
    coords = re.search(r'(\d+\.?\d*)\s*,\s*(\d+\.?\d*)', location)
    if coords:
        lat = float(coords.group(1))
        lon = float(coords.group(2))
        if 6 < lat < 38 and 68 < lon < 98:  # Rough India bounds
            return (lat, lon)

    return None


def assess_spatial_precision(location: str) -> SpatialPrecision:
    """Assess the precision of a location description."""
    if not location:
        return SpatialPrecision.UNKNOWN

    location_lower = location.lower()

    # GPS coordinates = exact
    if re.search(r'\d+\.?\d*\s*,\s*\d+\.?\d*', location):
        return SpatialPrecision.EXACT

    # Cell tower = tower precision (pattern: DEL_TWR_042, UP_TWR_012, MH_TWR_005)
    if re.search(r'[A-Z]{2,3}_TWR_', location, re.IGNORECASE) or "cell" in location_lower or "tower" in location_lower:
        return SpatialPrecision.TOWER

    # Vague indicators
    vague_keywords = ["near", "area", "vicinity", "around", "somewhere", "close to"]
    if any(kw in location_lower for kw in vague_keywords):
        return SpatialPrecision.VAGUE

    # City-level — check known cities
    known_cities = list(KNOWN_LOCATIONS.keys())
    if any(city in location_lower for city in known_cities):
        return SpatialPrecision.CITY

    # Specific address/hotel
    if any(kw in location_lower for kw in ["hotel", "road", "station", "stand", "shop", "airport", "nagar", "bagh"]):
        return SpatialPrecision.CITY  # Treat as city-level

    return SpatialPrecision.UNKNOWN


def get_radius_km(precision: SpatialPrecision) -> float:
    """Get approximate radius in km for a precision level."""
    radii = {
        SpatialPrecision.EXACT: 0.1,      # 100 meters
        SpatialPrecision.TOWER: 2.0,      # 2 km
        SpatialPrecision.CITY: 5.0,       # 5 km
        SpatialPrecision.VAGUE: 10.0,     # 10 km
        SpatialPrecision.UNKNOWN: 15.0,   # 15 km
    }
    return radii.get(precision, 15.0)


def normalize_location(location: str) -> Dict[str, str]:
    """
    Normalize a location string to structured data.
    Returns dict with latitude, longitude, city, state, precision, etc.
    """
    parsed = parse_location_string(location)

    # Ensure coordinates
    if "latitude" not in parsed:
        coords = get_coordinates(location)
        if coords:
            parsed["latitude"] = coords[0]
            parsed["longitude"] = coords[1]

    # Ensure precision
    if "precision" not in parsed:
        parsed["precision"] = assess_spatial_precision(location)

    # Add radius
    if isinstance(parsed.get("precision"), SpatialPrecision):
        parsed["radius_km"] = get_radius_km(parsed["precision"])
        parsed["precision"] = parsed["precision"].value

    return parsed
