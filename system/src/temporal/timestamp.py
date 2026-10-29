"""
Timestamp Normalization and Precision Modeling.
Converts various date/time formats to standardized ISO 8601.

Relative expressions ("evening", "between 14:00 and 16:00") are resolved
against a *deterministic* reference date — never the wall clock. Crime
records are historical (typically 2024), so stamping the execution date
onto them would silently falsify every derived interval.
"""

import re
from datetime import date, datetime
from typing import Optional, Tuple, List
from .schema import TemporalPrecision


# Common Indian date formats
DATE_FORMATS = [
    # ISO formats — offset-aware variants first: "%z" also accepts "Z", so a
    # timezone-bearing input is normalized to an aware datetime (and keeps its
    # offset in isoformat()) instead of falling through to the year-only regex.
    "%Y-%m-%dT%H:%M:%S%z",
    "%Y-%m-%dT%H:%M:%SZ",
    "%Y-%m-%dT%H:%M:%S",
    "%Y-%m-%d %H:%M:%S%z",
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%d %H:%M%z",
    "%Y-%m-%d %H:%M",
    "%Y-%m-%d",
    # Indian formats (offset-aware variants first, as above)
    "%d/%m/%Y %H:%M:%S%z",
    "%d/%m/%Y %H:%M:%S",
    "%d/%m/%Y %H:%M%z",
    "%d/%m/%Y %H:%M",
    "%d/%m/%Y",
    "%d-%m-%Y %H:%M:%S%z",
    "%d-%m-%Y %H:%M:%S",
    "%d-%m-%Y %H:%M%z",
    "%d-%m-%Y %H:%M",
    "%d-%m-%Y",
    "%d.%m.%Y",
    "%d %b %Y",
    "%d %B %Y",
    "%b %d, %Y",
    "%B %d, %Y",
    "%Y/%m/%d",
    "%m/%d/%Y",
]

# Date shapes accepted for an explicit reference_date argument.
REFERENCE_DATE_FORMATS = [
    "%Y-%m-%d",
    "%Y/%m/%d",
    "%d/%m/%Y",
    "%d-%m-%Y",
    "%d.%m.%Y",
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%dT%H:%M:%S",
    "%Y-%m-%d %H:%M",
    "%d %b %Y",
    "%d %B %Y",
    "%b %d, %Y",
    "%B %d, %Y",
    "%b %Y",
    "%B %Y",
]

# Deterministic historical fallback when neither reference_date nor the text
# itself yields a date. Never datetime.now() — see module docstring.
HISTORICAL_DEFAULT_DATE = date(2024, 1, 1)

# Timezone patterns
TZ_PATTERNS = [
    (r'\s*IST\s*', '+05:30'),
    (r'\s*UTC\s*', '+00:00'),
    (r'\s*GMT\s*', '+00:00'),
    (r'\s*\+05:30\s*', '+05:30'),
    (r'\s*\+00:00\s*', '+00:00'),
]

# Precision indicators
PRECISION_KEYWORDS = {
    TemporalPrecision.EXACT: [
        "am", "pm", ":", "hour", "minute", "exact",
    ],
    TemporalPrecision.APPROXIMATE: [
        "morning", "afternoon", "evening", "night",
        "around", "approximately", "about", "roughly",
        "early", "late", "before", "after",
    ],
    TemporalPrecision.RANGE: [
        "between", "from", "to", "-", "to",
    ],
}

# Time of day ranges
TIME_RANGES = {
    "morning": (6, 12),
    "afternoon": (12, 17),
    "evening": (17, 21),
    "night": (21, 6),
    "early morning": (5, 8),
    "late night": (22, 4),
}


def _parse_reference_date(value: str) -> Optional[date]:
    """Parse an explicit reference value ("2024-03-12") into a date.

    Accepts whole-string dates first, then any ISO date embedded in a longer
    phrase, then a bare year (Jan 1). Returns None when nothing matches.
    """
    if not value:
        return None
    candidate = str(value).strip()
    if not candidate:
        return None

    for fmt in REFERENCE_DATE_FORMATS:
        try:
            return datetime.strptime(candidate, fmt).date()
        except ValueError:
            continue

    try:
        return datetime.fromisoformat(candidate.replace("Z", "+00:00")).date()
    except ValueError:
        pass

    iso_date = re.search(r'\b(\d{4})-(\d{1,2})-(\d{1,2})\b', candidate)
    if iso_date:
        try:
            return date(int(iso_date.group(1)), int(iso_date.group(2)),
                        int(iso_date.group(3)))
        except ValueError:
            pass

    year = re.search(r'\b(19\d{2}|20\d{2})\b', candidate)
    if year:
        return date(int(year.group(1)), 1, 1)

    return None


def _resolve_reference_date(text: str = "",
                            reference_date: Optional[str] = None) -> date:
    """Deterministic base date for relative time expressions.

    Priority: explicit reference_date → date/year found in the text →
    HISTORICAL_DEFAULT_DATE (2024-01-01). The wall clock is never consulted.
    """
    if reference_date:
        parsed = _parse_reference_date(reference_date)
        if parsed:
            return parsed
    if text:
        parsed = _parse_reference_date(text)
        if parsed:
            return parsed
    return HISTORICAL_DEFAULT_DATE


def clean_timestamp_string(ts: str) -> str:
    """Clean a timestamp string, removing extra whitespace and normalizing.

    Timezone labels are substituted with standard numeric offsets via
    TZ_PATTERNS so they can be consumed by the "%z" DATE_FORMATS entries
    ("2024-03-12 14:30:00 IST" → "2024-03-12 14:30:00+05:30").
    """
    if not ts:
        return ""
    # Remove extra whitespace
    ts = re.sub(r'\s+', ' ', ts.strip())
    # Remove trailing periods
    ts = ts.rstrip('.')
    # Normalize timezone labels / offsets (IST, UTC, GMT, +05:30, +00:00)
    for pattern, replacement in TZ_PATTERNS:
        ts = re.sub(pattern, replacement, ts, flags=re.IGNORECASE)
    return ts


def detect_precision(text: str) -> TemporalPrecision:
    """Detect the precision of a timestamp from its text representation."""
    text_lower = text.lower()

    # Check for exact indicators (has time component) FIRST
    if re.search(r'\d{1,2}:\d{2}', text):
        return TemporalPrecision.EXACT

    # Check for date-only (day precision) — specific date = exact
    if re.search(r'\d{4}-\d{2}-\d{2}|\d{2}/\d{2}/\d{4}', text):
        return TemporalPrecision.EXACT

    # Check for range indicators (must be careful not to match date hyphens)
    # Only match "between X and Y" or "from X to Y" patterns, not hyphens in dates
    if re.search(r'between\s+.+\s+and\s+|from\s+.+\s+to\s+', text_lower):
        return TemporalPrecision.RANGE

    # Check for approximate indicators
    if any(kw in text_lower for kw in PRECISION_KEYWORDS[TemporalPrecision.APPROXIMATE]):
        return TemporalPrecision.APPROXIMATE

    # Check for month-only or year-only (lower precision)
    if re.search(r'^(january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{4}$', text_lower):
        return TemporalPrecision.RANGE  # Month precision
    if re.search(r'^\d{4}$', text):
        return TemporalPrecision.RANGE  # Year precision

    return TemporalPrecision.UNKNOWN


def parse_timestamp(text: str,
                    reference_date: Optional[str] = None
                    ) -> Optional[Tuple[str, TemporalPrecision]]:
    """
    Parse a timestamp string into ISO 8601 format.
    Returns (iso_timestamp, precision) or None if parsing fails.

    ``reference_date`` anchors relative expressions ("evening",
    "between 14:00 and 16:00"). When absent, the date is taken from ``text``
    itself, else HISTORICAL_DEFAULT_DATE (2024-01-01). The wall clock is
    never used.
    """
    if not text:
        return None

    original = text
    text = clean_timestamp_string(text)

    # Handle time-of-day descriptions
    for time_word, (start_hour, end_hour) in TIME_RANGES.items():
        if time_word in text.lower():
            # Anchor to the deterministic reference date, not today. Only the
            # start hour is materialized (as before); midnight-wrapping ranges
            # therefore cannot overflow into the wrong month.
            ref = _resolve_reference_date(text, reference_date)
            start = datetime(ref.year, ref.month, ref.day, start_hour, 0)
            return (start.isoformat(), TemporalPrecision.RANGE)

    # Handle range descriptions
    range_match = re.search(r'between\s+(\d{1,2}:\d{2})\s+and\s+(\d{1,2}:\d{2})', text, re.IGNORECASE)
    if range_match:
        t1 = range_match.group(1)
        t2 = range_match.group(2)
        ref = _resolve_reference_date(text, reference_date)
        h1, m1 = map(int, t1.split(':'))
        h2, m2 = map(int, t2.split(':'))
        start = datetime(ref.year, ref.month, ref.day, h1, m1)
        end = datetime(ref.year, ref.month, ref.day, h2, m2)
        return (start.isoformat(), TemporalPrecision.RANGE)

    # Try each format
    for fmt in DATE_FORMATS:
        try:
            dt = datetime.strptime(text, fmt)
            precision = detect_precision(original)
            # If no time component, set to start of day
            if '%H' not in fmt:
                dt = dt.replace(hour=0, minute=0, second=0)
            return (dt.isoformat(), precision)
        except ValueError:
            continue

    # Try partial date matching (month/year only)
    month_year = re.search(r'(\w+)\s+(\d{4})', text)
    if month_year:
        month_str = month_year.group(1)
        year = int(month_year.group(2))
        try:
            dt = datetime.strptime(f"{month_str} {year}", "%B %Y")
            return (dt.isoformat(), TemporalPrecision.RANGE)  # Month precision
        except ValueError:
            try:
                dt = datetime.strptime(f"{month_str} {year}", "%b %Y")
                return (dt.isoformat(), TemporalPrecision.RANGE)
            except ValueError:
                pass

    # Try year only
    year_match = re.search(r'\b(20\d{2})\b', text)
    if year_match:
        year = int(year_match.group(1))
        dt = datetime(year, 1, 1)
        return (dt.isoformat(), TemporalPrecision.RANGE)

    return None


def normalize_timestamp(ts: str,
                        reference_date: Optional[str] = None) -> Optional[str]:
    """Normalize a timestamp to ISO 8601 format."""
    result = parse_timestamp(ts, reference_date=reference_date)
    return result[0] if result else None


def get_time_range(text: str,
                   reference_date: Optional[str] = None) -> Optional[Tuple[str, str]]:
    """
    Extract a time range from text.
    Returns (start_iso, end_iso) or None.

    Relative ranges use the same deterministic reference date as
    ``parse_timestamp`` — never the wall clock.
    """
    if not text:
        return None

    # Handle "between X and Y" patterns
    range_match = re.search(r'between\s+(\d{1,2}:\d{2})\s+and\s+(\d{1,2}:\d{2})', text, re.IGNORECASE)
    if range_match:
        t1 = range_match.group(1)
        t2 = range_match.group(2)
        ref = _resolve_reference_date(text, reference_date)
        h1, m1 = map(int, t1.split(':'))
        h2, m2 = map(int, t2.split(':'))
        start = datetime(ref.year, ref.month, ref.day, h1, m1)
        end = datetime(ref.year, ref.month, ref.day, h2, m2)
        return (start.isoformat(), end.isoformat())

    # Handle date ranges (DD/MM/YYYY to DD/MM/YYYY)
    date_range = re.search(
        r'(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})\s+to\s+(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})',
        text
    )
    if date_range:
        start = normalize_timestamp(date_range.group(1), reference_date=reference_date)
        end = normalize_timestamp(date_range.group(2), reference_date=reference_date)
        if start and end:
            return (start, end)

    return None


def format_duration(seconds: float) -> str:
    """Format duration in human-readable form."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    elif seconds < 86400:
        hours = seconds / 3600
        return f"{hours:.1f}h"
    else:
        days = seconds / 86400
        return f"{days:.1f}d"
