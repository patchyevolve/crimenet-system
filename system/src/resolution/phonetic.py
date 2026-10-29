"""
Phonetic Matching Engine
Soundex + Metaphone + Custom Indian Name Transliteration Scheme.
Handles Hindi→English phonetic mapping for Indian names.
"""

import re
from typing import List, Tuple


# ──────────────────────────────────────────────
# Standard Soundex
# ──────────────────────────────────────────────

_SOUNDEX_MAP = str.maketrans("BFPVCGJKQSXZDTLMNR", "111122222222334556")


def soundex(name: str) -> str:
    """American Soundex algorithm. Returns 4-character code."""
    name = name.upper().strip()
    if not name:
        return ""

    # Keep first letter
    code = name[0]

    # Map remaining letters
    for ch in name[1:]:
        mapped = ch.translate(_SOUNDEX_MAP)
        if mapped and mapped != code[-1]:
            code += mapped

    # Pad or truncate to 4 characters
    code = (code + "000")[:4]
    return code


# ──────────────────────────────────────────────
# Metaphone (simplified)
# ──────────────────────────────────────────────

def metaphone(name: str) -> str:
    """Simplified Metaphone algorithm."""
    name = name.upper().strip()
    if not name:
        return ""

    # Drop duplicate adjacent letters (except C)
    result = name[0]
    for ch in name[1:]:
        if ch != result[-1] or ch == 'C':
            result += ch

    # Initial letter transformations
    if result[:2] in ("AE", "GN", "KN", "PN", "WR"):
        result = result[1:]
    elif result[0] == "A":
        if result[1] in "EI":
            result = result[1:]

    # Drop silent letters
    result = result.replace("GH", "H")
    result = result.replace("MB", "M")

    return result[:6]  # Metaphone is variable length, cap at 6


# ──────────────────────────────────────────────
# Custom Indian Name Phonetic Scheme
# ──────────────────────────────────────────────
# Maps Hindi/Devanagari sounds to English phonetic groups.
# Handles transliteration variants:
#   Rajesh / Raajesh / Rajesh / Rajeshkumar
#   Suresh / Sureš / Suresh Kumar
#   Amit / Ameet / Ameeth

# Hindi consonant groups → phonetic code
_HINDI_CONSONANT_GROUPS = {
    # Voiced stops
    'क': 'K', 'ख': 'KH', 'ग': 'G', 'घ': 'GH', 'ङ': 'NG',
    # Unvoiced stops
    'च': 'CH', 'छ': 'CHH', 'ज': 'J', 'झ': 'JH', 'ञ': 'NY',
    # Retroflex
    'ट': 'T', 'ठ': 'TH', 'ड': 'D', 'ढ': 'DH', 'ण': 'N',
    # Dental
    'त': 'T', 'थ': 'TH', 'द': 'D', 'ध': 'DH', 'न': 'N',
    # Nasals
    'प': 'P', 'फ': 'PH', 'ब': 'B', 'भ': 'BH', 'म': 'M',
    # Semivowels
    'य': 'Y', 'र': 'R', 'ल': 'L', 'व': 'V',
    # Sibilants / fricatives
    'श': 'SH', 'ष': 'SH', 'स': 'S', 'ह': 'H',
}

# Common Indian name spelling variants → canonical form
_INDIAN_NAME_VARIANTS = {
    # Kumar variants
    'kumar': 'KMR', 'kumr': 'KMR', 'kmaar': 'KMR',
    # Singh variants
    'singh': 'SNG', 'sinh': 'SNG', 'sing': 'SNG',
    # Devi variants
    'devi': 'DV', 'devii': 'DV', 'dvi': 'DV',
    # Sharma variants
    'sharma': 'SHRM', 'sharmma': 'SHRM',
    # Patel variants
    'patel': 'PTL', 'patll': 'PTL',
    # Reddy variants
    'reddy': 'RD', 'redi': 'RD', 'reddey': 'RD',
    # Nair variants
    'nair': 'NR', 'nayr': 'NR',
    # Gupta variants
    'gupta': 'GPT', 'gupata': 'GPT',
    # Verma variants
    'verma': 'VRM', 'varma': 'VRM', 'vermna': 'VRM',
    # Mishra variants
    'mishra': 'MSHR', 'mishhra': 'MSHR',
    # Common first name variants
    'rajesh': 'RJSH', 'rajsh': 'RJSH', 'rajeshkumar': 'RJSHKMR',
    'suresh': 'SRSH', 'sures': 'SRSH',
    'rakesh': 'RKSH', 'raksh': 'RKSH',
    'mahesh': 'MHSH', 'mahhesh': 'MHSH',
    'dinesh': 'DNSH', 'dines': 'DNSH',
    'umesh': 'UMSH', 'umes': 'UMSH',
    'ramesh': 'RMSH', 'rames': 'RMSH',
    'mukesh': 'MKSH', 'mukes': 'MKSH',
    'akash': 'AKSH', 'aakash': 'AKSH',
    'rishabh': 'RSHB', 'rishav': 'RSHB',
    'aditya': 'ADTY', 'adiitya': 'ADTY',
    'deepak': 'DPK', 'depak': 'DPK',
    'manoj': 'MNJ', 'manohar': 'MNHR',
    'anil': 'ANL', 'aneel': 'ANL',
    'vinod': 'VND', 'vinod': 'VND',
    'sanjay': 'SNJY', 'sanjey': 'SNJY',
    'vijay': 'VJY', 'vijey': 'VJY',
    'ajay': 'AJY', 'adjay': 'AJY',
    'pradeep': 'PRDP', 'pradip': 'PRDP',
    'sunil': 'SNL', 'suneel': 'SNL',
    'pankaj': 'PNKJ', 'pankaz': 'PNKJ',
    'ashok': 'ASHK', 'asjhok': 'ASHK',
    'surendra': 'SRND', 'surender': 'SRND',
    'devendra': 'DVND', 'devender': 'DVND',
    'girish': 'GRSH', 'gireesh': 'GRSH',
    'satish': 'STSH', 'sateesh': 'STSH',
    'raj': 'RJ', 'raaj': 'RJ',
    'amit': 'AMT', 'ameet': 'AMT',
    'sumit': 'SMT', 'sumeet': 'SMT',
    'rohit': 'RHT', 'roheat': 'RHT',
    'mohit': 'MHT', 'moheet': 'MHT',
    'ankit': 'NKT', 'ankeet': 'NKT',
}


def _transliterate_hindi_char(ch: str) -> str:
    """Transliterate a single Devanagari character to phonetic code."""
    return _HINDI_CONSONANT_GROUPS.get(ch, ch)


def indian_phonetic(name: str) -> str:
    """
    Custom Indian name phonetic encoding.
    Handles transliteration variants, common suffixes, and Hindi→English mapping.

    Examples:
        "Rajesh" → "RJSH"
        "Raajesh" → "RJSH"
        "Rajesh Kumar" → "RJSHKMR"
        "राकेश" → "RKSH" (Hindi)
    """
    name = name.lower().strip()
    if not name:
        return ""

    # Check known variants first (fast path)
    name_no_space = name.replace(" ", "")
    if name_no_space in _INDIAN_NAME_VARIANTS:
        return _INDIAN_NAME_VARIANTS[name_no_space]

    # Transliterate Hindi characters if present
    has_hindi = any('\u0900' <= ch <= '\u097F' for ch in name)
    if has_hindi:
        transliterated = ""
        for ch in name:
            if '\u0900' <= ch <= '\u097F':
                transliterated += _transliterate_hindi_char(ch)
            else:
                transliterated += ch
        name = transliterated.lower()

    # Remove common Indian suffixes/prefixes that don't affect identity
    suffixes_to_remove = ['ji', 'sahab', 'sahib', 'bhai', 'bai', 'ben',
                          'sa', 'shri', 'smt', 'dr', 'insp', 'si']
    for suffix in suffixes_to_remove:
        if name.endswith(suffix) and len(name) > len(suffix) + 2:
            name = name[:-len(suffix)].strip()

    # Remove doubled letters (Rajesh → RAJESH → RJSH)
    result = name[0].upper() if name else ""
    for ch in name[1:]:
        if ch.isalpha() and ch.upper() != result[-1]:
            result += ch.upper()

    # Collapse consecutive identical consonants
    collapsed = result[0] if result else ""
    for ch in result[1:]:
        if ch != collapsed[-1]:
            collapsed += ch

    return collapsed[:8]  # Cap length


# ──────────────────────────────────────────────
# Combined Phonetic Score
# ──────────────────────────────────────────────

def phonetic_similarity(name1: str, name2: str) -> float:
    """
    Multi-algorithm phonetic similarity score.
    Returns 0.0 (no match) to 1.0 (identical phonetic encoding).

    Runs Soundex + Metaphone + Indian Phonetic and returns the best match.
    """
    if not name1 or not name2:
        return 0.0

    # Exact match
    if name1.lower() == name2.lower():
        return 1.0

    scores = []

    # Soundex
    sx1, sx2 = soundex(name1), soundex(name2)
    if sx1 and sx2 and sx1 == sx2:
        scores.append(0.85)

    # Metaphone
    mp1, mp2 = metaphone(name1), metaphone(name2)
    if mp1 and mp2 and mp1 == mp2:
        scores.append(0.90)

    # Indian phonetic
    ip1, ip2 = indian_phonetic(name1), indian_phonetic(name2)
    if ip1 and ip2:
        if ip1 == ip2:
            scores.append(0.95)
        elif ip1.startswith(ip2) or ip2.startswith(ip1):
            # One is prefix of the other (e.g., "Raj" vs "Rajesh")
            shorter = min(len(ip1), len(ip2))
            longer = max(len(ip1), len(ip2))
            scores.append(0.70 + 0.20 * (shorter / longer))

    return max(scores) if scores else 0.0


def phonetic_candidates(
    target_name: str,
    candidate_names: List[Tuple[str, str]],  # [(name, entity_id), ...],
    threshold: float = 0.60,
) -> List[Tuple[str, str, float]]:
    """
    Find phonetically similar candidates for a target name.
    Returns: [(entity_id, name, score), ...] sorted by score descending.
    """
    results = []
    for name, eid in candidate_names:
        score = phonetic_similarity(target_name, name)
        if score >= threshold:
            results.append((eid, name, score))

    results.sort(key=lambda x: x[2], reverse=True)
    return results
