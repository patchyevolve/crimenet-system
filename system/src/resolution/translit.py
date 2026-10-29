"""
Devanagari → Latin transliteration.

Script conversion (orthographic, not a lexicon): maps Devanagari code
points to their standard Latin renderings so cross-script name variants
resolve to the same normalized form ("मीना देवी" ≈ "Meena Devi",
"नेहरू नगर" ≈ "Nehru Nagar").

Rules:
- A consonant followed by a vowel matra renders consonant + matra vowel.
- A consonant followed by another consonant (no matra) gets the inherent
  vowel 'a' (Devanagari inherent vowel between consonants).
- A consonant followed by virama (्) renders bare consonant (dead form).
- Word-final consonant drops the inherent vowel (schwa deletion, matching
  common English renderings: कुमार → "kumar", राजेश → "rajesh").
- Standalone vowels render as their vowel letters.
- Non-Devanagari characters pass through unchanged.
"""

# Vowel signs (matras)
_MATRAS = {
    "ा": "a",
    "ि": "i",
    "ी": "ee",
    "ु": "u",
    "ू": "u",      # long u: नेहरू → "neharu" (1 edit from "nehru", exact-typo rule)
    "ृ": "ri",
    "ॄ": "ri",
    "े": "e",
    "ै": "ai",
    "ो": "o",
    "ौ": "au",
    "ॉ": "o",
    "ॆ": "e",
    "ॊ": "o",
}

# Independent vowels
_VOWELS = {
    "अ": "a",
    "आ": "aa",
    "इ": "i",
    "ई": "ee",
    "उ": "u",
    "ऊ": "oo",
    "ऋ": "ri",
    "ॠ": "ri",
    "ऌ": "li",
    "ए": "e",
    "ऐ": "ai",
    "ओ": "o",
    "औ": "au",
    "ऑ": "o",
    "ऍ": "e",
    "ॲ": "e",
}

# Consonants (base + precomposed nukta forms as single code points)
_CONSONANTS = {
    "क": "k", "ख": "kh", "ग": "g", "घ": "gh", "ङ": "ng",
    "च": "ch", "छ": "chh", "ज": "j", "झ": "jh", "ञ": "ny",
    "ट": "t", "ठ": "th", "ड": "d", "ढ": "dh", "ण": "n",
    "त": "t", "थ": "th", "द": "d", "ध": "dh", "न": "n",
    "प": "p", "फ": "ph", "ब": "b", "भ": "bh", "म": "m",
    "य": "y", "र": "r", "ल": "l", "व": "v",
    "श": "sh", "ष": "sh", "स": "s", "ह": "h",
    "ळ": "l", "ऱ": "r",
    # Precomposed nukta letters (U+0958–U+095F)
    "क़": "k", "ख़": "kh", "ग़": "gh", "ज़": "z",
    "ड़": "r", "ढ़": "rh", "फ़": "f", "य़": "y",
    "\u0958": "k", "\u0959": "kh", "\u095a": "gh", "\u095b": "z",
    "\u095c": "r", "\u095d": "rh", "\u095e": "f", "\u095f": "y",
}

# Nukta combining mark (U+093C): remaps the consonant it follows
_NUKTA_REMAP = {"ph": "f", "j": "z", "d": "r", "dh": "rh", "y": "y",
                "k": "k", "kh": "kh", "g": "gh"}

# Signs carrying no phonetic content in this transliteration
_INERT = {"ँ", "ं", "ः", "ऺ", "ऻ", "ऽ", "्", "ॐ", "़", "॒", "॑"}

_DEVANAGARI_START = 0x0900
_DEVANAGARI_END = 0x097F


def _is_devanagari(ch: str) -> bool:
    cp = ord(ch)
    return _DEVANAGARI_START <= cp <= _DEVANAGARI_END


def transliterate_devanagari(text: str) -> str:
    """Convert Devanagari script text to Latin. Non-Devanagari unchanged."""
    if not text or not any(_is_devanagari(c) for c in text):
        return text

    out = []
    i = 0
    n = len(text)
    while i < n:
        ch = text[i]

        if ch in ("ं", "ँ"):  # anusvara / candrabindu → nasal 'n'
            out.append("n")
            i += 1
            continue

        if ch in _INERT:
            i += 1
            continue

        # Stray matra not directly after a consonant — no vowel to modify
        if ch in _MATRAS:
            i += 1
            continue

        if ch in _VOWELS:
            out.append(_VOWELS[ch])
            i += 1
            continue

        if ch in _CONSONANTS:
            out.append(_CONSONANTS[ch])
            i += 1
            # Nukta (combining) may sit between consonant and matra
            while i < n and text[i] == "़":
                remapped = _NUKTA_REMAP.get(out[-1])
                if remapped:
                    out[-1] = remapped
                i += 1
            # Inspect what follows: matra / virama / consonant / other
            if i < n:
                nxt = text[i]
                if nxt in _MATRAS:
                    out.append(_MATRAS[nxt])
                    i += 1
                elif nxt == "्":  # virama — dead consonant, no vowel
                    i += 1
                elif _is_devanagari(nxt) and nxt in _CONSONANTS:
                    out.append("a")  # inherent vowel between consonants
                # word-final or followed by non-Devanagari → drop inherent 'a'
            continue

        out.append(ch)
        i += 1

    return "".join(out)
