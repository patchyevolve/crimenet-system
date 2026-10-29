"""
Rule Pass — fast exact matching for obvious merges.
Also includes Jaro-Winkler fuzzy matching for near-duplicates.
No LLM needed for rule-based matches.
"""

import re
from typing import List, Dict, Optional, Tuple
from ..models.schema import MergeCandidate
from .translit import transliterate_devanagari
from .fuzzy_index import FuzzyIndex


# Honorifics/prefixes (word-initial)
_HONORIFIC_PREFIXES = {
    "si", "si.", "insp", "insp.", "inspector", "shri", "shri.", "smt", "smt.",
    "mr", "mr.", "mrs", "mrs.", "ms", "ms.", "dr", "dr.", "pt", "pt.",
    "const", "const.", "head", "ps", "ssp", "dcp", "acp",
}
# Trailing honorifics / kinship / courtesy suffixes common in Indian names
_HONORIFIC_SUFFIXES = {
    "ji", "ji.", "sahib", "sahib.", "babu", "bai", "begum", "khan",
    "bhai", "bhau", "saheb", "sah", "singh", "kumari", "kum.",
    "devi", "das", "lal", "prasad", "sharma?",  # keep sharma as surname; only strip if sole token handled below
}
# Tokens that are never a core name when trailing
_COURTESY_TOKENS = {"ji", "sahib", "babu", "bhai", "bhau", "saheb", "bai", "begum"}


def normalize_name(name: str, entity_type: str = "") -> str:
    """Normalize a name for comparison.

    Strips honorific prefixes, trailing courtesy tokens, possessives, and
    punctuation so 'Meena Devi', 'Meena Ji', "Meena Devi's" → 'meena',
    while preserving real surnames like 'Sharma' in 'Amit Sharma'.
    Cross-script variants normalize to the same form (Devanagari is
    transliterated first: "मीना देवी" ≈ "Meena Devi").

    Synthetic id-style suffixes are dropped for PERSON names only
    ('suresh_kumar_01' → 'suresh kumar'): they are identity handles, not
    part of the real name. Other entity types keep trailing digits —
    they are meaningful distinctions ('Karol Bagh Tower 1' is a different
    tower from 'Karol Bagh Tower 3', '12 March 2024' ≠ '12 March').
    entity_type="" (unknown) takes the conservative no-strip path.
    """
    name = transliterate_devanagari(name or "")
    name = name.lower().strip()
    # Identifier-style names: underscores are separators, not word chars
    name = name.replace("_", " ")
    # Possessive / punctuation cleanup
    name = name.replace("'s", " ").replace("'s ", " ")
    name = re.sub(r"[^\w\s]", " ", name)
    name = re.sub(r"\s+", " ", name).strip()
    tokens = name.split()
    # Strip honorific prefixes
    while tokens and tokens[0] in _HONORIFIC_PREFIXES:
        tokens.pop(0)
    # Strip trailing digit-only tokens (id/account suffixes: 'suresh_kumar_01',
    # 'meena_devi_2024'). PERSON only — for other types trailing digits are
    # signal ('Tower 1' vs 'Tower 3'). Requires >=3 tokens so bare numeric
    # names ('Rajesh 01' style handles aside) stay intact.
    if entity_type == "PERSON":
        while len(tokens) >= 3 and tokens[-1].isdigit():
            tokens.pop()
    # Strip trailing courtesy tokens (ji, bhai, sahib, ...) repeatedly.
    # Keep >=1 token: a courtesy-only name ("Bhai", "Ji") must not collapse
    # to "" — empty norms are indistinguishable across names and bypass the
    # exact rule's len>1 guard. Mirrors the devi/kumari rule below.
    while len(tokens) > 1 and tokens[-1] in _COURTESY_TOKENS:
        tokens.pop()
    # Strip trailing 'devi'/'kumari' only when there is a preceding given name
    # (keeps 'Devi' as a standalone name if it's the only token)
    while len(tokens) > 1 and tokens[-1] in {"devi", "kumari", "kum"}:
        tokens.pop()
    return " ".join(tokens).strip()


def name_tokens(name: str, entity_type: str = "") -> List[str]:
    """Normalized token list for a name."""
    norm = normalize_name(name, entity_type)
    return norm.split() if norm else []


def first_name(name: str) -> str:
    """First core token of a normalized name."""
    toks = name_tokens(name)
    return toks[0] if toks else ""


def last_name(name: str) -> str:
    """Last core token of a normalized name (surname when multi-token)."""
    toks = name_tokens(name)
    return toks[-1] if toks else ""


def reconcile_entity_types(
    entities: List[dict],
    relations: Optional[List[dict]] = None,
) -> List[dict]:
    """Settle type disagreements between entities with the same name.

    The NER model is inconsistent across spans and files: the same string
    can come back PERSON in one place and ORGANIZATION/LOCATION in another
    ("Suresh Kumar", "Meena Devi's"). Cross-type duplicates never merge
    downstream (union-find excludes cross-type verdicts), so they linger as
    separate entities.

    Resolution is evidence-based, not lexicon-based:
      1. PERSON wins any exact-name conflict — a bare name duplicated as
         ORG/LOC is NER noise; real orgs/places carry distinguishing words
         and don't exactly equal a bare person name.
      2. Otherwise the type with more relation endpoints wins (the corpus
         itself says what the entity participates in).
      3. Ties keep the existing type.

    Only entity_type is rewritten — IDs stay stable so relations, id_map
    and provenance keep pointing at the same records.
    """
    if not entities:
        return entities

    def _key(name: str) -> str:
        n = re.sub(r"\s+", " ", (name or "").strip().lower())
        return re.sub(r"'s$", "", n).strip()

    groups: Dict[str, List[dict]] = {}
    for e in entities:
        if not isinstance(e, dict):
            continue
        k = _key(e.get("name", ""))
        if k:
            groups.setdefault(k, []).append(e)

    rel_count: Dict[str, int] = {}
    for r in relations or []:
        if not isinstance(r, dict):
            continue
        for sid in (
            r.get("source_entity_id") or r.get("source_id"),
            r.get("target_entity_id") or r.get("target_id"),
        ):
            if sid:
                rel_count[sid] = rel_count.get(sid, 0) + 1

    # Corpus evidence: an ORGANIZATION whose tokens are all covered by a
    # single known LOCATION name is a place, not a company
    # ("Noida" ⊂ "Noida Phase 2"). Real orgs carry at least one token no
    # location has ("Delhi Herald" — "herald" appears in no place name).
    loc_token_sets = [
        set(_key(e.get("name", "")).split())
        for e in entities
        if isinstance(e, dict) and e.get("entity_type") == "LOCATION"
    ]
    for e in entities:
        if not isinstance(e, dict) or e.get("entity_type") != "ORGANIZATION":
            continue
        toks = set(_key(e.get("name", "")).split())
        if toks and any(toks <= lt for lt in loc_token_sets):
            e["entity_type"] = "LOCATION"

    for members in groups.values():
        types = {m.get("entity_type", "") for m in members}
        if len(types) < 2:
            continue
        if "PERSON" in types:
            target = "PERSON"
        else:
            type_votes: Dict[str, int] = {}
            for m in members:
                t = m.get("entity_type", "")
                type_votes[t] = type_votes.get(t, 0) + 1 + rel_count.get(m.get("id", ""), 0)
            target = max(type_votes, key=lambda t: type_votes[t])
        for m in members:
            if m.get("entity_type") != target:
                m["entity_type"] = target

    return entities


def person_names_compatible(name1: str, name2: str) -> bool:
    """Guard against cross-person fuzzy merges (Rajesh≠Suresh≠Rakesh).

    Compatible when:
    - Exact normalized match, OR
    - One name's tokens are a subset of the other's (Amit ⊂ Amit Sharma), OR
    - Same first name AND (single-token on either side OR surnames also similar),
      OR
    - Same surname AND first names are very similar (JW >= 0.95).

    Incompatible: shared surname with different first names (Rajesh Kumar vs
    Suresh Kumar), or same first name with clearly different surnames
    (Amit Sharma vs Amit Kumar → review, not auto-merge).
    """
    t1, t2 = name_tokens(name1, "PERSON"), name_tokens(name2, "PERSON")
    if not t1 or not t2:
        return False
    if t1 == t2:
        return True
    # Token subset (Amit ⊂ Amit Sharma)
    if set(t1) <= set(t2) or set(t2) <= set(t1):
        return True
    f1, f2 = t1[0], t2[0]
    # Same first name: allow if either is single-token (partial name) OR surnames close
    if f1 == f2:
        if len(t1) == 1 or len(t2) == 1:
            return True
        # Multi-token both: require surname similarity (Amit Sharma vs Amit Verma → no)
        return jaro_winkler(t1[-1], t2[-1]) >= 0.75
    # Shared surname but different first names → only if first names are
    # extremely close (typo-level). rajesh vs rakesh JW≈0.91 must NOT merge;
    # threshold 0.95 blocks distinct given names while allowing clear typos.
    if len(t1) > 1 and len(t2) > 1 and t1[-1] == t2[-1]:
        return jaro_winkler(f1, f2) >= 0.95
    # Different structure: require very strong first-token similarity
    # (blocks Rajesh↔Rakesh at JW≈0.91 while allowing clear typos)
    return jaro_winkler(f1, f2) >= 0.93


def normalize_phone(phone: str) -> str:
    """Normalize a phone number to 10 digits."""
    digits = re.sub(r'\D', '', phone)
    # Strip country code if present
    if len(digits) == 12 and digits.startswith('91'):
        digits = digits[2:]
    if len(digits) == 11 and digits.startswith('0'):
        digits = digits[1:]
    return digits


def get_entity_phones(entity: dict) -> List[str]:
    """Extract all phone numbers from an entity."""
    phones = set()
    # Direct phone field
    phone = entity.get("attributes", {}).get("phone_number", "")
    if phone:
        phones.add(normalize_phone(phone))
    # account_holder phone (from CDR)
    phone = entity.get("attributes", {}).get("phone", "")
    if phone:
        phones.add(normalize_phone(phone))
    return [p for p in phones if len(p) == 10]


def get_entity_accounts(entity: dict) -> List[str]:
    """Extract all account numbers from an entity."""
    accounts = set()
    acc = entity.get("attributes", {}).get("account_number", "")
    if acc:
        accounts.add(acc.strip())
    return list(accounts)


def jaro_winkler(s1: str, s2: str, p: float = 0.1) -> float:
    """
    Compute Jaro-Winkler similarity between two strings.
    p = scaling factor for common prefix (default 0.1).
    Returns value between 0.0 (no match) and 1.0 (exact match).
    """
    if not s1 or not s2:
        return 0.0
    if s1 == s2:
        return 1.0

    len1, len2 = len(s1), len(s2)
    if len1 == 0 or len2 == 0:
        return 0.0

    # Maximum distance for matching
    match_distance = max(len1, len2) // 2 - 1
    if match_distance < 0:
        match_distance = 0

    s1_matches = [False] * len1
    s2_matches = [False] * len2

    matches = 0
    transpositions = 0

    # Find matches
    for i in range(len1):
        start = max(0, i - match_distance)
        end = min(i + match_distance + 1, len2)
        for j in range(start, end):
            if s2_matches[j] or s1[i] != s2[j]:
                continue
            s1_matches[i] = True
            s2_matches[j] = True
            matches += 1
            break

    if matches == 0:
        return 0.0

    # Count transpositions
    k = 0
    for i in range(len1):
        if not s1_matches[i]:
            continue
        while k < len2 and not s2_matches[k]:
            k += 1
        if k >= len2:
            break
        if s1[i] != s2[k]:
            transpositions += 1
        k += 1

    jaro = (matches / len1 + matches / len2 + (matches - transpositions / 2) / matches) / 3

    # Common prefix (up to 4 characters)
    prefix_len = 0
    for i in range(min(4, len1, len2)):
        if s1[i] == s2[i]:
            prefix_len += 1
        else:
            break

    return jaro + prefix_len * p * (1 - jaro)


def fuzzy_name_match(name1: str, name2: str, threshold: float = 0.85) -> Tuple[bool, float]:
    """
    Check if two names are similar enough using Jaro-Winkler.
    Applies person_names_compatible guard for PERSON-like names.
    Returns (is_match, similarity_score).
    """
    n1 = normalize_name(name1, "PERSON")
    n2 = normalize_name(name2, "PERSON")

    if not n1 or not n2:
        return False, 0.0

    # Exact match
    if n1 == n2:
        return True, 1.0

    # Cross-person guard: block Rajesh/Suresh/Rakesh-style false merges
    # even when substring/JW would otherwise pass.
    if not person_names_compatible(name1, name2):
        # Still return the raw similarity for logging, but not a match
        return False, jaro_winkler(n1, n2)

    # One contains the other
    if n1 in n2 or n2 in n1:
        shorter = min(n1, n2, key=len)
        longer = max(n1, n2, key=len)
        ratio = len(shorter) / len(longer)
        if ratio >= 0.6:
            return True, 0.85 + ratio * 0.15
        # Short given name fully contained (Amit in Amit Sharma) with token subset
        if set(n1.split()) <= set(n2.split()) or set(n2.split()) <= set(n1.split()):
            return True, 0.90

    # Jaro-Winkler similarity
    similarity = jaro_winkler(n1, n2)
    return similarity >= threshold, similarity


def rule_pass(entities: List[dict]) -> List[MergeCandidate]:
    """
    Find obvious merges using exact matching and Jaro-Winkler fuzzy matching.
    Returns merge candidates that can be auto-merged.
    """
    candidates = []

    # Group entities by type — merge duplicates across ALL types
    entities_by_type: Dict[str, List[dict]] = {}
    for e in entities:
        etype = e.get("entity_type", "UNKNOWN")
        if etype not in entities_by_type:
            entities_by_type[etype] = []
        entities_by_type[etype].append(e)

    # --- Exact name match across ALL entity types (dedup) ---
    for etype, type_entities in entities_by_type.items():
        name_map: Dict[str, List[dict]] = {}
        for e in type_entities:
            norm = normalize_name(e.get("name", ""), etype)
            if norm and len(norm) > 1:
                if norm not in name_map:
                    name_map[norm] = []
                name_map[norm].append(e)

        for norm_name, group in name_map.items():
            if len(group) > 1:
                candidates.append(MergeCandidate(
                    entity_ids=[e["id"] for e in group],
                    signal="exact_name_match",
                    description=f"Same {etype} name: {group[0]['name']}",
                    confidence=0.95,
                    entities=group,
                ))

    # --- Near-identical name (single-token typo / transliteration artifact) ---
    # Same non-PERSON type, same shape (token count), all tokens identical
    # except one differing by a single edit ("neharu nagar" ≈ "nehru nagar").
    # Numeric tokens never count as the difference ("Camera 5" vs "Camera 7"
    # are distinct cameras). PERSON keeps its own guarded compatibility paths.
    for etype in ("LOCATION", "ORGANIZATION"):
        type_entities = entities_by_type.get(etype, [])
        for i, e1 in enumerate(type_entities):
            t1 = name_tokens(e1.get("name", ""), etype)
            if not t1:
                continue
            for e2 in type_entities[i + 1:]:
                t2 = name_tokens(e2.get("name", ""), etype)
                if not t2 or len(t1) != len(t2):
                    continue
                diffs = [(a, b) for a, b in zip(t1, t2) if a != b]
                if len(diffs) != 1:
                    continue
                a, b = diffs[0]
                if not (a.isalpha() and b.isalpha()):
                    continue
                if FuzzyIndex._edit_distance(a, b) > 1:
                    continue
                candidates.append(MergeCandidate(
                    entity_ids=[e1["id"], e2["id"]],
                    signal="near_identical_name",
                    description=f"Typo-level {etype} match: {e1.get('name', '')} ≈ {e2.get('name', '')}",
                    confidence=0.95,
                    entities=[e1, e2],
                ))

    # --- ORG structural matches (acronym / suffix expansion) ---
    # Evidence-based shape rules over the corpus's own strings, no lexicon:
    #   initials: single-token short form == first letters of a >=3-token
    #     expansion ("pnb" ↔ "punjab national bank", "bob" ↔ "bank of baroda")
    #   suffix expansion: shorter tokens are a strict prefix of the longer
    #     and the remaining tokens are organisation suffixes
    #     ("hdfc" ↔ "hdfc bank"). Non-suffix additions block the match
    #     ("sbi" vs "sbi customer care" stays split).
    org_entities = entities_by_type.get("ORGANIZATION", [])
    _ORG_SUFFIX_TOKENS = {
        "bank", "banks", "ltd", "limited", "inc", "corp", "corporation",
        "company", "group", "international",
    }
    for i, e1 in enumerate(org_entities):
        t1 = name_tokens(e1.get("name", ""), "ORGANIZATION")
        if not t1:
            continue
        for e2 in org_entities[i + 1:]:
            t2 = name_tokens(e2.get("name", ""), "ORGANIZATION")
            if not t2 or t1 == t2:
                continue
            matched = False
            # Initials of a >=3-token expansion
            short, long_ = (t1, t2) if len(t1) == 1 and len(t2) >= 3 else \
                           ((t2, t1) if len(t2) == 1 and len(t1) >= 3 else (None, None))
            if short is not None:
                initials = "".join(tok[0] for tok in long_ if tok)
                if initials == short[0]:
                    matched = True
            # Strict token prefix + org-suffix remainder
            if not matched:
                if len(t1) < len(t2) and t1 == t2[:len(t1)] \
                        and set(t2[len(t1):]) <= _ORG_SUFFIX_TOKENS:
                    matched = True
                elif len(t2) < len(t1) and t2 == t1[:len(t2)] \
                        and set(t1[len(t2):]) <= _ORG_SUFFIX_TOKENS:
                    matched = True
            if matched:
                candidates.append(MergeCandidate(
                    entity_ids=[e1["id"], e2["id"]],
                    signal="org_structural_match",
                    description=f"ORG acronym/expansion match: {e1.get('name', '')} ↔ {e2.get('name', '')}",
                    confidence=0.95,
                    entities=[e1, e2],
                ))

    # --- Exact phone match (PERSON only) ---
    # Shared phone alone does NOT prove identity (family phones, shared devices).
    # Only merge when names are also compatible; otherwise it's a co-occurrence edge.
    person_entities = entities_by_type.get("PERSON", [])
    phone_map: Dict[str, List[dict]] = {}
    for e in person_entities:
        for phone in get_entity_phones(e):
            if phone not in phone_map:
                phone_map[phone] = []
            phone_map[phone].append(e)

    for phone, group in phone_map.items():
        if len(group) > 1:
            # Require name compatibility within the group
            compatible = []
            for e in group:
                if all(person_names_compatible(e.get("name", ""), o.get("name", ""))
                       for o in group if o is not e):
                    compatible.append(e)
            if len(compatible) > 1:
                candidates.append(MergeCandidate(
                    entity_ids=[e["id"] for e in compatible],
                    signal="exact_phone_match",
                    description=f"Same phone + compatible names: {phone}",
                    confidence=0.95,
                    entities=compatible,
                ))

    # --- Exact account match ---
    account_map: Dict[str, List[dict]] = {}
    for e in person_entities:
        for acc in get_entity_accounts(e):
            if acc not in account_map:
                account_map[acc] = []
            account_map[acc].append(e)

    for acc, group in account_map.items():
        if len(group) > 1:
            compatible = []
            for e in group:
                if all(person_names_compatible(e.get("name", ""), o.get("name", ""))
                       for o in group if o is not e):
                    compatible.append(e)
            if len(compatible) > 1:
                candidates.append(MergeCandidate(
                    entity_ids=[e["id"] for e in compatible],
                    signal="exact_account_match",
                    description=f"Same account + compatible names: {acc}",
                    confidence=0.95,
                    entities=compatible,
                ))

    # --- Jaro-Winkler fuzzy name match (PERSON only) ---
    # Only check pairs not already matched by exact rules
    matched_ids = set()
    for c in candidates:
        matched_ids.update(c.entity_ids)

    fuzzy_candidates = []
    for i, e1 in enumerate(person_entities):
        if e1["id"] in matched_ids:
            continue
        for e2 in person_entities[i+1:]:
            if e2["id"] in matched_ids:
                continue
            is_match, similarity = fuzzy_name_match(e1.get("name", ""), e2.get("name", ""))
            if is_match and similarity < 1.0:  # Exact matches already handled
                # Extra safety: never auto-propose cross-family merges
                if not person_names_compatible(e1.get("name", ""), e2.get("name", "")):
                    continue
                fuzzy_candidates.append(MergeCandidate(
                    entity_ids=[e1["id"], e2["id"]],
                    signal="fuzzy_name_match",
                    description=f"Similar names: {e1.get('name', '')} ≈ {e2.get('name', '')} (sim={similarity:.2f})",
                    confidence=similarity,
                    entities=[e1, e2],
                ))

    # Add top fuzzy matches (limit to avoid explosion)
    candidates.extend(fuzzy_candidates[:20])

    return candidates
