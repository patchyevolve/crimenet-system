"""
Group Finder — find candidate groups for LLM evaluation.
Uses multiple similarity signals.
"""

from typing import List, Dict, Set, Tuple
from ..models.schema import MergeCandidate
from .rule_pass import normalize_name, normalize_phone, get_entity_phones, person_names_compatible


def get_first_name(name: str) -> str:
    """Extract first name from a full name."""
    norm = normalize_name(name, "PERSON")
    parts = norm.split()
    return parts[0] if parts else norm


def is_partial_name_match(name1: str, name2: str) -> bool:
    """Check if one name is a partial of the other."""
    n1 = normalize_name(name1, "PERSON")
    n2 = normalize_name(name2, "PERSON")
    if n1 == n2:
        return False  # Exact match, handled by rule_pass
    # Check if shorter name is contained in longer name
    shorter, longer = (n1, n2) if len(n1) <= len(n2) else (n2, n1)
    if len(shorter) < 2:
        return False
    # Check if shorter is a prefix of longer (first name match)
    if longer.startswith(shorter):
        return True
    # Check if shorter appears as a word in longer
    if f" {shorter} " in f" {longer} ":
        return True
    return False


def find_groups(entities: List[dict], rule_candidates: List[MergeCandidate]) -> List[MergeCandidate]:
    """
    Find groups of entities that might be the same real-world entity.
    Skips pairs already covered by rule_pass, but still includes those entities
    in group finding so partial matches with OTHER entities are not missed.
    """
    # Pairs already covered by rule_pass (skip these)
    covered_pairs: Set[Tuple[str, str]] = set()
    for rc in rule_candidates:
        pair = tuple(sorted(rc.entity_ids))
        covered_pairs.add(pair)

    # Consider ALL PERSON entities (don't exclude rule_pass-covered ones)
    person_entities = [
        e for e in entities
        if e.get("entity_type") == "PERSON"
    ]

    candidates = []
    seen_pairs: Set[Tuple[str, str]] = set()

    # Signal 1: First-name match
    first_name_map: Dict[str, List[dict]] = {}
    for e in person_entities:
        fn = get_first_name(e.get("name", ""))
        if fn and len(fn) > 1:
            if fn not in first_name_map:
                first_name_map[fn] = []
            first_name_map[fn].append(e)

    for fn, group in first_name_map.items():
        if len(group) > 1:
            # Create pairwise candidates — only when names are compatible
            for i in range(len(group)):
                for j in range(i + 1, len(group)):
                    pair = tuple(sorted([group[i]["id"], group[j]["id"]]))
                    if pair in seen_pairs or pair in covered_pairs:
                        continue
                    if not person_names_compatible(group[i].get("name", ""), group[j].get("name", "")):
                        continue
                    seen_pairs.add(pair)
                    candidates.append(MergeCandidate(
                        entity_ids=[group[i]["id"], group[j]["id"]],
                        signal="first_name_match",
                        description=f"Same first name: {fn}",
                        confidence=0.85,
                        entities=[group[i], group[j]],
                    ))

    # Signal 2: Partial name containment (PERSON only — must be name-compatible)
    for i in range(len(person_entities)):
        for j in range(i + 1, len(person_entities)):
            e1, e2 = person_entities[i], person_entities[j]
            pair = tuple(sorted([e1["id"], e2["id"]]))
            if pair in seen_pairs or pair in covered_pairs:
                continue
            if not person_names_compatible(e1.get("name", ""), e2.get("name", "")):
                continue
            if is_partial_name_match(e1.get("name", ""), e2.get("name", "")):
                seen_pairs.add(pair)
                candidates.append(MergeCandidate(
                    entity_ids=[e1["id"], e2["id"]],
                    signal="partial_name_match",
                    description=f"'{e1['name']}' ~ '{e2['name']}'",
                    confidence=0.8,
                    entities=[e1, e2],
                ))

    # Signal 3: Shared phone (across different entity types too)
    phone_map: Dict[str, List[dict]] = {}
    for e in entities:
        for phone in get_entity_phones(e):
            if phone not in phone_map:
                phone_map[phone] = []
            phone_map[phone].append(e)

    for phone, group in phone_map.items():
        # Only group PERSON entities with shared phones
        person_group = [e for e in group if e.get("entity_type") == "PERSON"]
        if len(person_group) > 1:
            for i in range(len(person_group)):
                for j in range(i + 1, len(person_group)):
                    pair = tuple(sorted([person_group[i]["id"], person_group[j]["id"]]))
                    if pair in seen_pairs or pair in covered_pairs:
                        continue
                    # Shared phone alone does NOT prove identity (family
                    # phones, shared devices) — same rule as rule_pass's
                    # exact_phone_match: names must also be compatible
                    # ("Bhai Sahab" and "Boss" both own 9876543220, but a
                    # shared number never makes them the same person).
                    if not person_names_compatible(person_group[i].get("name", ""),
                                                   person_group[j].get("name", "")):
                        continue
                    seen_pairs.add(pair)
                    candidates.append(MergeCandidate(
                        entity_ids=[person_group[i]["id"], person_group[j]["id"]],
                        signal="shared_phone",
                            description=f"Shared phone: {phone}",
                            confidence=0.9,
                            entities=[person_group[i], person_group[j]],
                        ))

    return candidates


def deduplicate_candidates(candidates: List[MergeCandidate]) -> List[MergeCandidate]:
    """Remove duplicate candidates (same entity pair)."""
    seen: Set[Tuple[str, ...]] = set()
    unique = []
    for c in candidates:
        key = tuple(sorted(c.entity_ids))
        if key not in seen:
            seen.add(key)
            unique.append(c)
    return unique
