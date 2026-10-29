"""
Entity Merger — create ResolvedEntities from validated merges.
Also handles UnknownEntities and Contradictions.
Implements identity uncertainty propagation and claim-level dependency.
"""

from typing import List, Dict, Tuple, Set
from datetime import datetime
from ..models.schema import (
    LLMVerdict, ResolvedEntity, UnknownEntity, Contradiction,
    ContradictionType, ProvenanceChain, ResolutionHistory,
    generate_id,
)


def _names_conflicting(names: List[str]) -> List[str]:
    """Subset of ``names`` that cannot be reconciled with every other name.

    Two names reconcile when they are aliases of each other: exact
    normalized match, containment subset ("Rakesh" ⊂ "Rakesh Kumar"),
    nickname/short form, or a fuzzy match clearing the name-matching rules
    (``person_names_compatible`` + Jaro-Winkler gate in ``fuzzy_name_match``).

    Returns ``[]`` when the whole set reconciles, otherwise the names that
    took part in at least one irreconcilable pair (order preserved, unique).
    """
    from .rule_pass import fuzzy_name_match, person_names_compatible

    conflicting = []
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            if person_names_compatible(a, b):
                continue
            if fuzzy_name_match(a, b)[0]:
                continue
            for name in (a, b):
                if name not in conflicting:
                    conflicting.append(name)
    return conflicting


def detect_contradictions(entities: List[dict], run_id: str = "") -> List[Contradiction]:
    """Detect all 8 contradiction types — per SYSTEM_STRUCTURE.md taxonomy."""
    contradictions = []
    entity_ids = [e["id"] for e in entities]

    # 1. IDENTITY: Same person, different identity
    #
    # Phones are deliberately NOT checked. A person holding several numbers
    # (personal/work/handset swapped out) is ordinary attribute aggregation —
    # flagging len(phones) > 1 produced a false IDENTITY contradiction on
    # virtually every multi-source merge.

    # Name conflicts (identity) — only when the names are irreconcilable.
    # Legitimate alias merges ("Rakesh" + "Rakesh Kumar") must not emit an
    # IDENTITY contradiction: Stage 8 would have to adjudicate it and
    # Stage 7 scored hypothesis confidence against it while unadjudicated.
    names = sorted({e.get("name", "") for e in entities if (e.get("name") or "").strip()})
    if len(names) > 1:
        conflicting = _names_conflicting(names)
        if conflicting:
            contradictions.append(Contradiction(
                id=generate_id("CON", f"name_{'_'.join(entity_ids)}"),
                type=ContradictionType.IDENTITY.value,
                entity_ids=entity_ids,
                attribute="name",
                values=conflicting,
                sources=[e.get("source", {}).get("file_name", "?") for e in entities],
                severity="medium",
                run_id=run_id,
            ))

    # DOB conflicts (identity)
    dobs = set()
    for e in entities:
        dob = e.get("attributes", {}).get("dob", "") or e.get("attributes", {}).get("date_of_birth", "")
        if dob:
            dobs.add(dob)
    if len(dobs) > 1:
        contradictions.append(Contradiction(
            id=generate_id("CON", f"dob_{'_'.join(entity_ids)}"),
            type=ContradictionType.IDENTITY.value,
            entity_ids=entity_ids,
            attribute="dob",
            values=list(dobs),
            sources=[e.get("source", {}).get("file_name", "?") for e in entities],
            severity="high",
            run_id=run_id,
        ))

    # 2. TEMPORAL: Conflicting timestamps
    timestamps = set()
    for e in entities:
        ts = e.get("attributes", {}).get("timestamp", "") or e.get("attributes", {}).get("call_time", "")
        if ts:
            timestamps.add(str(ts))
    if len(timestamps) > 1:
        contradictions.append(Contradiction(
            id=generate_id("CON", f"ts_{'_'.join(entity_ids)}"),
            type=ContradictionType.TEMPORAL.value,
            entity_ids=entity_ids,
            attribute="timestamp",
            values=list(timestamps),
            sources=[e.get("source", {}).get("file_name", "?") for e in entities],
            severity="medium",
            run_id=run_id,
        ))

    # 3. LOCATION: Conflicting locations
    addresses = set()
    for e in entities:
        addr = e.get("attributes", {}).get("address", "") or e.get("attributes", {}).get("location", "")
        if addr:
            addresses.add(addr)
    if len(addresses) > 1:
        contradictions.append(Contradiction(
            id=generate_id("CON", f"addr_{'_'.join(entity_ids)}"),
            type=ContradictionType.LOCATION.value,
            entity_ids=entity_ids,
            attribute="address",
            values=list(addresses),
            sources=[e.get("source", {}).get("file_name", "?") for e in entities],
            severity="medium",
            run_id=run_id,
        ))

    # 4. ATTRIBUTION: Different source claims different owner of same attribute
    accounts = set()
    for e in entities:
        acc = e.get("attributes", {}).get("account_number", "")
        if acc:
            accounts.add(acc)
    if len(accounts) > 1:
        contradictions.append(Contradiction(
            id=generate_id("CON", f"acct_{'_'.join(entity_ids)}"),
            type=ContradictionType.ATTRIBUTION.value,
            entity_ids=entity_ids,
            attribute="account_number",
            values=list(accounts),
            sources=[e.get("source", {}).get("file_name", "?") for e in entities],
            severity="high",
            run_id=run_id,
        ))

    # 5. SOURCE_CONTENT: Same fact described differently across sources
    descriptions = {}
    for e in entities:
        desc = e.get("attributes", {}).get("description", "") or e.get("attributes", {}).get("notes", "")
        src = e.get("source", {}).get("file_name", "unknown")
        if desc:
            descriptions[src] = desc
    if len(descriptions) > 1:
        desc_values = list(descriptions.values())
        # Check if descriptions are substantially different (not just rephrased)
        unique_words = set()
        for d in desc_values:
            unique_words.update(d.lower().split())
        avg_len = sum(len(d.split()) for d in desc_values) / len(desc_values) if desc_values else 0
        if avg_len > 3 and len(unique_words) > avg_len * 1.5:
            contradictions.append(Contradiction(
                id=generate_id("CON", f"src_{'_'.join(entity_ids)}"),
                type=ContradictionType.SOURCE_CONTENT.value,
                entity_ids=entity_ids,
                attribute="description",
                values=desc_values[:5],
                sources=list(descriptions.keys()),
                severity="low",
                run_id=run_id,
            ))

    # 6. EVENT_IDENTITY: Same event described as different events
    event_types = set()
    for e in entities:
        etype = e.get("entity_type", "")
        if etype == "EVENT":
            event_desc = e.get("name", "") or e.get("attributes", {}).get("description", "")
            if event_desc:
                event_types.add(event_desc[:50])
    if len(event_types) > 1:
        contradictions.append(Contradiction(
            id=generate_id("CON", f"evt_{'_'.join(entity_ids)}"),
            type=ContradictionType.EVENT_IDENTITY.value,
            entity_ids=entity_ids,
            attribute="event_description",
            values=list(event_types),
            sources=[e.get("source", {}).get("file_name", "?") for e in entities],
            severity="medium",
            run_id=run_id,
        ))

    # 7. DERIVATION: Derived claim contradicts source observations
    derivation_depths = set(e.get("derivation_depth", 0) for e in entities)
    if max(derivation_depths, default=0) > 0 and min(derivation_depths, default=0) == 0:
        contradictions.append(Contradiction(
            id=generate_id("CON", f"drv_{'_'.join(entity_ids)}"),
            type=ContradictionType.DERIVATION.value,
            entity_ids=entity_ids,
            attribute="derivation_depth",
            values=[str(d) for d in derivation_depths],
            sources=[e.get("source", {}).get("file_name", "?") for e in entities],
            severity="low",
            run_id=run_id,
        ))

    # 8. UNKNOWN: Generic catch-all for unclassifiable contradictions
    # (Only if we found attribute conflicts but couldn't classify them)

    # Father name conflicts
    fathers = set()
    for e in entities:
        father = e.get("attributes", {}).get("father_name", "")
        if father:
            fathers.add(father)
    if len(fathers) > 1:
        contradictions.append(Contradiction(
            id=generate_id("CON", f"father_{'_'.join(entity_ids)}"),
            type=ContradictionType.ATTRIBUTE.value,
            entity_ids=entity_ids,
            attribute="father_name",
            values=list(fathers),
            sources=[e.get("source", {}).get("file_name", "?") for e in entities],
            severity="medium",
            run_id=run_id,
        ))

    # Role conflicts
    roles = set()
    for e in entities:
        role = e.get("attributes", {}).get("role", "")
        if role:
            roles.add(role)
    if len(roles) > 1:
        contradictions.append(Contradiction(
            id=generate_id("CON", f"role_{'_'.join(entity_ids)}"),
            type=ContradictionType.ROLE.value,
            entity_ids=entity_ids,
            attribute="role",
            values=list(roles),
            sources=[e.get("source", {}).get("file_name", "?") for e in entities],
            severity="medium",
            run_id=run_id,
        ))

    return contradictions


def build_resolution_history(
    entity_id: str,
    resolution_type: str,
    merged_with: List[str],
    confidence: float,
    method: str,
    signals: dict,
    run_id: str,
    previous_state: str = "",
) -> ResolutionHistory:
    """Create a ResolutionHistory entry for audit trail."""
    return ResolutionHistory(
        entity_id=entity_id,
        resolution_type=resolution_type,
        merged_with=merged_with,
        confidence=confidence,
        method=method,
        signals=signals,
        timestamp=datetime.now().isoformat(),
        run_id=run_id,
        previous_state=previous_state,
    )


def consolidate_roles(resolved_entities: List[ResolvedEntity]) -> Dict[str, str]:
    """
    Role consolidation — when same person appears across multiple FIRs,
    consolidate their roles. Per docs: accused in multiple FIRs stays accused.
    """
    role_counts: Dict[str, Dict[str, int]] = {}  # canonical_name → {role: count}
    for r in resolved_entities:
        name = r.canonical_name
        if name not in role_counts:
            role_counts[name] = {}
        role = r.attributes.get("role", "")
        if role:
            role_counts[name][role] = role_counts[name].get(role, 0) + 1

    consolidated = {}
    for name, roles in role_counts.items():
        if not roles:
            consolidated[name] = ""
            continue
        # Pick the role with highest count; if tie, prefer "accused" > "witness" > "victim"
        priority = {"accused": 3, "complainant": 2, "witness": 1, "victim": 0}
        best_role = max(roles.keys(), key=lambda r: (roles[r], priority.get(r, -1)))
        consolidated[name] = best_role
        # Set consolidated role back on resolved entities
        for r in resolved_entities:
            if r.canonical_name == name:
                r.attributes["consolidated_role"] = best_role
                r.attributes["role_source_count"] = roles[best_role]

    return consolidated


def merge_attributes(entities: List[dict]) -> dict:
    """Merge attributes from multiple entities, keeping all values.

    List-valued attributes (e.g. `observed_in`) are unioned element-wise —
    a list is never appended inside another list — and lists are copied on
    first write so a member's own attributes are never mutated through a
    shared reference.
    """
    merged = {}
    for e in entities:
        attrs = e.get("attributes", {})
        for k, v in attrs.items():
            if v and v != "null" and v != "None":
                if k not in merged or not merged[k]:
                    # Copy: `merged[k] = v` would alias the member's list and
                    # later appends would corrupt the member's attributes.
                    merged[k] = list(v) if isinstance(v, list) else v
                elif merged[k] != v:
                    existing = merged[k]
                    if isinstance(existing, list):
                        incoming = list(v) if isinstance(v, list) else [v]
                        for item in incoming:
                            if item not in existing:
                                existing.append(item)
                    elif isinstance(v, list):
                        merged[k] = [existing] + [x for x in v if x != existing]
                    else:
                        merged[k] = [existing, v]
    return merged


def collect_aliases(entities: List[dict]) -> List[str]:
    """Collect all aliases from entities."""
    aliases = set()
    for e in entities:
        name = e.get("name", "")
        if name:
            aliases.add(name)
        for alias in e.get("aliases", []):
            if alias:
                aliases.add(alias)
    return sorted(aliases)


def collect_phones(entities: List[dict]) -> List[str]:
    """Collect all phone numbers from entities."""
    phones = set()
    for e in entities:
        phone = e.get("attributes", {}).get("phone_number", "")
        if phone:
            phones.add(phone)
        phone = e.get("attributes", {}).get("phone", "")
        if phone:
            phones.add(phone)
    return sorted(phones)


def collect_accounts(entities: List[dict]) -> List[str]:
    """Collect all account numbers from entities."""
    accounts = set()
    for e in entities:
        acc = e.get("attributes", {}).get("account_number", "")
        if acc:
            accounts.add(acc)
    return sorted(accounts)


def build_provenance(source_entities: List[dict]) -> List[str]:
    """Files that contribute evidence to a resolved entity.

    First-creation source first, then attributes.observed_in — the files
    that re-observed an already-indexed entity during extraction (unique,
    order preserved).
    """
    chain: List[str] = []
    for se in source_entities:
        f = (se.get("source") or {}).get("file_name", "")
        if f and f not in chain:
            chain.append(f)
        for f2 in ((se.get("attributes") or {}).get("observed_in")) or []:
            if f2 and f2 not in chain:
                chain.append(f2)
    return chain


def merge_entities(
    verdicts: List[LLMVerdict],
    entities_by_id: Dict[str, dict],
    relations: List[dict],
    run_id: str = "",
) -> Tuple[List[ResolvedEntity], List[UnknownEntity], List[Contradiction]]:
    """
    Create ResolvedEntities from validated merges.
    Uses union-find to handle chain merges (A↔B and B↔C → A,B,C in one group).
    Includes resolution history tracking and role consolidation.
    """
    from collections import defaultdict

    resolved = []
    unknowns = []
    contradictions = []
    resolution_history = []  # Track every resolution decision

    # Union-Find for grouping entities
    parent: Dict[str, str] = {}
    members: Dict[str, List[str]] = {}

    def find(x: str) -> str:
        if x not in parent:
            parent[x] = x
            members[x] = [x]
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def _digits(eid: str) -> set:
        import re
        name = (entities_by_id.get(eid, {}) or {}).get("name", "") or ""
        return set(re.findall(r"\d+", name))

    def union(x: str, y: str):
        rx, ry = find(x), find(y)
        if rx == ry:
            return
        # Digits are identifiers: components carrying different numeric
        # tokens never union. Catches transitive bridges the pairwise
        # candidate guard can't see (Camera 5 ↔ base ↔ Camera 7).
        ma, mb = members[rx], members[ry]
        for a in ma:
            da = _digits(a)
            if not da:
                continue
            for b in mb:
                db = _digits(b)
                if db and da != db:
                    return
        parent[rx] = ry
        members[ry] = ma + members[ry]
        del members[rx]

    # Build groups from merge verdicts — ONLY same-type verdicts.
    # Cross-type verdicts (e.g. PERSON ↔ LOCATION from content-hash ID collision)
    # must never enter union-find: they chain unrelated entities into one giant
    # group that then either gets wholesale-rejected or over-merged by type-split.
    merge_verdicts = [v for v in verdicts if v.merge]
    same_type_verdicts = []
    for verdict in merge_verdicts:
        v_types = set()
        for eid in verdict.entity_ids:
            e = entities_by_id.get(eid)
            if e:
                v_types.add(e.get("entity_type", "") or "UNKNOWN")
        if len(v_types) > 1:
            continue  # skip cross-type verdict entirely
        same_type_verdicts.append(verdict)
        for eid in verdict.entity_ids:
            find(eid)  # ensure exists
        for i in range(1, len(verdict.entity_ids)):
            union(verdict.entity_ids[0], verdict.entity_ids[i])
    merge_verdicts = same_type_verdicts

    # Group entities by their root
    groups: Dict[str, List[str]] = defaultdict(list)
    all_eids = set(entities_by_id.keys())
    for eid in all_eids:
        root = find(eid)
        groups[root].append(eid)

    # Defensive: split any residual mixed-type group (e.g. empty-string type
    # sneaking past the same-type filter above). LLM-reclassified groups are
    # left intact — the LLM's type judgment overrides the original assignment.
    split_groups: List[List[str]] = []
    for root, member_ids in groups.items():
        if len(member_ids) < 2:
            split_groups.append(member_ids)
            continue
        sample_entities = [entities_by_id[eid] for eid in member_ids if eid in entities_by_id]
        has_reclassified = any(
            e.get("attributes", {}).get("llm_reclassified", False)
            for e in sample_entities
        )
        if has_reclassified:
            split_groups.append(member_ids)
            continue
        by_type: Dict[str, List[str]] = defaultdict(list)
        for eid in member_ids:
            etype = entities_by_id.get(eid, {}).get("entity_type", "") or "UNKNOWN"
            by_type[etype].append(eid)
        if len(by_type) == 1:
            split_groups.append(member_ids)
        else:
            names = [entities_by_id.get(eid, {}).get("name", eid) for eid in member_ids]
            print(f"  [TYPE-SPLIT] Mixed-type group split by type: {names} types={set(by_type.keys())}")
            for _etype, ids in by_type.items():
                split_groups.append(ids)

    # PERSON name-cluster split: single-linkage via person_names_compatible.
    # Oversized / mixed-surname mega-groups (Rajesh+Rakesh+Suresh+Bhai chained
    # through shared-phone links) must split into compatible clusters BEFORE
    # SIZE-CAP / NAME-GUARD, so legitimate nickname merges (Suresh ⊂ Suresh
    # Kumar) still happen while cross-person chains are broken.
    from .rule_pass import person_names_compatible as _pnc
    final_groups: List[List[str]] = []
    for member_ids in split_groups:
        if len(member_ids) < 2:
            final_groups.append(member_ids)
            continue
        ents = [entities_by_id[eid] for eid in member_ids if eid in entities_by_id]
        types = set(e.get("entity_type", "") for e in ents)
        if types != {"PERSON"} or len(ents) <= 2:
            final_groups.append(member_ids)
            continue
        # Union-find on name compatibility within this PERSON group
        p_parent = {e["id"]: e["id"] for e in ents}

        def p_find(x):
            while p_parent[x] != x:
                p_parent[x] = p_parent[p_parent[x]]
                x = p_parent[x]
            return x

        def p_union(a, b):
            ra, rb = p_find(a), p_find(b)
            if ra != rb:
                p_parent[ra] = rb

        for i in range(len(ents)):
            for j in range(i + 1, len(ents)):
                if _pnc(ents[i].get("name", ""), ents[j].get("name", "")):
                    p_union(ents[i]["id"], ents[j]["id"])
        clusters: Dict[str, List[str]] = defaultdict(list)
        for e in ents:
            clusters[p_find(e["id"])].append(e["id"])
        if len(clusters) == 1:
            final_groups.append(member_ids)
        else:
            names = [e.get("name", "") for e in ents]
            print(f"  [NAME-SPLIT] PERSON group split into {len(clusters)} name-clusters: {names}")
            for _root, ids in clusters.items():
                final_groups.append(ids)

    # Detect contradictions and create resolved entities for merged groups
    for member_ids in final_groups:
        if len(member_ids) < 2:
            continue

        # Collect source entities
        source_entities = [entities_by_id[eid] for eid in member_ids if eid in entities_by_id]
        if not source_entities:
            continue

        # Defensive type guard: subgroups should already be homogeneous after the
        # split above; if a residual mixed group remains (e.g. empty-string type),
        # reject it rather than merging across types.
        entity_types = set(e.get("entity_type", "") for e in source_entities)
        if len(entity_types) > 1:
            has_reclassified = any(
                e.get("attributes", {}).get("llm_reclassified", False)
                for e in source_entities
            )
            if not has_reclassified:
                print(f"  [TYPE-GUARD] Rejected merge group: {[e.get('name','') for e in source_entities]} types={entity_types}")
                continue  # Skip cross-type merge groups

        # SIZE CAP: transitive single-linkage chaining through weak fuzzy links
        # creates mega-groups (37 dates, 26 locations, all phones in one node).
        # Reject groups above a reasonable size — legitimate merges (Meena×4,
        # Amit×2, Rakesh variants) stay small; mega-chains fall back to standalone.
        MAX_MERGE_GROUP_SIZE = 8
        if len(source_entities) > MAX_MERGE_GROUP_SIZE:
            print(f"  [SIZE-CAP] Rejected oversized merge group ({len(source_entities)} > {MAX_MERGE_GROUP_SIZE}): "
                  f"{[e.get('name','') for e in source_entities[:6]]}...")
            continue

        # VALUE-TYPE EXACT MATCH: PHONE/AMOUNT/DATE/ACCOUNT/EVENT identifiers must
        # be identical to merge — fuzzy edit-distance on sequential phone numbers
        # or similar dates wrongly chains distinct values.
        _value_types = {"PHONE", "AMOUNT", "DATE", "ACCOUNT", "EVENT"}
        if entity_types and entity_types <= _value_types:
            names = set(e.get("name", "").strip() for e in source_entities)
            if len(names) > 1:
                print(f"  [VALUE-GUARD] Rejected non-exact value merge: {sorted(names)[:5]}")
                continue

        # PERSON NAME COMPATIBILITY: prevent Rakesh+Suresh+Rajesh chaining via
        # shared-phone / weak-fuzzy links. All PERSON names in a merge group must
        # share a common first-name token (or be containment-compatible).
        if entity_types == {"PERSON"} and len(source_entities) > 2:
            from .rule_pass import normalize_name, person_names_compatible
            base = None
            compatible = True
            for e in source_entities:
                n = e.get("name", "")
                if base is None:
                    base = n
                    continue
                if not person_names_compatible(base, n):
                    # allow if shares first token with any other member
                    firsts = {normalize_name(x.get("name", ""), "PERSON").split()[0]
                              for x in source_entities if x.get("name")}
                    if len(firsts) > 2:
                        compatible = False
                        break
            if not compatible:
                firsts = sorted({(e.get("name") or "").split()[0] for e in source_entities if e.get("name")})
                print(f"  [NAME-GUARD] Rejected incompatible PERSON merge: {[e.get('name','') for e in source_entities]}")
                continue

        # Detect contradictions
        merge_contradictions = detect_contradictions(source_entities, run_id)
        contradictions.extend(merge_contradictions)

        # Find the best verdict for this group (highest confidence)
        best_verdict = None
        for v in merge_verdicts:
            if any(eid in member_ids for eid in v.entity_ids):
                if best_verdict is None or v.confidence > best_verdict.confidence:
                    best_verdict = v

        # Merge attributes
        merged_attrs = merge_attributes(source_entities)

        # Find related relations
        source_relation_ids = []
        for rel in relations:
            if rel.get("source_entity_id") in member_ids or \
               rel.get("target_entity_id") in member_ids:
                source_relation_ids.append(rel.get("id", ""))

        # Determine canonical name: longest base form, possessive 's stripped.
        # "Meena Devi's" and "Meena Devi" → canonical "Meena Devi".
        # Handle-style identifiers (underscored, e.g. 'suresh_kumar_01') must
        # never win over human-readable names ('Suresh Kumar'): rank the
        # underscore-free candidates first, falling back to the full set only
        # when a cluster consists solely of handles.
        import re as _re
        candidates = []
        for e in source_entities:
            name = e.get("name", "")
            if not name or "unknown" in name.lower():
                continue
            base = _re.sub(r"[''`]s$", "", name).strip() or name
            candidates.append((name, base))
        readable = [c for c in candidates if "_" not in c[0]]
        canonical_name = ""
        best_score = -1
        for name, base in (readable or candidates):
            # Prefer longer base; at equal length prefer non-possessive original
            score = len(base) * 2 + (0 if name == base else -1)
            if score > best_score:
                canonical_name = base
                best_score = score
        if not canonical_name:
            canonical_name = source_entities[0].get("name", "Unknown")

        # Create resolved entity with identity uncertainty propagation
        now = datetime.now().isoformat()
        # Include entity_type + source member IDs so different-type merges with the
        # same canonical name (e.g. LOCATION "Meena Devi's" vs PERSON "Meena Devi's")
        # never collide and silently overwrite each other in resolved_entities.json.
        type_prefix = (source_entities[0].get("entity_type") or "UNKNOWN").upper()
        res_id = generate_id("RES", f"{type_prefix}:{canonical_name}:{'|'.join(sorted(member_ids))}")

        # Calculate effective_confidence = f(merge_confidence, source_identity_confidence)
        # Per docs: effective_confidence = merge_confidence × avg_source_identity_confidence
        merge_conf = best_verdict.confidence if best_verdict else 1.0
        source_identity_confs = []
        for se in source_entities:
            conf = se.get("confidence", {})
            if isinstance(conf, dict):
                source_identity_confs.append(conf.get("source_reliability", 0.5))
        avg_identity_conf = sum(source_identity_confs) / len(source_identity_confs) if source_identity_confs else 0.5
        effective_confidence = merge_conf * avg_identity_conf

        # Build provenance chains from source entities (first-creation
        # sources + extraction re-observations via observed_in)
        provenance_chains = build_provenance(source_entities)

        # Determine epistemic status: if merge was LLM-evaluated, it's an inference
        epistemic_status = "inference" if best_verdict and best_verdict.reasoning else "observation"

        resolved_entity = ResolvedEntity(
            id=res_id,
            canonical_name=canonical_name,
            entity_type=source_entities[0].get("entity_type", "PERSON"),
            aliases=collect_aliases(source_entities),
            phones=collect_phones(source_entities),
            accounts=collect_accounts(source_entities),
            attributes=merged_attrs,
            merge_confidence=merge_conf,
            merge_type=best_verdict.merge_type if best_verdict else "auto",
            source_entities=member_ids,
            source_relations=source_relation_ids,
            llm_reasoning=best_verdict.reasoning if best_verdict else "chain merge",
            signals=best_verdict.signals if best_verdict else {},
            contradictions=[c.id for c in merge_contradictions],
            created_at=now,
            updated_at=now,
            epistemic_status=epistemic_status,
            derivation_depth=1,  # Merged entities are inferred
            provenance_chain=provenance_chains,
            effective_confidence=effective_confidence,
        )
        resolved.append(resolved_entity)

        # Track resolution history
        resolution_history.append(build_resolution_history(
            entity_id=res_id,
            resolution_type="merge",
            merged_with=member_ids,
            confidence=merge_conf,
            method=best_verdict.signals.get("method", "llm") if best_verdict and best_verdict.signals else "chain",
            signals=best_verdict.signals if best_verdict else {},
            run_id=run_id,
            previous_state=str(member_ids),
        ))

    # Create UnknownEntities for unresolved "Unknown" entities
    merged_ids = set()
    for r in resolved:
        merged_ids.update(r.source_entities)

    for eid, entity in entities_by_id.items():
        if eid in merged_ids:
            continue
        name = entity.get("name", "")
        name_lower = name.lower().strip()
        # Exact match only — don't misclassify real entities like "Rakesh Callery"
        if name_lower in ("unknown", "caller", "unknown caller", "unknown person"):
            unknowns.append(UnknownEntity(
                id=generate_id("UNK", eid),
                description=name,
                entity_type=entity.get("entity_type", "PERSON"),
                source_entity=eid,
                attributes=entity.get("attributes", {}),
                confidence=entity.get("confidence", {}).get("score", 0.3) if isinstance(entity.get("confidence"), dict) else 0.3,
            ))
            merged_ids.add(eid)

    # Remaining unmerged entities become resolved entities themselves (ALL types, not just PERSON).
    # Standalone singletons MUST be emitted: without them the canonical store
    # has no record of a never-merged entity and Stage 5 has to synthesize an
    # ad-hoc node from the raw extraction id.
    for eid, entity in entities_by_id.items():
        if eid in merged_ids:
            continue
        now = datetime.now().isoformat()
        canonical_name = entity.get("name", "") or ""
        # Include entity_type + canonical name + source ID so standalone RES
        # IDs never collide with a merge of the same name or type.
        stype = (entity.get("entity_type") or "UNKNOWN").upper()
        res_id = generate_id("RES", f"{stype}:{canonical_name or eid}:{eid}")

        # Relations touching this entity belong to the singleton too — they
        # are its only edges in the graph.
        source_relation_ids = [
            rel.get("id", "") for rel in relations
            if rel.get("source_entity_id") == eid or rel.get("target_entity_id") == eid
        ]

        conf = entity.get("confidence", {})
        effective_confidence = conf.get("score", 0.5) if isinstance(conf, dict) else 0.5

        resolved.append(ResolvedEntity(
            id=res_id,
            canonical_name=canonical_name,
            entity_type=entity.get("entity_type", "PERSON"),
            aliases=entity.get("aliases", []),
            phones=collect_phones([entity]),
            accounts=collect_accounts([entity]),
            attributes=entity.get("attributes", {}),
            merge_confidence=1.0,
            merge_type="single",
            source_entities=[eid],
            source_relations=source_relation_ids,
            llm_reasoning="No merge candidates — standalone entity",
            signals={},
            created_at=now,
            updated_at=now,
            epistemic_status="observation",
            derivation_depth=0,
            provenance_chain=build_provenance([entity]),
            effective_confidence=effective_confidence,
        ))
        merged_ids.add(eid)

    # Role consolidation across all resolved entities
    consolidate_roles(resolved)

    return resolved, unknowns, contradictions, resolution_history
