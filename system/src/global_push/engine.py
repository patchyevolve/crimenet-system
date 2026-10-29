"""Stage 11: Global Entity Push — cross-case identity index.

Design: PIPELINE_SYSTEM_DESIGN.md §2.9, DATA_FLOW.md "Stage 11: Global
Entity Push", INTERNAL_BINDINGS.md GlobalEntity/GlobalEntityLink/
CrossCaseAlert stores, OUTPUTS.md file shapes.

Pushes identity signals from this Case's ResolvedEntities into the shared
global index: match (phone_exact > account_exact > name_exact > name_fuzzy,
same entity_type only), then create/update GlobalEntity rows and
GlobalEntityLink rows; generate CrossCaseAlerts for entities that span
multiple jurisdictions. Exports global_entities.json,
global_entity_links.json, cross_case_alerts.json.
"""

import json
import os
import re
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path
from typing import List, Optional, Tuple

from ..models.schema import generate_id, GlobalEntity, GlobalEntityLink, CrossCaseAlert
from ..resolution.rule_pass import normalize_name

FUZZY_NAME_MIN_SIM = 0.92
ASSOCIATE_STRENGTHEN_BONUS = 0.05


def extract_identity_signals(entity: dict) -> dict:
    """Extract identity signals from one ResolvedEntity (design step 2)."""
    canonical_name = entity.get("canonical_name", "") or ""
    aliases = [a for a in entity.get("aliases", []) if a]
    names = [canonical_name] + aliases if canonical_name else list(aliases)

    phones = sorted({_canon_phone(p) for p in entity.get("phones", []) if p})
    accounts = sorted({_canon_account(a) for a in entity.get("accounts", []) if a})
    addresses = [str(a) for a in entity.get("addresses", []) if a]

    etype = entity.get("entity_type", "")
    norm_names = sorted({normalize_name(n, etype) for n in names if normalize_name(n, etype)})

    return {
        "entity_id": entity.get("id", ""),
        "entity_type": entity.get("entity_type", ""),
        "canonical_name": canonical_name,
        "names": names,
        "norm_names": norm_names,
        "phones": phones,
        "accounts": accounts,
        "addresses": addresses,
        "created_at": entity.get("created_at", ""),
        "updated_at": entity.get("updated_at", ""),
    }


def match_identity(sig: dict, globals_list: List[dict]) -> Optional[Tuple[dict, str, float]]:
    """Match one entity's signals against the global index.

    Returns (global_record, match_type, confidence) or None. Same
    entity_type required for every rule — a PERSON never links to a PHONE.
    """
    etype = sig.get("entity_type", "")
    phone_set = {_canon_phone(p) for p in sig.get("phones", []) if p}
    acct_set = {_canon_account(a) for a in sig.get("accounts", []) if a}
    norm_names = sig.get("norm_names", [])

    best: Optional[Tuple[dict, str, float]] = None
    priority = {"phone_exact": 0, "account_exact": 1, "name_exact": 2, "name_fuzzy": 3}

    for g in globals_list:
        if g.get("entity_type") != etype:
            continue

        g_phones = {_canon_phone(p) for p in g.get("phones", []) if p}
        g_accounts = {_canon_account(a) for a in g.get("accounts", []) if a}

        # A name match must not override contradictory unique identifiers.
        # Different recorded phone/account sets may reflect an updated
        # identifier; leave that identity unlinked until a person reviews it.
        if phone_set and g_phones and not (phone_set & g_phones):
            continue
        if acct_set and g_accounts and not (acct_set & g_accounts):
            continue

        if phone_set and g_phones & phone_set:
            cand = (g, "phone_exact", 0.99)
        elif acct_set and g_accounts & acct_set:
            cand = (g, "account_exact", 0.97)
        elif norm_names and set(g.get("norm_names", [])) & set(norm_names):
            cand = (g, "name_exact", 0.90)
        else:
            fuzzy = _best_fuzzy(norm_names, g.get("norm_names", []))
            if fuzzy is None:
                continue
            cand = (g, "name_fuzzy", round(fuzzy, 4))

        if best is None or priority[cand[1]] < priority[best[1]]:
            best = cand

    return best


def _digit_runs(s: str) -> tuple:
    """All digit runs in a normalized name, in order: '12th march 2024' →
    ('12', '2024')."""
    return tuple(re.findall(r"\d+", s))


def _best_fuzzy(names_a: List[str], names_b: List[str]) -> Optional[float]:
    """Best fuzzy ratio across name pairs.

    Digit guard: a pair only qualifies when its digit runs are identical.
    Character similarity alone would link different values —
    '12 march 2024' vs '14 march 2024' scores 0.923, 'rs 5,00,000' vs
    'rs 15,00,000' scores ~0.96 — while '12 march 2024' vs '12th march
    2024' (same date, different format) shares digit runs and may link.
    """
    best = None
    for a in names_a:
        da = _digit_runs(a)
        for b in names_b:
            if da != _digit_runs(b):
                continue
            r = SequenceMatcher(None, a, b).ratio()
            if r >= FUZZY_NAME_MIN_SIM and (best is None or r > best):
                best = r
    return best


def _canon_phone(p: str) -> str:
    return re.sub(r"[^\d+]", "", str(p))


def _canon_account(a: str) -> str:
    return re.sub(r"\s+", "", str(a)).upper()


def _new_global_id(sig: dict) -> str:
    content = ":".join([
        sig.get("entity_type", ""),
        normalize_name(sig.get("canonical_name", ""), sig.get("entity_type", "")),
        ",".join(sig.get("phones", [])),
        ",".join(sig.get("accounts", [])),
    ])
    return generate_id("GLOB", content)


def _parse_ts(value: str) -> Optional[datetime]:
    """Parse an ISO timestamp to a *naive UTC* datetime (None on failure).

    JSON payloads carry naive datetimes while Postgres returns offset-aware
    ones; returning naive UTC for both makes min()/max() comparisons in
    ``_absorb_signals`` total instead of raising TypeError.
    """
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        if dt.tzinfo is not None:
            dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
        return dt
    except (ValueError, TypeError):
        return None


def _make_global(sig: dict, canonical_id: str) -> dict:
    now = datetime.now().isoformat()
    return {
        "canonical_id": canonical_id,
        "entity_type": sig["entity_type"],
        "canonical_name": sig["canonical_name"],
        "phones": list(sig["phones"]),
        "accounts": list(sig["accounts"]),
        "addresses": list(sig["addresses"]),
        "norm_names": list(sig["norm_names"]),
        "names": list(sig["names"]),
        "first_seen": sig["created_at"] or now,
        "last_seen": sig["updated_at"] or now,
        "total_cases": 0,
        "total_jurisdictions": 0,
    }


def _absorb_signals(g: dict, sig: dict) -> None:
    """Merge an entity's identity signals into its global record."""
    for key in ("phones", "accounts", "addresses"):
        merged = sorted(set(g.get(key, [])) | set(sig.get(key, [])))
        g[key] = merged
    g["norm_names"] = sorted(set(g.get("norm_names", [])) | set(sig.get("norm_names", [])))
    g["names"] = sorted(set(g.get("names", [])) | set(sig.get("names", [])))

    created, updated = _parse_ts(sig["created_at"]), _parse_ts(sig["updated_at"])
    first, last = _parse_ts(g.get("first_seen", "")), _parse_ts(g.get("last_seen", ""))
    cands_first = [d for d in (first, created) if d]
    cands_last = [d for d in (last, updated) if d]
    if cands_first:
        g["first_seen"] = min(cands_first).isoformat()
    if cands_last:
        g["last_seen"] = max(cands_last).isoformat()


def _file_global(g: dict) -> dict:
    """File shape per OUTPUTS.md (snake_case, no index-internal fields)."""
    return {
        "canonical_id": g["canonical_id"],
        "entity_type": g["entity_type"],
        "canonical_name": g["canonical_name"],
        "phones": g["phones"],
        "accounts": g["accounts"],
        "addresses": g["addresses"],
        "first_seen": g["first_seen"],
        "last_seen": g["last_seen"],
        "total_cases": g["total_cases"],
        "total_jurisdictions": g["total_jurisdictions"],
    }


class GlobalPushEngine:
    """Stage 11 engine — push identity signals to the global index."""

    def __init__(self):
        self.summary = {}

    def push(self, output_dir: str, run_id: str = "", case_id: str | None = None,
             jurisdiction_node_id: str | None = None,
             index_dir: str | None = None,
             database_enabled: bool = True) -> dict:
        out = Path(output_dir)

        resolved_file = out / "resolved_entities.json"
        if not resolved_file.exists():
            print("[STAGE 11] WARNING: resolved_entities.json not found — skipping push")
            return {"mode": "skipped", "entities_pushed": 0}

        with open(resolved_file, encoding="utf-8") as f:
            resolved = json.load(f)
        entities = list(resolved.values()) if isinstance(resolved, dict) else resolved
        signals = [extract_identity_signals(e) for e in entities if e.get("id")]

        if database_enabled:
            conn, db_error = self._connect()
        else:
            conn, db_error = None, "database disabled by caller"
        jurisdiction = jurisdiction_node_id
        db_writable = conn is not None and bool(case_id)

        index_root = Path(index_dir) if index_dir else out
        existing = []
        if conn is not None:
            existing = self._load_index(conn)
        elif (index_root / "global_entities.json").exists():
            with (index_root / "global_entities.json").open(encoding="utf-8") as f:
                stored = json.load(f)
            existing = list(stored.values()) if isinstance(stored, dict) else stored
            for entity in existing:
                # File contract intentionally omits matching internals; rebuild
                # them from the canonical name when resuming file-only runs.
                entity.setdefault("names", [entity.get("canonical_name", "")])
                entity.setdefault("norm_names", [normalize_name(
                    entity.get("canonical_name", ""), entity.get("entity_type", "")
                )])

        existing_links = []
        links_path = index_root / "global_entity_links.json"
        if links_path.exists():
            with links_path.open(encoding="utf-8") as f:
                existing_links = json.load(f)

        globals_map = {g["canonical_id"]: g for g in existing}
        matched_by = {"phone_exact": 0, "account_exact": 0, "name_exact": 0, "name_fuzzy": 0}
        links: List[dict] = []
        created = matched = absorbed = ambiguous = 0

        for sig in signals:
            if not sig["entity_id"]:
                continue
            result = match_identity(sig, list(globals_map.values()))

            if result is None:
                gid = _new_global_id(sig)
                g = globals_map.get(gid)
                if g is None:
                    g = _make_global(sig, gid)
                    globals_map[gid] = g
                    created += 1
                else:
                    # Deterministic id already known to the index (e.g. an
                    # identity that existed in an earlier run) — absorb, and
                    # count it: otherwise signals silently fail to add up
                    # (created + matched != entities_pushed).
                    _absorb_signals(g, sig)
                    absorbed += 1
                matched_global, match_type, conf = g, "new_identity", 1.0
            else:
                matched_global, match_type, conf = result
                _absorb_signals(matched_global, sig)
                matched_by[match_type] += 1
                matched += 1
                if self._count_candidates(sig, globals_map) > 1:
                    ambiguous += 1

            links.append({
                "global_entity_id": matched_global["canonical_id"],
                "case_id": case_id or "",
                "local_entity_id": sig["entity_id"],
                "jurisdiction_node_id": jurisdiction or "",
                "confidence": conf,
                "match_type": match_type,
            })

        strengthened = self._strengthen_from_graph(links, out)

        if db_writable and jurisdiction is None:
            jurisdiction = self._lookup_case_jurisdiction(conn, case_id)
            for l in links:
                l["jurisdiction_node_id"] = l["jurisdiction_node_id"] or jurisdiction or ""

        touched_ids = sorted({l["global_entity_id"] for l in links})
        alerts_created = 0
        mode = "file_only"

        # Keep an append/replace-by-case file index when PostgreSQL is absent;
        # this makes local multi-case runs exercise the same identity matching.
        current_case_links = [l for l in existing_links if l.get("case_id") != (case_id or "")]
        all_links = current_case_links + links

        # Files first: a DB failure must not lose the stage's outputs.
        self._write_files(out, globals_map, links, created_ids=touched_ids,
                          case_id=case_id or "", jurisdiction=jurisdiction or "",
                          all_links=all_links)
        if index_root != out:
            index_root.mkdir(parents=True, exist_ok=True)
            self._write_files(index_root, globals_map, all_links, created_ids=touched_ids,
                              case_id="", jurisdiction="", all_links=all_links)

        if db_writable and jurisdiction:
            try:
                alerts_created = self._persist(
                    conn, globals_map, links, touched_ids, case_id
                )
                mode = "db"
                # Matching uses the database-wide index. Keep the shared
                # index artifacts equally authoritative instead of exporting
                # only links that happened to be loaded from a local folder.
                self._write_workspace_index(index_root, conn)
            except Exception as e:
                try:
                    conn.rollback()
                except Exception:
                    pass
                db_error = f"persist failed: {e}"
                print(f"[STAGE 11] WARNING: DB persist failed ({e}) — file outputs only")
        else:
            reason = db_error or ("case_id not provided" if not case_id else
                                  "jurisdiction could not be resolved")
            print(f"[STAGE 11] WARNING: DB push skipped ({reason}) — file outputs only")

        if conn is not None:
            conn.close()

        summary = {
            "mode": mode,
            "case_id": case_id or "",
            "entities_pushed": len(signals),
            "globals_created": created,
            "globals_matched": matched,
            "globals_absorbed": absorbed,
            "links": len(links),
            "matched_by": matched_by,
            "ambiguous_matches": ambiguous,
            "strengthened_links": strengthened,
            "alerts_created": alerts_created,
            "total_globals": len(globals_map),
            "files": ["global_entities.json", "global_entity_links.json",
                      "cross_case_alerts.json"],
        }
        self.summary = summary
        print(f"[STAGE 11] Global push ({summary['mode']}): "
              f"{summary['entities_pushed']} entities → "
              f"{created} new / {matched} matched / {absorbed} absorbed globals, "
              f"{len(links)} links, {alerts_created} alerts")
        return summary

    # ── matching helpers ──────────────────────────────────────────────

    def _count_candidates(self, sig: dict, globals_map: dict) -> int:
        """How many distinct globals this entity could match (ambiguity)."""
        count = 0
        seen = set()
        etype = sig.get("entity_type", "")
        phone_set = {_canon_phone(p) for p in sig.get("phones", []) if p}
        acct_set = {_canon_account(a) for a in sig.get("accounts", []) if a}
        for gid, g in globals_map.items():
            if g.get("entity_type") != etype or gid in seen:
                continue
            g_phones = {_canon_phone(p) for p in g.get("phones", []) if p}
            g_accounts = {_canon_account(a) for a in g.get("accounts", []) if a}
            if phone_set and g_phones & phone_set:
                pass
            elif acct_set and g_accounts & acct_set:
                pass
            elif set(g.get("norm_names", [])) & set(sig.get("norm_names", [])):
                pass
            elif _best_fuzzy(sig.get("norm_names", []), g.get("norm_names", [])) is not None:
                pass
            else:
                continue
            seen.add(gid)
            count += 1
        return count

    def _strengthen_from_graph(self, links: List[dict], out: Path) -> int:
        """Known-associate overlap strengthens existing links (design step 3).

        If two local entities linked to the SAME global are connected in the
        case graph, both link confidences gain a small bonus.
        """
        edges_file = out / "graph_edges.json"
        if not edges_file.exists():
            return 0
        with open(edges_file, encoding="utf-8") as f:
            edges = json.load(f)

        adjacency = {}
        for e in edges:
            s, t = e.get("source_id"), e.get("target_id")
            if s and t:
                adjacency.setdefault(s, set()).add(t)
                adjacency.setdefault(t, set()).add(s)

        by_global = {}
        for l in links:
            by_global.setdefault(l["global_entity_id"], []).append(l)

        strengthened = 0
        for gid, group in by_global.items():
            if len(group) < 2:
                continue
            boosted = set()
            for i, a in enumerate(group):
                for b in group[i + 1:]:
                    neighbors = adjacency.get(a["local_entity_id"], set())
                    if b["local_entity_id"] not in neighbors:
                        continue
                    pair = frozenset((id(a), id(b)))
                    if pair in boosted:
                        continue
                    boosted.add(pair)
                    for l in (a, b):
                        l["confidence"] = round(min(1.0, l["confidence"] + ASSOCIATE_STRENGTHEN_BONUS), 4)
                    strengthened += 1
        return strengthened

    # ── database ──────────────────────────────────────────────────────

    def _connect(self):
        try:
            import psycopg2
            db_url = os.environ.get(
                "DATABASE_URL",
                "postgresql://criminal:criminal_secret@localhost:5432/criminal_network_db",
            )
            if "?" in db_url:
                db_url = db_url.split("?")[0]
            conn = psycopg2.connect(db_url)
            conn.autocommit = False
            return conn, None
        except Exception as e:
            return None, str(e)

    def _load_index(self, conn) -> List[dict]:
        cur = conn.cursor()
        cur.execute("""
            SELECT "canonicalId", "entityType", "canonicalName", phones, accounts,
                   addresses, "firstSeen", "lastSeen", "totalCases", "totalJurisdictions"
            FROM "GlobalEntity"
        """)
        rows = []
        for r in cur.fetchall():
            rows.append({
                "canonical_id": r[0],
                "entity_type": r[1],
                "canonical_name": r[2],
                "phones": r[3] or [],
                "accounts": r[4] or [],
                "addresses": r[5] or [],
                "norm_names": sorted({normalize_name(r[2], r[1])} - {""}),
                "names": [r[2]],
                "first_seen": r[6].isoformat() if r[6] else "",
                "last_seen": r[7].isoformat() if r[7] else "",
                "total_cases": r[8],
                "total_jurisdictions": r[9],
            })
        cur.close()
        return rows

    def _lookup_case_jurisdiction(self, conn, case_id: str) -> Optional[str]:
        cur = conn.cursor()
        cur.execute('SELECT "jurisdictionNodeId" FROM "Case" WHERE id = %s', (case_id,))
        row = cur.fetchone()
        cur.close()
        return row[0] if row else None

    def _write_workspace_index(self, out: Path, conn) -> None:
        """Export the complete persisted workspace identity view."""
        cur = conn.cursor()
        cur.execute('SELECT "canonicalId", "entityType", "canonicalName", phones, '
                    'accounts, addresses, "firstSeen", "lastSeen", "totalCases", '
                    '"totalJurisdictions" FROM "GlobalEntity" ORDER BY "canonicalId"')
        globals_out = {}
        for row in cur.fetchall():
            entity = {
                "canonical_id": row[0], "entity_type": row[1],
                "canonical_name": row[2], "phones": row[3] or [],
                "accounts": row[4] or [], "addresses": row[5] or [],
                "first_seen": row[6].isoformat() if row[6] else "",
                "last_seen": row[7].isoformat() if row[7] else "",
                "total_cases": row[8] or 0,
                "total_jurisdictions": row[9] or 0,
            }
            globals_out[row[0]] = _file_global(entity)

        cur.execute('SELECT id, "globalEntityId", "caseId", "localEntityId", '
                    '"jurisdictionNodeId", confidence, "matchType", "createdAt" '
                    'FROM "GlobalEntityLink" ORDER BY "caseId", id')
        links_out = [{
            "id": r[0], "global_entity_id": r[1], "case_id": r[2],
            "local_entity_id": r[3], "jurisdiction_node_id": r[4],
            "confidence": r[5], "match_type": r[6],
            "created_at": r[7].isoformat() if r[7] else "",
        } for r in cur.fetchall()]

        cur.execute('SELECT id, "globalEntityId", severity, recommendation, '
                    '"createdAt", status, "reviewedById", "reviewedAt" '
                    'FROM "CrossCaseAlert" ORDER BY "createdAt", id')
        alert_rows = cur.fetchall()
        cur.execute('SELECT "alertId", "caseId", "jurisdictionNodeId" '
                    'FROM "CrossCaseAlertCase" ORDER BY "alertId", "caseId"')
        cases_by_alert = {}
        for alert_id, case_id, jurisdiction_id in cur.fetchall():
            item = cases_by_alert.setdefault(alert_id, {"cases": [], "jurisdictions": []})
            if case_id not in item["cases"]:
                item["cases"].append(case_id)
            if jurisdiction_id and jurisdiction_id not in item["jurisdictions"]:
                item["jurisdictions"].append(jurisdiction_id)
        cur.close()

        alerts_out = []
        for r in alert_rows:
            members = cases_by_alert.get(r[0], {"cases": [], "jurisdictions": []})
            alerts_out.append({
                "id": r[0], "global_entity_id": r[1], "severity": r[2],
                "case_ids": members["cases"],
                "jurisdiction_node_ids": members["jurisdictions"],
                "recommendation": r[3],
                "created_at": r[4].isoformat() if r[4] else "",
                "status": r[5], "reviewed_by_id": r[6],
                "reviewed_at": r[7].isoformat() if r[7] else None,
            })

        out.mkdir(parents=True, exist_ok=True)
        payloads = {
            "global_entities.json": globals_out,
            "global_entity_links.json": links_out,
            "cross_case_alerts.json": alerts_out,
        }
        for filename, payload in payloads.items():
            with (out / filename).open("w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, ensure_ascii=False)

    def _persist(self, conn, globals_map: dict, links: List[dict],
                 touched_ids: List[str], case_id: str) -> int:
        try:
            cur = conn.cursor()

            # Reconcile this case's links: local entity ids that no longer
            # exist in the current resolution output are superseded, and
            # locals remapped to a different global since the last push must
            # not leave a second row behind (one link per local per case).
            current_globals = {l["local_entity_id"]: l["global_entity_id"]
                               for l in links}
            cur.execute('SELECT "localEntityId", "globalEntityId" FROM "GlobalEntityLink" '
                        'WHERE "caseId" = %s', (case_id,))
            stale = [(loc, glob) for loc, glob in cur.fetchall()
                     if current_globals.get(loc) != glob]
            for loc, glob in stale:
                cur.execute('DELETE FROM "GlobalEntityLink" '
                            'WHERE "caseId" = %s AND "localEntityId" = %s '
                            'AND "globalEntityId" = %s',
                            (case_id, loc, glob))
            recompute_ids = sorted(set(touched_ids) | {glob for _, glob in stale})

            for gid in touched_ids:
                g = globals_map[gid]
                cur.execute("""
                    INSERT INTO "GlobalEntity"
                        ("canonicalId", "entityType", "canonicalName", phones,
                         accounts, addresses, "firstSeen", "lastSeen",
                         "totalCases", "totalJurisdictions", "createdAt", "updatedAt")
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, 0, 0, NOW(), NOW())
                    ON CONFLICT ("canonicalId") DO UPDATE SET
                        phones = EXCLUDED.phones,
                        accounts = EXCLUDED.accounts,
                        addresses = EXCLUDED.addresses,
                        "firstSeen" = LEAST("GlobalEntity"."firstSeen", EXCLUDED."firstSeen"),
                        "lastSeen" = GREATEST("GlobalEntity"."lastSeen", EXCLUDED."lastSeen"),
                        "updatedAt" = NOW()
                """, (
                    g["canonical_id"], g["entity_type"], g["canonical_name"] or "(unnamed)",
                    json.dumps(g["phones"]), json.dumps(g["accounts"]),
                    json.dumps(g["addresses"]),
                    g["first_seen"] or datetime.now().isoformat(),
                    g["last_seen"] or datetime.now().isoformat(),
                ))

            for l in links:
                cur.execute("""
                    INSERT INTO "GlobalEntityLink"
                        (id, "globalEntityId", "caseId", "localEntityId",
                         "jurisdictionNodeId", confidence, "matchType", "createdAt")
                    VALUES (%s, %s, %s, %s, %s, %s, %s, NOW())
                    ON CONFLICT ("globalEntityId", "caseId", "localEntityId") DO UPDATE SET
                        confidence = EXCLUDED.confidence,
                        "matchType" = EXCLUDED."matchType"
                """, (
                    generate_id("GLINK", f"{l['global_entity_id']}:{l['case_id']}:{l['local_entity_id']}"),
                    l["global_entity_id"], l["case_id"], l["local_entity_id"],
                    l["jurisdiction_node_id"], l["confidence"], l["match_type"],
                ))

            # Aggregate case/jurisdiction counts + seen window per touched global
            for gid in recompute_ids:
                cur.execute("""
                    SELECT COUNT(DISTINCT "caseId"), COUNT(DISTINCT "jurisdictionNodeId"),
                           MIN("createdAt"), MAX("createdAt")
                    FROM "GlobalEntityLink" WHERE "globalEntityId" = %s
                """, (gid,))
                cases, juris, first, last = cur.fetchone()
                cur.execute("""
                    UPDATE "GlobalEntity" SET
                        "totalCases" = %s, "totalJurisdictions" = %s,
                        "firstSeen" = LEAST("firstSeen", %s),
                        "lastSeen" = GREATEST("lastSeen", %s),
                        "updatedAt" = NOW()
                    WHERE "canonicalId" = %s
                """, (cases or 0, juris or 0, first or datetime.now(),
                      last or datetime.now(), gid))

            # Superseded globals (id-content changed, links remapped away)
            # must not accumulate: the index reflects present state. Globals
            # still linked from any case — or referenced by an alert — stay.
            cur.execute("""
                DELETE FROM "GlobalEntity" g
                WHERE NOT EXISTS (
                    SELECT 1 FROM "GlobalEntityLink" l
                    WHERE l."globalEntityId" = g."canonicalId")
                  AND NOT EXISTS (
                    SELECT 1 FROM "CrossCaseAlert" a
                    WHERE a."globalEntityId" = g."canonicalId")
            """)

            alerts_created = self._create_alerts(cur, recompute_ids)

            conn.commit()
            cur.close()
            return alerts_created
        except Exception:
            conn.rollback()
            raise

    def _create_alerts(self, cur, touched_ids: List[str]) -> int:
        created = 0
        for gid in touched_ids:
            cur.execute('SELECT "totalCases", "totalJurisdictions", "canonicalName" '
                        'FROM "GlobalEntity" WHERE "canonicalId" = %s', (gid,))
            row = cur.fetchone()
            if not row:
                continue
            cases, juris, name = row
            if (juris or 0) < 2:
                continue

            cur.execute('SELECT id FROM "CrossCaseAlert" '
                        'WHERE "globalEntityId" = %s AND status = %s',
                        (gid, "ACTIVE"))
            if cur.fetchone():
                continue

            # A resolved alert can recur when a later case links to this
            # identity. Preserve the old row as history and create a new
            # episode ID instead of colliding with its primary key.
            cur.execute('SELECT id FROM "CrossCaseAlert" '
                        'WHERE "globalEntityId" = %s ORDER BY "createdAt", id', (gid,))
            previous_alert_ids = {r[0] for r in cur.fetchall()}
            generation = 1
            alert_id = generate_id("ALERT", gid)
            while alert_id in previous_alert_ids:
                generation += 1
                alert_id = generate_id("ALERT", f"{gid}:{generation}")
            severity = "HIGH" if juris >= 3 else "MEDIUM"
            recommendation = (
                f"Entity '{name}' appears in {cases} cases across {juris} jurisdictions "
                f"— verify cross-case identity before merging investigations"
            )
            cur.execute("""
                INSERT INTO "CrossCaseAlert"
                    (id, "globalEntityId", severity, recommendation, "createdAt", status)
                VALUES (%s, %s, %s, %s, NOW(), 'ACTIVE')
            """, (alert_id, gid, severity, recommendation))

            cur.execute("""
                SELECT DISTINCT "caseId", "jurisdictionNodeId"
                FROM "GlobalEntityLink" WHERE "globalEntityId" = %s
            """, (gid,))
            for case_row in cur.fetchall():
                cur.execute("""
                    INSERT INTO "CrossCaseAlertCase"
                        (id, "alertId", "caseId", "jurisdictionNodeId")
                    VALUES (%s, %s, %s, %s)
                    ON CONFLICT DO NOTHING
                """, (generate_id("ALERTCASE", f"{alert_id}:{case_row[0]}"),
                      alert_id, case_row[0], case_row[1]))
            created += 1
        return created

    # ── file export ───────────────────────────────────────────────────

    def _write_files(self, out: Path, globals_map: dict, links: List[dict],
                     created_ids: List[str], case_id: str, jurisdiction: str,
                     all_links: Optional[List[dict]] = None) -> None:
        all_links = all_links if all_links is not None else links
        by_global: dict = {}
        for l in all_links:
            by_global.setdefault(l["global_entity_id"], []).append(l)

        # File totals: DB rows carry authoritative counts after persist; in
        # file-only mode derive them from this run's links. Computed before
        # the globals dump so the exported counts are current.
        for gid, g in globals_map.items():
            group = by_global.get(gid, [])
            if group:
                g["total_cases"] = max(
                    g.get("total_cases", 0), len({l["case_id"] for l in group if l["case_id"]})
                )
                g["total_jurisdictions"] = max(
                    g.get("total_jurisdictions", 0),
                    len({l["jurisdiction_node_id"] for l in group if l["jurisdiction_node_id"]}),
                )

        with open(out / "global_entities.json", "w", encoding="utf-8") as f:
            json.dump(
                {gid: _file_global(g) for gid, g in sorted(globals_map.items())},
                f, indent=2, ensure_ascii=False,
            )

        link_records = [
            GlobalEntityLink(
                id=generate_id("GLINK", f"{l['global_entity_id']}:{l['case_id']}:{l['local_entity_id']}"),
                global_entity_id=l["global_entity_id"],
                case_id=l["case_id"],
                local_entity_id=l["local_entity_id"],
                jurisdiction_node_id=l["jurisdiction_node_id"],
                confidence=l["confidence"],
                match_type=l["match_type"],
                created_at=datetime.now().isoformat(),
            ).to_dict()
            for l in links
        ]
        with open(out / "global_entity_links.json", "w", encoding="utf-8") as f:
            json.dump(link_records, f, indent=2, ensure_ascii=False)

        alerts: List[dict] = []
        for gid in created_ids:
            g = globals_map.get(gid)
            group = by_global.get(gid, [])
            case_ids = sorted({l["case_id"] for l in group if l.get("case_id")})
            juris_ids = sorted({l["jurisdiction_node_id"] for l in group
                                if l.get("jurisdiction_node_id")})
            # File alerts mirror DB rule: multi-jurisdiction entities only.
            # In a single-case push no link spans 2 jurisdictions, so the file
            # list reflects only entities whose global record already spans
            # jurisdictions (loaded from the index).
            total_juris = g.get("total_jurisdictions", 0) if g else 0
            if total_juris < 2 and len(set(juris_ids)) < 2:
                continue
            alerts.append(CrossCaseAlert(
                id=generate_id("ALERT", gid),
                global_entity_id=gid,
                severity="HIGH" if max(total_juris, len(juris_ids)) >= 3 else "MEDIUM",
                case_ids=case_ids,
                jurisdiction_node_ids=juris_ids,
                recommendation=(
                    f"Entity '{g['canonical_name'] if g else gid}' appears across "
                    f"{len(case_ids)} cases — verify cross-case identity before merging"
                ),
                created_at=datetime.now().isoformat(),
            ).to_dict())

        with open(out / "cross_case_alerts.json", "w", encoding="utf-8") as f:
            json.dump(alerts, f, indent=2, ensure_ascii=False)
