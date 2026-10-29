"""
Resolution Engine — Advanced Entity Resolution Pipeline
Playbook V4 Module C.1 + C.2

Pipeline:
  1. Rule Pass (fast exact matching)
  2. Phonetic Matching (Indian name transliteration)
  3. Fuzzy Index Search (~ operator)
  4. Multi-Signal Disambiguation (phone, vehicle, DOB, location, case)
  5. Group Similar (candidate detection for LLM)
  6. LLM Evaluate (intelligent judgment)
  7. Ranked Candidates ("Did you mean..." + confirmation gate)
  8. Merge + Unknown + Contradiction
  9. Knowledge Graph Construction (C.2 schema)
"""

import json
import re
import time
from typing import List, Dict, Tuple, Optional
from pathlib import Path
from datetime import datetime

from ..models.schema import (
    ResolvedEntity, UnknownEntity, Contradiction,
    MergeCandidate, LLMVerdict, ResolutionHistory,
)
from ..ai.caller import AICaller
from .rule_pass import (
    rule_pass, normalize_name, normalize_phone,
    person_names_compatible, reconcile_entity_types,
)
from .group import find_groups, deduplicate_candidates
from .llm_pass import llm_pass
from .merger import merge_entities
from .phonetic import phonetic_candidates, phonetic_similarity
from .fuzzy_index import FuzzyIndex
from .disambiguator import MultiSignalDisambiguator
from .candidate_formatter import CandidateFormatter
from .graph_schema import KnowledgeGraph


def _prune_artifacts(resolved: List, relations) -> Tuple[List, List]:
    """Drop post-merge extraction artifacts.

    Called AFTER merges so acronym orgs that merged with their expansions —
    HDFC/PNB/BoB — survive; those singletons would otherwise be caught below.

      - Single-token ORGANIZATION with no relation endpoints is an
        extraction artifact ("OTP", "SIM", "CDR", "FIR", "CCTV", "Phone",
        "Account") unless another multi-token ORG expands to it — its
        initials match this token ("PNB" beside "Punjab National Bank") —
        or it is connected (evidence edges). Structural only: no word lists.
      - AMOUNT entities without any digit are not amounts ("FIR" from a
        CARDINAL mislabel).
    """
    relation_members = set()
    for rel in relations:
        relation_members.add(rel.get("source_entity_id"))
        relation_members.add(rel.get("target_entity_id"))
    # Initials of every multi-token ORG ("punjab national bank" -> "pnb").
    # Single-token orgs contribute nothing, so a candidate never partners
    # with itself.
    org_initials = set()
    for r in resolved:
        if r.entity_type == "ORGANIZATION":
            toks = (r.canonical_name or "").split()
            if len(toks) >= 2 and all(t.isalnum() for t in toks):
                org_initials.add("".join(t[0] for t in toks).lower())
    keep_resolved = []
    pruned_artifacts = []
    for r in resolved:
        name = (r.canonical_name or "").strip()
        connected = any(sid in relation_members for sid in r.source_entities)
        if not connected and r.entity_type == "ORGANIZATION" and len(name.split()) == 1:
            if name.lower() not in org_initials:
                pruned_artifacts.append(f"ORG:{name}")
                continue
        if not connected and r.merge_type == "single" and r.entity_type == "AMOUNT" \
                and not re.search(r"\d", name):
            pruned_artifacts.append(f"AMOUNT:{name}")
            continue
        keep_resolved.append(r)
    return keep_resolved, pruned_artifacts


class ResolutionEngine:
    """Advanced entity resolution with phonetic, fuzzy, multi-signal, and graph construction."""

    def __init__(self, ai: Optional[AICaller] = None):
        self.ai = ai
        self.fuzzy_index = FuzzyIndex()
        self.disambiguator = MultiSignalDisambiguator()
        self.candidate_formatter = CandidateFormatter()
        self.graph = KnowledgeGraph()

    @staticmethod
    def _load_face_candidates(output_dir: str) -> List[MergeCandidate]:
        """Face-match candidates from Stage 2.5 (RESEARCH_FACE_RECOGNITION
        doc 08 §4) — emitted as MergeCandidates with signal ``face_match``.
        Callers filter to candidates whose endpoints are real entities."""
        path = Path(output_dir) / "face_embeddings.json"
        if not path.is_file():
            return []
        try:
            with path.open(encoding="utf-8") as fh:
                data = json.load(fh)
        except (OSError, ValueError):
            return []
        if not isinstance(data, dict):
            return []
        out: List[MergeCandidate] = []
        for cand in data.get("candidates") or []:
            if not isinstance(cand, dict):
                continue
            a = str(cand.get("source_entity_id") or "")
            b = str(cand.get("candidate_entity_id") or "")
            if not a or not b:
                continue
            signals = cand.get("signals") or {}
            out.append(MergeCandidate(
                entity_ids=[a, b],
                signal="face_match",
                description=(
                    f"Face embeddings match "
                    f"(similarity {float(signals.get('face_similarity', 0.0)):.3f})"
                ),
                confidence=float(cand.get("confidence") or 0.0),
            ))
        return out

    def resolve(
        self,
        entities: List[dict],
        relations: List[dict],
        output_dir: str = "output",
        run_id: str = "",
    ) -> dict:
        """
        Run full advanced entity resolution pipeline.
        Returns summary dict with resolved, unknowns, contradictions, graph stats.
        """
        start_time = time.time()

        # Step 0: settle cross-type exact-name duplicates before anything
        # indexes or groups entities (NER emits PERSON/ORG/LOC inconsistently).
        entities = reconcile_entity_types(entities, relations)

        # Index entities by ID
        entities_by_id = {e["id"]: e for e in entities if isinstance(e, dict) and "id" in e}

        print(f"\n[RESOLUTION] Starting with {len(entities)} entities")

        # Step 1: Build fuzzy index for ~ operator search
        print("[RESOLUTION] Step 1: Building fuzzy index...")
        self._build_index(entities)
        print(f"  -> Indexed {len(self.fuzzy_index)} entities")

        # Step 2: Rule Pass (exact matching)
        print("[RESOLUTION] Step 2: Rule pass (exact matching)...")
        rule_candidates = rule_pass(entities)
        print(f"  -> {len(rule_candidates)} auto-merge candidates found")

        # Step 3: Phonetic matching
        print("[RESOLUTION] Step 3: Phonetic matching (Indian names)...")
        phonetic_cands = self._phonetic_pass(entities)
        print(f"  -> {len(phonetic_cands)} phonetic candidates found")

        # Step 4: Fuzzy index search (~ operator)
        print("[RESOLUTION] Step 4: Fuzzy index search (~ operator)...")
        fuzzy_cands = self._fuzzy_index_pass(entities)
        print(f"  -> {len(fuzzy_cands)} fuzzy index candidates found")

        # Step 5: Multi-signal disambiguation on top candidates
        print("[RESOLUTION] Step 5: Multi-signal disambiguation...")
        # Face-based candidates (Stage 2.5 — RESEARCH_FACE_RECOGNITION doc 08 §4):
        # person↔person pairs whose faces matched at the review floor.
        face_cands = [
            c for c in self._load_face_candidates(output_dir)
            if all(eid in entities_by_id for eid in c.entity_ids)
        ]
        if face_cands:
            print(f"  -> {len(face_cands)} face-match candidates found")
        all_candidates = rule_candidates + phonetic_cands + fuzzy_cands + face_cands
        all_candidates = deduplicate_candidates(all_candidates)
        disambiguated = self._disambiguate_pass(all_candidates, entities_by_id)
        print(f"  -> {len(disambiguated)} disambiguated pairs")

        # Step 6: Group Similar (for LLM evaluation)
        print("[RESOLUTION] Step 6: Finding similar groups...")
        group_candidates = find_groups(entities, rule_candidates)
        group_candidates = deduplicate_candidates(group_candidates)
        # Merge with disambiguated candidates
        combined = disambiguated + group_candidates
        combined = deduplicate_candidates(combined)
        print(f"  -> {len(combined)} total candidates for evaluation")

        # Step 7: LLM Evaluate (if available)
        verdicts = []
        if combined and self.ai:
            print(f"[RESOLUTION] Step 7: LLM evaluating {len(combined)} groups...")
            # HARD REJECT: Different entity types cannot merge — filter before LLM
            # Skip type guard if any candidate has LLM-reclassified entities — the LLM
            # already judged these as merge-worthy after type correction.
            type_filtered = []
            for c in combined:
                # Check if any entity in this candidate was LLM-reclassified
                has_reclassified = any(
                    e.get("attributes", {}).get("llm_reclassified", False)
                    for e in c.entities
                )
                if has_reclassified:
                    # LLM already reclassified types — trust its judgment
                    type_filtered.append(c)
                else:
                    types = set()
                    for e in c.entities:
                        t = e.get("entity_type", "")
                        if t:
                            types.add(t)
                    if len(types) > 1:
                        print(f"  [TYPE-GUARD] Rejected cross-type merge (pre-LLM): {[e.get('name','') for e in c.entities]} types={types}")
                        continue
                    type_filtered.append(c)
            print(f"  -> {len(type_filtered)} candidates after type guard (rejected {len(combined) - len(type_filtered)} cross-type)")

            verdicts = llm_pass(type_filtered, self.ai)
            print(f"  -> {len([v for v in verdicts if v.merge])} merges approved")
            print(f"  -> {len([v for v in verdicts if not v.merge])} merges rejected")
        elif combined:
            print("[RESOLUTION] Step 7: No LLM, using rule-based verdicts...")
            for c in combined:
                # HARD REJECT: Different entity types cannot merge
                types = set()
                for e in c.entities:
                    t = e.get("entity_type", "")
                    if t:
                        types.add(t)
                if len(types) > 1:
                    continue  # Skip cross-type merges entirely

                # Check disambiguator recommendation from signal
                is_review = "multi-signal" in c.signal and "review" in c.signal
                is_merge_signal = "multi-signal" in c.signal and "merge" in c.signal

                # Merge if: high confidence OR disambiguator recommends merge/review
                # OR compatible name signals (first-name / partial) that were
                # already person_names_compatible-filtered in group.py.
                # first_name_match=0.85, partial_name_match=0.8 — accept these
                # when the pair passed person_names_compatible (nickname/short-form
                # merges like "Suresh Bhai" ⊂ "Suresh Kumar").
                should_merge = (
                    c.confidence >= 0.9
                    or is_merge_signal
                    or (is_review and c.confidence >= 0.6)
                    or (c.signal == "first_name_match" and c.confidence >= 0.85)
                    or (c.signal == "partial_name_match" and c.confidence >= 0.8)
                )

                verdicts.append(LLMVerdict(
                    entity_ids=c.entity_ids,
                    merge=should_merge,
                    confidence=c.confidence,
                    canonical_name=c.entities[0].get("name", "") if c.entities else "",
                    reasoning=f"Rule-based: {c.signal}",
                    signals={"rule_based": c.confidence, "method": "rule_based"},
                    merge_type="auto" if c.confidence > 0.95 else "review",
                ))

        # Step 8: Ranked Candidates + Confirmation Gate
        print("[RESOLUTION] Step 8: Ranked candidates + confirmation gate...")
        ranked_results = self._rank_candidates(verdicts, entities_by_id)
        print(f"  -> {len(ranked_results)} ranked candidate sets")
        auto_merges = len([r for r in ranked_results if r.status == "auto_merge"])
        review_required = len([r for r in ranked_results if r.status == "review_required"])
        print(f"  -> Auto-merges: {auto_merges}, Review required: {review_required}")

        # Step 9: Merge + Unknown + Contradiction
        print("[RESOLUTION] Step 9: Merging entities...")
        resolved, unknowns, contradictions, resolution_history = merge_entities(
            verdicts, entities_by_id, relations, run_id,
        )

        # Post-merge artifact prune (AFTER merges — see _prune_artifacts).
        resolved, pruned_artifacts = _prune_artifacts(resolved, relations)
        if pruned_artifacts:
            print(f"  -> Pruned {len(pruned_artifacts)} artifact entities: {pruned_artifacts[:10]}")

        for r in resolved:
            r.run_id = run_id

        print(f"  -> {len(resolved)} resolved entities")
        print(f"  -> {len(unknowns)} unknown entities")
        print(f"  -> {len(contradictions)} contradictions")

        # Step 10: Knowledge Graph Construction (deferred to GraphBuilder in pipeline Stage 5)
        # Resolution outputs resolved_entities.json; graph building happens in pipeline._stage_graph()

        # Build summary
        elapsed = time.time() - start_time
        summary = {
            "run_id": run_id,
            "pipeline_resolution_run_id": f"res_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "timestamp": datetime.now().isoformat(),
            "input_entities": len(entities),
            "fuzzy_index_size": len(self.fuzzy_index),
            "phonetic_candidates": len(phonetic_cands),
            "fuzzy_index_candidates": len(fuzzy_cands),
            "disambiguated_pairs": len(disambiguated),
            "auto_merges": auto_merges,
            "review_required": review_required,
            "llm_evaluated_groups": len(group_candidates),
            "llm_merges": len([v for v in verdicts if v.merge and v.merge_type == "llm"]),
            "rejected": len([v for v in verdicts if not v.merge]),
            "resolved_entities": len(resolved),
            "unknown_entities": len(unknowns),
            "contradictions": len(contradictions),
            "contradictions_flagged": len([c for c in contradictions if not c.resolved]),
            "resolution_history_entries": len(resolution_history),
            "graph_nodes": 0,
            "graph_edges": 0,
            "graph_node_types": {},
            "graph_edge_types": {},
            "processing_time_seconds": round(elapsed, 2),
        }

        # Save outputs
        self._save_outputs(resolved, unknowns, contradictions, summary, output_dir, run_id, resolution_history)

        # Export raw_id → res_id bridge so Stage 4/6 can join temporal/coverage/
        # contradictions onto graph nodes (fixes cross-stage ID fracture).
        id_map = {}
        for r in resolved:
            for seid in r.source_entities:
                id_map[seid] = r.id
            id_map[r.id] = r.id
        for u in unknowns:
            if u.source_entity:
                id_map[u.source_entity] = u.id
            id_map[u.id] = u.id
        with open(Path(output_dir) / "id_map.json", "w") as f:
            json.dump(id_map, f, indent=2)
        print(f"  -> Saved to {output_dir}/id_map.json ({len(id_map)} mappings)")

        print(f"\n[RESOLUTION] Complete in {elapsed:.2f}s")
        return summary

    def _build_index(self, entities: List[dict]):
        """Build fuzzy inverted index from all entities."""
        self.fuzzy_index = FuzzyIndex()
        for entity in entities:
            eid = entity.get("id", "")
            name = entity.get("name", "")
            etype = entity.get("entity_type", "")
            aliases = entity.get("aliases", [])
            if name:
                self.fuzzy_index.add_entity(eid, name, etype, aliases)

    def _phonetic_pass(self, entities: List[dict]) -> List[MergeCandidate]:
        """Find phonetically similar entity pairs."""
        candidates = []
        person_entities = [e for e in entities if e.get("entity_type") == "PERSON" and e.get("name")]

        # Build name list for phonetic search
        name_list = [(e["name"], e["id"]) for e in person_entities]
        by_id = {e["id"]: e for e in person_entities}

        seen_pairs = set()
        for entity in person_entities:
            name = entity["name"]
            matches = phonetic_candidates(name, name_list, threshold=0.65)

            for match_eid, match_name, score in matches:
                if match_eid == entity["id"]:
                    continue
                pair = tuple(sorted([entity["id"], match_eid]))
                if pair in seen_pairs:
                    continue
                # Same safety as the fuzzy pass: phonetic codes equate
                # different given names (Mahesh/Harish) — surname alone
                # never proves identity.
                if not person_names_compatible(name, match_name):
                    continue
                seen_pairs.add(pair)

                candidates.append(MergeCandidate(
                    entity_ids=[entity["id"], match_eid],
                    entities=[entity, by_id[match_eid]],
                    confidence=score,
                    signal=f"phonetic: {name} ↔ {match_name} (score: {score:.2f})",
                ))

        return candidates

    def _fuzzy_index_pass(self, entities: List[dict]) -> List[MergeCandidate]:
        """Find similar entities using fuzzy index (~ operator)."""
        candidates = []
        seen_pairs = set()
        by_id = {e.get("id", ""): e for e in entities}

        for entity in entities:
            name = entity.get("name", "")
            if not name:
                continue
            etype = entity.get("entity_type", "")

            # Search the fuzzy index with ~ operator (max edit distance 2)
            results = self.fuzzy_index.search_fuzzy(name, max_distance=2, max_results=10)

            for match_eid, match_name, similarity in results:
                if match_eid == entity["id"]:
                    continue
                pair = tuple(sorted([entity["id"], match_eid]))
                if pair in seen_pairs:
                    continue

                match_entity = by_id.get(match_eid, {})
                match_type = match_entity.get("entity_type", "")

                # Same-type only (type guard needs both entities present)
                if etype and match_type and etype != match_type:
                    continue

                # Digits are identifiers, not typography. Fuzzy index tokens
                # drop single characters, so 'Tower 1' ↔ 'Tower 3' both score
                # 1.00 on ['karol','bagh','tower']. Two names carrying
                # different numeric tokens are different things; one-sided
                # digits are granularity (handled by the ratio guard below).
                d1 = set(re.findall(r"\d+", name or ""))
                d2 = set(re.findall(r"\d+", match_name or ""))
                if d1 and d2 and d1 != d2:
                    continue

                # PERSON: shared surname / compatible given names required.
                # Blocks "Bhai" ↔ "Suresh Bhai" (courtesy token alone at
                # sim 1.00 would otherwise bypass the disambiguator).
                if etype == "PERSON" and match_type == "PERSON":
                    if not person_names_compatible(name, match_name):
                        continue

                # Threshold by type: PERSON can use 0.70 (nickname/typo path
                # is further guarded by person_names_compatible). ORG/LOCATION
                # need near-exact — "Delhi Herald" ≈ "Delhi Police" at JW 0.84
                # must NOT merge.
                if etype == "PERSON":
                    min_sim = 0.70
                else:
                    min_sim = 0.92

                if similarity < min_sim:
                    continue

                # Containment asymmetry: a short name fully contained in a
                # much longer one scores 1.0 on token match but names a
                # different granularity ("Noida" vs "Noida Sector 62 Gate
                # Camera"). Mirrors the disambiguator's 0.6 length-ratio
                # rule so geographic/part-whole hierarchies don't collapse.
                shorter, longer = sorted((name, match_name), key=len)
                if shorter.lower() != longer.lower() and shorter.lower() in longer.lower():
                    if len(shorter) / len(longer) < 0.6:
                        continue

                seen_pairs.add(pair)
                candidates.append(MergeCandidate(
                    entity_ids=[entity["id"], match_eid],
                    entities=[entity, match_entity],
                    confidence=similarity,
                    signal=f"fuzzy~: {name} ↔ {match_name} (similarity: {similarity:.2f})",
                ))

        return candidates

    def _disambiguate_pass(
        self,
        candidates: List[MergeCandidate],
        entities_by_id: dict,
    ) -> List[MergeCandidate]:
        """Run multi-signal disambiguation on candidate pairs."""
        disambiguated = []

        for candidate in candidates:
            eids = candidate.entity_ids
            if len(eids) < 2:
                continue

            entity_a = entities_by_id.get(eids[0], {})
            entity_b = entities_by_id.get(eids[1], {})

            if not entity_a or not entity_b:
                continue

            result = self.disambiguator.disambiguate(entity_a, entity_b)

            # Hard reject (type mismatch / conflicting values) is absolute —
            # never keep a candidate the disambiguator scored 0.0.
            if result.total_score == 0.0 and result.recommendation == "reject":
                continue

            # Grounded reject is absolute too: a positive low-confidence
            # signal on phone/dob/case ("Different phones: ...") is evidence
            # OF difference — mere absence of data emits no signal at all.
            # The conf >= 0.9 preserve below exists for the absence
            # (weak-signal) path; name similarity must never override
            # conflicting hard identity facts ("Bhai" 9876543210 vs
            # "Bhai Sahab" 9876543220). Location excluded: "No location
            # overlap" is contextual, not identity (people move).
            if result.recommendation == "reject" and any(
                s.signal_type in ("phone", "dob", "case") and s.confidence < 0.5
                for s in result.signals
            ):
                continue

            # Preserve high-confidence matches — disambiguator should not
            # downgrade them below merge threshold. Only blend for fuzzy/phonetic.
            # Also: if rule_pass already passed person_names_compatible and
            # confidence >= 0.9, absence of secondary signals (phones/accounts)
            # must not cause a reject — that's the disambiguator's weak-signal path.
            if candidate.confidence >= 0.9:
                candidate.signal += f" | multi-signal: {result.total_score:.2f} ({result.recommendation})"
                # Keep regardless of disambiguator recommendation when rule-pass
                # already validated name compatibility at conf >= 0.9
                disambiguated.append(candidate)
            else:
                # Fuzzy/phonetic: blend original confidence with disambiguation score
                blended = (candidate.confidence * 0.4 + result.total_score * 0.6)
                candidate.confidence = blended
                candidate.signal += f" | multi-signal: {result.total_score:.2f} ({result.recommendation})"
                if result.recommendation != "reject":
                    disambiguated.append(candidate)

        return disambiguated

    def _rank_candidates(
        self,
        verdicts: List[LLMVerdict],
        entities_by_id: dict,
    ) -> List:
        """Convert verdicts into ranked "Did you mean..." candidates."""
        results = []

        for verdict in verdicts:
            if not verdict.merge:
                continue

            eids = verdict.entity_ids
            if len(eids) < 2:
                continue

            # Format as ranked candidates
            candidates = []
            for eid in eids:
                entity = entities_by_id.get(eid, {})
                if entity:
                    source_file = ""
                    source = entity.get("source", {})
                    if isinstance(source, dict):
                        source_file = source.get("file_name", "")
                    candidates.append({
                        "entity_id": eid,
                        "name": entity.get("name", ""),
                        "confidence": verdict.confidence,
                        "evidence_count": 1,
                        "evidence_files": [source_file] if source_file else [],
                        "signals": verdict.signals,
                    })

            if candidates:
                formatted = self.candidate_formatter.format_candidates(
                    query_entity_id=eids[0],
                    query_name=entities_by_id.get(eids[0], {}).get("name", ""),
                    candidates=candidates,
                )
                results.append(formatted)

        return results

    def _save_outputs(
        self,
        resolved: List[ResolvedEntity],
        unknowns: List[UnknownEntity],
        contradictions: List[Contradiction],
        summary: dict,
        output_dir: str,
        run_id: str,
        resolution_history: List[ResolutionHistory] = None,
    ):
        """Save all resolution outputs including graph, confirmation log, and resolution history."""
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        # Resolved entities
        resolved_dict = {r.id: r.to_dict() for r in resolved}
        with open(out / "resolved_entities.json", "w") as f:
            json.dump(resolved_dict, f, indent=2)

        # Unknown entities
        unknowns_dict = {u.id: u.to_dict() for u in unknowns}
        with open(out / "unknown_entities.json", "w") as f:
            json.dump(unknowns_dict, f, indent=2)

        # Contradictions
        contradictions_dict = {c.id: c.to_dict() for c in contradictions}
        with open(out / "contradictions.json", "w") as f:
            json.dump(contradictions_dict, f, indent=2)

        # Resolution history
        if resolution_history:
            history_dict = {h.entity_id: h.to_dict() for h in resolution_history}
            with open(out / "resolution_history.json", "w") as f:
                json.dump(history_dict, f, indent=2)
            print(f"  -> Saved to {output_dir}/resolution_history.json ({len(resolution_history)} entries)")

        # Resolution log
        with open(out / "resolution_log.json", "w") as f:
            json.dump(summary, f, indent=2)

        # Knowledge graph
        self.graph.save(str(out))

        # Confirmation log (never-silently-auto-correct audit trail)
        self.candidate_formatter.save_confirmation_log(str(out))

        # Fuzzy index stats
        with open(out / "fuzzy_index_stats.json", "w") as f:
            json.dump(self.fuzzy_index.stats(), f, indent=2)

        print(f"  -> Saved to {output_dir}/resolved_entities.json")
        print(f"  -> Saved to {output_dir}/knowledge_graph_nodes.json")
        print(f"  -> Saved to {output_dir}/knowledge_graph_edges.json")
        print(f"  -> Saved to {output_dir}/confirmation_log.json")
        print(f"  -> Saved to {output_dir}/resolution_log.json")
