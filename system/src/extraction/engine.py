"""
Extraction Engine
Takes normalized ingestion output and extracts entities + relations.
Two modes: code-based (structured) and LLM-based (unstructured).
"""

import re
import os
import json
import hashlib
from datetime import datetime
from typing import List, Dict, Tuple, Optional
from ..models.schema import (
    EntityType, RelationType, ProvenanceClass, VerificationStatus,
    ExtractedEntity, ExtractedRelation, SourceMetadata, ConfidenceSchema,
    ConfidenceFactor, EpistemicCategory,
    generate_id, file_hash,
)
from ..ai.caller import AICaller
from ..ingestion.column_mapper import ColumnMapper
from ..faces.context import classify_image_evidence, filename_name, is_junk_person_name, should_mint_filename_person


def make_confidence(
    score: float,
    basis: List[str],
    source: SourceMetadata = None,
    epistemic: str = "observation",
    derivation_depth: int = 0,
    run_id: str = "",
) -> ConfidenceSchema:
    """Create ConfidenceSchema with proper factors and epistemic tagging."""
    factors = []

    # Source reliability factor
    if source:
        reliability = (source.reliability_occurrence + source.reliability_identity + source.reliability_intent) / 3
        factors.append(ConfidenceFactor(
            factor_type="source_reliability",
            value=reliability,
            weight=0.4,
            description=f"Average reliability for {source.source_type} source",
        ))

    # Basis factor
    if basis:
        factors.append(ConfidenceFactor(
            factor_type="evidence_basis",
            value=min(1.0, score),
            weight=0.6,
            description=basis[0] if basis else "",
        ))

    return ConfidenceSchema(
        score=score,
        basis=basis,
        source_reliability=source.reliability_occurrence if source else 0.5,
        derivation_depth=derivation_depth,
        semantic_type=epistemic,
        run_id=run_id,
        factors=factors,
    )


def get_semantic_edge_type(relation_type) -> str:
    """Map relation_type to semantic edge type."""
    rt = relation_type.value if hasattr(relation_type, 'value') else str(relation_type)
    mapping = {
        "CALLED": "associational",
        "MESSED": "associational",
        "VISITED": "associational",
        "ASSOCIATED_WITH": "associational",
        "FAMILY_OF": "associational",
        "FRIEND_OF": "associational",
        "ASSOCIATE_OF": "associational",
        "WORKS_WITH": "entrepreneurial",
        "OWNS_ACCOUNT": "entrepreneurial",
        "TRANSFERRED_TO": "entrepreneurial",
        "RECEIVED_FROM": "entrepreneurial",
        "WORKS_AT": "entrepreneurial",
        "MEMBER_OF": "entrepreneurial",
        "SUSPECT_OF": "quasi-governmental",
        "VICTIM_OF": "quasi-governmental",
        "WITNESS_OF": "quasi-governmental",
        "LIVES_AT": "associational",
    }
    return mapping.get(rt, "associational")


def extract_temporal_info(attributes: dict) -> Optional[dict]:
    """Extract temporal info from relation attributes."""
    if not attributes:
        return None
    timestamp = (
        attributes.get("timestamp")
        or attributes.get("call_time")
        or attributes.get("date")
        or attributes.get("date_time")
        or attributes.get("transaction_date")
    )
    if timestamp:
        return {"timestamp": str(timestamp), "source": "attributes"}
    return None


class ExtractionEngine:
    """Extracts entities and relations from ingested data."""

    ORG_KEYWORDS = [
        "electronics", "office", "bank", "ltd", "corp", "inc", "llc",
        "services", "enterprises", "traders", "trading", "mobile",
        "telecom", "communications", "associates", "consultancy",
        "hospital", "medical", "pharmacy", "school", "college",
        "university", "hospitality", "restaurant", "cafe",
        "motors", "automobiles", "industries", "manufacturing",
        "textiles", "jewellers", "jewelry", "diamonds", "gold",
        "impex", "export", "import", "logistics", "transport",
        "realty", "properties", "construction", "builders",
        "airtel", "jio", "vi", "bsnl", "airtel office", "jio office",
        "sbi", "hdfc", "icici", "axis", "kotak", "pnb", "bob",
    ]

    @staticmethod
    def _is_organization_name(name: str) -> bool:
        """Check if a name looks like an organization, not a person."""
        lower = name.lower().strip()
        for kw in ExtractionEngine.ORG_KEYWORDS:
            # Word boundary match to avoid false positives (e.g., "si" in "GIS")
            if re.search(r'\b' + re.escape(kw) + r'\b', lower):
                return True
        return False
    def __init__(self, ai_caller: Optional[AICaller] = None,
                 llm_extraction_threshold: int = 50,
                 max_text_length: int = 5000):
        self.entities = []
        self.relations = []
        # Stage 2 (face): embedding records promoted from Stage 1 detections.
        self.face_embeddings: List[dict] = []
        self.entity_index = {}  # id -> entity
        self.relation_index = {}  # id -> relation
        self.extraction_log = []
        self.ai = ai_caller
        self.llm_extraction_threshold = max(0, llm_extraction_threshold)
        self.max_text_length = max(1, max_text_length)

    def extract_from_file(self, file_info: dict, run_id: str = "") -> dict:
        """Extract entities and relations from one ingested file."""
        file_name = file_info.get("file_name", "unknown")
        file_type = file_info.get("detected_type", "text")
        content = file_info["content"]
        source_type = file_info.get("source_type", file_type)

        # Use source_type from ingestion for proper reliability mapping
        source = SourceMetadata(
            source_type=source_type,
            file_name=file_name,
            file_hash=file_hash(file_info["file_path"]),
            ingestion_time=datetime.now().isoformat(),
        )

        # Set reliability from ingestion matrix
        reliability = file_info.get("reliability", {})
        source.reliability_occurrence = reliability.get("occurrence", 0.5)
        source.reliability_identity = reliability.get("identity", 0.5)
        source.reliability_intent = reliability.get("intent", 0.5)
        source.reliability_location = reliability.get("location", 0.5)
        source.reliability_timing = reliability.get("timing", 0.5)
        # Propagate Stage 1 adversarial flag into source metadata → Stage 5 edges
        source.adversarial_suspicious = bool(
            file_info.get("adversarial_check", {}).get("is_suspicious", False)
        )

        new_entities = []
        new_relations = []

        if file_type == "tabular":
            new_entities, new_relations = self._extract_tabular(content, source)
        elif file_type == "json":
            new_entities, new_relations = self._extract_json(content, source)
        elif file_type == "text":
            new_entities, new_relations = self._extract_text(content, source)
        elif file_type == "image":
            new_entities, new_relations = self._extract_image(content, source)
        elif file_type == "pdf":
            new_entities, new_relations = self._extract_pdf(content, source)
        elif file_type == "docx":
            new_entities, new_relations = self._extract_docx(content, source)

        # Entity validation is limited to unstructured text. Running an LLM
        # over every row in structured CDR/bank/JSON files is costly and can
        # discard well-formed observations without adding extraction value.
        text_value = content.get("text", "") if isinstance(content, dict) else ""
        if (self.ai and new_entities and file_type == "text"
                and len(text_value) >= self.llm_extraction_threshold):
            new_entities = self.validate_entities_with_llm(new_entities)

        # Snapshot existing index keys BEFORE dedup
        existing_entity_keys = set(self.entity_index.keys())
        existing_relation_keys = set(self.relation_index.keys())

        # Deduplicate and index
        new_entity_ids = set(e.id for e in new_entities)
        new_relation_ids = set(r.id for r in new_relations)

        for entity in new_entities:
            # Set run_id and source_id on all entities
            entity.run_id = run_id
            entity.source_id = file_name
            entity.epistemic_category = "observation"  # Raw extraction = observation

            # Set semantic_type and run_id on confidence schema
            if entity.confidence:
                entity.confidence.semantic_type = "observation"
                entity.confidence.run_id = run_id
                entity.confidence.source_reliability = source.reliability_occurrence
                # Regenerate factors with correct source_reliability
                entity.confidence.factors = [
                    ConfidenceFactor(
                        factor_type="source_reliability",
                        value=source.reliability_occurrence,
                        weight=0.4,
                        description=f"Source reliability for {source.source_type}",
                    ),
                    ConfidenceFactor(
                        factor_type="evidence_basis",
                        value=min(1.0, entity.confidence.score),
                        weight=0.6,
                        description=entity.confidence.basis[0] if entity.confidence.basis else "",
                    ),
                ]

            if entity.id not in self.entity_index:
                self.entity_index[entity.id] = entity
                self.entities.append(entity)
            else:
                # Merge: update aliases and attributes
                existing = self.entity_index[entity.id]
                # This file also observed the entity — credit it.
                self._observe(existing, source)
                for alias in entity.aliases:
                    if alias not in existing.aliases:
                        existing.aliases.append(alias)
                for k, v in entity.attributes.items():
                    existing_val = existing.attributes.get(k)
                    # Overwrite if key is new OR existing value is empty/None
                    if k not in existing.attributes or not existing_val:
                        existing.attributes[k] = v

        for relation in new_relations:
            # Set run_id and source_id on all relations
            relation.run_id = run_id
            relation.source_id = file_name
            relation.epistemic_category = "observation"  # Raw extraction = observation

            # Set semantic_type and run_id on confidence schema
            if relation.confidence:
                relation.confidence.semantic_type = "observation"
                relation.confidence.run_id = run_id
                relation.confidence.source_reliability = source.reliability_occurrence
                # Regenerate factors with correct source_reliability
                relation.confidence.factors = [
                    ConfidenceFactor(
                        factor_type="source_reliability",
                        value=source.reliability_occurrence,
                        weight=0.4,
                        description=f"Source reliability for {source.source_type}",
                    ),
                    ConfidenceFactor(
                        factor_type="evidence_basis",
                        value=min(1.0, relation.confidence.score),
                        weight=0.6,
                        description=relation.confidence.basis[0] if relation.confidence.basis else "",
                    ),
                ]

            if relation.id not in self.relation_index:
                self.relation_index[relation.id] = relation
                self.relations.append(relation)
            else:
                # Update existing relation's source if new source has adversarial flag
                # (reprocessing a file should upgrade stale source metadata)
                existing = self.relation_index[relation.id]
                if existing.source and relation.source and relation.source.adversarial_suspicious:
                    existing.source.adversarial_suspicious = True
                    # Also merge any other source fields that may have changed
                    existing.source.reliability_occurrence = relation.source.reliability_occurrence
                    existing.source.reliability_identity = relation.source.reliability_identity
                    existing.source.reliability_intent = relation.source.reliability_intent
                    existing.source.reliability_location = relation.source.reliability_location
                    existing.source.reliability_timing = relation.source.reliability_timing

        # Stage 2 (face): promote face crops detected at ingestion into
        # embedding records — RESEARCH_FACE_RECOGNITION doc 08 §3.
        if file_info.get("face_crops"):
            self.face_embeddings.extend(
                self._extract_face_embeddings(file_info, run_id=run_id)
            )

        log_entry = {
            "file_name": file_name,
            "entities_found": len(new_entities),
            "relations_found": len(new_relations),
            "new_entities_added": len(new_entity_ids - existing_entity_keys),
            "new_relations_added": len(new_relation_ids - existing_relation_keys),
        }
        self.extraction_log.append(log_entry)

        return {
            "source_file": file_name,
            "entities": new_entities,
            "relations": new_relations,
            "log": log_entry,
        }

    def _extract_face_embeddings(self, file_info: dict, run_id: str = "") -> List[dict]:
        """Stage 2 face path: finalize face records detected at ingestion.

        Cutoffs per RESEARCH_FACE_RECOGNITION doc 08 §3.2: detector confidence
        below 0.5 → REJECTED, quality below 0.3 → LOW_QUALITY with the embedding
        dropped, otherwise EXTRACTED (embedding kept for Stage 2.5 matching).
        """
        from ..faces.engine import DET_CUTOFF, QUALITY_CUTOFF

        records: List[dict] = []
        for crop in file_info.get("face_crops") or []:
            if not isinstance(crop, dict):
                continue
            rec = dict(crop)
            if rec.get("status") in (None, "PENDING"):
                det = float(rec.get("detector_confidence") or 0.0)
                quality = float(rec.get("quality_score") or 0.0)
                if det < DET_CUTOFF:
                    rec["status"] = "REJECTED"
                    rec["embedding_vector"] = []
                elif quality < QUALITY_CUTOFF:
                    rec["status"] = "LOW_QUALITY"
                    rec["embedding_vector"] = []
                else:
                    rec["status"] = "EXTRACTED"
            rec["run_id"] = run_id
            records.append(rec)
        return records

    def resolve_dangling_references(self) -> int:
        """
        Post-extraction dedup: fix dangling social media references.
        Social media DM/visit relations use username-based IDs but person entities
        use display-name-based IDs. This method maps username -> display_name and
        rewrites the relation source/target IDs.
        Returns number of relations fixed.
        """
        # Build username -> person_id lookup from all person entities
        username_to_person_id = {}
        person_entities = [e for e in self.entities if e.entity_type.value == "PERSON"]
        for entity in person_entities:
            # Check social_username attribute
            username = entity.attributes.get("social_username", "")
            if username:
                username_to_person_id[username] = entity.id
            # Also check aliases (usernames like "rakesh_rocky")
            for alias in entity.aliases:
                if alias and alias != entity.name:
                    # Check if this alias matches a social username pattern (lowercase, underscores)
                    if re.match(r'^[a-z_]+$', alias):
                        username_to_person_id[alias] = entity.id

        # Also build from entity_index (entities from other files not in self.entities)
        for entity in self.entity_index.values():
            if hasattr(entity, 'entity_type') and hasattr(entity.entity_type, 'value') and entity.entity_type.value == "PERSON":
                username = entity.attributes.get("social_username", "")
                if username:
                    username_to_person_id[username] = entity.id
                for alias in entity.aliases:
                    if alias and alias != entity.name and re.match(r'^[a-z_]+$', alias):
                        username_to_person_id[alias] = entity.id

        # Get all valid entity IDs
        valid_entity_ids = {e.id for e in self.entities}

        # Fix dangling relations
        fixed = 0
        for relation in self.relations:
            src = relation.source_entity_id
            tgt = relation.target_entity_id
            new_src = src
            new_tgt = tgt

            # Check if source is a username-based ID that doesn't exist
            if src not in valid_entity_ids:
                # Try to find the person entity by username
                # Extract username from the ID pattern: PERSON_<hash> where hash = generate_id("PERSON", username)
                for username, person_id in username_to_person_id.items():
                    if generate_id("PERSON", username) == src:
                        new_src = person_id
                        break

            # Check if target is a username-based ID that doesn't exist
            if tgt not in valid_entity_ids:
                for username, person_id in username_to_person_id.items():
                    if generate_id("PERSON", username) == tgt:
                        new_tgt = person_id
                        break

            if new_src != src or new_tgt != tgt:
                relation.source_entity_id = new_src
                relation.target_entity_id = new_tgt
                fixed += 1

        if fixed:
            print(f"[DEDUP] Fixed {fixed} dangling references")

        return fixed

    def ensure_entity_completeness(self) -> int:
        """
        Post-extraction: create stub entities for any IDs referenced by relations
        but not present in the entity list. Prevents broken graph edges.
        Returns number of stub entities created.
        """
        valid_entity_ids = {e.id for e in self.entities}
        referenced_ids = set()
        for rel in self.relations:
            referenced_ids.add(rel.source_entity_id)
            referenced_ids.add(rel.target_entity_id)

        missing_ids = referenced_ids - valid_entity_ids
        if not missing_ids:
            return 0

        # Build a name lookup from relation attributes
        id_to_name = {}
        for rel in self.relations:
            attrs = rel.attributes or {}
            src_name = attrs.get("source_name", "")
            tgt_name = attrs.get("target_name", "")
            if src_name and rel.source_entity_id:
                id_to_name[rel.source_entity_id] = src_name
            if tgt_name and rel.target_entity_id:
                id_to_name[rel.target_entity_id] = tgt_name

        created = 0
        for mid in missing_ids:
            # Infer type from ID prefix
            entity_type_str = mid.split("_")[0] if "_" in mid else "PERSON"
            try:
                entity_type = EntityType(entity_type_str)
            except ValueError:
                entity_type = EntityType.PERSON

            name = id_to_name.get(mid, mid)
            stub = ExtractedEntity(
                id=mid,
                entity_type=entity_type,
                name=name,
                attributes={"stub": True, "reason": "relation-referenced"},
                confidence=ConfidenceSchema(score=0.5, basis=["inferred from relation"]),
                source=SourceMetadata(
                    source_type="inferred",
                    file_name="",
                    file_hash="",
                    ingestion_time=datetime.now().isoformat(),
                ),
                extraction_method="code",
            )
            self.entities.append(stub)
            self.entity_index[mid] = stub
            created += 1

        if created:
            print(f"[DEDUP] Created {created} stub entities for relation-referenced IDs")

        return created

    @staticmethod
    def _observe(entity, source) -> None:
        """Record that `source` re-observed an already-indexed entity.

        The single `source` field can only name the file that FIRST created
        an entity; every later file that sees the same identity would vanish
        from provenance. `attributes.observed_in` accumulates those later
        files (unique, first-creation order preserved) and is merged into the
        resolved entity's provenance chain in resolution.
        """
        fname = getattr(source, "file_name", "") or ""
        if not fname:
            return
        seen = entity.attributes.setdefault("observed_in", [])
        if fname not in seen:
            seen.append(fname)

    def _extract_tabular(self, content: dict, source: SourceMetadata) -> Tuple[List, List]:
        """Extract from CSV/tabular data."""
        rows = content.get("rows", [])
        columns = content.get("columns", [])

        # Detect what kind of data this is
        data_type = self._detect_tabular_type(columns, rows)

        if data_type == "cdr":
            return self._extract_cdr(rows, source)
        elif data_type == "bank":
            return self._extract_bank(rows, source)
        elif data_type == "cctv":
            return self._extract_cctv(rows, source)
        elif data_type in ("criminal_record", "suspect_list"):
            return self._extract_suspect_list(rows, source)
        elif data_type in ("device", "social", "evidence_log", "generic"):
            return self._extract_generic_tabular(rows, columns, source)
        else:
            return self._extract_generic_tabular(rows, columns, source)

    def _detect_tabular_type(self, columns: List[str], rows: List[dict]) -> str:
        """Detect what kind of tabular data this is using patterns first, LLM only for ambiguous cases."""
        col_names = [c.lower().replace(" ", "_") for c in columns if c is not None]
        
        # Pattern matching first — deterministic, instant
        if any("caller_number" in c or "callee_number" in c or "calling_party" in c or "called_party" in c for c in col_names):
            return "cdr"
        if any("transaction_id" in c or "account_number" in c or "debit_amount" in c or "credit_amount" in c for c in col_names):
            return "bank"
        if any("camera_id" in c for c in col_names):
            return "cctv"
        if any("case_number" in c or c == "charge" or c == "charges" or c == "criminal_charge" for c in col_names):
            return "criminal_record"
        if any("suspect_name" in c or "person_of_interest" in c for c in col_names):
            return "suspect_list"

        # Only use LLM for ambiguous data
        if self.ai and rows:
            sample = json.dumps(rows[:3], indent=2, default=str)
            prompt = f"""Analyze this tabular data and tell me what type of criminal investigation data it is.

COLUMNS: {json.dumps(columns)}

SAMPLE DATA:
{sample}

Types:
- cdr: Call Detail Records (phone calls, durations, tower locations)
- bank: Bank transactions (amounts, debits, credits, account numbers)
- cctv: CCTV surveillance logs (camera feeds, person sightings)
- device: Device extraction data (contacts, apps, messages)
- social: Social media data (posts, followers, messages)
- criminal_record: Criminal history (cases, charges, scores)
- suspect_list: List of suspects/persons of interest
- evidence_log: General evidence/investigation log

OUTPUT (JSON):
{{
  "data_type": "type_name",
  "reasoning": "why this type",
  "confidence": 0.95
}}"""
            try:
                result = self.ai.extract(
                    task="detect_data_type",
                    text=prompt,
                    source_file="type_detector",
                )
                if result.get("status") == "success" and result.get("parsed"):
                    detected = result["parsed"].get("data_type", "generic")
                    if detected in ("cdr", "bank", "cctv", "device", "social", "criminal_record", "suspect_list"):
                        return detected
            except Exception:
                pass

        return "generic"

    def _extract_cdr(self, rows: List[dict], source: SourceMetadata) -> Tuple[List, List]:
        """Extract entities and relations from CDR data."""
        entities = []
        relations = []
        phones_seen = set()
        locs_seen = set()

        # Detect column mapping using LLM if available
        mapper = ColumnMapper(ai_caller=self.ai)
        if rows:
            mapper.detect_mapping(list(rows[0].keys()), sample_rows=rows[:3])

        # Find caller and callee columns from LLM column_details or by position
        phone_cols = [c for c, f in mapper.mappings.items() if f == "phone"]
        caller_col = None
        callee_col = None

        # Try to use LLM column_details to identify caller/callee
        if hasattr(mapper, 'column_details') and mapper.column_details:
            for col, details in mapper.column_details.items():
                role = details.get("role", "")
                if role == "caller" and col in phone_cols:
                    caller_col = col
                elif role == "callee" and col in phone_cols:
                    callee_col = col

        # Fallback: use position (first phone = caller, second = callee)
        if not caller_col and phone_cols:
            caller_col = phone_cols[0]
        if not callee_col and len(phone_cols) >= 2:
            callee_col = phone_cols[1]

        for row in rows:
            # Get ALL attributes from row (nothing lost)
            all_attrs = mapper.get_all_attributes(row)

            caller = str(row.get(caller_col, "")).strip() if caller_col else ""
            callee = str(row.get(callee_col, "")).strip() if callee_col else ""
            call_date = mapper.get_timestamp(row)
            duration = mapper.get_duration(row)
            tower_loc = mapper.get_location(row)

            # Skip self-calls
            if caller and callee and caller == callee:
                callee = ""

            # Create phone entities (no duplicates)
            for phone in [caller, callee]:
                if phone and phone not in phones_seen:
                    phones_seen.add(phone)
                    entity = ExtractedEntity(
                        id=generate_id("PHONE", phone),
                        entity_type=EntityType.PHONE,
                        name=phone,
                        attributes={"phone_number": phone},
                        confidence=ConfidenceSchema(score=1.0, basis=["CDR direct record"]),
                        source=source,
                        extraction_method="code",
                    )
                    entities.append(entity)

            # Create location entity (no duplicates)
            if tower_loc:
                loc_id = generate_id("LOC", tower_loc)
                if loc_id not in locs_seen:
                    locs_seen.add(loc_id)
                    loc_entity = ExtractedEntity(
                        id=loc_id,
                        entity_type=EntityType.LOCATION,
                        name=tower_loc,
                        attributes={"tower_location": tower_loc},
                        confidence=ConfidenceSchema(score=0.8, basis=["CDR tower triangulation"]),
                        source=source,
                        extraction_method="code",
                    )
                    entities.append(loc_entity)

                # Caller was at location (with ALL row data)
                if caller:
                    caller_id = generate_id("PHONE", caller)
                    loc_rel = ExtractedRelation(
                        id=generate_id("REL", f"{caller}_LOC_{call_date}_{tower_loc}"),
                        source_entity_id=caller_id,
                        target_entity_id=loc_id,
                        relation_type=RelationType.VISITED,
                        confidence=ConfidenceSchema(score=0.7, basis=["CDR tower location"]),
                        source=source,
                        attributes={**all_attrs, "source_name": caller, "target_name": tower_loc},
                        extraction_method="code",
                        semantic_edge_type=get_semantic_edge_type(RelationType.VISITED),
                        temporal_info=extract_temporal_info(all_attrs),
                    )
                    relations.append(loc_rel)

            # Create call relation (skip self-calls)
            if caller and callee:
                rel = ExtractedRelation(
                    id=generate_id("REL", f"{caller}_CALLED_{callee}_{call_date}"),
                    source_entity_id=generate_id("PHONE", caller),
                    target_entity_id=generate_id("PHONE", callee),
                    relation_type=RelationType.CALLED,
                    confidence=ConfidenceSchema(score=1.0, basis=["CDR direct record"]),
                    source=source,
                    attributes={**all_attrs, "source_name": caller, "target_name": callee},
                    extraction_method="code",
                    semantic_edge_type=get_semantic_edge_type(RelationType.CALLED),
                    temporal_info=extract_temporal_info(all_attrs),
                )
                relations.append(rel)

        return entities, relations

    # Currency codes and abbreviations to exclude from person names
    CURRENCY_CODES = {"INR", "USD", "EUR", "GBP", "CNY", "JPY", "AED", "SAR", "KWD", "BHD", "QAR", "OMR", "NGN", "ZAR", "AUD", "CAD", "SGD", "HKD", "THB", "MYR", "IDR", "PHP", "VND", "KRW", "TRY", "RUB", "BRL", "MXN", "ARS", "CLP", "COP", "PEN", "EGP", "PKR", "BDT", "LKR", "NPR", "MMK", "KHR", "LAK"}

    def _extract_bank(self, rows: List[dict], source: SourceMetadata) -> Tuple[List, List]:
        """Extract entities and relations from bank transaction data."""
        entities = []
        relations = []
        accounts_seen = set()
        pending_persons = {}  # Track persons created via _find_or_create_person for export

        # Detect column mapping using LLM if available
        mapper = ColumnMapper(ai_caller=self.ai)
        if rows:
            mapper.detect_mapping(list(rows[0].keys()), sample_rows=rows[:3])

        for row in rows:
            account_num = mapper.get_account(row)
            # Holder name: try mapper first, fallback to raw column
            holder_name = mapper.get_name(row)
            if not holder_name:
                for col in row:
                    if "holder" in col.lower() and row[col]:
                        holder_name = str(row[col]).strip()
                        break
            txn_type = mapper.get_direction(row)
            amount = mapper.get_amount(row)
            counterparty = ""
            counterparty_name = ""
            txn_date = mapper.get_timestamp(row)
            description = mapper.get_field(row, "description", "")

            # Try to find counterparty from unmapped columns
            for col in mapper.unmapped:
                val = str(row.get(col, "")).strip()
                if not val:
                    continue
                # Account numbers: 10-18 digits (not dates or phones)
                if re.match(r'^\d{10,18}$', val) and val != account_num:
                    counterparty = val
                # Names: 1+ words with uppercase start, handles prefixes and initials
                elif re.match(r'^(?:[A-Z][a-z]+\.?\s*)*[A-Z][a-z]+$', val) or re.match(r'^[A-Z]{2,}$', val):
                    if self._is_organization_name(val):
                        # Skip org names as counterparty persons
                        continue
                    # Skip currency codes (e.g., "INR", "USD")
                    if val.upper() in self.CURRENCY_CODES:
                        continue
                    # Skip single-word uppercase values that are likely currency codes
                    if re.match(r'^[A-Z]{2,4}$', val):
                        continue
                    counterparty_name = val

            # Create account entity
            if account_num and account_num not in accounts_seen:
                accounts_seen.add(account_num)

                # Check if account holder already exists as person (skip empty names)
                if holder_name and holder_name.strip():
                    holder_entity = self._find_or_create_person(holder_name, source, pending_persons)
                    entities.append(holder_entity)

                    # Link person to account
                    person_to_acct = ExtractedRelation(
                        id=generate_id("REL", f"{holder_entity.id}_OWNS_{generate_id('ACCT', account_num)}"),
                        source_entity_id=holder_entity.id,
                        target_entity_id=generate_id("ACCT", account_num),
                        relation_type=RelationType.OWNS_ACCOUNT,
                        confidence=ConfidenceSchema(score=1.0, basis=["Bank record direct"]),
                        source=source,
                        extraction_method="code",
                        semantic_edge_type=get_semantic_edge_type(RelationType.OWNS_ACCOUNT),
                        temporal_info=None,
                        attributes={"source_name": holder_name, "target_name": f"Account {account_num[-4:]}"},
                    )
                    relations.append(person_to_acct)

                # Create account — try to get bank info from raw row
                bank_name = mapper.get_field(row, "bank", "")
                if not bank_name:
                    # Fallback: scan all columns for bank-like values
                    for col, val in row.items():
                        val_str = str(val).strip().lower()
                        if any(b in val_str for b in ["bank", "sbi", "hdfc", "icici", "axis", "kotak", "pnb", "bob", "bank of"]):
                            bank_name = str(val).strip()
                            break
                account_attrs = {
                    "account_number": account_num,
                    "account_holder": holder_name,
                }
                if bank_name:
                    account_attrs["bank"] = bank_name
                account_entity = ExtractedEntity(
                    id=generate_id("ACCT", account_num),
                    entity_type=EntityType.ACCOUNT,
                    name=f"Account {account_num[-4:]}",
                    attributes=account_attrs,
                    confidence=ConfidenceSchema(score=1.0, basis=["Bank record direct"]),
                    source=source,
                    extraction_method="code",
                )
                entities.append(account_entity)

                # Link person to account (only if holder exists and not already linked)
                if holder_name and holder_name.strip() and holder_entity is not None:
                    already_linked = any(
                        r.source_entity_id == holder_entity.id and r.target_entity_id == account_entity.id
                        for r in relations
                    )
                    if not already_linked:
                        person_to_acct = ExtractedRelation(
                            id=generate_id("REL", f"{holder_entity.id}_OWNS_{account_entity.id}"),
                            source_entity_id=holder_entity.id,
                            target_entity_id=account_entity.id,
                            relation_type=RelationType.OWNS_ACCOUNT,
                            confidence=ConfidenceSchema(score=1.0, basis=["Bank record direct"]),
                            source=source,
                            extraction_method="code",
                            semantic_edge_type=get_semantic_edge_type(RelationType.OWNS_ACCOUNT),
                            temporal_info=None,
                            attributes={"source_name": holder_name, "target_name": f"Account {account_num[-4:]}"},
                        )
                        relations.append(person_to_acct)

            # Create transfer relation
            if counterparty and counterparty != "NA" and counterparty != account_num:
                # Counterparty account
                if counterparty not in accounts_seen:
                    accounts_seen.add(counterparty)
                    cp_entity = ExtractedEntity(
                        id=generate_id("ACCT", counterparty),
                        entity_type=EntityType.ACCOUNT,
                        name=f"Account {counterparty[-4:]}",
                        attributes={
                            "account_number": counterparty,
                            "account_holder": counterparty_name,
                        },
                        confidence=ConfidenceSchema(score=1.0, basis=["Bank record direct"]),
                        source=source,
                        extraction_method="code",
                    )
                    entities.append(cp_entity)

                    if counterparty_name and counterparty_name != "Unknown":
                        cp_person = self._find_or_create_person(counterparty_name, source, pending_persons)
                        entities.append(cp_person)
                        cp_to_acct = ExtractedRelation(
                            id=generate_id("REL", f"{cp_person.id}_OWNS_{cp_entity.id}"),
                            source_entity_id=cp_person.id,
                            target_entity_id=cp_entity.id,
                            relation_type=RelationType.OWNS_ACCOUNT,
                            confidence=ConfidenceSchema(score=1.0, basis=["Bank record direct"]),
                            source=source,
                            extraction_method="code",
                            semantic_edge_type=get_semantic_edge_type(RelationType.OWNS_ACCOUNT),
                            temporal_info=None,
                            attributes={"source_name": counterparty_name, "target_name": f"Account {counterparty[-4:]}"},
                        )
                        relations.append(cp_to_acct)

                # Transfer relation
                from_acct = generate_id("ACCT", account_num)
                to_acct = generate_id("ACCT", counterparty)

                rel_type = RelationType.TRANSFERRED_TO if txn_type == "DEBIT" else RelationType.RECEIVED_FROM
                transfer = ExtractedRelation(
                    id=generate_id("REL", f"{from_acct}_{rel_type.value}_{to_acct}_{txn_date}"),
                    source_entity_id=from_acct,
                    target_entity_id=to_acct,
                    relation_type=rel_type,
                    confidence=ConfidenceSchema(score=1.0, basis=["Bank record direct"]),
                    source=source,
                    attributes={
                        "amount": amount,
                        "currency": mapper.get_field(row, "currency", ""),
                        "date": txn_date,
                        "description": description,
                        "is_flagged": "flagged" in description.lower(),
                        "source_name": f"Account {account_num[-4:]}",
                        "target_name": f"Account {counterparty[-4:]}",
                    },
                    extraction_method="code",
                    semantic_edge_type=get_semantic_edge_type(rel_type),
                    temporal_info=extract_temporal_info({"date": txn_date}),
                )
                relations.append(transfer)

        return entities, relations

    def _extract_cctv(self, rows: List[dict], source: SourceMetadata) -> Tuple[List, List]:
        """Extract entities and relations from CCTV log."""
        entities = []
        relations = []
        persons_seen = set()

        for row in rows:
            timestamp = str(row.get("timestamp", "")).strip()
            location = str(row.get("location", "")).strip()
            camera_id = str(row.get("camera_id", "")).strip()
            action = str(row.get("action", "")).strip()
            description = str(row.get("person_description", row.get("description", ""))).strip()

            # Extract person from description
            person_key = self._extract_person_from_description(description)
            if person_key and person_key not in persons_seen:
                persons_seen.add(person_key)
                entity = ExtractedEntity(
                    id=generate_id("PERSON", person_key),
                    entity_type=EntityType.PERSON,
                    name=person_key,
                    attributes={
                        "description": description,
                        "cctv_camera": camera_id,
                        "location": location,
                    },
                    confidence=ConfidenceSchema(score=0.6, basis=["CCTV visual description"]),
                    source=source,
                    extraction_method="code",
                )
                entities.append(entity)

            # Location entity
            if location:
                loc_id = generate_id("LOC", location)
                if loc_id not in self.entity_index:
                    loc_entity = ExtractedEntity(
                        id=loc_id,
                        entity_type=EntityType.LOCATION,
                        name=location,
                        attributes={"location_name": location},
                        confidence=ConfidenceSchema(score=0.9, basis=["CCTV location tag"]),
                        source=source,
                        extraction_method="code",
                    )
                    entities.append(loc_entity)
                else:
                    # Same location re-observed by this file — credit it.
                    self._observe(self.entity_index[loc_id], source)

            # Visit relation
            if person_key and location:
                person_id = generate_id("PERSON", person_key)
                loc_id = generate_id("LOC", location)
                visit_rel = ExtractedRelation(
                    id=generate_id("REL", f"{person_id}_VISITED_{loc_id}_{timestamp}"),
                    source_entity_id=person_id,
                    target_entity_id=loc_id,
                    relation_type=RelationType.VISITED,
                    confidence=ConfidenceSchema(score=0.7, basis=["CCTV visual record"]),
                    source=source,
                    attributes={
                        "timestamp": timestamp,
                        "camera_id": camera_id,
                        "action": action,
                        "duration_minutes": str(row.get("duration_minutes", "")).strip(),
                        "source_name": person_key,
                        "target_name": location,
                    },
                    extraction_method="code",
                    semantic_edge_type=get_semantic_edge_type(RelationType.VISITED),
                    temporal_info=extract_temporal_info({"timestamp": timestamp}),
                )
                relations.append(visit_rel)

        return entities, relations

    def _extract_image(self, content: dict, source: SourceMetadata) -> Tuple[List, List]:
        """Extract from image — uses OCR text if available, otherwise marks for vision model.

        Image-class gate: slides/diagrams and text reports rendered as images
        produce NO entities (their OCR text mints junk persons like 'GPU' and
        'Subject Moe'); only photographic evidence OCRs into the entity stream.
        """
        entities = []
        relations = []
        file_path = content.get("file_path", "")
        file_name = os.path.basename(file_path) if file_path else "unknown"
        ocr_text = content.get("ocr_text", "")

        image_class = content.get("image_class", "")
        if not image_class:
            image_class, _, _ = classify_image_evidence(file_name, ocr_text, file_path=file_path or None)

        # Slides/diagrams are not evidence text at all — their OCR ('Entity
        # Tracker', 'Kalman Filter', 'GPU') mints dozens of fake persons.
        if image_class == "SLIDE_OR_DIAGRAM":
            return entities, relations

        # If OCR produced useful text, extract from it (same as text extraction)
        if ocr_text and not content.get("requires_vision_model", True):
            # Treat OCR text like any other text content
            text_content = {"content": ocr_text}
            text_entities, text_relations = self._extract_text(text_content, source)
            # OCR of text reports/screenshots mints junk persons ('Subject Moe'
            # from 'Subject Male', 'CAM_KE7' from an overlay camera id).
            text_entities = [
                e for e in text_entities
                if e.entity_type != EntityType.PERSON or not is_junk_person_name(e.name)
            ]
            entities.extend(text_entities)
            relations.extend(text_relations)

        # Always extract filename metadata too (camera ID, date, etc.)
        loc_match = re.search(r'(?:CCTV|CAM|cam|cctv)[_-]?(\w+)', file_name, re.IGNORECASE)
        if loc_match:
            cam_id = loc_match.group(0)
            entities.append(ExtractedEntity(
                id=generate_id("CAM", cam_id),
                entity_type=EntityType.DEVICE,
                name=cam_id,
                attributes={"camera_id": cam_id, "source_image": file_name},
                confidence=ConfidenceSchema(score=0.6, basis=["Camera ID from filename"]),
                source=source,
                extraction_method="code",
            ))

        date_match = re.search(r'(\d{4}[-_]?\d{2}[-_]?\d{2})', file_name)
        if date_match:
            entities.append(ExtractedEntity(
                id=generate_id("DATE", date_match.group(1)),
                entity_type=EntityType.DATE,
                name=date_match.group(1),
                attributes={"date": date_match.group(1), "source_image": file_name},
                confidence=ConfidenceSchema(score=0.7, basis=["Date from filename"]),
                source=source,
                extraction_method="code",
            ))

        # Filename-derived subject (doc 11 §5 Priority 5): a name-shaped image
        # filename ('suresh.jpg') mints a weak PERSON the face stage can link
        # its detected face to — only on photographic evidence, only when the
        # file's own text named nobody, and only when the name is name-shaped
        # (surveillance words, digits and ID-document files never mint).
        person_name = filename_name(file_name) if should_mint_filename_person(image_class) else ""
        person_from_content = any(
            e.entity_type == EntityType.PERSON for e in entities
        )
        if person_name and not person_from_content:
            entities.append(ExtractedEntity(
                id=generate_id("PERSON", person_name),
                entity_type=EntityType.PERSON,
                name=person_name,
                attributes={"minted_from": "filename", "source_image": file_name},
                confidence=ConfidenceSchema(
                    score=0.55,
                    basis=["Person name inferred from filename (doc 11 filename context)"],
                ),
                source=source,
                extraction_method="code",
            ))

        # If image requires vision model (no useful OCR text), tag it
        if content.get("requires_vision_model", False):
            entities.append(ExtractedEntity(
                id=generate_id("IMG", file_name),
                entity_type=EntityType.EVENT,
                name=f"Unprocessed image: {file_name}",
                attributes={
                    "source_image": file_name,
                    "requires_vision_model": True,
                    "ocr_quality_score": content.get("ocr_quality_score", 0),
                    "ocr_words_extracted": content.get("ocr_words_extracted", 0),
                },
                confidence=ConfidenceSchema(score=0.0, basis=["Image pending vision model processing"]),
                source=source,
                extraction_method="code",
            ))

        return entities, relations

    def _extract_pdf(self, content: dict, source: SourceMetadata) -> Tuple[List, List]:
        """Extract from PDF: text + embedded tables."""
        entities = []
        relations = []

        # Extract from text content
        text_content = {"content": content.get("content", ""), "format": "text"}
        text_entities, text_relations = self._extract_text(text_content, source)
        entities.extend(text_entities)
        relations.extend(text_relations)

        # Extract from embedded tables
        for table in content.get("tables", []):
            table_content = {
                "format": "csv",
                "columns": table.get("columns", []),
                "row_count": table.get("row_count", 0),
                "rows": table.get("rows", []),
            }
            table_entities, table_relations = self._extract_tabular(table_content, source)
            # Tag table entities with page info
            for e in table_entities:
                e.attributes["from_table"] = True
                e.attributes["table_page"] = table.get("page", "?")
            entities.extend(table_entities)
            relations.extend(table_relations)

        return entities, relations

    def _extract_docx(self, content: dict, source: SourceMetadata) -> Tuple[List, List]:
        """Extract from DOCX: text + embedded tables."""
        entities = []
        relations = []

        # Extract from text content
        text_content = {"content": content.get("content", ""), "format": "text"}
        text_entities, text_relations = self._extract_text(text_content, source)
        entities.extend(text_entities)
        relations.extend(text_relations)

        # Extract from embedded tables
        for table in content.get("tables", []):
            table_content = {
                "format": "csv",
                "columns": table.get("columns", []),
                "row_count": table.get("row_count", 0),
                "rows": table.get("rows", []),
            }
            table_entities, table_relations = self._extract_tabular(table_content, source)
            for e in table_entities:
                e.attributes["from_table"] = True
                e.attributes["table_index"] = table.get("table_index", "?")
            entities.extend(table_entities)
            relations.extend(table_relations)

        return entities, relations

    def _extract_text(self, content: dict, source: SourceMetadata) -> Tuple[List, List]:
        """Extract from plain text using code heuristics + optional LLM."""
        text = content.get("content", "")
        entities = []
        relations = []

        # Extract code-based entities first (phones, amounts, dates, names)
        phones = re.findall(r'\b\d{10}\b', text)
        for phone in set(phones):
            entity = ExtractedEntity(
                id=generate_id("PHONE", phone),
                entity_type=EntityType.PHONE,
                name=phone,
                attributes={"phone_number": phone},
                confidence=ConfidenceSchema(score=0.9, basis=["Phone number found in text"]),
                source=source,
                extraction_method="code",
            )
            entities.append(entity)

        # Extract amounts
        amounts = re.findall(r'Rs\.\s*[\d,]+|₹[\d,]+|INR\s*[\d,]+', text)
        for amount in set(amounts):
            entity = ExtractedEntity(
                id=generate_id("AMOUNT", amount),
                entity_type=EntityType.AMOUNT,
                name=amount,
                attributes={"amount": amount, "type": "financial_amount"},
                confidence=ConfidenceSchema(score=0.8, basis=["Amount found in text"]),
                source=source,
                extraction_method="code",
            )
            entities.append(entity)

        # Extract names (common Indian patterns)
        # NOTE: separators use [ \t] (not \s) so matches never cross line
        # boundaries — otherwise "Name: Unknown\nFather's Name: ..." would
        # capture "Unknown\nFather" and yield a bogus "Unknown" person.
        name_patterns = [
            r'(?:Name|name|NAME)[: \t]+([A-Z][a-z]+(?:[ \t]+[A-Z][a-z]+)+)',
            r'(?:Mr|Mr\.|Ms|Ms\.|Mrs|Mrs\.)[ \t]+([A-Z][a-z]+(?:[ \t]+[A-Z][a-z]+)+)',
            r'(?:son of|daughter of|wife of)[ \t]+([A-Z][a-z]+(?:[ \t]+[A-Z][a-z]+)+)',
            # Officer titles: "SI Vikram Rathore" / "Sub Inspector Vikram Rathore".
            # spaCy tags these as ORG; emitting the same string as PERSON lets
            # reconcile_entity_types settle it (PERSON wins exact-name conflict).
            r'\b(SI[ \t]+[A-Z][a-z]+(?:[ \t]+[A-Z][a-z]+)+)',
            r'\b(Sub Inspector[ \t]+[A-Z][a-z]+(?:[ \t]+[A-Z][a-z]+)+)',
        ]
        for pattern in name_patterns:
            matches = re.findall(pattern, text, re.MULTILINE)
            for name in set(matches):
                # Clean: remove newlines, extra whitespace, trailing junk
                name = name.split('\n')[0].strip()
                if len(name) > 3 and not name.isdigit():
                    entity = ExtractedEntity(
                        id=generate_id("PERSON", name),
                        entity_type=EntityType.PERSON,
                        name=name,
                        aliases=[name],
                        attributes={"extracted_from": "text"},
                        confidence=ConfidenceSchema(score=0.7, basis=["Name extracted from text"]),
                        source=source,
                        extraction_method="code",
                    )
                    entities.append(entity)

        # Name:/Phone: field association (FIR-style records).
        # Name and phone are extracted independently, so the person from a
        # "Name:" line never receives the "Phone:" a few lines below in the
        # same record block — key persons end up with phones=[] and no
        # OWNS_PHONE edge. Walk lines top-down: a Name line arms the pending
        # record, blank lines disarm it, and the next Phone/Mobile line with
        # 10 digits attaches to the armed record. "Father's Name:" does not
        # arm (line does not start with the Name label); "Phone: Unknown"
        # never fabricates a number.
        pending_person_name = None
        for line in text.splitlines():
            stripped = line.strip()
            if not stripped:
                pending_person_name = None
                continue
            name_m = re.match(r"^Name:\s*(.+)$", stripped)
            if name_m:
                pending_person_name = name_m.group(1).strip()
                continue
            phone_m = re.match(r"^(?:Phone|Mobile):\s*(\d{10})\b", stripped)
            if phone_m and pending_person_name:
                pid = generate_id("PERSON", pending_person_name)
                for entity in entities:
                    if entity.id == pid and entity.entity_type == EntityType.PERSON:
                        entity.attributes.setdefault("phone", phone_m.group(1))
                        break
                pending_person_name = None

        # Extract dates
        dates = re.findall(r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b', text)
        for date in set(dates):
            entity = ExtractedEntity(
                id=generate_id("DATE", date),
                entity_type=EntityType.DATE,
                name=date,
                attributes={"date": date, "type": "date_reference"},
                confidence=ConfidenceSchema(score=0.9, basis=["Date found in text"]),
                source=source,
                extraction_method="code",
            )
            entities.append(entity)

        # NLP/NER extraction using spaCy — supplements regex-based extraction
        nlp_entities = self._extract_ner_spacy(text, source)
        entities.extend(nlp_entities)

        # Hindi NER — extracts from Devanagari script text
        hindi_entities = self._extract_hindi_ner(text, source)
        entities.extend(hindi_entities)

        # Role assignment — detect accused/victim/witness/complainant from context
        roles = self._assign_entity_roles(text, entities)
        for entity in entities:
            if entity.id in roles:
                entity.attributes["role"] = roles[entity.id]

        # Narrative relation extraction — pull directed links from prose
        # (called / visited / transferred / owns / associated / etc.)
        narrative_relations = self._extract_narrative_relations(text, entities, source)
        relations.extend(narrative_relations)

        # Low-confidence filter: PERSON entities with confidence < 0.5 → demoted
        entities, demoted = self._apply_low_confidence_filter(entities)
        for d in demoted:
            d.attributes["demoted_reason"] = "low_confidence"
            d.attributes["original_type"] = d.entity_type.value

        # LLM extraction — adds structured entities on top of code-extracted ones
        if self.ai and len(text) >= self.llm_extraction_threshold:
            task = self._detect_text_task(text)
            if task:
                try:
                    llm_result = self.ai.extract(
                        task=task,
                        text=text[:self.max_text_length],
                        source_file=source.file_name,
                    )
                    if llm_result.get("status") == "success" and llm_result.get("parsed"):
                        parsed = llm_result["parsed"]
                        llm_entities, llm_relations = self._convert_llm_parsed(parsed, source, llm_result)
                        entities.extend(llm_entities)
                        relations.extend(llm_relations)
                except Exception as e:
                    print(f"[EXTRACTION] LLM extraction failed for {source.file_name}: {e}")

        return entities, relations

    def _extract_json(self, content: dict, source: SourceMetadata) -> Tuple[List, List]:
        """Extract from JSON data — handles dicts and lists."""
        data = content.get("data", {})

        if isinstance(data, dict):
            if "device_info" in data:
                return self._extract_device_data(data, source)
            elif "profile" in data or "posts" in data:
                return self._extract_social_data(data, source)
            elif "criminal_history" in data:
                return self._extract_criminal_history(data, source)
            else:
                # Unknown dict — extract what we can
                return self._extract_generic_json_dict(data, source)

        if isinstance(data, list) and data:
            # JSON array of records — treat like tabular
            return self._extract_generic_json_list(data, source)

        return [], []

    def _extract_generic_json_dict(self, data: dict, source: SourceMetadata) -> Tuple[List, List]:
        """Extract entities from an unknown JSON dict by scanning all string values."""
        entities = []
        relations = []

        def scan_value(val, path=""):
            if isinstance(val, str) and val.strip():
                # Phone
                if re.match(r'^\d{10}$', val.strip()):
                    entities.append(ExtractedEntity(
                        id=generate_id("PHONE", val.strip()),
                        entity_type=EntityType.PHONE,
                        name=val.strip(),
                        attributes={"phone_number": val.strip(), "path": path},
                        confidence=ConfidenceSchema(score=0.8, basis=["Phone in JSON"]),
                        source=source, extraction_method="code",
                    ))
                # Name-like
                elif re.match(r'^[A-Z][a-z]+ [A-Z][a-z]+', val.strip()) and len(val.strip()) < 60:
                    entities.append(ExtractedEntity(
                        id=generate_id("PERSON", val.strip()),
                        entity_type=EntityType.PERSON,
                        name=val.strip(),
                        attributes={"path": path},
                        confidence=ConfidenceSchema(score=0.5, basis=["Name pattern in JSON"]),
                        source=source, extraction_method="code",
                    ))
            elif isinstance(val, dict):
                for k, v in val.items():
                    scan_value(v, f"{path}.{k}" if path else k)
            elif isinstance(val, list):
                for i, item in enumerate(val):
                    scan_value(item, f"{path}[{i}]")

        scan_value(data)
        return entities, relations

    def _extract_generic_json_list(self, data: list, source: SourceMetadata) -> Tuple[List, List]:
        """Extract entities from a JSON array of records."""
        entities = []
        relations = []

        # Detect if records have common keys
        if data and isinstance(data[0], dict):
            all_keys = set()
            for item in data:
                all_keys.update(item.keys())

            # Use column mapper to map JSON keys
            mapper = ColumnMapper(ai_caller=self.ai)
            mapper.detect_mapping(list(all_keys), sample_rows=data[:3])

            phones_seen = set()
            persons_seen = set()

            for row in data:
                attrs = mapper.get_all_attributes(row)

                # Phone entities
                phone = mapper.get_phone(row)
                if phone and phone not in phones_seen:
                    phones_seen.add(phone)
                    entities.append(ExtractedEntity(
                        id=generate_id("PHONE", phone),
                        entity_type=EntityType.PHONE,
                        name=phone,
                        attributes={"phone_number": phone},
                        confidence=ConfidenceSchema(score=0.8, basis=["Phone in JSON list"]),
                        source=source, extraction_method="code",
                    ))

                # Person entities
                name = mapper.get_name(row)
                if name and name not in persons_seen and re.match(r'^[A-Z][a-z]+ [A-Z][a-z]+', name):
                    persons_seen.add(name)
                    entities.append(ExtractedEntity(
                        id=generate_id("PERSON", name),
                        entity_type=EntityType.PERSON,
                        name=name,
                        aliases=[name],
                        attributes=attrs,
                        confidence=ConfidenceSchema(score=0.6, basis=["Name in JSON list"]),
                        source=source, extraction_method="code",
                    ))

        return entities, relations

    def _extract_device_data(self, data: dict, source: SourceMetadata) -> Tuple[List, List]:
        """Extract from device extraction JSON."""
        entities = []
        relations = []

        device = data.get("device_info", {})
        owner = device.get("device_id", "unknown")

        # Device entity
        device_id_str = device.get("imei", "") or device.get("device_id", "") or device.get("serial", "")
        device_entity = ExtractedEntity(
            id=generate_id("DEVICE", device_id_str),
            entity_type=EntityType.DEVICE,
            name=f"Device: {device.get('brand', '')} {device.get('model', '')}",
            attributes=device,
            confidence=ConfidenceSchema(score=1.0, basis=["Device extraction direct"]),
            source=source,
            extraction_method="code",
        )
        entities.append(device_entity)

        # Contacts
        for contact in data.get("contacts", []):
            name = contact.get("name", "")
            phone = contact.get("phone", "")

            if phone:
                phone_entity = ExtractedEntity(
                    id=generate_id("PHONE", phone),
                    entity_type=EntityType.PHONE,
                    name=phone,
                    attributes={"phone_number": phone, "contact_name": name},
                    confidence=ConfidenceSchema(score=1.0, basis=["Device contact direct"]),
                    source=source,
                    extraction_method="code",
                )
                entities.append(phone_entity)

            # Structured alias claim: {"relation": "alias", "phone": ...}
            # means this name IS an alias of whoever already owns that phone
            # (file17: {"name": "Bhai", "phone": "9876543210", "relation":
            # "alias"} — Rakesh's phone). Attach it to that person instead of
            # minting a standalone identity. Only when a known PERSON entity
            # already carries this phone; otherwise fall through to normal
            # contact handling.
            alias_of = None
            if name and phone and str(contact.get("relation", "")).lower() == "alias":
                for ex in self.entity_index.values():
                    if (hasattr(ex, "entity_type") and hasattr(ex.entity_type, "value")
                            and ex.entity_type.value == "PERSON"
                            and phone in {ex.attributes.get("phone"),
                                          ex.attributes.get("phone_number")}):
                        alias_of = ex
                        break

            if alias_of is not None:
                if name not in alias_of.aliases:
                    alias_of.aliases.append(name)
                self._observe(alias_of, source)
                if phone:
                    rel = ExtractedRelation(
                        id=generate_id("REL", f"{alias_of.id}_HAS_PHONE_{phone_entity.id}"),
                        source_entity_id=alias_of.id,
                        target_entity_id=phone_entity.id,
                        relation_type=RelationType.ASSOCIATED_WITH,
                        confidence=ConfidenceSchema(score=1.0, basis=["Device contact direct"]),
                        source=source,
                        attributes={"contact_name": name, "relation": "alias",
                                    "source_name": alias_of.name, "target_name": phone},
                        extraction_method="code",
                        semantic_edge_type=get_semantic_edge_type(RelationType.ASSOCIATED_WITH),
                        temporal_info=None,
                    )
                    relations.append(rel)
            elif name:
                # Location check first (e.g. "Shop - Lajpat Nagar" is not a person)
                _loc_contact = re.compile(
                    r"\b(Shop|Store|Showroom|Office|Market|Nagar|Bagh|Phase|"
                    r"Sector|Block|Complex|Mall|Building|Road|Street|Lane|"
                    r"Metro|Station|Tower|Colony|Enclave|Extension)\b",
                    re.IGNORECASE,
                )
                if _loc_contact.search(name):
                    entity_type = EntityType.LOCATION
                    prefix = "LOC"
                elif self._is_organization_name(name):
                    entity_type = EntityType.ORGANIZATION
                    prefix = "ORG"
                else:
                    entity_type = EntityType.PERSON
                    prefix = "PERSON"
                # Keep the contact's phone on the person: get_entity_phones()
                # reads attributes.phone, so without this a device contact's
                # number never reaches resolution's phone signals (it exists
                # only on the HAS_PHONE relation's PHONE node).
                person_attrs = {"contact_in_device": owner}
                if phone:
                    person_attrs["phone"] = phone
                person_entity = ExtractedEntity(
                    id=generate_id(prefix, name),
                    entity_type=entity_type,
                    name=name,
                    aliases=[name],
                    attributes=person_attrs,
                    confidence=ConfidenceSchema(score=0.9, basis=["Device contact"]),
                    source=source,
                    extraction_method="code",
                )
                entities.append(person_entity)

                # Link person to phone
                if phone:
                    rel = ExtractedRelation(
                        id=generate_id("REL", f"{person_entity.id}_HAS_PHONE_{phone_entity.id}"),
                        source_entity_id=person_entity.id,
                        target_entity_id=phone_entity.id,
                        relation_type=RelationType.ASSOCIATED_WITH,
                        confidence=ConfidenceSchema(score=1.0, basis=["Device contact direct"]),
                        source=source,
                        attributes={"contact_name": name, "relation": contact.get("relation", ""), "source_name": name, "target_name": phone},
                        extraction_method="code",
                        semantic_edge_type=get_semantic_edge_type(RelationType.ASSOCIATED_WITH),
                        temporal_info=None,
                    )
                    relations.append(rel)

        # WhatsApp messages
        for chat in data.get("whatsapp_chats", []):
            contact_name = chat.get("contact", "")
            for msg in chat.get("messages", []):
                sender = msg.get("from", "")
                text = msg.get("text", "")
                date = msg.get("date", "")
                time = msg.get("time", "")

                # Extract mentioned persons from message text — only proper names (2+ words)
                mentioned_names = re.findall(r'\b([A-Z][a-z]{2,}\s+[A-Z][a-z]{2,})\b', text)
                # Blacklist: common Hindi/English words that match name pattern
                _name_blacklist = {
                    "Theek", "Haan", "Thanks", "Received", "Police", "SIM", "UPI",
                    "Total", "Payment", "Check", "Safe", "Account", "Fraud", "Fake",
                    "First", "Last", "Next", "After", "Before", "Call", "Send",
                    "Good", "Very", "Much", "More", "Some", "What", "When", "Where",
                    "This", "That", "With", "From", "Have", "Been", "Will", "Your",
                    "Meeting", "Report", "Photo", "Video", "File", "Data", "Info",
                    "Karol Bagh", "Lajpat Nagar", "Nehru Nagar", "Connaught Place",
                    "Sharma Properties", "RK Telecom", "State Bank",
                }
                # Location patterns — skip these as person names
                _location_patterns = [
                    r"Bagh$", r"Nagar$", r"Place$", r"Colony$", r"Market$",
                    r"Delhi$", r"Goa$", r"UP$", r"Maharashtra$",
                ]
                for name in mentioned_names:
                    if name not in _name_blacklist and not any(w in name for w in ["Aadhaar", "Screenshot"]):
                        # Check if it's a location
                        is_location = any(re.search(pat, name) for pat in _location_patterns)
                        if is_location:
                            continue
                        is_org = self._is_organization_name(name)
                        entity_type = EntityType.ORGANIZATION if is_org else EntityType.PERSON
                        prefix = "ORG" if is_org else "PERSON"
                        entity = ExtractedEntity(
                            id=generate_id(prefix, name),
                            entity_type=entity_type,
                            name=name,
                            attributes={"mentioned_in_chat": contact_name, "date": date},
                            confidence=ConfidenceSchema(score=0.5, basis=["Name mentioned in chat"]),
                            source=source,
                            extraction_method="code",
                        )
                        entities.append(entity)

        # Photos — extract EXIF location if available, skip placeholder entities
        for photo in data.get("photos", []):
            exif_loc = photo.get("exif_location", "")
            if exif_loc:
                entities.append(ExtractedEntity(
                    id=generate_id("LOC", exif_loc),
                    entity_type=EntityType.LOCATION,
                    name=exif_loc,
                    attributes={"source": "photo_exif", "filename": photo.get("filename", "")},
                    confidence=ConfidenceSchema(score=0.6, basis=["EXIF location from photo"]),
                    source=source,
                    extraction_method="code",
                ))

        return entities, relations

    def _extract_social_data(self, data: dict, source: SourceMetadata) -> Tuple[List, List]:
        """Extract from social media data."""
        entities = []
        relations = []

        profile = data.get("profile", {})
        username = data.get("username", "")

        # Build username -> person_id lookup (username resolves to display_name-based ID)
        username_to_person_id = {}

        # Profile entity
        if username:
            # profile.name may be missing OR explicitly empty — fall back to username
            display_name = (profile.get("name") or "").strip() or username
            person_id = generate_id("PERSON", display_name)
            username_to_person_id[username] = person_id
            person = ExtractedEntity(
                id=person_id,
                entity_type=EntityType.PERSON,
                name=display_name,
                aliases=[username],
                attributes={
                    "social_username": username,
                    "platform": data.get("platform", ""),
                    "bio": profile.get("bio", ""),
                    "followers": profile.get("followers", 0),
                    "posts_count": profile.get("posts_count", 0),
                },
                confidence=ConfidenceSchema(score=0.9, basis=["Social media profile direct"]),
                source=source,
                extraction_method="code",
            )
            entities.append(person)

        # Posts
        for post in data.get("posts", []):
            post_entity = ExtractedEntity(
                id=generate_id("POST", post.get("post_id", "")),
                entity_type=EntityType.EVENT,
                name=f"Post: {post.get('caption', '')[:50]}",
                attributes={
                    "post_id": post.get("post_id", ""),
                    "date": post.get("date", ""),
                    "caption": post.get("caption", ""),
                    "location": post.get("location", ""),
                    "likes": post.get("likes", 0),
                    "comments": post.get("comments", 0),
                    "social_type": "post",
                },
                confidence=ConfidenceSchema(score=1.0, basis=["Social media post direct"]),
                source=source,
                extraction_method="code",
            )
            entities.append(post_entity)

            # Location entity
            loc = post.get("location", "")
            if loc:
                loc_entity = ExtractedEntity(
                    id=generate_id("LOC", loc),
                    entity_type=EntityType.LOCATION,
                    name=loc,
                    attributes={"location_name": loc, "source": "social_media"},
                    confidence=ConfidenceSchema(score=0.8, basis=["Social media location tag"]),
                    source=source,
                    extraction_method="code",
                )
                entities.append(loc_entity)

                # Author visited location — use display_name-based ID
                author_id = username_to_person_id.get(username, generate_id("PERSON", username))
                if username:
                    visit_rel = ExtractedRelation(
                        id=generate_id("REL", f"{author_id}_POSTED_AT_{generate_id('LOC', loc)}_{post.get('date', '')}"),
                        source_entity_id=author_id,
                        target_entity_id=generate_id("LOC", loc),
                        relation_type=RelationType.VISITED,
                        confidence=ConfidenceSchema(score=0.7, basis=["Social media location tag"]),
                        source=source,
                        attributes={"date": post.get("date", ""), "post_id": post.get("post_id", ""), "source_name": username, "target_name": loc},
                        extraction_method="code",
                        semantic_edge_type=get_semantic_edge_type(RelationType.VISITED),
                        temporal_info=extract_temporal_info({"date": post.get("date", "")}),
                    )
                    relations.append(visit_rel)

        # Direct messages — resolve contact username to their display_name-based person ID
        # Build a reverse lookup: if contact is someone we know, map to their person_id
        dm_threads_seen = set()

        # Build reverse lookup from ALL known person entities (including those from other files)
        all_person_username_lookup = {}
        for entity in self.entity_index.values():
            if hasattr(entity, 'entity_type') and hasattr(entity.entity_type, 'value') and entity.entity_type.value == "PERSON":
                social_uname = entity.attributes.get("social_username", "")
                if social_uname:
                    all_person_username_lookup[social_uname] = entity.id
                # Also check aliases for username patterns
                for alias in entity.aliases:
                    if alias and re.match(r'^[a-z_]+$', alias) and alias != entity.name:
                        all_person_username_lookup[alias] = entity.id

        for dm in data.get("direct_messages", []):
            contact = dm.get("contact", "")

            # Create ONE relation per DM thread (not per message)
            if username and contact:
                thread_key = f"{username}_{contact}"
                if thread_key not in dm_threads_seen:
                    dm_threads_seen.add(thread_key)
                    msg_count = len(dm.get("messages", []))
                    first_date = dm.get("messages", [{}])[0].get("date", "") if dm.get("messages") else ""
                    # Resolve both source and target to display_name-based IDs
                    source_id = username_to_person_id.get(username, generate_id("PERSON", username))
                    # Try to resolve contact username to known person entity
                    target_id = all_person_username_lookup.get(contact, generate_id("PERSON", contact))
                    comm_rel = ExtractedRelation(
                        id=generate_id("REL", f"{source_id}_DM_{target_id}"),
                        source_entity_id=source_id,
                        target_entity_id=target_id,
                        relation_type=RelationType.MESSED,
                        confidence=ConfidenceSchema(score=1.0, basis=["Social media DM direct"]),
                        source=source,
                        attributes={"message_count": msg_count, "first_date": first_date, "platform": data.get("platform", ""), "source_name": username, "target_name": contact},
                        extraction_method="code",
                        semantic_edge_type=get_semantic_edge_type(RelationType.MESSED),
                        temporal_info=extract_temporal_info({"date": first_date}),
                    )
                    relations.append(comm_rel)

        return entities, relations

    def _extract_criminal_history(self, data: dict, source: SourceMetadata) -> Tuple[List, List]:
        """Extract from criminal history JSON."""
        entities = []
        relations = []

        for person_data in data.get("criminal_history", []):
            name = person_data.get("name", "")
            phone = person_data.get("phone", "")
            criminal_score = person_data.get("criminal_score", 0)

            if name:
                person = ExtractedEntity(
                    id=generate_id("PERSON", name),
                    entity_type=EntityType.PERSON,
                    name=name,
                    aliases=[name],
                    attributes={
                        "criminal_score": criminal_score,
                        "phone": phone,
                        "previous_cases": len(person_data.get("previous_cases", [])),
                        "notes": person_data.get("notes", ""),
                    },
                    confidence=ConfidenceSchema(score=1.0, basis=["Criminal record direct"]),
                    source=source,
                    extraction_method="code",
                )
                entities.append(person)

            # Previous cases
            for case in person_data.get("previous_cases", []):
                case_entity = ExtractedEntity(
                    id=generate_id("CASE", case.get("case_number", "")),
                    entity_type=EntityType.EVENT,
                    name=f"Case: {case.get('case_number', '')}",
                    attributes=case,
                    confidence=ConfidenceSchema(score=1.0, basis=["Case record direct"]),
                    source=source,
                    extraction_method="code",
                )
                entities.append(case_entity)

                # Suspect relation
                if name:
                    suspect_rel = ExtractedRelation(
                        id=generate_id("REL", f"{generate_id('PERSON', name)}_SUSPECT_{case_entity.id}"),
                        source_entity_id=generate_id("PERSON", name),
                        target_entity_id=case_entity.id,
                        relation_type=RelationType.SUSPECT_OF,
                        confidence=ConfidenceSchema(score=1.0, basis=["Case record direct"]),
                        source=source,
                        attributes={"charge": case.get("charge", ""), "status": case.get("status", ""), "source_name": name, "target_name": case.get("case_number", "")},
                        extraction_method="code",
                        semantic_edge_type=get_semantic_edge_type(RelationType.SUSPECT_OF),
                        temporal_info=None,
                    )
                    relations.append(suspect_rel)

        # Known associations
        # Build ID->name map from criminal_history entries
        id_to_name = {}
        for person_data in data.get("criminal_history", []):
            pid = person_data.get("person_id", person_data.get("id", ""))
            pname = person_data.get("name", "")
            if pid and pname:
                id_to_name[pid.upper()] = pname
                id_to_name[pid.upper().replace("PERSON_", "")] = pname

        # Track persons created from known_associations for export
        pending_assoc_persons = {}
        for assoc in data.get("known_associations", []):
            p1_raw = assoc.get("person1", "")
            p2_raw = assoc.get("person2", "")
            relationship = assoc.get("relationship", "")
            confidence = assoc.get("confidence", 0)

            # Resolve to actual names
            p1 = id_to_name.get(p1_raw.upper(), id_to_name.get(p1_raw.replace("PERSON_", "").upper(), p1_raw))
            p2 = id_to_name.get(p2_raw.upper(), id_to_name.get(p2_raw.replace("PERSON_", "").upper(), p2_raw))

            # Create entities for associates if not already created
            for person_name in [p1, p2]:
                if person_name:
                    person_id = generate_id("PERSON", person_name)
                    if person_id not in self.entity_index:
                        person_entity = ExtractedEntity(
                            id=person_id,
                            entity_type=EntityType.PERSON,
                            name=person_name,
                            aliases=[person_name],
                            attributes={"extracted_from": "criminal_history_association"},
                            confidence=ConfidenceSchema(score=confidence, basis=["Criminal history association"]),
                            source=source,
                            extraction_method="code",
                        )
                        entities.append(person_entity)
                        pending_assoc_persons[person_name] = person_entity
                    else:
                        self._observe(self.entity_index[person_id], source)

            # Map relationship string to RelationType
            rel_type = RelationType.ASSOCIATE_OF
            if "family" in relationship.lower() or "brother" in relationship.lower() or "sister" in relationship.lower():
                rel_type = RelationType.FAMILY_OF
            elif "friend" in relationship.lower():
                rel_type = RelationType.FRIEND_OF
            elif "business" in relationship.lower() or "fraud" in relationship.lower():
                rel_type = RelationType.WORKS_WITH

            rel = ExtractedRelation(
                id=generate_id("REL", f"{p1}_{rel_type.value}_{p2}"),
                source_entity_id=generate_id("PERSON", p1),
                target_entity_id=generate_id("PERSON", p2),
                relation_type=rel_type,
                confidence=ConfidenceSchema(score=confidence, basis=[f"Criminal history: {relationship}"]),
                source=source,
                attributes={"relationship": relationship, "source_name": p1, "target_name": p2},
                extraction_method="code",
                semantic_edge_type=get_semantic_edge_type(rel_type),
                temporal_info=None,
            )
            relations.append(rel)

        return entities, relations

    def _extract_suspect_list(self, rows: List[dict], source: SourceMetadata) -> Tuple[List, List]:
        """Extract from suspect/criminal record lists."""
        entities = []
        relations = []
        persons_seen = set()
        phones_seen = set()

        mapper = ColumnMapper(ai_caller=self.ai)
        if rows:
            mapper.detect_mapping(list(rows[0].keys()), sample_rows=rows[:3])

        for row in rows:
            name = mapper.get_name(row)
            phone = mapper.get_phone(row)

            if name and name not in persons_seen:
                persons_seen.add(name)
                person = ExtractedEntity(
                    id=generate_id("PERSON", name),
                    entity_type=EntityType.PERSON,
                    name=name,
                    aliases=[name],
                    attributes={
                        k: v for k, v in row.items()
                        if v and k not in ("name", "phone")
                    },
                    confidence=ConfidenceSchema(score=0.9, basis=["Suspect list record"]),
                    source=source,
                    extraction_method="code",
                )
                entities.append(person)

            if phone and phone not in phones_seen:
                phones_seen.add(phone)
                phone_entity = ExtractedEntity(
                    id=generate_id("PHONE", phone),
                    entity_type=EntityType.PHONE,
                    name=phone,
                    attributes={"phone_number": phone},
                    confidence=ConfidenceSchema(score=0.9, basis=["Suspect list record"]),
                    source=source,
                    extraction_method="code",
                )
                entities.append(phone_entity)

                if name:
                    rel = ExtractedRelation(
                        id=generate_id("REL", f"{generate_id('PERSON', name)}_OWNS_{generate_id('PHONE', phone)}"),
                        source_entity_id=generate_id("PERSON", name),
                        target_entity_id=generate_id("PHONE", phone),
                        relation_type=RelationType.ASSOCIATED_WITH,
                        confidence=ConfidenceSchema(score=0.9, basis=["Suspect list record"]),
                        source=source,
                        attributes={"relationship": "owns_phone", "source_name": name, "target_name": phone},
                        extraction_method="code",
                        semantic_edge_type=get_semantic_edge_type(RelationType.ASSOCIATED_WITH),
                        temporal_info=None,
                    )
                    relations.append(rel)

        return entities, relations

    def _extract_generic_tabular(self, rows: List[dict], columns: List[str], source: SourceMetadata) -> Tuple[List, List]:
        """Generic extraction from unknown tabular data — finds phones, names, amounts, locations."""
        entities = []
        relations = []
        phones_seen = set()
        persons_seen = set()

        mapper = ColumnMapper(ai_caller=self.ai)
        if rows:
            mapper.detect_mapping(columns, sample_rows=rows[:3])

        for row in rows:
            attrs = mapper.get_all_attributes(row)

            # Phone entities
            phone = mapper.get_phone(row)
            if phone and phone not in phones_seen:
                phones_seen.add(phone)
                entities.append(ExtractedEntity(
                    id=generate_id("PHONE", phone),
                    entity_type=EntityType.PHONE,
                    name=phone,
                    attributes={"phone_number": phone},
                    confidence=ConfidenceSchema(score=0.8, basis=["Phone in generic tabular"]),
                    source=source, extraction_method="code",
                ))

            # Person entities — from mapped name field or pattern match
            name = mapper.get_name(row)
            if not name:
                for col, val in row.items():
                    if val and re.match(r'^[A-Z][a-z]+ [A-Z][a-z]+', str(val)):
                        name = str(val).strip()
                        break
            if name and name not in persons_seen:
                persons_seen.add(name)
                # Check if it's an organization, not a person
                is_org = self._is_organization_name(name)
                entity_type = EntityType.ORGANIZATION if is_org else EntityType.PERSON
                prefix = "ORG" if is_org else "PERSON"
                entities.append(ExtractedEntity(
                    id=generate_id(prefix, name),
                    entity_type=entity_type,
                    name=name,
                    aliases=[name],
                    attributes=attrs,
                    confidence=ConfidenceSchema(score=0.6, basis=["Name in generic tabular"]),
                    source=source, extraction_method="code",
                ))

        # Create phone-to-phone relations if multiple phones in same row
        rels_seen = set()
        for row in rows:
            phone = mapper.get_phone(row)
            if not phone:
                continue
            # Check for other phone-mapped columns
            phone_cols = [c for c, f in mapper.mappings.items() if f == "phone"]
            for col in phone_cols:
                val = row.get(col, "")
                if val and re.match(r'^\d{10}$', str(val)):
                    other_phone = str(val).strip()
                    if other_phone != phone and other_phone in phones_seen:
                        rel_key = (phone, other_phone)
                        if rel_key not in rels_seen:
                            rels_seen.add(rel_key)
                            rel = ExtractedRelation(
                            id=generate_id("REL", f"{phone}_LINKED_{other_phone}"),
                            source_entity_id=generate_id("PHONE", phone),
                            target_entity_id=generate_id("PHONE", other_phone),
                            relation_type=RelationType.ASSOCIATED_WITH,
                            confidence=ConfidenceSchema(score=0.5, basis=["Same row in generic tabular"]),
                            source=source,
                            attributes={**mapper.get_all_attributes(row), "source_name": phone, "target_name": other_phone},
                            extraction_method="code",
                            semantic_edge_type=get_semantic_edge_type(RelationType.ASSOCIATED_WITH),
                            temporal_info=None,
                        )
                        relations.append(rel)

        return entities, relations

    def _is_valid_entity_name(self, name: str, spacy_label: str) -> bool:
        """Filter out clearly invalid entities from spaCy NER.

        Structural filters + lightweight semantic filters for PERSON labels
        (form-label prefixes, Hindi verb phrases). Full semantic validation
        remains with the optional LLM step when enabled.
        """
        name_clean = name.strip()

        # Structural: names with newlines are boundary detection failures
        if '\n' in name:
            return False

        # Structural: too short to be meaningful
        if len(name_clean) < 2:
            return False

        # Structural: pure whitespace
        if not name_clean:
            return False

        # Semantic (PERSON only): form-label prefixes are not person names
        if spacy_label == "PERSON":
            # e.g. "नाम: राज", "हस्ताक्षर: राज", "पता: X", "मैं, राज"
            _form_label = re.compile(
                r"^(?:नाम|हस्ताक्षर|पता|शिकायत|बयान|निवास|नागरिकता|"
                r"Name|Address|Signature|Complaint|Statement|मैं)\s*[:\s,]+",
                re.IGNORECASE,
            )
            if _form_label.match(name_clean):
                return False

            # Devanagari garbage: incomplete fragment starting with a dependent
            # vowel sign / virama (e.g. "ेश कुमार" — boundary mid-syllable).
            # Vowel signs are U+093E–U+094C; virama is U+094D. Consonants
            # like "म" (U+092E) are fine.
            if name_clean and ("ऀ" <= name_clean[0] <= "्"):
                # Only reject if it's a combining mark (matra/virama), not a letter
                _cp = ord(name_clean[0])
                if 0x093E <= _cp <= 0x094D:
                    return False

            # Hindi function-word-only phrases mislabeled as PERSON by spaCy
            # (English model on Devanagari text). Not person names.
            if re.search(
                r"(?:रहता\s*हूं|रहती\s*हूं|रहता\s*है|रहती\s*है|"
                r"उनमें\s*से|कहा\s*कि|बताया\s*कि|"
                r"मैंने\s*देखा|मैंने\s*सुना|मैंने\s*हमारे|हो\s*जाएगा|"
                r"चाहिए\s*था|कर\s*रहा\s*हूं|कर\s*रही\s*हूं|"
                r"और\s+उसके|खाते\s+से|बैंक\s+खाते)",
                name_clean,
            ):
                return False

            # Pure-Devanagari PERSON: reject if every token is a Hindi
            # function word (sentence fragment, not a name)
            if name_clean and all("ऀ" <= c <= "ॿ" or c.isspace() for c in name_clean):
                _hi_func = {
                    "मैंने", "हमारे", "हमारा", "हमारी", "और", "उसके", "उसकी",
                    "उसका", "खाते", "से", "हैं", "है", "के", "की", "का", "को",
                    "में", "पर", "ने", "नहीं", "तो", "भी", "ही", "यह", "वह",
                    "इस", "उस", "उन", "ये", "वे", "एक", "मेरे", "मेरा",
                    "मेरी", "तुम्हारे", "आपके", "उनके", "उनकी", "उनका",
                    "वाले", "वाला", "वाली", "करने", "किया", "होता", "होती",
                    "होते", "रहा", "रही", "रहे", "गया", "गई", "गए", "बोला",
                    "बोली", "कहा", "बताया", "दिया", "लिया", "था", "थी",
                    "थे", "हुआ", "हुई", "हुए", "बैंक", "खाता", "खाते",
                }
                tokens = name_clean.split()
                if tokens and all(t in _hi_func for t in tokens):
                    return False

            # Common Hindi time/frequency words falsely tagged as PERSON
            if name_clean in {"रोज", "कल", "आज", "कभी", "हमेशा", "रात", "सुबह", "शाम", "दोपहर"}:
                return False

            # Pure-Hindi single-token filler (spaCy path doesn't check HINDI_FILLERS)
            if name_clean in {
                "हूं", "हैं", "है", "था", "थी", "थे", "गया", "गई", "गए",
                "कर", "किया", "मैं", "तुम", "आप", "वह", "यह", "वे", "ये",
                "और", "या", "लेकिन", "पर", "में", "से", "को", "के", "ने",
                "जी", "हाँ", "नहीं", "बिल्कुल", "बहुत", "एक",
            }:
                return False

        return True

    def validate_entities_with_llm(self, entities: list, source_texts: dict = None) -> list:
        """Use LLM to validate and reclassify extracted entities.

        This replaces ALL hardcoded heuristic filters with LLM understanding.
        For each entity, the LLM determines:
        1. Is this a valid entity? (garbage detection — Hindi filler, partial dates, etc.)
        2. Is the type correct? (reclassification — location-as-person, etc.)
        3. Should it be split? (e.g., "PAN Card No" → remove, "Delhi" from "Connaught Place, Delhi" → keep)

        Args:
            entities: List of ExtractedEntity objects
            source_texts: Optional dict mapping entity ID to source text context
        """
        if not self.ai or not entities:
            return entities

        # Batch entities for efficiency (max ~30 per LLM call)
        batch_size = 30
        validated = []
        for i in range(0, len(entities), batch_size):
            batch = entities[i:i + batch_size]
            batch_validated = self._validate_entity_batch(batch, source_texts)
            validated.extend(batch_validated)

        return validated

    def _validate_entity_batch(self, entities: list, source_texts: dict = None) -> list:
        """Validate a batch of entities using LLM."""
        # Build entity list for LLM
        entity_list = []
        for e in entities:
            entity_list.append({
                "id": e.id,
                "name": e.name,
                "type": e.entity_type.value,
                "source_file": e.source.file_name if e.source else "",
                "attributes": e.attributes,
                "confidence": e.confidence.score if e.confidence else 0.5,
            })

        entity_json = json.dumps(entity_list, indent=1)

        # Build context snippets if available
        context = ""
        if source_texts:
            context_parts = []
            for e in entities[:5]:  # Only first 5 for context
                if e.id in source_texts:
                    snippet = source_texts[e.id][:200]
                    context_parts.append(f"- {e.name}: ...{snippet}...")
            if context_parts:
                context = "\n\nSOURCE CONTEXT:\n" + "\n".join(context_parts)

        prompt = f"""Review these extracted entities from criminal investigation data.
For each entity, determine if it should be KEPT (with correct type), RECLASSIFIED (wrong type), or REMOVED (garbage).

Entities:
{entity_json}
{context}

RULES:
1. REMOVE entities that are NOT real-world entities:
    - Hindi filler words: haan, nahi, ji, acha, kal tak, ho jayega, bilkul, chalo, arre, oye, bhai, yaar, boss, senior, bhai sahab, bhai log, sir ji, uncle, aunty, bhaiya, didi, bhai sahab, bhai log, sir ji, uncle, aunty, bhaiya, didi, bhai sahab, bhai log, sir ji, uncle, aunty
   - Placeholder/label text: subject, none, unknown, other, complainant, accused, witness, victim
   - Technical abbreviations used as labels: OTP (one-time password), LOG (log file), SIM (SIM card), H.No (house number)
   - Document references: PAN Card No, Aadhaar, Driving License, Block C
   - Partial dates: 18/03, time-only values: 10:30, 15:45
   - Duration expressions: 26 years, about 30 minutes
   - Case numbers: 1234/2024
   - Conjunction of names: "Rakesh and Suresh" (this is two people, not one entity)
   - Single letters, pure numbers

2. RECLASSIFY entities that have wrong type:
   - Locations tagged as PERSON: "Karol Bagh", "Connaught Place", "Noida", "Bangalore", "Goa", "Lajpat Nagar", "Nehru Nagar" → LOCATION
   - People tagged as ORG: "SI Vikram Rathore" (has police rank prefix SI = Sub Inspector) → PERSON
   - Organizations tagged as PERSON: "Rajesh Electronics" → ORGANIZATION

3. KEEP entities that are valid names, locations, organizations, amounts, dates, events

OUTPUT FORMAT (JSON):
{{
  "validations": [
    {{"id": "entity_id", "action": "keep|remove|reclassify", "new_type": "PERSON|LOCATION|ORGANIZATION|PHONE|AMOUNT|DATE|EVENT|null", "reason": "brief explanation"}}
  ]
}}

OUTPUT ONLY VALID JSON. No extra text."""

        result = self.ai.extract(
            task="validate_entities",
            text=prompt,
            custom_prompt="You are an entity validator for criminal investigation data. Output ONLY valid JSON.",
        )

        if result.get("status") != "success" or not result.get("parsed"):
            # If LLM fails, keep all entities (fail open — don't lose data)
            print(f"[EXTRACTION] LLM entity validation failed, keeping all entities")
            return entities

        parsed = result["parsed"]
        validations = parsed.get("validations", [])

        # Build validation lookup
        validation_map = {}
        for v in validations:
            validation_map[v["id"]] = v

        # Apply validations
        validated = []
        for e in entities:
            v = validation_map.get(e.id)
            if not v:
                # No validation entry — keep entity as-is
                validated.append(e)
                continue

            action = v.get("action", "keep")

            if action == "remove":
                # Entity is garbage — skip it
                print(f"[EXTRACTION] LLM removed: {e.name} ({e.entity_type.value}) — {v.get('reason', '')}")
                continue

            elif action == "reclassify":
                new_type = v.get("new_type")
                if new_type and new_type != e.entity_type.value:
                    try:
                        e.entity_type = EntityType(new_type)
                        e.attributes["llm_reclassified"] = True
                        e.attributes["original_type"] = e.entity_type.value
                        e.confidence.score = min(e.confidence.score + 0.1, 1.0)  # Boost confidence after validation
                        print(f"[EXTRACTION] LLM reclassified: {e.name} {e.entity_type.value} → {new_type}")
                    except ValueError:
                        pass  # Invalid type — keep original

            elif action == "keep":
                # Valid entity — boost confidence slightly
                e.confidence.score = min(e.confidence.score + 0.05, 1.0)

            validated.append(e)

        print(f"[EXTRACTION] LLM validation: {len(entities)} → {len(validated)} entities")
        return validated

    def _extract_ner_spacy(self, text: str, source: SourceMetadata) -> list:
        """Extract entities using spaCy NER — supplements regex-based extraction."""
        entities = []
        if not text or len(text) < 20:
            return entities

        try:
            import spacy
            # Load small English model (fast, good enough for entity types)
            if not hasattr(self, '_spacy_nlp'):
                self._spacy_nlp = spacy.load("en_core_web_sm", disable=["parser", "lemmatizer"])
            nlp = self._spacy_nlp

            # Process in chunks for large texts (spaCy has token limits)
            max_len = 100000
            doc = nlp(text[:max_len]) if len(text) > max_len else nlp(text)

            # Map spaCy entity labels to our EntityType
            LABEL_MAP = {
                "PERSON": ("PERSON", EntityType.PERSON),
                "ORG": ("ORGANIZATION", EntityType.ORGANIZATION),
                "GPE": ("LOCATION", EntityType.LOCATION),       # countries, cities, states
                "LOC": ("LOCATION", EntityType.LOCATION),       # non-GPE locations
                "FAC": ("LOCATION", EntityType.LOCATION),       # facilities
                "DATE": ("DATE", EntityType.DATE),
                "TIME": ("DATE", EntityType.DATE),
                "MONEY": ("AMOUNT", EntityType.AMOUNT),
                "CARDINAL": ("AMOUNT", EntityType.AMOUNT),
            }

            # Location / organization keyword patterns — used to demote
            # PERSON spans that are really places or organizations.
            _loc_pat = re.compile(
                r"\b(Tower|Market|Nagar|Bagh|Place|Colony|Extension|Enclave|"
                r"Apartments?|Vihar|Pur|Garh|City|District|Zone|Road|Street|"
                r"Lane|Marg|Gali|Chowk|Phase|Sector|Block|Part|"
                r"Shop|Store|Showroom|Office|Complex|Mall|"
                r"Metro|Station|Airport|Bus\s+Stand|"
                r"Garden|Park|Lake|River|Bridge|Building|"
                r"Industrial|Campus)\b",
                re.IGNORECASE,
            )
            _org_pat = re.compile(
                r"\b(Herald|Times|Tribune|Express|Gazette|Chronicle|"
                r"News|Journal|Reporter|Media|Network|"
                r"Ltd|Limited|Corp|Inc|Company|Agency|Group|"
                r"Department|Ministry|Commission|Authority|Board|"
                r"Foundation|Trust|Institute|University|College)\b",
                re.IGNORECASE,
            )
            # Trailing phone numbers glued onto name spans by boundary
            # detection: "Rakesh Kumar (9876543210" → "Rakesh Kumar".
            _phone_suffix = re.compile(r"\s*[\(\[]?\s*\+?\d[\d\s\-]{6,14}\s*\)?\s*$")

            seen_ids = set()
            for ent in doc.ents:
                if ent.label_ not in LABEL_MAP:
                    continue
                name = ent.text.strip()

                # Name-like labels: strip trailing phones glued on by
                # boundary detection, dangling brackets ("Rakesh Kumar ("),
                # and possessive 's ("Meena Devi's").
                if ent.label_ in ("PERSON", "ORG", "GPE", "LOC", "FAC"):
                    name = _phone_suffix.sub("", name)
                    name = re.sub(r"[\s\(\[\{,;:]+$", "", name).strip()
                    name = re.sub(r"'s$", "", name).strip()

                # Skip very short, very long, or numeric-only
                if len(name) < 3 or len(name) > 100:
                    continue
                if name.isdigit():
                    continue

                # FILTER: Skip garbage/misclassified entities
                if not self._is_valid_entity_name(name, ent.label_):
                    continue

                # FILTER: single-token lowercase ASCII place spans are function
                # words mislabeled GPE/LOC ("Theek hai bhai..." → "hai").
                # Real place names are capitalized or multi-token; Devanagari
                # spans have no case and are never .islower().
                if ent.label_ in ("GPE", "LOC") and len(name.split()) == 1 \
                        and name.isascii() and name.islower():
                    continue

                # FILTER: document-header boilerplate tagged ORG in the first
                # lines ("Confidential - For Investigation Purposes",
                # "SUPPLEMENTARY"). Evidence organizations appear in the body,
                # past the header zone.
                if ent.label_ == "ORG" and ent.start_char < 80:
                    continue

                # FILTER: spaCy English model mislabels Devanagari text as
                # ORG/CARDINAL/etc. (e.g. "मैंने"→ORG, "मैं"→CARDINAL→AMOUNT).
                # Only PERSON/LOCATION/GPE/DATE/etc. survive pure-Devanagari
                # spans; ORG requires an English org keyword (handled below).
                _devanagari_chars = sum(1 for c in name if "ऀ" <= c <= "ॿ")
                _devanagari_ratio = _devanagari_chars / max(len(name), 1)
                if _devanagari_ratio > 0.5 and ent.label_ in (
                    "ORG", "CARDINAL", "ORDINAL", "MONEY", "PERCENT",
                    "QUANTITY", "NORP", "FAC", "PRODUCT", "WORK_OF_ART",
                    "LANGUAGE", "EVENT",
                ):
                    continue

                # FILTER: Devanagari DATE/TIME spans with no digit are
                # sentence fragments ("पहले देखा था"), not dates — the
                # English model mislabels verb phrases as DATE. Real Devanagari
                # dates ("मार्च 2024", "9:15 बजे") always carry digits.
                if _devanagari_ratio > 0.5 and ent.label_ in ("DATE", "TIME"):
                    if not re.search(r"\d", name):
                        continue

                prefix, entity_type = LABEL_MAP[ent.label_]
                # Heuristic: reclassify obvious location / organization names
                # mislabeled as PERSON by spaCy NER.
                if ent.label_ == "PERSON":
                    if _loc_pat.search(name):
                        prefix = "LOCATION"
                        entity_type = EntityType.LOCATION
                    elif _org_pat.search(name) or self._is_organization_name(name):
                        prefix = "ORG"
                        entity_type = EntityType.ORGANIZATION
                entity_id = generate_id(prefix, name)

                # Dedupe by content-hash id (not raw name): identical
                # cleaned name + type collapse to one entity, while the same
                # name under different NER labels (PERSON "Meena Devi" vs
                # FAC "Meena Devi's") survives for resolution to reconcile.
                if entity_id in seen_ids:
                    continue
                seen_ids.add(entity_id)

                entities.append(ExtractedEntity(
                    id=entity_id,
                    entity_type=entity_type,
                    name=name,
                    aliases=[name],
                    attributes={
                        "extracted_from": "spacy_ner",
                        "spacy_label": ent.label_,
                        "spacy_start": ent.start_char,
                        "spacy_end": ent.end_char,
                    },
                    confidence=ConfidenceSchema(
                        score=0.75,
                        basis=["spaCy NER extraction"],
                    ),
                    source=source,
                    extraction_method="code",
                ))

        except ImportError:
            pass  # spaCy not installed — skip NER
        except Exception:
            pass  # Model not loaded or processing error — skip silently

        return entities

    def _extract_hindi_ner(self, text: str, source: SourceMetadata) -> list:
        """Extract entities from Hindi text using regex patterns for common Hindi name/address patterns.
        
        Enhanced Hindi NER:
        1. Better name patterns with context (e.g., "नाम: X", "X ने कहा")
        2. Exclusion list for common Hindi filler words
        3. Validation for extracted entities
        """
        entities = []
        if not text or len(text) < 10:
            return entities

        # Detect if text contains Devanagari script
        devanagari_chars = sum(1 for c in text if '\u0900' <= c <= '\u097F')
        if devanagari_chars < 5:
            return entities

        # Common Hindi filler words to exclude (not entities)
        HINDI_FILLERS = {
            'है', 'हैं', 'हूं', 'हो', 'था', 'थे', 'थी', 'थे', 'गया', 'गई', 'गए',
            'कर', 'करें', 'किया', 'करते', 'करने', 'करना', 'होने', 'होना', 'होता',
            'मैं', 'तुम', 'आप', 'वह', 'यह', 'वे', 'ये', 'उस', 'इस', 'उन', 'इन',
            'और', 'या', 'लेकिन', 'पर', 'पर', 'में', 'से', 'को', 'के', 'ने', 'ना',
            'हो', 'जा', 'आ', 'दे', 'ले', 'पा', 'सक', 'चाह', 'पड़', 'रह', 'कह',
            'अच्छा', 'ठीक', 'बिल्कुल', 'जी', 'हाँ', 'नहीं', 'शायद', 'शुरू', 'खत्म',
            'बहुत', 'ज्यादा', 'कम', 'थोड़ा', 'सारा', 'पूरा', 'आधा', 'दोगुना',
            'एक', 'दो', 'तीन', 'चार', 'पाँच', 'छह', 'सात', 'आठ', 'नौ', 'दस',
            'पहले', 'बाद', 'फिर', 'अब', 'तब', 'कल', 'आज', 'कभी', 'हमेशा',
            'यहाँ', 'वहाँ', 'कहीं', 'कहीं', 'ऐसे', 'वैसे', 'जैसे', 'तैसे',
        }

        # Hindi name patterns: more specific to avoid garbage
        hindi_name_patterns = [
            # "नाम: X" or "नाम X"
            r'(?:नाम|naam|Name)\s*[:\s]+([^\n,]+?)(?:\s+है|\s+हैं|\s*,|\s*$)',
            # "X ने कहा" (X said) - common in statements
            r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+ने\s+(?:कहा|बताया|देखा|सुना)',
            # "X का/की/के" (X's) - possession
            r'([^\s]+(?:का|की|के))\s+(?:फ़ोन|नंबर|खाता|पता)',
            # Names with common Hindi surnames
            r'([^\s]+\s+(?:कुमार|शर्मा|गुप्ता|यादव|पटेल|सिंह|देवी|राज|वर्मा|जैन|अग्रवाल|मिश्रा|त्रिपाठी|पांडेय|चौहान|रावत|नेगी|बिष्ट|राव|चौधरी|मीणा|जाट|गुर्जर|यादव|कुशवाहा|सेन|तिवारी|चतुर्वेदी))',
            # "श्री X" or "श्रीमती X" (Mr./Mrs. X)
            r'(?:श्री|श्रीमती|सुश्री)\s+([^\s,]+(?:\s+[^\s,]+){0,2})',
        ]
        # Form-label prefixes that pattern 4 (surname list) can capture with the label
        _hindi_form_label = re.compile(
            r"^(?:नाम|हस्ताक्षर|पता|शिकायत|बयान|निवास|मैं)\s*[:\s,]+",
            re.IGNORECASE,
        )
        seen_names = set()
        for pattern in hindi_name_patterns:
            matches = re.findall(pattern, text)
            for name in matches:
                name = name.strip()
                # Strip form-label prefix accidentally captured: "नाम: राज" → "राज"
                name = _hindi_form_label.sub("", name).strip(" ,:।")
                # Reject mid-syllable fragments starting with a dependent vowel
                # sign / virama (e.g. "ेश कुमार"). Consonants are fine.
                if name and 0x093E <= ord(name[0]) <= 0x094D:
                    continue
                # Validate: must be at least 3 chars, not a filler, not digits
                if (len(name) > 2 and
                    name not in seen_names and
                    not name.isdigit() and
                    name.lower() not in HINDI_FILLERS and
                    not all('ऀ' <= c <= 'ॿ' for c in name)):  # Not all Devanagari (likely garbage)
                    seen_names.add(name)
                    entities.append(ExtractedEntity(
                        id=generate_id("PERSON", name),
                        entity_type=EntityType.PERSON,
                        name=name,
                        aliases=[name],
                        attributes={"extracted_from": "hindi_ner", "script": "devanagari"},
                        confidence=ConfidenceSchema(score=0.65, basis=["Hindi text NER pattern"]),
                        source=source,
                        extraction_method="code",
                    ))

        # Hindi phone patterns: 10-digit numbers (same as English)
        phones = re.findall(r'\b\d{10}\b', text)
        for phone in set(phones):
            entities.append(ExtractedEntity(
                id=generate_id("PHONE", phone),
                entity_type=EntityType.PHONE,
                name=phone,
                attributes={"phone_number": phone, "extracted_from": "hindi_text"},
                confidence=ConfidenceSchema(score=0.8, basis=["Phone in Hindi text"]),
                source=source,
                extraction_method="code",
            ))

        # Hindi location patterns: addresses with common Hindi locality markers
        hindi_loc_patterns = [
            # "थाना: X" (Police station: X)
            r'(?:थाना|पता|इलाका|जिला)\s*[:\s]+([^\n,]+?)(?:\s*,|\s*$)',
            # "X नगर" or "X कॉलोनी" etc.
            r'([^\s]+(?:नगर|पुर|बाग|मंडी|कॉलोनी|गली|मोहल्ला|शहर|जिला|राज्य|मार्ग|रोड|सड़क|पार्क|विहार))',
            # "लाजपत नगर" style locations
            r'((?:[A-Z][a-z]+\s+){1,3}(?:नगर|पुर|बाग|मंडी|कॉलोनी|गली|मोहल्ला))',
        ]
        seen_locs = set()
        for pattern in hindi_loc_patterns:
            matches = re.findall(pattern, text)
            for loc in matches:
                loc = loc.strip()
                if len(loc) > 3 and loc not in seen_locs:
                    seen_locs.add(loc)
                    entities.append(ExtractedEntity(
                        id=generate_id("LOC", loc),
                        entity_type=EntityType.LOCATION,
                        name=loc,
                        attributes={"extracted_from": "hindi_text", "script": "devanagari"},
                        confidence=ConfidenceSchema(score=0.55, basis=["Hindi location pattern"]),
                        source=source,
                        extraction_method="code",
                    ))

        # Hindi amount patterns: "15 लाख", "₹5000" etc.
        hindi_amount_patterns = [
            r'(\d+(?:,\d+)*(?:\.\d+)?)\s*(?:लाख|लाख|hundred thousand)',
            r'₹\s*(\d+(?:,\d+)*)',
            r'(\d+(?:,\d+)*)\s*(?:रुपये|rs\.?|inr)',
        ]
        for pattern in hindi_amount_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for amt in matches:
                amt_clean = amt.replace(",", "").strip()
                if amt_clean.isdigit() and int(amt_clean) > 0:
                    entities.append(ExtractedEntity(
                        id=generate_id("AMOUNT", amt_clean),
                        entity_type=EntityType.AMOUNT,
                        name=amt_clean,
                        attributes={"amount": amt_clean, "extracted_from": "hindi_text"},
                        confidence=ConfidenceSchema(score=0.7, basis=["Hindi amount pattern"]),
                        source=source,
                        extraction_method="code",
                    ))

        return entities

    def _assign_entity_roles(self, text: str, entities: list) -> dict:
        """Detect roles (accused, victim, witness, complainant) from context around entity mentions."""
        roles = {}
        text_lower = text.lower()

        # Role keyword patterns with context
        ROLE_PATTERNS = {
            "accused": [
                r'(?:accused|alleged|suspect|offender|प्रतिवादी|आरोपी|शक)\s*[:\s]+([^\n,]+)',
                r'([^\n]+)\s+(?:is accused|was arrested|has been charged)',
                r'(?:named|identified)\s+([^\n]+)\s+(?:as|in connection)',
            ],
            "victim": [
                r'(?:victim| complainant|deceased|मृतक|पीड़ित)\s*[:\s]+([^\n,]+)',
                r'([^\n]+)\s+(?:was killed|was murdered|was attacked|lost)',
            ],
            "witness": [
                r'(?:witness|eyewitness|साक्षी)\s*[:\s]+([^\n,]+)',
                r'([^\n]+)\s+(?:witnessed|saw|observed|testified)',
            ],
            "complainant": [
                r'(?:complainant|reporter|शिकायतकर्ता)\s*[:\s]+([^\n,]+)',
                r'complaint\s+(?:by|filed by)\s+([^\n,]+)',
            ],
        }

        for role, patterns in ROLE_PATTERNS.items():
            for pattern in patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                for match in matches:
                    name = match.strip()
                    if len(name) > 2:
                        # Find entity by name match
                        for entity in entities:
                            if entity.name.lower() == name.lower() or name.lower() in [a.lower() for a in entity.aliases]:
                                roles[entity.id] = role
                                break
                        # Also try partial match
                        if name not in [e.name for e in entities if hasattr(e, 'name')]:
                            for entity in entities:
                                if hasattr(entity, 'name') and name.lower() in entity.name.lower():
                                    roles[entity.id] = role
                                    break

        return roles

    def _extract_narrative_relations(
        self, text: str, entities: list, source: SourceMetadata
    ) -> list:
        """Extract directed relations from prose sentences.

        Patterns cover the common narrative verbs found in FIRs / investigation
        notes: called, visited, transferred, owns, associated, family, works, etc.
        Only links entities already extracted from the same text — never invents
        new entities. All returned relations are epistemic_status=observation
        with derivation_depth=0 (raw text evidence).
        """
        if not text or not entities:
            return []

        # Index person/location/organization entities by lowercase name and aliases
        type_groups = {
            "person": [],
            "location": [],
            "organization": [],
            "phone": [],
            "account": [],
            "amount": [],
        }
        etype_map = {
            "PERSON": "person",
            "LOCATION": "location",
            "ORGANIZATION": "organization",
            "PHONE": "phone",
            "ACCOUNT": "account",
            "AMOUNT": "amount",
            "BANK_ACCOUNT": "account",
        }
        for e in entities:
            key = etype_map.get(e.entity_type.value if hasattr(e.entity_type, "value") else str(e.entity_type), None)
            if not key:
                continue
            names = [e.name] + list(getattr(e, "aliases", []) or [])
            for nm in names:
                if nm and len(nm) >= 2:
                    type_groups[key].append((nm.lower(), e.id, e.name))

        persons = type_groups["person"]
        locations = type_groups["location"]
        orgs = type_groups["organization"]
        if len(persons) < 1:
            return []

        # Relation patterns: (regex capturing subject-group and object-group)
        # Group keys: P=person, L=location, O=organization, PH=phone, A=account, AM=amount
        PATTERNS = [
            # called / phoned / rang
            (r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})\s+(?:called|phoned|rang|dialed)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})\b",
             "person", "person", RelationType.CALLED, 0.8),
            # visited / went to / arrived at / stayed at
            (r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})\s+(?:visited|went to|arrived at|stayed at|reached|entered)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})\b",
             "person", "location", RelationType.VISITED, 0.75),
            # lives at / residing at / resident of
            (r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})\s+(?:lives at|residing at|resident of|stays at|resides in)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})\b",
             "person", "location", RelationType.LIVES_AT, 0.8),
            # works at / employed at
            (r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})\s+(?:works at|employed at|working at|employee of)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})\b",
             "person", "organization", RelationType.WORKS_AT, 0.8),
            # transferred / paid / sent … to
            (r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})\s+(?:transferred|paid|sent|deposited|remitted)\s+(?:Rs\.?\s*[\d,]+|₹[\d,]+|INR\s*[\d,]+)?\s*(?:to)?\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})\b",
             "person", "person", RelationType.TRANSFERRED_TO, 0.7),
            # owns / possesses / holds
            (r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})\s+(?:owns|possesses|holds|maintains)\s+(?:account|phone|number|SIM)?\s*([A-Z0-9][\w\s-]{2,})\b",
             "person", "account", RelationType.OWNS_ACCOUNT, 0.7),
            # associated with / known as / alias of
            (r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})\s+(?:is associated with|associated with|known as|alias of|also known as)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})\b",
             "person", "person", RelationType.ASSOCIATED_WITH, 0.65),
            # brother of / son of / wife of / relative of / family of
            (r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})\s+(?:is brother of|brother of|son of|daughter of|wife of|husband of|relative of|family of|sibling of)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})\b",
             "person", "person", RelationType.FAMILY_OF, 0.85),
            # friend of
            (r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})\s+(?:is friend of|friend of|buddy of)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})\b",
             "person", "person", RelationType.FRIEND_OF, 0.75),
            # works with / colleague of / partner of
            (r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})\s+(?:works with|colleague of|partner of|associated colleague)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})\b",
             "person", "person", RelationType.WORKS_WITH, 0.75),
            # member of / belongs to
            (r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})\s+(?:is member of|member of|belongs to|joined)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})\b",
             "person", "organization", RelationType.MEMBER_OF, 0.8),
            # suspect of / involved in
            (r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})\s+(?:is suspect of|suspect of|involved in|accused in)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})\b",
             "person", "person", RelationType.SUSPECT_OF, 0.7),
        ]

        def find_entity(name: str, group_key: str):
            name_l = name.lower().strip()
            if not name_l:
                return None
            for nm, eid, _disp in type_groups.get(group_key, []):
                if nm == name_l:
                    return eid
            # fallback: substring
            for nm, eid, _disp in type_groups.get(group_key, []):
                if name_l in nm or nm in name_l:
                    return eid
            return None

        relations = []
        seen_pairs = set()
        text_original = text  # keep original for regex case sensitivity on names

        # ── Phase 1: Entity-pair proximity extraction ──
        # Find where known entity names co-occur within a sentence/window,
        # with a relation verb between them. This handles FIR prose that says
        # "introduced himself as Rajesh from SBI" rather than "Rajesh called Meena".
        import re as _re
        sentences = _re.split(r'[.!?;\n]+', text_original)
        rel_verbs = {
            "called": RelationType.CALLED, "phoned": RelationType.CALLED,
            "rang": RelationType.CALLED, "dialed": RelationType.CALLED,
            "visited": RelationType.VISITED, "went to": RelationType.VISITED,
            "arrived at": RelationType.VISITED, "stayed at": RelationType.VISITED,
            "reached": RelationType.VISITED, "entered": RelationType.VISITED,
            "transferred": RelationType.TRANSFERRED_TO, "paid": RelationType.TRANSFERRED_TO,
            "sent": RelationType.TRANSFERRED_TO, "deposited": RelationType.TRANSFERRED_TO,
            "remitted": RelationType.TRANSFERRED_TO,
            "owns": RelationType.OWNS_ACCOUNT, "possesses": RelationType.OWNS_ACCOUNT,
            "holds": RelationType.OWNS_ACCOUNT, "maintains": RelationType.OWNS_ACCOUNT,
            "associated with": RelationType.ASSOCIATED_WITH,
            "known as": RelationType.ASSOCIATED_WITH,
            "lives at": RelationType.LIVES_AT, "residing at": RelationType.LIVES_AT,
            "resident of": RelationType.LIVES_AT, "stays at": RelationType.LIVES_AT,
            "resides in": RelationType.LIVES_AT,
            "works at": RelationType.WORKS_AT, "employed at": RelationType.WORKS_AT,
            "working at": RelationType.WORKS_AT, "employee of": RelationType.WORKS_AT,
            "operates from": RelationType.LIVES_AT,
            "brother of": RelationType.FAMILY_OF, "son of": RelationType.FAMILY_OF,
            "daughter of": RelationType.FAMILY_OF, "wife of": RelationType.FAMILY_OF,
            "husband of": RelationType.FAMILY_OF, "father of": RelationType.FAMILY_OF,
            "mother of": RelationType.FAMILY_OF, "relative of": RelationType.FAMILY_OF,
            "friend of": RelationType.FRIEND_OF,
            "works with": RelationType.WORKS_WITH, "colleague of": RelationType.WORKS_WITH,
            "partner of": RelationType.WORKS_WITH,
            "member of": RelationType.MEMBER_OF, "belongs to": RelationType.MEMBER_OF,
            "involved in": RelationType.SUSPECT_OF, "accused in": RelationType.SUSPECT_OF,
        }

        # FIR-specific: "introduced himself as X from Y" → X ASSOCIATED_WITH Y (or WORKS_AT)
        for m in _re.finditer(
            r'introduced\s+(?:himself|herself|themselves)\s+as\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\s+from\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*?)(?:\s+and\s|\s*$|\s*\.)',
            text_original, _re.IGNORECASE | _re.MULTILINE
        ):
            subj_name, obj_name = m.group(1).strip(), m.group(2).strip()
            # Clean up object name (strip trailing conjunctions)
            obj_name = _re.sub(r'\s+(and|who|that|which).*$', '', obj_name, flags=_re.IGNORECASE).strip()
            sid = find_entity(subj_name, "person")
            oid = find_entity(obj_name, "organization") or find_entity(obj_name, "person")
            if sid and oid and sid != oid:
                pair_key = (sid, oid, "ASSOCIATED_WITH")
                if pair_key not in seen_pairs:
                    seen_pairs.add(pair_key)
                    span = m.start(0)
                    context = text_original[max(0, span - 40):span + len(m.group(0)) + 40]
                    relations.append(ExtractedRelation(
                        id=generate_id("REL", f"{sid}_{oid}_ASSOCIATED_WITH"),
                        source_entity_id=sid, target_entity_id=oid,
                        relation_type=RelationType.ASSOCIATED_WITH,
                        confidence=make_confidence(0.75, basis=["Narrative pattern: introduced_as"], source=source, epistemic="observation", derivation_depth=0),
                        source=source,
                        attributes={"narrative_pattern": True, "verb": "introduced_as"},
                        raw_text=context.strip(), extraction_method="code",
                        epistemic_category="observation", derivation_depth=0,
                    ))

        # FIR-specific: "as X from Y" (variant without "introduced himself")
        for m in _re.finditer(
            r'(?:as|identified\s+as|known\s+as)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\s+from\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*?)(?:\s+and\s|\s*$|\s*\.|,)',
            text_original, _re.IGNORECASE | _re.MULTILINE
        ):
            subj_name, obj_name = m.group(1).strip(), m.group(2).strip()
            obj_name = _re.sub(r'\s+(and|who|that|which|stating).*$', '', obj_name, flags=_re.IGNORECASE).strip()
            if len(subj_name) < 3 or len(obj_name) < 3:
                continue
            sid = find_entity(subj_name, "person")
            oid = find_entity(obj_name, "organization") or find_entity(obj_name, "person")
            if sid and oid and sid != oid:
                pair_key = (sid, oid, "ASSOCIATED_WITH")
                if pair_key not in seen_pairs:
                    seen_pairs.add(pair_key)
                    span = m.start(0)
                    context = text_original[max(0, span - 40):span + len(m.group(0)) + 40]
                    relations.append(ExtractedRelation(
                        id=generate_id("REL", f"{sid}_{oid}_ASSOCIATED_WITH"),
                        source_entity_id=sid, target_entity_id=oid,
                        relation_type=RelationType.ASSOCIATED_WITH,
                        confidence=make_confidence(0.7, basis=["Narrative pattern: known_as_from"], source=source, epistemic="observation", derivation_depth=0),
                        source=source,
                        attributes={"narrative_pattern": True, "verb": "known_as_from"},
                        raw_text=context.strip(), extraction_method="code",
                        epistemic_category="observation", derivation_depth=0,
                    ))

        # FIR-specific: "Rs. X transferred to account number Y" → AMOUNT TRANSFERRED_TO ACCOUNT
        for m in _re.finditer(
            r'Rs\.?\s*[\d,]+\s+transferred\s+to\s+account\s+number\s+(\d{10,})',
            text_original, _re.IGNORECASE
        ):
            acct_num = m.group(1)
            oid = find_entity(acct_num, "account")
            # Find an amount entity in the same context
            amt_match = _re.search(r'Rs\.?\s*[\d,]+', text_original[max(0, m.start()-50):m.end()])
            if amt_match and oid:
                amt_name = amt_match.group(0).strip()
                sid = find_entity(amt_name, "amount")
                if sid and sid != oid:
                    pair_key = (sid, oid, "TRANSFERRED_TO")
                    if pair_key not in seen_pairs:
                        seen_pairs.add(pair_key)
                        span = m.start(0)
                        context = text_original[max(0, span - 40):span + len(m.group(0)) + 40]
                        relations.append(ExtractedRelation(
                            id=generate_id("REL", f"{sid}_{oid}_TRANSFERRED_TO"),
                            source_entity_id=sid, target_entity_id=oid,
                            relation_type=RelationType.TRANSFERRED_TO,
                            confidence=make_confidence(0.7, basis=["Narrative pattern: amount_transfer"], source=source, epistemic="observation", derivation_depth=0),
                            source=source,
                            attributes={"narrative_pattern": True, "verb": "transferred"},
                            raw_text=context.strip(), extraction_method="code",
                            epistemic_category="observation", derivation_depth=0,
                        ))

        # Generic: two known entities + relation verb in same sentence
        for sent in sentences:
            sent_lower = sent.lower()
            # Detect which relation verb(s) appear
            matched_verb_type = None
            matched_verb = None
            for verb, rtype in rel_verbs.items():
                if verb in sent_lower:
                    matched_verb_type = rtype
                    matched_verb = verb
                    break
            if not matched_verb_type:
                continue

            # Find entity names appearing in this sentence
            sent_entities = []
            for group_key in ["person", "location", "organization", "phone", "account", "amount"]:
                for nm, eid, _disp in type_groups.get(group_key, []):
                    if len(nm) >= 3 and nm in sent_lower:
                        sent_entities.append((eid, group_key, nm))

            # Dedup by entity id, keep first occurrence
            seen_eids = set()
            unique_ents = []
            for eid, gk, nm in sent_entities:
                if eid not in seen_eids:
                    seen_eids.add(eid)
                    unique_ents.append((eid, gk, nm))

            # Need at least 2 entities for a relation
            if len(unique_ents) < 2:
                continue

            # Smarter subject/object selection based on relation type and entity types
            # For LIVES_AT: person → location (not location → location)
            # For TRANSFERRED_TO: person/amount → account/organization
            # For FAMILY_OF/FRIEND_OF: person → person
            # Default: first person as subject, non-person as object
            
            persons_in_sent = [(eid, gk, nm) for eid, gk, nm in unique_ents if gk == "person"]
            non_persons = [(eid, gk, nm) for eid, gk, nm in unique_ents if gk != "person"]
            
            # Select subject and object based on relation semantics
            if matched_verb_type == RelationType.LIVES_AT:
                # Person lives at location
                if persons_in_sent and non_persons:
                    # Prefer location-type non-persons
                    locations = [(eid, gk, nm) for eid, gk, nm in non_persons if gk == "location"]
                    if locations:
                        sid, s_gk, s_nm = persons_in_sent[0]
                        oid, o_gk, o_nm = locations[0]
                    else:
                        sid, s_gk, s_nm = persons_in_sent[0]
                        oid, o_gk, o_nm = non_persons[0]
                else:
                    continue  # Need person + location
            elif matched_verb_type in (RelationType.FAMILY_OF, RelationType.FRIEND_OF, RelationType.WORKS_WITH, RelationType.CALLED, RelationType.SUSPECT_OF):
                # Person → person
                if len(persons_in_sent) >= 2:
                    sid, s_gk, s_nm = persons_in_sent[0]
                    oid, o_gk, o_nm = persons_in_sent[1]
                else:
                    continue
            elif matched_verb_type == RelationType.TRANSFERRED_TO:
                # person/amount → account/organization
                if persons_in_sent and non_persons:
                    # Prefer accounts and organizations as targets
                    accounts = [(eid, gk, nm) for eid, gk, nm in non_persons if gk in ("account", "organization")]
                    if accounts:
                        sid, s_gk, s_nm = persons_in_sent[0]
                        oid, o_gk, o_nm = accounts[0]
                    else:
                        sid, s_gk, s_nm = persons_in_sent[0]
                        oid, o_gk, o_nm = non_persons[0]
                elif non_persons and len(non_persons) >= 2:
                    # amount → account
                    amounts = [(eid, gk, nm) for eid, gk, nm in non_persons if gk == "amount"]
                    accts = [(eid, gk, nm) for eid, gk, nm in non_persons if gk in ("account", "organization")]
                    if amounts and accts:
                        sid, s_gk, s_nm = amounts[0]
                        oid, o_gk, o_nm = accts[0]
                    else:
                        continue
                else:
                    continue
            elif matched_verb_type == RelationType.WORKS_AT:
                # person → organization
                if persons_in_sent and non_persons:
                    orgs = [(eid, gk, nm) for eid, gk, nm in non_persons if gk == "organization"]
                    if orgs:
                        sid, s_gk, s_nm = persons_in_sent[0]
                        oid, o_gk, o_nm = orgs[0]
                    else:
                        continue
                else:
                    continue
            elif matched_verb_type == RelationType.OWNS_ACCOUNT:
                # person → account
                if persons_in_sent and non_persons:
                    accts = [(eid, gk, nm) for eid, gk, nm in non_persons if gk in ("account", "phone")]
                    if accts:
                        sid, s_gk, s_nm = persons_in_sent[0]
                        oid, o_gk, o_nm = accts[0]
                    else:
                        continue
                else:
                    continue
            else:
                # Default: first as subject, second as object (but skip same non-person types)
                if len(unique_ents) >= 2:
                    sid, s_gk, s_nm = unique_ents[0]
                    oid, o_gk, o_nm = unique_ents[1]
                    if s_gk == o_gk and s_gk != "person":
                        continue
                else:
                    continue

            if sid == oid:
                continue
            pair_key = (sid, oid, matched_verb_type.value)
            if pair_key in seen_pairs:
                continue
            seen_pairs.add(pair_key)
            context = sent.strip()[:200]
            relations.append(ExtractedRelation(
                id=generate_id("REL", f"{sid}_{oid}_{matched_verb_type.value}"),
                source_entity_id=sid, target_entity_id=oid,
                relation_type=matched_verb_type,
                confidence=make_confidence(0.65, basis=[f"Narrative pattern: {matched_verb}"], source=source, epistemic="observation", derivation_depth=0),
                source=source,
                attributes={"narrative_pattern": True, "verb": matched_verb},
                raw_text=context, extraction_method="code",
                epistemic_category="observation", derivation_depth=0,
            ))

        # ── Phase 2: Legacy regex patterns (keep for non-FIR prose) ──
        for pattern, subj_type, obj_type, rel_type, conf in PATTERNS:
            try:
                matches = _re.finditer(pattern, text_original, _re.IGNORECASE)
            except _re.error:
                continue
            for m in matches:
                try:
                    subj_name = m.group(1).strip()
                    obj_name = m.group(2).strip()
                except IndexError:
                    continue
                if not subj_name or not obj_name:
                    continue
                if subj_name.lower() == obj_name.lower():
                    continue
                sid = find_entity(subj_name, subj_type)
                oid = find_entity(obj_name, obj_type)
                if not sid or not oid or sid == oid:
                    continue
                pair_key = (sid, oid, rel_type.value)
                if pair_key in seen_pairs:
                    continue
                seen_pairs.add(pair_key)
                span = m.start(0)
                context = text_original[max(0, span - 40):span + len(m.group(0)) + 40]
                relations.append(ExtractedRelation(
                    id=generate_id("REL", f"{sid}_{oid}_{rel_type.value}"),
                    source_entity_id=sid,
                    target_entity_id=oid,
                    relation_type=rel_type,
                    confidence=make_confidence(
                        conf,
                        basis=[f"Narrative pattern: {rel_type.value}"],
                        source=source,
                        epistemic="observation",
                        derivation_depth=0,
                    ),
                    source=source,
                    attributes={"narrative_pattern": True, "verb": rel_type.value.lower()},
                    raw_text=context.strip(),
                    extraction_method="code",
                    epistemic_category="observation",
                    derivation_depth=0,
                ))

        return relations

    def _apply_low_confidence_filter(self, entities: list) -> Tuple[list, list]:
        """Convert entities with confidence < 0.5 to UnknownEntity — per docs."""
        kept = []
        demoted = []
        for entity in entities:
            conf = entity.confidence.score if entity.confidence else 0.0
            if conf < 0.5 and entity.entity_type.value == "PERSON":
                demoted.append(entity)
            else:
                kept.append(entity)
        return kept, demoted

    def _find_or_create_person(self, name: str, source: SourceMetadata, pending_persons: dict = None) -> ExtractedEntity:
        """
        Find existing person entity or create new one.

        When pending_persons is provided, uses it for intra-method dedup instead
        of the global entity_index. This prevents pre-indexing which causes
        entities to be indexed but never exported to self.entities.

        Args:
            name: Person name
            source: Source metadata
            pending_persons: Optional dict[name, entity] for intra-method dedup
        """
        person_id = generate_id("PERSON", name)

        # Check global index first (existing entity from prior file or already exported)
        if person_id in self.entity_index:
            return self.entity_index[person_id]

        # Check pending persons (same file, not yet exported)
        if pending_persons is not None and name in pending_persons:
            return pending_persons[name]

        entity = ExtractedEntity(
            id=person_id,
            entity_type=EntityType.PERSON,
            name=name,
            aliases=[name],
            attributes={},
            confidence=ConfidenceSchema(score=1.0, basis=["Direct record"]),
            source=source,
            extraction_method="code",
        )
        # Track in pending (caller must add to entities list for export)
        if pending_persons is not None:
            pending_persons[name] = entity
        return entity

    def _extract_person_from_description(self, description: str) -> Optional[str]:
        """Extract a person identifier from CCTV description. Returns None for generic descriptions."""
        if not description:
            return None

        # Reject generic descriptions (age/gender only)
        age_match = re.search(r'(\d+)\s*years?', description)
        gender_match = re.search(r'\b(Male|Female|Man|Woman)\b', description, re.IGNORECASE)
        if age_match and gender_match:
            return None  # Too generic — store as attribute, not person entity

        # Check for a proper name (First Last pattern)
        name_match = re.search(r'\b([A-Z][a-z]{2,}\s+[A-Z][a-z]{2,})\b', description)
        if name_match:
            return name_match.group(1)

        return None  # Don't create person entities from raw descriptions

    def _detect_text_task(self, text: str) -> Optional[str]:
        """Determine if text is investigation evidence worth LLM extraction."""
        text_lower = text.lower()

        # Quick keyword check — no LLM call needed, just decides IF we extract
        # FIR: always use extract_fir (structured format needs specific prompt)
        fir_structure = ["first information report", "fir number", "police station", "complainant"]
        fir_matches = sum(1 for kw in fir_structure if kw in text_lower)
        if fir_matches >= 3:
            return "extract_fir"

        # Any other investigation text → extract_generic (handles all types)
        evidence_phrases = [
            "received a tip", "informant reported", "witness statement",
            "witness testimony", "observed the suspect", "identified the suspect",
            "complaint received", "alert received", "report from",
            "the accused", "the complainant", "section 420", "ipc section",
            "surveillance report", "observation report", "stakeout",
            "source:", "article draft", "key points", "publish",
            "arrested", "accused", "victim", "investigation",
            "journalist", "police officer", "suspect",
        ]
        if any(phrase in text_lower for phrase in evidence_phrases):
            return "extract_generic"

        return None

    def _convert_llm_parsed(
        self, parsed: dict, source: SourceMetadata, llm_result: dict
    ) -> Tuple[List, List]:
        """Convert LLM parsed JSON to entities and relations."""
        entities = []
        relations = []
        seen_entity_ids = set()
        entity_types_by_name = {}

        def add_entity(entity):
            if entity.id in seen_entity_ids:
                return
            seen_entity_ids.add(entity.id)
            entities.append(entity)

        # Convert entities
        for e in parsed.get("entities", []):
            entity_type_str = e.get("type", "UNKNOWN")
            # Normalize: map common LLM variants to standard EntityType values
            TYPE_ALIASES = {
                "phone": "PHONE", "phone_number": "PHONE", "phone number": "PHONE",
                "person": "PERSON", "name": "PERSON",
                "location": "LOCATION", "addr": "LOCATION", "address": "LOCATION",
                "account": "ACCOUNT", "bank_account": "ACCOUNT",
                "organization": "ORGANIZATION", "org": "ORGANIZATION",
                "device": "DEVICE", "imei": "DEVICE",
                "event": "EVENT", "case": "EVENT",
                "amount": "AMOUNT", "money": "AMOUNT",
                "date": "DATE", "time": "DATE",
            }
            normalized_type = TYPE_ALIASES.get(entity_type_str.lower(), entity_type_str.upper())
            try:
                entity_type = EntityType(normalized_type)
            except ValueError:
                entity_type = EntityType.UNKNOWN

            name = e.get("name", "").strip()
            if not name or len(name) < 2:
                continue
            entity_types_by_name[name] = normalized_type

            entity_id = generate_id(normalized_type, name)
            if entity_id in seen_entity_ids:
                continue
            seen_entity_ids.add(entity_id)

            entity = ExtractedEntity(
                id=entity_id,
                entity_type=entity_type,
                name=name,
                aliases=e.get("aliases", []),
                attributes=e.get("attributes", {}),
                confidence=ConfidenceSchema(
                    score=e.get("confidence", 0.5),
                    basis=["LLM extraction", f"provider:{llm_result.get('provider', 'unknown')}"],
                    source_reliability=0.7,
                ),
                source=source,
                extraction_method="llm",
            )
            entities.append(entity)

        # Convert relations
        for r in parsed.get("relations", []):
            rel_type_str = r.get("type", "ASSOCIATED_WITH")
            try:
                rel_type = RelationType(rel_type_str.upper())
            except ValueError:
                rel_type = RelationType.ASSOCIATED_WITH

            source_name = r.get("source", "").strip()
            target_name = r.get("target", "").strip()
            if not source_name or not target_name:
                continue

            source_type = r.get("source_type") or entity_types_by_name.get(source_name, "PERSON")
            target_type = r.get("target_type") or entity_types_by_name.get(target_name, "PERSON")

            relation = ExtractedRelation(
                id=generate_id("LLM_REL", f"{source_name}_{target_name}_{rel_type_str}"),
                source_entity_id=generate_id(source_type.upper(), source_name),
                target_entity_id=generate_id(target_type.upper(), target_name),
                relation_type=rel_type,
                confidence=ConfidenceSchema(
                    score=r.get("confidence", 0.5),
                    basis=["LLM extraction"],
                    source_reliability=0.7,
                ),
                source=source,
                attributes={**r.get("attributes", {}), "source_name": source_name, "target_name": target_name},
                extraction_method="llm",
                semantic_edge_type=get_semantic_edge_type(rel_type),
                temporal_info=extract_temporal_info(r.get("attributes", {})),
            )
            relations.append(relation)

        # Also extract structured entities from nested fields (complainant, accused, etc.)
        for field_name in ["complainant", "complaintant", "accused", "suspect", "offender", "victims"]:
            field_data = parsed.get(field_name, [])
            if isinstance(field_data, dict):
                field_data = [field_data]
            elif isinstance(field_data, str):
                # Flat format: field is a string name
                field_data = [{"name": field_data}]
            for person in field_data:
                if isinstance(person, dict) and person.get("name"):
                    entity = ExtractedEntity(
                        id=generate_id("PERSON", person["name"]),
                        entity_type=EntityType.PERSON,
                        name=person["name"],
                        aliases=[person["name"]],
                        attributes={**person, "extracted_from_field": field_name},
                        confidence=ConfidenceSchema(score=0.8, basis=["LLM extraction"]),
                        source=source,
                        extraction_method="llm",
                    )
                    add_entity(entity)

        # Extract entities from flat fields (name, address, phone, location, etc.)
        flat_person_fields = ["name", "complainant_name", "suspect_name", "victim_name"]
        for field_name in flat_person_fields:
            val = parsed.get(field_name, "")
            if val and isinstance(val, str) and len(val) > 2 and not val.startswith("Unknown"):
                entity = ExtractedEntity(
                    id=generate_id("PERSON", val),
                    entity_type=EntityType.PERSON,
                    name=val,
                    aliases=[val],
                    attributes={"extracted_from_field": field_name},
                    confidence=ConfidenceSchema(score=0.8, basis=["LLM extraction"]),
                    source=source,
                    extraction_method="llm",
                )
                add_entity(entity)

        # Extract location from flat field or nested incident
        loc = parsed.get("location", "") or parsed.get("fir_location", "")
        if not loc:
            incident = parsed.get("incident", {})
            if isinstance(incident, dict):
                loc = incident.get("location", "")
        if loc and isinstance(loc, str) and len(loc) > 2:
            entity = ExtractedEntity(
                id=generate_id("LOCATION", loc),
                entity_type=EntityType.LOCATION,
                name=loc,
                attributes={"extracted_from_field": "location"},
                confidence=ConfidenceSchema(score=0.8, basis=["LLM extraction"]),
                source=source,
                extraction_method="llm",
            )
            add_entity(entity)

        # Extract amounts from flat fields or nested incident
        for field_name in ["amount_lost", "amount", "fraud_amount"]:
            val = parsed.get(field_name, "")
            if not val:
                incident = parsed.get("incident", {})
                if isinstance(incident, dict):
                    val = incident.get("amount", "")
            if val:
                entity = ExtractedEntity(
                    id=generate_id("AMOUNT", str(val)),
                    entity_type=EntityType.AMOUNT,
                    name=f"Rs. {val}" if isinstance(val, (int, float)) else str(val),
                    attributes={"amount": val, "extracted_from_field": field_name},
                    confidence=ConfidenceSchema(score=0.8, basis=["LLM extraction"]),
                    source=source,
                    extraction_method="llm",
                )
                add_entity(entity)

        # Extract phone numbers from the text
        phones = re.findall(r'\b\d{10}\b', json.dumps(parsed))
        for phone in set(phones):
            entity = ExtractedEntity(
                id=generate_id("PHONE", phone),
                entity_type=EntityType.PHONE,
                name=phone,
                attributes={"phone_number": phone},
                confidence=ConfidenceSchema(score=1.0, basis=["Extracted from LLM response"]),
                source=source,
                extraction_method="llm",
            )
            add_entity(entity)

        return entities, relations

    def get_extraction_summary(self) -> dict:
        """Summary of all extractions."""
        entity_types = {}
        for e in self.entities:
            t = e.entity_type.value
            entity_types[t] = entity_types.get(t, 0) + 1

        relation_types = {}
        for r in self.relations:
            t = r.relation_type.value
            relation_types[t] = relation_types.get(t, 0) + 1

        return {
            "total_entities": len(self.entities),
            "total_relations": len(self.relations),
            "entity_types": entity_types,
            "relation_types": relation_types,
            "files_processed": len(self.extraction_log),
        }

    def export_for_db(self) -> dict:
        """Export all extracted data in DB-ready format."""
        return {
            "entities": [e.to_dict() for e in self.entities],
            "relations": [r.to_dict() for r in self.relations],
            "summary": self.get_extraction_summary(),
            "extraction_log": self.extraction_log,
            "export_time": datetime.now().isoformat(),
        }
