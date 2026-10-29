"""
File Context Generator
Creates a context record for each file that the reasoner can use.
NOT extraction — this is about understanding what the file IS and what it CONTAINS.
"""

import json
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass, field, asdict
from datetime import datetime


@dataclass
class FileContext:
    """What the reasoner needs to know about a file."""
    file_name: str
    file_type: str
    file_path: str

    # What is this file?
    description: str  # "FIR filed by Priya Mehta on 13/03/2024"
    document_type: str  # "fir", "cdr", "bank_record", "cctv_log", "social_media", "device_extraction", "criminal_history"

    # What did extraction get?
    entities_extracted: List[Dict] = field(default_factory=list)  # [{id, type, name}]
    relations_extracted: List[Dict] = field(default_factory=list)  # [{id, type, source, target}]
    extraction_method: str = "code"  # "code", "llm", "both"

    # Key evidence this file contains
    key_evidence: List[str] = field(default_factory=list)
    # ["Phone 9876543210 called victim on 12/03/2024",
    #  "Rs. 67,000 debited from victim's account"]

    # What this file proves / suggests
    suggests: List[str] = field(default_factory=list)
    # ["Rakesh Kumar involved in fraud",
    #  "Money flowed from victim to Rakesh's account"]

    # What this file DOESN'T contain (gaps)
    missing: List[str] = field(default_factory=list)
    # ["No bank records for victim",
    #  "No CCTV footage of suspect"]

    # Links to other files (cross-references)
    linked_files: List[str] = field(default_factory=list)
    # ["02_CDR_Rakesh.csv (same phone number)",
    #  "05_Bank_Rakesh.csv (same account)"]

    # Reliability assessment
    source_reliability: float = 0.0  # 0-1
    reliability_notes: str = ""

    # Metadata
    ingestion_time: str = ""
    record_count: int = 0
    raw_summary: str = ""  # LLM-generated summary (if available)


class FileContextGenerator:
    """Generates context records for each file."""

    def __init__(self):
        self.contexts: Dict[str, FileContext] = {}

    def generate_all(self, ingestion_results: List[dict], extraction_output: dict) -> List[FileContext]:
        """Generate context for all ingested files."""
        # Index entities and relations by source file
        entities_by_file = {}
        for e in extraction_output.get("entities", []):
            src = e.get("source", {}).get("file_name", "unknown")
            if src not in entities_by_file:
                entities_by_file[src] = []
            entities_by_file[src].append(e)

        relations_by_file = {}
        for r in extraction_output.get("relations", []):
            src = r.get("source", {}).get("file_name", "unknown")
            if src not in relations_by_file:
                relations_by_file[src] = []
            relations_by_file[src].append(r)

        # Generate context for each file
        for file_info in ingestion_results:
            file_name = file_info.get("file_name", "unknown")
            file_type = file_info.get("detected_type", "unknown")
            file_path = file_info.get("file_path", "")

            entities = entities_by_file.get(file_name, [])
            relations = relations_by_file.get(file_name, [])

            context = self._build_context(file_info, entities, relations)
            self.contexts[file_name] = context

        # Cross-reference files
        self._find_linked_files()

        return list(self.contexts.values())

    def _build_context(self, file_info: dict, entities: list, relations: list) -> FileContext:
        """Build context record for a single file."""
        file_name = file_info.get("file_name", "unknown")
        file_type = file_info.get("detected_type", "unknown")
        content = file_info.get("content", {})

        # Detect document type
        doc_type = self._detect_doc_type(file_name, content)

        # Build description
        description = self._build_description(file_name, doc_type, content, entities)

        # Build entity summary
        entity_summary = [
            {"id": e.get("id", ""), "type": e.get("entity_type", ""), "name": e.get("name", "")}
            for e in entities
        ]

        # Build relation summary
        relation_summary = [
            {"id": r.get("id", ""), "type": r.get("relation_type", ""),
             "source": r.get("source_entity_id", ""), "target": r.get("target_entity_id", "")}
            for r in relations
        ]

        # Extract key evidence
        key_evidence = self._extract_key_evidence(doc_type, entities, relations, content)

        # Extract suggestions
        suggests = self._extract_suggestions(doc_type, entities, relations)

        # Identify gaps
        missing = self._identify_gaps(doc_type, entities, content)

        # Assess reliability
        reliability, notes = self._assess_reliability(doc_type, file_info)

        # Determine extraction method
        methods = set(e.get("extraction_method", "code") for e in entities)
        extraction_method = "both" if len(methods) > 1 else (methods.pop() if methods else "code")

        return FileContext(
            file_name=file_name,
            file_type=file_type,
            file_path=file_info.get("file_path", ""),
            description=description,
            document_type=doc_type,
            entities_extracted=entity_summary,
            relations_extracted=relation_summary,
            extraction_method=extraction_method,
            key_evidence=key_evidence,
            suggests=suggests,
            missing=missing,
            source_reliability=reliability,
            reliability_notes=notes,
            ingestion_time=file_info.get("ingestion_time", datetime.now().isoformat()),
            record_count=file_info.get("record_count", 0),
        )

    def _detect_doc_type(self, file_name: str, content: dict) -> str:
        """Detect document type from filename and content."""
        name_lower = file_name.lower()

        if "fir" in name_lower:
            return "fir"
        if "cdr" in name_lower:
            return "cdr"
        if "bank" in name_lower:
            return "bank_record"
        if "cctv" in name_lower:
            return "cctv_log"
        if "social" in name_lower or "instagram" in name_lower:
            return "social_media"
        if "device" in name_lower:
            return "device_extraction"
        if "criminal" in name_lower:
            return "criminal_history"
        if "crime_story" in name_lower:
            return "case_narrative"

        # Check content for clues
        text = content.get("content", "")
        if isinstance(text, str):
            if "first information report" in text.lower():
                return "fir"
            if "caller_number" in str(content.get("columns", [])):
                return "cdr"
            if "transaction_id" in str(content.get("columns", [])):
                return "bank_record"

        return "unknown"

    def _build_description(self, file_name: str, doc_type: str, content: dict, entities: list) -> str:
        """Build human-readable description of what this file is."""
        entity_count = len(entities)

        if doc_type == "fir":
            # Try to get FIR details
            complainant = ""
            for e in entities:
                if e.get("attributes", {}).get("extracted_from_field") == "complainant":
                    complainant = e.get("name", "")
                    break
            date = ""
            for e in entities:
                if e.get("entity_type") == "UNKNOWN" and "date" in str(e.get("attributes", {})):
                    date = e.get("name", "")
                    break
            return f"FIR document. {f'Filed by {complainant}.' if complainant else ''} {f'Date: {date}.' if date else ''} {entity_count} entities extracted."

        if doc_type == "cdr":
            return f"Call Detail Records. {entity_count} entities extracted from {content.get('row_count', '?')} call records."

        if doc_type == "bank_record":
            return f"Bank transaction records. {entity_count} entities extracted from {content.get('row_count', '?')} transactions."

        if doc_type == "cctv_log":
            return f"CCTV entry/exit log. {entity_count} entities extracted from {content.get('row_count', '?')} records."

        if doc_type == "device_extraction":
            return f"Mobile device extraction. {entity_count} entities extracted (contacts, chats, photos, SMS)."

        if doc_type == "social_media":
            return f"Social media data. {entity_count} entities extracted (profile, posts, DMs)."

        if doc_type == "criminal_history":
            return f"Criminal history records. {entity_count} entities extracted (prior cases, associations)."

        if doc_type == "case_narrative":
            return f"Case narrative document. {entity_count} entities extracted."

        return f"Document. {entity_count} entities extracted."

    def _extract_key_evidence(self, doc_type: str, entities: list, relations: list, content: dict) -> list:
        """Extract key evidence points from this file."""
        evidence = []

        if doc_type == "fir":
            for e in entities:
                if e.get("entity_type") == "PHONE":
                    evidence.append(f"Phone number {e['name']} mentioned in FIR")
                if e.get("entity_type") == "UNKNOWN" and "₹" in e.get("name", ""):
                    evidence.append(f"Amount {e['name']} involved in fraud")

        if doc_type == "cdr":
            # Find most frequent contacts
            phone_counts = {}
            for r in relations:
                if r.get("relation_type") == "CALLED":
                    target = r.get("target_entity_id", "")
                    phone_counts[target] = phone_counts.get(target, 0) + 1
            if phone_counts:
                top_contact = max(phone_counts, key=phone_counts.get)
                evidence.append(f"Most frequent contact: {top_contact} ({phone_counts[top_contact]} calls)")

        if doc_type == "bank_record":
            for e in entities:
                attrs = e.get("attributes", {})
                if attrs.get("is_flagged"):
                    evidence.append(f"Flagged transaction: {e.get('name', '')}")

        if doc_type == "cctv_log":
            for r in relations:
                if r.get("relation_type") == "VISITED":
                    attrs = r.get("attributes", {})
                    if attrs.get("duration_minutes") and int(attrs.get("duration_minutes", 0) or 0) > 180:
                        evidence.append(f"Extended visit: {attrs.get('duration_minutes')} minutes")

        if doc_type == "device_extraction":
            for e in entities:
                attrs = e.get("attributes", {})
                if attrs.get("requires_vision_model"):
                    evidence.append(f"Photo evidence pending: {attrs.get('filename', e.get('name', ''))}")

        if doc_type == "criminal_history":
            for e in entities:
                score = e.get("attributes", {}).get("criminal_score", 0)
                if score and score > 0.5:
                    evidence.append(f"High criminal score: {e.get('name', '')} (score: {score})")

        return evidence

    def _extract_suggestions(self, doc_type: str, entities: list, relations: list) -> list:
        """What this file suggests about the case."""
        suggestions = []

        if doc_type == "fir":
            suggestions.append("Documents initial complaint and incident details")
            suggestions.append("Establishes victim and alleged crime")

        if doc_type == "cdr":
            suggestions.append("Reveals communication patterns between suspects")
            suggestions.append("Provides timeline and location data")

        if doc_type == "bank_record":
            suggestions.append("Shows money flow patterns")
            suggestions.append("Can trace funds between accounts")

        if doc_type == "cctv_log":
            suggestions.append("Provides physical presence verification")
            suggestions.append("Can confirm/deny meetings between suspects")

        if doc_type == "device_extraction":
            suggestions.append("Contains direct evidence of criminal planning")
            suggestions.append("Chat messages may prove conspiracy")

        if doc_type == "social_media":
            suggestions.append("Shows lifestyle and movements of suspect")
            suggestions.append("DMs may contain evidence of coordination")

        if doc_type == "criminal_history":
            suggestions.append("Establishes pattern of criminal behavior")
            suggestions.append("Links suspect to prior cases")

        return suggestions

    def _identify_gaps(self, doc_type: str, entities: list, content: dict) -> list:
        """What this file DOESN'T contain."""
        gaps = []

        if doc_type == "fir":
            if not any(e.get("entity_type") == "PERSON" for e in entities):
                gaps.append("No suspect identified in FIR")

        if doc_type == "cdr":
            if content.get("row_count", 0) < 10:
                gaps.append("Limited call history available")

        if doc_type == "bank_record":
            gaps.append("Does not show cash transactions")
            gaps.append("Does not show hawala transfers")

        if doc_type == "device_extraction":
            gaps.append("Deleted messages may be missing")
            gaps.append("Encrypted chats not captured")

        if doc_type == "social_media":
            gaps.append("Only public/available data")
            gaps.append("Private accounts may be excluded")

        return gaps

    def _assess_reliability(self, doc_type: str, file_info: dict) -> tuple:
        """Assess source reliability."""
        reliability_map = {
            "fir": (0.9, "Official police document"),
            "cdr": (0.95, "Direct telecom provider record"),
            "bank_record": (0.95, "Direct bank record"),
            "cctv_log": (0.85, "Automated sensor record"),
            "device_extraction": (0.9, "Forensic extraction"),
            "social_media": (0.7, "Platform data, may be incomplete"),
            "criminal_history": (0.9, "Official police records"),
            "case_narrative": (0.8, "Investigative summary"),
        }
        return reliability_map.get(doc_type, (0.5, "Unknown source"))

    def _find_linked_files(self):
        """Find cross-references between files."""
        # Build entity-to-file index
        entity_files = {}
        for fname, ctx in self.contexts.items():
            for e in ctx.entities_extracted:
                eid = e.get("id", "")
                if eid not in entity_files:
                    entity_files[eid] = []
                entity_files[eid].append(fname)

        # Find files sharing entities
        for fname, ctx in self.contexts.items():
            linked = set()
            for e in ctx.entities_extracted:
                eid = e.get("id", "")
                for other_file in entity_files.get(eid, []):
                    if other_file != fname:
                        linked.add(other_file)
            ctx.linked_files = sorted(linked)

    def export_for_reasoner(self) -> dict:
        """Export all contexts formatted for reasoner consumption."""
        return {
            "files": {name: asdict(ctx) for name, ctx in self.contexts.items()},
            "summary": {
                "total_files": len(self.contexts),
                "by_type": {},
                "cross_references": sum(len(ctx.linked_files) for ctx in self.contexts.values()),
            },
            "generated_at": datetime.now().isoformat(),
        }
