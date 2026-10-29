"""
Column Mapper — maps arbitrary columns to unified schema.
Pattern matching first (deterministic), LLM only for unknown columns.
"""

import re
import json
from typing import Dict, List, Optional


# Schema fields available for mapping
SCHEMA_FIELDS = [
    "phone", "timestamp", "location", "amount", "name", "account",
    "duration", "direction", "imei", "tower_id", "event_type",
    "description", "currency", "bank", "counterparty", "counterparty_name",
]


class ColumnMapper:
    """Maps arbitrary columns to unified schema using LLM."""

    def __init__(self, ai_caller=None):
        self.ai = ai_caller
        self.mappings: Dict[str, str] = {}
        self.unmapped: List[str] = []
        self.split_groups: Dict[str, List[str]] = {}
        self.column_details: Dict[str, dict] = {}
        self.extraction_hints: Dict[str, dict] = {}

    def detect_mapping(self, columns: List[str], sample_rows: List[dict] = None) -> Dict[str, str]:
        """Auto-detect column mappings. Pattern matching first, LLM for unknown."""
        self.mappings = {}
        self.unmapped = []
        self.split_groups = {}

        # Pattern matching (deterministic for known formats)
        self._pattern_match(columns)

        # LLM only for truly unknown columns (skip if pattern matching mapped enough)
        mapped_ratio = len(self.mappings) / max(len(columns), 1)
        if self.ai and sample_rows and len(self.unmapped) > 4 and mapped_ratio < 0.6:
            llm_result = self._llm_map(columns, sample_rows)
            if llm_result:
                llm_mappings = llm_result.get("mappings", {})
                self.column_details = llm_result.get("column_details", {})
                self.extraction_hints = llm_result.get("hints", {})
                # Only fill unmapped columns, never override pattern matches
                for col in list(self.unmapped):
                    if col in llm_mappings:
                        self.mappings[col] = self._normalize_field(llm_mappings[col])
                self.unmapped = [c for c in columns if c not in self.mappings]

        # Always detect split groups from column names
        self._detect_split_groups(columns)

        return self.mappings

    def _normalize_field(self, field: str) -> str:
        """Map LLM-invented field names to our exact schema."""
        aliases = {
            "caller": "phone", "callee": "phone",
            "caller_number": "phone", "callee_number": "phone",
            "from_number": "phone", "to_number": "phone",
            "caller_entity": "phone", "callee_entity": "phone",
            "subscriber_entity": "name", "subscriber": "name",
            "person": "name", "accused": "name", "suspect": "name",
            "victim": "name", "holder": "name",
            "cell_site": "location", "cell_site_entity": "location",
            "tower": "location", "cell_tower": "location",
            "device_entity": "imei",
            "status": "direction", "call_status": "direction",
            "txn_status": "direction", "flag_status": "direction",
            "notes": "description", "remark": "description",
            "incident_reference": "case_number", "incident": "case_number",
            "fir_number": "case_number", "case_reference": "case_number",
            "data_type": "other", "reasoning": "other",
            "currency": "other", "bank_name": "bank", "branch": "bank",
            "counterparty": "account", "counterparty_account": "account",
            "counterparty_name": "name",
            "call_amount": "amount", "call_type": "description",
            "ip_address": "ip_address", "organization": "other",
            "vehicle_plate": "vehicle_plate",
        }
        return aliases.get(field, field)

    def _detect_split_groups(self, columns: List[str]):
        """Auto-detect split columns from names like _A/_B/_C or _1/_2."""
        suffix_pattern = re.compile(r'^(.+?)[_\-](\d+|[a-zA-Z])$')
        groups = {}
        for col in columns:
            m = suffix_pattern.match(col)
            if m:
                base = m.group(1)
                groups.setdefault(base, []).append(col)
        for base, cols in groups.items():
            if len(cols) >= 2:
                self.split_groups[base] = cols

    def _pattern_match(self, columns: List[str]) -> Dict[str, str]:
        """Fast regex-based pattern matching."""
        patterns = {
            "phone": [r"^phone$", r"^msisdn$", r"^number$", r"^calling.?party$", r"^called.?party$", r"^caller$", r"^callee$", r"^caller.?number$", r"^callee.?number$", r"^src$", r"^dst$"],
            "timestamp": [r"time", r"date", r"datetime", r"timestamp"],
            "location": [r"location", r"tower", r"cell", r"site", r"address", r"place", r"area", r"city"],
            "amount": [r"amount", r"credit", r"debit", r"balance", r"sum", r"money", r"rs", r"inr", r"total"],
            "name": [r"^name$", r"^person$", r"^individual$", r"^accused$", r"^suspect$", r"^victim$", r"holder", r"counterparty.?name"],
            "account": [r"account", r"acc_no", r"acct", r"iban"],
            "duration": [r"duration", r"seconds", r"minutes", r"call_length", r"secs"],
            "direction": [r"direction", r"call.?type", r"txn.?type", r"transaction.?type"],
            "imei": [r"imei", r"device_id", r"handset", r"imeitrac"],
            "tower_id": [r"tower.?id", r"cell.?id", r"site.?id", r"lac", r"cid"],
            "description": [r"description", r"narration", r"details", r"remark", r"notes?$"],
            "bank": [r"bank", r"branch", r"ifsc"],
            "email": [r"email", r"e-mail", r"mail"],
            "vehicle_plate": [r"vehicle", r"plate", r"registration", r"reg_no", r"car.?no"],
            "ip_address": [r"ip.?address", r"ipv4", r"ipv6"],
            "case_number": [r"case", r"fir", r"case.?no", r"fir.?no"],
            "organization": [r"organization", r"company", r"firm", r"agency", r"department"],
            "flag_status": [r"flag.?status", r"status"],
        }

        # Detect split groups — columns ending with _A, _B, _C or _1, _2, _3
        split_pattern = re.compile(r'^(.+?)[_\-](\d+|[a-zA-Z])$')
        split_groups = {}  # field -> {suffix: original_col}

        for col in columns:
            col_clean = col.lower().strip()
            match = split_pattern.match(col_clean)
            if match:
                base = match.group(1)
                suffix = match.group(2)
                for field, pats in patterns.items():
                    for pat in pats:
                        if re.search(pat, base):
                            split_groups.setdefault(field, {})[suffix] = col
                            break
                    if field in split_groups and suffix in split_groups[field]:
                        break

        # Map split groups — all parts map to same schema field
        mapped_cols = set()
        mapped_fields = set()

        for field, parts in split_groups.items():
            sorted_parts = sorted(parts.items(), key=lambda x: x[0])
            for suffix, col in sorted_parts:
                self.mappings[col] = field
                mapped_cols.add(col)
            mapped_fields.add(field)
            self.split_groups[field] = [col for _, col in sorted_parts]

        # Map remaining columns (non-split)
        multi_fields = {"phone", "location", "amount", "name"}

        for col in columns:
            col_lower = re.sub(r'[^a-z0-9]', '', col.lower().strip())
            if col in mapped_cols:
                continue
            matched = False
            for field, pats in patterns.items():
                if field in mapped_fields and field not in multi_fields:
                    continue
                for pat in pats:
                    if re.search(pat, col_lower):
                        self.mappings[col] = field
                        mapped_fields.add(field)
                        mapped_cols.add(col)
                        matched = True
                        break
                if matched:
                    break
            if not matched:
                self.unmapped.append(col)

        return self.mappings

    def _llm_map(self, columns: List[str], sample_rows: List[dict]) -> Optional[dict]:
        """Use LLM to map columns to schema fields."""
        if not self.ai:
            return None

        prompt = f"""Map these columns to EXACT schema fields. Do NOT invent new field names.

EXACT SCHEMA FIELDS (use ONLY these):
phone, timestamp, location, amount, name, account, duration, direction, imei, tower_id, description, email, vehicle_plate, ip_address, case_number, organization, other

COLUMNS: {json.dumps(columns)}

SAMPLE DATA:
{json.dumps(sample_rows[:3], indent=2, default=str)}

OUTPUT FORMAT (JSON only):
{{
  "mappings": {{"column_name": "exact_schema_field"}},
  "column_details": {{
    "column_name": {{
      "schema_field": "exact_schema_field",
      "connects_to": "caller_entity|callee_entity|location_entity|relation_attribute",
      "role": "caller|callee|timestamp|location|description|extra",
      "reasoning": "why"
    }}
  }},
  "data_type": "cdr|bank|cctv|device|social|generic"
}}

RULES:
1. Use ONLY the exact schema field names listed above
2. ALL columns must appear in mappings — no column left out
3. Output ONLY valid JSON"""

        try:
            result = self.ai.extract(
                task="map_columns",
                text=prompt,
                source_file="column_mapper",
            )
            if result.get("status") == "success" and result.get("parsed"):
                return result["parsed"]
        except Exception as e:
            print(f"[MAPPER] LLM mapping failed: {e}")

        return None

    def get_all_attributes(self, row: dict) -> dict:
        """Get ALL attributes from a row. Handles split groups and duplicate mappings."""
        result = {}

        # Track duplicate fields to suffix them
        field_counts = {}
        for col, value in row.items():
            if value is None or str(value).strip() == "":
                continue
            val = str(value).strip()
            key = self.mappings.get(col, col)
            if key in field_counts:
                field_counts[key] += 1
                result[f"{key}_{field_counts[key]}"] = val
            else:
                field_counts[key] = 1
                result[key] = val

        # Second pass: combine split groups
        for field, group_cols in self.split_groups.items():
            parts = []
            for col in group_cols:
                val = row.get(col, "")
                if val and str(val).strip():
                    parts.append(str(val).strip())
            if parts:
                result[field] = " ".join(parts)
                # Also store individual parts
                for col in group_cols:
                    original_val = row.get(col, "")
                    if original_val and str(original_val).strip():
                        result[col] = str(original_val).strip()

        return result

    def get_field(self, row: dict, field: str, default: str = "") -> str:
        """Get a mapped field value from a row."""
        if field in row:
            return str(row[field]).strip()
        for col, schema_field in self.mappings.items():
            if schema_field == field and col in row:
                return str(row[col]).strip()
        return default

    def get_phone(self, row: dict) -> str:
        return self.get_field(row, "phone")

    def get_timestamp(self, row: dict) -> str:
        return self.get_field(row, "timestamp")

    def get_location(self, row: dict) -> str:
        return self.get_field(row, "location")

    def get_amount(self, row: dict) -> str:
        return self.get_field(row, "amount")

    def get_name(self, row: dict) -> str:
        return self.get_field(row, "name")

    def get_account(self, row: dict) -> str:
        return self.get_field(row, "account")

    def get_duration(self, row: dict) -> str:
        return self.get_field(row, "duration")

    def get_direction(self, row: dict) -> str:
        return self.get_field(row, "direction")

    def get_hint(self, column: str) -> dict:
        """Get extraction hints for a column."""
        return self.extraction_hints.get(column, {})
