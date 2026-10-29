"""
Multi-Signal Disambiguator
Uses secondary signals beyond name to resolve entity identity:
phone, vehicle, DOB, location overlap, case overlap.
"""

from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
import re


@dataclass
class DisambiguationSignal:
    """A single signal used for entity disambiguation."""
    signal_type: str        # "phone", "vehicle", "dob", "location", "case", "name"
    confidence: float       # 0.0-1.0, how strong this signal is
    evidence: str           # Human-readable description
    source_entity_ids: List[str] = field(default_factory=list)


@dataclass
class DisambiguationResult:
    """Result of multi-signal disambiguation."""
    entity_id_a: str
    entity_id_b: str
    total_score: float              # Weighted combination of all signals
    signals: List[DisambiguationSignal]
    recommendation: str             # "merge", "review", "reject"
    explanation: str                # Human-readable explanation


class MultiSignalDisambiguator:
    """
    Disambiguates entity pairs using multiple signals.
    Each signal contributes a weighted score to the final decision.
    """

    # Signal weights (sum = 1.0)
    WEIGHTS = {
        "name": 0.25,
        "phone": 0.25,
        "vehicle": 0.10,
        "dob": 0.15,
        "location": 0.15,
        "case": 0.10,
    }

    # Indian vehicle plate regex: XX-00-XX-0000 or XX-XX-0000
    VEHICLE_PATTERN = re.compile(
        r'^[A-Z]{2}[\s-]?\d{1,2}[\s-]?[A-Z]{1,2}[\s-]?\d{4}$', re.IGNORECASE
    )

    # DOB patterns
    DOB_PATTERNS = [
        re.compile(r'\d{1,2}[/-]\d{1,2}[/-]\d{4}'),      # DD/MM/YYYY
        re.compile(r'\d{4}[/-]\d{1,2}[/-]\d{1,2}'),      # YYYY/MM/DD
        re.compile(r'\d{1,2}\s+\w+\s+\d{4}'),             # 15 March 2026
    ]

    def _extract_phones(self, entity: dict) -> List[str]:
        """Extract normalized phone numbers from entity."""
        phones = []
        attrs = entity.get("attributes", {})

        for key in ["phone", "phone_number", "mobile", "contact_number"]:
            val = attrs.get(key, "")
            if val:
                normalized = self._normalize_phone(str(val))
                if normalized:
                    phones.append(normalized)

        # Check aliases
        for alias in entity.get("aliases", []):
            if re.match(r'^\d{10}$', alias):
                phones.append(alias)

        return list(set(phones))

    def _extract_vehicles(self, entity: dict) -> List[str]:
        """Extract vehicle numbers from entity."""
        vehicles = []
        attrs = entity.get("attributes", {})

        for key in ["vehicle_number", "vehicle", "car_number", "plate"]:
            val = str(attrs.get(key, "")).strip()
            if val and self.VEHICLE_PATTERN.match(val):
                vehicles.append(val.upper().replace(" ", "").replace("-", ""))

        return list(set(vehicles))

    def _extract_dob(self, entity: dict) -> Optional[str]:
        """Extract date of birth from entity."""
        attrs = entity.get("attributes", {})
        for key in ["dob", "date_of_birth", "birth_date", "birthdate"]:
            val = str(attrs.get(key, "")).strip()
            if val:
                return self._normalize_date(val)
        return None

    def _extract_locations(self, entity: dict) -> List[str]:
        """Extract location identifiers from entity."""
        locations = []
        attrs = entity.get("attributes", {})

        for key in ["location", "address", "city", "area", "district", "state"]:
            val = str(attrs.get(key, "")).strip().lower()
            if val and len(val) > 2:
                locations.append(val)

        # Check source file for location hints (pattern-based, not hardcoded)
        source = entity.get("source", {})
        if isinstance(source, dict):
            fname = source.get("file_name", "").lower()
            # Detect location patterns in filenames: city names, state abbreviations
            # Pattern: word followed by common Indian location indicators
            import re
            loc_patterns = re.findall(
                r'\b([a-z]+(?:\s+[a-z]+)?)\b(?=_(?:cd|cdr|bank|fir|device|social|cctv|text|witness|surveillance|journalist))',
                fname
            )
            for loc in loc_patterns:
                if len(loc) > 2:
                    locations.append(loc)

        return list(set(locations))

    def _extract_cases(self, entity: dict) -> List[str]:
        """Extract case/FIR numbers from entity."""
        cases = []
        attrs = entity.get("attributes", {})

        for key in ["case_number", "fir_number", "case", "fir", "case_id"]:
            val = str(attrs.get(key, "")).strip()
            if val:
                cases.append(val.upper())

        return list(set(cases))

    def _normalize_phone(self, phone: str) -> Optional[str]:
        """Normalize phone to 10-digit Indian format."""
        digits = re.sub(r'\D', '', phone)
        if len(digits) == 12 and digits.startswith('91'):
            digits = digits[2:]
        elif len(digits) == 11 and digits.startswith('0'):
            digits = digits[1:]
        if len(digits) == 10 and digits[0] in '6789':
            return digits
        return None

    def _normalize_date(self, date_str: str) -> str:
        """Normalize date to YYYY-MM-DD format."""
        # Try common formats
        for fmt in ["%d/%m/%Y", "%d-%m-%Y", "%Y/%m/%d", "%Y-%m-%d",
                     "%d %B %Y", "%d %b %Y"]:
            try:
                from datetime import datetime
                dt = datetime.strptime(date_str.strip(), fmt)
                return dt.strftime("%Y-%m-%d")
            except ValueError:
                continue
        return date_str.strip()

    def _compute_signal(self, entity_a: dict, entity_b: dict, signal_type: str) -> Optional[DisambiguationSignal]:
        """Compute a single disambiguation signal between two entities."""
        if signal_type == "name":
            return self._signal_name(entity_a, entity_b)
        elif signal_type == "phone":
            return self._signal_phone(entity_a, entity_b)
        elif signal_type == "vehicle":
            return self._signal_vehicle(entity_a, entity_b)
        elif signal_type == "dob":
            return self._signal_dob(entity_a, entity_b)
        elif signal_type == "location":
            return self._signal_location(entity_a, entity_b)
        elif signal_type == "case":
            return self._signal_case(entity_a, entity_b)
        return None

    def _signal_name(self, a: dict, b: dict) -> DisambiguationSignal:
        """Name similarity signal."""
        name_a = a.get("name", "").lower()
        name_b = b.get("name", "").lower()

        if name_a == name_b:
            return DisambiguationSignal("name", 1.0, f"Exact name match: '{name_a}'")

        # Conjunction guard: "Rakesh and Suresh" is TWO people, not one
        if " and " in name_a or " and " in name_b:
            return DisambiguationSignal("name", 0.05, f"Conjunction name detected: '{name_a}' or '{name_b}'")

        if name_a in name_b or name_b in name_a:
            shorter = min(len(name_a), len(name_b))
            longer = max(len(name_a), len(name_b))
            ratio = shorter / longer

            # Area-within-city pattern: "Delhi" in "Connaught Place, Delhi"
            # The shorter name is a generic substring of the longer name
            # These are DIFFERENT locations (area vs city)
            if ratio < 0.5:
                conf = 0.15  # Very weak — likely different locations
            elif shorter < 5:
                # Short name contained in longer: "Priya" in "Priya Mehta" = OK
                # But "Rajesh" in "Rajesh Electronics" = NOT same
                conf = 0.3 + 0.2 * ratio
            else:
                conf = 0.6 + 0.3 * ratio
            return DisambiguationSignal("name", conf, f"Partial name match: '{name_a}' ↔ '{name_b}'")

        # Check aliases
        aliases_a = set(a.get("aliases", []))
        aliases_b = set(b.get("aliases", []))
        if aliases_a & aliases_b:
            return DisambiguationSignal("name", 0.80, f"Shared alias: {aliases_a & aliases_b}")

        # Jaro-Winkler
        from .rule_pass import jaro_winkler
        score = jaro_winkler(name_a, name_b)
        if score > 0.75:
            return DisambiguationSignal("name", score, f"Fuzzy name match: '{name_a}' ↔ '{name_b}' (similarity: {score:.2f})")

        return DisambiguationSignal("name", 0.1, f"Name mismatch: '{name_a}' vs '{name_b}'")

    def _signal_phone(self, a: dict, b: dict) -> Optional[DisambiguationSignal]:
        """Phone number signal."""
        phones_a = set(self._extract_phones(a))
        phones_b = set(self._extract_phones(b))

        if not phones_a and not phones_b:
            return None

        shared = phones_a & phones_b
        if shared:
            return DisambiguationSignal("phone", 0.95, f"Shared phone: {shared}", list(shared))

        if phones_a and phones_b:
            return DisambiguationSignal("phone", 0.1, f"Different phones: {phones_a} vs {phones_b}")

        return None

    def _signal_vehicle(self, a: dict, b: dict) -> Optional[DisambiguationSignal]:
        """Vehicle number signal."""
        vehicles_a = set(self._extract_vehicles(a))
        vehicles_b = set(self._extract_vehicles(b))

        if not vehicles_a and not vehicles_b:
            return None

        shared = vehicles_a & vehicles_b
        if shared:
            return DisambiguationSignal("vehicle", 0.90, f"Shared vehicle: {shared}")

        return None

    def _signal_dob(self, a: dict, b: dict) -> Optional[DisambiguationSignal]:
        """Date of birth signal."""
        dob_a = self._extract_dob(a)
        dob_b = self._extract_dob(b)

        if not dob_a and not dob_b:
            return None

        if dob_a and dob_b:
            if dob_a == dob_b:
                return DisambiguationSignal("dob", 0.90, f"Same DOB: {dob_a}")
            else:
                return DisambiguationSignal("dob", 0.05, f"Different DOB: {dob_a} vs {dob_b}")

        return None

    def _signal_location(self, a: dict, b: dict) -> Optional[DisambiguationSignal]:
        """Location overlap signal."""
        locs_a = set(self._extract_locations(a))
        locs_b = set(self._extract_locations(b))

        if not locs_a and not locs_b:
            return None

        # Get entity names for containment check
        name_a = a.get("name", "").lower()
        name_b = b.get("name", "").lower()

        # Check for significant location overlap
        # A shared location counts only if it's a significant portion of BOTH entity names
        significant_matches = set()
        weak_matches = set()

        for la in locs_a:
            for lb in locs_b:
                if la == lb:
                    # Exact match — check if it's a significant portion of both names
                    shorter_name = min(len(name_a), len(name_b))
                    if shorter_name > 0 and len(la) / shorter_name >= 0.5:
                        significant_matches.add(la)
                    else:
                        weak_matches.add(la)
                elif la in lb or lb in la:
                    # Substring match — check containment ratio
                    shorter = min(len(la), len(lb))
                    longer = max(len(la), len(lb))
                    if shorter >= 3 and shorter / longer >= 0.6:
                        significant_matches.add(f"{la}⊂{lb}")

        if significant_matches:
            # Check if matches came from entity attributes (stronger)
            attrs_a = a.get("attributes", {})
            attrs_b = b.get("attributes", {})
            specific_count = 0
            for m in significant_matches:
                base = m.split("⊂")[0] if "⊂" in m else m
                in_a = any(base in str(v).lower() for v in attrs_a.values() if isinstance(v, str))
                in_b = any(base in str(v).lower() for v in attrs_b.values() if isinstance(v, str))
                if in_a and in_b:
                    specific_count += 1

            if specific_count > 0:
                conf = min(0.85, 0.5 + 0.15 * specific_count)
                return DisambiguationSignal("location", conf, f"Shared locations: {significant_matches}")
            else:
                return DisambiguationSignal("location", 0.4, f"Location overlap: {significant_matches}")

        if weak_matches:
            return DisambiguationSignal("location", 0.25, f"Weak location overlap: {weak_matches}")

        return DisambiguationSignal("location", 0.2, f"No location overlap")

    def _signal_case(self, a: dict, b: dict) -> Optional[DisambiguationSignal]:
        """Case/FIR overlap signal."""
        cases_a = set(self._extract_cases(a))
        cases_b = set(self._extract_cases(b))

        if not cases_a and not cases_b:
            return None

        shared = cases_a & cases_b
        if shared:
            return DisambiguationSignal("case", 0.85, f"Shared case: {shared}")

        # Different case numbers = strong reject signal
        return DisambiguationSignal("case", 0.1, f"Different cases: {cases_a} vs {cases_b}")

    def disambiguate(self, entity_a: dict, entity_b: dict) -> DisambiguationResult:
        """
        Full multi-signal disambiguation between two entities.
        Returns ranked result with all signals and recommendation.
        """
        signals = []
        weighted_sum = 0.0
        weight_total = 0.0

        # HARD REJECT: Different entity types cannot merge
        type_a = entity_a.get("entity_type", "")
        type_b = entity_b.get("entity_type", "")
        if type_a and type_b and type_a != type_b:
            return DisambiguationResult(
                entity_id_a=entity_a.get("id", ""),
                entity_id_b=entity_b.get("id", ""),
                total_score=0.0,
                signals=[],
                recommendation="reject",
                explanation=f"Type mismatch: {type_a} vs {type_b} — different entity types cannot merge",
            )

        # HARD REJECT: Specific type-specific guards
        if type_a == "AMOUNT":
            # Different amounts should never merge
            name_a = entity_a.get("name", "").strip()
            name_b = entity_b.get("name", "").strip()
            if name_a != name_b:
                return DisambiguationResult(
                    entity_id_a=entity_a.get("id", ""),
                    entity_id_b=entity_b.get("id", ""),
                    total_score=0.0,
                    signals=[],
                    recommendation="reject",
                    explanation=f"Different amounts: {name_a} vs {name_b}",
                )

        if type_a == "ACCOUNT":
            # Different accounts should not merge
            name_a = entity_a.get("name", "").strip()
            name_b = entity_b.get("name", "").strip()
            accs_a = set(entity_a.get("accounts", []))
            accs_b = set(entity_b.get("accounts", []))
            # If names are different AND they share no account numbers, reject
            if name_a != name_b:
                if accs_a and accs_b and not (accs_a & accs_b):
                    return DisambiguationResult(
                        entity_id_a=entity_a.get("id", ""),
                        entity_id_b=entity_b.get("id", ""),
                        total_score=0.0,
                        signals=[],
                        recommendation="reject",
                        explanation=f"Different accounts: {name_a} vs {name_b}",
                    )
                # If no account data available, different names = different accounts
                if not accs_a and not accs_b and name_a != name_b:
                    return DisambiguationResult(
                        entity_id_a=entity_a.get("id", ""),
                        entity_id_b=entity_b.get("id", ""),
                        total_score=0.0,
                        signals=[],
                        recommendation="reject",
                        explanation=f"Different account names: {name_a} vs {name_b}",
                    )

        if type_a == "DATE":
            # Different dates should not merge (except aliases like "March 15" = "March 15, 2024")
            name_a = entity_a.get("name", "").strip().lower()
            name_b = entity_b.get("name", "").strip().lower()
            if name_a != name_b and name_a not in name_b and name_b not in name_a:
                return DisambiguationResult(
                    entity_id_a=entity_a.get("id", ""),
                    entity_id_b=entity_b.get("id", ""),
                    total_score=0.0,
                    signals=[],
                    recommendation="reject",
                    explanation=f"Different dates: {name_a} vs {name_b}",
                )

        if type_a == "EVENT":
            # Different case numbers should not merge
            name_a = entity_a.get("name", "").strip().lower()
            name_b = entity_b.get("name", "").strip().lower()
            if name_a != name_b:
                # Check if they're both case entries
                if ("case" in name_a or "case" in name_b) and name_a != name_b:
                    return DisambiguationResult(
                        entity_id_a=entity_a.get("id", ""),
                        entity_id_b=entity_b.get("id", ""),
                        total_score=0.0,
                        signals=[],
                        recommendation="reject",
                        explanation=f"Different cases: {name_a} vs {name_b}",
                    )
                # Different social media posts should not merge
                if ("post:" in name_a or "post:" in name_b) and name_a != name_b:
                    return DisambiguationResult(
                        entity_id_a=entity_a.get("id", ""),
                        entity_id_b=entity_b.get("id", ""),
                        total_score=0.0,
                        signals=[],
                        recommendation="reject",
                        explanation=f"Different posts: {name_a[:40]} vs {name_b[:40]}",
                    )

        if type_a == "LOCATION":
            # Different cell towers should not merge
            name_a = entity_a.get("name", "").strip().lower()
            name_b = entity_b.get("name", "").strip().lower()
            if re.match(r'^(del|up|mum|blr|che|kol)_twr_', name_a) or re.match(r'^(del|up|mum|blr|che|kol)_twr_', name_b):
                if name_a != name_b:
                    return DisambiguationResult(
                        entity_id_a=entity_a.get("id", ""),
                        entity_id_b=entity_b.get("id", ""),
                        total_score=0.0,
                        signals=[],
                        recommendation="reject",
                        explanation=f"Different cell towers: {name_a} vs {name_b}",
                    )
            # Geographic scale check: regions vs cities/states should not merge
            REGION_PATTERNS = [
                r'south\s+india', r'north\s+india', r'east\s+india', r'west\s+india',
                r'central\s+india', r'northeast\s+india', r'northwest\s+india',
            ]
            is_region_a = any(re.search(p, name_a) for p in REGION_PATTERNS)
            is_region_b = any(re.search(p, name_b) for p in REGION_PATTERNS)
            if is_region_a != is_region_b:
                # One is a region, other is not — different geographic scales
                return DisambiguationResult(
                    entity_id_a=entity_a.get("id", ""),
                    entity_id_b=entity_b.get("id", ""),
                    total_score=0.0,
                    signals=[],
                    recommendation="reject",
                    explanation=f"Different geographic scales: {name_a} (region) vs {name_b} (not region)",
                )
            # Different locations with no name overlap should not merge
            if name_a != name_b and name_a not in name_b and name_b not in name_a:
                # Only merge if they share a significant substring (e.g., "Delhi" in "New Delhi")
                shared_words = set(name_a.split()) & set(name_b.split())
                if len(shared_words) < 1:
                    return DisambiguationResult(
                        entity_id_a=entity_a.get("id", ""),
                        entity_id_b=entity_b.get("id", ""),
                        total_score=0.0,
                        signals=[],
                        recommendation="reject",
                        explanation=f"Different locations: {name_a} vs {name_b}",
                    )

        for signal_type, weight in self.WEIGHTS.items():
            signal = self._compute_signal(entity_a, entity_b, signal_type)
            if signal:
                signals.append(signal)
                weighted_sum += signal.confidence * weight
                weight_total += weight

        total_score = weighted_sum / weight_total if weight_total > 0 else 0.0

        # ENHANCED name signal: penalize partial containment across types
        name_a = entity_a.get("name", "").lower()
        name_b = entity_b.get("name", "").lower()
        if name_a != name_b and (name_a in name_b or name_b in name_a):
            # If one name is a proper subset of the other, check if they share
            # the same core identity. E.g., "Priya" and "Priya Mehta" = OK
            # But "Rajesh" and "Rajesh Electronics" = NOT same entity
            shorter = min(len(name_a), len(name_b))
            longer = max(len(name_a), len(name_b))
            if shorter < 4:
                # Very short names (e.g., "Priya") should NOT be merged with
                # longer names unless they share phone/account/location
                phones_a = set(self._extract_phones(entity_a))
                phones_b = set(self._extract_phones(entity_b))
                accounts_a = set(entity_a.get("accounts", []))
                accounts_b = set(entity_b.get("accounts", []))
                if not (phones_a & phones_b or accounts_a & accounts_b):
                    total_score = min(total_score, 0.4)

        # Decision thresholds — raised for safety
        if total_score >= 0.90:
            recommendation = "merge"
        elif total_score >= 0.65:
            recommendation = "review"
        else:
            recommendation = "reject"

        # Build explanation
        strong_signals = [s for s in signals if s.confidence > 0.7]
        weak_signals = [s for s in signals if s.confidence <= 0.3]

        explanation_parts = []
        if strong_signals:
            explanation_parts.append("Strong signals: " + "; ".join(s.evidence for s in strong_signals))
        if weak_signals:
            explanation_parts.append("Weak signals: " + "; ".join(s.evidence for s in weak_signals))

        explanation = " | ".join(explanation_parts) if explanation_parts else "No strong signals found"

        return DisambiguationResult(
            entity_id_a=entity_a.get("id", ""),
            entity_id_b=entity_b.get("id", ""),
            total_score=total_score,
            signals=signals,
            recommendation=recommendation,
            explanation=explanation,
        )
