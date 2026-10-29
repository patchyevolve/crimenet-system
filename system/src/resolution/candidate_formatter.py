"""
Ranked Candidate Formatter
"Did you mean..." output + confirmation gate (never silently auto-correct).
"""

from typing import List, Dict, Optional
from dataclasses import dataclass, field
import json
import os
from datetime import datetime


@dataclass
class RankedCandidate:
    """A single candidate in a "Did you mean..." suggestion."""
    entity_id: str
    name: str
    confidence: float              # 0.0-1.0
    evidence_count: int            # How many source records link to this entity
    evidence_files: List[str] = field(default_factory=list)
    signals: Dict[str, float] = field(default_factory=dict)  # signal_type → score


@dataclass
class DidYouMeanResult:
    """A complete "Did you mean..." suggestion for an entity."""
    query_entity_id: str
    query_name: str
    candidates: List[RankedCandidate]
    status: str                    # "auto_merge", "review_required", "no_match", "confirmed", "rejected"
    merge_type: str                # "auto", "review", "reject"
    created_at: str = ""


class CandidateFormatter:
    """
    Formats disambiguation results into "Did you mean..." suggestions.
    Enforces the never-silently-auto-correct rule:
    - confidence > 0.9 → auto_merge (but still logged)
    - confidence 0.6-0.9 → review_required
    - confidence < 0.6 → no_match
    """

    AUTO_MERGE_THRESHOLD = 0.90
    REVIEW_THRESHOLD = 0.60

    def __init__(self, output_dir: str = "output"):
        self.output_dir = output_dir
        self.pending_confirmations: List[DidYouMeanResult] = []
        self.confirmed_merges: List[dict] = []
        self.rejected_merges: List[dict] = []

    def format_candidates(
        self,
        query_entity_id: str,
        query_name: str,
        candidates: List[dict],  # [{entity_id, name, confidence, evidence_count, evidence_files, signals}]
    ) -> DidYouMeanResult:
        """
        Format a list of ranked candidates into a "Did you mean..." result.

        Enforces confirmation gate:
        - High confidence → auto_merge (but still added to pending for audit)
        - Medium confidence → review_required (must be confirmed)
        - Low confidence → no_match (no action)
        """
        # Sort candidates by confidence descending
        sorted_candidates = []
        for c in candidates:
            sorted_candidates.append(RankedCandidate(
                entity_id=c.get("entity_id", ""),
                name=c.get("name", ""),
                confidence=c.get("confidence", 0.0),
                evidence_count=c.get("evidence_count", 0),
                evidence_files=c.get("evidence_files", []),
                signals=c.get("signals", {}),
            ))
        sorted_candidates.sort(key=lambda x: x.confidence, reverse=True)

        # Determine status
        if not sorted_candidates:
            status = "no_match"
            merge_type = "reject"
        elif sorted_candidates[0].confidence >= self.AUTO_MERGE_THRESHOLD:
            status = "auto_merge"
            merge_type = "auto"
        elif sorted_candidates[0].confidence >= self.REVIEW_THRESHOLD:
            status = "review_required"
            merge_type = "review"
        else:
            status = "no_match"
            merge_type = "reject"

        result = DidYouMeanResult(
            query_entity_id=query_entity_id,
            query_name=query_name,
            candidates=sorted_candidates[:5],  # Top 5 candidates
            status=status,
            merge_type=merge_type,
            created_at=datetime.now().isoformat(),
        )

        # Track pending confirmations (for review_required)
        if status == "review_required":
            self.pending_confirmations.append(result)

        return result

    def confirm_merge(self, query_entity_id: str, confirmed_entity_id: str, investigator: str = "system") -> dict:
        """
        Investigator confirms a merge candidate.
        This is the ONLY way a merge becomes permanent.
        """
        confirmation = {
            "query_entity_id": query_entity_id,
            "confirmed_entity_id": confirmed_entity_id,
            "investigator": investigator,
            "timestamp": datetime.now().isoformat(),
            "action": "confirmed",
        }
        self.confirmed_merges.append(confirmation)

        # Remove from pending
        self.pending_confirmations = [
            p for p in self.pending_confirmations
            if p.query_entity_id != query_entity_id
        ]

        return confirmation

    def reject_merge(self, query_entity_id: str, investigator: str = "system") -> dict:
        """Investigator rejects a merge candidate."""
        rejection = {
            "query_entity_id": query_entity_id,
            "investigator": investigator,
            "timestamp": datetime.now().isoformat(),
            "action": "rejected",
        }
        self.rejected_merges.append(rejection)

        # Remove from pending
        self.pending_confirmations = [
            p for p in self.pending_confirmations
            if p.query_entity_id != query_entity_id
        ]

        return rejection

    def get_pending_confirmations(self) -> List[dict]:
        """Get all pending confirmations awaiting investigator review."""
        return [
            {
                "query_entity_id": p.query_entity_id,
                "query_name": p.query_name,
                "candidates": [
                    {
                        "entity_id": c.entity_id,
                        "name": c.name,
                        "confidence": c.confidence,
                        "evidence_count": c.evidence_count,
                        "evidence_files": c.evidence_files,
                        "signals": c.signals,
                    }
                    for c in p.candidates
                ],
                "status": p.status,
                "created_at": p.created_at,
            }
            for p in self.pending_confirmations
        ]

    def to_display_string(self, result: DidYouMeanResult) -> str:
        """
        Human-readable "Did you mean..." string.
        Example: "Did you mean: Rahul Sharma (94%, 2 case links) / Rahul Verma (71%)?"
        """
        if not result.candidates:
            return f"No match found for '{result.query_name}'"

        parts = []
        for c in result.candidates:
            evidence_str = f"{c.evidence_count} records" if c.evidence_count else "no records"
            parts.append(f"{c.name} ({c.confidence:.0%}, {evidence_str})")

        return f"Did you mean: {' / '.join(parts)}?"

    def save_confirmation_log(self, output_dir: str = None):
        """Save all confirmations and rejections to audit log."""
        out = output_dir or self.output_dir
        os.makedirs(out, exist_ok=True)

        log = {
            "pending": self.get_pending_confirmations(),
            "confirmed": self.confirmed_merges,
            "rejected": self.rejected_merges,
            "total_pending": len(self.pending_confirmations),
            "total_confirmed": len(self.confirmed_merges),
            "total_rejected": len(self.rejected_merges),
            "saved_at": datetime.now().isoformat(),
        }

        with open(os.path.join(out, "confirmation_log.json"), "w") as f:
            json.dump(log, f, indent=2)

        return log
