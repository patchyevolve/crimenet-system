"""
LLM Pass — intelligent entity resolution using LLM.
Sends candidate groups to LLM for evaluation.
"""

import json
import re
from typing import List, Optional
from ..models.schema import MergeCandidate, LLMVerdict
from ..ai.caller import AICaller


RESOLUTION_PROMPT = """You are an investigation entity resolver for Indian criminal investigation data.

Determine if these entities refer to the same real-world person.

For each pair, evaluate:
1. NAME_SIMILARITY: first name, last name, nicknames, handles
2. SHARED_ATTRIBUTES: phones, accounts, addresses
3. CONTEXT_ALIGNMENT: same case, same timeline, same location
4. CONTRADICTIONS: conflicting attributes
5. CULTURAL_KNOWLEDGE: Indian naming patterns

CULTURAL CONTEXT:
- "Kumar" is often omitted in casual use
- "Bhai" is an honorific (like "brother"), not legal name
- Social media handles (rakesh_rocky) map to real names
- Same phone number = very strong merge signal

RULES:
- confidence > 0.9 → merge_type: "auto"
- confidence 0.7-0.9 → merge_type: "review"
- confidence < 0.7 → merge_type: "reject"
- Never merge organizations with persons

OUTPUT (compact JSON only):
{
  "groups": [
    {
      "entity_ids": ["id1", "id2"],
      "merge": true,
      "confidence": 0.92,
      "canonical_name": "Suresh Kumar",
      "reasoning": "short reason",
      "signals": {
        "name_similarity": 0.8,
        "shared_attributes": 1.0,
        "context_alignment": 0.9,
        "contradictions": 1.0,
        "cultural_knowledge": 0.95
      },
      "merge_type": "auto"
    }
  ]
}"""


def build_group_prompt(candidate: MergeCandidate) -> str:
    """Build a prompt for a single candidate group."""
    lines = ["Evaluate if these entities are the same person:\n"]

    for i, e in enumerate(candidate.entities, 1):
        lines.append(f"Entity {i}:")
        lines.append(f"  ID: {e.get('id', '?')}")
        lines.append(f"  Name: {e.get('name', '?')}")
        lines.append(f"  Type: {e.get('entity_type', '?')}")
        attrs = e.get("attributes", {})
        if attrs:
            for k, v in attrs.items():
                if v and k not in ["extracted_from", "extracted_from_field"]:
                    lines.append(f"  {k}: {v}")
        src = e.get("source", {})
        if src:
            lines.append(f"  Source: {src.get('file_name', '?')}")
        lines.append("")

    lines.append(f"Detection signal: {candidate.signal}")
    lines.append(f"Description: {candidate.description}")

    return "\n".join(lines)


def parse_llm_verdict(response_text: str, candidate: MergeCandidate) -> Optional[LLMVerdict]:
    """Parse LLM response into a verdict."""
    try:
        # Extract JSON
        json_match = re.search(r'\{[\s\S]*\}', response_text)
        if not json_match:
            return None

        raw_json = json_match.group()

        # Try direct parse
        try:
            data = json.loads(raw_json)
        except json.JSONDecodeError:
            # Try repair
            repaired = re.sub(r',\s*([}\]])', r'\1', raw_json)
            repaired = re.sub(r'"\s*\n\s*"', '",\n  "', repaired)
            try:
                data = json.loads(repaired)
            except json.JSONDecodeError:
                return None

        groups = data.get("groups", [])
        if not groups:
            return None

        # Use first group that has a merge decision
        g = None
        for grp in groups:
            if grp.get("merge"):
                g = grp
                break
        if g is None:
            g = groups[0]  # Fallback to first group

        return LLMVerdict(
            entity_ids=candidate.entity_ids,
            merge=g.get("merge", False),
            confidence=g.get("confidence", 0.5),
            canonical_name=g.get("canonical_name", ""),
            reasoning=g.get("reasoning", ""),
            signals={**g.get("signals", {}), "method": "llm"},
            merge_type=g.get("merge_type", "reject"),
        )

    except Exception:
        return None


def llm_pass(candidates: List[MergeCandidate], ai: AICaller) -> List[LLMVerdict]:
    """
    Send candidate groups to LLM for evaluation.
    Returns verdicts for each group.
    """
    verdicts = []

    for candidate in candidates:
        prompt = build_group_prompt(candidate)

        result = ai.extract(
            task="resolve_entities",
            text=prompt,
            source_file="entity_resolution",
            custom_prompt=RESOLUTION_PROMPT,
            temperature=0.1,
        )

        if result.get("status") == "success" and result.get("parsed"):
            # Parse from parsed response — ensure it's a dict
            parsed = result["parsed"]
            if not isinstance(parsed, dict):
                parsed = {}
            groups = parsed.get("groups", [])
            if groups:
                g = groups[0]
                verdict = LLMVerdict(
                    entity_ids=candidate.entity_ids,
                    merge=g.get("merge", False),
                    confidence=g.get("confidence", 0.5),
                    canonical_name=g.get("canonical_name", ""),
                    reasoning=g.get("reasoning", ""),
                    signals={**g.get("signals", {}), "method": "llm"},
                    merge_type=g.get("merge_type", "reject"),
                )
                verdicts.append(verdict)
                continue

        # Fallback: try to parse raw response
        if result.get("raw_response"):
            verdict = parse_llm_verdict(result["raw_response"], candidate)
            if verdict:
                verdicts.append(verdict)
                continue

        # If LLM fails, use rule-based confidence
        verdicts.append(LLMVerdict(
            entity_ids=candidate.entity_ids,
            merge=candidate.confidence > 0.8,
            confidence=candidate.confidence,
            canonical_name=candidate.entities[0].get("name", "") if candidate.entities else "",
            reasoning=f"LLM unavailable, using rule-based signal: {candidate.signal}",
            signals={"rule_based": candidate.confidence, "method": "rule_based"},
            merge_type="auto" if candidate.confidence > 0.9 else "review",
        ))

    return verdicts
