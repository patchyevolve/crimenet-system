"""Graph Builder — Stage 5
Creates graph nodes and edges from resolved entities, relations, temporal data,
raw evidence, and unknown entities.

Architecture refs: STAGE_REASONERS.md §Stage5, DATA_FLOW.md §Stage5,
INTERNAL_BINDINGS.md §Stage5, OUTPUTS.md §Stage5, SYSTEM_STRUCTURE.md §Stage5

Reads from: RESOLVED_ENTITIES_STORE, UNKNOWN_ENTITIES_STORE, EXTRACTED_RELATIONS_STORE,
            TEMPORAL_SPATIAL_STORE, RAW_EVIDENCE_STORE
Writes to:  EVIDENCE_EDGES_STORE, PROVENANCE_STORE
"""

import json
import time
from typing import List, Dict, Optional, Set, Tuple
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field, asdict
from collections import defaultdict
from itertools import combinations
from ..models.schema import (
    ExtractedEntity, ExtractedRelation, ResolvedEntity, UnknownEntity,
    SemanticEdgeType, ProvenanceChain, DependencyGroup,
    ConfidenceSchema, ConfidenceFactor, generate_id, get_reliability,
)

# ──────────────────────────────────────────────
# Policy Thresholds (per STAGE_REASONERS.md §Stage5)
# ──────────────────────────────────────────────
MIN_EDGE_CONFIDENCE = 0.50
MIN_SOURCE_RELIABILITY = 0.30
MAX_PROVENANCE_DEPTH = 5
CONFIDENCE_DECAY_RATE = 0.85  # per depth step beyond MAX_PROVENANCE_DEPTH
DEPENDENCY_CONFIDENCE_PENALTY = 0.85  # multiply by this for dependent edges

# ──────────────────────────────────────────────
# Semantic Classification (4 types per architecture)
# ──────────────────────────────────────────────
RELATION_SEMANTIC_MAP = {
    # Entrepreneurial (business/profit)
    "TRANSFERRED_TO": "entrepreneurial",
    "RECEIVED_FROM": "entrepreneurial",
    "OWNS_ACCOUNT": "entrepreneurial",
    "WORKS_WITH": "entrepreneurial",
    # Associational (social/bonding)
    "CALLED": "associational",
    "MESSED": "associational",
    "ASSOCIATED_WITH": "associational",
    "FAMILY_OF": "associational",
    "FRIEND_OF": "associational",
    "ASSOCIATE_OF": "associational",
    "VISITED": "associational",
    "MEMBER_OF": "associational",
    # Quasi-governmental (enforcement)
    "SUSPECT_OF": "quasi-governmental",
    "VICTIM_OF": "quasi-governmental",
    "WITNESS_OF": "quasi-governmental",
    "LIVES_AT": "quasi-governmental",
    "WORKS_AT": "quasi-governmental",
    # Upperworld bridge (corrupt officials, political backers, legitimate fronts)
    "REGISTERED_OWNER": "upperworld_bridge",
    "ACTUAL_USER": "upperworld_bridge",
    "DRIVER": "upperworld_bridge",
    "PURCHASER": "upperworld_bridge",
    "INSURER": "upperworld_bridge",
    "PASSENGER": "upperworld_bridge",
    # Entity-to-attribute links (Category B)
    "OWNS_PHONE": "associational",
    "LOCATED_AT": "associational",
    "USES_DEVICE": "associational",
    # Shared-attribute co-occurrence (Category C)
    "SHARED_PHONE": "associational",
    "SHARED_LOCATION": "associational",
    "SHARED_ACCOUNT": "entrepreneurial",
    "SHARED_DEVICE": "associational",
    # Provenance (internal)
    "EXTRACTED_FROM": "associational",
    "DERIVED_FROM": "associational",
    "POSSIBLE_IDENTITY": "associational",
    "SAME_AS": "associational",
}

# Domain mapping for edge types
EDGE_DOMAIN_MAP = {
    "entrepreneurial": "legitimate",
    "associational": "personal",
    "quasi-governmental": "criminal",
    "upperworld_bridge": "legitimate",
}

def classify_semantic_edge(relation_type: str) -> str:
    """Classify a relation into semantic edge type (4 categories)."""
    return RELATION_SEMANTIC_MAP.get(relation_type, "associational")

# ──────────────────────────────────────────────
# Data Models (per OUTPUTS.md EvidenceEdge spec)
# ──────────────────────────────────────────────
@dataclass
class GraphNode:
    """A node in the evidence graph — per OUTPUTS.md"""
    id: str
    node_type: str              # "person", "phone", "vehicle", "location", "account", "organization", "unknown"
    name: str
    attributes: dict = field(default_factory=dict)
    confidence: float = 0.0
    epistemic_status: str = "observation"
    derivation_depth: int = 0
    source_entities: List[str] = field(default_factory=list)
    provenance_chain: List[str] = field(default_factory=list)
    effective_confidence: float = 0.0
    resolution_status: str = "unresolved"  # "resolved", "unresolved", "partial"
    created_at: str = ""
    updated_at: str = ""
    run_id: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

@dataclass
class GraphEdge:
    """An edge in the evidence graph — per OUTPUTS.md EvidenceEdge spec.

    Field name is 'relationship_type' per architecture (not 'relation_type').
    """
    id: str
    source_id: str
    target_id: str
    relationship_type: str      # per OUTPUTS.md: "called" / "met_at" / "owns" / etc.
    edge_type: str              # "entrepreneurial", "associational", "quasi-governmental", "upperworld_bridge"
    confidence: dict = field(default_factory=dict)  # ConfidenceSchema as dict
    supporting_evidence: List[str] = field(default_factory=list)
    contradicting_evidence: List[str] = field(default_factory=list)
    temporal_info: Optional[dict] = None
    provenance_chain: List[str] = field(default_factory=list)
    dependency_group: str = ""
    epistemic_status: str = "observation"
    derivation_depth: int = 0
    is_independent: bool = True
    adversarial_score: float = 0.0
    created_at: str = ""
    updated_at: str = ""
    run_id: str = ""
    source_name: str = ''         # human-readable source name
    target_name: str = ''         # human-readable target name

    def to_dict(self) -> dict:
        return asdict(self)

@dataclass
class MissingEdge:
    """Tracks expected edges that do not exist — per STAGE_REASONERS.md Step 5"""
    source_id: str
    target_id: str
    expected_relation: str
    confidence: float = 0.0
    source_files: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

@dataclass
class MultiplexityTie:
    """Multi-domain tie between two entities — per OUTPUTS.md §MultiplexityTie."""
    id: str
    entity_a: str
    entity_b: str
    tie_types: List[str] = field(default_factory=list)  # ["communication", "financial", "social", "criminal"]
    tie_count: int = 0
    tie_strength: float = 0.0   # 0-1
    is_cross_domain: bool = False
    domain_mapping: dict = field(default_factory=dict)  # {criminal: [...], personal: [...], legitimate: [...]}

    def to_dict(self) -> dict:
        return asdict(self)

@dataclass
class AdversarialEdgeScore:
    """Adversarial risk assessment for an edge — per OUTPUTS.md §AdversarialEdgeScore."""
    edge_id: str
    naturalness_score: float = 0.0     # 0-1
    temporal_consistency: float = 0.0   # 0-1
    source_reliability: float = 0.0     # 0-1
    network_coherence: float = 0.0      # 0-1
    behavioral_consistency: float = 0.0 # 0-1
    overall_score: float = 0.0          # 0-1
    is_suspicious: bool = False
    flags: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)
# ──────────────────────────────────────────────
# Confidence Propagation (per STAGE_REASONERS.md)
# ──────────────────────────────────────────────
def propagate_confidence(
    source_confidence: float,
    target_confidence: float,
    extraction_confidence: float,
    source_reliability: float = 0.5,
    derivation_depth: int = 0,
    supporting_count: int = 1,
    contradicting_count: int = 0,
    coverage_ratio: float = 1.0,
    is_independent: bool = True,
) -> ConfidenceSchema:
    """Edge confidence = min(source, target, extraction) per STAGE_REASONERS.md.
    Applies decay for deep provenance chains.
    Applies dependency penalty for dependent edges.
    Applies coverage adjustment.
    """
    # Core propagation: min of all confidence components
    base_confidence = min(source_confidence, target_confidence, extraction_confidence)
    # Apply decay for deep provenance (D5 fix)
    if derivation_depth > MAX_PROVENANCE_DEPTH:
        excess_depth = derivation_depth - MAX_PROVENANCE_DEPTH
        base_confidence *= (CONFIDENCE_DECAY_RATE ** excess_depth)
    # Apply dependency penalty (B7 fix: dependent edges get reduced confidence)
    if not is_independent:
        base_confidence *= DEPENDENCY_CONFIDENCE_PENALTY
    # Apply coverage adjustment (G7 fix)
    if coverage_ratio < 1.0:
        # Low coverage reduces confidence proportionally
        base_confidence *= (0.5 + 0.5 * coverage_ratio)
    # Adjust for corroboration (only independent sources count)
    if supporting_count > 1 and is_independent:
        corroboration_boost = min(0.1, 0.02 * (supporting_count - 1))
        base_confidence = min(1.0, base_confidence + corroboration_boost)
    # Reduce for contradictions — identity/name-variant contradictions are
    # resolution issues, not evidence that THIS edge is wrong. Cap their impact
    # so a pending alias gate cannot zero out legitimate victim/witness edges.
    if contradicting_count > 0:
        # Full penalty only up to 0.15 for ordinary contradictions;
        # identity-style disputes contribute at most 0.05 each (handled by caller
    # filtering which contradictions count). Hard cap remains 0.30.
        contradiction_penalty = min(0.30, 0.08 * contradicting_count)
        base_confidence = max(0.0, base_confidence - contradiction_penalty)
    base_confidence = max(0.0, min(1.0, base_confidence))
    # Build factors
    factors = [
        ConfidenceFactor(
            factor_type="source_reliability",
            value=source_reliability,
            weight=0.3,
            description=f"Source reliability: {source_reliability:.2f}",
        ),
        ConfidenceFactor(
            factor_type="entity_resolution",
            value=min(source_confidence, target_confidence),
            weight=0.25,
            description=f"Source entity: {source_confidence:.2f}, Target: {target_confidence:.2f}",
        ),
        ConfidenceFactor(
            factor_type="extraction_confidence",
            value=extraction_confidence,
            weight=0.25,
            description=f"Extraction confidence: {extraction_confidence:.2f}",
        ),
        ConfidenceFactor(
            factor_type="coverage_adjustment",
            value=coverage_ratio,
            weight=0.2,
            description=f"Coverage ratio: {coverage_ratio:.2f}",
        ),
    ]
    if derivation_depth > MAX_PROVENANCE_DEPTH:
        factors.append(ConfidenceFactor(
            factor_type="provenance_decay",
            value=CONFIDENCE_DECAY_RATE ** (derivation_depth - MAX_PROVENANCE_DEPTH),
            weight=0.5,
            description=f"Depth decay at {derivation_depth} steps",
        ))
    if not is_independent:
        factors.append(ConfidenceFactor(
            factor_type="dependency_penalty",
            value=DEPENDENCY_CONFIDENCE_PENALTY,
            weight=0.5,
            description=f"Dependent edge penalty: {DEPENDENCY_CONFIDENCE_PENALTY}",
        ))
    return ConfidenceSchema(
        score=base_confidence,
        basis=[f"min(source={source_confidence:.2f}, target={target_confidence:.2f}, extraction={extraction_confidence:.2f})"],
        supporting_count=supporting_count,
        contradicting_count=contradicting_count,
        source_reliability=source_reliability,
        derivation_depth=derivation_depth,
        is_independent=is_independent,
        semantic_type="inference",
        factors=factors,
    )


# ──────────────────────────────────────────────
# Adversarial Edge Detection (per STAGE_REASONERS.md + OUTPUTS.md)
# ──────────────────────────────────────────────
def compute_adversarial_score(
    source_id: str,
    target_id: str,
    relation_type: str,
    confidence: float,
    temporal_info: Optional[dict],
    source_reliability: float,
    all_edges: List[dict],
    network_edge_count: Dict[str, int],
    has_adversarial_source: bool = False,
    adversarial_source_files: Optional[set] = None,
) -> AdversarialEdgeScore:
    """Compute adversarial edge score per OUTPUTS.md §AdversarialEdgeScore.
    Returns AdversarialEdgeScore with component scores and overall assessment.
    
    Args:
        has_adversarial_source: True if this edge's supporting evidence includes
            a file flagged as adversarial in Stage 1.
        adversarial_source_files: Set of adversarial file names (for flag messages).
    """
    flags = []
    # Stage 1 adversarial source files → immediate suspicion
    if has_adversarial_source:
        flags.append("Source file flagged as adversarial in Stage 1")
        if adversarial_source_files:
            for fname in adversarial_source_files:
                flags.append(f"Adversarial source: {fname}")
    
    # Naturalness: how natural is this edge type for these entities?
    naturalness = 1.0
    if has_adversarial_source:
        naturalness -= 0.4
    if confidence > 0.95 and relation_type in ("ASSOCIATED_WITH", "MESSED"):
        naturalness -= 0.3
        flags.append("Suspiciously high confidence on weak relation type")
    # Temporal consistency
    temporal_consistency = 1.0
    if relation_type in ("CALLED", "TRANSFERRED_TO", "RECEIVED_FROM") and not temporal_info:
        temporal_consistency -= 0.3
        flags.append(f"Missing temporal info on {relation_type}")
    elif temporal_info and isinstance(temporal_info, dict):
        if temporal_info.get("precision") == "unknown":
            temporal_consistency -= 0.1
    # Source reliability
    source_rel = source_reliability
    if source_reliability < MIN_SOURCE_RELIABILITY:
        flags.append(f"Source reliability below minimum: {source_reliability:.2f}")
    # Network coherence: hub pattern detection
    network_coherence = 1.0
    outgoing = network_edge_count.get(source_id, 0)
    if outgoing > 10:
        network_coherence -= 0.2
        flags.append(f"Hub pattern: {outgoing} outgoing edges")
    incoming = network_edge_count.get(target_id, 0)
    if incoming > 10:
        network_coherence -= 0.2
        flags.append(f"Hub pattern: {incoming} incoming edges")
    # Behavioral consistency
    behavioral = 1.0
    if source_id == target_id:
        behavioral -= 0.5
        flags.append("Self-loop detected")
    # Overall score
    overall = (
        (1.0 - naturalness) * 0.25
        + (1.0 - temporal_consistency) * 0.20
        + (1.0 - source_rel) * 0.20
        + (1.0 - network_coherence) * 0.20
        + (1.0 - behavioral) * 0.15
    )
    overall = max(0.0, min(1.0, overall))
    # Adversarial source → minimum suspicion threshold
    if has_adversarial_source:
        overall = max(overall, 0.5)
    return AdversarialEdgeScore(
        edge_id="",
        naturalness_score=naturalness,
        temporal_consistency=temporal_consistency,
        source_reliability=source_rel,
        network_coherence=network_coherence,
        behavioral_consistency=behavioral,
        overall_score=overall,
        is_suspicious=overall > 0.3 or has_adversarial_source,
        flags=flags,
    )


# ──────────────────────────────────────────────
# Provenance Chain Builder (step-by-step)
# ──────────────────────────────────────────────
def build_derivation_chain(
    entity_id: str,
    source_file: str,
    derivation_depth: int,
    extraction_id: str = '',
    resolution_id: str = '',
) -> List[dict]:
    """Build step-by-step derivation chain per OUTPUTS.md: derivation: [{step, type, id}].
    """
    chain = []
    chain.append({"step": 0, "type": "RawEvidence", "id": source_file})
    if extraction_id:
        chain.append({"step": 1, "type": "ExtractedEntity", "id": extraction_id})
    if resolution_id and derivation_depth >= 1:
        chain.append({"step": 2, "type": "ResolvedEntity", "id": resolution_id})
    for step in range(3, derivation_depth + 1):
        chain.append({"step": step, "type": "Derived", "id": f"derived_step_{step}"})
    return chain


# ──────────────────────────────────────────────
# Node Creation
# ──────────────────────────────────────────────
def create_entity_nodes(
    resolved_entities: List[dict],
    unknown_entities: List[dict],
    all_entities: List[dict],
    run_id: str = '',
) -> List[GraphNode]:
    """Create graph nodes from resolved entities + unknown entities + standalone entities.
    Uses entity_type field from entity dict (not ID prefix).
    
    Standalone entities (not merged during resolution) are also included
    to prevent data loss — e.g., AMOUNT, DATE, DEVICE entities that are
    unique and don't need resolution.
    """
    nodes = []
    now = datetime.now().isoformat()
    entity_by_id = {e["id"]: e for e in all_entities if isinstance(e, dict) and "id" in e}
    
    # Track which entity IDs are covered by resolved entities
    covered_entity_ids = set()
    
    for res in resolved_entities:
        res_id = res.get("id", "")
        entity_type = res.get("entity_type", "PERSON").lower()
        name = res.get("canonical_name", "")
        if not name:
            aliases = res.get("aliases", [])
            if aliases:
                name = aliases[0]
            else:
                name = res.get("id", "unknown")
        source_entities = res.get("source_entities", [])
        
        # Mark source entities as covered
        for se_id in source_entities:
            covered_entity_ids.add(se_id)
        
        source_confs = []
        for se_id in source_entities:
            se = entity_by_id.get(se_id, {})
            conf = se.get("confidence", {})
            if isinstance(conf, dict):
                source_confs.append(conf.get("score", 0.5))
            elif isinstance(conf, (int, float)):
                source_confs.append(float(conf))
        avg_conf = sum(source_confs) / len(source_confs) if source_confs else 0.5
        provenance = res.get("provenance_chain", [])
        if not provenance:
            for se_id in source_entities:
                se = entity_by_id.get(se_id, {})
                source_file = se.get("source", {}).get("file_name", "")
                if source_file and source_file not in provenance:
                    provenance.append(source_file)
        node = GraphNode(
            id=res_id,
            node_type=entity_type,
            name=name,
            attributes=res.get("attributes", {}),
            confidence=avg_conf,
            epistemic_status=res.get("epistemic_status", "observation"),
            derivation_depth=res.get("derivation_depth", 0),
            source_entities=source_entities,
            provenance_chain=provenance,
            effective_confidence=res.get("effective_confidence", avg_conf),
            resolution_status="resolved",
            created_at=res.get("created_at", now),
            updated_at=res.get("updated_at", now),
            run_id=run_id,
        )
        nodes.append(node)
    
    for unk in unknown_entities:
        unk_id = unk.get("id", "")
        if not unk_id:
            continue
        covered_entity_ids.add(unk_id)
        node = GraphNode(
            id=unk_id,
            node_type=unk.get("entity_type", "unknown").lower(),
            name=unk.get("description", unk_id),
            attributes=unk.get("attributes", {}),
            confidence=unk.get("confidence", 0.0),
            epistemic_status="hypothesis",
            derivation_depth=1,
            source_entities=[unk.get("source_entity", "")] if unk.get("source_entity") else [],
            provenance_chain=[],
            effective_confidence=unk.get("confidence", 0.0),
            resolution_status="unresolved",
            created_at=now,
            updated_at=now,
            run_id=run_id,
        )
        nodes.append(node)
    
    # Add standalone entities not covered by resolved/unknown
    # These are entities that were extracted but not merged (unique entities)
    for eid, edata in entity_by_id.items():
        if eid in covered_entity_ids:
            continue
        
        entity_type = edata.get("entity_type", "unknown").lower()
        name = edata.get("name", "")
        if not name:
            continue
        
        conf = edata.get("confidence", {})
        if isinstance(conf, dict):
            confidence = conf.get("score", 0.5)
        elif isinstance(conf, (int, float)):
            confidence = float(conf)
        else:
            confidence = 0.5
        
        source_file = edata.get("source", {}).get("file_name", "")
        provenance = [source_file] if source_file else []
        
        node = GraphNode(
            id=eid,
            node_type=entity_type,
            name=name,
            attributes=edata.get("attributes", {}),
            confidence=confidence,
            epistemic_status=edata.get("epistemic_category", "observation"),
            derivation_depth=edata.get("derivation_depth", 0),
            source_entities=[],
            provenance_chain=provenance,
            effective_confidence=confidence,
            resolution_status="standalone",
            created_at=now,
            updated_at=now,
            run_id=run_id,
        )
        nodes.append(node)
    
    return nodes


# ──────────────────────────────────────────────
# Category B: Entity-to-Attribute Edges
# ──────────────────────────────────────────────
def _normalize_attr_values(val) -> List[str]:
    """Normalize an attribute value to a list of non-empty strings."""
    if isinstance(val, list):
        return [str(v).strip() for v in val if v and str(v).strip()]
    elif val and str(val).strip():
        return [str(val).strip()]
    return []

def create_attribute_edges(
    resolved_entities: List[dict],
    nodes: List[GraphNode],
    run_id: str = '',
) -> List[GraphEdge]:
    """Create edges from resolved entity attributes to attribute nodes.
    
    Category B: Rakesh → OWNS_PHONE → 9876543210
                Rakesh → OWNS_ACCOUNT → Account 1234
                Rakesh → LOCATED_AT → Location
                Rakesh → USES_DEVICE → Device
    """
    now = datetime.now().isoformat()
    edges = []

    # Build node lookup by id and by name
    node_by_id = {n.id: n for n in nodes}
    # Map lowercased name → node for finding existing attribute nodes
    node_by_name = {}
    for n in nodes:
        if n.name:
            node_by_name[n.name.lower()] = n

    def _get_entity_conf(res: dict) -> float:
        """Extract confidence from a resolved entity dict."""
        conf = res.get("confidence", {})
        if isinstance(conf, dict) and conf.get("score", 0) > 0:
            return conf.get("score", 0.5)
        elif isinstance(conf, (int, float)) and conf > 0:
            return float(conf)
        return res.get("effective_confidence", 0.5)

    def _find_or_skip_node(attr_name: str, attr_type: str) -> Optional[GraphNode]:
        """Find existing node by name. Skip if not found (standalone entities handle these)."""
        return node_by_name.get(attr_name.lower())

    for res in resolved_entities:
        res_id = res.get("id", "")
        res_type = (res.get("entity_type") or "").lower()
        res_name = res.get("canonical_name", "")
        if not res_name:
            aliases = res.get("aliases", [])
            res_name = aliases[0] if aliases else res.get("id", "unknown")
        if not res_id:
            continue

        res_conf = _get_entity_conf(res)

        # --- Phones ---
        phones = _normalize_attr_values(res.get("phones", []))
        attrs = res.get("attributes", {}) or {}
        if isinstance(attrs, dict):
            phones.extend(_normalize_attr_values(attrs.get("phone_number", [])))
            phones.extend(_normalize_attr_values(attrs.get("phone", [])))
        # Deduplicate
        seen_phones = set()
        unique_phones = []
        for p in phones:
            if p not in seen_phones:
                seen_phones.add(p)
                unique_phones.append(p)

        for phone_val in unique_phones:
            # Find existing PHONE node
            phone_node = node_by_name.get(phone_val)
            if not phone_node:
                continue
            # Skip self-loops (PHONE node → same PHONE node)
            if phone_node.id == res_id:
                continue
            edge_type = classify_semantic_edge("OWNS_PHONE")
            edge = GraphEdge(
                id=generate_id("EDGE", f"{res_id}_OWNS_PHONE_{phone_node.id}"),
                source_id=res_id,
                target_id=phone_node.id,
                source_name=res_name,
                target_name=phone_node.name,
                relationship_type="OWNS_PHONE",
                edge_type=edge_type,
                confidence={"score": res_conf, "basis": ["entity_attribute"], "semantic_type": "observation"},
                supporting_evidence=res.get("source_entities", []),
                provenance_chain=[],
                created_at=now,
                updated_at=now,
                run_id=run_id,
            )
            edges.append(edge)

        # --- Accounts ---
        accounts = _normalize_attr_values(res.get("accounts", []))
        if isinstance(attrs, dict):
            acct_num = attrs.get("account_number", "")
            if acct_num:
                accounts.extend(_normalize_attr_values(acct_num))
        seen_accts = set()
        unique_accts = []
        for a in accounts:
            if a not in seen_accts:
                seen_accts.add(a)
                unique_accts.append(a)

        for acct_val in unique_accts:
            acct_node = node_by_name.get(acct_val)
            if not acct_node:
                continue
            if acct_node.id == res_id:
                continue
            edge_type = classify_semantic_edge("OWNS_ACCOUNT")
            edge = GraphEdge(
                id=generate_id("EDGE", f"{res_id}_OWNS_ACCOUNT_{acct_node.id}"),
                source_id=res_id,
                target_id=acct_node.id,
                source_name=res_name,
                target_name=acct_node.name,
                relationship_type="OWNS_ACCOUNT",
                edge_type=edge_type,
                confidence={"score": res_conf, "basis": ["entity_attribute"], "semantic_type": "observation"},
                supporting_evidence=res.get("source_entities", []),
                provenance_chain=[],
                created_at=now,
                updated_at=now,
                run_id=run_id,
            )
            edges.append(edge)

        # --- Locations (addresses) ---
        addresses = _normalize_attr_values(res.get("addresses", []))
        for addr_val in addresses:
            loc_node = node_by_name.get(addr_val)
            if not loc_node:
                continue
            if loc_node.id == res_id:
                continue
            edge_type = classify_semantic_edge("LOCATED_AT")
            edge = GraphEdge(
                id=generate_id("EDGE", f"{res_id}_LOCATED_AT_{loc_node.id}"),
                source_id=res_id,
                target_id=loc_node.id,
                source_name=res_name,
                target_name=loc_node.name,
                relationship_type="LOCATED_AT",
                edge_type=edge_type,
                confidence={"score": res_conf, "basis": ["entity_attribute"], "semantic_type": "observation"},
                supporting_evidence=res.get("source_entities", []),
                provenance_chain=[],
                created_at=now,
                updated_at=now,
                run_id=run_id,
            )
            edges.append(edge)

        # --- Devices (IMEI) ---
        if isinstance(attrs, dict):
            imeis = _normalize_attr_values(attrs.get("imei", []))
            for imei_val in imeis:
                dev_node = node_by_name.get(imei_val)
                if not dev_node:
                    continue
                if dev_node.id == res_id:
                    continue
                edge_type = classify_semantic_edge("USES_DEVICE")
                edge = GraphEdge(
                    id=generate_id("EDGE", f"{res_id}_USES_DEVICE_{dev_node.id}"),
                    source_id=res_id,
                    target_id=dev_node.id,
                    source_name=res_name,
                    target_name=dev_node.name,
                    relationship_type="USES_DEVICE",
                    edge_type=edge_type,
                    confidence={"score": res_conf, "basis": ["entity_attribute"], "semantic_type": "observation"},
                    supporting_evidence=res.get("source_entities", []),
                    provenance_chain=[],
                    created_at=now,
                    updated_at=now,
                    run_id=run_id,
                )
                edges.append(edge)

    return edges


# ──────────────────────────────────────────────
# Category C: Shared-Attribute Co-occurrence Edges
# ──────────────────────────────────────────────
def create_cooccurrence_edges(
    resolved_entities: List[dict],
    entity_by_id: Dict[str, dict],
    run_id: str = '',
) -> List[GraphEdge]:
    """Create Person↔Person edges when entities share the same attribute value.
    
    Category C: Rakesh ←──[SHARED_PHONE]──→ Suresh (both have 9876543210)
    """
    now = datetime.now().isoformat()
    edges = []

    # Helper to get entity confidence
    def _get_conf(eid: str) -> float:
        e = entity_by_id.get(eid, {})
        conf = e.get("confidence", {})
        if isinstance(conf, dict) and conf.get("score", 0) > 0:
            return conf.get("score", 0.5)
        elif isinstance(conf, (int, float)) and conf > 0:
            return float(conf)
        return e.get("effective_confidence", 0.5)

    def _get_name(eid: str) -> str:
        e = entity_by_id.get(eid, {})
        return e.get("canonical_name", e.get("name", eid))

    # Build reverse indexes: attribute_value → [entity_ids]
    phone_index: Dict[str, List[str]] = defaultdict(list)
    location_index: Dict[str, List[str]] = defaultdict(list)
    account_index: Dict[str, List[str]] = defaultdict(list)
    device_index: Dict[str, List[str]] = defaultdict(list)

    for res in resolved_entities:
        eid = res.get("id", "")
        if not eid:
            continue
        attrs = res.get("attributes", {}) or {}

        # Phones
        phones = _normalize_attr_values(res.get("phones", []))
        if isinstance(attrs, dict):
            phones.extend(_normalize_attr_values(attrs.get("phone_number", [])))
            phones.extend(_normalize_attr_values(attrs.get("phone", [])))
        seen = set()
        for p in phones:
            if p and p not in seen and p.lower() != "unknown":
                seen.add(p)
                phone_index[p].append(eid)

        # Locations (from addresses + location attributes)
        addresses = _normalize_attr_values(res.get("addresses", []))
        if isinstance(attrs, dict):
            addresses.extend(_normalize_attr_values(attrs.get("location", [])))
            addresses.extend(_normalize_attr_values(attrs.get("location_name", [])))
        seen = set()
        for loc in addresses:
            if loc and loc not in seen:
                seen.add(loc)
                location_index[loc].append(eid)

        # Accounts
        accounts = _normalize_attr_values(res.get("accounts", []))
        if isinstance(attrs, dict):
            acct_num = attrs.get("account_number", "")
            if acct_num:
                accounts.extend(_normalize_attr_values(acct_num))
        seen = set()
        for a in accounts:
            if a and a not in seen:
                seen.add(a)
                account_index[a].append(eid)

        # Devices (IMEI)
        if isinstance(attrs, dict):
            imeis = _normalize_attr_values(attrs.get("imei", []))
            seen = set()
            for i in imeis:
                if i and i not in seen:
                    seen.add(i)
                    device_index[i].append(eid)

    # Generate co-occurrence edges
    def _make_cooccurrence_edges(
        index: Dict[str, List[str]],
        rel_type: str,
        attr_label: str,
    ) -> List[GraphEdge]:
        result = []
        for attr_val, eids in index.items():
            if len(eids) < 2:
                continue
            # Deduplicate entity IDs
            unique_eids = list(dict.fromkeys(eids))
            for eid_a, eid_b in combinations(unique_eids, 2):
                if eid_a == eid_b:
                    continue
                conf_a = _get_conf(eid_a)
                conf_b = _get_conf(eid_b)
                edge_conf = min(conf_a, conf_b)
                edge_type = classify_semantic_edge(rel_type)
                result.append(GraphEdge(
                    id=generate_id("EDGE", f"{eid_a}_{rel_type}_{eid_b}_{attr_val}"),
                    source_id=eid_a,
                    target_id=eid_b,
                    source_name=_get_name(eid_a),
                    target_name=_get_name(eid_b),
                    relationship_type=rel_type,
                    edge_type=edge_type,
                    confidence={
                        "score": edge_conf,
                        "basis": [f"shared_{attr_label}: {attr_val}"],
                        "semantic_type": "inference",
                    },
                    supporting_evidence=[f"shared_{attr_label}:{attr_val}"],
                    provenance_chain=[],
                    created_at=now,
                    updated_at=now,
                    run_id=run_id,
                ))
        return result

    edges.extend(_make_cooccurrence_edges(phone_index, "SHARED_PHONE", "phone"))
    edges.extend(_make_cooccurrence_edges(location_index, "SHARED_LOCATION", "location"))
    edges.extend(_make_cooccurrence_edges(account_index, "SHARED_ACCOUNT", "account"))
    edges.extend(_make_cooccurrence_edges(device_index, "SHARED_DEVICE", "device"))

    return edges


# ──────────────────────────────────────────────
# Category D: Multiplexity Ties
# ──────────────────────────────────────────────
def compute_multiplexity_ties(edges: List[GraphEdge]) -> List[MultiplexityTie]:
    """Aggregate multiple relationship types between the same entity pair.
    
    Category D: Rakesh ↔ Suresh: {communication, social, financial}
    """
    # Group edges by (source_id, target_id) — both directions
    pair_edges: Dict[Tuple[str, str], List[GraphEdge]] = defaultdict(list)
    for edge in edges:
        # Normalize pair key: sorted to handle both directions
        pair = tuple(sorted([edge.source_id, edge.target_id]))
        if pair[0] == pair[1]:
            continue
        pair_edges[pair].append(edge)

    ties = []
    for (eid_a, eid_b), pair_edge_list in pair_edges.items():
        if len(pair_edge_list) < 2:
            continue

        tie_types = set()
        rel_types = set()
        domain_mapping: Dict[str, List[str]] = defaultdict(list)

        for edge in pair_edge_list:
            et = edge.edge_type
            rt = edge.relationship_type
            tie_types.add(et)
            rel_types.add(rt)
            domain = EDGE_DOMAIN_MAP.get(et, "personal")
            if rt not in domain_mapping[domain]:
                domain_mapping[domain].append(rt)

        is_cross_domain = len(domain_mapping) > 1
        tie_strength = len(tie_types) / 4.0  # normalize by max 4 semantic types

        tie = MultiplexityTie(
            id=generate_id("MT", f"{eid_a}_{eid_b}"),
            entity_a=eid_a,
            entity_b=eid_b,
            tie_types=sorted(tie_types),
            tie_count=len(rel_types),
            tie_strength=min(1.0, tie_strength),
            is_cross_domain=is_cross_domain,
            domain_mapping=dict(domain_mapping),
        )
        ties.append(tie)

    return ties


# ──────────────────────────────────────────────
# Category F: Graph Statistics
# ──────────────────────────────────────────────
def compute_graph_statistics(
    nodes: List[GraphNode],
    edges: List[GraphEdge],
    multiplexity_ties: List[MultiplexityTie],
) -> dict:
    """Compute aggregate graph metrics."""
    from collections import Counter

    node_type_counts = Counter(n.node_type for n in nodes)
    edge_type_counts = Counter(e.edge_type for e in edges)
    rel_type_counts = Counter(e.relationship_type for e in edges)

    # Degree computation
    degree = Counter()
    for e in edges:
        degree[e.source_id] += 1
        degree[e.target_id] += 1

    degrees = list(degree.values()) if degree else [0]
    avg_degree = sum(degrees) / len(degrees) if degrees else 0
    max_degree = max(degrees) if degrees else 0

    # Connected components (simple BFS)
    adj: Dict[str, Set[str]] = defaultdict(set)
    for e in edges:
        adj[e.source_id].add(e.target_id)
        adj[e.target_id].add(e.source_id)

    visited = set()
    components = 0
    for n in nodes:
        if n.id in visited:
            continue
        if n.id not in adj:
            components += 1
            visited.add(n.id)
            continue
        components += 1
        queue = [n.id]
        while queue:
            cur = queue.pop()
            if cur in visited:
                continue
            visited.add(cur)
            for nb in adj.get(cur, []):
                if nb not in visited:
                    queue.append(nb)

    # Density: actual edges / possible edges
    n = len(nodes)
    possible = n * (n - 1) / 2 if n > 1 else 1
    density = len(edges) / possible if possible > 0 else 0

    return {
        "total_nodes": len(nodes),
        "total_edges": len(edges),
        "node_type_counts": dict(node_type_counts),
        "edge_type_counts": dict(edge_type_counts),
        "relationship_type_counts": dict(rel_type_counts),
        "multiplexity_ties_count": len(multiplexity_ties),
        "avg_degree": round(avg_degree, 2),
        "max_degree": max_degree,
        "connected_components": components,
        "density": round(density, 6),
    }


# ──────────────────────────────────────────────
# Edge Creation with Confidence Propagation
# ──────────────────────────────────────────────
def create_relationship_edges(
    relations: List[dict],
    resolved_entities: List[dict],
    unknown_entities: List[dict],
    entity_by_id: Dict[str, dict],
    temporal_infos: List[dict],
    spatial_infos: List[dict],
    coverage_intervals: List[dict],
    contradictions: Dict[str, dict],
    raw_evidence: List[dict],
    run_id: str = '',
) -> Tuple[List[GraphEdge], List[MissingEdge], List[dict], List[AdversarialEdgeScore]]:
    """Create graph edges from extracted relations with:
    - Confidence propagation (min of source, target, extraction)
    - Confidence threshold filtering (< 0.5 → reject)
    - Contradicting evidence tracking
    - Provenance derivation chains
    - Adversarial edge detection (full AdversarialEdgeScore)
    - Missing edge tracking
    - Temporal spatial enrichment
    - Source reliability check (min 0.30)
    - Coverage-based confidence adjustment
    - Dependent edge confidence penalty
    """
    now = datetime.now().isoformat()
    # Build mapping: original entity IDs → resolved entity IDs
    original_to_resolved = {}
    for res in resolved_entities:
        for orig_id in res.get("source_entities", []):
            original_to_resolved[orig_id] = res["id"]
    # Reverse map used for contradiction joins (defined early so lookups work)
    raw_to_resolved = dict(original_to_resolved)
    # Build mapping: any entity ID → human-readable name
    id_to_name = {}
    for ent in entity_by_id.values():
        eid = ent.get("id", "")
        name = ent.get("name", "")
        if eid and name:
            id_to_name[eid] = name
    for res in resolved_entities:
        rid = res.get("id", "")
        canonical = res.get("canonical_name", "")
        if rid and canonical:
            id_to_name[rid] = canonical
        for orig_id in res.get("source_entities", []):
            if canonical:
                id_to_name[orig_id] = canonical
    for unk in unknown_entities:
        uid = unk.get("id", "")
        desc = unk.get("description", uid)
        if uid and desc:
            id_to_name[uid] = desc
    # Also build from relation attributes (source_name/target_name stored by extraction)
    for rel in relations:
        src_id = rel.get("source_entity_id", "")
        tgt_id = rel.get("target_entity_id", "")
        attrs = rel.get("attributes", {}) or {}
        src_name = attrs.get("source_name", "")
        tgt_name = attrs.get("target_name", "")
        if isinstance(src_name, str) and src_name and src_id:
            id_to_name[src_id] = src_name
        if isinstance(tgt_name, str) and tgt_name and tgt_id:
            id_to_name[tgt_id] = tgt_name
    # Build temporal lookup by entity_id
    temporal_by_entity = {}
    for t in temporal_infos:
        eid = t.get("entity_id", "")
        if eid:
            if eid not in temporal_by_entity:
                temporal_by_entity[eid] = []
            temporal_by_entity[eid].append(t)
    # Build spatial lookup by entity_id
    spatial_by_entity = {}
    for s in spatial_infos:
        eid = s.get("entity_id", "")
        if eid:
            spatial_by_entity[eid] = s
    # Build coverage lookup by entity_id
    coverage_by_entity = {}
    for c in coverage_intervals:
        eid = c.get("entity_id", "")
        if eid:
            coverage_by_entity[eid] = c.get("coverage_ratio", 1.0)
    # Build contradiction lookup: entity_id → list of contradictions
    # Map raw IDs → resolved IDs so identity contradictions attach to graph nodes.
    contradictions_by_entity = {}
    for cid, contr in contradictions.items():
        for eid in contr.get("entity_ids", []):
            mapped = raw_to_resolved.get(eid, eid)
            for key in {eid, mapped}:
                if key not in contradictions_by_entity:
                    contradictions_by_entity[key] = []
                contradictions_by_entity[key].append(contr)
    # Build raw evidence lookup: file_name → evidence record
    raw_evidence_by_file = {}
    for ev in raw_evidence:
        fname = ev.get("file_name", "") or ev.get("original_filename", "")
        if fname:
            raw_evidence_by_file[fname] = ev
    # Index resolved + unknown entities
    all_resolved = {}
    for res in resolved_entities:
        all_resolved[res["id"]] = res
    for unk in unknown_entities:
        all_resolved[unk["id"]] = unk
    # Also index standalone/raw entities so edge confidence lookup never falls
    # back to 0.5 for victims/witnesses that weren't merged (Meena bug).
    for eid, ent in entity_by_id.items():
        if eid not in all_resolved and isinstance(ent, dict):
            all_resolved[eid] = ent
    # First pass: group relations by (source, target, type)
    edge_groups = {}
    for rel in relations:
        src_id = rel.get("source_entity_id", "")
        tgt_id = rel.get("target_entity_id", "")
        rel_type = rel.get("relation_type", "")
        resolved_src = original_to_resolved.get(src_id, src_id)
        resolved_tgt = original_to_resolved.get(tgt_id, tgt_id)
        if resolved_src == resolved_tgt:
            continue
        key = (resolved_src, resolved_tgt, rel_type)
        if key not in edge_groups:
            edge_groups[key] = []
        edge_groups[key].append(rel)
    # Compute network edge counts for adversarial detection
    network_edge_count = {}
    for rel in relations:
        src = rel.get("source_entity_id", "")
        tgt = rel.get("target_entity_id", "")
        network_edge_count[src] = network_edge_count.get(src, 0) + 1
        network_edge_count[tgt] = network_edge_count.get(tgt, 0) + 1
    # Second pass: create deduplicated edges
    edges = []
    missing_edges = []
    rejected_edges = []
    adversarial_scores = []
    for (resolved_src, resolved_tgt, rel_type), rels in edge_groups.items():
        base = rels[0]
        # Source entity confidence — prefer resolved effective_confidence,
        # then extraction score, then a source-quality floor (never crush to 0.5
        # for well-evidenced standalone entities like the victim).
        def _entity_conf(ent: dict) -> float:
            if not ent:
                return 0.5
            conf = ent.get("confidence", {})
            if isinstance(conf, dict) and conf.get("score", 0) > 0:
                return float(conf.get("score", 0.5))
            if isinstance(conf, (int, float)) and conf > 0:
                return float(conf)
            eff = ent.get("effective_confidence")
            if isinstance(eff, (int, float)) and eff > 0:
                return float(eff)
            return 0.5

        src_entity = all_resolved.get(resolved_src, {})
        src_confidence = _entity_conf(src_entity)
        tgt_entity = all_resolved.get(resolved_tgt, {})
        tgt_confidence = _entity_conf(tgt_entity)
        # Extraction confidence (average)
        extraction_confs = []
        for r in rels:
            conf = r.get("confidence", {})
            if isinstance(conf, dict):
                extraction_confs.append(conf.get("score", 0.5))
            elif isinstance(conf, (int, float)):
                extraction_confs.append(float(conf))
            else:
                extraction_confs.append(0.5)
        avg_extraction = sum(extraction_confs) / len(extraction_confs)
        # Source reliability (G5 fix: check min_source_reliability)
        source_reliability = 0.5
        for r in rels:
            src_meta = r.get("source", {})
            if isinstance(src_meta, dict):
                sr = src_meta.get("reliability_occurrence", 0.0)
                if sr > 0:
                    source_reliability = max(source_reliability, sr)
        # Also check raw evidence
        for r in rels:
            src_meta = r.get("source", {})
            if isinstance(src_meta, dict):
                fname = src_meta.get("file_name", "")
                if fname:
                    raw_ev = raw_evidence_by_file.get(fname, {})
                    if raw_ev:
                        sr = raw_ev.get("reliability_occurrence", 0.0)
                        if sr > 0:
                            source_reliability = max(source_reliability, sr)
        # Coverage ratio (G7 fix)
        coverage_ratio = 1.0
        for eid in [resolved_src, resolved_tgt]:
            cr = coverage_by_entity.get(eid, 1.0)
            coverage_ratio = min(coverage_ratio, cr)
        # Supporting / contradicting counts
        # Only count contradictions that actually dispute evidence for this edge.
        # Pure identity/name-alias disagreements (pending review gates) do not
        # reduce edge confidence — they are resolution work, not edge falseness.
        supporting = []
        contradicting = []
        for r in rels:
            source_file = r.get("source", {}).get("file_name", "")
            if source_file and source_file not in supporting:
                supporting.append(source_file)
        for eid in [resolved_src, resolved_tgt]:
            for contr in contradictions_by_entity.get(eid, []):
                ctype = (contr.get("type") or "").lower()
                cattr = (contr.get("attribute") or "").lower()
                # Skip identity name/alias conflicts — resolution issue, not edge dispute
                if ctype == "identity" and cattr in ("name", "alias", "names"):
                    continue
                for src in contr.get("sources", []):
                    if src not in contradicting and src not in supporting:
                        contradicting.append(src)
        # Stage 1 adversarial-flagged source files weaken the edge —
        # BUT only when the edge has no clean support: an edge corroborated
        # by non-adversarial files must not be condemned by one flagged
        # file among many (well-corroborated pairs from 30 stay clean;
        # sole-source edges from 31 are still flagged).
        adversarial_files = set()
        for r in rels:
            src_meta = r.get("source", {})
            if isinstance(src_meta, dict) and src_meta.get("adversarial_suspicious"):
                fname = src_meta.get("file_name", "")
                if fname:
                    adversarial_files.add(fname)
        clean_support = [s for s in supporting if s not in adversarial_files]
        has_adversarial_source = bool(adversarial_files) and not clean_support
        if has_adversarial_source:
            for fname in adversarial_files:
                if fname not in contradicting:
                    contradicting.append(f"adversarial:{fname}")
        # Derivation depth — floor at 1 for multi-source edges, 0 for single observation
        derivation_depth = base.get("derivation_depth", 0) or 0
        if len(supporting) > 1 and derivation_depth < 1:
            derivation_depth = 1
        if not isinstance(derivation_depth, int):
            try:
                derivation_depth = int(derivation_depth)
            except (TypeError, ValueError):
                derivation_depth = 0
        # Initially assume independent, will be updated by dependency check
        is_independent = True
        # ── Propagate Confidence (B5 fix + D5 fix + B7 fix + G7 fix) ──
        confidence_schema = propagate_confidence(
            source_confidence=src_confidence,
            target_confidence=tgt_confidence,
            extraction_confidence=avg_extraction,
            source_reliability=source_reliability,
            derivation_depth=derivation_depth,
            supporting_count=len(supporting),
            contradicting_count=len(contradicting),
            coverage_ratio=coverage_ratio,
            is_independent=is_independent,
        )
        # ── Threshold Filtering (B4 fix) ──
        # Never reject a direct observation below threshold if both endpoints
        # are well-evidenced (conf >= 0.6) and there is at least one supporting
        # file — the min() + penalty pipeline can under-count confidence for
        # standalone victims/witnesses. Floor only when evidence is real.
        effective_score = confidence_schema.score
        if (
            effective_score < MIN_EDGE_CONFIDENCE
            and src_confidence >= 0.6
            and tgt_confidence >= 0.6
            and avg_extraction >= 0.5
            and supporting
        ):
            # Apply penalty-aware floor: keep contradiction discount but not below 0.5
            floor = max(MIN_EDGE_CONFIDENCE, min(src_confidence, tgt_confidence, avg_extraction) - 0.15)
            effective_score = max(effective_score, floor)
            confidence_schema.score = effective_score
        if effective_score < MIN_EDGE_CONFIDENCE:
            rejected_edges.append({
                "source_id": resolved_src,
                "target_id": resolved_tgt,
                "relationship_type": rel_type,
                "confidence": effective_score,
                "reason": f"Below threshold ({MIN_EDGE_CONFIDENCE})",
            })
            continue
        # ── Source Reliability Check (G5 fix) ──
        if source_reliability < MIN_SOURCE_RELIABILITY:
            rejected_edges.append({
                "source_id": resolved_src,
                "target_id": resolved_tgt,
                "relationship_type": rel_type,
                "confidence": confidence_schema.score,
                "reason": f"Source reliability below minimum ({MIN_SOURCE_RELIABILITY}): {source_reliability:.2f}",
            })
            continue
        # ── Semantic Classification ──
        edge_type = classify_semantic_edge(rel_type)
        # ── Temporal Info Enrichment ──
        temporal = None
        for r in rels:
            t = r.get("temporal_info")
            if t:
                temporal = t
                break
        if not temporal:
            src_temporal = temporal_by_entity.get(resolved_src, [])
            if src_temporal:
                temporal = src_temporal[0]
        if not temporal:
            temporal = {"precision": "unknown", "confidence": 0.0}
        # ── Dependency Group ──
        dep_group = ""
        for r in rels:
            dg = r.get("dependency_group", "")
            if dg:
                dep_group = dg
                break
        # ── Derivation Chain (C14 fix: step-by-step) ──
        derivation = build_derivation_chain(
            entity_id=resolved_src,
            source_file=supporting[0] if supporting else "",
            derivation_depth=derivation_depth,
            extraction_id=base.get("id", ""),
            resolution_id=resolved_src,
        )
        # Record expected-but-absent reverse/symmetric edges as missing
        if rel_type in ("CALLED", "TRANSFERRED_TO", "VISITED", "ASSOCIATED_WITH"):
            rev_key = (resolved_tgt, resolved_src, rel_type)
            # If no reverse relation exists in the grouped set, note it for analysts
            if rev_key not in edge_groups and rel_type == "CALLED":
                missing_edges.append(MissingEdge(
                    source_id=resolved_tgt,
                    target_id=resolved_src,
                    expected_relation="CALLED",
                    confidence=round(min(src_confidence, tgt_confidence) * 0.5, 4),
                    source_files=supporting[:],
                ))
        # ── Adversarial Score (G4 fix: full AdversarialEdgeScore) ──
        adv = compute_adversarial_score(
            source_id=resolved_src,
            target_id=resolved_tgt,
            relation_type=rel_type,
            confidence=confidence_schema.score,
            temporal_info=temporal,
            source_reliability=source_reliability,
            all_edges=relations,
            network_edge_count=network_edge_count,
            has_adversarial_source=has_adversarial_source,
            adversarial_source_files=adversarial_files,
        )
# ── Create Edge ──
        edge_id = generate_id("EDGE", f"{resolved_src}_{rel_type}_{resolved_tgt}")
        adv.edge_id = edge_id
        # ── Check for LLM-reclassified entities ──
        # If source or target entity was LLM-reclassified, flag the edge
        # so investigators know the ID prefix may not match entity_type
        reclassified = False
        src_entity = all_resolved.get(resolved_src, {})
        tgt_entity = all_resolved.get(resolved_tgt, {})
        if src_entity.get("attributes", {}).get("llm_reclassified"):
            reclassified = True
        if tgt_entity.get("attributes", {}).get("llm_reclassified"):
            reclassified = True
        # Resolve source/target IDs to human-readable names
        src_name = id_to_name.get(resolved_src, resolved_src)
        tgt_name = id_to_name.get(resolved_tgt, resolved_tgt)
        # Build edge confidence with optional reclassified flag
        if reclassified:
            base_conf = confidence_schema.to_dict()
            base_conf["reclassified"] = True
            edge_confidence = base_conf
        else:
            edge_confidence = confidence_schema.to_dict()
        edge = GraphEdge(
            id=edge_id,
            source_id=resolved_src,
            target_id=resolved_tgt,
            source_name=src_name,
            target_name=tgt_name,
            relationship_type=rel_type,  # C4 fix: renamed field
            edge_type=edge_type,
            confidence=edge_confidence,
            supporting_evidence=supporting,
            contradicting_evidence=contradicting,
            temporal_info=temporal,
            provenance_chain=supporting,
            dependency_group=dep_group,
            epistemic_status=base.get("epistemic_category", "observation"),
            derivation_depth=derivation_depth,
            is_independent=is_independent,
            adversarial_score=adv.overall_score,
            created_at=now,
            updated_at=now,
        )
        edges.append(edge)
        adversarial_scores.append(adv)
    # ── Post-creation validation: fix invalid relationship types ──
    # VICTIM_OF should not point to AMOUNT/DATE entities
    # SUSPECT_OF should not point to AMOUNT/DATE entities
    VALID_VICTIM_TARGETS = {"person", "phone", "organization", "location", "unknown"}
    VALID_SUSPECT_TARGETS = {"person", "phone", "organization", "location", "event", "unknown"}
    cleaned_edges = []
    for edge in edges:
        target_node = None
        for node_candidate in all_resolved.values():
            if node_candidate.get("id") == edge.target_id:
                target_node = node_candidate
                break
        target_type = ""
        if target_node:
            target_type = (target_node.get("entity_type", "") or "").lower()
        if edge.relationship_type == "VICTIM_OF" and target_type not in VALID_VICTIM_TARGETS and target_type:
            # Reclassify: person VICTIM_OF amount → person OWNS_ACCOUNT or drop
            edge.relationship_type = "ASSOCIATED_WITH"
            edge.edge_type = classify_semantic_edge("ASSOCIATED_WITH")
        elif edge.relationship_type == "SUSPECT_OF" and target_type not in VALID_SUSPECT_TARGETS and target_type:
            edge.relationship_type = "ASSOCIATED_WITH"
            edge.edge_type = classify_semantic_edge("ASSOCIATED_WITH")
        cleaned_edges.append(edge)
    return cleaned_edges, missing_edges, rejected_edges, adversarial_scores


# ──────────────────────────────────────────────
# Hidden connection detection (shared-associate + multi-hop paths)
# ──────────────────────────────────────────────
def _detect_hidden_connections(
    edges: List[GraphEdge],
    nodes: List[GraphNode],
) -> Tuple[List[MissingEdge], List[dict]]:
    """Find expected-but-absent edges and multi-hop paths.

    1. Shared associate: persons A and B both connect to common neighbor X
       but have no direct edge → MissingEdge(A, B, SHARED_ASSOCIATE).
    2. Multi-hop paths: BFS paths of length 2-4 between person nodes that
       don't already share a direct edge (e.g. suspects → victim aggregation).
    """
    missing: List[MissingEdge] = []
    multi_hop: List[dict] = []

    person_nodes = {n.id for n in nodes if n.node_type == "person"}
    if not person_nodes:
        return missing, multi_hop

    # Adjacency (undirected)
    adj: Dict[str, Set[str]] = defaultdict(set)
    edge_between: Dict[frozenset, List[GraphEdge]] = defaultdict(list)
    for e in edges:
        if e.source_id == e.target_id:
            continue
        adj[e.source_id].add(e.target_id)
        adj[e.target_id].add(e.source_id)
        edge_between[frozenset((e.source_id, e.target_id))].append(e)

    # 1. Shared-associate: for each common neighbor of two persons
    person_list = sorted(person_nodes)
    seen_pairs: Set[Tuple[str, str]] = set()
    for i, a in enumerate(person_list):
        for b in person_list[i + 1:]:
            if b in adj.get(a, set()):
                continue  # already directly connected
            common = adj.get(a, set()) & adj.get(b, set())
            # Ignore trivial shared attribute nodes (phone/account/location alone)
            meaningful = set()
            for c in common:
                # keep if any edge into c is relational (not pure attribute link)
                for e in edge_between.get(frozenset((a, c)), []) + edge_between.get(frozenset((b, c)), []):
                    if e.relationship_type not in ("OWNS_PHONE", "OWNS_ACCOUNT", "LOCATED_AT", "USES_DEVICE"):
                        meaningful.add(c)
                        break
            if not meaningful:
                continue
            pair = (a, b)
            if pair in seen_pairs:
                continue
            seen_pairs.add(pair)
            conf = round(min(0.7, 0.3 + 0.1 * len(meaningful)), 4)
            missing.append(MissingEdge(
                source_id=a,
                target_id=b,
                expected_relation="SHARED_ASSOCIATE",
                confidence=conf,
                source_files=sorted({
                    f for c in list(meaningful)[:5]
                    for e in edge_between.get(frozenset((a, c)), []) + edge_between.get(frozenset((b, c)), [])
                    for f in (e.supporting_evidence or [])
                })[:5],
            ))

    # 2. Multi-hop paths (length 2-4) between persons without direct edge
    for a in person_list:
        # BFS up to depth 4
        frontier = [(a, [a])]
        visited = {a}
        depth = 0
        while frontier and depth < 4:
            next_frontier = []
            for cur, path in frontier:
                for nb in adj.get(cur, set()):
                    if nb in visited:
                        continue
                    new_path = path + [nb]
                    visited.add(nb)
                    if nb in person_nodes and nb != a and frozenset((a, nb)) not in edge_between:
                        # path found a → ... → nb with no direct edge
                        path_conf = 1.0
                        path_types = []
                        for u, v in zip(new_path, new_path[1:]):
                            es = edge_between.get(frozenset((u, v)), [])
                            if es:
                                path_conf *= max(
                                    (e.confidence.get("score", 0.5) if isinstance(e.confidence, dict) else 0.5)
                                    for e in es
                                )
                                path_types.append(es[0].relationship_type)
                        multi_hop.append({
                            "source_id": a,
                            "target_id": nb,
                            "path": new_path,
                            "hops": len(new_path) - 1,
                            "relationship_types": path_types,
                            "path_confidence": round(path_conf, 4),
                            "epistemic_status": "inference",
                        })
                    next_frontier.append((nb, new_path))
            frontier = next_frontier
            depth += 1

    # Deduplicate multi-hop by (source, target) keeping shortest/highest conf
    best: Dict[Tuple[str, str], dict] = {}
    for mh in multi_hop:
        key = tuple(sorted((mh["source_id"], mh["target_id"])))
        if key not in best or mh["path_confidence"] > best[key]["path_confidence"]:
            best[key] = mh
    multi_hop = sorted(best.values(), key=lambda x: (-x["path_confidence"], x["hops"]))

    return missing, multi_hop


def create_face_edges(
    output_dir: str,
    entity_by_id: Dict[str, dict],
    resolved_entities: Optional[List[dict]] = None,
    run_id: str = "",
) -> List["GraphEdge"]:
    """FACE_MATCH edges between person entities linked by face similarity
    (RESEARCH_FACE_RECOGNITION doc 08 §6.2).

    Reads ``face_embeddings.json`` (Stage 2.5). Only person↔person alias
    candidates count — identity candidates link a face record to a person,
    not two graph nodes. Pairs resolution already merged collapse onto the
    same node and are skipped.
    """
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

    orig_to_resolved: Dict[str, str] = {}
    for res in resolved_entities or []:
        if isinstance(res, dict) and res.get("id"):
            for orig in res.get("source_entities") or []:
                orig_to_resolved[str(orig)] = str(res["id"])

    def _entity_score(entity: dict) -> float:
        conf = entity.get("confidence") if isinstance(entity, dict) else getattr(entity, "confidence", None)
        if isinstance(conf, dict):
            try:
                return float(conf.get("score", 0.5))
            except (TypeError, ValueError):
                return 0.5
        try:
            return float(getattr(conf, "score", 0.5))
        except (TypeError, ValueError):
            return 0.5

    now = datetime.now().isoformat()
    edges: List[GraphEdge] = []
    seen: Set[Tuple[str, str]] = set()
    for cand in data.get("candidates") or []:
        if not isinstance(cand, dict) or cand.get("kind") != "person_alias":
            continue
        src0 = str(cand.get("source_entity_id") or "")
        tgt0 = str(cand.get("candidate_entity_id") or "")
        if not src0 or not tgt0:
            continue
        src = orig_to_resolved.get(src0, src0)
        tgt = orig_to_resolved.get(tgt0, tgt0)
        if src == tgt or src not in entity_by_id or tgt not in entity_by_id:
            continue
        key = (src, tgt) if src <= tgt else (tgt, src)
        if key in seen:
            continue
        seen.add(key)
        signals = cand.get("signals") or {}
        similarity = float(signals.get("face_similarity", 0.0))
        conf_schema = propagate_confidence(
            source_confidence=_entity_score(entity_by_id[src]),
            target_confidence=_entity_score(entity_by_id[tgt]),
            extraction_confidence=float(cand.get("confidence") or 0.0),
        )
        edges.append(GraphEdge(
            id=f"face_match_{key[0]}_{key[1]}",
            source_id=src,
            target_id=tgt,
            relationship_type="FACE_MATCH",
            edge_type="associational",
            confidence=conf_schema.to_dict(),
            supporting_evidence=[str(f) for f in cand.get("files") or [] if f],
            provenance_chain=[f"face_similarity:{similarity:.3f}"],
            epistemic_status="inference",  # resolution-class inference, not a raw observation
            created_at=now,
            updated_at=now,
            run_id=run_id,
            source_name=str(entity_by_id[src].get("name", "")) if isinstance(entity_by_id[src], dict) else "",
            target_name=str(entity_by_id[tgt].get("name", "")) if isinstance(entity_by_id[tgt], dict) else "",
        ))
    return edges


class GraphBuilder:
    """Graph Builder — Stage 5: Creates graph nodes and edges from resolved data."""

    def __init__(self):
        pass

    def create_nodes(self, resolved_entities, unknown_entities, all_entities, run_id=""):
        from .builder import create_entity_nodes
        return create_entity_nodes(resolved_entities, unknown_entities, all_entities, run_id)

    def create_edges(self, relations, resolved_entities, unknown_entities, entity_by_id, temporal_infos, spatial_infos, coverage_intervals, contradictions, raw_evidence, run_id=""):
        from .builder import create_relationship_edges
        return create_relationship_edges(relations, resolved_entities, unknown_entities, entity_by_id, temporal_infos, spatial_infos, coverage_intervals, contradictions, raw_evidence, run_id)

    def build(
        self,
        entities: list,
        relations: list,
        resolved_entities: list = None,
        unknown_entities: list = None,
        dependency_groups: list = None,
        temporal_infos: list = None,
        spatial_infos: list = None,
        coverage_intervals: list = None,
        contradictions: dict = None,
        raw_evidence: list = None,
        output_dir: str = "output",
        run_id: str = "",
    ) -> dict:
        """Full graph build: nodes + edges + output files + summary."""
        import json
        from pathlib import Path

        resolved_entities = resolved_entities or []
        unknown_entities = unknown_entities or []
        dependency_groups = dependency_groups or []
        temporal_infos = temporal_infos or []
        spatial_infos = spatial_infos or []
        coverage_intervals = coverage_intervals or []
        contradictions = contradictions or {}
        raw_evidence = raw_evidence or []

        # Build entity_by_id from all entities
        entity_by_id = {}
        for e in entities:
            if isinstance(e, dict) and "id" in e:
                entity_by_id[e["id"]] = e
            elif hasattr(e, "id"):
                entity_by_id[e.id] = e.to_dict() if hasattr(e, "to_dict") else {
                    "id": e.id,
                    "name": e.name,
                    "entity_type": e.entity_type.value if hasattr(e.entity_type, "value") else str(e.entity_type),
                    "confidence": e.confidence.score if e.confidence else 0.5,
                    "attributes": e.attributes if hasattr(e, "attributes") else {},
                    "source": e.source.to_dict() if e.source and hasattr(e.source, "to_dict") else {},
                }
        # Also index resolved entities by ID
        for res in resolved_entities:
            if isinstance(res, dict) and "id" in res:
                entity_by_id[res["id"]] = res

        # Convert relation dicts if needed
        rel_dicts = []
        for r in relations:
            if isinstance(r, dict):
                rel_dicts.append(r)
            elif hasattr(r, "to_dict"):
                rel_dicts.append(r.to_dict())
            else:
                rel_dicts.append({
                    "id": r.id,
                    "source_entity_id": r.source_entity_id,
                    "target_entity_id": r.target_entity_id,
                    "relation_type": r.relation_type.value if hasattr(r.relation_type, "value") else str(r.relation_type),
                    "confidence": r.confidence.to_dict() if r.confidence and hasattr(r.confidence, "to_dict") else {"score": 0.5},
                    "source": r.source.to_dict() if r.source and hasattr(r.source, "to_dict") else {},
                    "attributes": r.attributes if hasattr(r, "attributes") else {},
                    "raw_text": r.raw_text if hasattr(r, "raw_text") else None,
                    "extraction_method": r.extraction_method if hasattr(r, "extraction_method") else "code",
                    "semantic_edge_type": r.semantic_edge_type if hasattr(r, "semantic_edge_type") else None,
                    "temporal_info": r.temporal_info if hasattr(r, "temporal_info") else None,
                })

        # Create nodes
        nodes = create_entity_nodes(resolved_entities, unknown_entities, entities, run_id)

        # Create edges from explicit relations (Category A)
        rel_edges, missing_edges, rejected_edges, adversarial_scores = create_relationship_edges(
            relations=rel_dicts,
            resolved_entities=resolved_entities,
            unknown_entities=unknown_entities,
            entity_by_id=entity_by_id,
            temporal_infos=temporal_infos,
            spatial_infos=spatial_infos,
            coverage_intervals=coverage_intervals,
            contradictions=contradictions,
            raw_evidence=raw_evidence,
            run_id=run_id,
        )

        # Populate missing edges from multi-hop / shared-associate expectations
        # (populated further below after all edge categories are merged)

        # Create entity-to-attribute edges (Category B)
        attr_edges = create_attribute_edges(
            resolved_entities=resolved_entities,
            nodes=nodes,
            run_id=run_id,
        )

        # Create shared-attribute co-occurrence edges (Category C)
        cooccurrence_edges = create_cooccurrence_edges(
            resolved_entities=resolved_entities,
            entity_by_id=entity_by_id,
            run_id=run_id,
        )

        # Category E: face-based edges (RESEARCH_FACE_RECOGNITION doc 08 §6.2)
        face_edges = create_face_edges(
            output_dir=output_dir,
            entity_by_id=entity_by_id,
            resolved_entities=resolved_entities,
            run_id=run_id,
        )
        if face_edges:
            print(f"[GRAPH]   -> {len(face_edges)} FACE_MATCH edge(s) from face recognition")

        # Merge all edge categories + deduplicate
        # Dedup key: (source_id, target_id, relationship_type) — keep higher confidence
        all_edge_candidates = rel_edges + attr_edges + cooccurrence_edges + face_edges
        merged_edges: Dict[Tuple[str, str, str], GraphEdge] = {}
        for edge in all_edge_candidates:
            key = (edge.source_id, edge.target_id, edge.relationship_type)
            reverse_key = (edge.target_id, edge.source_id, edge.relationship_type)
            # Check both directions for symmetric edges
            existing_key = key if key in merged_edges else (reverse_key if reverse_key in merged_edges else None)
            if existing_key is None:
                merged_edges[key] = edge
            else:
                existing = merged_edges[existing_key]
                # Keep the edge with higher confidence, merge evidence
                existing_conf = existing.confidence.get("score", 0) if isinstance(existing.confidence, dict) else 0
                new_conf = edge.confidence.get("score", 0) if isinstance(edge.confidence, dict) else 0
                if new_conf > existing_conf:
                    merged_edges[existing_key] = edge
                # Merge supporting evidence
                for ev in edge.supporting_evidence:
                    if ev not in existing.supporting_evidence:
                        existing.supporting_evidence.append(ev)

        edges = list(merged_edges.values())

        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        # ── Derived hidden-connection detection (shared-associate / multi-hop) ──
        # Shared associate: A→X and B→X with no A→B edge → missing A—B link.
        # Also aggregate multi-hop paths (e.g. all suspects → victim).
        try:
            derived_missing, multi_hop_paths = _detect_hidden_connections(edges, nodes)
            for m in derived_missing:
                if not any(
                    (m.source_id, m.target_id, m.expected_relation) ==
                    (x.source_id, x.target_id, x.expected_relation)
                    for x in missing_edges
                ):
                    missing_edges.append(m)
            if multi_hop_paths:
                with open(out_path / "multi_hop_paths.json", "w") as f:
                    json.dump(multi_hop_paths, f, indent=2, default=str)
        except Exception as _hc_err:
            print(f"[GRAPH] Hidden-connection detection skipped: {_hc_err}")

        # Filter by confidence threshold
        threshold_edges = []
        for edge in edges:
            conf_score = edge.confidence.get("score", 0) if isinstance(edge.confidence, dict) else 0
            if conf_score < MIN_EDGE_CONFIDENCE:
                rejected_edges.append({
                    "source_id": edge.source_id,
                    "target_id": edge.target_id,
                    "relationship_type": edge.relationship_type,
                    "confidence": conf_score,
                    "reason": f"Below threshold ({MIN_EDGE_CONFIDENCE})",
                })
            else:
                threshold_edges.append(edge)
        edges = threshold_edges

        # Keep adversarial scores in sync with the final edge set: dedup
        # (reverse-key matches) and thresholding drop edges after scores are
        # computed in the relation loop, which would otherwise export scores
        # for edges that no longer exist.
        final_edge_ids = {e.id for e in edges}
        adversarial_scores = [a for a in adversarial_scores if a.edge_id in final_edge_ids]

        # Ensure provenance depth is non-zero for multi-evidence edges
        for edge in edges:
            if len(edge.supporting_evidence) > 1 and edge.derivation_depth < 1:
                edge.derivation_depth = 1
            # Non-empty dependency_group when evidence shares a source file batch
            if not edge.dependency_group and edge.supporting_evidence:
                edge.dependency_group = f"src:{edge.supporting_evidence[0]}"

        # Compute multiplexity ties (Category D)
        multiplexity_ties = compute_multiplexity_ties(edges)

        # Compute graph statistics (Category F)
        graph_stats = compute_graph_statistics(nodes, edges, multiplexity_ties)

        # Count adversarial flagged edges
        adversarial_flagged = sum(1 for a in adversarial_scores if a.is_suspicious)

        # Write output files
        out_path.mkdir(parents=True, exist_ok=True)

        with open(out_path / "graph_nodes.json", "w") as f:
            json.dump([n.to_dict() for n in nodes], f, indent=2, default=str)

        with open(out_path / "graph_edges.json", "w") as f:
            json.dump([e.to_dict() for e in edges], f, indent=2, default=str)

        with open(out_path / "multiplexity_ties.json", "w") as f:
            json.dump([t.to_dict() for t in multiplexity_ties], f, indent=2, default=str)

        with open(out_path / "graph_statistics.json", "w") as f:
            json.dump(graph_stats, f, indent=2, default=str)

        with open(out_path / "missing_edges.json", "w") as f:
            json.dump([m.to_dict() for m in missing_edges], f, indent=2, default=str)

        if rejected_edges:
            with open(out_path / "rejected_edges.json", "w") as f:
                json.dump(rejected_edges, f, indent=2, default=str)

        if adversarial_scores:
            with open(out_path / "adversarial_scores.json", "w") as f:
                json.dump([a.to_dict() for a in adversarial_scores], f, indent=2, default=str)

        # Write provenance chains for all edges — depth reflects supporting evidence
        provenance_chains = []
        for edge in edges:
            depth = edge.derivation_depth
            if len(edge.supporting_evidence) > 1 and depth < 1:
                depth = 1
            if len(edge.supporting_evidence) > 3 and depth < 2:
                depth = 2
            chain = {
                "id": f"prov_{edge.id}",
                "node_id": edge.id,
                "derivation": build_derivation_chain(
                    entity_id=edge.source_id,
                    source_file=edge.supporting_evidence[0] if edge.supporting_evidence else "",
                    derivation_depth=depth,
                    extraction_id=edge.id,
                    resolution_id=edge.source_id,
                ),
                "depth": depth,
                "is_independent": edge.is_independent,
                "dependency_group": edge.dependency_group or (
                    f"src:{edge.supporting_evidence[0]}" if edge.supporting_evidence else ""
                ),
                "source_evidence": edge.supporting_evidence,
            }
            provenance_chains.append(chain)
        with open(out_path / "provenance_chains.json", "w") as f:
            json.dump(provenance_chains, f, indent=2, default=str)

        # Build edge type summary
        edge_types = {}
        for edge in edges:
            et = edge.edge_type
            edge_types[et] = edge_types.get(et, 0) + 1

        # Populate knowledge graph from resolved nodes/edges (was always empty)
        try:
            from ..resolution.graph_schema import KnowledgeGraph
            kg = KnowledgeGraph()
            kg_nodes_src = [
                {
                    "id": n.id,
                    "entity_type": n.node_type.upper(),
                    "name": n.name,
                    "attributes": n.attributes,
                    "confidence": {"score": n.effective_confidence or n.confidence},
                    "epistemic_category": n.epistemic_status,
                    "source": {"file_name": (n.provenance_chain or [""])[0]},
                }
                for n in nodes
            ]
            kg_rels_src = [
                {
                    "id": e.id,
                    "source_entity_id": e.source_id,
                    "target_entity_id": e.target_id,
                    "relation_type": e.relationship_type,
                    "temporal_info": e.temporal_info or {},
                    "confidence": e.confidence if isinstance(e.confidence, dict) else {"score": e.confidence},
                    "source": {"file_name": (e.supporting_evidence or [""])[0]},
                    "attributes": {},
                    "epistemic_category": e.epistemic_status,
                }
                for e in edges
            ]
            kg.build_from_entities(kg_nodes_src, kg_rels_src, run_id=run_id)
            kg.save(str(out_path))
        except Exception as kg_err:
            print(f"[GRAPH] Knowledge graph export skipped: {kg_err}")

        # Print summary
        print(f"[GRAPH] Nodes: {len(nodes)}, Edges: {len(edges)}")
        print(f"[GRAPH] Attribute edges (B): {len(attr_edges)}, Co-occurrence edges (C): {len(cooccurrence_edges)}")
        print(f"[GRAPH] Multiplexity ties: {len(multiplexity_ties)}")
        print(f"[GRAPH] Missing edges: {len(missing_edges)}, Rejected edges: {len(rejected_edges)}")
        print(f"[GRAPH] Adversarial flagged: {adversarial_flagged}")
        print(f"[GRAPH] Edge types: {edge_types}")

        return {
            "nodes": len(nodes),
            "edges": len(edges),
            "edge_types": edge_types,
            "attribute_edges": len(attr_edges),
            "cooccurrence_edges": len(cooccurrence_edges),
            "provenance_chains": len(provenance_chains),
            "missing_edges": len(missing_edges),
            "rejected_edges": len(rejected_edges),
            "adversarial_flagged": adversarial_flagged,
            "multiplexity_ties": len(multiplexity_ties),
        }
