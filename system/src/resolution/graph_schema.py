"""
Graph Schema — Playbook V4 Module C.2
DEPRECATED: This file is retained for reference only. The active graph builder is
src/graph/builder.py which implements the full architecture (confidence propagation,
provenance chains, adversarial detection, missing edge tracking, etc.).

This file contains the original Playbook V4 knowledge graph schema with 8 node types
and 7 edge types. It is NOT used by the pipeline.
"""

from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional
from datetime import datetime
import json
import os


# ──────────────────────────────────────────────
# Node Types (Playbook V4 §4 Module C.2)
# ──────────────────────────────────────────────

NODE_TYPES = {
    "Person": {
        "description": "Individual with photo thumbnail + linked face-embedding IDs",
        "required_fields": ["name"],
        "optional_fields": ["photo_thumbnail", "face_embedding_ids", "dob", "gender", "aliases"],
    },
    "Phone": {
        "description": "Phone number with carrier info",
        "required_fields": ["number"],
        "optional_fields": ["carrier", "type", "registered_to"],
    },
    "Vehicle": {
        "description": "Vehicle with registration",
        "required_fields": ["plate_number"],
        "optional_fields": ["make", "model", "color", "registered_to"],
    },
    "Location": {
        "description": "Physical location",
        "required_fields": ["name"],
        "optional_fields": ["lat", "lng", "address", "type"],
    },
    "Organization": {
        "description": "Organization or business",
        "required_fields": ["name"],
        "optional_fields": ["type", "registration_number", "address"],
    },
    "BankAccount": {
        "description": "Bank account",
        "required_fields": ["account_number"],
        "optional_fields": ["bank_name", "account_holder", "balance"],
    },
    "Case": {
        "description": "Case or event (FIR, investigation)",
        "required_fields": ["case_number"],
        "optional_fields": ["title", "date_filed", "police_station", "status", "offense_type"],
    },
    "PoliceStation": {
        "description": "Police station / jurisdiction",
        "required_fields": ["name"],
        "optional_fields": ["jurisdiction", "address", "contact"],
    },
}


# ──────────────────────────────────────────────
# Edge Types (Playbook V4 §4 Module C.2)
# ──────────────────────────────────────────────

EDGE_TYPES = {
    "Call": {
        "description": "Phone call between two parties",
        "source_types": ["Phone", "Person"],
        "target_types": ["Phone", "Person"],
        "required_fields": ["timestamp", "source_file", "confidence"],
        "optional_fields": ["duration_seconds", "tower_location"],
    },
    "Transaction": {
        "description": "Financial transaction between accounts",
        "source_types": ["BankAccount", "Person"],
        "target_types": ["BankAccount", "Person"],
        "required_fields": ["timestamp", "source_file", "confidence"],
        "optional_fields": ["amount", "currency", "description"],
    },
    "CoOccurrence": {
        "description": "Two entities appear in the same FIR/case",
        "source_types": ["Person", "Phone", "Vehicle", "Location"],
        "target_types": ["Person", "Phone", "Vehicle", "Location"],
        "required_fields": ["timestamp", "source_file", "confidence", "case_number"],
        "optional_fields": ["context"],
    },
    "FamilyOrAssociate": {
        "description": "Family relationship or known association",
        "source_types": ["Person"],
        "target_types": ["Person"],
        "required_fields": ["timestamp", "source_file", "confidence"],
        "optional_fields": ["relationship_type"],
    },
    "Ownership": {
        "description": "Entity owns another entity",
        "source_types": ["Person", "Organization"],
        "target_types": ["Phone", "Vehicle", "BankAccount"],
        "required_fields": ["timestamp", "source_file", "confidence"],
        "optional_fields": ["since_date"],
    },
    "Communication": {
        "description": "Social media / messaging communication",
        "source_types": ["Person", "Phone"],
        "target_types": ["Person", "Phone"],
        "required_fields": ["timestamp", "source_file", "confidence"],
        "optional_fields": ["platform", "message_count"],
    },
    "PhysicalProximity": {
        "description": "Entities seen at the same location (CCTV, surveillance)",
        "source_types": ["Person", "Vehicle"],
        "target_types": ["Location"],
        "required_fields": ["timestamp", "source_file", "confidence"],
        "optional_fields": ["camera_id", "duration_minutes"],
    },
}


# ──────────────────────────────────────────────
# Graph Node
# ──────────────────────────────────────────────

@dataclass
class GraphNode:
    """A node in the knowledge graph (Playbook V4 Module C.2)."""
    id: str
    node_type: str          # One of NODE_TYPES keys
    name: str
    attributes: dict = field(default_factory=dict)
    source_files: List[str] = field(default_factory=list)
    confidence: float = 0.0
    epistemic_status: str = "observation"  # observation | inference | hypothesis
    created_at: str = ""
    run_id: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


# ──────────────────────────────────────────────
# Graph Edge
# ──────────────────────────────────────────────

@dataclass
class GraphEdge:
    """
    An edge in the knowledge graph.
    Per Playbook V4: each edge has timestamp + source + confidence score.
    """
    id: str
    source_id: str
    target_id: str
    edge_type: str          # One of EDGE_TYPES keys
    timestamp: str          # When this relationship occurred
    source_file: str        # Which evidence file proves this edge
    confidence: float       # 0.0-1.0
    attributes: dict = field(default_factory=dict)
    epistemic_status: str = "observation"
    run_id: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


# ──────────────────────────────────────────────
# Graph Builder
# ──────────────────────────────────────────────

class KnowledgeGraph:
    """
    Builds and manages the knowledge graph per Playbook V4 Module C.2.
    Creates properly typed nodes and edges from resolved entities + relations.
    """

    def __init__(self):
        self.nodes: Dict[str, GraphNode] = {}
        self.edges: List[GraphEdge] = []

    def add_node(self, node: GraphNode):
        """Add or update a node in the graph."""
        if node.id in self.nodes:
            existing = self.nodes[node.id]
            # Merge source files
            for sf in node.source_files:
                if sf not in existing.source_files:
                    existing.source_files.append(sf)
            # Update confidence if higher
            if node.confidence > existing.confidence:
                existing.confidence = node.confidence
        else:
            self.nodes[node.id] = node

    def add_edge(self, edge: GraphEdge):
        """Add an edge to the graph."""
        self.edges.append(edge)

    def build_from_entities(
        self,
        entities: List[dict],
        relations: List[dict],
        run_id: str = "",
    ) -> dict:
        """
        Build the knowledge graph from extracted entities and relations.
        Maps entity types to proper node types, relation types to edge types.
        """
        # Entity type mapping
        ENTITY_TYPE_MAP = {
            "PERSON": "Person",
            "PHONE": "Phone",
            "VEHICLE": "Vehicle",
            "LOCATION": "Location",
            "ORGANIZATION": "Organization",
            "ACCOUNT": "BankAccount",
            "EVENT": "Case",
            "DATE": "Date",
            "AMOUNT": "Amount",
            "DEVICE": "Device",
        }

        # Create nodes from entities
        for entity in entities:
            eid = entity.get("id", "")
            etype = entity.get("entity_type", "")
            node_type = ENTITY_TYPE_MAP.get(etype)

            if node_type is None:
                continue  # Skip date/amount/device-only entities

            source_file = ""
            source = entity.get("source", {})
            if isinstance(source, dict):
                source_file = source.get("file_name", "")

            conf_dict = entity.get("confidence", {})
            confidence = conf_dict.get("score", 0.5) if isinstance(conf_dict, dict) else 0.5

            node = GraphNode(
                id=eid,
                node_type=node_type,
                name=entity.get("name", eid),
                attributes=entity.get("attributes", {}),
                source_files=[source_file] if source_file else [],
                confidence=confidence,
                epistemic_status=entity.get("epistemic_category", "observation"),
                run_id=run_id,
                created_at=datetime.now().isoformat(),
            )
            self.add_node(node)

        # Edge type mapping
        EDGE_TYPE_MAP = {
            "CALLED": "Call",
            "TRANSFERRED_TO": "Transaction",
            "RECEIVED_FROM": "Transaction",
            "OWNS_ACCOUNT": "Ownership",
            "ASSOCIATED_WITH": "Communication",
            "MESSED": "Communication",
            "FAMILY_OF": "FamilyOrAssociate",
            "FRIEND_OF": "FamilyOrAssociate",
            "ASSOCIATE_OF": "FamilyOrAssociate",
            "WORKS_WITH": "FamilyOrAssociate",
            "VISITED": "PhysicalProximity",
            "SUSPECT_OF": "CoOccurrence",
            "VICTIM_OF": "CoOccurrence",
            "WITNESS_OF": "CoOccurrence",
        }

        # Create edges from relations
        for rel in relations:
            rel_type = rel.get("relation_type", "")
            edge_type = EDGE_TYPE_MAP.get(rel_type, "Communication")

            # Get timestamp
            temporal = rel.get("temporal_info", {})
            timestamp = ""
            if isinstance(temporal, dict):
                timestamp = temporal.get("timestamp", "")

            source_file = ""
            source = rel.get("source", {})
            if isinstance(source, dict):
                source_file = source.get("file_name", "")

            conf_dict = rel.get("confidence", {})
            confidence = conf_dict.get("score", 0.5) if isinstance(conf_dict, dict) else 0.5

            edge = GraphEdge(
                id=rel.get("id", ""),
                source_id=rel.get("source_entity_id", ""),
                target_id=rel.get("target_entity_id", ""),
                edge_type=edge_type,
                timestamp=timestamp,
                source_file=source_file,
                confidence=confidence,
                attributes=rel.get("attributes", {}),
                epistemic_status=rel.get("epistemic_category", "observation"),
                run_id=run_id,
            )
            self.add_edge(edge)

        return self.get_summary()

    def get_summary(self) -> dict:
        """Graph summary statistics."""
        node_types = {}
        for node in self.nodes.values():
            node_types[node.node_type] = node_types.get(node.node_type, 0) + 1

        edge_types = {}
        for edge in self.edges:
            edge_types[edge.edge_type] = edge_types.get(edge.edge_type, 0) + 1

        return {
            "total_nodes": len(self.nodes),
            "total_edges": len(self.edges),
            "node_types": node_types,
            "edge_types": edge_types,
        }

    def save(self, output_dir: str):
        """Save graph to JSON files."""
        os.makedirs(output_dir, exist_ok=True)

        with open(os.path.join(output_dir, "knowledge_graph_nodes.json"), "w") as f:
            json.dump([n.to_dict() for n in self.nodes.values()], f, indent=2)

        with open(os.path.join(output_dir, "knowledge_graph_edges.json"), "w") as f:
            json.dump([e.to_dict() for e in self.edges], f, indent=2)

        with open(os.path.join(output_dir, "knowledge_graph_summary.json"), "w") as f:
            json.dump(self.get_summary(), f, indent=2)

    def query_by_type(self, node_type: str) -> List[GraphNode]:
        """Query all nodes of a specific type."""
        return [n for n in self.nodes.values() if n.node_type == node_type]

    def get_neighbors(self, node_id: str) -> List[GraphEdge]:
        """Get all edges connected to a node."""
        return [e for e in self.edges if e.source_id == node_id or e.target_id == node_id]

    def get_subgraph(self, node_id: str, max_hops: int = 2) -> 'KnowledgeGraph':
        """Get a subgraph around a specific node (N-hop neighborhood)."""
        subgraph = KnowledgeGraph()
        visited_nodes = set()
        current_nodes = {node_id}

        for hop in range(max_hops + 1):
            next_nodes = set()
            for nid in current_nodes:
                if nid in visited_nodes:
                    continue
                visited_nodes.add(nid)

                if nid in self.nodes:
                    subgraph.add_node(self.nodes[nid])

                for edge in self.get_neighbors(nid):
                    subgraph.add_edge(edge)
                    other = edge.target_id if edge.source_id == nid else edge.source_id
                    next_nodes.add(other)

            current_nodes = next_nodes - visited_nodes

        return subgraph
