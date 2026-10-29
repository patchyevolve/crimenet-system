"""Graph utilities — build NetworkX graph from Stage 5 JSON output."""

import json
from pathlib import Path
from typing import List, Dict, Tuple
import networkx as nx


def load_json(path: Path) -> any:
    """Load a JSON file, return empty structure on error."""
    if not path.exists():
        return [] if "list" in str(path) else {}
    with open(path) as f:
        return json.load(f)


def build_networkx_graph(nodes_data: List[dict], edges_data: List[dict]) -> nx.Graph:
    """Convert Stage 5 JSON nodes/edges into a NetworkX graph.

    Node attributes: node_type, name, effective_confidence, epistemic_status,
                     derivation_depth, provenance_chain, resolution_status.
    Edge attributes: relationship_type, edge_type, confidence (dict), adversarial_score.
    """
    G = nx.Graph()

    for node in nodes_data:
        node_id = node["id"]
        G.add_node(
            node_id,
            node_type=node.get("node_type", "unknown"),
            name=node.get("name", ""),
            effective_confidence=node.get("effective_confidence", 0.5),
            epistemic_status=node.get("epistemic_status", "observation"),
            derivation_depth=node.get("derivation_depth", 0),
            provenance_chain=node.get("provenance_chain", []),
            resolution_status=node.get("resolution_status", ""),
        )

    for edge in edges_data:
        src = edge["source_id"]
        tgt = edge["target_id"]
        if src not in G or tgt not in G:
            continue
        if src == tgt:
            continue

        conf = edge.get("confidence", {})
        conf_score = conf.get("score", 0.5) if isinstance(conf, dict) else 0.5

        G.add_edge(
            src, tgt,
            relationship_type=edge.get("relationship_type", ""),
            edge_type=edge.get("edge_type", ""),
            confidence=conf,
            confidence_score=conf_score,
            adversarial_score=edge.get("adversarial_score", 0.0),
            source_name=edge.get("source_name", ""),
            target_name=edge.get("target_name", ""),
        )

    return G


def load_json_file(output_dir: Path, filename: str, default=None):
    """Load a JSON file from the output directory."""
    path = output_dir / filename
    if not path.exists():
        return default if default is not None else []
    with open(path) as f:
        return json.load(f)
