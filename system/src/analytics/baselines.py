"""Behavioral baseline computation — establishes normal per entity type."""

from typing import List
from collections import defaultdict, Counter
import networkx as nx
import numpy as np

from .models import BehavioralBaseline


def compute_behavioral_baselines(
    G: nx.Graph,
    temporal_infos: List[dict],
    spatial_infos: List[dict],
    run_id: str = "",
) -> List[BehavioralBaseline]:
    """Compute per-entity-type behavioral baselines.

    Groups nodes by type and computes statistics: degree, confidence,
    temporal/spatial counts, edge confidence, provenance depth.
    """
    if len(G.nodes) == 0:
        return []

    # Group nodes by type
    nodes_by_type = defaultdict(list)
    for node_id in G.nodes():
        node_type = G.nodes[node_id].get("node_type", "unknown")
        nodes_by_type[node_type].append(node_id)

    # Build temporal/spatial lookup by entity_id
    temporal_by_entity = defaultdict(int)
    for t in temporal_infos:
        temporal_by_entity[t.get("entity_id", "")] += 1

    spatial_by_entity = defaultdict(int)
    for s in spatial_infos:
        spatial_by_entity[s.get("entity_id", "")] += 1

    baselines = []
    for node_type, node_ids in nodes_by_type.items():
        degrees = [G.degree(n) for n in node_ids]
        confidences = [G.nodes[n].get("effective_confidence", 0.5) for n in node_ids]
        temporal_counts = [temporal_by_entity.get(n, 0) for n in node_ids]
        spatial_counts = [spatial_by_entity.get(n, 0) for n in node_ids]
        depths = [G.nodes[n].get("derivation_depth", 0) for n in node_ids]

        # Edge confidence for edges involving this type
        edge_confs = []
        relationship_types = []
        for u, v, data in G.edges(data=True):
            if G.nodes[u].get("node_type") == node_type or G.nodes[v].get("node_type") == node_type:
                conf = data.get("confidence", {})
                if isinstance(conf, dict):
                    edge_confs.append(conf.get("score", 0.5))
                relationship_types.append(data.get("relationship_type", ""))

        dominant_rel = Counter(relationship_types).most_common(1)[0][0] if relationship_types else ""

        temporal_coverage = sum(1 for c in temporal_counts if c > 0) / len(node_ids) if node_ids else 0
        spatial_coverage = sum(1 for c in spatial_counts if c > 0) / len(node_ids) if node_ids else 0

        baselines.append(BehavioralBaseline(
            entity_type=node_type,
            sample_size=len(node_ids),
            avg_degree=round(float(np.mean(degrees)), 4) if degrees else 0.0,
            median_degree=round(float(np.median(degrees)), 4) if degrees else 0.0,
            std_degree=round(float(np.std(degrees)), 4) if degrees else 0.0,
            avg_confidence=round(float(np.mean(confidences)), 4) if confidences else 0.0,
            median_confidence=round(float(np.median(confidences)), 4) if confidences else 0.0,
            avg_temporal_count=round(float(np.mean(temporal_counts)), 4) if temporal_counts else 0.0,
            avg_spatial_count=round(float(np.mean(spatial_counts)), 4) if spatial_counts else 0.0,
            avg_edge_confidence=round(float(np.mean(edge_confs)), 4) if edge_confs else 0.0,
            avg_provenance_depth=round(float(np.mean(depths)), 4) if depths else 0.0,
            temporal_coverage_fraction=round(temporal_coverage, 4),
            spatial_coverage_fraction=round(spatial_coverage, 4),
            dominant_relationship_type=dominant_rel,
            run_id=run_id,
        ))

    baselines.sort(key=lambda b: b.sample_size, reverse=True)
    return baselines
