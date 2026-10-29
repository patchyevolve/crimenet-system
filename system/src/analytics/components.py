"""Component analysis — analyzes each connected component in the graph."""

from typing import List, Dict, Optional
from collections import defaultdict, Counter
import networkx as nx

from .models import ComponentAnalysis


def compute_component_analyses(
    G: nx.Graph,
    temporal_infos: List[dict],
    spatial_infos: List[dict],
    run_id: str = "",
) -> List[ComponentAnalysis]:
    """Analyze each connected component in the graph.

    For each component: size, edge count, density, node type breakdown,
    investigation candidacy, average confidence, diameter, key entities.
    """
    if len(G.nodes) == 0:
        return []

    temporal_by_entity = defaultdict(int)
    for t in temporal_infos:
        temporal_by_entity[t.get("entity_id", "")] += 1

    spatial_by_entity = defaultdict(int)
    for s in spatial_infos:
        spatial_by_entity[s.get("entity_id", "")] += 1

    analyses = []
    for comp_id, comp_nodes in enumerate(nx.connected_components(G)):
        subgraph = G.subgraph(comp_nodes)

        type_counts = Counter(G.nodes[n].get("node_type", "unknown") for n in comp_nodes)

        has_person = type_counts.get("person", 0) > 0
        has_financial = type_counts.get("account", 0) > 0 or type_counts.get("amount", 0) > 0
        has_communication = type_counts.get("phone", 0) > 0

        has_temporal = any(temporal_by_entity.get(n, 0) > 0 for n in comp_nodes)
        has_spatial = any(spatial_by_entity.get(n, 0) > 0 for n in comp_nodes)

        # Heuristic: person + financial + communication = investigation candidate
        is_candidate = has_person and (has_financial or has_communication)

        # Average confidence
        confs = [G.nodes[n].get("effective_confidence", 0.5) for n in comp_nodes]
        avg_conf = sum(confs) / len(confs) if confs else 0.0

        # Diameter (only for small components)
        path_length = None
        if 2 <= len(comp_nodes) <= 50:
            try:
                path_length = nx.diameter(subgraph)
            except nx.NetworkXError:
                path_length = None

        # Key entities (highest degree within component)
        sorted_by_degree = sorted(comp_nodes, key=lambda n: subgraph.degree(n), reverse=True)
        key_entities = sorted_by_degree[:min(3, len(sorted_by_degree))]

        analyses.append(ComponentAnalysis(
            component_id=comp_id,
            size=len(comp_nodes),
            edge_count=subgraph.number_of_edges(),
            density=round(nx.density(subgraph), 6) if len(comp_nodes) > 1 else 0.0,
            node_types=dict(type_counts),
            has_person=has_person,
            has_financial=has_financial,
            has_communication=has_communication,
            has_temporal_data=has_temporal,
            has_spatial_data=has_spatial,
            is_candidate_for_investigation=is_candidate,
            avg_confidence=round(avg_conf, 4),
            path_length=path_length,
            key_entities=key_entities,
            run_id=run_id,
        ))

    analyses.sort(key=lambda a: a.size, reverse=True)
    return analyses
