"""Community detection — finds clusters of related entities."""

from typing import List
from collections import Counter
import networkx as nx

from .models import Community


def compute_communities(G: nx.Graph, run_id: str = "") -> List[Community]:
    """Detect communities using greedy modularity optimization.

    Returns only communities of size >= 2 (isolated nodes are not communities).
    Falls back to connected-component decomposition if modularity fails.
    """
    if len(G.nodes) == 0 or G.number_of_edges() == 0:
        return []

    try:
        communities_gen = nx.community.greedy_modularity_communities(G, resolution=1.0)
        communities_list = list(communities_gen)
    except Exception:
        # Fallback: each connected component with >= 2 nodes is a community
        communities_list = [
            comp for comp in nx.connected_components(G) if len(comp) >= 2
        ]

    node_types = {n: G.nodes[n].get("node_type", "unknown") for n in G.nodes}

    result = []
    for idx, community_nodes in enumerate(communities_list):
        if len(community_nodes) < 2:
            continue

        subgraph = G.subgraph(community_nodes)
        internal_edges = subgraph.number_of_edges()
        n = len(community_nodes)
        max_edges = n * (n - 1) / 2 if n > 1 else 1
        density = internal_edges / max_edges if max_edges > 0 else 0.0

        # Dominant node type
        type_counts = Counter(node_types.get(n, "unknown") for n in community_nodes)
        dominant_type = type_counts.most_common(1)[0][0] if type_counts else "unknown"

        # Modularity contribution (approximate)
        modularity = density * len(community_nodes) / len(G.nodes) if len(G.nodes) > 0 else 0.0

        result.append(Community(
            community_id=f"COMM_{idx:03d}",
            node_ids=sorted(list(community_nodes)),
            size=len(community_nodes),
            modularity_contribution=round(modularity, 4),
            internal_edge_count=internal_edges,
            density=round(density, 4),
            dominant_node_type=dominant_type,
            run_id=run_id,
        ))

    result.sort(key=lambda c: c.size, reverse=True)
    return result
