"""Centrality computation — identifies the most important nodes in the graph."""

from typing import List
import networkx as nx
import numpy as np

from .models import CentralityScore


def compute_centrality_scores(
    G: nx.Graph, nodes_data: List[dict], run_id: str = ""
) -> List[CentralityScore]:
    """Compute centrality metrics for all nodes.

    Metrics: degree, betweenness, closeness, eigenvector, pagerank.
    Falls back gracefully for disconnected graphs.
    """
    if len(G.nodes) == 0:
        return []

    # Build node type lookup
    node_types = {n: G.nodes[n].get("node_type", "unknown") for n in G.nodes}
    node_names = {n: G.nodes[n].get("name", n) for n in G.nodes}

    # Degree centrality (trivial — normalized degree)
    degree_cent = nx.degree_centrality(G)

    # Betweenness centrality (works on disconnected graphs via nx)
    betweenness_cent = nx.betweenness_centrality(G, normalized=True)

    # Closeness centrality — only meaningful for connected components
    closeness_cent = {}
    for component in nx.connected_components(G):
        if len(component) < 2:
            for n in component:
                closeness_cent[n] = 0.0
            continue
        subgraph = G.subgraph(component)
        cc = nx.closeness_centrality(subgraph)
        closeness_cent.update(cc)

    # Eigenvector centrality — may not converge for disconnected graphs
    eigenvector_cent = {}
    try:
        eigenvector_cent = nx.eigenvector_centrality(G, max_iter=1000, tol=1e-06)
    except nx.PowerIterationFailedConvergence:
        # Fallback: compute per connected component
        for component in nx.connected_components(G):
            if len(component) < 2:
                for n in component:
                    eigenvector_cent[n] = 0.0
                continue
            subgraph = G.subgraph(component)
            try:
                ec = nx.eigenvector_centrality(subgraph, max_iter=1000, tol=1e-06)
                eigenvector_cent.update(ec)
            except nx.PowerIterationFailedConvergence:
                for n in component:
                    eigenvector_cent[n] = 0.0

    # PageRank — requires scipy; fall back to degree-based approximation
    try:
        pagerank = nx.pagerank(G, alpha=0.85)
    except ModuleNotFoundError:
        total_degree = sum(dict(G.degree()).values()) or 1
        pagerank = {n: G.degree(n) / total_degree for n in G.nodes}

    scores = []
    for node_id in G.nodes:
        scores.append(CentralityScore(
            node_id=node_id,
            node_type=node_types.get(node_id, "unknown"),
            name=node_names.get(node_id, node_id),
            degree_centrality=round(degree_cent.get(node_id, 0.0), 6),
            betweenness_centrality=round(betweenness_cent.get(node_id, 0.0), 6),
            closeness_centrality=round(closeness_cent.get(node_id, 0.0), 6),
            eigenvector_centrality=round(eigenvector_cent.get(node_id, 0.0), 6),
            pagerank=round(pagerank.get(node_id, 0.0), 6),
            degree=G.degree(node_id),
            run_id=run_id,
        ))

    scores.sort(key=lambda s: s.degree_centrality, reverse=True)
    return scores
