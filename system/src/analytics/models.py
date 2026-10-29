"""Analytics output models — Stage 6 outputs."""

from dataclasses import dataclass, field
from typing import List, Dict, Optional


@dataclass
class CentralityScore:
    """Centrality metrics for a single node."""
    node_id: str
    node_type: str
    name: str
    degree_centrality: float = 0.0
    betweenness_centrality: float = 0.0
    closeness_centrality: float = 0.0
    eigenvector_centrality: float = 0.0
    pagerank: float = 0.0
    degree: int = 0
    run_id: str = ""

    def to_dict(self):
        return {
            "node_id": self.node_id,
            "node_type": self.node_type,
            "name": self.name,
            "degree_centrality": self.degree_centrality,
            "betweenness_centrality": self.betweenness_centrality,
            "closeness_centrality": self.closeness_centrality,
            "eigenvector_centrality": self.eigenvector_centrality,
            "pagerank": self.pagerank,
            "degree": self.degree,
            "run_id": self.run_id,
        }


@dataclass
class Community:
    """A detected community (cluster) in the graph."""
    community_id: str
    node_ids: List[str]
    size: int
    modularity_contribution: float = 0.0
    internal_edge_count: int = 0
    density: float = 0.0
    dominant_node_type: str = ""
    run_id: str = ""

    def to_dict(self):
        return {
            "community_id": self.community_id,
            "node_ids": self.node_ids,
            "size": self.size,
            "modularity_contribution": self.modularity_contribution,
            "internal_edge_count": self.internal_edge_count,
            "density": self.density,
            "dominant_node_type": self.dominant_node_type,
            "run_id": self.run_id,
        }


@dataclass
class AnomalySignal:
    """A detected anomaly in the graph data."""
    signal_id: str
    signal_type: str  # degree_outlier, temporal_burst, confidence_gap, contradiction_cluster, isolated_high_value, coverage_gap
    severity: str     # low, medium, high
    entity_id: str
    entity_type: str
    description: str
    evidence_count: int = 0
    score: float = 0.0
    run_id: str = ""

    def to_dict(self):
        return {
            "signal_id": self.signal_id,
            "signal_type": self.signal_type,
            "severity": self.severity,
            "entity_id": self.entity_id,
            "entity_type": self.entity_type,
            "description": self.description,
            "evidence_count": self.evidence_count,
            "score": self.score,
            "run_id": self.run_id,
        }


@dataclass
class BehavioralBaseline:
    """Baseline statistics for a specific entity type."""
    entity_type: str
    sample_size: int
    avg_degree: float = 0.0
    median_degree: float = 0.0
    std_degree: float = 0.0
    avg_confidence: float = 0.0
    median_confidence: float = 0.0
    avg_temporal_count: float = 0.0
    avg_spatial_count: float = 0.0
    avg_edge_confidence: float = 0.0
    avg_provenance_depth: float = 0.0
    temporal_coverage_fraction: float = 0.0
    spatial_coverage_fraction: float = 0.0
    dominant_relationship_type: str = ""
    run_id: str = ""

    def to_dict(self):
        return {
            "entity_type": self.entity_type,
            "sample_size": self.sample_size,
            "avg_degree": self.avg_degree,
            "median_degree": self.median_degree,
            "std_degree": self.std_degree,
            "avg_confidence": self.avg_confidence,
            "median_confidence": self.median_confidence,
            "avg_temporal_count": self.avg_temporal_count,
            "avg_spatial_count": self.avg_spatial_count,
            "avg_edge_confidence": self.avg_edge_confidence,
            "avg_provenance_depth": self.avg_provenance_depth,
            "temporal_coverage_fraction": self.temporal_coverage_fraction,
            "spatial_coverage_fraction": self.spatial_coverage_fraction,
            "dominant_relationship_type": self.dominant_relationship_type,
            "run_id": self.run_id,
        }


@dataclass
class Correlation:
    """A computed correlation between two variables."""
    variable_a: str
    variable_b: str
    correlation_type: str  # rank, point_biserial, cramers_v
    coefficient: float
    p_value: float = 1.0
    is_significant: bool = False
    sample_size: int = 0
    run_id: str = ""

    def to_dict(self):
        return {
            "variable_a": self.variable_a,
            "variable_b": self.variable_b,
            "correlation_type": self.correlation_type,
            "coefficient": self.coefficient,
            "p_value": self.p_value,
            "is_significant": self.is_significant,
            "sample_size": self.sample_size,
            "run_id": self.run_id,
        }


@dataclass
class ComponentAnalysis:
    """Analysis of a single connected component."""
    component_id: int
    size: int
    edge_count: int
    density: float = 0.0
    node_types: Dict[str, int] = field(default_factory=dict)
    has_person: bool = False
    has_financial: bool = False
    has_communication: bool = False
    has_temporal_data: bool = False
    has_spatial_data: bool = False
    is_candidate_for_investigation: bool = False
    avg_confidence: float = 0.0
    path_length: Optional[int] = None
    key_entities: List[str] = field(default_factory=list)
    run_id: str = ""

    def to_dict(self):
        return {
            "component_id": self.component_id,
            "size": self.size,
            "edge_count": self.edge_count,
            "density": self.density,
            "node_types": self.node_types,
            "has_person": self.has_person,
            "has_financial": self.has_financial,
            "has_communication": self.has_communication,
            "has_temporal_data": self.has_temporal_data,
            "has_spatial_data": self.has_spatial_data,
            "is_candidate_for_investigation": self.is_candidate_for_investigation,
            "avg_confidence": self.avg_confidence,
            "path_length": self.path_length,
            "key_entities": self.key_entities,
            "run_id": self.run_id,
        }


@dataclass
class TemporalPattern:
    """A detected temporal pattern."""
    pattern_id: str
    pattern_type: str  # activity_burst, sequence_gap, temporal_anomaly, regularity
    entity_id: str
    entity_type: str
    description: str
    event_count: int = 0
    time_span_seconds: float = 0.0
    severity: str = "low"
    run_id: str = ""

    def to_dict(self):
        return {
            "pattern_id": self.pattern_id,
            "pattern_type": self.pattern_type,
            "entity_id": self.entity_id,
            "entity_type": self.entity_type,
            "description": self.description,
            "event_count": self.event_count,
            "time_span_seconds": self.time_span_seconds,
            "severity": self.severity,
            "run_id": self.run_id,
        }


@dataclass
class AnalyticsOutput:
    """Complete output from Stage 6 analytics."""
    run_id: str
    timestamp: str
    graph_nodes_analyzed: int
    graph_edges_analyzed: int
    connected_components: int
    largest_component_size: int
    centrality_scores: List[CentralityScore]
    communities: List[Community]
    anomaly_signals: List[AnomalySignal]
    behavioral_baselines: List[BehavioralBaseline]
    correlations: List[Correlation]
    component_analyses: List[ComponentAnalysis]
    temporal_patterns: List[TemporalPattern]
    zone_scores: List = field(default_factory=list)
    processing_time_seconds: float = 0.0
    data_quality_notes: List[str] = field(default_factory=list)

    def to_dict(self):
        return {
            "run_id": self.run_id,
            "timestamp": self.timestamp,
            "graph_nodes_analyzed": self.graph_nodes_analyzed,
            "graph_edges_analyzed": self.graph_edges_analyzed,
            "connected_components": self.connected_components,
            "largest_component_size": self.largest_component_size,
            "centrality_scores": [c.to_dict() for c in self.centrality_scores],
            "communities": [c.to_dict() for c in self.communities],
            "anomaly_signals": [a.to_dict() for a in self.anomaly_signals],
            "behavioral_baselines": [b.to_dict() for b in self.behavioral_baselines],
            "correlations": [c.to_dict() for c in self.correlations],
            "component_analyses": [c.to_dict() for c in self.component_analyses],
            "temporal_patterns": [t.to_dict() for t in self.temporal_patterns],
            "zone_scores": [z.to_dict() for z in self.zone_scores] if self.zone_scores else [],
            "processing_time_seconds": self.processing_time_seconds,
            "data_quality_notes": self.data_quality_notes,
        }
