"""Analytics Engine — Stage 6 orchestrator.

Reads all Stage 5 output files, computes derived analytics, and writes results.
No graph modification — only reads and computes derived properties.
"""

import json
import time
from pathlib import Path
from datetime import datetime
from typing import List, Dict

from .graph_utils import build_networkx_graph, load_json_file
from .centrality import compute_centrality_scores
from .communities import compute_communities
from .anomalies import compute_anomaly_signals
from .baselines import compute_behavioral_baselines
from .correlation import compute_correlations
from .components import compute_component_analyses
from .temporal_patterns import compute_temporal_patterns
from .geospatial import compute_zone_scores
from .derived_connections import compute_derived_connections
from .models import AnalyticsOutput


def _apply_id_bridge(records: List[dict], id_map: Dict[str, str], key: str = "entity_id") -> List[dict]:
    """Remap raw entity IDs to resolved graph node IDs via id_map.json."""
    if not id_map:
        return records
    out = []
    for rec in records:
        if not isinstance(rec, dict):
            out.append(rec)
            continue
        rec = dict(rec)
        eid = rec.get(key, "")
        if eid and eid in id_map:
            rec[key] = id_map[eid]
        out.append(rec)
    return out


class AnalyticsEngine:
    """Stage 6: Computes derived analytics over the evidence graph."""

    def analyze(self, output_dir: str, run_id: str = "") -> dict:
        start_time = time.time()
        out = Path(output_dir)

        # 1. Load all inputs from disk
        nodes_data = load_json_file(out, "graph_nodes.json", [])
        edges_data = load_json_file(out, "graph_edges.json", [])
        graph_stats = load_json_file(out, "graph_statistics.json", {})
        temporal_infos = load_json_file(out, "temporal_infos.json", [])
        spatial_infos = load_json_file(out, "spatial_infos.json", [])
        coverage_intervals = load_json_file(out, "coverage_intervals.json", [])
        adversarial_scores = load_json_file(out, "adversarial_scores.json", [])
        id_map = load_json_file(out, "id_map.json", {})
        missing_edges = load_json_file(out, "missing_edges.json", [])
        multi_hop_paths = load_json_file(out, "multi_hop_paths.json", [])

        # Load contradictions (dict keyed by ID)
        contradictions_file = out / "contradictions.json"
        if contradictions_file.exists():
            with open(contradictions_file) as f:
                contradictions = json.load(f)
        else:
            contradictions = {}

        temporal_contradictions = load_json_file(out, "temporal_contradictions.json", [])

        # ── ID bridge: remap raw IDs → graph node IDs (fixes cross-stage fracture) ──
        temporal_infos = _apply_id_bridge(temporal_infos, id_map)
        spatial_infos = _apply_id_bridge(spatial_infos, id_map)
        coverage_intervals = _apply_id_bridge(coverage_intervals, id_map)
        temporal_contradictions = _apply_id_bridge(temporal_contradictions, id_map)
        # Remap contradiction entity_ids (list)
        remapped_contradictions = {}
        for cid, contr in contradictions.items():
            if isinstance(contr, dict):
                contr = dict(contr)
                eids = contr.get("entity_ids", [])
                contr["entity_ids"] = [id_map.get(e, e) for e in eids]
            remapped_contradictions[cid] = contr
        contradictions = remapped_contradictions

        # 2. Build NetworkX graph
        G = build_networkx_graph(nodes_data, edges_data)

        # 3. Compute all analytics
        centrality_scores = compute_centrality_scores(G, nodes_data, run_id)
        communities = compute_communities(G, run_id)
        behavioral_baselines = compute_behavioral_baselines(G, temporal_infos, spatial_infos, run_id)
        component_analyses = compute_component_analyses(G, temporal_infos, spatial_infos, run_id)
        anomaly_signals = compute_anomaly_signals(
            G, centrality_scores, temporal_infos, contradictions,
            temporal_contradictions, coverage_intervals, adversarial_scores, run_id
        )
        correlations = compute_correlations(G, temporal_infos, run_id)
        temporal_patterns = compute_temporal_patterns(G, temporal_infos, run_id)

        # Derived hidden connections: co-location, temporal co-occurrence, shared-associate
        derived_connections = compute_derived_connections(
            G, nodes_data, edges_data, temporal_infos, spatial_infos,
            missing_edges, multi_hop_paths, run_id,
        )

        # Geospatial zone risk scoring — only location-type nodes
        location_entities = [n for n in nodes_data if n.get("node_type") == "location"]
        zone_scores = compute_zone_scores(
            location_entities,
            nodes_data,
            contradictions,
            temporal_infos,
            run_id,
        )

        # 4. Compute data quality notes
        data_quality_notes = self._assess_data_quality(
            len(nodes_data), len(edges_data), graph_stats,
            len(temporal_infos), spatial_infos, len(coverage_intervals),
            communities, component_analyses
        )

        # 5. Build output
        processing_time = round(time.time() - start_time, 2)
        output = AnalyticsOutput(
            run_id=run_id,
            timestamp=datetime.now().isoformat(),
            graph_nodes_analyzed=len(nodes_data),
            graph_edges_analyzed=len(edges_data),
            connected_components=graph_stats.get("connected_components", 0),
            largest_component_size=max((c.size for c in component_analyses), default=0),
            centrality_scores=centrality_scores,
            communities=communities,
            anomaly_signals=anomaly_signals,
            behavioral_baselines=behavioral_baselines,
            correlations=correlations,
            component_analyses=component_analyses,
            temporal_patterns=temporal_patterns,
            zone_scores=zone_scores,
            processing_time_seconds=processing_time,
            data_quality_notes=data_quality_notes,
        )

        # 6. Write outputs
        self._write_outputs(output, output_dir)

        # Write derived connections separately
        with open(out / "derived_connections.json", "w") as f:
            json.dump([d if isinstance(d, dict) else d for d in derived_connections], f, indent=2, default=str)

        # 7. Print summary
        print(f"[ANALYTICS] Nodes analyzed: {len(centrality_scores)}")
        print(f"[ANALYTICS] Communities: {len(communities)} (size>=2)")
        print(f"[ANALYTICS] Anomaly signals: {len(anomaly_signals)}")
        print(f"[ANALYTICS] Behavioral baselines: {len(behavioral_baselines)}")
        print(f"[ANALYTICS] Correlations: {len(correlations)}")
        print(f"[ANALYTICS] Component analyses: {len(component_analyses)}")
        print(f"[ANALYTICS] Temporal patterns: {len(temporal_patterns)}")
        print(f"[ANALYTICS] Zone scores: {len(zone_scores)}")
        print(f"[ANALYTICS] Derived connections: {len(derived_connections)}")
        print(f"[ANALYTICS] Data quality notes: {len(data_quality_notes)}")
        print(f"[ANALYTICS] Processing time: {processing_time}s")

        return {
            "centrality_scores": len(centrality_scores),
            "communities": len(communities),
            "anomaly_signals": len(anomaly_signals),
            "behavioral_baselines": len(behavioral_baselines),
            "correlations": len(correlations),
            "component_analyses": len(component_analyses),
            "temporal_patterns": len(temporal_patterns),
            "zone_scores": len(zone_scores),
            "derived_connections": len(derived_connections),
        }

    def _assess_data_quality(
        self, num_nodes, num_edges, graph_stats,
        num_temporal, spatial_infos, num_coverage,
        communities, component_analyses
    ) -> List[str]:
        """Produce honest documentation of what the data supports."""
        notes = []
        num_spatial = len(spatial_infos)

        density = graph_stats.get("density", 0)
        if density < 0.01:
            notes.append(f"Graph is very sparse (density={density}). Many algorithms produce trivial results for isolated nodes.")

        num_components = graph_stats.get("connected_components", 0)
        if num_components > num_nodes * 0.5:
            multi = len([c for c in component_analyses if c.size > 1])
            notes.append(f"Graph has {num_components} connected components for {num_nodes} nodes. "
                         f"Most nodes are isolated. Graph algorithms limited to {multi} components of size >= 2.")

        if num_spatial > 0:
            null_spatial = sum(
                1 for s in spatial_infos
                if s.get("latitude") is None and s.get("lat") is None
            )
            if null_spatial == num_spatial:
                notes.append(
                    f"Spatial data: {num_spatial} records available but ALL have null latitude/longitude. "
                    "Extracted spatial text was geocoded separately for zone scoring."
                )

        if num_temporal < num_nodes:
            notes.append(f"Temporal coverage is uneven: {num_temporal} temporal_infos for {num_nodes} nodes. "
                         f"~{num_nodes - num_temporal} nodes have no temporal data.")

        if not communities:
            notes.append("No multi-node communities detected. The graph may be too sparse for community analysis.")

        return notes

    def _write_outputs(self, output: AnalyticsOutput, output_dir: str):
        """Write all analytics output files."""
        out = Path(output_dir)

        # Main analytics output
        with open(out / "analytics_output.json", "w") as f:
            json.dump(output.to_dict(), f, indent=2)

        # Individual output files
        files = {
            "centrality_scores.json": [c.to_dict() for c in output.centrality_scores],
            "community_assignments.json": [c.to_dict() for c in output.communities],
            "anomaly_signals.json": [a.to_dict() for a in output.anomaly_signals],
            "behavioral_baselines.json": [b.to_dict() for b in output.behavioral_baselines],
            "correlations.json": [c.to_dict() for c in output.correlations],
            "component_analysis.json": [c.to_dict() for c in output.component_analyses],
            "temporal_patterns.json": [t.to_dict() for t in output.temporal_patterns],
            "zone_scores.json": [z.to_dict() for z in output.zone_scores] if output.zone_scores else [],
        }

        for filename, data in files.items():
            with open(out / filename, "w") as f:
                json.dump(data, f, indent=2)

        # Summary dict
        summary = {
            "run_id": output.run_id,
            "timestamp": output.timestamp,
            "graph_nodes_analyzed": output.graph_nodes_analyzed,
            "graph_edges_analyzed": output.graph_edges_analyzed,
            "connected_components": output.connected_components,
            "largest_component_size": output.largest_component_size,
            "processing_time_seconds": output.processing_time_seconds,
            "centrality_scores": len(output.centrality_scores),
            "communities": len(output.communities),
            "anomaly_signals": len(output.anomaly_signals),
            "behavioral_baselines": len(output.behavioral_baselines),
            "correlations": len(output.correlations),
            "component_analyses": len(output.component_analyses),
            "temporal_patterns": len(output.temporal_patterns),
            "zone_scores": len(output.zone_scores),
            "data_quality_notes": output.data_quality_notes,
        }
        with open(out / "analytics_summary.json", "w") as f:
            json.dump(summary, f, indent=2)
