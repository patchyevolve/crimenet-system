"""
Database connection module for Criminal Network Analysis System.
Uses Prisma Client Python (prisma-client-py) for PostgreSQL + pgvector.

Source of truth: src/models/schema.py (pipeline v4, Revision 3+)
Derived from: db_schema/DB_SCHEMA.md (v5)
"""

import os
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Optional, List

from prisma import Prisma
from prisma.models import (
    PipelineRun,
    IngestedFile,
    ExtractedEntity,
    ExtractedRelation,
    ResolvedEntity,
    GraphNode,
    GraphEdge,
    Contradiction,
    TemporalInfo,
    SpatialInfo,
    Case,
    FIR,
    Person,
    Suspect,
    Evidence,
    FaceEmbedding,
)

# Global Prisma client instance
_client: Prisma | None = None


async def get_client() -> Prisma:
    """Get or create the Prisma client singleton."""
    global _client
    if _client is None:
        _client = Prisma()
        await _client.connect()
    return _client


async def disconnect():
    """Disconnect the Prisma client."""
    global _client
    if _client is not None:
        await _client.disconnect()
        _client = None


@asynccontextmanager
async def get_db() -> AsyncGenerator[Prisma, None]:
    """Context manager for database operations."""
    client = await get_client()
    try:
        yield client
    finally:
        pass


async def health_check() -> bool:
    """Check database connectivity."""
    try:
        client = await get_client()
        await client.pipelinerun.find_first()
        return True
    except Exception as e:
        print(f"Database health check failed: {e}")
        return False


# ============================================================
# Pipeline Run Operations
# ============================================================

async def create_pipeline_run(
    case_id: str | None = None,
    jurisdiction_node_id: str | None = None,
    trigger_type: str = "MANUAL",
    input_files: list | None = None,
    pipeline_version: str = "v4.0",
) -> PipelineRun:
    """Create a new pipeline run."""
    async with get_db() as db:
        return await db.pipelinerun.create(
            data={
                "caseId": case_id,
                "jurisdictionNodeId": jurisdiction_node_id,
                "triggerType": trigger_type,
                "inputFiles": input_files or [],
                "pipelineVersion": pipeline_version,
                "status": "RUNNING",
            }
        )


async def update_pipeline_run(run_id: str, **kwargs) -> PipelineRun:
    """Update a pipeline run with results."""
    async with get_db() as db:
        return await db.pipelinerun.update(where={"id": run_id}, data=kwargs)


async def complete_pipeline_run(
    run_id: str,
    files_count: int = 0,
    entities_count: int = 0,
    relations_count: int = 0,
    contradictions_count: int = 0,
    graph_nodes_count: int = 0,
    graph_edges_count: int = 0,
) -> PipelineRun:
    """Mark a pipeline run as completed."""
    from datetime import datetime
    return await update_pipeline_run(
        run_id,
        status="COMPLETED",
        completedAt=datetime.utcnow(),
        filesCount=files_count,
        entitiesCount=entities_count,
        relationsCount=relations_count,
        contradictionsCount=contradictions_count,
        graphNodesCount=graph_nodes_count,
        graphEdgesCount=graph_edges_count,
    )


async def fail_pipeline_run(run_id: str, error: str) -> PipelineRun:
    """Mark a pipeline run as failed."""
    from datetime import datetime
    return await update_pipeline_run(
        run_id,
        status="FAILED",
        completedAt=datetime.utcnow(),
        errorMessage=error,
    )


# ============================================================
# Ingested File Operations
# ============================================================

async def create_ingested_file(
    run_id: str,
    file_path: str,
    file_hash: str,
    file_name: str,
    file_type: str,
    file_size: int | None = None,
    source_reliability: int = 3,
    source_origin_type: str = "OBSERVED",
    document_type: str = "OTHER",
    narrative_type: str | None = None,
    entity_density: float | None = None,
) -> IngestedFile:
    """Create an ingested file record."""
    async with get_db() as db:
        return await db.ingestedfile.create(
            data={
                "runId": run_id,
                "filePath": file_path,
                "fileHash": file_hash,
                "fileName": file_name,
                "fileType": file_type,
                "fileSize": file_size,
                "sourceReliability": source_reliability,
                "sourceOriginType": source_origin_type,
                "documentType": document_type,
                "narrativeType": narrative_type,
                "entityDensity": entity_density,
                "status": "INGESTED",
            }
        )


async def update_ingested_file(file_id: str, **kwargs) -> IngestedFile:
    """Update an ingested file record."""
    async with get_db() as db:
        return await db.ingestedfile.update(where={"id": file_id}, data=kwargs)


# ============================================================
# Extracted Entity Operations
# ============================================================

async def create_extracted_entity(
    run_id: str,
    ingested_file_id: str,
    entity_type: str,
    mention_text: str,
    confidence_score: float,
    start_offset: int | None = None,
    end_offset: int | None = None,
    epistemic_category: str = "INFERENCE",
    source_reliability: int = 3,
    evidence_basis: int = 1,
    derivation_depth: int = 0,
) -> ExtractedEntity:
    """Create an extracted entity record."""
    async with get_db() as db:
        return await db.extractedentity.create(
            data={
                "runId": run_id,
                "ingestedFileId": ingested_file_id,
                "entityType": entity_type,
                "mentionText": mention_text,
                "confidenceScore": confidence_score,
                "startOffset": start_offset,
                "endOffset": end_offset,
                "epistemicCategory": epistemic_category,
                "sourceReliability": source_reliability,
                "evidenceBasis": evidence_basis,
                "derivationDepth": derivation_depth,
            }
        )


async def create_extracted_entities_batch(entities: List[dict]) -> List[ExtractedEntity]:
    """Batch create extracted entities."""
    async with get_db() as db:
        return await db.extractedentity.create_many(data=entities, skip_duplicates=True)


# ============================================================
# Extracted Relation Operations
# ============================================================

async def create_extracted_relation(
    run_id: str,
    ingested_file_id: str,
    source_entity_id: str,
    target_entity_id: str,
    relation_type: str,
    confidence_score: float,
    epistemic_category: str = "INFERENCE",
    source_reliability: int = 3,
    evidence_basis: int = 1,
    derivation_depth: int = 0,
) -> ExtractedRelation:
    """Create an extracted relation record."""
    async with get_db() as db:
        return await db.extractedrelation.create(
            data={
                "runId": run_id,
                "ingestedFileId": ingested_file_id,
                "sourceEntityId": source_entity_id,
                "targetEntityId": target_entity_id,
                "relationType": relation_type,
                "confidenceScore": confidence_score,
                "epistemicCategory": epistemic_category,
                "sourceReliability": source_reliability,
                "evidenceBasis": evidence_basis,
                "derivationDepth": derivation_depth,
            }
        )


async def create_extracted_relations_batch(relations: List[dict]) -> List[ExtractedRelation]:
    """Batch create extracted relations."""
    async with get_db() as db:
        return await db.extractedrelation.create_many(data=relations, skip_duplicates=True)


# ============================================================
# Resolved Entity Operations
# ============================================================

async def create_resolved_entity(
    run_id: str,
    entity_type: str,
    canonical_name: str,
    merge_confidence: float,
    merge_type: str,
    epistemic_category: str = "INFERENCE",
    source_reliability: int = 3,
    evidence_basis: int = 1,
    derivation_depth: int = 0,
    phone_numbers: list | None = None,
    plate_numbers: list | None = None,
    bank_accounts: list | None = None,
    location_names: list | None = None,
) -> ResolvedEntity:
    """Create a resolved entity record."""
    async with get_db() as db:
        return await db.resolvedentity.create(
            data={
                "runId": run_id,
                "entityType": entity_type,
                "canonicalName": canonical_name,
                "mergeConfidence": merge_confidence,
                "mergeType": merge_type,
                "epistemicCategory": epistemic_category,
                "sourceReliability": source_reliability,
                "evidenceBasis": evidence_basis,
                "derivationDepth": derivation_depth,
                "phoneNumbers": phone_numbers,
                "plateNumbers": plate_numbers,
                "bankAccounts": bank_accounts,
                "locationNames": location_names,
            }
        )


# ============================================================
# Temporal Info Operations
# ============================================================

async def create_temporal_info(
    run_id: str,
    entity_id: str,
    time_expression: str,
    normalized_start: str | None = None,
    normalized_end: str | None = None,
    time_type: str = "POINT",
    confidence: float = 0.0,
    source_field: str | None = None,
) -> TemporalInfo:
    """Create a temporal info record."""
    async with get_db() as db:
        return await db.temporalinfo.create(
            data={
                "runId": run_id,
                "entityId": entity_id,
                "timeExpression": time_expression,
                "normalizedStart": normalized_start,
                "normalizedEnd": normalized_end,
                "timeType": time_type,
                "confidence": confidence,
                "sourceField": source_field,
            }
        )


# ============================================================
# Spatial Info Operations
# ============================================================

async def create_spatial_info(
    run_id: str,
    entity_id: str,
    location_text: str,
    normalized_location: str | None = None,
    latitude: float | None = None,
    longitude: float | None = None,
    confidence: float = 0.0,
    source_field: str | None = None,
) -> SpatialInfo:
    """Create a spatial info record."""
    async with get_db() as db:
        return await db.spatialinfo.create(
            data={
                "runId": run_id,
                "entityId": entity_id,
                "locationText": location_text,
                "normalizedLocation": normalized_location,
                "latitude": latitude,
                "longitude": longitude,
                "confidence": confidence,
                "sourceField": source_field,
            }
        )


# ============================================================
# Graph Node Operations
# ============================================================

async def create_graph_node(
    run_id: str,
    node_type: str,
    canonical_name: str,
    confidence: float,
    resolution_status: str = "UNRESOLVED",
    provenance_chain: dict | None = None,
) -> GraphNode:
    """Create a graph node record."""
    async with get_db() as db:
        return await db.graphnode.create(
            data={
                "runId": run_id,
                "nodeType": node_type,
                "canonicalName": canonical_name,
                "confidence": confidence,
                "resolutionStatus": resolution_status,
                "provenanceChain": provenance_chain,
            }
        )


async def create_graph_nodes_batch(nodes: List[dict]) -> List[GraphNode]:
    """Batch create graph nodes."""
    async with get_db() as db:
        return await db.graphnode.create_many(data=nodes, skip_duplicates=True)


# ============================================================
# Graph Edge Operations
# ============================================================

async def create_graph_edge(
    run_id: str,
    source_id: str,
    target_id: str,
    relationship_type: str,
    semantic_edge_type: str,
    edge_type: str,
    confidence_score: float,
    adversarial_score: float | None = None,
    supporting_evidence: dict | None = None,
) -> GraphEdge:
    """Create a graph edge record."""
    async with get_db() as db:
        return await db.graphedge.create(
            data={
                "runId": run_id,
                "sourceId": source_id,
                "targetId": target_id,
                "relationshipType": relationship_type,
                "semanticEdgeType": semantic_edge_type,
                "edgeType": edge_type,
                "confidenceScore": confidence_score,
                "adversarialScore": adversarial_score,
                "supportingEvidence": supporting_evidence,
            }
        )


async def create_graph_edges_batch(edges: List[dict]) -> List[GraphEdge]:
    """Batch create graph edges."""
    async with get_db() as db:
        return await db.graphedge.create_many(data=edges, skip_duplicates=True)


# ============================================================
# Contradiction Operations
# ============================================================

async def create_contradiction(
    run_id: str,
    contradiction_type: str,
    severity: str,
    description: str,
    entities: List[str],
) -> Contradiction:
    """Create a contradiction record."""
    async with get_db() as db:
        return await db.contradiction.create(
            data={
                "runId": run_id,
                "contradictionType": contradiction_type,
                "severity": severity,
                "description": description,
                "entities": entities,
            }
        )


# ============================================================
# FIR Operations
# ============================================================

async def create_fir(
    case_id: str,
    fir_number: str,
    jurisdiction_node_id: str,
    description: str,
    filed_by_id: str,
    investigation_status: str = "REGISTERED",
    legal_disposition: str = "NONE",
    record_status: str = "ACTIVE",
    fir_document_r2_key: str | None = None,
) -> FIR:
    """Create a FIR record."""
    async with get_db() as db:
        return await db.fir.create(
            data={
                "caseId": case_id,
                "firNumber": fir_number,
                "jurisdictionNodeId": jurisdiction_node_id,
                "description": description,
                "filedById": filed_by_id,
                "investigationStatus": investigation_status,
                "legalDisposition": legal_disposition,
                "recordStatus": record_status,
                "firDocumentR2Key": fir_document_r2_key,
            }
        )


# ============================================================
# Case Operations
# ============================================================

async def create_case(
    jurisdiction_node_id: str,
    location_id: str | None = None,
    case_status: str = "ACTIVE",
    description: str | None = None,
) -> Case:
    """Create a case record."""
    async with get_db() as db:
        return await db.case.create(
            data={
                "jurisdictionNodeId": jurisdiction_node_id,
                "locationId": location_id,
                "caseStatus": case_status,
                "description": description,
            }
        )


# ============================================================
# Person Operations
# ============================================================

async def create_person(
    canonical_name: str,
    gender: str | None = None,
    dob: str | None = None,
    national_id_number: str | None = None,
    photo_r2_key: str | None = None,
    identity_source: str = "MANUAL",
    name_confidence: float | None = None,
    created_by_id: str | None = None,
) -> Person:
    """Create a person record."""
    async with get_db() as db:
        return await db.person.create(
            data={
                "canonicalName": canonical_name,
                "gender": gender,
                "dob": dob,
                "nationalIdNumber": national_id_number,
                "photoR2Key": photo_r2_key,
                "identitySource": identity_source,
                "nameConfidence": name_confidence,
                "createdById": created_by_id,
            }
        )


# ============================================================
# Suspect Operations
# ============================================================

async def create_suspect(
    case_id: str,
    name: str,
    gender: str | None = None,
    person_id: str | None = None,
    dob: str | None = None,
    national_id_number: str | None = None,
    status: str = "UNDER_INVESTIGATION",
    arrest_date: str | None = None,
    contact_info: str | None = None,
    photo_r2_key: str | None = None,
    notes: str | None = None,
) -> Suspect:
    """Create a suspect record."""
    async with get_db() as db:
        return await db.suspect.create(
            data={
                "caseId": case_id,
                "name": name,
                "gender": gender,
                "personId": person_id,
                "dob": dob,
                "nationalIdNumber": national_id_number,
                "status": status,
                "arrestDate": arrest_date,
                "contactInfo": contact_info,
                "photoR2Key": photo_r2_key,
                "notes": notes,
            }
        )


# ============================================================
# Evidence Operations
# ============================================================

async def create_evidence(
    case_id: str,
    type: str,
    logged_by_id: str,
    suspect_id: str | None = None,
    description: str | None = None,
    status: str = "LOGGED",
) -> Evidence:
    """Create an evidence record."""
    async with get_db() as db:
        return await db.evidence.create(
            data={
                "caseId": case_id,
                "type": type,
                "loggedById": logged_by_id,
                "suspectId": suspect_id,
                "description": description,
                "status": status,
            }
        )


# ============================================================
# Face Embedding Operations
# ============================================================

async def create_face_embedding(
    source_image_r2_key: str,
    embedding_vector: List[float],
    detector_confidence: float,
    person_id: str | None = None,
    suspect_id: str | None = None,
    evidence_file_id: str | None = None,
    source_type: str | None = None,
) -> FaceEmbedding:
    """Create a face embedding record."""
    async with get_db() as db:
        return await db.faceembedding.create(
            data={
                "sourceImageR2Key": source_image_r2_key,
                "embeddingVector": embedding_vector,
                "detectorConfidence": detector_confidence,
                "personId": person_id,
                "suspectId": suspect_id,
                "evidenceFileId": evidence_file_id,
                "sourceType": source_type,
            }
        )


async def find_similar_faces(
    query_embedding: List[float],
    threshold: float = 0.6,
    limit: int = 10,
) -> List[dict]:
    """Find similar face embeddings using pgvector cosine similarity."""
    async with get_db() as db:
        results = await db.query_raw(
            """
            SELECT 
                id,
                person_id,
                suspect_id,
                source_image_r2_key,
                detector_confidence,
                match_confidence,
                status,
                1 - (embedding_vector <=> $1::vector) as similarity
            FROM face_embeddings
            WHERE status = 'CONFIRMED'
            ORDER BY embedding_vector <=> $1::vector
            LIMIT $2
            """,
            str(query_embedding),
            limit,
        )
        return [r for r in results if r["similarity"] >= threshold]


# ============================================================
# Audit Log Operations
# ============================================================

async def create_audit_log(
    user_id: str,
    action: str,
    resource_type: str,
    resource_id: str,
    metadata: dict | None = None,
    ip_address: str | None = None,
) -> dict:
    """Create an audit log entry."""
    async with get_db() as db:
        return await db.auditlog.create(
            data={
                "userId": user_id,
                "action": action,
                "resourceType": resource_type,
                "resourceId": resource_id,
                "metadata": metadata,
                "ipAddress": ip_address,
            }
        )


# ============================================================
# Query Helpers
# ============================================================

async def get_pipeline_run_with_details(run_id: str) -> dict | None:
    """Get a pipeline run with all its related data."""
    async with get_db() as db:
        run = await db.pipelinerun.find_unique(
            where={"id": run_id},
            include={
                "ingestedFiles": True,
                "graphNodes": True,
                "graphEdges": True,
                "contradictions": True,
            }
        )
        return run


async def get_case_with_entities(case_id: str) -> dict | None:
    """Get a case with all its entities and evidence."""
    async with get_db() as db:
        case = await db.case.find_unique(
            where={"id": case_id},
            include={
                "entities": {"include": {"person": True}},
                "suspects": True,
                "evidence": True,
                "fir": True,
            }
        )
        return case


async def search_persons_by_name(name: str) -> List[Person]:
    """Search persons by name (case-insensitive partial match)."""
    async with get_db() as db:
        return await db.person.find_many(
            where={"canonicalName": {"contains": name, "mode": "insensitive"}},
            take=20,
        )


async def get_suspect_with_person(suspect_id: str) -> dict | None:
    """Get a suspect with its resolved person."""
    async with get_db() as db:
        suspect = await db.suspect.find_unique(
            where={"id": suspect_id},
            include={"person": True, "case": True}
        )
        return suspect