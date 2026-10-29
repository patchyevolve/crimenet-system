-- Initialize Criminal Network Analysis Database
-- This script runs when the PostgreSQL container starts for the first time

-- Enable pgvector extension for face embeddings and semantic search
CREATE EXTENSION IF NOT EXISTS vector;

-- Enable uuid-ossp for UUID generation
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- ============================================================
-- Core scope tables
-- ============================================================

CREATE TABLE IF NOT EXISTS "JurisdictionNode" (
    "id"          VARCHAR(255) PRIMARY KEY,
    "name"        VARCHAR(255) NOT NULL,
    "nodeType"    VARCHAR(50)  NOT NULL DEFAULT 'CITY',
    "status"      VARCHAR(50)  NOT NULL DEFAULT 'ACTIVE',
    "createdAt"   TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS "Case" (
    "id"                 VARCHAR(255) PRIMARY KEY,
    "jurisdictionNodeId" VARCHAR(255) REFERENCES "JurisdictionNode"("id") ON DELETE SET NULL,
    "caseStatus"         VARCHAR(50)  NOT NULL DEFAULT 'ACTIVE',
    "createdAt"          TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS "PipelineRun" (
    "id"                 VARCHAR(255) PRIMARY KEY,
    "caseId"             VARCHAR(255) REFERENCES "Case"("id") ON DELETE CASCADE,
    "jurisdictionNodeId" VARCHAR(255) REFERENCES "JurisdictionNode"("id") ON DELETE SET NULL,
    "triggerType"        VARCHAR(50)  NOT NULL,
    "inputFiles"         JSONB        NOT NULL DEFAULT '[]'::jsonb,
    "pipelineVersion"    VARCHAR(50)  NOT NULL DEFAULT 'v4.0',
    "status"             VARCHAR(50)  NOT NULL DEFAULT 'RUNNING',
    "startTime"          TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    "endTime"            TIMESTAMP WITH TIME ZONE,
    "summary"            JSONB
);

-- ============================================================
-- Ingestion tables
-- ============================================================

CREATE TABLE IF NOT EXISTS "IngestedFile" (
    "id"             VARCHAR(255) PRIMARY KEY,
    "runId"          VARCHAR(255) REFERENCES "PipelineRun"("id") ON DELETE CASCADE,
    "caseId"         VARCHAR(255),
    "fileName"       VARCHAR(255) NOT NULL,
    "filePath"       TEXT         NOT NULL,
    "fileExt"        VARCHAR(50),
    "fileHash"       VARCHAR(128),
    "fileSizeBytes"  BIGINT       DEFAULT 0,
    "detectedType"   VARCHAR(50),
    "sourceType"     VARCHAR(50),
    "ingestionTime"  TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS "AdversarialCheck" (
    "id"                VARCHAR(255) PRIMARY KEY,
    "runId"             VARCHAR(255) REFERENCES "PipelineRun"("id") ON DELETE CASCADE,
    "ingestedFileId"    VARCHAR(255) REFERENCES "IngestedFile"("id") ON DELETE CASCADE,
    "isSuspicious"      BOOLEAN      DEFAULT FALSE,
    "behavioralAnomaly" BOOLEAN      DEFAULT FALSE,
    "temporalAnomaly"   BOOLEAN      DEFAULT FALSE,
    "contentAnomaly"    BOOLEAN      DEFAULT FALSE,
    "duplicateSuspect"  BOOLEAN      DEFAULT FALSE,
    "score"             DOUBLE PRECISION DEFAULT 0.0,
    "reasons"           JSONB        DEFAULT '[]'::jsonb
);

-- ============================================================
-- Extraction tables
-- ============================================================

CREATE TABLE IF NOT EXISTS "ExtractedEntity" (
    "id"                 VARCHAR(255) PRIMARY KEY,
    "runId"              VARCHAR(255) REFERENCES "PipelineRun"("id") ON DELETE CASCADE,
    "caseId"             VARCHAR(255),
    "ingestedFileId"     VARCHAR(255) REFERENCES "IngestedFile"("id") ON DELETE CASCADE,
    "entityType"         VARCHAR(50)  NOT NULL,
    "mentionText"        TEXT         NOT NULL,
    "confidenceScore"    DOUBLE PRECISION DEFAULT 0.0,
    "epistemicCategory"  VARCHAR(50)  DEFAULT 'INFERENCE',
    "sourceReliability"  INT          DEFAULT 3,
    "evidenceBasis"      INT          DEFAULT 1,
    "derivationDepth"    INT          DEFAULT 0,
    "createdAt"          TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS "ExtractedRelation" (
    "id"                 VARCHAR(255) PRIMARY KEY,
    "runId"              VARCHAR(255) REFERENCES "PipelineRun"("id") ON DELETE CASCADE,
    "caseId"             VARCHAR(255),
    "ingestedFileId"     VARCHAR(255) REFERENCES "IngestedFile"("id") ON DELETE CASCADE,
    "sourceEntityId"     VARCHAR(255),
    "targetEntityId"     VARCHAR(255),
    "relationType"       VARCHAR(100) NOT NULL,
    "confidenceScore"    DOUBLE PRECISION DEFAULT 0.0,
    "epistemicCategory"  VARCHAR(50)  DEFAULT 'INFERENCE',
    "createdAt"          TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- ============================================================
-- Resolution + graph tables
-- ============================================================

CREATE TABLE IF NOT EXISTS "ResolvedEntity" (
    "id"                 VARCHAR(255) PRIMARY KEY,
    "runId"              VARCHAR(255) REFERENCES "PipelineRun"("id") ON DELETE CASCADE,
    "caseId"             VARCHAR(255),
    "entityType"         VARCHAR(50)  NOT NULL,
    "canonicalName"      TEXT         NOT NULL,
    "mergeConfidence"    DOUBLE PRECISION DEFAULT 0.0,
    "mergeType"          VARCHAR(50)  DEFAULT 'auto',
    "epistemicCategory"  VARCHAR(50)  DEFAULT 'INFERENCE',
    "phoneNumbers"       JSONB        DEFAULT '[]'::jsonb,
    "plateNumbers"       JSONB        DEFAULT '[]'::jsonb,
    "bankAccounts"       JSONB        DEFAULT '[]'::jsonb,
    "locationNames"      JSONB        DEFAULT '[]'::jsonb,
    "createdAt"          TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS "GraphNode" (
    "id"                 VARCHAR(255) PRIMARY KEY,
    "runId"              VARCHAR(255) REFERENCES "PipelineRun"("id") ON DELETE CASCADE,
    "caseId"             VARCHAR(255),
    "nodeType"           VARCHAR(50)  NOT NULL,
    "canonicalName"      TEXT         NOT NULL,
    "confidence"         DOUBLE PRECISION DEFAULT 0.0,
    "resolutionStatus"   VARCHAR(50)  DEFAULT 'UNRESOLVED',
    "provenanceChain"    JSONB        DEFAULT '[]'::jsonb,
    "createdAt"          TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS "GraphEdge" (
    "id"                 VARCHAR(255) PRIMARY KEY,
    "runId"              VARCHAR(255) REFERENCES "PipelineRun"("id") ON DELETE CASCADE,
    "caseId"             VARCHAR(255),
    "sourceId"           VARCHAR(255),
    "targetId"           VARCHAR(255),
    "relationshipType"   VARCHAR(100) NOT NULL,
    "semanticEdgeType"   VARCHAR(50),
    "edgeType"           VARCHAR(50),
    "confidenceScore"    DOUBLE PRECISION DEFAULT 0.0,
    "adversarialScore"   DOUBLE PRECISION,
    "supportingEvidence" JSONB        DEFAULT '[]'::jsonb,
    "createdAt"          TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- ============================================================
-- Analysis tables
-- ============================================================

CREATE TABLE IF NOT EXISTS "Contradiction" (
    "id"                 VARCHAR(255) PRIMARY KEY,
    "runId"              VARCHAR(255) REFERENCES "PipelineRun"("id") ON DELETE CASCADE,
    "contradictionType"  VARCHAR(100) NOT NULL,
    "severity"           VARCHAR(50)  NOT NULL,
    "description"        TEXT         NOT NULL,
    "entities"           JSONB        NOT NULL DEFAULT '[]'::jsonb,
    "isResolved"         BOOLEAN      DEFAULT FALSE,
    "resolvedById"       VARCHAR(255),
    "resolvedAt"         TIMESTAMP WITH TIME ZONE,
    "createdAt"          TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS "TemporalInfo" (
    "id"                VARCHAR(255) PRIMARY KEY,
    "runId"             VARCHAR(255) REFERENCES "PipelineRun"("id") ON DELETE CASCADE,
    "entityId"          VARCHAR(255),
    "timeExpression"    TEXT         NOT NULL,
    "normalizedStart"   TIMESTAMP WITH TIME ZONE,
    "normalizedEnd"     TIMESTAMP WITH TIME ZONE,
    "granularity"       VARCHAR(50),
    "confidence"        DOUBLE PRECISION DEFAULT 0.0,
    "createdAt"         TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS "SpatialInfo" (
    "id"            VARCHAR(255) PRIMARY KEY,
    "runId"         VARCHAR(255) REFERENCES "PipelineRun"("id") ON DELETE CASCADE,
    "entityId"      VARCHAR(255),
    "locationName"  TEXT,
    "lat"           DOUBLE PRECISION,
    "lng"           DOUBLE PRECISION,
    "radius"        DOUBLE PRECISION,
    "confidence"    DOUBLE PRECISION DEFAULT 0.0,
    "createdAt"     TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- ============================================================
-- Global (cross-case) tables
-- ============================================================

CREATE TABLE IF NOT EXISTS "GlobalEntity" (
    "canonicalId"        VARCHAR(255) PRIMARY KEY,
    "entityType"         VARCHAR(50)  NOT NULL,
    "canonicalName"      TEXT         NOT NULL,
    "phones"             JSONB        DEFAULT '[]'::jsonb,
    "accounts"           JSONB        DEFAULT '[]'::jsonb,
    "addresses"          JSONB        DEFAULT '[]'::jsonb,
    "firstSeen"          TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    "lastSeen"           TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    "totalCases"         INT          DEFAULT 0,
    "totalJurisdictions" INT          DEFAULT 0,
    "createdAt"          TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    "updatedAt"          TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS "GlobalEntityLink" (
    "id"                 VARCHAR(255) PRIMARY KEY,
    "globalEntityId"     VARCHAR(255) REFERENCES "GlobalEntity"("canonicalId") ON DELETE CASCADE,
    "caseId"             VARCHAR(255),
    "localEntityId"      VARCHAR(255),
    "jurisdictionNodeId" VARCHAR(255) REFERENCES "JurisdictionNode"("id") ON DELETE SET NULL,
    "confidence"         DOUBLE PRECISION DEFAULT 0.0,
    "matchType"          VARCHAR(50)  NOT NULL,
    "createdAt"          TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    CONSTRAINT "uq_global_entity_link" UNIQUE ("globalEntityId", "caseId", "localEntityId")
);

CREATE TABLE IF NOT EXISTS "CrossCaseAlert" (
    "id"              VARCHAR(255) PRIMARY KEY,
    "globalEntityId"  VARCHAR(255) REFERENCES "GlobalEntity"("canonicalId") ON DELETE CASCADE,
    "severity"        VARCHAR(50)  DEFAULT 'MEDIUM',
    "recommendation"  TEXT,
    "createdAt"       TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    "status"          VARCHAR(50)  DEFAULT 'ACTIVE',
    -- Queried by src/global_push/engine.py workspace-index export.
    "reviewedById"    VARCHAR(255),
    "reviewedAt"      TIMESTAMP WITH TIME ZONE
);

CREATE TABLE IF NOT EXISTS "CrossCaseAlertCase" (
    "id"                 VARCHAR(255) PRIMARY KEY,
    "alertId"            VARCHAR(255) REFERENCES "CrossCaseAlert"("id") ON DELETE CASCADE,
    "caseId"             VARCHAR(255),
    "jurisdictionNodeId" VARCHAR(255),
    CONSTRAINT "uq_cross_case_alert_case" UNIQUE ("alertId", "caseId")
);

-- ============================================================
-- Indexes on the pipeline's hot join/filter columns
-- ============================================================

CREATE INDEX IF NOT EXISTS idx_pipeline_run_case_id        ON "PipelineRun"        ("caseId");
CREATE INDEX IF NOT EXISTS idx_ingested_file_run_id        ON "IngestedFile"       ("runId");
CREATE INDEX IF NOT EXISTS idx_ingested_file_case_id       ON "IngestedFile"       ("caseId");
CREATE INDEX IF NOT EXISTS idx_adversarial_check_run_id    ON "AdversarialCheck"   ("runId");
CREATE INDEX IF NOT EXISTS idx_extracted_entity_run_id     ON "ExtractedEntity"    ("runId");
CREATE INDEX IF NOT EXISTS idx_extracted_entity_case_id    ON "ExtractedEntity"    ("caseId");
CREATE INDEX IF NOT EXISTS idx_extracted_relation_run_id   ON "ExtractedRelation"  ("runId");
CREATE INDEX IF NOT EXISTS idx_extracted_relation_case_id  ON "ExtractedRelation"  ("caseId");
CREATE INDEX IF NOT EXISTS idx_resolved_entity_run_id      ON "ResolvedEntity"     ("runId");
CREATE INDEX IF NOT EXISTS idx_resolved_entity_case_id     ON "ResolvedEntity"     ("caseId");
CREATE INDEX IF NOT EXISTS idx_graph_node_run_id           ON "GraphNode"          ("runId");
CREATE INDEX IF NOT EXISTS idx_graph_node_case_id          ON "GraphNode"          ("caseId");
CREATE INDEX IF NOT EXISTS idx_graph_edge_run_id           ON "GraphEdge"          ("runId");
CREATE INDEX IF NOT EXISTS idx_graph_edge_case_id          ON "GraphEdge"          ("caseId");
CREATE INDEX IF NOT EXISTS idx_contradiction_run_id        ON "Contradiction"      ("runId");
CREATE INDEX IF NOT EXISTS idx_temporal_info_run_id        ON "TemporalInfo"       ("runId");
CREATE INDEX IF NOT EXISTS idx_spatial_info_run_id         ON "SpatialInfo"        ("runId");
CREATE INDEX IF NOT EXISTS idx_global_entity_link_case_id  ON "GlobalEntityLink"   ("caseId");
CREATE INDEX IF NOT EXISTS idx_cross_case_alert_case_id    ON "CrossCaseAlertCase" ("caseId");

-- Verify extensions
SELECT extname, extversion FROM pg_extension WHERE extname IN ('vector', 'uuid-ossp');
