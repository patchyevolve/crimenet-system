# ============================================================
# FULL SYSTEM DESIGN
# ============================================================
#
# Complete architecture: Frontend → Backend → Pipeline → DB
# Includes scalability analysis, flaws, and improvements.
# ============================================================
#
# DEPRECATION NOTE:
# PoliceStation references (found in IMPLEMENTATION_PLAN.md and
# earlier schema drafts) have been replaced by JurisdictionNode.
# PoliceStation model is retired. All new entities reference
# JurisdictionNode for jurisdictional hierarchy.
# ============================================================


## ============================================================
## 1. FLAWS IN PREVIOUS DESIGN
## ============================================================

### 1.1 Critical Flaws

```
FLAW 1: Monolithic Pipeline
  Problem:  All stages run in sequence, one failure stops everything
  Fix:      Stage isolation with queue-based handoff

FLAW 2: No Async Processing
  Problem:  Synchronous — blocks while processing, can't handle concurrent runs
  Fix:      Message queue (Redis/BullMQ) for async stage execution

FLAW 3: No Incremental Processing
  Problem:  Can't add new evidence without re-running everything
  Fix:      Track processed files, only process new/changed files

FLAW 4: No Rollback
  Problem:  Can't undo a bad pipeline run
  Fix:      Soft deletes + run versioning, ability to revert to previous state

FLAW 5: Tight Coupling
  Problem:  Stages depend on each other directly
  Fix:      Event-driven architecture, stages communicate via DB/events

FLAW 6: No Worker Scaling
  Problem:  Single worker processes everything, can't add more
  Fix:      Worker pool with horizontal scaling

FLAW 7: No Monitoring
  Problem:  No visibility into pipeline health
  Fix:      Health checks, metrics, alerting

FLAW 8: No Retry Logic
  Problem:  Transient failures crash the pipeline
  Fix:      Exponential backoff, dead letter queue
```

### 1.2 Scalability Issues

```
ISSUE 1: Single Database Connection
  Problem:  Bottleneck under load
  Fix:      Connection pooling + read replicas

ISSUE 2: No Caching
  Problem:  Repeated queries hit DB every time
  Fix:      Redis cache for hot data

ISSUE 3: No Table Partitioning
  Problem:  Large tables slow down
  Fix:      Partition by case_id or timestamp

ISSUE 4: No Horizontal Scaling
  Problem:  Can't add more servers
  Fix:      Stateless services + load balancer

ISSUE 5: No Load Balancing
  Problem:  Single point of failure
  Fix:      Multiple instances behind load balancer
```


## ============================================================
## 2. REVISED ARCHITECTURE
## ============================================================

### 2.1 Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              FRONTEND                                       │
│                                                                             │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐          │
│  │  Dashboard   │ │ Case Detail │ │   Graph     │ │  Timeline   │          │
│  │  (Case List) │ │   View      │ │   View      │ │    View     │          │
│  └──────┬──────┘ └──────┬──────┘ └──────┬──────┘ └──────┬──────┘          │
│         │               │               │               │                   │
│         └───────────────┴───────┬───────┴───────────────┘                   │
│                                 │                                           │
│                          REST API calls                                     │
└─────────────────────────────────┼───────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              BACKEND                                        │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                        API Server (FastAPI)                         │   │
│  │                                                                     │   │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐              │   │
│  │  │ /cases   │ │ /evidence│ │ /graph   │ │ /pipeline│              │   │
│  │  │ endpoints│ │ endpoints│ │ endpoints│ │ endpoints│              │   │
│  │  └────┬─────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘              │   │
│  │       │             │             │             │                    │   │
│  │       └─────────────┴──────┬──────┴─────────────┘                    │   │
│  │                            │                                         │   │
│  │                    ┌───────┴───────┐                                 │   │
│  │                    │  Auth Layer   │                                 │   │
│  │                    │  (JWT + RBAC) │                                 │   │
│  │                    └───────┬───────┘                                 │   │
│  └────────────────────────────┼─────────────────────────────────────────┘   │
│                               │                                             │
│  ┌────────────────────────────┼─────────────────────────────────────────┐   │
│  │                    File Upload Handler                               │   │
│  │                               │                                     │   │
│  │                    ┌──────────┴──────────┐                          │   │
│  │                    │  Upload to R2       │                          │   │
│  │                    │  Create Evidence    │                          │   │
│  │                    │  Trigger Pipeline   │                          │   │
│  │                    └──────────┬──────────┘                          │   │
│  └───────────────────────────────┼─────────────────────────────────────┘   │
│                                  │                                         │
└──────────────────────────────────┼─────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           MESSAGE QUEUE                                     │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                      Redis + BullMQ                                 │   │
│  │                                                                     │   │
│  │  Queue: pipeline                                                    │   │
│  │  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐     │   │
│  │  │ Stage 1 │→│ Stage 2 │→│ Stage 3 │→│ Stage 4 │→│ Stage 5 │     │   │
│  │  │ Ingest  │ │ Extract │ │ Resolve │ │ Temporal│ │  Graph  │     │   │
│  │  └─────────┘ └─────────┘ └─────────┘ └─────────┘ └────┬────┘     │   │
│  │                                                        │           │   │
│  │  ┌─────────┐ ┌─────────┐ ┌─────────┐                  │           │   │
│  │  │ Stage 6 │→│ Stage 11 │→│ Stage 12 │←─────────────────┘           │   │
│  │  │Analytics│ │ Global  │ │ Scoped  │                             │   │
│  │  │         │ │ Entity  │ │Analytics│                             │   │
│  │  └─────────┘ └─────────┘ └─────────┘                             │   │
│  │                                                                     │   │
│  │  Dead Letter Queue: pipeline-failed                                 │   │
│  │  Retry Queue: pipeline-retry                                        │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└──────────────────────────────────┬─────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                            WORKER POOL                                      │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    Pipeline Workers (2-8 instances)                  │   │
│  │                                                                     │   │
│  │  Worker 1: Stage 1-2 (Ingestion + Extraction)                      │   │
│  │  Worker 2: Stage 3-4 (Resolution + Temporal)                       │   │
│  │  Worker 3: Stage 5 (Graph Building)                                │   │
│  │  Worker 4: Face Recognition (when enabled)                         │   │
│  │                                                                     │   │
│  │  Each worker:                                                       │   │
│  │  - Pulls job from queue                                             │   │
│  │  - Processes data                                                   │   │
│  │  - Writes results to DB                                             │   │
│  │  - Pushes next stage job to queue                                   │   │
│  │  - Reports status back                                              │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└──────────────────────────────────┬─────────────────────────────────────────┘
                                   │
                    ┌──────────────┴──────────────┐
                    ▼                              ▼
┌────────────────────────────────┐  ┌────────────────────────────────────┐
│       PRIMARY DATABASE         │  │           CACHE LAYER              │
│       (PostgreSQL + pgvector)  │  │           (Redis)                  │
│                                │  │                                    │
│  ┌──────────────────────────┐  │  │  ┌──────────────────────────────┐ │
│  │  Pipeline Tables (35+)    │  │  │  │ Session Cache                │ │
│  │  - PipelineRun            │  │  │  │ - User sessions              │ │
│  │  - IngestedFile           │  │  │  │ - Auth tokens                │ │
│  │  - ExtractedEntity        │  │  │  └──────────────────────────────┘ │
│  │  - ResolvedEntity         │  │  │                                    │
│  │  - GraphNode/Edge         │  │  │  ┌──────────────────────────────┐ │
│  │  - Person/Phone/Vehicle   │  │  │  │ Query Cache                  │ │
│  │  - FaceEmbedding          │  │  │  │ - Frequent queries           │ │
│  │  - AuditLog               │  │  │  │ - Graph traversals           │ │
│  │  - JurisdictionNode       │  │  │  │ - Search results             │ │
│  │  - GlobalEntityIndex      │  │  │  └──────────────────────────────┘ │
│  │  - Case (with FIR link)   │  │  │                                    │
│  └──────────────────────────┘  │  │  ┌──────────────────────────────┐ │
│  │ Read Replica             │  │  │  ┌──────────────────────────────┐ │
│  │ (for queries)            │  │  │  │ Pipeline State               │ │
│  └──────────────────────────┘  │  │  │ - Run status                 │ │
│                                │  │  │ - Stage progress              │ │
└────────────────────────────────┘  │  │ - Error counts               │ │
                                    │  └──────────────────────────────┘ │
                                    └────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           FILE STORAGE                                      │
│                           (Cloudflare R2)                                   │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  Bucket: criminal-evidence                                          │   │
│  │                                                                     │   │
│  │  ├── cases/{case_id}/evidence/{file_id}/                           │   │
│  │  │   ├── original/{filename}.{ext}                                  │   │
│  │  │   ├── processed/                                                 │   │
│  │  │   └── thumbnails/                                                │   │
│  │  │                                                                  │   │
│  │  ├── faces/{embedding_id}/                                          │   │
│  │  │   ├── crop.jpg                                                   │   │
│  │  │   └── embedding.npy                                              │   │
│  │  │                                                                  │   │
│  │  └── exports/{run_id}/                                              │   │
│  │      ├── extraction_output.json                                     │   │
│  │      └── graph_output.json                                          │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘


┌─────────────────────────────────────────────────────────────────────────────┐
│                        TEAMMATE'S BACKEND                                   │
│                        (Stub / Future Integration)                          │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  Their API                                                         │   │
│  │  - Person, Phone, Vehicle tables                                    │   │
│  │  - CaseEntityLink                                                  │   │
│  │  - Frontend display                                                 │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```


## ============================================================
## 2.5 MULTI-CASE ARCHITECTURE (JurisdictionNode)
## ============================================================

### 2.5.1 JurisdictionNode Hierarchy

```
JurisdictionNode replaces PoliceStation as the organizational
unit for cases. JurisdictionNode models a jurisdictional
boundary (district, station, sub-division) with parent-child
hierarchy.

Schema:
  JurisdictionNode {
    id              UUID PRIMARY KEY
    name            TEXT NOT NULL
    code            TEXT UNIQUE NOT NULL        -- e.g. "MH-PUN-01"
    node_type       ENUM('COUNTRY','STATE','DISTRICT','SUB_DIVISION','STATION')
    parent_id       UUID FK → JurisdictionNode
    depth           INT                         -- root=0, station=4
    metadata        JSONB
    created_at      TIMESTAMPTZ DEFAULT now()
  }

Hierarchy:
  COUNTRY (root)
  ├── STATE
  │   ├── DISTRICT
  │   │   ├── SUB_DIVISION
  │   │   │   └── STATION (leaf — closest to PoliceStation)
  │   │   └── STATION
  │   └── DISTRICT
  └── STATE

Each Case belongs to exactly one JurisdictionNode (leaf).
All ancestor JurisdictionNodes are inherited via parent_id.
```

### 2.5.2 Global Entity Identity Index

```
Purpose: Deduplicate entities across cases within the same
JurisdictionNode hierarchy.

Schema:
  GlobalEntityIndex {
    id              UUID PRIMARY KEY
    canonical_id    UUID NOT NULL               -- references ResolvedEntity
    entity_type     ENUM('PERSON','PHONE','VEHICLE','ADDRESS')
    fingerprint     TEXT NOT NULL               -- normalized key (phone: digits, name: soundex+dob)
    confidence      FLOAT
    created_at      TIMESTAMPTZ DEFAULT now()

    -- Which jurisdictions has this entity appeared in?
    jurisdiction_ids  UUID[]                    -- array of JurisdictionNode.id
  }

Uniqueness:
  UNIQUE (entity_type, fingerprint)             -- one canonical per entity type+key

On new ResolvedEntity creation (Stage 3):
  1. Compute fingerprint from entity attributes
  2. Query GlobalEntityIndex for existing match
  3. If match found → link existing ResolvedEntity to current case
  4. If no match   → create new canonical entry + new ResolvedEntity
  5. Append current JurisdictionNode.id to jurisdiction_ids[]
```

### 2.5.3 Scoped Analytical Graphs

```
Purpose: Build case-local subgraphs that are queryable and
composable across jurisdiction boundaries.

Two graph scopes:
  A) Case-Scoped Graph (default)
     - Nodes/Edges belong to one case
     - Built in Stage 5 (per-case)
     - Queried via /api/cases/:id/graph/*

  B) Jurisdiction-Scoped Graph (cross-case)
     - Merges all case graphs within a JurisdictionNode
     - Built in Stage 12 (post-push)
     - Queried via /api/jurisdictions/:id/graph/*
     - Enables cross-case link discovery

Storage:
  - Case-Scoped: GraphNode.case_id = the_case
  - Jurisdiction-Scoped: GraphNode.jurisdiction_id = the_jurisdictions
  - Both share the same schema; differentiated by scope column
```

### 2.5.4 Case-FIR Invariant

```
Invariant: Case ↔ FIR is 1:1 (each Case has exactly one FIR,
each FIR belongs to exactly one Case).

  Case {
    id              UUID PRIMARY KEY
    title           TEXT NOT NULL
    description     TEXT
    jurisdiction_id UUID FK → JurisdictionNode
    status          ENUM('DRAFT','ACTIVE','CLOSED','ARCHIVED')
    primary_fir_id  UUID UNIQUE NOT NULL → FIR  -- 1:1 invariant
    created_at      TIMESTAMPTZ DEFAULT now()
    updated_at      TIMESTAMPTZ DEFAULT now()
  }

  FIR {
    id                    UUID PRIMARY KEY
    fir_number            TEXT NOT NULL
    jurisdiction_id       UUID FK → JurisdictionNode
    case_id               UUID UNIQUE NOT NULL → Case  -- 1:1 invariant
    filing_date           DATE NOT NULL
    investigation_status  ENUM('PENDING','UNDER_INVESTIGATION','CHARGE_SHEET_FILED','CLOSED')
    legal_disposition     ENUM('PENDING','DISMISSED','CHARGESHEET','CONVICTED','ACQUITTED')
    record_status         ENUM('ACTIVE','SEALED','EXPUNGED')
    created_at            TIMESTAMPTZ DEFAULT now()
    updated_at            TIMESTAMPTZ DEFAULT now()
  }

Rules:
  1. Case ↔ FIR is 1:1 (FIR.case_id unique, Case.primary_fir_id unique)
  2. FIR.fir_number is unique WITHIN a JurisdictionNode (not globally)
  3. Case.jurisdiction_id must be a leaf (STATION) node
  4. JurisdictionNode ancestors are derived via parent traversal
  5. A Case cannot be created without a valid JurisdictionNode
  6. FIR lifecycle (investigation_status, legal_disposition, record_status) is independent
     from CaseRelationship, Workspace, and AnalysisRun
  7. FIR lifecycle changes tracked in FIRLifecycleHistory
```


### 2.5.5 Designed Entities (Not Yet Implemented)

```
The following entities are designed but not yet implemented:

CaseRelationship:
  - Models relationships between Cases (e.g., linked investigations)
  - Has CaseRelationshipHistory for audit trail
  - Status lifecycle tracks relationship state
  - FIR lifecycle is independent from CaseRelationship

CaseAccess:
  - Controls authorization for Case access
  - Complements role-based auth with per-Case permissions

InvestigationWorkspace:
  - Organizes investigative work for a Case
  - Contains WorkspaceCase (linking Cases to Workspaces)
  - Contains WorkspaceMember (user membership)
  - Workspace organization is independent from FIR lifecycle

AnalysisRun:
  - Immutable historical execution record (cannot be modified after creation)
  - Captures input snapshots in AnalysisRunResult (inputs frozen at execution time)
  - Separate from PipelineRun (which tracks pipeline execution)
  - AnalysisRunCase links AnalysisRuns to Cases
  - Finding records analytical findings from the run

FIRLifecycleHistory:
  - Audit trail for FIR lifecycle dimension changes
  - Tracks transitions in investigation_status, legal_disposition, record_status
```


## ============================================================
## 3. COMPONENT DETAILS
## ============================================================

### 3.1 Frontend (Stub)

```
Framework: React + TypeScript + Tailwind

Pages:
  1. Dashboard
     - Case list with stats
     - Recent pipeline runs
     - System health

  2. Case Detail
     - Evidence files
     - Entities
     - Graph visualization
     - Timeline
     - Contradictions

  3. Evidence Viewer
     - File preview
     - OCR results
     - Face detection results
     - Quality scores

  4. Search
     - Entity search
     - Graph search
     - Full-text search

  5. Settings
     - Pipeline config
     - User management
     - Audit log

API Calls:
  - GET  /api/cases
  - GET  /api/cases/:id
  - GET  /api/cases/:id/entities
  - GET  /api/cases/:id/graph
  - GET  /api/cases/:id/timeline
  - POST /api/evidence/upload
  - POST /api/pipeline/run
  - GET  /api/pipeline/runs/:id
```

### 3.2 Backend (Stub)

```
Framework: FastAPI (Python)

Components:
  1. REST API Server
     - Route handlers
     - Request validation
     - Response formatting

  2. Auth Layer
     - JWT tokens
     - Role-based access (INSPECTOR, ADMIN, AUDIT_LOGGER)

  3. File Upload Handler
     - Receive files
     - Upload to R2
     - Create Evidence/EvidenceFile records
     - Trigger pipeline via queue

  4. Query Layer
     - Read from DB (primary for writes, replica for reads)
     - Cache hot queries in Redis

  5. Sync Service (Stub)
     - Placeholder for teammate's backend integration
     - POST /api/sync/to-backend

Endpoints:
  Cases:
    GET    /api/cases
    GET    /api/cases/:id
    POST   /api/cases
    PUT    /api/cases/:id

  Evidence:
    GET    /api/cases/:id/evidence
    POST   /api/evidence/upload
    GET    /api/evidence/:id

  Pipeline:
    POST   /api/pipeline/run
    GET    /api/pipeline/runs/:id
    GET    /api/pipeline/runs/:id/status

  Graph:
    GET    /api/cases/:id/graph/nodes
    GET    /api/cases/:id/graph/edges
    GET    /api/cases/:id/graph/adjacency/:nodeId

  Jurisdiction (NEW):
    GET    /api/jurisdictions
    GET    /api/jurisdictions/:id
    GET    /api/jurisdictions/:id/graph/nodes
    GET    /api/jurisdictions/:id/graph/edges
    GET    /api/jurisdictions/:id/cases

  Global Entity (NEW):
    GET    /api/global-entities
    GET    /api/global-entities/:id

  Search:
    GET    /api/search/entities?q=
    GET    /api/search/graph?q=

  Audit:
    GET    /api/audit/logs
```

### 3.3 Message Queue

```
Technology: Redis + BullMQ

Queues:
  1. pipeline (main)
     - Jobs: { run_id, case_id, stage, input }
     - Priority: HIGH for new evidence, MEDIUM for re-processing

  2. pipeline-processing (stage-specific)
     - ingestion: File intake jobs
     - extraction: NER jobs
     - resolution: Merge jobs
     - temporal: Timeline jobs
     - graph: Graph build jobs (case-scoped)
     - global_entity: Global Entity Push jobs (cross-case)
     - scoped_analytics: Jurisdiction-scoped analytics jobs

  3. pipeline-failed (dead letter)
     - Failed jobs after max retries
     - Manual inspection needed

  4. pipeline-retry
     - Jobs being retried
     - Exponential backoff

Job Flow:
  Backend → pipeline queue → Worker picks up → Processes → 
  Writes to DB → Pushes next stage job → Repeat

Retry Logic:
  - Max retries: 3
  - Backoff: 1s, 5s, 25s
  - After max retries → dead letter queue
```

### 3.4 Worker Pool

```
Technology: Python + BullMQ worker

Worker Types:
  1. Ingestion Worker
     - Reads files from R2
     - Computes hashes
     - Creates IngestedFile records
     - Uploads to R2
     - Creates quality/adversarial checks

  2. Extraction Worker
     - Downloads content from R2
     - Runs NER
     - Creates ExtractedEntity/Relation records
     - Stores embeddings (future)

  3. Resolution Worker
     - Loads entities from DB
     - Runs matching algorithms
     - Creates ResolvedEntity records
     - Detects contradictions

  4. Temporal Worker
     - Extracts timestamps
     - Builds timeline
     - Detects temporal contradictions

  5. Graph Worker
     - Creates case-scoped GraphNode/Edge records
     - Creates Knowledge Graph Entities
     - Runs adversarial scoring

  6. Global Entity Worker (NEW)
     - Pushes resolved entities to GlobalEntityIndex
     - Cross-case entity deduplication
     - Manages canonical entity linking

  7. Scoped Analytics Worker (NEW)
     - Merges case graphs into jurisdiction-scoped graphs
     - Runs cross-case link discovery
     - Updates jurisdiction analytics cache

Scaling:
  - Start with 2 workers
  - Scale to 8 workers under load
  - Each worker is stateless
  - Horizontal scaling via adding instances
```

### 3.5 Database

```
Technology: PostgreSQL 15 + pgvector

Primary Database:
  - All writes go here
  - Connection pool: 20 connections
  - Transactions for multi-table writes

Read Replica:
  - All reads go here
  - Reduces load on primary
  - Async replication (lag < 1s)

Caching (Redis):
  - Session data: 24h TTL
  - Query results: 5min TTL
  - Pipeline state: 1h TTL
  - Search results: 10min TTL

Table Partitioning:
  - PipelineRun: partition by start_time (monthly)
  - IngestedFile: partition by ingestion_time (monthly)
  - ExtractedEntity: partition by created_at (monthly)
  - AuditLog: partition by created_at (monthly)

Indexes:
  - All FK columns: btree
  - Confidence scores: btree (for filtering)
  - Timestamps: btree (for range queries)
  - pgvector columns: hnsw (for similarity search)
  - JSONB columns: gin (for containment queries)
```

### 3.6 File Storage

```
Technology: Cloudflare R2

Bucket: criminal-evidence

Structure:
  cases/
    {case_id}/
      evidence/
        {file_id}/
          original/     ← raw evidence file
          processed/    ← extracted text, metadata
          thumbnails/   ← image thumbnails

  faces/
    {embedding_id}/
      crop.jpg          ← face crop
      embedding.npy     ← 512-d vector

  exports/
    {run_id}/
      extraction_output.json
      graph_output.json

Lifecycle:
  - Original files: retained forever (evidence chain)
  - Processed files: retained for 7 years
  - Temporary files: deleted after processing

Access:
  - Private bucket (no public access)
  - Presigned URLs for temporary access (1h TTL)
  - API proxy for authenticated access
```


## ============================================================
## 4. DATA FLOW
## ============================================================

### 4.1 Evidence Upload Flow

```
1. Investigator uploads file via Frontend
2. Frontend → POST /api/evidence/upload
3. Backend receives file
4. Backend uploads to R2: r2://cases/{case_id}/evidence/{file_id}/original/
5. Backend creates Evidence + EvidenceFile records in DB
6. Backend pushes job to pipeline queue
7. Queue acknowledges job
8. Backend returns { evidence_id, file_id, status: 'queued' }
```

### 4.2 Pipeline Processing Flow

```
1. Worker picks up job from queue
2. Worker reads job: { run_id, case_id, file_ids }
3. Worker creates PipelineRun record (status: RUNNING)

--- STAGES 1-6: Per-Case Processing ---

Stage 1: Ingestion
  4. Worker downloads files from R2
  5. Worker computes hashes
  6. Worker creates IngestedFile records
  7. Worker creates DataQualityScore records
  8. Worker creates AdversarialCheck records
  9. Worker pushes Stage 2 job to queue
  10. Worker marks Stage 1 complete

Stage 2: Extraction
  11. Worker picks up Stage 2 job
  12. Worker loads IngestedFile records from DB
  13. Worker downloads content from R2
  14. Worker runs NER
  15. Worker creates ExtractedEntity records
  16. Worker creates ExtractedRelation records
  17. Worker pushes Stage 3 job to queue

Stage 3: Resolution
  18. Worker picks up Stage 3 job
  19. Worker loads ExtractedEntity records from DB
  20. Worker runs matching algorithms
  21. Worker creates ResolvedEntity records
  22. Worker creates Contradiction records
  23. Worker pushes Stage 4 job to queue

Stage 4: Temporal
  24. Worker picks up Stage 4 job
  25. Worker loads entities/relations from DB
  26. Worker extracts timestamps
  27. Worker creates TemporalInfo records
  28. Worker pushes Stage 5 job to queue

Stage 5: Graph (Case-Scoped)
  29. Worker picks up Stage 5 job
  30. Worker loads ResolvedEntity records from DB
  31. Worker creates case-scoped GraphNode records
  32. Worker creates case-scoped GraphEdge records
  33. Worker creates Person/Phone/Vehicle records
  34. Worker pushes Stage 6 job to queue

Stage 6: Analytics
  35. Worker picks up Stage 6 job
  36. Worker runs final contradiction detection
  37. Worker computes case-level confidence scores
  38. Worker marks PipelineRun status: CASE_COMPLETE
  39. Worker pushes Stage 11 job to queue (Global Entity Push)

--- STAGE 11: Global Entity Push (Cross-Case, designed not implemented) ---

Stage 11: Global Entity Push
  40. Worker picks up Stage 11 job
  41. Worker loads all ResolvedEntities from current case
  42. For each entity: compute fingerprint
  43. Query GlobalEntityIndex for existing canonical match
  44. If match found:
      - Link existing canonical ResolvedEntity to current case
      - Append current JurisdictionNode.id to jurisdiction_ids[]
  45. If no match:
      - Create new GlobalEntityIndex entry (canonical_id = new ResolvedEntity)
      - Set jurisdiction_ids[] = [current jurisdiction]
  46. Worker pushes Stage 12 job to queue (Scoped Analytics)

--- STAGE 12: Scoped Analytics (Cross-Case, designed not implemented) ---

Stage 12: Scoped Analytics
  47. Worker picks up Stage 12 job
  48. Worker loads all JurisdictionNodes in case's ancestor chain
  49. For each jurisdiction in chain (bottom-up):
      a. Load all case-scoped graphs within this jurisdiction
      b. Merge into jurisdiction-scoped GraphNode/GraphEdge
      c. Run cross-case link discovery (shared entities, patterns)
      d. Update jurisdiction-scoped analytics cache
  50. Worker marks PipelineRun status: COMPLETED
  51. Worker sends notification (WebSocket/Webhook)
```

### 4.3 Query Flow

```
1. Frontend requests data: GET /api/cases/:id/graph
2. Backend checks Redis cache
3. If cache hit → return cached data
4. If cache miss → query PostgreSQL read replica
5. Store result in Redis (5min TTL)
6. Return data to Frontend
```

### 4.4 Search Flow

```
1. Frontend searches: GET /api/search/entities?q=suresh
2. Backend queries PostgreSQL (full-text search)
3. Backend queries pgvector (similarity search for faces)
4. Backend combines results
5. Backend ranks by relevance + confidence
6. Return results to Frontend
```


## ============================================================
## 5. SCALABILITY
## ============================================================

### 5.1 Horizontal Scaling

```
Frontend:
  - Static files → CDN (Cloudflare)
  - No server scaling needed

Backend:
  - Multiple instances behind load balancer
  - Stateless (no session in memory)
  - Scale: 2-4 instances

Workers:
  - Multiple worker instances
  - Pull from queue (competing consumers)
  - Scale: 2-8 instances based on load

Database:
  - Primary: single (writes)
  - Read replica: 1-2 (reads)
  - Future: sharding by case_id

Cache:
  - Redis cluster (3+ nodes)
  - Scale: add nodes as needed
```

### 5.2 Performance Targets

```
Metric                    Target      Current
──────────────────────────────────────────────
File upload               < 2s        ~1s
Pipeline run (50 files)   < 2min      ~70s
Pipeline run (500 files)  < 10min     N/A
Pipeline run (5000 files) < 60min     N/A
API response time         < 200ms     N/A
Search response time      < 500ms     N/A
Graph query time          < 1s        N/A
Face match time           < 100ms     N/A
```

### 5.3 Storage Limits

```
Component               Limit           Current
──────────────────────────────────────────────
R2 storage              10 TB           ~1 MB
PostgreSQL              1 TB            ~10 MB
Redis cache             10 GB           ~1 MB
Concurrent connections  1000            1
Pipeline runs/day       100             1
```


## ============================================================
## 6. MONITORING
## ============================================================

```
Metrics:
  - Pipeline runs: success/fail count, duration
  - Queue depth: jobs waiting
  - Worker status: active/idle/error
  - DB connections: active/available
  - Cache hit rate: percentage
  - API response time: p50, p95, p99
  - Error rate: errors per minute

Alerts:
  - Pipeline failure rate > 10%
  - Queue depth > 100
  - Worker error rate > 5%
  - DB connection pool exhausted
  - Cache hit rate < 50%

Tools:
  - Prometheus (metrics collection)
  - Grafana (dashboards)
  - Sentry (error tracking)
  - Structured logging (JSON)
```


## ============================================================
## 7. SECURITY
## ============================================================

```
Authentication:
  - JWT tokens (1h expiry)
  - Refresh tokens (7d expiry)
  - Role-based access control

Authorization:
  - INSPECTOR: read/write own cases
  - ADMIN: read/write all cases, manage users
  - AUDIT_LOGGER: read-only, audit logs

Data Protection:
  - Database: encrypted at rest, SSL in transit
  - R2: private bucket, presigned URLs
  - Secrets: .env file, never in code
  - PII: field-level encryption (national_id_number)

Audit:
  - All API calls logged
  - All pipeline runs logged
  - All data changes logged
  - Retention: 7 years
```


## ============================================================
## 8. DEPLOYMENT
## ============================================================

```
Environment: Docker + Docker Compose (development)
             Kubernetes (production)

Services:
  1. frontend (React) → Port 3000
  2. backend (FastAPI) → Port 8000
  3. worker (Python) → Port 8001
  4. postgresql → Port 5432
  5. redis → Port 6379
  6. r2 (external)

Docker Compose:
  - All services in one network
  - Persistent volumes for DB + Redis
  - Environment variables from .env
```


## ============================================================
## 9. IMPLEMENTATION ORDER
## ============================================================

```
Phase 1: Database (Week 1-2)
  - PostgreSQL setup
  - Prisma schema
  - Initial migration
  - Connection pool

Phase 2: Storage (Week 2-3)
  - R2 bucket
  - Upload/download
  - File management

Phase 3: Pipeline Queue (Week 3-4)
  - Redis + BullMQ
  - Worker setup
  - Stage isolation
  - Error handling

Phase 4: Pipeline Stages (Week 4-6)
  - Ingestion worker
  - Extraction worker
  - Resolution worker
  - Temporal worker
  - Graph worker

Phase 5: Backend Stub (Week 6-7)
  - FastAPI server
  - REST endpoints
  - Auth layer
  - Query layer

Phase 6: Frontend Stub (Week 7-8)
  - React app
  - Dashboard
  - Case detail
  - Graph view

Phase 7: Integration (Week 8-9)
  - Frontend ↔ Backend
  - Backend ↔ Pipeline
  - Pipeline ↔ DB
  - End-to-end testing

Phase 9: Monitoring (Week 9-10)
  - Prometheus metrics
  - Grafana dashboards
  - Alerting
  - Logging
```


## ============================================================
## 10. SUMMARY
## ============================================================

```
Architecture:
  Frontend (React) → Backend (FastAPI) → Queue (Redis) → Workers → DB (PostgreSQL)

Pipeline Stages (8 total):
  Stages 1-6: Per-Case (Ingestion → Extraction → Resolution → Temporal → Graph → Case Final)
  Stage 11:    Global Entity Push (cross-case dedup via GlobalEntityIndex)
  Stage 12:    Scoped Analytics (cross-case graph merge via JurisdictionNode hierarchy)

Key Improvements:
  1. Queue-based async processing
  2. Worker pool with horizontal scaling
  3. Stage isolation (each stage独立)
  4. Caching layer (Redis)
  5. Read replicas (PostgreSQL)
  6. Monitoring + alerting
  7. Error recovery + retry
  8. Incremental processing
  9. Multi-case architecture (JurisdictionNode replaces PoliceStation)
  10. Global Entity Identity Index for cross-case dedup
  11. Scoped Analytical Graphs (case + jurisdiction scope)

Scalability:
  - 2-8 workers (horizontal)
  - 2-4 backend instances (horizontal)
  - 1-2 read replicas (horizontal)
  - Redis cluster (horizontal)

Implementation:
  ~10 weeks for full stack
  ~6 weeks for pipeline + DB only
```
