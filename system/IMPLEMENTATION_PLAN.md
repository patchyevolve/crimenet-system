# ============================================================
# IMPLEMENTATION PLAN
# ============================================================


## ============================================================
## 1. WHAT WE'RE BUILDING
## ============================================================

```
Our system has 3 layers:

┌─────────────────────────────────────────┐
│  LAYER 1: PIPELINE (exists)             │
│  Stages 1-5: Ingestion → Graph         │
│  Currently outputs JSON files           │
└─────────────────┬───────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────┐
│  LAYER 2: DATABASE (to build)           │
│  PostgreSQL + Prisma                    │
│  Stores all pipeline output             │
│  Stores entities, graph, audit          │
└─────────────────┬───────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────┐
│  LAYER 3: API/BRIDGE (stub)             │
│  Connects to teammate's backend         │
│  Syncs entities, graph, confidence      │
└─────────────────────────────────────────┘

Face Recognition = separate feature (later)
```


## ============================================================
## 2. PHASE 1: DATABASE (Week 1-2)
## ============================================================

### 2.1 Setup

```
Install:
  - PostgreSQL 15
  - pgvector extension
  - Prisma ORM

Create:
  - prisma/schema.prisma (translated from DB_SCHEMA.md)
  - Initial migration
  - Database connection pool
  - Seed script for reference data
```

### 2.2 Schema Translation

Convert our DBML to Prisma:

```
Tables to create:
  Core:
    - PipelineRun
    - Case
    - PoliceStation
    - User
    - Location

  Ingestion (Stage 1):
    - IngestedFile
    - DataQualityScore
    - AdversarialCheck
    - EvidenceIntegrity
    - DependencyGroup

  Extraction (Stage 2):
    - ExtractedEntity
    - ExtractedRelation
    - ExtractionLog

  Resolution (Stage 3):
    - ResolvedEntity
    - EntityResolutionCandidate
    - Contradiction
    - ResolutionHistory

  Temporal (Stage 4):
    - TemporalInfo
    - SpatialInfo
    - CoverageInterval

  Graph (Stage 5):
    - GraphNode
    - GraphEdge
    - MissingEdge
    - RejectedEdge
    - AdversarialEdgeScore
    - EventCluster

  Knowledge Graph Entities:
    - Person
    - PersonAlias
    - Phone
    - Vehicle
    - BankAccount
    - CaseEntityLink

  Evidence Management:
    - Evidence
    - EvidenceFile
    - OcrResult
    - AudioTranscript

  Suspect Lifecycle:
    - Suspect

  Audit:
    - AuditLog

Total: 35+ tables
```

### 2.3 Files to Create

```
prisma/
├── schema.prisma
├── migrations/
│   └── 20260831_init/
│       └── migration.sql
└── seed.ts

src/db/
├── connection.ts        ← Prisma client singleton
├── pipeline-run.ts      ← CRUD
├── ingested-file.ts
├── extracted-entity.ts
├── extracted-relation.ts
├── resolved-entity.ts
├── contradiction.ts
├── graph-node.ts
├── graph-edge.ts
├── person.ts
├── phone.ts
├── vehicle.ts
├── bank-account.ts
├── location.ts
├── suspect.ts
└── case-entity-link.ts
```


## ============================================================
## 3. PHASE 2: STORAGE (Week 2-3)
## ============================================================

### 3.1 Cloudflare R2

```
Setup:
  - Create R2 bucket
  - Generate API tokens
  - Configure S3-compatible client

Bucket structure:
  evidence/
  ├── {case_id}/
  │   ├── {file_id}/
  │   │   ├── original.{ext}
  │   │   ├── metadata.json
  │   │   └── thumbnail.jpg (if image)
  │   └── ...
  └── ...

  faces/
  ├── {embedding_id}/
  │   ├── crop.jpg
  │   └── embedding.npy
  └── ...
```

### 3.2 Files to Create

```
src/storage/
├── r2-client.ts         ← S3-compatible client
├── upload.ts            ← upload file to R2
├── download.ts          ← download from R2
└── presigned.ts         ← generate presigned URLs
```


## ============================================================
## 4. PHASE 3: PIPELINE → DATABASE (Week 3-4)
## ============================================================

### 4.1 Update Pipeline Stages

```
Current: Pipeline outputs JSON files
New: Pipeline writes to database

Stage 1: Ingestion
  - Create PipelineRun record
  - Create IngestedFile records
  - Upload files to R2
  - Create DataQualityScore records
  - Create AdversarialCheck records
  - Create EvidenceIntegrity records

Stage 2: Extraction
  - Create ExtractedEntity records
  - Create ExtractedRelation records
  - Create ExtractionLog records

Stage 3: Resolution
  - Create ResolvedEntity records
  - Create EntityResolutionCandidate records
  - Create Contradiction records
  - Create ResolutionHistory records

Stage 4: Temporal
  - Create TemporalInfo records
  - Create SpatialInfo records
  - Create CoverageInterval records

Stage 5: Graph
  - Create GraphNode records
  - Create GraphEdge records
  - Create MissingEdge/RejectedEdge records
  - Create AdversarialEdgeScore records
  - Create EventCluster records
  - Create Knowledge Graph Entity records (Person, Phone, etc.)
```

### 4.2 Files to Update

```
src/pipeline.py              ← add database writes
src/extraction/engine.py     ← store to DB
src/resolution/engine.py     ← store to DB
src/graph/builder.py         ← store to DB
src/temporal/engine.py       ← store to DB
```


## ============================================================
## 5. PHASE 4: API/BRIDGE STUB (Week 4-5)
## ============================================================

### 5.1 What This Is

```
A thin API layer that:
  1. Exposes our data via REST endpoints
  2. Can sync to teammate's backend when ready
  3. Is just a stub for now (endpoints exist, logic placeholder)
```

### 5.2 Endpoints

```
GET  /api/cases                    ← list cases
GET  /api/cases/:id                ← get case details
GET  /api/cases/:id/entities       ← get entities for case
GET  /api/cases/:id/graph          ← get knowledge graph
GET  /api/cases/:id/contradictions ← get contradictions
GET  /api/cases/:id/timeline       ← get temporal timeline
GET  /api/cases/:id/evidence       ← get evidence files
POST /api/pipeline/run             ← trigger pipeline
GET  /api/pipeline/runs/:id        ← get pipeline run status
POST /api/sync/to-backend          ← sync to teammate's backend (stub)
```

### 5.3 Files to Create

```
src/api/
├── server.ts            ← Express/Fastify server
├── routes/
│   ├── cases.ts
│   ├── pipeline.ts
│   └── sync.ts
├── middleware/
│   └── auth.ts
└── stubs/
    └── backend-sync.ts  ← placeholder for teammate's API
```


## ============================================================
## 6. PHASE 5: TESTING + POLISH (Week 5-6)
## ============================================================

```
Tasks:
  - Unit tests for DB operations
  - Integration tests for pipeline → DB
  - API endpoint tests
  - Seed demo data
  - Update run.py to use database
  - Documentation
```


## ============================================================
## 7. FILE STRUCTURE
## ============================================================

```
system/
├── prisma/
│   ├── schema.prisma
│   ├── migrations/
│   └── seed.ts
├── src/
│   ├── db/
│   │   ├── connection.ts
│   │   ├── pipeline-run.ts
│   │   ├── ingested-file.ts
│   │   ├── extracted-entity.ts
│   │   ├── extracted-relation.ts
│   │   ├── resolved-entity.ts
│   │   ├── contradiction.ts
│   │   ├── graph-node.ts
│   │   ├── graph-edge.ts
│   │   ├── person.ts
│   │   ├── phone.ts
│   │   ├── vehicle.ts
│   │   ├── bank-account.ts
│   │   ├── location.ts
│   │   ├── suspect.ts
│   │   └── case-entity-link.ts
│   ├── storage/
│   │   ├── r2-client.ts
│   │   ├── upload.ts
│   │   ├── download.ts
│   │   └── presigned.ts
│   ├── api/
│   │   ├── server.ts
│   │   ├── routes/
│   │   │   ├── cases.ts
│   │   │   ├── pipeline.ts
│   │   │   └── sync.ts
│   │   └── stubs/
│   │       └── backend-sync.ts
│   ├── extraction/
│   │   └── engine.ts       ← update to write DB
│   ├── resolution/
│   │   └── engine.ts       ← update to write DB
│   ├── graph/
│   │   └── builder.ts      ← update to write DB
│   ├── temporal/
│   │   └── engine.ts       ← update to write DB
│   └── pipeline.py         ← update to use DB
├── db_schema/
│   ├── DB_SCHEMA.md
│   ├── DB_SCHEMA_DIAGRAM.md
│   └── DB_SCHEMA_DIAGRAM.svg
├── RESEARCH_FACE_RECOGNITION/  ← separate feature
├── SCHEMA_COMPARISON.md
├── INTEGRATION_BRIDGE.md
├── IMPLEMENTATION_PLAN.md
└── run.py
```


## ============================================================
## 8. DEPENDENCIES
## ============================================================

### Python (existing + new)
```
# Existing
python-dateutil
pandas
networkx

# New - Database
prisma
asyncpg

# New - Storage
boto3  # R2 is S3-compatible

# New - API
fastapi
uvicorn
```

### Node.js (optional, for Prisma CLI only)
```
prisma
@prisma/client
```


## ============================================================
## 9. TIMELINE
## ============================================================

```
Week 1-2:  Database setup + schema translation
Week 2-3:  R2 storage integration
Week 3-4:  Pipeline → Database integration
Week 4-5:  API/bridge stub
Week 5-6:  Testing + polish

Total: ~6 weeks
```
