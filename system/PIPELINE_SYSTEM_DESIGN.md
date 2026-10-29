# ============================================================
# PIPELINE SYSTEM DESIGN
# ============================================================
#
# This document defines the complete architecture of the
# intelligence pipeline — how data flows, how stages connect,
# how storage works, and how everything fits together.
#
# Architecture: 8-stage pipeline with multi-case support
# ============================================================


## ============================================================
## 1. SYSTEM OVERVIEW
## ============================================================

### 1.1 What the Pipeline Does

```
Input:  Raw evidence files (PDFs, CSVs, images, docs) scoped to a Case
Output: Local knowledge graph per Case + Global Entity Identity Index

Pipeline Stages:
  Stage 1: INGESTION           → File intake, integrity, quality (per-Case)
  Stage 2: EXTRACTION          → NER, relations, confidence scoring (per-Case)
  Stage 3: ENTITY RESOLUTION   → Entity merging, deduplication (per-Case, scoped to case_id)
  Stage 4: TEMPORAL ENRICHMENT → Timeline, contradictions (per-Case)
  Stage 5: GRAPH CONSTRUCTION  → Local knowledge graph per Case
  Stage 6: ANALYTICS ENGINE    → Analytics on local graph (per-Case)
  Stage 11: GLOBAL ENTITY PUSH  → Identity signals to Global Entity Identity Index (cross-case)
  Stage 12: SCOPED ANALYTICS    → Merged cross-case analytics (on-demand)
```

### 1.2 Architecture Diagram

```
┌──────────────────────────────────────────────────────────────────────────┐
│                           ENTRY POINT                                    │
│                                                                          │
│  run.py --case-id <id> --input ../demo_data --output output --use-llm  │
│                                                                          │
└─────────────────────────────┬────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                      PIPELINE ORCHESTRATOR                               │
│                                                                          │
│  src/pipeline.py                                                         │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │ 1. Create PipelineRun record (linked to case_id)                │   │
│  │ 2. Call Stage 1 → IngestedFiles                                 │   │
│  │ 3. Call Stage 2 → ExtractedEntities + ExtractedRelations        │   │
│  │ 4. Call Stage 3 → ResolvedEntities (scoped to case_id)          │   │
│  │ 5. Call Stage 4 → TemporalInfo + Contradictions                 │   │
│  │ 6. Call Stage 5 → Local graph (GraphNode + GraphEdge)           │   │
│  │ 7. Call Stage 6 → Case-level analytics                          │   │
│  │ 8. Call Stage 11 → Push identity signals to global index         │   │
│  │ 9. Update PipelineRun status = COMPLETED                        │   │
│  │ 10. Return summary                                              │   │
│  └──────────────────────────────────────────────────────────────────┘   │
└────┬───────┬───────┬───────┬───────┬───────┬───────┬───────┬────────────┘
     │       │       │       │       │       │       │       │
     ▼       ▼       ▼       ▼       ▼       ▼       ▼       ▼
┌────────┐┌────────┐┌────────┐┌────────┐┌────────┐┌────────┐┌────────┐┌────────┐
│  DB    ││  R2    ││ Stage1 ││ Stage2 ││ Stage3 ││ Stage4 ││ Stage5 ││ Stage6 │
│ Client ││ Client ││ Engine ││ Engine ││ Engine ││ Engine ││ Engine ││ Engine │
└───┬────┘└───┬────┘└───┬────┘└───┬────┘└───┬────┘└───┬────┘└───┬────┘└───┬────┘
    │         │         │         │         │         │         │         │
    ▼         ▼         ▼         ▼         ▼         ▼         ▼         ▼
┌────────┐┌────────┐┌────────┐┌────────┐┌────────┐┌────────┐┌────────┐┌────────┐
│Postgres││Cloud-  ││Ingested││Extracted││Resolved││Temporal││Graph   ││Analytics│
│   +    ││flare   ││File    ││Entity   ││Entity  ││Info    ││Node    ││Result  │
│pgvector││  R2    ││Record  ││Record   ││Record  ││Record  ││Record  ││Record  │
└────────┘└────────┘└────────┘└────────┘└────────┘└────────┘└────────┘└────────┘

     ┌────────────────────────────────────────────────────────────────┐
     │                     CROSS-CASE LAYER                           │
     │                                                                │
     │  Stage 11: GLOBAL ENTITY PUSH                                   │
     │  ┌──────────────────────────────────────────────────────────┐  │
     │  │ Scans local ResolvedEntities → extracts identity        │  │
     │  │ signals (phone, name, DOB, etc.) → upserts into         │  │
     │  │ GlobalEntityIdentityIndex (cross-case deduplication)     │  │
     │  └──────────────────────────────────────────────────────────┘  │
     │                                                                │
     │  Stage 12: SCOPED ANALYTICS (on-demand)                         │
     │  ┌──────────────────────────────────────────────────────────┐  │
     │  │ Accepts scope (set of case_ids) → merges local graphs   │  │
     │  │ using GlobalEntityIdentityIndex links → produces         │  │
     │  │ ScopedAnalyticalGraph (not persisted, computed fresh)    │  │
     │  └──────────────────────────────────────────────────────────┘  │
     │                                                                │
     │  ┌──────────────────────────────────────────────────────────┐  │
     │  │              GlobalEntityIdentityIndex                    │  │
     │  │  Maps canonical identity → list of (case_id, entity_id)  │  │
     │  │  Updated by Stage 11, read by Stage 12                    │  │
     │  └──────────────────────────────────────────────────────────┘  │
     └────────────────────────────────────────────────────────────────┘
```


## ============================================================
## 2. COMPONENT DESIGN
## ============================================================

### 2.1 Entry Point: run.py

```
Responsibilities:
  - Parse CLI arguments
  - Load configuration
  - Initialize pipeline
  - Execute pipeline for a single Case
  - Print summary

Flow:
  1. Parse args: --case-id, --input, --output, --use-llm
  2. Load config from system/config/
  3. Validate case_id exists and has exactly one primary FIR
  4. Create Pipeline instance
  5. Call pipeline.execute(case_id, input_folder, output_folder)
  6. Print results
```

### 2.2 Pipeline Orchestrator: src/pipeline.py

```
Responsibilities:
  - Create PipelineRun record (linked to case_id)
  - Call each stage in sequence (Stages 1-6 local; Stages 11-12 cross-case, designed)
  - Pass data between stages via pipeline context
  - Handle errors with per-stage rollback
  - Update run status
  - Return summary

State Management:
  - self.run_id: PipelineRun ID (created at start)
  - self.case_id: Case ID (required parameter)
  - self.db: Database connection
  - self.r2: R2 storage client

Stage Execution:
  Each stage receives:
    - Input data from previous stage
    - case_id for scoping
    - Database connection
    - R2 client
    - Configuration

  Each stage returns:
    - Output data for next stage
    - Statistics (counts, timing)
    - Any errors encountered

Multi-Case Support:
  Stages 1-6 operate on a single Case (scoped by case_id).
  Stage 11 operates cross-case (pushes to global index).
  Stage 12 operates on-demand across multiple Cases.
```

### 2.3 Stage 1: Ingestion Engine (per-Case)

```
Input:  Folder path with evidence files + case_id
Output: List of IngestedFile records (linked to case_id)

Flow:
  1. Validate case_id exists
  2. Scan folder for files
  3. For each file:
     a. Read file
     b. Compute hash (SHA-256)
     c. Detect file type
     d. Classify source type
     e. Extract metadata
     f. Check for duplicates within this Case
     g. Upload original to R2 (under case_id path)
     h. Create IngestedFile record (case_id FK)
     i. Create EvidenceIntegrity record
     j. Assess data quality → DataQualityScore record
     k. Check for adversarial signals → AdversarialCheck record
  4. Group files by sequence number
  5. Return list of IngestedFile IDs

Dependencies:
  - Database: IngestedFile, EvidenceIntegrity, DataQualityScore, AdversarialCheck
  - Storage: R2 (upload original files under case_id path)
```

### 2.4 Stage 2: Extraction Engine (per-Case)

```
Input:  List of IngestedFile records (all scoped to case_id)
Output: List of ExtractedEntity + ExtractedRelation records

Flow:
  1. For each IngestedFile:
     a. Download content from R2 (or read from disk)
     b. Parse based on file type:
        - CSV/Excel → tabular extraction
        - PDF → text extraction + OCR
        - DOCX → text extraction
        - Image → OCR + face detection (if enabled)
     c. Run NER on text content
     d. Extract entities (PERSON, PHONE, LOCATION, etc.)
     e. Extract relations (CALLED, TRANSFERRED_TO, etc.)
     f. Score confidence (multi-factor)
     g. Create ExtractedEntity records (case_id FK)
     h. Create ExtractedRelation records (case_id FK)
     i. Create ExtractionLog record
  2. Return lists of entity/relation IDs

Dependencies:
  - Database: ExtractedEntity, ExtractedRelation, ExtractionLog
  - Models: NER model (code-based or LLM)
  - Storage: R2 (download files)
```

### 2.5 Stage 3: Entity Resolution Engine (per-Case, scoped to case_id)

```
Input:  Lists of ExtractedEntity + ExtractedRelation records (all for this case_id)
Output: List of ResolvedEntity records (scoped to case_id)

Flow:
  1. Load all entities for this case_id only
  2. Group by entity type
  3. For each entity type:
     a. Find potential merges:
        - Exact match (same phone number)
        - Fuzzy match (similar names)
        - Phonetic match (similar sounding)
        - Disambiguation (same name, different person)
     b. Score merge confidence
     c. For high-confidence merges:
        - Auto-merge
        - Create ResolvedEntity record
     d. For medium-confidence merges:
        - Create EntityResolutionCandidate (PENDING)
     e. For low-confidence:
        - Reject merge
  4. Detect contradictions:
     - Same entity, different attributes
     - Same entity, different locations at same time
  5. Create Contradiction records
  6. Create ResolutionHistory records
  7. Return ResolvedEntity IDs

Scope Constraint:
  Resolution is strictly scoped to case_id. Entities from different
  Cases are NOT merged at this stage. Cross-case merging happens in
  Stage 11 via the Global Entity Identity Index.

Dependencies:
  - Database: ResolvedEntity, EntityResolutionCandidate, Contradiction, ResolutionHistory
  - Algorithms: Fuzzy matching, phonetic matching
```

### 2.6 Stage 4: Temporal Enrichment Engine (per-Case)

```
Input:  Lists of ExtractedEntity + ExtractedRelation + ResolvedEntity records (case_id scoped)
Output: TemporalInfo, SpatialInfo, CoverageInterval, Contradiction records

Flow:
  1. Extract timestamps from entities/relations
  2. Build timeline for each entity
  3. Detect temporal contradictions:
     - Same person at different locations at same time
     - Sequence violations (event B before event A)
  4. Compute coverage intervals:
     - When entity first appears
     - When entity last appears
     - Gaps in coverage
  5. Create TemporalInfo records
  6. Create SpatialInfo records
  7. Create CoverageInterval records
  8. Update Contradiction records (temporal ones)
  9. Return temporal data

Dependencies:
  - Database: TemporalInfo, SpatialInfo, CoverageInterval, Contradiction
  - Algorithms: Timeline construction, contradiction detection
```

### 2.7 Stage 5: Graph Construction Engine (local graph per Case)

```
Input:  All previous stage outputs (case_id scoped)
Output: Local knowledge graph: GraphNode + GraphEdge records

Flow:
  1. Create GraphNode for each ResolvedEntity (case_id scoped)
  2. Create GraphNode for each standalone entity (not resolved)
  3. Create GraphEdge for each ExtractedRelation:
     a. Map relation type to edge type
     b. Score edge confidence (4-factor model)
     c. Check for adversarial signals
     d. Track provenance
  4. Create Knowledge Graph Entity records:
     - Person (from ResolvedEntity PERSON)
     - Phone (from ResolvedEntity PHONE)
     - Vehicle (from ResolvedEntity VEHICLE)
     - BankAccount (from ResolvedEntity ACCOUNT)
     - Location (from ResolvedEntity LOCATION)
  5. Create CaseEntityLink records (SUSPECT_OF, VICTIM_OF, etc.)
  6. Detect missing edges (low confidence)
  7. Detect rejected edges (below threshold)
  8. Create AdversarialEdgeScore records
  9. Create EventCluster records
  10. Graph is stored permanently — local graph for this Case
  11. Return graph summary

Storage:
  The local graph is built once per Case and stored permanently.
  It is the canonical representation of all entities and relations
  for that Case. Stages 1-6 may be re-run, but the graph persists
  until explicitly deleted.

Dependencies:
  - Database: GraphNode, GraphEdge, MissingEdge, RejectedEdge, AdversarialEdgeScore, EventCluster
  - Database: Person, Phone, Vehicle, BankAccount, Location, CaseEntityLink
  - Algorithms: 4-factor confidence scoring, adversarial detection
```

### 2.8 Stage 6: Analytics Engine (per-Case, on local graph)

```
Input:  Local graph from Stage 5 (GraphNode + GraphEdge)
Output: Case-level analytics results

Flow:
  1. Load local graph for this case_id
  2. Compute network metrics:
     - Centrality scores for each entity node
     - Community detection (if clusters exist)
     - Shortest paths between key entities
     - Hub identification (high-connectivity nodes)
  3. Compute communication patterns:
     - Most active communicators
     - Time-of-day patterns
     - Communication density over time
  4. Compute geographic patterns:
     - Movement trajectories
     - Hotspot locations
     - Geographic clustering
  5. Store analytics results linked to case_id
  6. Return analytics summary

Dependencies:
  - Database: AnalyticsResult, NetworkMetric, CommunicationPattern, GeographicPattern
  - Algorithms: Graph centrality, community detection, pathfinding
```

### 2.9 Stage 11: Global Entity Push (cross-case, designed not implemented)

```
Input:  ResolvedEntities from this Case (case_id scoped)
Output: Updated Global Entity Identity Index

Purpose:
  Links same-entity across different Cases by pushing identity
  signals from each Case's local Resolution into a shared global index.

Flow:
  1. Load all ResolvedEntities for this case_id
  2. For each ResolvedEntity, extract identity signals:
     - Phone numbers (canonicalized)
     - Full name + aliases
     - Date of birth / age
     - National ID number (if available)
     - Physical descriptors
     - Known associates (entity links)
  3. Query Global Entity Identity Index for existing matches:
     - Exact phone match → link to existing canonical identity
     - Fuzzy name + DOB match → candidate link
     - Known associate overlap → strengthen existing link
  4. For new entities (no match):
     - Create new canonical identity in Global Entity Identity Index
  5. For matched entities:
     - Update global index with new case_id + entity_id mapping
     - Update confidence score of the link
     - Record provenance (which Case contributed which signal)
  6. Return push statistics

Global Entity Identity Index Structure:
  GlobalEntityIdentityIndex {
    canonical_id: string          # unique global identity
    identity_signals: {
      phones: string[]            # canonicalized phone numbers
      names: string[]             # all known names/aliases
      date_of_birth: string?      # if available
      national_id: string?        # if available
    }
    case_links: [{
      case_id: string             # which Case
      entity_id: string           # local entity ID in that Case
      resolved_entity_id: string  # local ResolvedEntity ID
      confidence: float           # link confidence
      contributed_signals: string[] # which signals this Case provided
    }]
    created_at: datetime
    updated_at: datetime
  }

Dependencies:
  - Database: GlobalEntityIdentityIndex, EntityIdentityLink
  - Algorithms: Identity matching, confidence scoring
```

### 2.10 Stage 12: Scoped Analytics (on-demand, designed not implemented)

```
Input:  Scope definition (set of case_ids or jurisdiction filter)
Output: ScopedAnalyticalGraph (computed, not stored)

Purpose:
  Merges local graphs from multiple Cases using global identity links
  to produce a cross-case analytical view. Computed on-demand and not
  persisted — each request recomputes from the source data.

Flow:
  1. Accept scope definition:
     - Explicit: list of case_ids
     - Implicit: jurisdiction-based filter (e.g., all Cases under
       a given JurisdictionNode)
  2. Load local graphs for all in-scope Cases
  3. Load Global Entity Identity Index entries for all entities
     in those graphs
  4. Merge graphs:
     a. Identify shared entities (same canonical_id across Cases)
     b. Create merged nodes (canonical identity → combined attributes)
     c. Union all edges (preserving provenance per Case)
     d. Resolve cross-case edges (entity A in Case 1 connected to
        entity B in Case 2 via shared phone number)
  5. Compute cross-case analytics and ML:
     - Cross-case network centrality
     - Inter-case communication patterns
     - Shared entity influence scores
     - Jurisdiction-level hotspot analysis
     - Community detection on merged graph (same as Stage 6)
     - GNN-based anomaly detection on cross-case patterns
     - Bayesian Network inference across Cases
  6. Return ScopedAnalyticalGraph (in-memory, not persisted)

Storage Policy:
  ScopedAnalyticalGraphs are NOT stored. They are computed on-demand
  from:
    - Local graphs (stored permanently, per Case)
    - Global Entity Identity Index (updated by Stage 11)
  This ensures the analytical view is always fresh and reflects the
  latest data from all contributing Cases.
  ML results on scoped graphs carry source_case_ids for traceability.

Dependencies:
  - Read: GraphNode, GraphEdge (from local graphs, multiple Cases)
  - Read: GlobalEntityIdentityIndex, EntityIdentityLink
  - Algorithms: Graph merging, cross-case centrality, influence scoring,
    GNN, Bayesian Networks, community detection (same as Stage 6)
```


## ============================================================
## 3. MULTI-CASE ARCHITECTURE
## ============================================================

### 3.1 Case Entity Model

```
Case {
  id: string (UUID)
  title: string
  description: string?
  jurisdiction_id: string → JurisdictionNode
  status: DRAFT | ACTIVE | CLOSED | ARCHIVED
  primary_fir_id: string → FIR (exactly one, invariant)
  created_at: datetime
  updated_at: datetime
}

FIR {
  id: string (UUID)
  fir_number: string
  jurisdiction_id: string → JurisdictionNode
  case_id: string → Case (unique — 1:1 invariant)
  filing_date: date
  investigation_status: FIRInvestigationStatus
  legal_disposition: FIRLegalDisposition
  record_status: FIRRecordStatus
  created_at: datetime
  updated_at: datetime
}

FIRInvestigationStatus: PENDING | UNDER_INVESTIGATION | CHARGE_SHEET_FILED | CLOSED
FIRLegalDisposition: PENDING | DISMISSED | CHARGESHEET | CONVICTED | ACQUITTED
FIRRecordStatus: ACTIVE | SEALED | EXPUNGED

Invariant: Case ↔ FIR is 1:1.
  - FIR.case_id is unique (each Case has exactly one FIR)
  - FIR lifecycle is independent from CaseRelationship, Workspace, AnalysisRun
  - Lifecycle changes tracked in FIRLifecycleHistory
```

### 3.2 FIR Number Uniqueness

```
FIR {
  id: string (UUID)
  fir_number: string
  jurisdiction_id: string → JurisdictionNode
  case_id: string → Case
  filing_date: date
  ...
}

Constraint: fir_number is unique WITHIN a JurisdictionNode.
  - "FIR/2024/0456" in Jurisdiction "Delhi Police" is distinct from
    "FIR/2024/0456" in Jurisdiction "Mumbai Police"
  - Enforced by unique constraint on (fir_number, jurisdiction_id)
  - Cross-jurisdiction FIR numbers may collide — this is expected
```

### 3.3 Data Isolation Model

```
Each Case maintains its own local graph. Entities and relations
are NOT shared across Cases at the pipeline level.

  Case A:
    Local Graph → {EntA1, EntA2, EntA3} + {RelA1, RelA2}

  Case B:
    Local Graph → {EntB1, EntB2, EntB3} + {RelB1, RelB2}

  Shared Entities:
    EntA1 and EntB1 might be the same person (e.g., same phone number).
    This link is NOT detected in Stages 1-6.
    It IS detected in Stage 11 (Global Entity Push) and exposed in
    Stage 12 (Scoped Analytics) via the Global Entity Identity Index.
```

### 3.4 Pipeline Execution Modes

```
Mode 1: Single-Case Pipeline
  - Processes one Case through Stages 1-6
  - Stage 11 pushes to global index
  - Typical: investigator uploads evidence for one Case

Mode 2: Multi-Case Batch
  - Processes multiple Cases sequentially
  - Each Case runs Stages 1-7 independently
  - Useful for backfilling or bulk ingestion

Mode 3: Cross-Case Analysis
  - Stage 12 on-demand
  - Requires Stages 1-7 to have completed for in-scope Cases
  - Typical: analyst requests cross-case network view
```

### 3.5 Designed Entities (Not Yet Implemented)

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


## ============================================================
## 4. JURISDICTION HIERARCHY
## ============================================================

### 4.1 JurisdictionNode Model

```
JurisdictionNode replaces PoliceStation. It supports a configurable
hierarchy of geographic/administrative units.

JurisdictionNode {
  id: string (UUID)
  name: string
  code: string                    # e.g., "DL-04", "MH-12"
  type: JURISDICTION_TYPE        # NATIONAL | STATE | DISTRICT | CITY | STATION
  parent_id: string? → JurisdictionNode   # nullable for root
  metadata: JSON?                # additional jurisdiction-specific data
  created_at: datetime
  updated_at: datetime
}

JURISDICTION_TYPE hierarchy:
  NATIONAL
    └── STATE
          └── DISTRICT
                └── CITY
                      └── STATION

Example:
  NATIONAL: "India"
    STATE: "Delhi"
      DISTRICT: "Central Delhi"
        STATION: "Karol Bagh PS"
        STATION: "Daryaganj PS"
      DISTRICT: "South Delhi"
        STATION: "Saket PS"
    STATE: "Maharashtra"
      DISTRICT: "Mumbai City"
        STATION: "Colaba PS"
```

### 4.2 Jurisdiction Scoping Rules

```
1. Case is always linked to exactly one JurisdictionNode (the
   investigating station's jurisdiction).

2. FIR numbers are unique within their JurisdictionNode.

3. Evidence files inherit the Case's jurisdiction for storage paths.

4. Cross-jurisdiction Cases (e.g., crimes spanning Delhi and Mumbai)
   are modeled as a single Case linked to the primary jurisdiction,
   with FIRs from other jurisdictions linked as secondary FIRs.

5. Stage 12 (Scoped Analytics) can filter by jurisdiction hierarchy:
   - "All Cases under Delhi" → recursively finds all child jurisdictions
   - "All Cases in Central Delhi" → finds only Cases in that district
```


## ============================================================
## 5. DATABASE LAYER
## ============================================================

### 5.1 Connection Management

```
src/db/connection.ts

Responsibilities:
  - Initialize Prisma client
  - Connection pooling
  - Transaction management
  - Error handling

Usage:
  import { db } from './db/connection';

  // Single query
  const person = await db.person.findUnique({ where: { id: '...' } });

  // Transaction
  await db.$transaction(async (tx) => {
    await tx.person.create({ data: { ... } });
    await tx.phone.create({ data: { ... } });
  });
```

### 5.2 Data Access Patterns

```
Pattern 1: Stage writes to DB
  Stage engine → db.operation.create(data) → PostgreSQL

Pattern 2: Stage reads from DB (case-scoped)
  Stage engine → db.operation.findMany({ where: { case_id } }) → data

Pattern 3: Pipeline passes data between stages
  Stage 1 output → pipeline context → Stage 2 input

  Pipeline context:
    case_id: string
    run_id: string
    ingested_files: string[]      (IDs, all for this case_id)
    extracted_entities: string[]  (IDs, all for this case_id)
    extracted_relations: string[] (IDs, all for this case_id)
    resolved_entities: string[]   (IDs, all for this case_id)

Pattern 4: Cross-case reads (Stage 11, Stage 12)
  Stage engine → db.operation.findMany({
    where: { case_id: { in: [...caseIds] } }
  }) → data
```

### 5.3 Audit Trail

```
Every write operation logs:
  - Who (user_id or pipeline system)
  - What (operation type)
  - When (timestamp)
  - Where (resource ID + case_id)
  - How (method: code, llm, hybrid)
  - Why (reason for operation)

Stored in:
  - PipelineRun (run-level audit, linked to case_id)
  - ExtractionLog (file-level audit)
  - ResolutionHistory (merge audit)
  - AuditLog (user action audit)
```


## ============================================================
## 6. STORAGE LAYER
## ============================================================

### 6.1 R2 Bucket Structure

```
r2://criminal-evidence/
├── cases/
│   └── {case_id}/
│       └── evidence/
│           └── {file_id}/
│               ├── original/
│               │   └── {filename}.{ext}
│               ├── processed/
│               │   ├── extracted_text.txt
│               │   └── metadata.json
│               └── thumbnails/
│                   └── thumb.jpg
│
├── faces/
│   └── {embedding_id}/
│       ├── crop.jpg
│       └── embedding.npy
│
└── exports/
    └── {run_id}/
        ├── extraction_output.json
        ├── graph_nodes.json
        └── graph_edges.json
```

### 6.2 File Lifecycle

```
1. Upload:
   Investigator uploads file → Backend stores in R2 (under case_id) → Pipeline reads from R2

2. Processing:
   Pipeline downloads from R2 → Processes locally → Stores results to DB

3. Output:
   Pipeline exports to JSON → Uploads to R2 → Available for download

4. Retention:
   Original files: retained forever (evidence)
   Processed files: retained for 7 years
   Temporary files: deleted after processing
```


## ============================================================
## 7. CONFIGURATION
## ============================================================

### 7.1 Config Files

```
system/config/
├── pipeline.yaml          ← pipeline settings
├── extraction.yaml        ← NER extraction settings
├── resolution.yaml        ← entity resolution settings
├── temporal.yaml          ← temporal enrichment settings
├── graph.yaml             ← graph building settings
├── analytics.yaml         ← analytics engine settings
├── global-push.yaml       ← Stage 11 global entity push settings
├── scoped-analytics.yaml  ← Stage 12 scoped analytics settings
├── jurisdiction.yaml      ← jurisdiction hierarchy configuration
├── source-types.yaml      ← source type definitions
└── adversarial.yaml       ← adversarial detection settings
```

### 7.2 pipeline.yaml

```yaml
pipeline:
  version: "2.0"

  database:
    url: "postgresql://user:pass@localhost:5432/criminal_analysis"
    pool_size: 10

  storage:
    r2_bucket: "criminal-evidence"
    r2_endpoint: "https://<account_id>.r2.cloudflarestorage.com"
    r2_access_key: "${R2_ACCESS_KEY}"
    r2_secret_key: "${R2_SECRET_KEY}"

  stages:
    ingestion:
      skip_duplicates: true
      quality_threshold: 0.3
      adversarial_check: true

    extraction:
      use_llm: false
      llm_provider: "groq"
      confidence_threshold: 0.5
      ner_model: "code"  # or "llm"

    resolution:
      auto_merge_threshold: 0.85
      review_threshold: 0.60
      reject_threshold: 0.40
      use_llm: false

    temporal:
      contradiction_window_minutes: 30
      coverage_gap_threshold_hours: 24

    graph:
      edge_confidence_threshold: 0.5
      adversarial_score_threshold: 0.7
      semantic_classification: true
      persist_local_graph: true     # local graphs stored permanently

    analytics:
      compute_centrality: true
      compute_communities: true
      compute_paths: true

    global_push:
      identity_match_threshold: 0.80
      phone_exact_match: true
      name_fuzzy_threshold: 0.75
      max_signals_per_entity: 20

    scoped_analytics:
      max_cases_per_scope: 100
      recomputed_on_demand: true    # never persisted
```


## ============================================================
## 8. DATA FLOW EXAMPLE
## ============================================================

```
Input: 03_CDR_Suresh.csv (for Case "case_abc123")

Stage 1: INGESTION (per-Case)
  IngestedFile {
    id: "file_abc123"
    case_id: "case_abc123"
    file_name: "03_CDR_Suresh.csv"
    file_hash: "sha256:abc..."
    detected_type: "tabular"
    source_type: "CDR"
    document_type: "CDR"
  }

  DataQualityScore {
    quality_score: 0.85
    completeness_score: 0.90
  }

  AdversarialCheck {
    is_suspicious: false
    risk_level: "LOW"
  }

  R2: Upload original → r2://cases/case_abc123/evidence/file_abc123/original/03_CDR_Suresh.csv

Stage 2: EXTRACTION (per-Case)
  ExtractedEntity [
    { id: "ent_001", case_id: "case_abc123", type: "PHONE", name: "9876543211" },
    { id: "ent_002", case_id: "case_abc123", type: "PHONE", name: "9876543213" },
    { id: "ent_003", case_id: "case_abc123", type: "PERSON", name: "Suresh Kumar" },
    { id: "ent_004", case_id: "case_abc123", type: "LOCATION", name: "Karol Bagh Tower 1" }
  ]

  ExtractedRelation [
    { id: "rel_001", case_id: "case_abc123", source: "ent_001", target: "ent_002", type: "CALLED", count: 15 },
    { id: "rel_002", case_id: "case_abc123", source: "ent_001", target: "ent_004", type: "VISITED", count: 3 }
  ]

Stage 3: ENTITY RESOLUTION (per-Case, scoped to case_id)
  ResolvedEntity [
    { id: "res_001", case_id: "case_abc123", name: "Suresh Kumar", entities: ["ent_003"], merge_type: "SINGLE" },
    { id: "res_002", case_id: "case_abc123", name: "9876543211", entities: ["ent_001"], merge_type: "SINGLE" }
  ]

Stage 4: TEMPORAL ENRICHMENT (per-Case)
  TemporalInfo [
    { entity: "ent_001", case_id: "case_abc123", timestamp: "2024-03-10 14:25:00", source: "file_abc123" }
  ]

  SpatialInfo [
    { entity: "ent_001", case_id: "case_abc123", location: "Karol Bagh Tower 1", lat: 28.6519, lng: 77.1890 }
  ]

Stage 5: GRAPH CONSTRUCTION (local graph per Case)
  GraphNode [
    { id: "res_001", case_id: "case_abc123", type: "PERSON", name: "Suresh Kumar" },
    { id: "res_002", case_id: "case_abc123", type: "PHONE", name: "9876543211" }
  ]

  GraphEdge [
    { id: "edge_001", case_id: "case_abc123", source: "res_002", target: "res_001", type: "OWNS_PHONE", confidence: 0.95 }
  ]

  Local graph stored permanently for case_abc123.

Stage 6: ANALYTICS ENGINE (per-Case)
  AnalyticsResult {
    case_id: "case_abc123",
    network_metrics: { centrality: {...}, communities: [...] },
    communication_patterns: { peak_hours: [...], density: 0.72 },
    geographic_patterns: { hotspots: [...], trajectory: [...] }
  }

Stage 11: GLOBAL ENTITY PUSH (cross-case)
  // Suresh Kumar (phone 9876543211) might also appear in case_def456
  GlobalEntityIdentityIndex update:
    canonical_id: "global_ent_789"
    identity_signals: {
      phones: ["+919876543211"],
      names: ["Suresh Kumar"]
    }
    case_links: [
      { case_id: "case_abc123", entity_id: "res_001", confidence: 0.95 },
      { case_id: "case_def456", entity_id: "res_010", confidence: 0.88 }
    ]

Stage 12: SCOPED ANALYTICS (on-demand)
  // Analyst requests: "Show network across case_abc123 and case_def456"
  ScopedAnalyticalGraph {
    merged_nodes: [
      { canonical_id: "global_ent_789", name: "Suresh Kumar", source_cases: ["case_abc123", "case_def456"] }
    ],
    merged_edges: [
      // edges from both local graphs, connected via shared entity
    ],
    cross_case_analytics: {
      shared_entities: 3,
      inter_case_connections: 5,
      influence_scores: {...}
    }
  }
  // NOT persisted — computed fresh each time
```


## ============================================================
## 9. ERROR HANDLING
## ============================================================

### 9.1 Error Types

```
1. FileReadError: Cannot read file
2. HashError: Cannot compute hash
3. R2UploadError: Cannot upload to R2
4. ExtractionError: NER failed
5. ResolutionError: Merge failed
6. GraphError: Edge creation failed
7. DatabaseError: DB write failed
8. ValidationError: Data validation failed
9. JurisdictionError: Invalid jurisdiction reference
10. FIRInvariantError: Case does not have exactly one primary FIR
11. IdentityPushError: Global entity push failed
12. ScopedAnalyticsError: Cross-case merge failed
```

### 9.2 Error Strategy

```
Non-fatal errors (log and continue):
  - Single file fails to process
  - Single entity fails to resolve
  - Single edge fails to create
  - Stage 11 fails for one entity (retry later)

Fatal errors (stop pipeline):
  - Database connection lost
  - R2 connection lost
  - All files fail to process
  - Out of memory
  - Case has no primary FIR (invariant violation)

Error recovery:
  - Stages 1-6 are idempotent per Case (can re-run)
  - PipelineRun tracks status (RUNNING → FAILED)
  - Can resume from last successful stage
  - Stage 11 can be retried independently (cross-case, not blocking)
  - Stage 12 is stateless (compute-on-demand, no recovery needed)
```


## ============================================================
## 10. PERFORMANCE
## ============================================================

### 10.1 Current Performance (per-Case)

```
50 files per Case:
  Ingestion:          ~5s
  Extraction:         ~30s (code) / ~120s (LLM)
  Entity Resolution:  ~20s
  Temporal:           ~10s
  Graph Construction: ~5s
  Analytics:          ~15s
  Global Push:        ~5s
  Total:              ~90s (code) / ~200s (LLM)
```

### 10.2 Scaling Considerations

```
For 1000 Cases:
  - Process Cases in parallel (4-8 workers)
  - Stage 11 global push: batch updates to Global Entity Identity Index
  - Stage 12: cache frequently-requested scopes
  - Connection pooling (10-20 connections)
  - Bulk inserts (batch size 1000)

For 10,000 Cases:
  - Queue-based processing per Case
  - Multiple pipeline workers
  - Read replicas for queries
  - Partition large tables by case_id
  - Global Entity Identity Index: use pgvector for similarity search
```


## ============================================================
## 11. SECURITY
## ============================================================

```
1. Database:
   - Encrypted at rest (PostgreSQL TDE)
   - Encrypted in transit (SSL/TLS)
   - Field-level encryption (national_id_number)

2. R2 Storage:
   - Private bucket (no public access)
   - Presigned URLs for temporary access
   - Versioning enabled

3. API:
   - JWT authentication
   - Role-based access control
   - Audit logging for all operations

4. Secrets:
   - Stored in .env file
   - Never committed to git
   - Rotated quarterly

5. Multi-Case Isolation:
   - Stages 1-6: strict case_id scoping, no cross-case data leak
   - Stage 11: controlled cross-case writes (identity signals only)
   - Stage 12: read-only cross-case access, scope enforced by caller
```


## ============================================================
## 12. KEY DESIGN DECISIONS
## ============================================================

```
1. JurisdictionNode replaces PoliceStation
   - Configurable hierarchy (NATIONAL → STATE → DISTRICT → CITY → STATION)
   - FIR number uniqueness scoped to jurisdiction
   - Supports cross-jurisdiction Cases with primary/secondary FIR model

2. Case has exactly one primary FIR (invariant)
   - Created at Case creation time
   - Additional FIRs linked as secondary
   - Enforced at application and database level

3. Local graphs are built per Case and stored permanently
   - Stages 1-6 process a single Case
   - Graph is the canonical output, persisted until deleted
   - Enables re-analysis without re-running the pipeline

4. Global Entity Identity Index links same-entity across Cases
   - Updated by Stage 11 (push model)
   - Identity signals: phone, name, DOB, national ID, associates
   - Confidence-scored links with provenance tracking

5. Scoped Analytical Graphs are computed on-demand, not stored
   - Stage 12 merges local graphs using global identity links
   - Fresh computation ensures data is always current
   - No stale cross-case caches

6. Stage 11 is async-friendly
   - Global push does not block the Case pipeline
   - Can be retried independently
   - Failures in Stage 11 do not affect local graph quality

7. Stage 12 is stateless
   - No persistent cross-case graph
   - Each request defines its own scope
   - Enables flexible ad-hoc analysis
```


## ============================================================
## 13. SUMMARY
## ============================================================

```
Pipeline Architecture:
  - 8 stages: 6 per-Case + 1 cross-case push + 1 on-demand analysis
  - Pipeline orchestrator manages flow
  - Database stores all results (local graphs permanently)
  - R2 stores all files (scoped by case_id)
  - Configuration drives behavior

Key Data Stores:
  - Local Graphs: per-Case, stored permanently (Stages 1-6)
  - Global Entity Identity Index: cross-case, updated by Stage 11
  - ScopedAnalyticalGraphs: computed on-demand, not stored (Stage 12)

Data Flow:
  Files → Ingestion → Extraction → Resolution → Temporal → Graph → Analytics → Global Push
    ↓         ↓           ↓            ↓           ↓         ↓         ↓            ↓
   R2      DB:Files    DB:Entities  DB:Resolved  DB:Time  DB:Graph  DB:Analytics  Global Index
                                                                                      ↓
                                                                              Scoped Analytics (on-demand)
```


## ============================================================
## 14. PIPELINE STAGE SUMMARY
## ============================================================

| Stage | Name                    | Scope     | Input                      | Output                       | Stored? |
|-------|-------------------------|-----------|----------------------------|------------------------------|---------|
| 1     | Ingestion               | Per-Case  | Files + case_id            | IngestedFile records         | Yes     |
| 2     | Extraction              | Per-Case  | IngestedFiles              | ExtractedEntity/Relation     | Yes     |
| 3     | Entity Resolution       | Per-Case  | ExtractedEntities          | ResolvedEntity records       | Yes     |
| 4     | Temporal Enrichment     | Per-Case  | Entities + Relations       | TemporalInfo + Contradictions| Yes     |
| 5     | Graph Construction      | Per-Case  | All prior outputs          | Local graph (Node + Edge)    | Yes     |
| 6     | Analytics Engine        | Per-Case  | Local graph                | Analytics results            | Yes     |
| 7     | Global Entity Push      | Cross-Case| ResolvedEntities           | Global Entity Identity Index | Yes     |
| 8     | Scoped Analytics        | On-Demand | Scope + Global Index       | ScopedAnalyticalGraph        | No      |
