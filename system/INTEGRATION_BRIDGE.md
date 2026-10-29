# ============================================================
# INTEGRATION BRIDGE: Our Pipeline → Their Backend
# ============================================================
#
# This document defines how our pipeline output maps to their
# backend database tables via API calls.
# ============================================================


## ============================================================
## 1. ENTITY MAPPING
## ============================================================

### Our ExtractedEntity → Their Tables

| Our EntityType | Their Table | Mapping Rules |
|---------------|-------------|---------------|
| PERSON | Person | canonical_name → name, gender → gender, dob → dob |
| PHONE | Phone | number → number (E.164 format) |
| VEHICLE | Vehicle | plate_number → plate_number, type → type, color → color |
| ACCOUNT | BankAccount | account_number → account_number_encrypted (encrypt), bank_name → bank_name |
| LOCATION | Location | name → name, lat → lat, lng → lng |
| ORGANIZATION | (No table — out of scope in v3) | Skip or store in attributes |
| AMOUNT | (No table — attribute on Evidence) | Store in Evidence.description or Suspect.notes |
| DATE | (No table — attribute on Evidence) | Store in Evidence.collected_at |
| DEVICE | (No table — attribute on Evidence) | Store in Evidence.description |
| EVENT | (No table — maps to Case) | Store in Case.fir_description |

### Our ResolvedEntity → Their Person Table

```
Pipeline Output                    →    Their Database
─────────────────────────────────────────────────────
ResolvedEntity (PERSON)                 Person
  .canonical_name                  →    .canonical_name
  .aliases[0]                     →    PersonAlias.alias
  .phones[0]                      →    Phone.number
  .accounts[0]                    →    BankAccount.account_number
  .addresses[0]                   →    Location.name
  .attributes.national_id_number  →    Person.national_id_number
  .attributes.gender              →    Person.gender
  .attributes.dob                 →    Person.dob
  .provenance_chain               →    (stored in AuditLog.metadata)
  .effective_confidence           →    (stored in AuditLog.metadata)
```

### Our GraphEdge → Their CaseEntityLink

```
Pipeline Output                    →    Their Database
─────────────────────────────────────────────────────
GraphEdge (SUSPECT_OF)                  CaseEntityLink
  .source_id (Person node)        →    .person_id
  .target_id (Case node)          →    .case_id
  .relationship_type              →    .role = "SUSPECT"
  .confidence_score               →    (stored in metadata)
  .supporting_evidence            →    (stored in metadata)

GraphEdge (VICTIM_OF)                   CaseEntityLink
  .source_id (Person node)        →    .person_id
  .target_id (Case node)          →    .case_id
  .relationship_type              →    .role = "VICTIM"

GraphEdge (WITNESS_OF)                  CaseEntityLink
  .source_id (Person node)        →    .person_id
  .target_id (Case node)          →    .case_id
  .relationship_type              →    .role = "WITNESS"
```


## ============================================================
## 2. BRIDGE SERVICE API ENDPOINTS
## ============================================================

The bridge service exposes these endpoints for their backend to call:

### POST /api/pipeline/sync
Syncs pipeline output to their database.

**Request:**
```json
{
  "run_id": "run_20260830_193829",
  "case_id": "case_uuid_here",
  "pipeline_output": {
    "entities": [...],
    "relations": [...],
    "resolved_entities": {...},
    "graph_nodes": [...],
    "graph_edges": [...],
    "contradictions": [...]
  }
}
```

**Response:**
```json
{
  "status": "success",
  "synced": {
    "persons": 12,
    "phones": 8,
    "vehicles": 3,
    "bank_accounts": 5,
    "locations": 15,
    "case_entity_links": 25,
    "contradictions_flagged": 3
  },
  "warnings": [
    "Person 'Unknown #3' has no confirmed identity — stored as suspect"
  ]
}
```

### GET /api/pipeline/status/{run_id}
Check sync status for a pipeline run.

### GET /api/pipeline/entities/{case_id}
Get all pipeline-extracted entities for a case.

### GET /api/pipeline/graph/{case_id}
Get knowledge graph nodes and edges for a case.

### GET /api/pipeline/contradictions/{case_id}
Get all contradictions detected for a case.

### POST /api/pipeline/re-sync/{run_id}
Re-sync a pipeline run (after corrections).


## ============================================================
## 3. DATA FLOW SEQUENCE
## ============================================================

```
Inspector uploads evidence files
        │
        ▼
┌─────────────────────────────────────────┐
│ 1. THEIR BACKEND receives files         │
│    - Stores in Cloudflare R2            │
│    - Creates Evidence + EvidenceFile    │
│    - Calls POST /api/pipeline/ingest    │
└────────────────┬────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────┐
│ 2. OUR PIPELINE processes files         │
│    - Stage 1: Ingestion (hash, quality) │
│    - Stage 2: Extraction (NER, relations)│
│    - Stage 3: Resolution (merge, dedup) │
│    - Stage 4: Temporal (timeline)       │
│    - Stage 5: Graph (nodes, edges)      │
│    - Output: JSON files                 │
└────────────────┬────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────┐
│ 3. BRIDGE SERVICE syncs to their DB     │
│    - Reads pipeline output JSON         │
│    - Maps to their schema               │
│    - Calls their API endpoints          │
│    - Stores confidence scores           │
│    - Flags contradictions               │
│    - Logs to AuditLog                   │
└────────────────┬────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────┐
│ 4. THEIR FRONTEND displays results      │
│    - Person cards with aliases          │
│    - Phone call network graph           │
│    - Bank transaction flow              │
│    - Timeline view                      │
│    - Contradiction alerts               │
│    - Risk scores                        │
└─────────────────────────────────────────┘
```


## ============================================================
## 4. EXAMPLE: FULL SYNC FLOW
## ============================================================

### Step 1: Inspector uploads CDR file
```
POST /api/evidence/upload
Body: { file: "03_CDR_Suresh.csv", case_id: "case_123" }
Response: { evidence_id: "ev_456", file_id: "file_789" }
```

### Step 2: Trigger pipeline
```
POST /api/pipeline/run
Body: { case_id: "case_123", files: ["file_789"] }
Response: { run_id: "run_20260830_193829", status: "running" }
```

### Step 3: Pipeline completes → Bridge syncs
```
POST /api/pipeline/sync
Body: {
  run_id: "run_20260830_193829",
  case_id: "case_123",
  pipeline_output: { ... }
}
Response: {
  synced: {
    persons: 3,        // Suresh, Rakesh, Meena
    phones: 4,         // 9876543210, 9876543211, 9876543213, 9876543214
    locations: 2,      // Karol Bagh Tower 1, Lajpat Nagar Tower 1
    case_entity_links: 6,
    contradictions_flagged: 0
  }
}
```

### Step 4: Frontend displays
```
GET /api/cases/case_123/graph
Response: {
  nodes: [
    { id: "person_1", name: "Suresh Kumar", type: "suspect", confidence: 0.92 },
    { id: "person_2", name: "Rakesh Kumar", type: "suspect", confidence: 0.95 },
    { id: "phone_1", number: "9876543211", owner: "person_1" },
    { id: "phone_2", number: "9876543210", owner: "person_2" },
    { id: "loc_1", name: "Karol Bagh Tower 1" }
  ],
  edges: [
    { source: "phone_1", target: "phone_2", type: "CALLED", count: 5 },
    { source: "phone_1", target: "loc_1", type: "VISITED", count: 3 }
  ]
}
```


## ============================================================
## 5. WHAT THEIR BACKEND NEEDS TO ADD
## ============================================================

### New API Endpoints (for pipeline integration):
1. `POST /api/pipeline/ingest` — Receive file metadata from pipeline
2. `POST /api/pipeline/sync` — Sync pipeline output to database
3. `GET /api/pipeline/status/{run_id}` — Check pipeline run status
4. `GET /api/pipeline/graph/{case_id}` — Get knowledge graph
5. `GET /api/pipeline/contradictions/{case_id}` — Get contradictions

### New Database Fields (on existing tables):
1. `Person.confidence_score` — float, from our pipeline
2. `Person.epistemic_status` — varchar, observation/inference/hypothesis
3. `Phone.confidence_score` — float
4. `CaseEntityLink.confidence_score` — float
5. `CaseEntityLink.supporting_evidence` — jsonb, array of file names
6. `CaseEntityLink.adversarial_score` — float

### New Tables (optional, for full pipeline data):
1. `PipelineRun` — Track each pipeline execution
2. `ExtractedEntity` — Raw entities before resolution
3. `Contradiction` — Detected contradictions
4. `AdversarialCheck` — Adversarial detection results


## ============================================================
## 6. WHAT WE NEED TO BUILD
## ============================================================

### Bridge Service (Node.js/Python):
1. Read pipeline output JSON files
2. Map entities to their schema
3. Call their API endpoints
4. Handle errors and retries
5. Log sync operations

### API Client:
1. Authentication (API key or JWT)
2. Rate limiting
3. Request/response logging
4. Error handling

### Configuration:
1. Their API base URL
2. Authentication credentials
3. Case ID mapping
4. Sync frequency (real-time vs batch)


## ============================================================
## 7. SUMMARY
## ============================================================

| Component | Their Side | Our Side | Bridge |
|-----------|-----------|----------|--------|
| Files | EvidenceFile (R2) | IngestedFile | Maps file metadata |
| Entities | Person, Phone, Vehicle, BankAccount | ExtractedEntity, ResolvedEntity | Maps entity types |
| Relationships | CaseEntityLink | GraphEdge | Maps relationship types |
| Confidence | (new field) | ConfidenceSchema | Passes through |
| Epistemic | (new field) | EpistemicCategory | Passes through |
| Adversarial | (new field) | AdversarialCheck | Passes through |
| Contradictions | (new table) | Contradiction | Creates alerts |
| Audit | AuditLog | PipelineRun, ExtractionLog | Maps to audit entries |

**The bridge is the missing piece.** Our pipeline produces intelligence. Their backend stores and serves it. The bridge connects them.
