# ============================================================
# GAP ANALYSIS: Teammate's Schema vs Our Pipeline Schema
# ============================================================
#
# This document compares the teammate's full-system schema (v4)
# with our pipeline's DB schema (v1) and identifies:
#   1. What they have that we DON'T extract
#   2. What we extract that they DON'T have
#   3. Where our schemas align
#   4. What needs to be built to bridge the gap
# ============================================================


## ============================================================
## SECTION 1: TABLE-BY-TABLE COMPARISON
## ============================================================

### Tables THEY have that WE have (aligned):
| Table | Their Schema | Our Schema | Status |
|-------|-------------|------------|--------|
| Case | ✅ | ✅ | ALIGNED — same fields |
| PoliceStation | ✅ | ✅ | ALIGNED |
| User | ✅ | ✅ | ALIGNED |
| FIR | ✅ | ❌ | SEE BELOW — they have FIR as separate table, we embed in Case |
| Suspect | ✅ | ✅ | ALIGNED — same lifecycle model |
| Evidence | ✅ | ✅ | ALIGNED |
| EvidenceFile | ✅ | ✅ | ALIGNED |
| OcrResult | ✅ | ✅ | ALIGNED |
| AudioTranscript | ✅ | ✅ | ALIGNED |
| Person | ✅ | ✅ | ALIGNED |
| PersonAlias | ✅ | ✅ | ALIGNED |
| Phone | ✅ | ✅ | ALIGNED |
| Vehicle | ✅ | ✅ | ALIGNED |
| Location | ✅ | ✅ | ALIGNED |
| BankAccount | ✅ | ✅ | ALIGNED |
| CaseEntityLink | ✅ | ✅ | ALIGNED |
| FaceEmbedding | ✅ | ✅ | ALIGNED |
| EntityResolutionCandidate | ✅ | ✅ | ALIGNED |
| CrimeZoneScore | ✅ | ✅ | ALIGNED |
| CaseNarrativeEmbedding | ✅ | ✅ | ALIGNED |
| AuditLog | ✅ | ✅ | ALIGNED |


### Tables THEY have that we DON'T:
| Table | What It Does | Priority | Notes |
|-------|-------------|----------|-------|
| **FIR** | Separate FIR table with firNumber, status, description, filedById | HIGH | They treat FIR as first-class entity; we embed FIR info in Case.fir_description |
| **Organization** | REMOVED in v3 — out of scope | N/A | Dropped in their v3 |
| **OrganizationMember** | REMOVED in v3 — out of scope | N/A | Dropped in their v3 |


### Tables WE have that THEY don't:
| Table | What It Does | Priority | Notes |
|-------|-------------|----------|-------|
| **PipelineRun** | Tracks each pipeline execution | HIGH | Critical for audit trail and incremental runs |
| **IngestedFile** | File intake record with hash, type, source | HIGH | They have EvidenceFile but not the ingestion pipeline layer |
| **DataQualityScore** | Quality assessment per file | MEDIUM | Their schema assumes clean data; we score quality |
| **AdversarialCheck** | Adversarial data detection per file | HIGH | They don't have adversarial detection at ingestion |
| **EvidenceIntegrity** | Chain of custody for raw files | HIGH | They have Evidence but not file-level integrity tracking |
| **DependencyGroup** | Files that must be processed together | LOW | Pipeline-specific optimization |
| **ExtractionLog** | Per-file extraction stats | MEDIUM | Pipeline debugging/monitoring |
| **ExtractedEntity** | Raw extracted entities before resolution | HIGH | They only have resolved entities; we preserve raw extraction |
| **ExtractedRelation** | Raw extracted relations before resolution | HIGH | Same — they only have resolved |
| **Contradiction** | Detected contradictions between sources | HIGH | They don't track contradictions |
| **ResolutionHistory** | Audit trail of merge decisions | HIGH | They have EntityResolutionCandidate but no history |
| **TemporalInfo** | Timeline events extracted from evidence | HIGH | They don't have temporal extraction |
| **SpatialInfo** | Location presence events | MEDIUM | They have Location but not spatial extraction events |
| **CoverageInterval** | Time ranges entities are active | LOW | Derived analytics |
| **GraphNode** | Knowledge graph nodes | HIGH | They have Person/Phone/Vehicle but not the graph abstraction |
| **GraphEdge** | Knowledge graph edges with confidence | HIGH | They have CaseEntityLink but not full graph edges |
| **MissingEdge** | Detected but uncreated edges | MEDIUM | Graph quality signal |
| **RejectedEdge** | Filtered-out low-confidence edges | MEDIUM | Graph quality signal |
| **AdversarialEdgeScore** | Adversarial scoring per edge | HIGH | They don't have adversarial detection on relationships |
| **EventCluster** | Groups of related events | LOW | Analytics feature |


## ============================================================
## SECTION 2: FIELD-LEVEL GAPS
## ============================================================

### Fields THEY have that WE don't:
| Table | Field | What It Does | Priority |
|-------|-------|-------------|----------|
| Case | fir_number | Official jurisdiction FIR number | HIGH |
| FIR (table) | fir_document_r2_key | Scan/PDF of FIR stored in R2 | HIGH |
| FIR (table) | status | FIR lifecycle: REGISTERED → UNDER_INVESTIGATION → CHARGE_SHEET_FILED → QUASHED | HIGH |
| Suspect | criminal_record_number | Internal record number (assigned on conviction) | MEDIUM |
| Suspect | convicted_by_id | Who confirmed CONVICTED status | MEDIUM |
| Suspect | convicted_at | When convicted | MEDIUM |
| EvidenceFile | r2_key | Cloudflare R2 object key for binary storage | HIGH |
| EvidenceFile | uploaded_by_id | Who uploaded this specific file | HIGH |
| Person | photo_r2_key | Primary photo R2 key | HIGH |
| Person | identity_source | How identity was established | HIGH |
| FaceEmbedding | embedding_vector | pgvector 512-d face embedding | HIGH (future) |
| FaceEmbedding | detector_confidence | Face detection confidence | HIGH (future) |
| CaseNarrativeEmbedding | embedding_vector | pgvector 1536-d text embedding | HIGH (future) |

### Fields WE have that THEY don't:
| Table | Field | What It Does | Priority |
|-------|-------|-------------|----------|
| ExtractedEntity | confidence_score | Multi-factor confidence | HIGH |
| ExtractedEntity | source_reliability | Source reliability from matrix | HIGH |
| ExtractedEntity | derivation_depth | How many inference steps from raw data | HIGH |
| ExtractedEntity | epistemic_category | observation vs inference vs hypothesis | HIGH |
| ExtractedRelation | semantic_edge_type | entrepreneurial/associational/quasi-governmental | HIGH |
| ExtractedRelation | temporal_info | When the relation occurred | HIGH |
| ResolvedEntity | merge_confidence | Confidence in the merge decision | HIGH |
| ResolvedEntity | merge_type | auto/review/reject/single/llm | HIGH |
| ResolvedEntity | llm_reasoning | LLM reasoning for merge | MEDIUM |
| ResolvedEntity | effective_confidence | Final confidence after resolution | HIGH |
| GraphEdge | adversarial_score | Adversarial risk per edge | HIGH |
| GraphEdge | supporting_evidence | Files supporting this edge | HIGH |
| GraphEdge | contradicting_evidence | Files contradicting this edge | HIGH |
| GraphEdge | 4-factor confidence | source_reliability + entity_resolution + extraction + coverage | HIGH |
| AdversarialCheck | behavioral_anomaly | Unusual data patterns | HIGH |
| AdversarialCheck | temporal_anomaly | Time-based anomalies | HIGH |
| AdversarialCheck | content_anomaly | Content-based anomalies | HIGH |
| DataQualityScore | completeness_score | Field completeness | MEDIUM |
| DataQualityScore | structural_consistency | Schema adherence | MEDIUM |
| DataQualityScore | content_richness | Information density | MEDIUM |


## ============================================================
## SECTION 3: ENUM GAPS
## ============================================================

### Enums THEY have that WE don't:
| Enum | Values | Notes |
|------|--------|-------|
| Role | INSPECTOR, ADMIN, AUDIT_LOGGER | We have this in User.role as varchar |
| UserStatus | FORCE_PASSWORD_CHANGE, ACTIVE, SUSPENDED | We have this in User.status as varchar |
| CaseStatus | ACTIVE, PENDING, SOLVED, CLOSED | We have this as varchar |
| FirStatus | REGISTERED, UNDER_INVESTIGATION, CHARGE_SHEET_FILED, QUASHED | **WE DON'T HAVE THIS** — FIR lifecycle tracking |
| SuspectStatus | UNKNOWN → CONVICTED (9 states) | We have this as varchar |
| EvidenceType | PHYSICAL, DIGITAL, DOCUMENTARY, BIOLOGICAL, TESTIMONIAL | We have this as varchar |
| EvidenceStatus | LOGGED → COURT_RELEASED (5 states) | We have this as varchar |
| Gender | MALE, FEMALE, OTHER | We have this as varchar |
| EvidenceFileType | AUDIO, IMAGE, DOCUMENT, VIDEO, CSV | We have this as varchar |
| IdentitySource | MANUAL, FIR_INFERRED, FACE_MATCH_CONFIRMED | We have this as varchar |
| ResolutionMatchType | FUZZY, PHONETIC, FACE, MANUAL | We have match_type as varchar |
| ResolutionStatus | PENDING, CONFIRMED, REJECTED | We have status as varchar |
| RiskBand | GREEN, AMBER, RED | We have band as varchar |
| CaseEntityRole | SUSPECT, VICTIM, WITNESS, ASSOCIATE, VEHICLE_OWNER | We have this as varchar |
| AuditAction | CREATE, UPDATE, DELETE, READ_SENSITIVE, EXPORT, FACE_MATCH_QUERY, ENTITY_RESOLUTION_CONFIRM, SUSPECT_STATUS_CHANGE, LOGIN | We have this as varchar |

### Enums WE have that THEY don't:
| Enum | Values | Notes |
|------|--------|-------|
| EntityType | 16 values (PERSON, PHONE, VEHICLE, LOCATION, etc.) | **THEY DON'T HAVE THIS** — they map directly to tables |
| RelationType | 21 values (CALLED, TRANSFERRED_TO, VISITED, etc.) | **THEY DON'T HAVE THIS** — they use CaseEntityLink |
| SemanticEdgeType | ENTREPRENEURIAL, ASSOCIATIONAL, QUASI_GOVERNMENTAL | **THEY DON'T HAVE THIS** — graph-specific |
| EpistemicCategory | OBSERVATION, INFERENCE, HYPOTHESIS, UNKNOWN_PROVENANCE | **THEY DON'T HAVE THIS** — epistemic tracking |
| ProvenanceClass | OBSERVATIONAL, DERIVED, INVESTIGATIVE, MODEL_GENERATED | **THEY DON'T HAVE THIS** — provenance tracking |
| VerificationStatus | UNVERIFIED, CORROBORATED, INDEPENDENTLY_VERIFIED, LEGALLY_ESTABLISHED | **THEY DON'T HAVE THIS** — verification tracking |
| ContradictionType | 8 values (IDENTITY, TEMPORAL, LOCATION, etc.) | **THEY DON'T HAVE THIS** — contradiction detection |
| SourceType | CDR, BANK, FIR, CCTV, SOCIAL, DEVICE, TEXT, etc. | They have EvidenceType but not source_type |
| DocumentType | FIR, CDR, BANK_RECORD, CCTV_LOG, etc. | **THEY DON'T HAVE THIS** — document classification |
| MergeType | AUTO, REVIEW, REJECT, SINGLE, LLM | **THEY DON'T HAVE THIS** — merge decision tracking |
| AdversarialRiskLevel | LOW, MEDIUM, HIGH, CRITICAL | **THEY DON'T HAVE THIS** — adversarial risk |


## ============================================================
## SECTION 4: WHAT NEEDS TO BE BUILT
## ============================================================

### Immediate (Stage 6 Analytics):
1. **FIR Table** — Extract FIR number, status, and description into a separate table
2. **R2 Storage Integration** — Map file ingestion to Cloudflare R2 keys
3. **Embedding Pipeline** — Face embeddings (ArcFace/InsightFace 512-d) and text embeddings (1536-d)
4. **Crime Zone Scoring** — H3 hex-based geospatial risk scoring
5. **Semantic Search** — Case narrative embedding + similarity search

### Pipeline Improvements:
6. **Vehicle Extraction** — Currently extracting PHONE, PERSON, LOCATION well; VEHICLE needs work
7. **Bank Account Extraction** — Extract account numbers with IFSC codes
8. **Person Photo Extraction** — Extract and store face photos from CCTV/social media
9. **FIR Status Tracking** — Track FIR lifecycle (REGISTERED → CHARGE_SHEET_FILED)

### Database Integration:
10. **Prisma Schema Translation** — Convert our DBML to Prisma schema
11. **pgvector Setup** — Install and configure pgvector extension
12. **R2 Bucket Setup** — Configure Cloudflare R2 for binary storage
13. **Audit Log Integration** — Map pipeline audit trail to AuditLog table


## ============================================================
## SECTION 5: SUMMARY
## ============================================================

### Coverage Score:
- **Tables aligned**: 20/21 (95%) — FIR is the only missing table
- **Fields covered**: ~85% — we have more confidence/epistemic fields; they have more R2/storage fields
- **Enums covered**: ~70% — we have extraction-specific enums; they have lifecycle enums

### Key Differences:
1. **Philosophy**: They treat entities as first-class tables (Person, Phone, Vehicle). We treat them as extracted data that gets resolved into those tables.
2. **Confidence**: We have multi-factor confidence scoring (4 factors). They don't track confidence at the entity level.
3. **Epistemic**: We track observation vs inference vs hypothesis. They don't.
4. **Adversarial**: We have first-class adversarial detection. They don't.
5. **Contradictions**: We detect and track contradictions. They don't.
6. **Provenance**: We have full provenance chains. They have partial.

### What We're Missing:
1. **FIR as first-class entity** — They have a separate FIR table with status lifecycle
2. **R2 Storage** — They use Cloudflare R2 for all binary/file storage
3. **Embeddings** — They have pgvector for face and text embeddings
4. **H3 Geospatial** — They have H3 hex-based crime zone scoring

### What They're Missing:
1. **Pipeline Audit Trail** — We track every pipeline run, ingestion, extraction, resolution
2. **Confidence Scoring** — We have multi-factor confidence at entity, relation, and edge level
3. **Epistemic Tracking** — We distinguish observations from inferences from hypotheses
4. **Adversarial Detection** — We detect and score adversarial data at file and edge level
5. **Contradiction Detection** — We find and track contradictions between sources
6. **Provenance Chains** — Every fact traces back to source evidence files
7. **Semantic Edge Types** — We classify relationships as entrepreneurial/associational/quasi-governmental
