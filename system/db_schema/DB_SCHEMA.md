# AI-POWERED CRIMINAL NETWORK ANALYSIS — DATABASE SCHEMA (v5)

**Source of truth:** `src/models/schema.py` (v4 — Revision 3+ Architecture)
**Derived from:** schema.py → prisma/schema.prisma → DB_SCHEMA.md
**Target:** PostgreSQL 16 + pgvector

---

## Design Principles

- Every row traces back to source evidence via provenance chains
- Confidence scores are multi-factor (source reliability × evidence basis × derivation depth)
- Epistemic status tracks observation vs inference vs hypothesis
- The system may increase confidence but never upgrades inference → observation
- Adversarial detection is first-class, not bolted on
- FIR identity is immutable; lifecycle fields are mutable and authorization-controlled
- Case ↔ FIR is a 1:1 invariant (FIR.caseId is unique)
- FIR number uniqueness is jurisdiction-scoped, not global
- Lifecycle transitions are explicit controlled actions with full audit trail
- CaseRelationship preserves evolution of relationship assessments over time
- AnalysisRun is an immutable historical analytical execution
- InvestigationWorkspace organizes work; CaseAccess controls authorization

---

## Changelog

- **v5 (2026-09-09):** Revision 3+ Architecture
  - FIR: replaced `status` with three lifecycle dimensions (`investigation_status`, `legal_disposition`, `record_status`)
  - Added `FIRLifecycleHistory` for lifecycle audit trail
  - Added `CaseRelationship`, `CaseRelationshipHistory`
  - Added `CaseAccess` (authorization model)
  - Added `InvestigationWorkspace`, `WorkspaceCase`, `WorkspaceMember`
  - Added `AnalysisRun`, `AnalysisRunCase`, `AnalysisRunResult`
  - Added `Finding` (conclusions/hypotheses/critic output)
  - Case ↔ FIR 1:1 invariant enforced (`FIR.caseId` unique)
  - `FirStatus` enum removed
- **v4 (2026-09-09):** Multi-Case Architecture (jurisdiction hierarchy, global entity index)
- **v3 (2026-08-31):** Derived from pipeline schema.py v2
- **v2 (2026-08-31):** Aligned with teammate's schema
- **v1 (2026-08-30):** Initial schema

---

## Entity Summary

| Category | Count | Tables |
|----------|-------|--------|
| Core Domain | 7 | Case, FIR, User, Person, Suspect, Evidence, EvidenceFile |
| Pipeline Run | 1 | PipelineRun |
| Pipeline Stage 1 | 5 | IngestedFile, DataQualityScore, AdversarialCheck, EvidenceIntegrity, DependencyGroup |
| Pipeline Stage 2 | 3 | ExtractedEntity, ExtractedRelation, ExtractionLog |
| Pipeline Stage 3 | 5 | ResolvedEntity, EntityResolutionCandidate, Contradiction, ResolutionHistory, MergeCandidate |
| Pipeline Stage 4 | 3 | TemporalInfo, SpatialInfo, CoverageInterval |
| Pipeline Stage 5 | 6 | GraphNode, GraphEdge, MissingEdge, RejectedEdge, AdversarialEdgeScore, EventCluster |
| Knowledge Graph | 6 | Person, PersonAlias, Phone, Vehicle, BankAccount, CaseEntityLink |
| Evidence Management | 4 | Evidence, EvidenceFile, OcrResult, AudioTranscript |
| Face Recognition | 1 | FaceEmbedding |
| Geospatial | 1 | CrimeZoneScore |
| Semantic Search | 1 | CaseNarrativeEmbedding |
| Audit | 1 | AuditLog |
| Jurisdiction | 6 | JurisdictionNode, GeographicJurisdiction, UnresolvedJurisdiction, JurisdictionHistory, CaseJurisdictionLink |
| Global Entity | 3 | GlobalEntity, GlobalEntityLink, CrossCaseAlert, CrossCaseAlertCase |
| **CaseRelationship** | **2** | **CaseRelationship, CaseRelationshipHistory** |
| **CaseAccess** | **1** | **CaseAccess** |
| **InvestigationWorkspace** | **3** | **InvestigationWorkspace, WorkspaceCase, WorkspaceMember** |
| **AnalysisRun** | **3** | **AnalysisRun, AnalysisRunCase, AnalysisRunResult** |
| **Finding** | **1** | **Finding** |
| **FIR Lifecycle** | **1** | **FIRLifecycleHistory** |
| **Total** | **50** | |

---

## Enums

### Pipeline Core Enums

| Enum | Values | Purpose |
|------|--------|---------|
| EntityType | PERSON, PHONE, VEHICLE, LOCATION, ACCOUNT, ORGANIZATION, EVENT, DEVICE, AMOUNT, DATE, DOCUMENT, EMAIL, VEHICLE_PLATE, IP_ADDRESS, CASE_NUMBER, UNKNOWN | Entity types |
| RelationType | CALLED, MESSED, ASSOCIATED_WITH, TRANSFERRED_TO, RECEIVED_FROM, OWNS_ACCOUNT, FAMILY_OF, FRIEND_OF, ASSOCIATE_OF, WORKS_WITH, LIVES_AT, WORKS_AT, VISITED, SUSPECT_OF, VICTIM_OF, WITNESS_OF, MEMBER_OF, EXTRACTED_FROM, DERIVED_FROM, POSSIBLE_IDENTITY, SAME_AS | Relation types |
| SemanticEdgeType | ENTREPRENEURIAL, ASSOCIATIONAL, QUASI_GOVERNMENTAL | Relationship semantics |
| EpistemicCategory | OBSERVATION, INFERENCE, HYPOTHESIS, UNKNOWN_PROVENANCE | Knowledge claim type |
| ProvenanceClass | OBSERVATIONAL, DERIVED, INVESTIGATIVE, MODEL_GENERATED | How data was obtained |
| VerificationStatus | UNVERIFIED, CORROBORATED, INDEPENDENTLY_VERIFIED, LEGALLY_ESTABLISHED | Verification level |
| ContradictionType | IDENTITY, TEMPORAL, LOCATION, ATTRIBUTION, SOURCE_CONTENT, EVENT_IDENTITY, DERIVATION, UNKNOWN | Contradiction type |
| ContradictionSeverity | LOW, MEDIUM, HIGH, CRITICAL | Contradiction severity |

### Domain Enums

| Enum | Values | Purpose |
|------|--------|---------|
| Gender | MALE, FEMALE, OTHER | Person gender |
| CaseStatus | ACTIVE, PENDING, SOLVED, CLOSED | Case lifecycle |
| UserRole | INSPECTOR, ADMIN, AUDIT_LOGGER | System roles |
| UserStatus | ACTIVE, INACTIVE, FORCE_PASSWORD_CHANGE | Account status |
| SuspectStatus | UNDER_INVESTIGATION, ARRESTED, CHARGED, CONVICTED, ACQUITTED, RELEASED | Suspect lifecycle |
| EvidenceStatus | LOGGED, COLLECTED, ANALYZED, ADMITTED, DISMISSED | Evidence lifecycle |
| CaseEntityRole | SUSPECT, VICTIM, WITNESS, ASSOCIATE, VEHICLE_OWNER | Person's role in case |
| MergeType | AUTO, REVIEW, REJECT, SINGLE, LLM | Entity resolution merge type |
| CandidateStatus | PENDING, CONFIRMED, REJECTED | Resolution candidate status |
| GraphNodeResolutionStatus | RESOLVED, STANDALONE, UNRESOLVED | Graph node resolution |
| AdversarialRiskLevel | LOW, MEDIUM, HIGH, CRITICAL | Adversarial risk |
| FaceStatus | PENDING, EXTRACTED, MATCHED, CONFIRMED, REJECTED, LOW_QUALITY | Face embedding status |
| SourceType | CDR, BANK, FIR, CCTV, SOCIAL, DEVICE, TEXT, DOCUMENT, IMAGE, AUDIO, VIDEO, UNKNOWN | Ingested file source |
| DocumentType | FIR, CDR, BANK_RECORD, CCTV_LOG, SOCIAL_MEDIA, DEVICE_FORENSIC, WITNESS_STATEMENT, SURVEILLANCE_REPORT, JOURNALIST_NOTE, INVESTIGATOR_SUMMARY, UNKNOWN | Document type |
| PipelineRunStatus | RUNNING, COMPLETED, FAILED, PARTIAL | Pipeline run status |

### Jurisdiction Enums

| Enum | Values | Purpose |
|------|--------|---------|
| JurisdictionNodeType | NATION, STATE_UT, COMMISSIONERATE, RANGE, DISTRICT, CITY, ZONE, DIVISION, SUB_DIVISION, POLICE_STATION, OUT_POST | Org hierarchy levels |
| JurisdictionNodeStatus | ACTIVE, INACTIVE, MERGED, SPLIT | Node status |
| CaseJurisdictionLinkType | PRIMARY, ASSIST, TRANSFER, LINKED | Jurisdiction link type |
| UnresolvedJurisdictionStatus | UNRESOLVED, RESOLVED, REJECTED | Resolution status |
| CrossCaseAlertSeverity | LOW, MEDIUM, HIGH, CRITICAL | Alert severity |

### CaseRelationship Enums (NEW)

| Enum | Values | Purpose |
|------|--------|---------|
| CaseRelationshipAssessment | SHARED_ENTITY, POSSIBLE_SAME_INCIDENT, POSSIBLE_RELATED_CASE, SAME_INCIDENT, RELATED_CASE | Relationship assessment |
| CaseRelationshipStatus | SUGGESTED, CONFIRMED, REJECTED | Relationship status |

### CaseAccess Enums (NEW)

| Enum | Values | Purpose |
|------|--------|---------|
| CaseAccessLevel | READ, WRITE, ADMIN | Authorization level |

### InvestigationWorkspace Enums (NEW)

| Enum | Values | Purpose |
|------|--------|---------|
| WorkspaceRole | OWNER, MEMBER, VIEWER | Workspace role |
| WorkspaceStatus | ACTIVE, ARCHIVED | Workspace status |

### AnalysisRun Enums (NEW)

| Enum | Values | Purpose |
|------|--------|---------|
| AnalysisRunScopeType | LOCAL, SCOPED | Analysis scope |
| AnalysisRunStatus | RUNNING, COMPLETED, FAILED | Execution status |

### Finding Enums (NEW)

| Enum | Values | Purpose |
|------|--------|---------|
| FindingType | HYPOTHESIS, ANOMALY, PATTERN, CRITIC_VERDICT | Finding type |
| CriticStatus | PENDING, CORROBORATED, REFUTED, INCONCLUSIVE | Critic evaluation |

### FIR Lifecycle Enums (NEW)

| Enum | Values | Purpose |
|------|--------|---------|
| FIRInvestigationStatus | REGISTERED, UNDER_INVESTIGATION, CHARGE_SHEETED, FINAL_REPORT, CLOSED | Police operational lifecycle |
| FIRLegalDisposition | NONE, CANCELLED, QUASHED, DISPOSED, CONVICTED, ACQUITTED | Court/legal disposition |
| FIRRecordStatus | ACTIVE, ARCHIVED, SEALED | Administrative/archive state |
| FIRLifecycleDimension | INVESTIGATION_STATUS, LEGAL_DISPOSITION, RECORD_STATUS | Which lifecycle dimension |

---

## Tables

### Core Domain

#### Case

| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| id | UUID | PK, default uuid() | Case identifier |
| jurisdictionNodeId | String | FK → JurisdictionNode, NOT NULL | Primary jurisdiction |
| locationId | String? | FK → Location | Primary location |
| caseStatus | Enum(CaseStatus) | default ACTIVE | Case lifecycle status |
| description | String? | | Case description |
| createdAt | DateTime | default now() | Creation timestamp |
| updatedAt | DateTime? | | Last update timestamp |

**Relations:** jurisdictionNode (1:N), location (N:1), pipelineRuns (1:N), fir (1:1), entities (1:N), evidence (1:N), suspects (1:N), narrativeEmbeddings (1:N), jurisdictionLinks (1:N), caseRelationshipsA (1:N), caseRelationshipsB (1:N), caseAccesses (1:N), workspaceCases (1:N), analysisRunCases (1:N), globalEntityLinks (1:N), crossCaseAlertCases (1:N)

---

#### FIR

| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| id | UUID | PK, default uuid() | FIR identifier |
| caseId | String | FK → Case, **UNIQUE** | One FIR per Case (invariant) |
| firNumber | String | NOT NULL | FIR number (jurisdiction-scoped uniqueness) |
| jurisdictionNodeId | String | FK → JurisdictionNode, NOT NULL | Registering station |
| description | String | NOT NULL | FIR description |
| filedById | String | FK → User, NOT NULL | Who filed the FIR |
| firDocumentR2Key | String? | | R2 key for FIR document |
| filedAt | DateTime? | | When the FIR was filed |
| **investigationStatus** | Enum(FIRInvestigationStatus) | default REGISTERED | Police operational lifecycle |
| **legalDisposition** | Enum(FIRLegalDisposition) | default NONE | Court/legal disposition |
| **recordStatus** | Enum(FIRRecordStatus) | default ACTIVE | Administrative/archive state |
| createdAt | DateTime | default now() | Creation timestamp |
| updatedAt | DateTime? | | Last update timestamp |

**Unique constraints:** `@@unique([firNumber, jurisdictionNodeId])`, `@@unique([caseId])`

**Relations:** case (N:1), jurisdictionNode (N:1), filedBy (N:1), lifecycleHistory (1:N)

**Lifecycle semantics:**
- `investigationStatus`: Police operational state (REGISTERED → UNDER_INVESTIGATION → CHARGE_SHEETED/FINAL_REPORT → CLOSED)
- `legalDisposition`: Court/legal state (NONE → CANCELLED/QUASHED/DISMISSED/CONVICTED/ACQUITTED)
- `recordStatus`: Administrative state (ACTIVE → ARCHIVED/SEALED)
- These are independent dimensions — a FIR can be CLOSED + NONE + ACTIVE
- Lifecycle fields are authorization-controlled, not free-edit
- FIR identity (firNumber, jurisdictionNodeId, filedById, filedAt) is immutable

---

#### User

| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| id | UUID | PK, default uuid() | User identifier |
| email | String | UNIQUE, NOT NULL | Email address |
| name | String | NOT NULL | Full name |
| passwordHash | String | NOT NULL | Hashed password |
| role | Enum(UserRole) | NOT NULL | System role |
| status | Enum(UserStatus) | default FORCE_PASSWORD_CHANGE | Account status |
| jurisdictionNodeId | String? | FK → JurisdictionNode | Assigned jurisdiction |
| createdAt | DateTime | default now() | Creation timestamp |
| updatedAt | DateTime? | | Last update timestamp |

**Relations:** jurisdictionNode (N:1), firsFiled (1:N), caseEntityLinks (1:N), evidenceLogged (1:N), evidenceUploaded (1:N), suspectsConvicted (1:N), facesConfirmed (1:N), contradictionsResolved (1:N), resolutionHistories (1:N), auditLogs (1:N), caseAccessesGranted (1:N), caseAccessesOwned (1:N), workspaceMemberships (1:N), workspacesCreated (1:N), workspacesCaseAdded (1:N), analysisRunsCreated (1:N), findingsCreated (1:N), firLifecycleChanges (1:N), caseRelationshipChanges (1:N), crossCaseAlertsReviewed (1:N), unresolvedJurisdictionsResolved (1:N)

---

#### Person

| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| id | UUID | PK, default uuid() | Person identifier |
| canonicalName | String | NOT NULL | Canonical name |
| gender | Enum(Gender)? | | Gender |
| dob | DateTime? | | Date of birth |
| nationalIdNumber | String? | | National ID |
| photoR2Key | String? | | Photo R2 key |
| identitySource | String | default "MANUAL" | How identity was established |
| nameConfidence | Float? | | Name confidence |
| createdById | String? | FK → User | Who created this person |
| createdAt | DateTime | default now() | Creation timestamp |

**Relations:** aliases (1:N), phones (1:N), vehicles (1:N), bankAccounts (1:N), caseLinks (1:N), suspects (1:N), faceEmbeddings (1:N)

---

#### Suspect

| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| id | UUID | PK, default uuid() | Suspect identifier |
| caseId | String | FK → Case, NOT NULL | Associated case |
| personId | String? | FK → Person | Resolved person |
| name | String | NOT NULL | Suspect name |
| gender | Enum(Gender)? | | Gender |
| dob | DateTime? | | Date of birth |
| nationalIdNumber | String? | | National ID |
| status | Enum(SuspectStatus) | default UNDER_INVESTIGATION | Suspect lifecycle |
| arrestDate | DateTime? | | Arrest date |
| contactInfo | String? | | Contact information |
| photoR2Key | String? | | Photo R2 key |
| criminalRecordNumber | String? | UNIQUE | Criminal record number |
| convictedById | String? | FK → User | Who convicted |
| convictedAt | DateTime? | | Conviction timestamp |
| notes | String? | | Notes |
| createdAt | DateTime | default now() | Creation timestamp |
| updatedAt | DateTime? | | Last update timestamp |

**Relations:** case (N:1), person (N:1), convictedBy (N:1), evidence (1:N), faceEmbeddings (1:N)

**Indexes:** `@@index([personId, caseId])`, `@@index([status])`

---

### Pipeline Run

#### PipelineRun

| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| id | UUID | PK, default uuid() | Run identifier |
| caseId | String? | FK → Case | Associated case |
| jurisdictionNodeId | String? | FK → JurisdictionNode | Associated jurisdiction |
| pipelineVersion | String | default "v1.0" | Pipeline version |
| modelVersions | Json? | | Model versions used |
| policyVersion | String | default "policy_v1.0" | Policy version |
| triggerType | String | NOT NULL | Trigger (new_evidence/reprocessing/manual) |
| status | Enum(PipelineRunStatus) | default RUNNING | Run status |
| startTime | DateTime | default now() | Start timestamp |
| endTime | DateTime? | | End timestamp |
| parentRunId | String? | FK → PipelineRun | Parent run (for reprocessing) |
| inputFiles | Json | NOT NULL | Input file list |
| summary | Json? | | Run summary |

---

### Pipeline Stage 1: Ingestion

#### IngestedFile

| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| id | UUID | PK, default uuid() | File identifier |
| runId | String | FK → PipelineRun, NOT NULL | Pipeline run |
| caseId | String? | FK → Case | Associated case |
| fileName | String | NOT NULL | Original filename |
| filePath | String | NOT NULL | File path |
| fileExt | String | NOT NULL | File extension |
| fileHash | String | NOT NULL | SHA-256 hash |
| fileSizeBytes | Int | NOT NULL | File size |
| detectedType | String | NOT NULL | Detected file type |
| sourceType | Enum(SourceType) | NOT NULL | Source type |
| documentType | Enum(DocumentType)? | | Document type |
| ingestionTime | DateTime | default now() | Ingestion timestamp |
| modifiedTime | DateTime? | | File modification time |
| recordCount | Int? | | Number of records |
| rawSummary | String? | | Raw summary |

**Relations:** run (N:1), dataQualityScore (1:1), adversarialChecks (1:N), evidenceIntegrity (1:1), extractionLogs (1:N), extractedEntities (1:N), extractedRelations (1:N)

---

#### DataQualityScore

| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| id | UUID | PK, default uuid() | Score identifier |
| ingestedFileId | String | FK → IngestedFile, UNIQUE, NOT NULL | Associated file |
| qualityScore | Float | NOT NULL | Overall quality (0.0–1.0) |
| completenessScore | Float | NOT NULL | Completeness (0.0–1.0) |
| structuralConsistency | Float | NOT NULL | Structural consistency |
| contentRichness | Float | NOT NULL | Content richness |
| issues | Json? | | Issues found |
| runId | String | FK → PipelineRun, NOT NULL | Pipeline run |

---

#### AdversarialCheck

| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| id | UUID | PK, default uuid() | Check identifier |
| ingestedFileId | String | FK → IngestedFile, NOT NULL | Associated file |
| isSuspicious | Boolean | default false | Is suspicious |
| behavioralAnomaly | Boolean | default false | Behavioral anomaly |
| temporalAnomaly | Boolean | default false | Temporal anomaly |
| contentAnomaly | Boolean | default false | Content anomaly |
| duplicateSuspect | Boolean | default false | Duplicate suspect |
| riskLevel | Enum(AdversarialRiskLevel) | default LOW | Risk level |
| score | Float | NOT NULL | Risk score (0.0–1.0) |
| reasons | Json? | | Reasons |
| runId | String | FK → PipelineRun, NOT NULL | Pipeline run |

---

#### EvidenceIntegrity

| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| id | UUID | PK, default uuid() | Integrity identifier |
| ingestedFileId | String | FK → IngestedFile, UNIQUE, NOT NULL | Associated file |
| fileHash | String | NOT NULL | File hash |
| hashVerified | Boolean | NOT NULL | Hash verified |
| isAdversarial | Boolean | default false | Is adversarial |
| chainOfCustody | Json? | | Chain of custody |
| transformations | Json? | | Transformations |
| createdAt | DateTime | default now() | Creation timestamp |

---

#### DependencyGroup

| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| id | UUID | PK, default uuid() | Group identifier |
| runId | String | FK → PipelineRun, NOT NULL | Pipeline run |
| groupName | String | NOT NULL | Group name |
| fileCount | Int | default 0 | File count |
| totalEntities | Int | default 0 | Total entities |
| totalRelations | Int | default 0 | Total relations |
| createdAt | DateTime | default now() | Creation timestamp |

---

### Pipeline Stage 2: Extraction

#### ExtractedEntity

| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| id | UUID | PK, default uuid() | Entity identifier |
| runId | String | FK → PipelineRun, NOT NULL | Pipeline run |
| caseId | String? | FK → Case | Associated case |
| ingestedFileId | String | FK → IngestedFile, NOT NULL | Source file |
| entityType | Enum(EntityType) | NOT NULL | Entity type |
| mentionText | String | NOT NULL | Mentioned text |
| confidenceScore | Float | NOT NULL | Confidence (0.0–1.0) |
| startOffset | Int? | | Start offset in source |
| endOffset | Int? | | End offset in source |
| epistemicCategory | Enum(EpistemicCategory) | default INFERENCE | Knowledge claim type |
| sourceReliability | Int | default 3 | Source reliability (1-5) |
| evidenceBasis | Int | default 1 | Evidence basis count |
| derivationDepth | Int | default 0 | Derivation depth |
| createdAt | DateTime | default now() | Creation timestamp |

**Relations:** run (N:1), ingestedFile (N:1), sourceRelations (1:N), targetRelations (1:N), candidates (1:N)

---

#### ExtractedRelation

| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| id | UUID | PK, default uuid() | Relation identifier |
| runId | String | FK → PipelineRun, NOT NULL | Pipeline run |
| caseId | String? | FK → Case | Associated case |
| ingestedFileId | String | FK → IngestedFile, NOT NULL | Source file |
| sourceEntityId | String | FK → ExtractedEntity, NOT NULL | Source entity |
| targetEntityId | String | FK → ExtractedEntity, NOT NULL | Target entity |
| relationType | Enum(RelationType) | NOT NULL | Relation type |
| confidenceScore | Float | NOT NULL | Confidence (0.0–1.0) |
| epistemicCategory | Enum(EpistemicCategory) | default INFERENCE | Knowledge claim type |
| sourceReliability | Int | default 3 | Source reliability (1-5) |
| evidenceBasis | Int | default 1 | Evidence basis count |
| derivationDepth | Int | default 0 | Derivation depth |
| createdAt | DateTime | default now() | Creation timestamp |

---

#### ExtractionLog

| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| id | UUID | PK, default uuid() | Log identifier |
| runId | String | FK → PipelineRun, NOT NULL | Pipeline run |
| ingestedFileId | String | FK → IngestedFile, NOT NULL | Source file |
| stage | String | NOT NULL | Pipeline stage |
| message | String? | | Log message |
| createdAt | DateTime | default now() | Creation timestamp |

---

### Pipeline Stage 3: Resolution

#### ResolvedEntity

| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| id | UUID | PK, default uuid() | Entity identifier |
| runId | String | FK → PipelineRun, NOT NULL | Pipeline run |
| caseId | String? | FK → Case | Associated case |
| entityType | Enum(EntityType) | NOT NULL | Entity type |
| canonicalName | String | NOT NULL | Canonical name |
| mergeConfidence | Float | NOT NULL | Merge confidence |
| mergeType | Enum(MergeType) | NOT NULL | Merge type |
| epistemicCategory | Enum(EpistemicCategory) | default INFERENCE | Knowledge claim type |
| sourceReliability | Int | default 3 | Source reliability |
| evidenceBasis | Int | default 1 | Evidence basis |
| derivationDepth | Int | default 0 | Derivation depth |
| phoneNumbers | Json? | | Phone numbers |
| plateNumbers | Json? | | Plate numbers |
| bankAccounts | Json? | | Bank accounts |
| locationNames | Json? | | Location names |
| createdAt | DateTime | default now() | Creation timestamp |

**Relations:** run (N:1), resolutionHistories (1:N), candidates (1:N)

---

#### EntityResolutionCandidate

| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| id | UUID | PK, default uuid() | Candidate identifier |
| runId | String | FK → PipelineRun, NOT NULL | Pipeline run |
| sourceEntityId | String | FK → ExtractedEntity, NOT NULL | Source entity |
| candidateEntityId | String | FK → ResolvedEntity, NOT NULL | Candidate entity |
| similarityScore | Float | NOT NULL | Similarity score |
| matchType | String | NOT NULL | Match type |
| status | Enum(CandidateStatus) | default PENDING | Candidate status |
| createdAt | DateTime | default now() | Creation timestamp |

---

#### Contradiction

| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| id | UUID | PK, default uuid() | Contradiction identifier |
| runId | String | FK → PipelineRun, NOT NULL | Pipeline run |
| contradictionType | Enum(ContradictionType) | NOT NULL | Contradiction type |
| severity | Enum(ContradictionSeverity) | NOT NULL | Severity |
| description | String | NOT NULL | Description |
| entities | Json | NOT NULL | Affected entities |
| isResolved | Boolean | default false | Is resolved |
| resolvedById | String? | FK → User | Who resolved |
| resolvedAt | DateTime? | | Resolution timestamp |
| createdAt | DateTime | default now() | Creation timestamp |

---

#### ResolutionHistory

| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| id | UUID | PK, default uuid() | History identifier |
| runId | String | FK → PipelineRun, NOT NULL | Pipeline run |
| entityId | String | FK → ResolvedEntity, NOT NULL | Entity |
| action | String | NOT NULL | Action taken |
| oldConfidence | Float? | | Previous confidence |
| newConfidence | Float? | | New confidence |
| changedById | String | FK → User, NOT NULL | Who changed |
| reason | String? | | Reason |
| contradictionId | String? | FK → Contradiction | Related contradiction |
| createdAt | DateTime | default now() | Creation timestamp |

---

### Pipeline Stage 4: Temporal

#### TemporalInfo

| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| id | UUID | PK, default uuid() | Info identifier |
| runId | String | FK → PipelineRun, NOT NULL | Pipeline run |
| entityId | String | NOT NULL | Entity identifier |
| timeExpression | String | NOT NULL | Original time expression |
| normalizedStart | DateTime? | | Normalized start |
| normalizedEnd | DateTime? | | Normalized end |
| granularity | String? | | Time granularity |
| confidence | Float | NOT NULL | Confidence |
| createdAt | DateTime | default now() | Creation timestamp |

---

#### SpatialInfo

| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| id | UUID | PK, default uuid() | Info identifier |
| runId | String | FK → PipelineRun, NOT NULL | Pipeline run |
| entityId | String | NOT NULL | Entity identifier |
| locationName | String? | | Location name |
| lat | Float? | | Latitude |
| lng | Float? | | Longitude |
| radius | Float? | | Radius |
| confidence | Float | NOT NULL | Confidence |
| createdAt | DateTime | default now() | Creation timestamp |

---

#### CoverageInterval

| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| id | UUID | PK, default uuid() | Interval identifier |
| runId | String | FK → PipelineRun, NOT NULL | Pipeline run |
| entityId | String | NOT NULL | Entity identifier |
| startYear | Int | NOT NULL | Start year |
| endYear | Int? | | End year |
| coveragePercent | Float | NOT NULL | Coverage percentage |
| createdAt | DateTime | default now() | Creation timestamp |

---

### Pipeline Stage 5: Graph

#### GraphNode

| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| id | UUID | PK, default uuid() | Node identifier |
| runId | String | FK → PipelineRun, NOT NULL | Pipeline run |
| caseId | String? | FK → Case | Associated case |
| nodeType | Enum(EntityType) | NOT NULL | Node type |
| canonicalName | String | NOT NULL | Canonical name |
| confidence | Float | NOT NULL | Confidence |
| resolutionStatus | Enum(GraphNodeResolutionStatus) | default UNRESOLVED | Resolution status |
| provenanceChain | Json? | | Provenance chain |
| createdAt | DateTime | default now() | Creation timestamp |

**Relations:** run (N:1), outgoingEdges (1:N), incomingEdges (1:N), missingOutgoing (1:N), missingIncoming (1:N), rejectedOutgoing (1:N), rejectedIncoming (1:N)

---

#### GraphEdge

| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| id | UUID | PK, default uuid() | Edge identifier |
| runId | String | FK → PipelineRun, NOT NULL | Pipeline run |
| caseId | String? | FK → Case | Associated case |
| sourceId | String | FK → GraphNode, NOT NULL | Source node |
| targetId | String | FK → GraphNode, NOT NULL | Target node |
| relationshipType | Enum(RelationType) | NOT NULL | Relationship type |
| semanticEdgeType | Enum(SemanticEdgeType) | NOT NULL | Semantic type |
| edgeType | String | NOT NULL | Edge type |
| confidenceScore | Float | NOT NULL | Confidence |
| adversarialScore | Float? | | Adversarial score |
| supportingEvidence | Json? | | Supporting evidence |
| createdAt | DateTime | default now() | Creation timestamp |

---

#### MissingEdge

| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| id | UUID | PK, default uuid() | Edge identifier |
| runId | String | FK → PipelineRun, NOT NULL | Pipeline run |
| sourceId | String | FK → GraphNode, NOT NULL | Source node |
| targetId | String | FK → GraphNode, NOT NULL | Target node |
| expectedType | String | NOT NULL | Expected type |
| confidence | Float | NOT NULL | Confidence |
| createdAt | DateTime | default now() | Creation timestamp |

---

#### RejectedEdge

| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| id | UUID | PK, default uuid() | Edge identifier |
| runId | String | FK → PipelineRun, NOT NULL | Pipeline run |
| sourceId | String | FK → GraphNode, NOT NULL | Source node |
| targetId | String | FK → GraphNode, NOT NULL | Target node |
| reason | String | NOT NULL | Rejection reason |
| confidence | Float | NOT NULL | Confidence |
| createdAt | DateTime | default now() | Creation timestamp |

---

#### AdversarialEdgeScore

| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| id | UUID | PK, default uuid() | Score identifier |
| runId | String | FK → PipelineRun, NOT NULL | Pipeline run |
| edgeId | String | FK → GraphEdge, NOT NULL | Edge |
| riskLevel | Enum(AdversarialRiskLevel) | NOT NULL | Risk level |
| score | Float | NOT NULL | Risk score |
| factors | Json? | | Risk factors |
| createdAt | DateTime | default now() | Creation timestamp |

---

#### EventCluster

| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| id | UUID | PK, default uuid() | Cluster identifier |
| runId | String | FK → PipelineRun, NOT NULL | Pipeline run |
| clusterName | String | NOT NULL | Cluster name |
| eventCount | Int | NOT NULL | Event count |
| avgConfidence | Float | NOT NULL | Average confidence |
| timeRange | Json? | | Time range |
| createdAt | DateTime | default now() | Creation timestamp |

---

### Knowledge Graph

#### PersonAlias

| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| id | UUID | PK, default uuid() | Alias identifier |
| personId | String | FK → Person, NOT NULL | Person |
| alias | String | NOT NULL | Alias |
| source | String? | | Source |

**Unique constraints:** `@@unique([personId, alias])`

---

#### Phone

| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| id | UUID | PK, default uuid() | Phone identifier |
| number | String | UNIQUE, NOT NULL | Phone number |
| personId | String? | FK → Person | Owner |
| carrier | String? | | Carrier |
| verifiedAt | DateTime? | | Verification timestamp |

---

#### Vehicle

| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| id | UUID | PK, default uuid() | Vehicle identifier |
| plateNumber | String | UNIQUE, NOT NULL | Plate number |
| personId | String? | FK → Person | Owner |
| type | String? | | Vehicle type |
| color | String? | | Color |

---

#### BankAccount

| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| id | UUID | PK, default uuid() | Account identifier |
| accountNumberEncrypted | String | NOT NULL | Encrypted account number |
| bankName | String | NOT NULL | Bank name |
| ifsc | String? | | IFSC code |
| personId | String? | FK → Person | Owner |

---

#### CaseEntityLink

| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| id | UUID | PK, default uuid() | Link identifier |
| caseId | String | FK → Case, NOT NULL | Case |
| personId | String | FK → Person, NOT NULL | Person |
| role | Enum(CaseEntityRole) | NOT NULL | Role in case |
| addedById | String | FK → User, NOT NULL | Who added |
| addedAt | DateTime | default now() | Addition timestamp |

**Unique constraints:** `@@unique([caseId, personId])`

---

### Evidence Management

#### Evidence

| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| id | UUID | PK, default uuid() | Evidence identifier |
| caseId | String | FK → Case, NOT NULL | Case |
| suspectId | String? | FK → Suspect | Suspect |
| type | String | NOT NULL | Evidence type |
| status | Enum(EvidenceStatus) | default LOGGED | Status |
| description | String? | | Description |
| collectedAt | DateTime | default now() | Collection timestamp |
| loggedById | String | FK → User, NOT NULL | Who logged |

---

#### EvidenceFile

| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| id | UUID | PK, default uuid() | File identifier |
| evidenceId | String | FK → Evidence, NOT NULL | Evidence |
| fileType | String | NOT NULL | File type |
| r2Key | String | NOT NULL | R2 storage key |
| mimeType | String? | | MIME type |
| originalFileName | String? | | Original filename |
| sizeBytes | Int? | | File size |
| uploadedById | String | FK → User, NOT NULL | Who uploaded |
| uploadedAt | DateTime | default now() | Upload timestamp |

---

#### OcrResult

| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| id | UUID | PK, default uuid() | Result identifier |
| evidenceFileId | String | FK → EvidenceFile, UNIQUE, NOT NULL | Source file |
| extractedText | String? | | Extracted text |
| confidence | Float? | | OCR confidence |
| language | String? | | Language |
| processedAt | DateTime | default now() | Processing timestamp |

---

#### AudioTranscript

| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| id | UUID | PK, default uuid() | Transcript identifier |
| evidenceFileId | String | FK → EvidenceFile, UNIQUE, NOT NULL | Source file |
| transcriptText | String? | | Transcript text |
| language | String? | | Language |
| durationSeconds | Int? | | Duration |
| confidence | Float? | | Confidence |
| processedAt | DateTime | default now() | Processing timestamp |

---

### Face Recognition

#### FaceEmbedding

| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| id | UUID | PK, default uuid() | Embedding identifier |
| personId | String? | FK → Person | Person |
| suspectId | String? | FK → Suspect | Suspect |
| evidenceFileId | String? | FK → EvidenceFile | Source file |
| sourceImageR2Key | String | NOT NULL | Source image R2 key |
| sourceType | String? | | Source type |
| embeddingVector | vector(512) | NOT NULL | 512-d embedding vector |
| detectorConfidence | Float | NOT NULL | Detector confidence |
| matchConfidence | Float? | | Match confidence |
| status | Enum(FaceStatus) | default PENDING | Status |
| confirmedById | String? | FK → User | Who confirmed |
| confirmedAt | DateTime? | | Confirmation timestamp |
| createdAt | DateTime | default now() | Creation timestamp |

**Indexes:** `@@index([personId, status])`, `@@index([suspectId])`, `@@index([evidenceFileId])`

---

### Geospatial

#### CrimeZoneScore

| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| id | UUID | PK, default uuid() | Score identifier |
| h3Index | String | NOT NULL | H3 hex index |
| riskScore | Float | NOT NULL | Risk score |
| band | String | NOT NULL | Risk band |
| offenseType | String? | | Offense type |
| jurisdictionNodeId | String | FK → JurisdictionNode, NOT NULL | Jurisdiction |
| computedAt | DateTime | default now() | Computation timestamp |

**Indexes:** `@@index([h3Index])`, `@@index([jurisdictionNodeId, computedAt])`

---

### Semantic Search

#### CaseNarrativeEmbedding

| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| id | UUID | PK, default uuid() | Embedding identifier |
| caseId | String | FK → Case, NOT NULL | Case |
| sourceText | String | NOT NULL | Source text |
| embeddingVector | vector(1536) | NOT NULL | 1536-d embedding vector |
| createdAt | DateTime | default now() | Creation timestamp |

**Indexes:** `@@index([caseId])`

---

### Audit

#### AuditLog

| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| id | UUID | PK, default uuid() | Log identifier |
| userId | String | FK → User, NOT NULL | User |
| action | String | NOT NULL | Action |
| resourceType | String | NOT NULL | Resource type |
| resourceId | String | NOT NULL | Resource ID |
| metadata | Json? | | Metadata |
| ipAddress | String? | | IP address |
| createdAt | DateTime | default now() | Creation timestamp |

**Indexes:** `@@index([userId, createdAt])`, `@@index([action])`, `@@index([resourceType, resourceId])`

---

### Jurisdiction Hierarchy

#### JurisdictionNode

| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| id | UUID | PK, default uuid() | Node identifier |
| name | String | NOT NULL | Node name |
| nodeType | Enum(JurisdictionNodeType) | NOT NULL | Node type |
| parentId | String? | FK → JurisdictionNode | Parent node |
| status | Enum(JurisdictionNodeStatus) | default ACTIVE | Node status |
| code | String? | | Official code |
| validFrom | DateTime | default now() | Valid from |
| validTo | DateTime? | | Valid to |
| createdAt | DateTime | default now() | Creation timestamp |
| updatedAt | DateTime? | | Last update timestamp |

**Relations:** parent (N:1), children (1:N), history (1:N), caseJurisdictionLinks (1:N), globalEntityLinks (1:N), unresolvedJurisdictions (1:N)

**Indexes:** `@@index([nodeType])`, `@@index([parentId])`, `@@index([status])`

---

#### GeographicJurisdiction

| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| id | UUID | PK, default uuid() | Identifier |
| name | String | NOT NULL | Name |
| jurisdictionType | String | NOT NULL | Type (state/district/city/zone/area) |
| parentId | String? | | Parent |
| boundaryPolygon | Json? | | GeoJSON polygon |
| centerLat | Float? | | Center latitude |
| centerLng | Float? | | Center longitude |
| validFrom | DateTime | default now() | Valid from |
| validTo | DateTime? | | Valid to |
| createdAt | DateTime | default now() | Creation timestamp |

**Indexes:** `@@index([jurisdictionType])`

---

#### UnresolvedJurisdiction

| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| id | UUID | PK, default uuid() | Identifier |
| rawText | String | NOT NULL | Raw parser text |
| sourceFile | String | NOT NULL | Source file |
| sourceType | String | NOT NULL | Source type |
| confidence | Float | default 0 | Parser confidence |
| suggestedJurisdictionNodeId | String? | FK → JurisdictionNode | Suggested match |
| resolvedById | String? | FK → User | Who resolved |
| resolvedJurisdictionNodeId | String? | | Resolved node |
| status | Enum(UnresolvedJurisdictionStatus) | default UNRESOLVED | Status |
| createdAt | DateTime | default now() | Creation timestamp |
| resolvedAt | DateTime? | | Resolution timestamp |

**Indexes:** `@@index([status])`, `@@index([sourceType])`

---

#### JurisdictionHistory

| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| id | UUID | PK, default uuid() | History identifier |
| jurisdictionNodeId | String | FK → JurisdictionNode, NOT NULL | Node |
| eventType | String | NOT NULL | Event type |
| eventDate | DateTime | default now() | Event date |
| previousState | Json? | | Previous state |
| newState | Json? | | New state |
| description | String? | | Description |
| recordedById | String? | FK → User | Who recorded |
| createdAt | DateTime | default now() | Creation timestamp |

**Indexes:** `@@index([jurisdictionNodeId])`, `@@index([eventType])`

---

#### CaseJurisdictionLink

| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| id | UUID | PK, default uuid() | Link identifier |
| caseId | String | FK → Case, NOT NULL | Case |
| jurisdictionNodeId | String | FK → JurisdictionNode, NOT NULL | Jurisdiction |
| linkType | Enum(CaseJurisdictionLinkType) | NOT NULL | Link type |
| description | String? | | Description |
| linkedAt | DateTime | default now() | Link timestamp |
| linkedById | String? | FK → User | Who linked |
| validFrom | DateTime | default now() | Valid from |
| validTo | DateTime? | | Valid to |

**Unique constraints:** `@@unique([caseId, jurisdictionNodeId, linkType])`

**Indexes:** `@@index([caseId])`, `@@index([jurisdictionNodeId])`, `@@index([linkType])`

---

### Global Entity Identity Index

#### GlobalEntity

| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| canonicalId | UUID | PK, default uuid() | Canonical identifier |
| entityType | String | NOT NULL | Entity type |
| canonicalName | String | NOT NULL | Canonical name |
| phones | Json? | | Phone numbers |
| accounts | Json? | | Account numbers |
| addresses | Json? | | Addresses |
| firstSeen | DateTime | default now() | First seen |
| lastSeen | DateTime | default now() | Last seen |
| totalCases | Int | default 0 | Total cases |
| totalJurisdictions | Int | default 0 | Total jurisdictions |
| createdAt | DateTime | default now() | Creation timestamp |
| updatedAt | DateTime | default now() | Last update |

**Relations:** links (1:N), crossCaseAlerts (1:N)

**Indexes:** `@@index([entityType])`, `@@index([canonicalName])`

---

#### GlobalEntityLink

| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| id | UUID | PK, default uuid() | Link identifier |
| globalEntityId | String | FK → GlobalEntity, NOT NULL | Global entity |
| caseId | String | FK → Case, NOT NULL | Case |
| localEntityId | String | NOT NULL | Local entity ID |
| jurisdictionNodeId | String | FK → JurisdictionNode, NOT NULL | Jurisdiction |
| confidence | Float | default 0 | Matching confidence |
| matchType | String | NOT NULL | Match type |
| createdAt | DateTime | default now() | Creation timestamp |

**Unique constraints:** `@@unique([globalEntityId, caseId, localEntityId])`

**Indexes:** `@@index([globalEntityId])`, `@@index([caseId])`, `@@index([jurisdictionNodeId])`

---

#### CrossCaseAlert

| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| id | UUID | PK, default uuid() | Alert identifier |
| globalEntityId | String | FK → GlobalEntity, NOT NULL | Global entity |
| severity | Enum(CrossCaseAlertSeverity) | default MEDIUM | Severity |
| recommendation | String? | | Recommendation |
| createdAt | DateTime | default now() | Creation timestamp |
| status | String | default "ACTIVE" | Status (ACTIVE/REVIEWED/DISMISSED) |
| reviewedById | String? | FK → User | Who reviewed |
| reviewedAt | DateTime? | | Review timestamp |

**Relations:** globalEntity (N:1), cases (1:N)

**Indexes:** `@@index([globalEntityId])`, `@@index([severity])`, `@@index([status])`

---

#### CrossCaseAlertCase

| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| id | UUID | PK, default uuid() | Junction identifier |
| alertId | String | FK → CrossCaseAlert, NOT NULL | Alert |
| caseId | String | FK → Case, NOT NULL | Case |
| jurisdictionNodeId | String? | FK → JurisdictionNode | Jurisdiction |

**Unique constraints:** `@@unique([alertId, caseId])`

**Indexes:** `@@index([alertId])`, `@@index([caseId])`

---

### CaseRelationship (NEW — Revision 3+)

#### CaseRelationship

| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| id | UUID | PK, default uuid() | Relationship identifier |
| caseAId | String | FK → Case, NOT NULL | Case A (smaller id) |
| caseBId | String | FK → Case, NOT NULL | Case B (larger id) |
| initialObservation | Enum(CaseRelationshipAssessment) | NOT NULL | First observation (immutable) |
| currentAssessment | Enum(CaseRelationshipAssessment) | NOT NULL | Current assessment |
| confidence | Float | default 0 | Confidence (0.0–1.0) |
| evidenceSummary | String? | | Evidence summary |
| detectedBy | String | default "SYSTEM" | SYSTEM or INVESTIGATOR |
| status | Enum(CaseRelationshipStatus) | default SUGGESTED | Status |
| confirmedById | String? | FK → User | Who confirmed |
| confirmedAt | DateTime? | | Confirmation timestamp |
| createdAt | DateTime | default now() | Creation timestamp |
| updatedAt | DateTime? | | Last update timestamp |

**Unique constraints:** `@@unique([caseAId, caseBId])`

**Relations:** caseA (N:1), caseB (N:1), confirmedBy (N:1), history (1:N)

**Indexes:** `@@index([caseAId])`, `@@index([caseBId])`, `@@index([currentAssessment])`, `@@index([status])`

**Semantics:**
- `initialObservation` is immutable — records what the system first detected
- `currentAssessment` evolves over time: SHARED_ENTITY → POSSIBLE_SAME_INCIDENT → SAME_INCIDENT
- Stage 11 generates SHARED_ENTITY observations; only investigators can confirm SAME_INCIDENT/RELATED_CASE
- CaseRelationshipHistory preserves the full evolution trail

---

#### CaseRelationshipHistory

| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| id | UUID | PK, default uuid() | History identifier |
| caseRelationshipId | String | FK → CaseRelationship, NOT NULL | Relationship |
| previousAssessment | Enum(CaseRelationshipAssessment)? | | Previous assessment (null for initial) |
| newAssessment | Enum(CaseRelationshipAssessment) | NOT NULL | New assessment |
| previousStatus | Enum(CaseRelationshipStatus)? | | Previous status |
| newStatus | Enum(CaseRelationshipStatus) | NOT NULL | New status |
| previousConfidence | Float? | | Previous confidence |
| newConfidence | Float | NOT NULL | New confidence |
| changedById | String? | FK → User | Who changed (null for system) |
| changeReason | String? | | Reason for change |
| createdAt | DateTime | default now() | Creation timestamp |

**Indexes:** `@@index([caseRelationshipId])`, `@@index([createdAt])`

---

### CaseAccess (NEW — Revision 3+)

#### CaseAccess

| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| id | UUID | PK, default uuid() | Access identifier |
| userId | String | FK → User, NOT NULL | User |
| caseId | String | FK → Case, NOT NULL | Case |
| accessLevel | Enum(CaseAccessLevel) | default READ | Authorization level |
| grantedById | String | FK → User, NOT NULL | Who granted |
| grantedAt | DateTime | default now() | Grant timestamp |
| expiresAt | DateTime? | | Expiration timestamp |

**Unique constraints:** `@@unique([userId, caseId])`

**Relations:** user (N:1), case (N:1), grantedBy (N:1)

**Indexes:** `@@index([userId])`, `@@index([caseId])`, `@@index([accessLevel])`

**Semantics:**
- Determines whether a user is permitted to access a Case
- Separate from InvestigationWorkspace (workspace organizes work, CaseAccess controls access)
- When a Case is created, the creator gets ADMIN access
- Senior officers/admins can grant access to others
- FIR lifecycle state does NOT revoke CaseAccess

---

### InvestigationWorkspace (NEW — Revision 3+)

#### InvestigationWorkspace

| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| id | UUID | PK, default uuid() | Workspace identifier |
| name | String | NOT NULL | Workspace name |
| description | String? | | Description |
| createdById | String | FK → User, NOT NULL | Creator |
| jurisdictionNodeId | String? | FK → JurisdictionNode | Jurisdiction |
| status | Enum(WorkspaceStatus) | default ACTIVE | Status |
| createdAt | DateTime | default now() | Creation timestamp |
| updatedAt | DateTime? | | Last update timestamp |

**Relations:** createdBy (N:1), jurisdictionNode (N:1), cases (1:N), members (1:N)

---

#### WorkspaceCase

| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| id | UUID | PK, default uuid() | Junction identifier |
| workspaceId | String | FK → InvestigationWorkspace, NOT NULL | Workspace |
| caseId | String | FK → Case, NOT NULL | Case |
| addedById | String | FK → User, NOT NULL | Who added |
| addedAt | DateTime | default now() | Addition timestamp |

**Unique constraints:** `@@unique([workspaceId, caseId])`

**Relations:** workspace (N:1), case (N:1), addedBy (N:1)

**Indexes:** `@@index([workspaceId])`, `@@index([caseId])`

**Semantics:**
- Cases are not owned by the workspace
- The same Case may exist in multiple workspaces
- Workspace membership does NOT grant Case access

---

#### WorkspaceMember

| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| id | UUID | PK, default uuid() | Member identifier |
| workspaceId | String | FK → InvestigationWorkspace, NOT NULL | Workspace |
| userId | String | FK → User, NOT NULL | User |
| role | Enum(WorkspaceRole) | default MEMBER | Role |
| joinedAt | DateTime | default now() | Join timestamp |

**Unique constraints:** `@@unique([workspaceId, userId])`

**Relations:** workspace (N:1), user (N:1)

**Indexes:** `@@index([workspaceId])`, `@@index([userId])`

---

### AnalysisRun (NEW — Revision 3+)

#### AnalysisRun

| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| id | UUID | PK, default uuid() | Run identifier |
| workspaceId | String? | FK → InvestigationWorkspace | Workspace |
| createdById | String | FK → User, NOT NULL | Creator |
| scopeType | Enum(AnalysisRunScopeType) | default LOCAL | Scope (LOCAL/SCOPED) |
| description | String? | | Description |
| pipelineVersion | String? | | Pipeline version |
| modelVersions | Json? | | Model versions |
| status | Enum(AnalysisRunStatus) | default RUNNING | Status |
| startedAt | DateTime | default now() | Start timestamp |
| completedAt | DateTime? | | Completion timestamp |
| createdAt | DateTime | default now() | Creation timestamp |

**Relations:** workspace (N:1), createdBy (N:1), cases (1:N), result (1:1), findings (1:N)

**Indexes:** `@@index([workspaceId])`, `@@index([createdById])`, `@@index([status])`

**Semantics:**
- Immutable historical analytical execution
- Separate from PipelineRun (which tracks Case processing)
- When scope_type = LOCAL (single Case), AnalysisRun still creates a persistent record
- A completed AnalysisRun must never silently change

---

#### AnalysisRunCase

| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| id | UUID | PK, default uuid() | Junction identifier |
| analysisRunId | String | FK → AnalysisRun, NOT NULL | Analysis run |
| caseId | String | FK → Case, NOT NULL | Case |

**Unique constraints:** `@@unique([analysisRunId, caseId])`

**Relations:** analysisRun (N:1), case (N:1)

**Indexes:** `@@index([analysisRunId])`, `@@index([caseId])`

**Semantics:**
- Authoritative relational representation of which Cases were analyzed
- The exact Case scope must not silently change later

---

#### AnalysisRunResult

| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| id | UUID | PK, default uuid() | Result identifier |
| analysisRunId | String | FK → AnalysisRun, UNIQUE, NOT NULL | Analysis run |
| **caseRelationshipsSnapshot** | Json? | | CaseRelationship records at execution time |
| **globalEntityLinksSnapshot** | Json? | | GlobalEntityLink records at execution time |
| mergedGraphNodes | Json? | | Merged graph nodes |
| mergedGraphEdges | Json? | | Merged graph edges |
| analytics | Json? | | Analytics outputs |
| mlOutputs | Json? | | ML outputs |
| createdAt | DateTime | default now() | Creation timestamp |

**Relations:** analysisRun (N:1)

**Semantics:**
- Input snapshots are written once at completion, never updated
- Derived outputs are NOT authoritative source for Case/FIR/Entity data
- A completed AnalysisRun's meaning is fixed because its input snapshot is immutable
- Later changes to CaseRelationship or GlobalEntityLink affect future analyses, not completed ones

---

### Finding (NEW — Revision 3+)

#### Finding

| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| id | UUID | PK, default uuid() | Finding identifier |
| analysisRunId | String | FK → AnalysisRun, NOT NULL | Analysis run |
| findingType | Enum(FindingType) | NOT NULL | Finding type |
| title | String | NOT NULL | Title |
| description | String? | | Description |
| confidence | Float | default 0 | Confidence (0.0–1.0) |
| supportingCaseIds | Json? | | Supporting case IDs |
| supportingEntityIds | Json? | | Supporting entity IDs |
| supportingEdgeDescriptions | Json? | | Supporting edge descriptions |
| supportingEvidenceIds | Json? | | Supporting evidence IDs |
| criticStatus | Enum(CriticStatus) | default PENDING | Critic evaluation |
| criticNotes | String? | | Critic notes |
| criticConfidence | Float? | | Critic confidence |
| createdById | String? | FK → User | Who created (null for system) |
| createdAt | DateTime | default now() | Creation timestamp |
| updatedAt | DateTime? | | Last update timestamp |

**Relations:** analysisRun (N:1), createdBy (N:1)

**Indexes:** `@@index([analysisRunId])`, `@@index([findingType])`, `@@index([criticStatus])`

**Semantics:**
- Persistent objects associated with an AnalysisRun
- System-generated findings (created_by_id = null): HYPOTHESIS, ANOMALY, PATTERN
- Investigator-created findings have created_by_id set
- Critic evaluates persisted AnalysisRun/findings, not a temporary session
- Critic evaluation does not delete or modify the original finding

---

### FIR Lifecycle History (NEW — Revision 3+)

#### FIRLifecycleHistory

| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| id | UUID | PK, default uuid() | History identifier |
| firId | String | FK → FIR, NOT NULL | FIR |
| dimension | Enum(FIRLifecycleDimension) | NOT NULL | Lifecycle dimension |
| previousValue | String? | | Previous value (null for initial) |
| newValue | String | NOT NULL | New value |
| changedById | String | FK → User, NOT NULL | Who changed |
| changedAt | DateTime | default now() | Change timestamp |
| reason | String? | | Reason |
| authorityReference | String? | | Authority/order reference |
| createdAt | DateTime | default now() | Creation timestamp |

**Relations:** fir (N:1), changedBy (N:1)

**Indexes:** `@@index([firId])`, `@@index([dimension])`, `@@index([changedAt])`

**Semantics:**
- Each lifecycle transition generates a FIRLifecycleHistory record
- FIRLifecycleHistory is domain-specific lifecycle history
- AuditLog is system-wide security/operations audit
- Both are generated for lifecycle transitions (different query purposes)
- Lifecycle state affects permissions/presentation/allowed operations, not historical existence

---

## Relationship Summary

```
Case ──1:1── FIR (FIR.caseId unique)
Case ──1:N── Evidence
Case ──1:N── CaseEntityLink ──N:1── Person
Case ──N:M── JurisdictionNode (via CaseJurisdictionLink)
Case ──N:M── Case (via CaseRelationship, self-referential)
Case ──N:M── InvestigationWorkspace (via WorkspaceCase)
Case ──N:M── AnalysisRun (via AnalysisRunCase)
Case ──N:M── User (via CaseAccess)
FIR ──1:N── FIRLifecycleHistory
User ──N:M── InvestigationWorkspace (via WorkspaceMember)
AnalysisRun ──1:1── AnalysisRunResult
AnalysisRun ──1:N── Finding
GlobalEntity ──1:N── GlobalEntityLink
GlobalEntity ──1:N── CrossCaseAlert (via CrossCaseAlertCase)
```

---

## Authorization Model

| Mechanism | Purpose | Scope |
|-----------|---------|-------|
| CaseAccess | Who may access which Cases | Per user × Case |
| WorkspaceMember | Who is in which Workspace | Per user × Workspace |
| FIR lifecycle | Controlled state transitions | Per FIR × dimension |
| UserRole | System-level permissions | Per user |

**Key distinction:** Workspace membership organizes work. CaseAccess controls access. These are independent.

---

## Lifecycle Semantics

### FIR Lifecycle

Three independent dimensions, each with its own history:

```
investigation_status: REGISTERED → UNDER_INVESTIGATION → CHARGE_SHEETED → CLOSED
legal_disposition:    NONE → CANCELLED/QUASHED/DISMISSED/CONVICTED/ACQUITTED
record_status:        ACTIVE → ARCHIVED/SEALED
```

A lifecycle transition generates:
1. FIRLifecycleHistory record (domain-specific)
2. AuditLog entry (system-wide)

### CaseRelationship Evolution

```
SHARED_ENTITY (system observation)
    ↓
POSSIBLE_SAME_INCIDENT (system suggestion)
    ↓
SAME_INCIDENT (investigator confirmation)
```

The original observation is never lost. CaseRelationshipHistory preserves every transition.

### AnalysisRun Immutability

- Input snapshots (case_relationships_snapshot, global_entity_links_snapshot) written once at completion
- Later changes to CaseRelationship or GlobalEntityLink affect future analyses, not completed ones
- AnalysisRunResult is a derived historical artifact, not authoritative source

---

## Persistence Layers

### Authoritative Domain Data
FIR, Case, Evidence, Jurisdiction, Entities, CaseRelationship, CaseAccess, FIR Lifecycle History

### Investigation Context
InvestigationWorkspace, WorkspaceCase, WorkspaceMember

### Analytical Execution
PipelineRun, AnalysisRun, AnalysisRunCase, AnalysisRunResult

### Interpretation
Finding

### Audit
AuditLog, FIRLifecycleHistory, CaseRelationshipHistory
