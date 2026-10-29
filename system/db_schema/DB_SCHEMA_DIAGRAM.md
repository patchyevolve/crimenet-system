# ============================================================
# ENTITY-RELATIONSHIP DIAGRAM
# AI-Powered Criminal Network Analysis System — Pipeline Schema v3
# Derived from: src/models/schema.py (pipeline v2)
# ============================================================

```mermaid
erDiagram
    %% ============================================================
    %% CORE TABLES
    %% ============================================================
    PipelineRun {
        varchar id PK
        varchar case_id FK
        varchar pipeline_version
        jsonb model_versions
        varchar policy_version
        varchar trigger_type
        varchar status "RUNNING|COMPLETED|FAILED|PARTIAL"
        timestamp start_time
        timestamp end_time
        varchar parent_run_id FK
        jsonb input_files
        jsonb summary
    }

    Case {
        varchar id PK
        varchar police_station_id FK
        varchar location_id FK
        varchar case_status "ACTIVE|PENDING|SOLVED|CLOSED"
        varchar fir_number UK
        text fir_description
        timestamp created_at
        timestamp updated_at
    }

    FIR {
        varchar id PK
        varchar case_id FK
        varchar fir_number UK
        varchar status "REGISTERED|UNDER_INVESTIGATION|CHARGE_SHEET_FILED|QUASHED"
        text description
        varchar filed_by_id FK
        varchar fir_document_r2_key
        timestamp created_at
        timestamp updated_at
    }

    PoliceStation {
        varchar id PK
        varchar name
        varchar district
    }

    User {
        varchar id PK
        varchar email UK
        varchar name
        varchar password_hash
        varchar role "INSPECTOR|ADMIN|AUDIT_LOGGER"
        varchar status "ACTIVE|INACTIVE|FORCE_PASSWORD_CHANGE"
        varchar police_station_id FK
        timestamp created_at
        timestamp updated_at
    }

    Location {
        varchar id PK
        varchar name
        float lat
        float lng
        varchar address
        varchar police_station_id FK
    }

    %% ============================================================
    %% STAGE 1: INGESTION
    %% ============================================================
    IngestedFile {
        varchar id PK
        varchar run_id FK
        varchar file_name
        varchar file_path
        varchar file_ext
        varchar file_hash
        bigint file_size_bytes
        varchar detected_type
        varchar source_type "CDR|BANK|FIR|CCTV|SOCIAL|DEVICE|TEXT"
        varchar document_type
        timestamp ingestion_time
        int record_count
        text raw_summary
    }

    DataQualityScore {
        varchar id PK
        varchar ingested_file_id FK
        float quality_score
        float completeness_score
        float structural_consistency
        float content_richness
        jsonb issues
        varchar run_id FK
    }

    AdversarialCheck {
        varchar id PK
        varchar ingested_file_id FK
        boolean is_suspicious
        boolean behavioral_anomaly
        boolean temporal_anomaly
        boolean content_anomaly
        boolean duplicate_suspect
        varchar risk_level "LOW|MEDIUM|HIGH|CRITICAL"
        float score
        jsonb reasons
        varchar run_id FK
    }

    EvidenceIntegrity {
        varchar id PK
        varchar ingested_file_id FK UK
        varchar file_hash
        boolean hash_verified
        boolean is_adversarial
        jsonb chain_of_custody
        jsonb transformations
        timestamp created_at
    }

    DependencyGroup {
        varchar id PK
        varchar run_id FK
        varchar group_name
        int file_count
        int total_entities
        int total_relations
        timestamp created_at
    }

    %% ============================================================
    %% STAGE 2: EXTRACTION
    %% ============================================================
    ExtractedEntity {
        varchar id PK
        varchar ingested_file_id FK
        varchar run_id FK
        varchar entity_type "PERSON|PHONE|VEHICLE|LOCATION|ACCOUNT|ORGANIZATION|EVENT|DEVICE|AMOUNT|DATE|DOCUMENT|EMAIL|VEHICLE_PLATE|IP_ADDRESS|CASE_NUMBER|UNKNOWN"
        varchar mention_text
        float confidence_score
        int start_offset
        int end_offset
        varchar epistemic_category "OBSERVATION|INFERENCE|HYPOTHESIS|UNKNOWN_PROVENANCE"
        int source_reliability
        int evidence_basis
        int derivation_depth
        timestamp created_at
    }

    ExtractedRelation {
        varchar id PK
        varchar ingested_file_id FK
        varchar run_id FK
        varchar source_entity_id FK
        varchar target_entity_id FK
        varchar relation_type "CALLED|MESSED|ASSOCIATED_WITH|TRANSFERRED_TO|RECEIVED_FROM|OWNS_ACCOUNT|FAMILY_OF|FRIEND_OF|ASSOCIATE_OF|WORKS_WITH|LIVES_AT|WORKS_AT|VISITED|SUSPECT_OF|VICTIM_OF|WITNESS_OF|MEMBER_OF|EXTRACTED_FROM|DERIVED_FROM|POSSIBLE_IDENTITY|SAME_AS"
        float confidence_score
        varchar epistemic_category "OBSERVATION|INFERENCE|HYPOTHESIS|UNKNOWN_PROVENANCE"
        int source_reliability
        int evidence_basis
        int derivation_depth
        timestamp created_at
    }

    ExtractionLog {
        varchar id PK
        varchar run_id FK
        varchar ingested_file_id FK
        varchar stage
        text message
        timestamp created_at
    }

    %% ============================================================
    %% STAGE 3: RESOLUTION
    %% ============================================================
    ResolvedEntity {
        varchar id PK
        varchar run_id FK
        varchar entity_type "PERSON|PHONE|VEHICLE|LOCATION|ACCOUNT|ORGANIZATION|EVENT|DEVICE|AMOUNT|DATE|DOCUMENT|EMAIL|VEHICLE_PLATE|IP_ADDRESS|CASE_NUMBER|UNKNOWN"
        varchar canonical_name
        float merge_confidence
        varchar merge_type "AUTO|REVIEW|REJECT|SINGLE|LLM"
        varchar epistemic_category "OBSERVATION|INFERENCE|HYPOTHESIS|UNKNOWN_PROVENANCE"
        int source_reliability
        int evidence_basis
        int derivation_depth
        jsonb phone_numbers
        jsonb plate_numbers
        jsonb bank_accounts
        jsonb location_names
        timestamp created_at
    }

    EntityResolutionCandidate {
        varchar id PK
        varchar run_id FK
        varchar source_entity_id FK
        varchar candidate_entity_id FK
        float similarity_score
        varchar match_type
        varchar status "PENDING|CONFIRMED|REJECTED"
        timestamp created_at
    }

    Contradiction {
        varchar id PK
        varchar run_id FK
        varchar contradiction_type "IDENTITY|TEMPORAL|LOCATION|ATTRIBUTION|SOURCE_CONTENT|EVENT_IDENTITY|DERIVATION|UNKNOWN"
        varchar severity "LOW|MEDIUM|HIGH|CRITICAL"
        text description
        jsonb entities
        boolean is_resolved
        varchar resolved_by_id FK
        timestamp resolved_at
        timestamp created_at
    }

    ResolutionHistory {
        varchar id PK
        varchar run_id FK
        varchar entity_id FK
        varchar action
        float old_confidence
        float new_confidence
        varchar changed_by_id FK
        text reason
        varchar contradiction_id FK
        timestamp created_at
    }

    %% ============================================================
    %% STAGE 4: TEMPORAL
    %% ============================================================
    TemporalInfo {
        varchar id PK
        varchar run_id FK
        varchar entity_id
        varchar time_expression
        timestamp normalized_start
        timestamp normalized_end
        varchar granularity
        float confidence
        timestamp created_at
    }

    SpatialInfo {
        varchar id PK
        varchar run_id FK
        varchar entity_id
        varchar location_name
        float lat
        float lng
        float radius
        float confidence
        timestamp created_at
    }

    CoverageInterval {
        varchar id PK
        varchar run_id FK
        varchar entity_id
        int start_year
        int end_year
        float coverage_percent
        timestamp created_at
    }

    %% ============================================================
    %% STAGE 5: GRAPH
    %% ============================================================
    GraphNode {
        varchar id PK
        varchar run_id FK
        varchar node_type "PERSON|PHONE|VEHICLE|LOCATION|ACCOUNT|ORGANIZATION|EVENT|DEVICE|AMOUNT|DATE|DOCUMENT|EMAIL|VEHICLE_PLATE|IP_ADDRESS|CASE_NUMBER|UNKNOWN"
        varchar canonical_name
        float confidence
        varchar resolution_status "RESOLVED|STANDALONE|UNRESOLVED"
        jsonb provenance_chain
        timestamp created_at
    }

    GraphEdge {
        varchar id PK
        varchar run_id FK
        varchar source_id FK
        varchar target_id FK
        varchar relationship_type "CALLED|MESSED|ASSOCIATED_WITH|TRANSFERRED_TO|RECEIVED_FROM|OWNS_ACCOUNT|FAMILY_OF|FRIEND_OF|ASSOCIATE_OF|WORKS_WITH|LIVES_AT|WORKS_AT|VISITED|SUSPECT_OF|VICTIM_OF|WITNESS_OF|MEMBER_OF|EXTRACTED_FROM|DERIVED_FROM|POSSIBLE_IDENTITY|SAME_AS"
        varchar semantic_edge_type "ENTREPRENEURIAL|ASSOCIATIONAL|QUASI_GOVERNMENTAL"
        varchar edge_type
        float confidence_score
        float adversarial_score
        jsonb supporting_evidence
        timestamp created_at
    }

    MissingEdge {
        varchar id PK
        varchar run_id FK
        varchar source_id FK
        varchar target_id FK
        varchar expected_type
        float confidence
        timestamp created_at
    }

    RejectedEdge {
        varchar id PK
        varchar run_id FK
        varchar source_id FK
        varchar target_id FK
        text reason
        float confidence
        timestamp created_at
    }

    AdversarialEdgeScore {
        varchar id PK
        varchar run_id FK
        varchar edge_id FK
        varchar risk_level "LOW|MEDIUM|HIGH|CRITICAL"
        float score
        jsonb factors
        timestamp created_at
    }

    EventCluster {
        varchar id PK
        varchar run_id FK
        varchar cluster_name
        int event_count
        float avg_confidence
        jsonb time_range
        timestamp created_at
    }

    %% ============================================================
    %% KNOWLEDGE GRAPH ENTITIES
    %% ============================================================
    Person {
        varchar id PK
        varchar canonical_name
        varchar gender "MALE|FEMALE|OTHER"
        timestamp dob
        varchar national_id_number
        varchar photo_r2_key
        varchar identity_source "MANUAL|FIR_INFERRED|FACE_MATCH_CONFIRMED"
        float name_confidence
        varchar created_by_id FK
        timestamp created_at
    }

    PersonAlias {
        varchar id PK
        varchar person_id FK
        varchar alias
        varchar source
    }

    Phone {
        varchar id PK
        varchar number
        varchar person_id FK
        varchar carrier
        timestamp verified_at
    }

    Vehicle {
        varchar id PK
        varchar plate_number
        varchar person_id FK
        varchar type
        varchar color
    }

    BankAccount {
        varchar id PK
        varchar account_number_encrypted
        varchar bank_name
        varchar ifsc
        varchar person_id FK
    }

    CaseEntityLink {
        varchar id PK
        varchar case_id FK
        varchar person_id FK
        varchar role "SUSPECT|VICTIM|WITNESS|ASSOCIATE|VEHICLE_OWNER"
        varchar added_by_id FK
        timestamp added_at
    }

    %% ============================================================
    %% EVIDENCE MANAGEMENT
    %% ============================================================
    Evidence {
        varchar id PK
        varchar case_id FK
        varchar suspect_id FK
        varchar type "PHYSICAL|DIGITAL|DOCUMENTARY|BIOLOGICAL|TESTIMONIAL"
        varchar status "LOGGED|COLLECTED|ANALYZED|ADMITTED|DISMISSED"
        text description
        timestamp collected_at
        varchar logged_by_id FK
    }

    EvidenceFile {
        varchar id PK
        varchar evidence_id FK
        varchar file_type "AUDIO|IMAGE|DOCUMENT|VIDEO|CSV"
        varchar r2_key
        varchar mime_type
        varchar original_file_name
        bigint size_bytes
        varchar uploaded_by_id FK
        timestamp uploaded_at
    }

    OcrResult {
        varchar id PK
        varchar evidence_file_id FK UK
        text extracted_text
        float confidence
        varchar language
        timestamp processed_at
    }

    AudioTranscript {
        varchar id PK
        varchar evidence_file_id FK UK
        text transcript_text
        varchar language
        int duration_seconds
        float confidence
        timestamp processed_at
    }

    %% ============================================================
    %% SUSPECT LIFECYCLE
    %% ============================================================
    Suspect {
        varchar id PK
        varchar case_id FK
        varchar person_id FK
        varchar name
        varchar gender "MALE|FEMALE|OTHER"
        timestamp dob
        varchar national_id_number
        varchar status "UNDER_INVESTIGATION|ARRESTED|CHARGED|CONVICTED|ACQUITTED|RELEASED"
        timestamp arrest_date
        varchar contact_info
        varchar photo_r2_key
        varchar criminal_record_number UK
        varchar convicted_by_id FK
        timestamp convicted_at
        text notes
        timestamp created_at
        timestamp updated_at
    }

    %% ============================================================
    %% FACE RECOGNITION
    %% ============================================================
    FaceEmbedding {
        varchar id PK
        varchar person_id FK
        varchar suspect_id FK
        varchar evidence_file_id FK
        varchar source_image_r2_key
        varchar source_type "FIR_ANNEXURE|CCTV_STILL|SOCIAL_MEDIA|SUSPECT_BOOKING_PHOTO"
        vector512 embedding_vector
        float detector_confidence
        float match_confidence
        varchar status "PENDING|EXTRACTED|MATCHED|CONFIRMED|REJECTED|LOW_QUALITY"
        varchar confirmed_by_id FK
        timestamp confirmed_at
        timestamp created_at
    }

    %% ============================================================
    %% GEOSPATIAL RISK
    %% ============================================================
    CrimeZoneScore {
        varchar id PK
        varchar h3_index
        float risk_score
        varchar band "GREEN|AMBER|RED"
        varchar offense_type
        varchar police_station_id FK
        timestamp computed_at
    }

    %% ============================================================
    %% SEMANTIC SEARCH
    %% ============================================================
    CaseNarrativeEmbedding {
        varchar id PK
        varchar case_id FK
        text source_text
        vector1536 embedding_vector
        timestamp created_at
    }

    %% ============================================================
    %% AUDIT LOG
    %% ============================================================
    AuditLog {
        varchar id PK
        varchar user_id FK
        varchar action "CREATE|UPDATE|DELETE|READ_SENSITIVE|EXPORT"
        varchar resource_type
        varchar resource_id
        jsonb metadata
        varchar ip_address
        timestamp created_at
    }

    %% ============================================================
    %% RELATIONSHIPS
    %% ============================================================

    %% Pipeline Run → all stages
    PipelineRun ||--o{ IngestedFile : "ingests files"
    PipelineRun ||--o{ DataQualityScore : "assesses quality"
    PipelineRun ||--o{ AdversarialCheck : "checks adversarial"
    PipelineRun ||--o{ DependencyGroup : "groups dependencies"
    PipelineRun ||--o{ ExtractionLog : "logs extraction"
    PipelineRun ||--o{ ResolvedEntity : "resolves entities"
    PipelineRun ||--o{ EntityResolutionCandidate : "resolves candidates"
    PipelineRun ||--o{ Contradiction : "finds contradictions"
    PipelineRun ||--o{ ResolutionHistory : "tracks history"
    PipelineRun ||--o{ TemporalInfo : "extracts temporal"
    PipelineRun ||--o{ SpatialInfo : "extracts spatial"
    PipelineRun ||--o{ CoverageInterval : "computes coverage"
    PipelineRun ||--o{ GraphNode : "builds nodes"
    PipelineRun ||--o{ GraphEdge : "builds edges"
    PipelineRun ||--o{ MissingEdge : "detects missing"
    PipelineRun ||--o{ RejectedEdge : "rejects edges"
    PipelineRun ||--o{ AdversarialEdgeScore : "scores adversarial"
    PipelineRun ||--o{ EventCluster : "clusters events"

    %% Core
    PoliceStation ||--o{ User : "employs"
    PoliceStation ||--o{ Case : "investigates"
    PoliceStation ||--o{ Location : "jurisdiction"
    PoliceStation ||--o{ CrimeZoneScore : "monitors zones"
    Location ||--o{ Case : "incident location"

    %% FIR
    Case ||--o{ FIR : "has FIRs"
    User ||--o{ FIR : "files FIRs"

    %% Ingestion
    IngestedFile ||--o{ DataQualityScore : "has quality"
    IngestedFile ||--o{ AdversarialCheck : "has adversarial"
    IngestedFile ||--o{ EvidenceIntegrity : "has integrity"
    IngestedFile ||--o{ ExtractionLog : "logged by"
    IngestedFile ||--o{ ExtractedEntity : "produces entities"
    IngestedFile ||--o{ ExtractedRelation : "produces relations"

    %% Extraction
    ExtractedEntity ||--o{ ExtractedRelation : "source of"
    ExtractedEntity ||--o{ ExtractedRelation : "target of"

    %% Resolution
    ExtractedEntity ||--o{ EntityResolutionCandidate : "candidate for"
    ResolvedEntity ||--o{ EntityResolutionCandidate : "resolves to"
    ResolvedEntity ||--o{ ResolutionHistory : "tracked in"

    %% Contradiction → ResolutionHistory
    Contradiction ||--o{ ResolutionHistory : "linked to"

    %% Graph
    GraphNode ||--o{ GraphEdge : "source of"
    GraphNode ||--o{ GraphEdge : "target of"
    GraphNode ||--o{ MissingEdge : "source of"
    GraphNode ||--o{ MissingEdge : "target of"
    GraphNode ||--o{ RejectedEdge : "source of"
    GraphNode ||--o{ RejectedEdge : "target of"
    GraphEdge ||--o{ AdversarialEdgeScore : "scored for adversarial"

    %% Knowledge Graph
    Person ||--o{ PersonAlias : "has aliases"
    Person ||--o{ Phone : "owns"
    Person ||--o{ Vehicle : "owns"
    Person ||--o{ BankAccount : "owns"
    Person ||--o{ CaseEntityLink : "linked to cases"
    Person ||--o{ FaceEmbedding : "has embeddings"
    Person ||--o{ Suspect : "resolved from"
    Case ||--o{ CaseEntityLink : "has entities"
    Case ||--o{ Evidence : "has evidence"
    Case ||--o{ Suspect : "has suspects"
    Case ||--o{ CaseNarrativeEmbedding : "has narratives"
    User ||--o{ CaseEntityLink : "adds links"

    %% Evidence
    Evidence ||--o{ EvidenceFile : "has files"
    EvidenceFile ||--o{ OcrResult : "ocr output"
    EvidenceFile ||--o{ AudioTranscript : "transcript output"
    EvidenceFile ||--o{ FaceEmbedding : "source of face"
    User ||--o{ Evidence : "logs evidence"
    User ||--o{ EvidenceFile : "uploads files"

    %% Suspect
    Suspect ||--o{ Evidence : "implicated by"
    Suspect ||--o{ FaceEmbedding : "has face"
    User ||--o{ Suspect : "convicts"

    %% Face Recognition
    User ||--o{ FaceEmbedding : "confirms matches"

    %% Audit
    User ||--o{ AuditLog : "performs actions"
```