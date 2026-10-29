# CrimeNet

**AI-powered criminal network analysis for investigators.**
Turn FIRs, call records, bank statements and photos into a living knowledge graph, find the people who hold a network together, and keep every step explainable, audited and tamper-evident.

> Built for **SIH26189** (Ministry of Home Affairs, Blockchain & Cybersecurity theme).
> CrimeNet is an investigative *lead-generation* tool. It never decides guilt, never auto-merges identities, and never writes AI suggestions into a case without a human's approval.

---

## Table of contents

1. [What it does](#what-it-does)
2. [Key ideas](#key-ideas)
3. [Architecture](#architecture)
4. [Tech stack](#tech-stack)
5. [Repository layout](#repository-layout)
6. [Quick start](#quick-start)
7. [Configuration](#configuration)
8. [First run: create the admin, load data](#first-run-create-the-admin-load-data)
9. [Using the platform](#using-the-platform)
10. [Security and privacy model](#security-and-privacy-model)
11. [Testing and CI](#testing-and-ci)
12. [Project status and known gaps](#project-status-and-known-gaps)
13. [Roadmap](#roadmap)
14. [Contributing](#contributing)
15. [Legal and ethics notice](#legal-and-ethics-notice)

---

## What it does

CrimeNet is a link-analysis and criminal-intelligence platform in the same category as IBM i2 Analyst's Notebook, Palantir Gotham and Cognyte NEXYTE. It works as a **retrospective analysis system over accumulated case data**, not a real-time stream.

| Step | What happens |
|---|---|
| **Ingest** | FIR PDFs/scans, CDR and bank-statement CSV/XLSX, surveillance notes, social-media exports and photos are uploaded, hashed and stored. |
| **Extract** | OCR and NLP (NER and relation extraction, with Hindi/Gujarati/Marathi support) pull out people, phones, plates, accounts, locations and events. |
| **Resolve** | Fuzzy, phonetic and cross-script matching proposes that "Rahul", "Rahool" and "राहुल" may be one person. An investigator confirms or rejects. |
| **Graph** | Confirmed entities and confidence-scored relationships land in a Neo4j knowledge graph. |
| **Analyse** | Centrality (key players), community detection (gangs/cells), link prediction (hidden ties), statistical anomaly detection (bursts, structuring) and H3 geospatial risk zones. |
| **Ask** | A natural-language assistant answers questions with citations to the underlying evidence (RAG plus typed graph tools). Any proposed graph change pauses for human approval. |
| **Prove** | Evidence hashes and decisions are anchored on a hash-chained ledger; every action is written to an append-only audit log. |

---

## Key ideas

- **Human-in-the-loop, always.** Entity merges, face matches and agent-proposed graph edits are *candidates* until an investigator confirms them (`PENDING → CONFIRMED / REJECTED`).
- **Explainability is not optional.** Every flag ships with a "why is this flagged?" panel: source records, algorithm and score.
- **Tamper-evident, not "trust us".** Original evidence hashes, audit checkpoints, export manifests and warrant grants are anchored on a signed, sequenced hash chain. Only hashes ever leave the database, never personal data.
- **Least privilege and need-to-know.** Six roles, case-level assignment, break-glass access, originator-controlled access requests and a disclosure ladder (blind hit → metadata → view → download).
- **Insider threat is the realistic threat.** Audit trails, behavioural risk scoring, tripwire records, watermarked secure viewing and conflict-of-interest checks.
- **Sovereignty-ready.** LLM, embeddings, vector store, object store and reranker sit behind provider interfaces so a deployment can run in-country or fully air-gapped.

---

## Architecture

```mermaid
flowchart TD
    subgraph Sources
      A[FIRs / scans] --- B[CDRs / bank CSV] --- C[Photos] --- D[Social / surveillance]
    end
    Sources --> GW[Zero-trust ingestion<br/>type sniffing, hashing, AV scan, dedupe]
    GW --> OB[(Postgres + outbox)]
    OB --> PIPE[Inngest pipeline<br/>anchor → analyse → extract → index → link → project → faces]
    PIPE --> DOC[document-enhancement-service<br/>OCR, tables, enhancement]
    PIPE --> NER[ner-service<br/>rules + GLiNER + IndicNER]
    PIPE --> FACE[face-recognition-service<br/>detect, embed, quality]
    PIPE --> VEC[(Vector store<br/>Pinecone or pgvector)]
    PIPE --> NEO[(Neo4j<br/>knowledge graph + GDS)]
    PIPE --> LED[ledger-service<br/>signed hash chain]
    NEO --> API[Express API<br/>RBAC, case access, audit]
    VEC --> API
    API --> UI[Next.js investigator console<br/>graph studio, map, vault, security center, chat]
```

**Design rules that shape the system**

- Postgres is the **source of truth**. Neo4j is a projection built from Postgres through an outbox, and vectors are stored by id with no text or PII in their metadata.
- The blockchain-style ledger sits *after* entity resolution and *before* the graph write. It anchors integrity of evidence and access decisions. It does not store intelligence.
- Heavy or slow work (OCR, NER, embeddings, face detection, zone recompute, audit checkpoints) runs as durable, retryable **Inngest** functions.

---

## Tech stack

| Layer | Technology |
|---|---|
| API | Bun, TypeScript, Express 5, Zod 4, Better Auth (Google SSO and optional OIDC, MFA) |
| Database | PostgreSQL 16 with Prisma 7 (pgvector, pg_trgm) |
| Graph | Neo4j 5 with Graph Data Science |
| Vector search | Pinecone (default) or pgvector (on-prem backend) |
| Background jobs | Inngest |
| Object storage | S3-compatible (Cloudflare R2, MinIO, AWS S3) |
| AI services | FastAPI microservices: NER, face recognition, document analysis; LangChain.js and LangGraph for the agent |
| Ledger | Node service, Ed25519-signed hash chain with Merkle checkpoints; optional immudb / RFC 3161 anchoring |
| Frontend | Next.js 16, React, TanStack Query, shadcn/ui, reagraph, MapLibre and deck.gl, Recharts |
| Ops | Docker Compose, OpenTelemetry, Prometheus, pino |

---

## Repository layout

```
CrimeNet/
├── server/                    # Bun + Express API, Prisma schema, Inngest pipeline, agent, RAG, memory
├── client/                    # Next.js investigator console
├── ner-service/               # FastAPI: entity + relation extraction (English + Indic)
├── face-recognition-service/  # FastAPI: face detect / embed / quality
├── document-enhancement-service/  # FastAPI: OCR, tables, image enhancement, SVG sanitising
├── ledger-service/            # Evidence and audit anchoring, warrant verification
├── docs/                      # Architecture, RBAC, status, runbooks, compliance
├── scripts/                   # Dev and ops scripts (dev-only seeders live under server/scripts/dev)
├── AGENTS.md                  # Conventions and security invariants for contributors and coding agents
└── docker-compose.yml
```

---

## Quick start

### Prerequisites

- [Bun](https://bun.sh) 1.x
- Docker and Docker Compose
- Node 20+ (client tooling) and Python 3.11+ (only if you run the AI services outside Docker)
- Accounts or keys you will need depending on your setup:
  - an **OpenAI-compatible** LLM and embedding endpoint (OpenAI, or a self-hosted vLLM/Ollama)
  - a **Pinecone** API key, *or* use the pgvector backend
  - **Google OAuth** client credentials (or an OIDC provider) for sign-in
  - an S3-compatible bucket (Cloudflare R2, MinIO, etc.)

> Placeholder or invalid API keys are fine for booting and for the test suite. Tests run offline against fakes.

### 1. Clone and configure

```bash
git clone https://github.com/MEHULARORA11/CrimeNet.git
cd CrimeNet

cp server/.env.example server/.env
cp client/.env.example client/.env
# fill in the values described in "Configuration" below
```

### 2. Start the stack

```bash
docker compose up --build
```

This brings up Postgres (pgvector), Neo4j, the Inngest dev server, the AI services, the ledger service, the API and the client. Internal ports are not published by default. Use the `dev-ports` profile if you want direct access to them:

```bash
docker compose --profile dev-ports up --build
```

Typical local addresses:

| Service | URL |
|---|---|
| Client | http://localhost:3000 |
| API | http://localhost:4000 (`/healthz`, `/readyz`) |
| Inngest dev UI | http://localhost:8288 |
| Neo4j browser | http://localhost:7474 (needs `dev-ports`) |

### 3. Prepare data stores (first time only)

```bash
cd server
bun install
bun run db:generate               # prisma client
bun run db:migrate:deploy         # apply migrations
bun run graph:constraints         # Neo4j constraints and indexes
bun run scripts/provision-vector-indexes.ts   # Pinecone indexes (skip if using pgvector)
```

The server container performs the migrate and constraint steps on start, so this is only needed when running the API on your host.

### 4. Running without Docker (optional)

```bash
# API
cd server && bun install && bun run dev

# Client
cd client && bun install && bun run dev

# Each Python service (own terminal)
cd ner-service && pip install -r requirements.txt && uvicorn main:app --port 8000
cd face-recognition-service && pip install -r requirements.txt && uvicorn main:app --port 8001
cd document-enhancement-service && pip install -r requirements.txt && uvicorn main:app --port 8002

# Ledger
cd ledger-service && bun install && bun run start
```

Postgres, Neo4j and Inngest still need to be running. Easiest is `docker compose up postgres neo4j inngest-dev-server`.

---

## Configuration

All server configuration goes through `server/src/env.ts`, a Zod schema that fails at boot if something required is missing. The generated `server/.env.example` is the authoritative list. Key groups:

```bash
# Database
DATABASE_URL=postgresql://postgres:password@localhost:5432/crimenet

# Auth
BETTER_AUTH_SECRET=              # >= 16 chars: openssl rand -base64 32
BETTER_AUTH_URL=http://localhost:3000
GOOGLE_CLIENT_ID=
GOOGLE_CLIENT_SECRET=
ALLOWED_GOOGLE_WORKSPACE_DOMAIN= # restricts sign-in to your Workspace domain
REQUIRE_MFA=true                 # default in production

# Internal service-to-service auth (shared by server + all services)
INTERNAL_SERVICE_KEY=            # >= 32 chars, never a dev default in production
GATEWAY_SECRET_TOKEN=

# Graph
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=password

# Vector search / embeddings / LLM
PINECONE_API_KEY=
PINECONE_INDEX_NAME=
EMBEDDING_MODEL=text-embedding-3-small
EMBEDDING_DIM=1536
OPENAI_API_KEY=
LLM_BASE_URL=                    # point at vLLM/Ollama for in-country or air-gapped use
RAG_PIPELINE_MODE=agentic        # or "legacy"

# Object storage
S3_ENDPOINT=
S3_BUCKET=
S3_REGION=
S3_ACCESS_KEY_ID=
S3_SECRET_ACCESS_KEY=

# Service URLs
NER_SERVICE_URL=http://localhost:8000
FACE_RECOGNITION_SERVICE_URL=http://localhost:8001
DOCUMENT_ENHANCEMENT_SERVICE_URL=http://localhost:8002
LEDGER_SERVICE_URL=http://localhost:3002

# Inngest
INNGEST_BASE_URL=http://localhost:8288
INNGEST_DEV=1                    # unset in production; keys required in production
INNGEST_EVENT_KEY=
INNGEST_SIGNING_KEY=

# Deployment posture
EGRESS_MODE=open                 # open | restricted | airgapped
NODE_ENV=development
```

**Client** (`client/.env`): `BACKEND_URL` (server-side rewrite target) and `NEXT_PUBLIC_APP_URL`.

> Never commit real secrets. In production, secrets come from Docker secrets or a vault, and the server rejects placeholder values like `password` or `changeme`.

---

## First run: create the admin, load data

CrimeNet is **invite-only**: there is no self-service sign-up and no password login.

1. **Bootstrap the first administrator**
   ```bash
   cd server
   BOOTSTRAP_ADMIN_EMAIL=you@your-domain.in bun run bootstrap:admin
   ```
2. Sign in at http://localhost:3000 with that Google (or OIDC) account. The invite hook matches your email, creates your user with the invited role and enrols MFA.
3. As admin, create police stations and invite users (INSPECTOR, SUPERVISOR, ANALYST, FORENSIC_EXPERT, AUDIT_LOGGER). Invites expire (default 72 hours).
4. As an inspector, create a case, upload evidence, confirm the detected source type and, for CDR or bank files, review the **import preview** before committing.

### Try it with synthetic data

No real police data is used or required. A companion repository, **Synthetic_Data_Generator**, produces:

- 1,000 synthetic FIR PDFs (English, plus Hindi, Gujarati and Marathi samples)
- CDR and bank-transaction CSVs for the first 50 people (one genuinely connected network)
- AI-generated face images

Only those first 50 people form a connected network. The other 950 are standalone single-incident cases, so use the connected case for graph and map demos.

```bash
# dev only; never in production images
cd server
bun run scripts/dev/import-folder-dataset.ts /path/to/Synthetic_Data_Generator
```

The script pushes every file through the real intake and pipeline, so it doubles as an end-to-end test. The UI shows a **SANDBOX** ribbon when `APP_ENV=sandbox`.

---

## Using the platform

| Area | What you can do |
|---|---|
| **Home** | Role-aware dashboard: your cases, pipeline queue, review tasks, approvals and SLA timers. |
| **Cases** | Workbench with filters and saved views; Case 360 overview, timeline, notes and journal. |
| **Evidence Vault** | Grid/list/timeline view, secure watermarked viewer, custody timeline, one-click integrity verification, access-request cart. |
| **Graph Studio** | Typed nodes and edges, time slider, expand, path finding, communities, centrality, inferred links, explain-this-subgraph, export. |
| **Command Map** | Self-hosted MapLibre with H3 hex risk, incidents, call trails, tower coverage sectors, time playback. |
| **CDR toolkit** | Common numbers, top contacts, IMEI/IMSI swap detection, tower dump (warrant-gated), location trail, follow-up and conference calls. |
| **Money trail** | Time-respecting fund tracing, explainable mule-suspicion scores, watch-list export. |
| **Ask (chat)** | Cited answers, clarification prompts, file attachments, human approval for any graph change. |
| **Review** | Entity-resolution and face-match review with side-by-side evidence; supervisors can undo merges. |
| **Exports** | Court-ready ZIP with signed manifest and BSA s.63 certificate drafts (statutory wording supplied by counsel). |
| **Security Center** | Audit explorer, chain verification, alerts, user risk, approvals inbox, tripwires, ledger explorer. |

---

## Security and privacy model

- **Roles:** `INSPECTOR`, `SUPERVISOR`, `ANALYST`, `FORENSIC_EXPERT`, `ADMIN` (no case content by default), `AUDIT_LOGGER` (no case content ever). See `docs/rbac-matrix.md`.
- **Authorization** is deny-by-default and derived from the *target resource*, never a client-supplied case id. Warrant-gated actions are verified against the ledger.
- **Authentication:** Google Workspace SSO or generic OIDC, invite-only, MFA (TOTP and passkeys), short idle and absolute session limits.
- **Audit:** synchronous, append-only, hash-chained; denied requests are logged; a nightly job verifies the chain and checkpoints it externally.
- **Agent safety:** free-form LLM Cypher is off by default. Typed, case-scoped tools are used instead, structurally guarded; writes go to Postgres first after human approval.
- **Data minimisation:** no text or PII in vector metadata; sensitive identifiers (Aadhaar etc.) stored masked; victims and minors masked everywhere unless an audited unmask flow runs.
- **Face recognition** is the most sensitive module: calibrated thresholds, quality gating, person-level matching, no automatic identity assignment, "investigative lead only" labelling.
- **Insider-threat controls:** view-first secure viewer, dynamic and forensic watermarks, download budgets, tripwire records, risk scoring, conflict-of-interest matching.

Read `AGENTS.md` for the full list of security invariants.

---

## Testing and CI

```bash
cd server
bun run typecheck        # tsc --noEmit
bun test                 # offline: network is stubbed
bun run check:size       # enforces the 500-line file limit
bun run eval:rag         # RAG evaluation harness (fakes by default)

cd ../client
bun run lint && bun run build

# Python services
cd ../ner-service && pytest
```

CI runs the file-size guard, typecheck, unit tests, Python tests and the client build. Integration tests (pipeline end to end with Postgres, Neo4j, MinIO, ClamAV) are part of the release hardening work.

---

## Project status and known gaps

CrimeNet is under active, commit-by-commit hardening toward production use. Be aware of the following:

- The authoritative per-module status lives in `docs/STATUS.md`. If it disagrees with this README, trust `STATUS.md`.
- The default ledger is a signed hash chain in a **separate** database. It is tamper-*evident* within that trust boundary. For stronger guarantees enable external anchoring (RFC 3161 timestamping or immudb). An NBF/Vishvasya adapter is an interface only.
- InsightFace's pretrained models are licensed for **non-commercial research**. For commercial or government use, switch to the permissive default engine or obtain a licence (`docs/face-licensing.md`).
- Neo4j Community has no role-based access control, so case scoping is enforced in application code and through typed tools.
- Pinecone's published serverless regions do not include India. For data-residency requirements use the pgvector backend and the in-country or air-gapped topology (`docs/DEPLOYMENT-TOPOLOGIES.md`).
- LangGraph approval checkpoints are durable only when the Postgres checkpointer is enabled.
- Hindi/Gujarati/Marathi extraction quality depends on the OCR and NER models and should be measured on real samples before relying on it.

---

## Roadmap

Planned, roughly in this order: modus-operandi similarity and series detection, watch-lists with alerting, WhatsApp/Telegram/Google export parsers with Section 63 certificates, crypto-wallet tracing, voice-to-FIR drafting for field officers, CCTNS/ICJS/NAFIS connectors where access is provided, a face-recognition accuracy dashboard, and a single-node appliance image for stations without a data centre.

---

## Contributing

1. Read `AGENTS.md` (conventions, known traps, security invariants).
2. Keep source files under **500 lines**; no mock or demo data in runtime code.
3. All configuration through `server/src/env.ts`; new routes need authentication, a permission check, a case-access check and an audit tag.
4. One conventional commit per change, with typecheck and tests green first.

---

## Legal and ethics notice

- CrimeNet analyses network position and relationship patterns within data gathered for an **active, authorised investigation**. It surfaces structural signals for a human to review. It is **not** predictive policing and no output is a determination of guilt.
- Dataset and demo content is entirely synthetic. No real individual's likeness or records are used.
- This repository is a software project, not legal or compliance certification. Evidence-law handling (BSA s.63), victim and minor identity protection, CERT-In log retention and DPDP obligations must be reviewed by qualified counsel before any real deployment.

---

*Questions, ideas or bug reports? Open an issue on the repository.*