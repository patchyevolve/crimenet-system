# Criminal Network Evidence Pipeline

## Current implementation

Stages 1–11 run in the case pipeline. Stage 10 is a read-only critic that can
flag supplied records for human review. Stage 12 runs for a declared multi-case
scope and reports per-case totals plus reviewed global identity links. Evidence
and risk scores remain case-scoped. GNN inference, Bayesian posterior
inference, and information gain are not implemented; see
[`../documents/ML_ENGINE.md`](../documents/ML_ENGINE.md).

## Run the synthetic multi-case demo

From this directory:

```bash
python run.py --multi-case-manifest demo_data/MULTI_CASE_MANIFEST.json
```

Each case gets its own output directory under `output/cases/<case_id>`. The
workspace-wide Stage 11 index is exported under `output/global_index` from
PostgreSQL when enabled, and uses a shared file index in file-only mode. Stage
12 writes `output/scoped_analytics.json`. The demo contains synthetic cases in
Delhi, Mumbai, Jaipur, and Hyderabad. A shared phone is expected to link across
cases and produce a cross-jurisdiction alert. Each case can also run alone;
Stage 11 still consults the persistent database-wide identity index.

To run without any model calls:

```bash
python run.py --multi-case-manifest demo_data/MULTI_CASE_MANIFEST.json --no-llm
```

To run entirely in file-only mode, with no PostgreSQL writes:

```bash
python run.py --multi-case-manifest demo_data/MULTI_CASE_MANIFEST.json --no-database
```

## LLM configuration

`config/pipeline.json` enables Groq's exact model `openai/gpt-oss-20b`.
Mistral is not on the pipeline route. Hosted API calls can incur charges. The
LLM is used for unstructured narrative extraction, resolution of ambiguous
entity candidates, and Stage 10's bounded output review. Structured CDR, bank,
and JSON extraction stays deterministic. The critic only writes a separate
review artifact and never changes evidence or analytical stores.

Provider/model used for each call are recorded in `output/llm_responses`.
Pipeline calls use Groq directly and do not attempt Ollama or Mistral.
`python run.py --status` reports configured provider readiness without making
a remote API call; `CONFIGURED` means credentials and local limits are present,
not that Groq will accept a request. Groq may reject requests because of
account, workspace, or model limits. Groq API use is metered and requires a
valid Groq API key.

## PostgreSQL in Docker

The pipeline's default `DATABASE_URL` targets the `criminal_network_db`
container on host port 5432. A separate PostgreSQL container on another port
(for example 5433) is a different database and is not used unless
`DATABASE_URL` is changed. Check the target with `python run.py --status` for
provider configuration and `docker compose ps` for this project's containers.
The database uses the persistent `postgres_data` Docker volume; do not run
`docker compose down -v` when you want to keep its data.

Database pruning is scoped to the same `case_id`. Runs without a case ID skip
case-data synchronization; use `--no-database` for disposable verification.

## Tests

```bash
python -m pytest -q
```
