# 🇸🇬 Singapore Job Market Intelligence Platform

> **Production-grade, cloud-native job market intelligence platform** for Singapore, leveraging GCP services, ML/NLP pipelines, and real-time data processing. Scrapes 10,000+ jobs daily from JobStreet & MyCareersFuture, analyzes with SBERT embeddings, predicts salaries, and surfaces insights through interactive dashboards.

---

## 🎯 Project Overview

A fully automated, end-to-end data pipeline that:
- **Scrapes** job postings from major Singapore job portals (JobStreet, MyCareersFuture)
- **Processes** with event-driven ETL (Cloud Functions)
- **Analyzes** using NLP embeddings and ML models (salary prediction, role clustering)
- **Augments** with GenAI agents (RAG, LangGraph) for intelligent query handling
- **Serves** insights via FastAPI, MCP Server, and visual dashboards

---

## Prerequisites
- Python 3.13
- Docker (for Cloud Run testing)
- Google Cloud SDK (`gcloud`)
- GCP Project with billing enabled

## Environment Variables
GCP_PROJECT_ID=your-project-id
BIGQUERY_DATASET_ID=sg_job_market
GCS_BUCKET=your-bucket-name
GCP_REGION=asia-southeast1
SCRAPER_USER_AGENTS="Mozilla/5.0 (...), Mozilla/5.0 (...)"
GCS_UPLOAD_ENABLED=false  # Set to true for cloud uploads
LOCAL_RETENTION_DAYS=30

---

## 📊 Architecture

Cloud Scheduler → Cloud Run (Docker) → GCS → Cloud Functions (ETL) → BigQuery → Vertex AI → FastAPI/MCP → Dashboards/Agents

**Data Flow:**
1. **Scraping Layer:** Cloud Run Jobs (Docker containers) scrape job portals daily
2. **Storage Layer:** Raw JSONL stored in Google Cloud Storage
3. **ETL Layer:** Cloud Functions triggered by GCS events, clean and transform data
4. **Data Warehouse:** BigQuery stores cleaned jobs, embeddings, and features
5. **ML Layer:** Vertex AI trains models (salary prediction, clustering, classification)
6. **GenAI Layer:** LangGraph agents & MCP Server for external tool access
7. **API Layer:** FastAPI serves predictions and insights
8. **Presentation:** Looker Studio & Streamlit dashboards

All components are modular, testable, documented, and cloud-ready.

---

## ✨ Key Features

- ✅ **Automated Daily Scraping:** JobStreet (GraphQL) & MyCareersFuture (Selenium + API)
- ✅ **Production Infrastructure:** Dockerized scrapers deployed to GCP Cloud Run
- ✅ **Event-Driven ETL:** Cloud Functions auto-triggered on new data uploads
- ✅ **Smart Deduplication:** Hash-based job matching with incremental updates
- ✅ **NLP Pipeline:** SBERT embeddings for semantic job similarity
- ✅ **ML Models:** Salary prediction (LightGBM), role classification, clustering (KMeans)
- ✅ **Resilient Design:** Retry logic, exponential backoff, graceful error handling
- ✅ **Observability:** Structured logging, Cloud Monitoring integration
- ✅ **Cost-Optimized:** Uses GCP free tier, auto-scaling, and efficient resource allocation
- ✅ **Agentic RAG:** LangGraph-orchestrated retrieval pipeline with Gemini Pro
- ✅ **MCP Server:** Exposes job data as tools to external AI assistants (Claude/Cursor)

---

## 📈 Current Status

**Phase 1: Scraping Infrastructure** ✅ **COMPLETE**
- JobStreet scraper (GraphQL, two-phase strategy)
- MyCareersFuture scraper (Selenium + API hybrid)
- GCS integration with auto-upload
- Docker containerization
- Cloud Run deployment
- Cloud Scheduler automation

**Phase 2: ETL Pipeline** ✅ **COMPLETE** (Deployed to Production)
- ✅ Stage 1: GCS → raw_jobs (deployed & operational)
- ✅ Stage 2: raw_jobs → cleaned_jobs (deployed & operational)
- ✅ Two-stage transformation in single Cloud Function
- ✅ Text cleaning: HTML removal, unicode normalization, whitespace cleanup
- ✅ Salary parsing: Range extraction, period detection, monthly conversion
- ✅ Query-time deduplication with ROW_NUMBER() pattern
- ✅ BigQuery streaming API with 100% success rate (5,861+ jobs tested)
- ✅ Cloud Function `etl-gcs-to-bigquery` deployed in asia-southeast1

**Phase 3: ML/NLP** � **IN PROGRESS**
- ✅ Phase 3A: SBERT embeddings (6,775 jobs, Cloud Run Job deployed, vector index operational, scheduled 3 AM SGT daily)
- 🔄 Phase 3B: Feature engineering (IN PROGRESS)
- 🔲 Phase 3C: Model training (salary prediction, classification, clustering)
- 🔲 Phase 3D: Model artifacts & deployment

**Phase 4: API & Dashboards** 🔲 **PLANNED**
- FastAPI REST endpoints

**Phase 5: GenAI & Agents** 🔲 **PLANNED**
- RAG Pipeline (Vertex AI + BigQuery Vector Search)
- LangGraph Orchestration
- MCP Server Implementation
- Streamlit dashboard
- Looker Studio integration



## 📁 FOLDER STRUCTURE

/scraper/           → jobsite scrapers, base classes, parsers
/etl/               → cleaning, transforms, salary parsing
/nlp/               → embeddings, tokenization, language cleaning
/ml/                → training pipelines & evaluation
/genai/             → RAG, LangGraph agents, MCP server
/api/               → FastAPI app for Cloud Run
/dashboard/         → Streamlit UI
/utils/             → bq.py, gcs.py, config.py, logging.py
/models/            → saved ML artifacts
/notebooks/         → exploration only
/data/raw/          → local raw dumps (gitignored)
data/processed/     → cleaned datasets (gitignored)


## 👥 TEAM AGENTS

- Python 3.13 only.
- Strict PEP8 & typing.
- Every file must include docstrings.
- Functions must be pure when possible.
- Use dependency injection, avoid global state.
- Add logging with timestamps + context info.
- Include retry logic for:
  - network calls
  - GCS operations
  - BigQuery operations
- All outputs must be deterministic and consistent with BigQuery schema.
- Use datetime objects for timestamps (not strings) - auto-converted to TIMESTAMP in BigQuery.
- All BigQuery operations are append-only - never update or delete rows.


## 📁 FOLDER STRUCTURE (COPILOT MUST FOLLOW)

/scraper/           → jobsite scrapers, base classes, parsers
/etl/               → cleaning, transforms, salary parsing, pipeline
/nlp/               → embeddings, tokenization, language cleaning
/ml/                → training pipelines & evaluation
/api/               → FastAPI app for Cloud Run
/dashboard/         → Streamlit UI
/utils/             → bq.py, gcs.py, config.py, logging.py, schemas.py
/models/            → saved ML artifacts
/notebooks/         → exploration only
/data/raw/          → local raw dumps (gitignored)
/data/processed/     → cleaned datasets (gitignored)
/.venv/             → Python virtual environment (ALWAYS USE THIS)
/tests/             → unit tests for all modules


## 👥 TEAM AGENTS (COPILOT MUST OBEY ROLE RULES)

### 1. PROJECT LEAD AGENT
- Ensures folder structure compliance.
- Enforces architecture: Scheduler → Cloud Run → GCS → BigQuery → VertexAI.
- Prevents spaghetti code & duplication.
- Generates TODO lists, roadmaps, UML if needed.

### 2. SCRAPER ENGINEER AGENT
- Use async scraping.
- Implement:
  - BaseScraper class
  - Site-specific scraper (Jobstreet, MCF)
  - Parser and normalizer utilities
- Output JSONL with fields:
  title, company, location, description, date_posted, url, source, salary_text
- Must include retry + user-agent rotation.

### 3. CLOUD BACKEND AGENT
- Writes Dockerfile (Python FastAPI + scraper jobs).
- Creates cloudrun.sh deployment script.
- Writes BigQuery schema modules.
- Writes GCS upload/download helpers.
- Ensures Python code is Cloud Run compatible (stateless).

### 4. ETL ENGINEER AGENT
- Cleans and normalizes text.
- Parses salary ranges robustly.
- Deduplicates by title + company + description hash.
- Writes cleaned Parquet.
- Loads into BigQuery using the official Python client.
- Creates schema-matching dicts.

### 5. ML ENGINEER AGENT
- Creates feature engineering pipelines.
- Trains ML:
  - Regression (salary)
  - Classification (role categories)
  - Clustering (KMeans, PCA)
- Generates embeddings using Sentence-BERT.
- Saves models to GCS + /models.
- Produces evaluation metrics: RMSE, accuracy, F1, silhouette.

### 6. API ENGINEER AGENT
- Builds FastAPI microservice for Cloud Run.
- Endpoints must include:
  /predict-salary
  /similar-jobs
  /embedding
  /cluster-insights
- Include request validation (Pydantic).

### 7. DASHBOARD ENGINEER AGENT
- Write Streamlit pages:
  - Job trends
  - Salary analytics
  - Skill clusters
  - ML model comparison
- Connects to BigQuery using service account.


## ✔️ Delegation Plan (End-to-End)

This document splits the platform into agent-owned workstreams and defines contracts (schemas, interfaces, handoffs) so implementation can proceed without ambiguity.

## 0. Pipeline Contract (Single Source of Truth)

Target pipeline:

Cloud Scheduler → Cloud Run (Docker: scraper runner) → GCS (raw JSONL) → ETL job → GCS (processed Parquet) → BigQuery (raw_jobs, cleaned_jobs, embeddings, features) → Vertex AI (training/evaluation) → Cloud Run (FastAPI) → Looker Studio / Streamlit.

Non-negotiables:
- All code reads config from env (`.env` locally) via `utils.config.Settings`.
- All external calls must use retry (`utils.retry`) + logging (`utils.logging`).
- Scraper output must match the `raw_jobs` schema contract.

## 1. Workstream Split (Agents, Deliverables)

### A. Project Lead Agent
Deliverables:
- Folder structure compliance (as in README.md).
- Schema contract definitions and module boundaries.
- PR checklist + acceptance criteria for each workstream.
- Integration guidance: how components connect and what each produces/consumes.

Done criteria:
- Every folder contains at least one base file and docstrings.
- `utils/` exposes stable interfaces for config/logging/retry/schemas.

### B. Scraper Engineer Agent ✅ COMPLETED
Scope:
- Implement `scraper.base.BaseScraper` concrete scrapers for:
  - JobStreet
  - MyCareersFuture
- Strategy: GraphQL APIs (preferred) or Selenium (headless).
- Produce JSONL lines aligned to `utils.schemas.RawJob` fields.

Deliverables:
- ✅ `scraper/jobstreet.py`, `scraper/mcf.py` implemented.
- ✅ Deterministic JSONL writer (local + GCS upload with compression).
- ✅ Smoke script to run a small crawl and write `data/raw/.../dump.jsonl`.

Acceptance criteria:
- ✅ Each record contains: `title, company, location, description, date_posted, url, source, salary_text`.
- ✅ Records validate against `raw_jobs` BigQuery schema (`utils.bq_schemas.raw_jobs_schema`).

**Implementation Details (Completed):**
- **JobStreet:** Two-phase strategy using v5 listing API + GraphQL detail queries
  - Phase 1: Collect job IDs from paginated listing API with incremental checkpointing
  - Phase 2: Batch GraphQL queries (max 32 per batch) for full job details
  - Progressive rate limiting: 10min → 30min → 1hr → abort (cumulative tracking)
  - Robust error handling: individual job failures don't break batches
  - GraphQL fragment in `scraper/jobstreet_queries.py`
- **MyCareersFuture:** Selenium + API hybrid
  - Phase 1: Selenium WebDriver collects job UUIDs from dynamic search pages
  - Phase 2: REST API calls for detailed job information
- **Base Infrastructure:**
  - `scraper/base.py`: Abstract base class with retry logic, user-agent rotation, session management
  - GCS upload support: `GCS_UPLOAD_ENABLED=true` in environment
  - 30-day local retention policy: `LOCAL_RETENTION_DAYS=30`
  - `scraper/validation.py`: JSONL validation utilities
  - `scraper/smoke_test.py`: Quick local testing script
- **Entry Points:**
  - `python -m scraper --site jobstreet`
  - `python -m scraper --site mcf`
  - `python scraper/smoke_test.py jobstreet`
- **Cloud Deployment:**
  - Docker images: `Dockerfile.scraper.jobstreet`, `Dockerfile.scraper.mcf`
  - Cloud Build: `cloudbuild.jobstreet.yaml`, `cloudbuild.mcf.yaml`
  - Artifact Registry: Images pushed and ready
  - Cloud Run Jobs: Created and configured

### C. Cloud Backend Agent 🔄 70% COMPLETE
Scope:
- Cloud Run packaging, scheduler trigger contract, storage and BigQuery helpers.

Deliverables:
- ✅ Dockerfile(s) for:
  - ✅ scraper runner (JobStreet, MCF)
  - 🔲 API service (pending API Engineer implementation)
1. ✅ Scaffolding: create mandatory folders + base contracts.
2. ✅ Backend Agent: implement GCS helpers + dockerization + deployment scripts.
3. ✅ Scraper Agent: implement scrapers end-to-end and write raw JSONL locally + GCS.
4. ✅ Cloud Backend Agent: wire raw JSONL upload to GCS path contract.
5. 🔄 ETL Agent: implement transform + streaming + load to BigQuery.
6. 🔲 ML Agent: add embeddings + baseline models + artifact persistence.
7. 🔲 API Agent: implement endpoints backed by BigQuery and models.
8. 🔲 Dashboard Agent: build Streamlit dashboards and/or Looker Studio config.

Development Rules:
- ✅ **Base Infrastructure:** Config, logging, retry, schemas complete
- 🔲 **BigQuery:** Stub only, needs `stream_rows_to_bq()`, `ensure_dataset()`, `ensure_table()`
- 🔲 **Cloud Scheduler:** Not configured yet (Phase 5)
- 🔲 **Deployment Scripts:** Need automation scripts in `deployment/`

**Next Priority:** Implement BigQuery streaming API to unblock ETL Cloud Function

### D. ETL Engineer Agent 🔲 NOT STARTED (BLOCKED)
**Status:** Waiting for BigQuery streaming API from Cloud Backend Agent

Scope:
- Normalize/clean raw records and create `CleanedJob` outputs.
- Dedup by stable hash (title+company+description_clean).
- Stream cleaned data directly to BigQuery (no intermediate Parquet - cost optimization).
- Deploy as Cloud Function with GCS finalize trigger.
- Note: Embeddings are handled by ML Agent.

Deliverables:
- 🔲 Implement `etl/cloud_function_main.py` (Cloud Function entry point)
- 🔲 Implement `etl/pipeline.py` plus supporting modules
- 🔲 Robust salary parser in `etl/salary_parser.py`
- 🔲 BigQuery streaming using `utils.bq.stream_rows_to_bq()` from Cloud Backend

Acceptance criteria:
- 🔲 Outputs match `utils.schemas.CleanedJob` and `utils.bq_schemas.cleaned_jobs_schema()`.
- 🔲 Deterministic job_id generation.
- 🔲 Event-driven: Automatically triggered when scraper uploads to GCS
- 🔲 Idempotent: Can reprocess same file safely

**Blocker:** Cloud Backend must implement `utils/bq.py` BigQuery streaming API first

### E. ML Engineer Agent
Scope:
- Embeddings, features, training/evaluation.

Deliverables:
- Implement `nlp/embeddings.py` (choose model + dependency strategy).
- Implement `ml/train.py` and evaluation outputs.
- Persist artifacts to `/models` and optionally to GCS.

Acceptance criteria:
- Produces embeddings table and/or files with stable IDs aligned to `job_id`.

### F. API Engineer Agent
Scope:
- FastAPI service deployed to Cloud Run.

Deliverables:
- Implement `api/app.py` with endpoints:
  - `/predict-salary`
  - `/similar-jobs`
  - `/embedding`
  - `/cluster-insights`
- Request/response schemas (likely Pydantic).

Acceptance criteria:
- No hard-coded project IDs.
- Uses BigQuery for reads; models loaded from GCS/local image.

### G. Dashboard Engineer Agent
Scope:
- Streamlit dashboards and/or Looker Studio guidance.

Deliverables:
- Implement `dashboard/app.py` and pages per README.

Acceptance criteria:
- Reads from BigQuery dataset only.

## 2. Integration Handoffs (Contracts)

### Scraper → GCS (raw)
- Format: JSONL
- Naming: `raw/{site}/{YYYY-MM-DD}/dump.jsonl`
- Record contract: `utils.schemas.RawJob`

### ETL → GCS (processed)
- Format: Parquet
- Naming: `processed/{date}/clean.parquet`
- Record contract: `utils.schemas.CleanedJob`

### ETL → BigQuery
- Tables:
  - `raw_jobs` (append)
  - `cleaned_jobs` (merge/upsert by `job_id` or append + dedupe logic)

### NLP/ML → BigQuery + Models
- `embeddings` table: `job_id` + vector
- `features` table: engineered features per `job_id`
- `/models` artifacts and mirrored in GCS.

## 3. Execution Steps (Lifecycle Initiation)

1. Scaffolding (done in this commit): create mandatory folders + base contracts.
2. Backend Agent: implement GCS helpers + dockerization + deployment scripts.
3. Scraper Agent: implement 1 scraper end-to-end and write raw JSONL locally.
4. Cloud Backend Agent: wire raw JSONL upload to GCS path contract.
5. ETL Agent: implement transform + Parquet + load to BigQuery.
6. ML Agent: add embeddings + baseline models + artifact persistence.
7. API Agent: implement endpoints backed by BigQuery and models.
8. Dashboard Agent: build Streamlit dashboards and/or Looker Studio config.


## ☁️ CLOUD DEPLOYMENT (PRODUCTION PIPELINE)

### Architecture Overview
```
Cloud Scheduler 
    ↓
Cloud Run (Docker: scraper-jobstreet, scraper-mcf)
    ↓
GCS (gs://<bucket>/raw/{site}/{timestamp}/dump.jsonl)
    ↓ (GCS finalize event trigger)
Cloud Function (ETL: Clean & Transform)
    ↓ (direct streaming)
BigQuery (raw_jobs, cleaned_jobs, embeddings)
    ↓
Vertex AI (Model Training & Evaluation)
    ↓
Cloud Run API (FastAPI)
    ↓
Looker Studio / Streamlit Dashboard
```

### Pipeline Flow Details:
1. **Cloud Scheduler** triggers scraper (2 AM for JobStreet, 3 AM for MCF)
2. **Cloud Run Scraper** executes and uploads `.jsonl` to GCS
3. **GCS Event Trigger** (finalize) automatically invokes Cloud Function
4. **Cloud Function ETL** reads, cleans, transforms data
5. **BigQuery Streaming** inserts cleaned data directly (no intermediate Parquet)
6. **ML/API/Dashboard** consume data from BigQuery

## Current Implementation Status

### ✅ Completed
- **Scrapers:** JobStreet (GraphQL) and MyCareersFuture (Selenium + API)
  - Local JSONL output: `data/raw/{site}/{timestamp}/dump.jsonl`
  - Auto-cleanup: Keeps last 10 runs per source
  - Schema validation against `RawJob` dataclass
- **Config Management:** Environment-based configuration via `.env`
- **Logging:** Dual console + file output with auto-rotation
- **Retry Logic:** Exponential backoff for network calls
- **Schemas:** `RawJob` and `CleanedJob` dataclasses with BigQuery schema generation

### 🔲 In Progress (Cloud Backend Agent)
- **BigQuery Integration:** `utils/bq.py` - dataset/table creation, streaming inserts
- **GCS Integration:** `utils/gcs.py` - upload/download with compression, path helpers
- **Cloud Function ETL:** Event-driven cleaning triggered by GCS finalize event
- **Docker:** Separate Dockerfiles for JobStreet (Python only) and MCF (with Chrome)
- **Cloud Run:** Deployment scripts, service accounts, IAM setup
- **Cloud Scheduler:** Daily triggers at 2 AM (JobStreet) and 3 AM (MCF) SGT

### 🔲 Planned
- **ETL Pipeline:** Text cleaning, salary parsing, deduplication
- **NLP & ML:** Sentence-BERT embeddings, regression/classification models
- **API Layer:** FastAPI endpoints for predictions and insights
- **Dashboard:** Streamlit or Looker Studio for visualization


## 📝 COPILOT

```
Help me build a cloud-ready, production-grade, ML-powered Singapore job market intelligence platform deployable on GCP.
```

**DEVELOPMENT RULES:**
- Not a homework project; this is a portfolio-level system.
- Always produce production-style code.
- Never produce “student exercises”.
- NEVER skip error handling.
- Prefer async where appropriate.
- For any script touching GCP:
  - Use environment variables, never hardcode project IDs.
- Ensure the entire pipeline is reproducible end-to-end.

**WORKFLOW EXPECTATIONS:**
1. Identify which agent(s) should act.
2. Follow architecture constraints.
3. Follow folder structure.
4. Provide complete modules, not fragments.
5. Provide test examples when relevant.
6. Document every class & function.