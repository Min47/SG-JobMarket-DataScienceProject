You are part of my engineering team building a full end-to-end Singapore Job Market Intelligence Platform.

Follow all instructions below strictly.

========================================================
🎯 PROJECT SUMMARY
========================================================
We are building a fully production-style GCP project:

Cloud Scheduler → Cloud Run (Docker) → GCS → ETL → BigQuery → Vertex AI → Cloud Run API → Looker Studio / Streamlit.

All components must be modular, testable, documented, and cloud-ready.

========================================================
🧱 ARCHITECTURE RULES
========================================================
1. **Scraping**
   - Runs in Cloud Run (Docker), triggered by Cloud Scheduler.
   - Use Python: GraphQL APIs (preferred) or Selenium (headless).
   - Write raw JSONL to GCS:
     gs://<bucket>/raw/{site}/{YYYY-MM-DD}/dump.jsonl

2. **ETL Pipeline**
   - Runs as Cloud Function (event-driven, triggered by GCS).
   - Triggered automatically when scraper uploads `.jsonl` to GCS.
   - Cleans text, parses salary, dedupes, normalizes.
   - Streams cleaned data directly to BigQuery (no intermediate Parquet).
   - BigQuery dataset: sg_job_market
   - BigQuery tables: raw_jobs, cleaned_jobs

3. **NLP + ML**
   - Use Sentence-BERT for embeddings.
   - ML models: Linear Regression, LightGBM, Logistic Classification, KMeans, PCA.
   - Save trained models in:
     /models (local) AND gs://<bucket>/models/
   - Vertex AI used for training/evaluation where possible.
   - Loads embeddings and features into BigQuery.

4. **API Layer**
   - Python FastAPI.
   - Runs in Cloud Run.
   - Provides endpoints:
     /predict-salary
     /similar-jobs
     /embedding
     /role-cluster

5. **Dashboard**
   - Two options:
     - Looker Studio (connects to BigQuery)
     - Streamlit app (Python)
   - Include job trends, salary ranges, clusters, ML comparison.

========================================================
💻 CODE QUALITY & CONVENTIONS
========================================================
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

========================================================
📁 FOLDER STRUCTURE (COPILOT MUST FOLLOW)
========================================================
/scraper/           → jobsite scrapers, base classes, parsers
/etl/               → cleaning, transforms, salary parsing
/nlp/               → embeddings, tokenization, language cleaning
/ml/                → training pipelines & evaluation
/api/               → FastAPI app for Cloud Run
/dashboard/         → Streamlit UI
/utils/             → bq.py, gcs.py, config.py, logging.py
/models/            → saved ML artifacts
/notebooks/         → exploration only
/data/raw/          → local raw dumps (gitignored)
data/processed/     → cleaned datasets (gitignored)

========================================================
👥 TEAM AGENTS (COPILOT MUST OBEY ROLE RULES)
========================================================

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

========================================================
✔️ Delegation Plan (End-to-End)
========================================================

This document splits the platform into agent-owned workstreams and defines contracts (schemas, interfaces, handoffs) so implementation can proceed without ambiguity.

## 0. Pipeline Contract (Single Source of Truth)

Target pipeline:

Cloud Scheduler → Cloud Run (Docker: scraper runner) → GCS (raw JSONL) → ETL job → GCS (processed Parquet) → BigQuery (raw_jobs, cleaned_jobs, embeddings, features) → Vertex AI (training/evaluation) → Cloud Run (FastAPI) → Looker Studio / Streamlit.

Non-negotiables:
- All code reads config from env (`.env` locally) via `utils.config.Settings`.
- All external calls must use retry (`utils.retry`) + logging (`utils.logging`).
- Scraper output must match the `raw_jobs` schema contract.

## 1. Workstream Split (Agents, Deliverables)

### A. Project Lead Agent (you + Copilot acting as lead)
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
- 🔲 Cloud Run deployment scripts (in `deployment/`).
- ✅ Implement `utils/gcs.py` - **FULLY IMPLEMENTED**
- 🔲 Expand `utils/bq.py` - **IN PROGRESS (PRIORITY)**

Acceptance criteria:
- ✅ Stateless container runs with env vars only - **VERIFIED**
- ✅ Uploads raw JSONL to `gs://<bucket>/raw/{site}/{timestamp}/dump.jsonl.gz` - **WORKING**
- 🔲 Creates dataset/tables if missing - **NEEDS BQ IMPLEMENTATION**

**Current Status:**
- ✅ **GCS Integration:** Fully implemented with upload/download/compression
- ✅ **Docker:** Multi-stage builds, optimized sizes (<500MB JobStreet, <800MB MCF)
- ✅ **Cloud Run Jobs:** Deployed to Artifact Registry and Cloud Run
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


========================================================
☁️ CLOUD DEPLOYMENT (PRODUCTION PIPELINE)
========================================================

## Architecture Overview
```
Cloud Scheduler (2 AM SGT) 
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

## Quick Start (Local Development)

### Prerequisites
- Python 3.13
- Virtual environment (`.venv`) - **REQUIRED for all commands**
- Docker (for Cloud Run testing)
- Google Cloud SDK (`gcloud`)
- GCP Project with billing enabled

## Setup Virtual Environment

**IMPORTANT:** Always use the virtual environment for running any Python commands:

```bash
# Create virtual environment (first time only)
python -m venv .venv

# Activate virtual environment
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# Windows CMD:
.venv\Scripts\activate.bat
# Linux/Mac:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run commands (example)
python -m scraper --site jobstreet
```

**For all code examples in this README, use:**
- Windows: `.venv\Scripts\python.exe <command>`
- Linux/Mac: `.venv/bin/python <command>`

### Setup
1. Clone repository:
   ```bash
   git clone <repo-url>
   cd SG_Job_Market
   ```

2. Create virtual environment:
   ```bash
   python -m venv .venv
   .venv\Scripts\activate  # Windows
   # source .venv/bin/activate  # Linux/Mac
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Configure environment variables (`.env`):
   ```env
   GCP_PROJECT_ID=your-project-id
   BIGQUERY_DATASET_ID=sg_job_market
   GCS_BUCKET=your-bucket-name
   GCP_REGION=asia-southeast1
   SCRAPER_USER_AGENTS="Mozilla/5.0 (...), Mozilla/5.0 (...)"
   GCS_UPLOAD_ENABLED=false  # Set to true for cloud uploads
   LOCAL_RETENTION_DAYS=30
   ```

5. Run scrapers locally:
   ```bash
   # JobStreet scraper
   python -m scraper --site jobstreet

   # MyCareersFuture scraper
   python -m scraper --site mcf

   # Smoke test (quick validation)
   python scraper/smoke_test.py jobstreet
   ```

## Cloud Deployment (Production)

### One-Time Setup
1. **GCP Configuration:**
   ```bash
   # Authenticate
   gcloud auth login
   gcloud config set project YOUR_PROJECT_ID

   # Enable APIs
   gcloud services enable \
     run.googleapis.com \
     scheduler.googleapis.com \
     cloudbuild.googleapis.com \
     artifactregistry.googleapis.com \
     bigquery.googleapis.com \
     storage.googleapis.com
   ```

2. **Create GCS Bucket:**
   ```bash
   gsutil mb -l asia-southeast1 gs://YOUR_BUCKET_NAME
   gsutil lifecycle set gcs_lifecycle.json gs://YOUR_BUCKET_NAME
   ```

3. **Create BigQuery Dataset:**
   ```bash
   bq mk --location=asia-southeast1 sg_job_market
   ```

4. **Setup Service Accounts:**
   ```bash
   # See deployment/iam_setup.md for detailed instructions
   gcloud iam service-accounts create scraper-runner \
     --display-name="Scraper Cloud Run Service Account"
   ```

### Deploy Scrapers
```bash
# Deploy JobStreet scraper
./deployment/deploy_scraper.sh --site jobstreet --env prod

# Deploy MCF scraper
./deployment/deploy_scraper.sh --site mcf --env prod

# Setup Cloud Scheduler
./deployment/scheduler_setup.sh
```

### Verify Deployment
```bash
# List Cloud Run services
gcloud run services list --region asia-southeast1

# List Cloud Scheduler jobs
gcloud scheduler jobs list --location asia-southeast1

# Manually trigger scraper (for testing)
gcloud scheduler jobs run scrape-jobstreet-daily --location asia-southeast1

# View logs
gcloud logs tail --filter="resource.type=cloud_run_revision" --limit 50
```

## Cost Estimates (Free Tier Optimized)

| Service | Usage | Free Tier Limit | Monthly Cost (USD) |
|---------|-------|----------------|-------------------|
| Cloud Run (Scrapers) | 2 runs/day × 15 min × 0.5-1 vCPU | 360K vCPU-sec | **$0** (within free tier) |
| Cloud Functions (ETL) | 2 invocations/day × ~2 min | 2M invocations + 400K GB-sec | **$0** (within free tier) |
| Cloud Scheduler | 2 jobs × daily | 3 jobs free | **$0** |
| GCS Storage | ~2-3 GB (30-day retention, gzipped) | 5 GB free | **$0** |
| BigQuery | <5 GB storage, <100 GB queries | 10 GB + 1 TB free | **$0** |
| Cloud Logging | <10 GB/month | 50 GB free | **$0** |
| **Total** | | | **~$0-2/month** |

**Cost Optimization Tips:**
- Use minimum CPU/memory allocations (0.5 vCPU, 512MB for JobStreet)
- Enable gzip compression on all GCS uploads (5-10x size reduction)
- Delete raw data after 30 days (not 90)
- Partition BigQuery tables by date to reduce query costs
- Set max instances to 1 to prevent concurrent runs
- Build Docker images locally to avoid Cloud Build charges
- Use environment variables instead of Secret Manager

## Monitoring & Alerts

- **Cloud Monitoring Dashboard:** `deployment/monitoring_setup.md`
- **Log-based Alerts:**
  - Scraper failure (2 consecutive failures)
  - Scraper duration >20 minutes
  - GCS upload failures
- **Billing Alerts:** 50%, 80%, 100% of budget

## Documentation

- **Deployment Runbook:** `deployment/RUNBOOK.md` (coming soon)
- **Architecture Diagram:** `deployment/ARCHITECTURE.md` (coming soon)
- **Cloud Backend Agent:** `.github/agents/02_cloud_backend.agent.md`
- **Scraper Agent:** `.github/agents/01_scraper.agent.md`

## Troubleshooting

### Common Issues

**Scraper fails in Docker but works locally:**
- Check Chrome/ChromeDriver versions (MCF)
- Verify environment variables are set
- Check Cloud Run memory allocation (1GB for MCF)

**GCS upload permission denied:**
- Verify service account has `roles/storage.objectAdmin`
- Check bucket name matches `GCS_BUCKET` env var

**BigQuery schema mismatch:**
- Validate JSONL against `RawJob` schema
- Check field types (string, integer, timestamp)
- Use `python -m utils.bq --smoke-test` to test

**Cloud Scheduler not triggering:**
- Verify service account has `roles/run.invoker`
- Check timezone is set to `Asia/Singapore`
- Test with manual trigger: `gcloud scheduler jobs run <job-name>`

For detailed troubleshooting, see `deployment/RUNBOOK.md`.

========================================================
📝 DEVELOPMENT RULES FOR COPILOT
========================================================
- Always produce production-style code.
- Never produce “student exercises”.
- NEVER skip error handling.
- Prefer async where appropriate.
- For any script touching GCP:
  - Use environment variables, never hardcode project IDs.
- Ensure the entire pipeline is reproducible end-to-end.

========================================================
📌 WORKFLOW EXPECTATIONS FOR COPILOT
========================================================
When I ask for code, you must:

1. Identify which agent(s) should act.
2. Follow architecture constraints.
3. Follow folder structure.
4. Provide complete modules, not fragments.
5. Provide test examples when relevant.
6. Document every class & function.

========================================================
🎯 PRIMARY GOAL
========================================================
Help me build a cloud-ready, production-grade, ML-powered Singapore job market intelligence platform deployable on GCP within 7 days.

This is not a homework project; this is a portfolio-level, interview-ready system.

BEGIN NOW.