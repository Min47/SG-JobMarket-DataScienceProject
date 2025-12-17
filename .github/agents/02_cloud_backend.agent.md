---
name: Cloud Backend
description: Handles BigQuery integration, FastAPI development, Docker, and Cloud Run deployment.
---
You are the Cloud Backend Engineer.

# Goal
Enable seamless integration between application code and Google Cloud services (BigQuery, GCS, Cloud Run).

# Current Project Status

**Overall Progress:** ~85% Complete  
**Status:** Production infrastructure deployed and operational. Only BigQuery integration remains to complete the data pipeline.

## ✅ Completed (Foundation & Deployment)
- **Config Management:** `utils/config.py` loads from `.env`
  - `GCP_PROJECT_ID`, `BIGQUERY_DATASET_ID`, `GCP_REGION`, `GCS_BUCKET`
  - Validates required variables at startup
- **Logging Infrastructure:** `utils/logging.py`
  - Dual console + file output
  - Timestamped log files in `logs/`
  - Auto-rotation (keeps last 10 runs)
  - Format: `YYYY-MM-DD HH:MM:SS (LEVEL) | logger | message`
- **Retry Logic:** `utils/retry.py`
  - Exponential backoff with jitter
  - Async-compatible
  - Used by scrapers for network calls
- **Schemas:** `utils/schemas.py` + `utils/bq_schemas.py`
  - `RawJob` dataclass (scraper output contract)
  - `CleanedJob` dataclass (ETL output contract)
  - Auto-generated BigQuery schemas from dataclasses
  - Schema validation tools in `utils/schema_tools.py`
- **Scrapers:** Fully implemented (see 01_scraper.agent.md)
  - JobStreet: GraphQL-based with progressive rate limiting (10min→30min→1hr→abort)
  - MyCareersFuture: Selenium + API with Chrome restart every 10 pages
  - Output: Local JSONL files in `data/raw/{site}/{timestamp}/`
  - Automatic GCS upload with compression (`GCS_UPLOAD_ENABLED=true`)
  - 30-day local retention policy
  - Robust error handling (individual job/page failures don't break batches)
  - Incremental job ID checkpointing
  - Timeout protection: Page load (30s), script execution (30s), element wait (15s)
- **GCS Integration:** ✅ `utils/gcs.py` fully implemented and deployed
  - Upload/download with compression
  - List blobs with pagination
  - Path helpers and URI validation
  - Comprehensive error handling
  - Bucket `gs://sg-job-market-data` created and operational
  - Scrapers automatically upload to GCS after local write
- **Docker Containerization:** ✅ Production-ready and deployed
  - `Dockerfile.scraper.jobstreet` (multi-stage, <500MB)
  - `Dockerfile.scraper.mcf` (with Chrome, <800MB)
  - `docker-compose.yml` with full deployment documentation
  - `.dockerignore` configured
  - `cloudbuild.jobstreet.yaml` and `cloudbuild.mcf.yaml` created
  - Images built and pushed to Artifact Registry (asia-southeast1)
- **GCP Production Infrastructure:** ✅ Fully deployed and operational
  - **Service Account:** `GCP-general-sa@sg-job-market.iam.gserviceaccount.com`
  - **IAM Roles:** storage.admin, bigquery.dataEditor, run.invoker, cloudscheduler.admin
  - **Artifact Registry:** scraper-jobstreet-docker, scraper-mcf-docker (asia-southeast1)
  - **GCS Bucket:** `gs://sg-job-market-data` (asia-southeast1)
  - **Cloud Run Jobs:**
    - `cloudjob-scraper-jobstreet` (512Mi, 1 CPU, 1d timeout, max-retries=3)
    - `cloudjob-scraper-mcf` (1Gi, 1 CPU, 1d timeout, max-retries=3)
  - **Cloud Scheduler:**
    - `scheduler-scraper-jobstreet` (0 13 * * * = 9 PM SGT daily)
    - `scheduler-scraper-mcf` (0 1 * * * = 9 AM SGT daily)
  - **Automation:** Scrapers run daily automatically, upload to GCS with compression

## 🔲 To Be Implemented

### Phase 1: BigQuery Integration

Expand `utils/bq.py` with production-ready functions:

#### Phase 1A: Core Infrastructure
- [ ] `ensure_dataset()`: Create dataset if missing, handle already exists gracefully
- [ ] `ensure_table()`: Create tables with schema, support partitioning/clustering
- [ ] `get_table_schema()`: Retrieve existing table schema for validation
- [ ] `delete_table()`: Helper for testing/cleanup (optional)
- All operations use retry logic from `utils/retry.py`

#### Phase 1B: BigQuery Streaming API
- [ ] `stream_rows_to_bq()`: Stream data rows directly to BigQuery table
  - Use `insert_rows_json()` for efficient streaming
  - Batch rows in chunks (recommended: 500 rows per request)
  - Handle rate limiting and retry on transient errors
  - Return failed rows for logging/debugging
  - Support both raw_jobs and cleaned_jobs tables
- [ ] `load_jsonl_to_bq()`: Load local JSONL for testing/backfill
  - Use `load_table_from_file()` with `autodetect=False`
  - `write_disposition=WRITE_APPEND` for raw_jobs
  - Add data validation before load
- Test with existing scraped data in `data/raw/jobstreet/` and `data/raw/mcf/`

#### Phase 1C: BigQuery Query Helpers (Optional)
- [ ] `query_table()`: Execute queries with parameters
- [ ] `deduplicate_table()`: Remove duplicates by job_id
- [ ] `get_row_count()`: Get table statistics

#### Phase 1D: Testing & Validation
- [ ] Create comprehensive tests: `tests/test_bq_integration.py`
  - Mock BQ client for unit tests
  - Integration test with test dataset (requires GCP credentials)
  - Test schema validation
  - Test error handling (permission denied, quota exceeded)
  - Test idempotent operations (run twice, same result)
- [ ] Add smoke test: `python -m utils.bq --smoke-test`
  - Creates test dataset/table
  - Loads sample JSONL
  - Queries data back
  - Cleanup

**Acceptance Criteria:**
- ✅ Tables auto-created on first run
- ✅ Schema validation before load
- ✅ Idempotent operations
- ✅ Proper error logging with retry
- ✅ Handles large datasets (10K+ rows)

---

### Phase 2: GCS Integration ✅ COMPLETED

**Status:** Production implementation complete in `utils/gcs.py`
**GCS Bucket:** `gs://sg-job-market-data` created in `asia-southeast1`

#### Phase 2A: Setup & Dependencies ✅
- [x] Add `google-cloud-storage>=2.10.0` to requirements.txt
- [x] Create `GCSClient` class with Storage client initialization
- [x] Add GCS-specific retry policy (network errors, 503, 429)
- [x] Document IAM requirements: Storage Object Admin role

#### Phase 2B: Upload Operations ✅
- [x] `upload_file(local_path, gcs_uri)`: Upload single file to GCS
  - Support resumable uploads for files >5MB
  - Calculate and verify MD5 checksum
  - Add progress callback for large files
  - Return blob metadata (size, created timestamp)
- [x] `upload_jsonl(local_path, gcs_uri, compress=True)`: Upload JSONL with optional gzip
  - Compress on-the-fly to save bandwidth/storage
  - Naming: `dump.jsonl.gz` if compress=True
  - Stream upload for memory efficiency

#### Phase 2C: Download & List Operations ✅
- [x] `download_file(gcs_uri, local_path)`: Download from GCS to local
  - Create parent directories automatically
  - Verify checksum after download
  - Support resume for interrupted downloads
- [x] `list_blobs(bucket, prefix)`: List files in bucket/prefix
  - Return list of blob metadata (name, size, updated)
  - Support pagination for large result sets
  - Filter by file extension (optional)
- [x] `exists(gcs_uri)`: Check if blob exists

#### Phase 2D: Path Helpers & Standards ✅
- [x] `build_raw_path(source, timestamp)`: `gs://{bucket}/raw/{source}/{timestamp}/dump.jsonl`
- [x] `build_model_path(model_name, version)`: `gs://{bucket}/models/{model_name}/{version}/`
- [x] `parse_gcs_uri(uri)`: Extract bucket and blob_name from `gs://` URI
- [x] `validate_gcs_uri(uri)`: Check URI format
- **Note:** No processed/ path needed - Cloud Function streams directly to BigQuery

#### Phase 2E: Testing & Error Handling ✅
- [x] Handle common errors with helpful messages:
  - 403: Permission denied (check IAM roles)
  - 404: Bucket/blob not found
  - 429: Rate limit exceeded (use retry)
  - Network errors (use retry)

#### Phase 2F: Extend BaseScraper for GCS Upload ✅
- [x] Modify `scraper/base.py` to support GCS upload:
  - Add optional `gcs_upload` parameter to `BaseScraper.__init__`
  - Update `run()` method:
    - After writing local JSONL, check if GCS upload is enabled
    - If enabled, upload to `gs://{bucket}/raw/{source}/{timestamp}/dump.jsonl`
    - Use `utils.gcs.upload_jsonl` with compression
    - Log upload success with GCS URI
    - Handle upload failures gracefully (local file remains as backup)
  - Environment variable: `GCS_UPLOAD_ENABLED=true|false`
  - Keep local file regardless of upload success (backup strategy)

#### Phase 2G: Local Storage Cleanup - 30-Day Retention ✅
- [x] Modify `_cleanup_old_runs()` in `scraper/base.py`:
  - Current behavior: Keep last 10 runs
  - New behavior: Keep runs from last 30 days OR last 10 runs (whichever is more)
  - Calculate date threshold: today - 30 days
  - Parse timestamp from folder name (YYYY-MM-DD_HHMMSS)
  - Delete runs older than 30 days (excluding most recent 10)
- [x] Add environment variable: `LOCAL_RETENTION_DAYS=30` (configurable)
- [x] Log cleanup actions with dates and sizes freed

**Acceptance Criteria:** ✅ ALL MET
- ✅ Works with `Settings.gcs_bucket` - VERIFIED
- ✅ Retry logic on all network errors - IMPLEMENTED
- ✅ Streaming for large files (memory-efficient) - IMPLEMENTED
- ✅ Proper IAM permission error messages - IMPLEMENTED
- ✅ Scrapers auto-upload to GCS after local write - IMPLEMENTED
- ✅ Local cleanup keeps 30-day retention - IMPLEMENTED

---

### Phase 2H: Cloud Function ETL (Event-Driven)
**Decision:** Use Cloud Function (not Cloud Dataflow) for cost optimization  
**Owner:** 🎯 **ETL Engineer Agent** (see `03_etl_engineer.agent.md`)  
**Cloud Backend:** Provide BigQuery streaming API only

- [ ] **Cloud Backend responsibilities (THIS AGENT):**
  - Implement `stream_rows_to_bq()` in `utils/bq.py` (Phase 1B) - **PRIORITY**
  - Provide BigQuery schema validation helpers
  - Ensure retry logic works with Cloud Function timeouts
  - Test streaming API with sample data (>1000 rows)

- [ ] **ETL Engineer responsibilities (DIFFERENT AGENT - see 03_etl_engineer.agent.md):**
  - Create `etl/cloud_function_main.py` entry point
  - Implement data cleaning and transformation logic
  - Parse salary ranges, deduplicate, validate schemas
  - Call `stream_rows_to_bq()` to write to BigQuery
  - Handle errors gracefully (log and continue)
  - Deploy Cloud Function with GCS trigger

- [ ] **Deployment configuration:**
  - Cloud Functions Gen 2, Python 3.13 runtime
  - Event trigger: `google.storage.object.finalize`
  - Event filter: `--event-filters="bucket={bucket},prefix=raw/"`
  - Memory: 512MB, Timeout: 540s (9 min max)
  - Service account: `GCP-general-sa@sg-job-market.iam.gserviceaccount.com`
  - IAM: Storage Object Viewer, BigQuery Data Editor

**Why Cloud Function over Cloud Dataflow:**
- ✅ FREE (within free tier: 2M invocations/month)
- ✅ Simple deployment and maintenance
- ✅ Sub-minute processing for typical scrape sizes (<5K jobs)
- ✅ Event-driven (automatic trigger on GCS upload)
- ✅ Sufficient for current data volume

**Acceptance Criteria:**
- ✅ Cloud Function deploys successfully
- ✅ GCS finalize event triggers function automatically
- ✅ Cleaned data streams to BigQuery within 2 minutes of scraper finish
- ✅ Error handling with detailed logs
- ✅ Idempotent (can reprocess same file safely)
- ✅ Handles up to 10K jobs per scrape within timeout

---

### Phase 3: Docker Containerization

Create Docker infrastructure for Cloud Run deployment:

#### Phase 3A: Base Configuration Files ✅
- [x] Create `.dockerignore`:
  - Exclude: `.venv/`, `__pycache__/`, `*.pyc`, `.git/`, `logs/`, `data/`, `.env`, `.pytest_cache/`, `notebooks/`
  - Include: `requirements.txt`, `scraper/`, `utils/`, `etl/`, `api/`
- [x] Create `docker-compose.yml` for local testing:
  - Define services: `scraper-jobstreet`, `scraper-mcf`
  - Mount volumes for testing
  - Environment variable configuration

#### Phase 3B: Dockerfile.scraper.jobstreet (JobStreet) ✅
- [x] Create `Dockerfile.scraper.jobstreet`:
  - Base image: `python:3.13-slim-bookworm`
  - Multi-stage build:
    - Stage 1 (builder): Install build dependencies, create wheels
    - Stage 2 (runtime): Copy wheels and install
  - Install dependencies from `requirements.txt` (subset for scraper)
  - Create non-root user: `scraperuser`
  - Set working directory: `/app`
  - Copy code: `scraper/`, `utils/`
  - Entrypoint: `python -m scraper --site jobstreet`
  - Environment variables: `GCP_PROJECT_ID`, `GCS_BUCKET`, `SCRAPER_USER_AGENTS`
  - Health check: `HEALTHCHECK CMD python -c 'import sys; sys.exit(0)'`
  - **Memory target:** 512MB
  - Label with build metadata (git SHA, build date)

#### Phase 3C: Dockerfile.scraper.mcf (MCF with Chrome) ✅
- [x] Create `Dockerfile.scraper.mcf`:
  - Base image: `python:3.13-slim-bookworm`
  - Install Chrome dependencies:
    - `wget`, `gnupg`, `unzip`
    - Google Chrome Stable (from official repo)
    - ChromeDriver (matching Chrome version)
    - Set up headless Chrome environment
  - Install Python dependencies (including selenium)
  - Configure Chrome for containerized environment:
    - `--no-sandbox`, `--disable-dev-shm-usage` flags
    - Set `CHROME_BIN` and `CHROMEDRIVER_PATH` env vars
  - Create non-root user: `scraperuser`
  - Copy code: `scraper/`, `utils/`
  - Entrypoint: `python -m scraper --site mcf`
  - **Memory target:** 1GB (Chrome needs more RAM)
  - Add script to verify Chrome installation at build time

#### Phase 3D: Local Testing & Optimization ✅
- [x] Build both images and check sizes (target <500MB base, <800MB MCF)
- [x] Test locally with `docker run`:
  - Mount `.env` file
  - Test with `--dry-run` flag
  - Verify logs output correctly
  - Test GCS upload (if enabled)
- [x] Optimize layer caching:
  - Copy `requirements.txt` first, then `pip install`
  - Copy code last to maximize cache hits
- [x] Security scanning: `docker scan <image>`
- [x] Document build commands in `docs/docker_build.md`

#### Phase 3E: Dockerfile.api (Future)
- [ ] Create `Dockerfile.api` for FastAPI service:
  - Base image: `python:3.13-slim`
  - Install FastAPI, uvicorn, dependencies for ML inference
  - Copy `api/`, `utils/`, `ml/`, `models/`
  - Expose port 8080 (Cloud Run default)
  - Entrypoint: `uvicorn api.app:app --host 0.0.0.0 --port 8080`
  - Health check: `curl http://localhost:8080/health`
  - Memory target: 512MB (increase if loading large models)
  - Support model loading from GCS or baked into image
- **Note:** Implementation deferred until API Engineer completes `api/app.py`

**Acceptance Criteria:**
- ✅ Images build without errors
- ✅ Scrapers run successfully in containers
- ✅ Stateless operation (no local state dependencies)
- ✅ Image sizes optimized (<500MB base, <1GB MCF)
- ✅ Non-root user for security
- ✅ Environment-based configuration only

---

### Phase 4: Cloud Run Deployment

Deploy containerized scrapers to Google Cloud Run:

#### Phase 4A: Artifact Registry Setup ✅ COMPLETED
- [x] Create repository: `scraper-jobstreet-docker`, `scraper-mcf-docker`
  - Location: `asia-southeast1` (Singapore)
  - Format: Docker
- [x] Configure Docker authentication:
  - `gcloud auth configure-docker asia-southeast1-docker.pkg.dev`
- [x] Define image naming convention:
  - `asia-southeast1-docker.pkg.dev/sg-job-market/scraper-jobstreet-docker/sg-job-scraper:jobstreet`
  - `asia-southeast1-docker.pkg.dev/sg-job-market/scraper-mcf-docker/sg-job-scraper:mcf`
  - Tag with git SHA for versioning (future enhancement)
- [x] Service account: `GCP-general-sa@sg-job-market.iam.gserviceaccount.com`
  - Roles: `storage.admin`, `bigquery.dataEditor`, `run.invoker`, `cloudscheduler.admin`

#### Phase 4B: Service Account & IAM ✅ COMPLETED
- [x] Create service account: `GCP-general-sa@sg-job-market.iam.gserviceaccount.com`
- [x] Grant required roles:
  - Storage Admin: `roles/storage.admin`
  - BigQuery Data Editor: `roles/bigquery.dataEditor`
  - Cloud Run Invoker: `roles/run.invoker`
  - Cloud Scheduler Admin: `roles/cloudscheduler.admin`
- [x] Bind service account to Cloud Run Jobs
- [ ] Create JSON key for local testing (optional)
- [ ] Document IAM setup in `deployment/iam_setup.md` (Optional)
- [ ] Test permissions with `gcloud auth login --impersonate-service-account` (Optional)

#### Phase 4C: Cloud Build & Cloud Run Jobs ✅ COMPLETED
- [x] Build Docker images with Cloud Build:
  - `gcloud builds submit . --config=cloudbuild.jobstreet.yaml`
  - `gcloud builds submit . --config=cloudbuild.mcf.yaml`
- [x] Push to Artifact Registry - Images successfully pushed
- [x] Deploy Cloud Run Jobs:
  - `cloudjob-scraper-jobstreet` - Region: `asia-southeast1`
  - `cloudjob-scraper-mcf` - Region: `asia-southeast1`
  - Environment variables configured:
    - `GCP_PROJECT_ID=sg-job-market`
    - `BIGQUERY_DATASET_ID=sg_job_market`
    - `GCP_REGION=asia-southeast1`
    - `GCS_BUCKET=sg-job-market-data`
    - `GCS_UPLOAD_ENABLED=true`
    - `LOG_LEVEL=INFO`
  - Service account: `GCP-general-sa@sg-job-market.iam.gserviceaccount.com`
  - **CPU: 1 vCPU** (both scrapers)
  - **Memory: 512Mi (jobstreet), 1Gi (mcf)**
  - Timeout: 1 day (task-timeout)
  - Max retries: 3
- [ ] Optimize CPU allocation (consider 0.5 vCPU for jobstreet) (Future optimization)

#### Phase 4D: Environment Variable Management
- [ ] Create `.env.production` template:
  - `GCP_PROJECT_ID`
  - `BIGQUERY_DATASET_ID`
  - `GCS_BUCKET`
  - `GCP_REGION`
  - `SCRAPER_USER_AGENTS` (comma-separated, rotating list)
  - `GCS_UPLOAD_ENABLED=true`
  - `LOCAL_RETENTION_DAYS=30`
  - `LOG_LEVEL=INFO`
- [ ] Use Secret Manager for sensitive values:
  - `gcloud secrets create scraper-user-agents`
  - Mount as environment variable in Cloud Run
- [ ] Document in `deployment/environment_variables.md`
- [ ] Add validation in deploy script to check required vars

#### Phase 4E: Health Check & Graceful Shutdown
- [ ] Add production readiness features to scrapers:
  - Health check endpoint (optional HTTP server during scrape):
    - Simple Flask/FastAPI endpoint: `/health` → 200 OK
    - Check: ChromeDriver alive (MCF), Session initialized (JobStreet)
    - Runs on separate thread during scrape
  - Graceful shutdown handling:
    - Listen for SIGTERM signal
    - Finish current job batch before exit
    - Upload partial results to GCS
    - Set timeout grace period: 10 seconds
  - Add `--cloud-run` flag to scraper CLI to enable Cloud Run mode
  - Log startup/shutdown events with timestamps

#### Phase 4F: Monitoring & Logging
- [ ] Configure structured logging:
  - JSON format for Cloud Logging
  - Include trace ID, source, timestamp
  - `utils/logging.py`: Add `CloudLoggingHandler`
- [ ] Define key metrics:
  - Scrape duration (histogram)
  - Jobs scraped count (counter)
  - Error rate (counter)
  - GCS upload status (counter)
- [ ] Create Cloud Monitoring dashboard:
  - Scrape success rate (last 7 days)
  - Average scrape duration
  - Error log entries
- [ ] Setup log-based alerts:
  - Alert if scrape fails 2 times in a row
  - Alert if duration >20 minutes
- [ ] Document in `deployment/monitoring_setup.md`

**Acceptance Criteria:**
- ✅ Stateless containers (no local state)
- ✅ Env-based configuration only
- ✅ Graceful shutdown handling
- ✅ Health check endpoint (optional)
- ✅ Services deployable with single script
- ✅ Logs structured for Cloud Logging

---

### Phase 5: Cloud Scheduler Orchestration ✅ COMPLETED

Setup automated scheduling for daily scrapes:

#### Phase 5A: Job Creation (JobStreet) ✅
- [x] Create Cloud Scheduler job: `scheduler-scraper-jobstreet`
  - Schedule: `0 13 * * *` (1 PM UTC = 9 PM SGT daily)
  - Timezone: `Asia/Singapore`
  - Target: Cloud Run Job (`cloudjob-scraper-jobstreet`)
  - HTTP method: POST
  - Service account: `GCP-general-sa@sg-job-market.iam.gserviceaccount.com`
    - Has `roles/run.invoker` permission
  - URI: `https://run.googleapis.com/v2/projects/sg-job-market/locations/asia-southeast1/jobs/cloudjob-scraper-jobstreet:run`
  - Retry config: Default Cloud Scheduler retry policy
- [x] Created with gcloud command

#### Phase 5B: Job Creation (MCF) ✅
- [x] Create Cloud Scheduler job: `scheduler-scraper-mcf`
  - Schedule: `0 1 * * *` (1 AM UTC = 9 AM SGT daily, 12 hours after JobStreet)
  - Stagger to avoid overlapping resource usage
  - Target: Cloud Run Job (`cloudjob-scraper-mcf`)
  - HTTP method: POST
  - Service account: `GCP-general-sa@sg-job-market.iam.gserviceaccount.com`
  - URI: `https://run.googleapis.com/v2/projects/sg-job-market/locations/asia-southeast1/jobs/cloudjob-scraper-mcf:run`
  - Retry config: Default Cloud Scheduler retry policy
- [x] Test with `gcloud scheduler jobs run scheduler-scraper-mcf --location asia-southeast1`

#### Phase 5C: Testing & Validation ✅ COMPLETED
- [x] Manually trigger jobs:
  - `gcloud scheduler jobs run scheduler-scraper-jobstreet --location asia-southeast1`
  - `gcloud scheduler jobs run scheduler-scraper-mcf --location asia-southeast1`
- [x] Verify execution:
  - `gcloud run jobs executions list --region=asia-southeast1`
  - Jobs execute successfully
- [x] View logs:
  - `gcloud run jobs logs read cloudjob-scraper-jobstreet --region=asia-southeast1 --limit=20`
  - `gcloud run jobs logs read cloudjob-scraper-mcf --region=asia-southeast1 --limit=20`
- [x] JSONL uploaded to GCS (verified)
- [ ] BigQuery integration (pending Phase 1 implementation)
- [ ] Test failure scenarios and alerting (Future enhancement)

**Acceptance Criteria:**
- ✅ Daily triggers configured (9 PM SGT for JobStreet, 9 AM SGT for MCF) - **VERIFIED**
- ✅ Proper IAM permissions - **VERIFIED**
- ✅ Manual trigger capability for testing - **VERIFIED**
- [ ] Retry logic on transient failures (Default Cloud Scheduler retry)
- [ ] Alerts on persistent failures (Future enhancement)

**Evidence:** Commands executed and documented in `docker-compose.yml` lines 245-272

---

### Phase 6: Production Readiness & Optimization

Final polish for production deployment:

#### Phase 6A: Integration Testing - Local to Cloud Flow
- [ ] Create end-to-end integration test: `tests/integration/test_cloud_pipeline.py`
- [ ] Test scenarios:
  - Run scraper locally → file in local → verify file exists
  - Run scraper in Cloud Run → verify GCS output
  - Load GCS JSONL → BigQuery → query data back
  - Full pipeline: Scraper → GCS → BigQuery → verify row count
- [ ] Use test dataset/bucket for isolation
- [ ] Cleanup test resources after each run
- [ ] CI/CD compatible (can run in GitHub Actions)
- [ ] Document in `tests/README.md`

#### Phase 6B: Cost Optimization & Quotas
- [ ] Set resource limits:
  - **Cloud Run max instances: 1 per service (prevent concurrent runs)**
  - **Cloud Run CPU: 0.5 (JobStreet), 1 (MCF) - minimum allocation**
  - **Cloud Run memory: 512MB (JobStreet), 1GB (MCF)**
  - BigQuery: Set daily quota on project (if needed)
  - **GCS: Enable lifecycle policy (delete raw data >30 days, not 90)**
- [ ] Maximize free tier usage:
  - Cloud Run: 360K vCPU-seconds/month free
  - Cloud Scheduler: 3 jobs free
  - GCS: 5 GB storage free
  - BigQuery: 10 GB storage + 1 TB queries/month free
- [ ] Document in `docs/cost_analysis.md`
- [ ] Setup billing alerts: Alert at $1, $5, $10 (targeting $0-2/month)
- [ ] **Avoid paid services: Skip Secret Manager, Artifact Registry scanning, Cloud Build**

#### Phase 6C: Documentation - Deployment Runbook
- [ ] Create `deployment/RUNBOOK.md`:
  - Prerequisites (gcloud, Docker, permissions)
  - First-time setup (Artifact Registry, IAM, Scheduler)
  - Deployment procedure (step-by-step)
  - Rollback procedure
  - Troubleshooting common issues
  - Emergency contacts/escalation
- [ ] Create `deployment/ARCHITECTURE.md`:
  - System diagram (Cloud Scheduler → Cloud Run → GCS → BigQuery)
  - Data flow diagram
  - Security boundaries
  - Failure modes and recovery
- [ ] Include screenshots and example commands
- [ ] Add to README.md with links

#### Phase 6D: Terraform Infrastructure as Code (Optional)
- [ ] Create `terraform/main.tf`:
  - Define GCS buckets with lifecycle rules
  - Define BigQuery dataset and tables
  - Define Cloud Run services
  - Define Cloud Scheduler jobs
  - Define IAM bindings
- [ ] Create `terraform/variables.tf`: Project ID, region, etc.
- [ ] Create `terraform/outputs.tf`: Service URLs, bucket names
- [ ] Use modules for reusability
- [ ] State backend: GCS bucket (terraform-state)
- [ ] Document in `terraform/README.md`
- [ ] Commands: `terraform init`, `plan`, `apply`
- **Note:** Optional but recommended for production

#### Phase 6E: CI/CD Pipeline (GitHub Actions)
- [ ] Create `.github/workflows/deploy-scraper.yml`:
  - Trigger: Push to main branch, changes in `scraper/` or `utils/`
  - Steps:
    1. Checkout code
    2. Authenticate to GCP (Workload Identity)
    3. Build Docker images
    4. Run tests
    5. Push to Artifact Registry
    6. Deploy to Cloud Run (staging first, then prod)
  - Environment secrets: `GCP_PROJECT_ID`, `SERVICE_ACCOUNT_KEY`
- [ ] Create `.github/workflows/test.yml`:
  - Run on PR: Lint, unit tests, integration tests
- [ ] Setup branch protection: Require tests to pass
- [ ] Document in `.github/CONTRIBUTING.md`

**Acceptance Criteria:**
- ✅ End-to-end tests pass
- ✅ Cost estimates documented
- ✅ Deployment runbook complete
- ✅ Terraform config functional (optional)
- ✅ CI/CD pipeline operational

---

### Phase 7: Documentation Updates (HIGH PRIORITY)
**Status:** Not Started | **Estimated:** Half day

Update project documentation to reflect Cloud Backend progress:

- [ ] Update `README.md`:
  - Add Cloud Deployment section
  - Update architecture diagram with Cloud Run
  - Add links to deployment docs
  - Update Prerequisites (gcloud, Docker)
- [ ] Update `.github/agents/02_cloud_backend.agent.md`:
  - Mark completed phases as ✅
  - Update Current Project Status section
  - Add lessons learned / gotchas section
  - Add troubleshooting FAQ
- [ ] Create `deployment/README.md`: Central hub for all deployment docs
- [ ] Add badges: Build status, deployment status

# Current Code Locations & Deployment Status
- Config: `/utils/config.py` ✅
- Logging: `/utils/logging.py` ✅
- Retry: `/utils/retry.py` ✅
- Schemas: `/utils/schemas.py`, `/utils/bq_schemas.py` ✅
- **BigQuery:** `/utils/bq.py` 🔴 **NEEDS IMPLEMENTATION (CRITICAL BLOCKER)**
- **GCS:** `/utils/gcs.py` ✅ **FULLY IMPLEMENTED & DEPLOYED**
- Scrapers: `/scraper/jobstreet.py`, `/scraper/mcf.py` ✅ **PRODUCTION-READY**
- Base Scraper: `/scraper/base.py` ✅ (with GCS upload support)
- **Dockerfiles:** ✅ **DEPLOYED TO PRODUCTION**
  - `Dockerfile.scraper.jobstreet` ✅
  - `Dockerfile.scraper.mcf` ✅
  - `docker-compose.yml` ✅ (with full deployment history)
  - `.dockerignore` ✅
- **Cloud Build:** ✅ **IMAGES BUILT & PUSHED**
  - `cloudbuild.jobstreet.yaml` ✅
  - `cloudbuild.mcf.yaml` ✅
  - Images in `asia-southeast1-docker.pkg.dev/sg-job-market/`
- **GCP Infrastructure:** ✅ **FULLY OPERATIONAL**
  - Service Account: `GCP-general-sa@sg-job-market.iam.gserviceaccount.com`
  - GCS Bucket: `gs://sg-job-market-data`
  - Cloud Run Jobs: `cloudjob-scraper-jobstreet`, `cloudjob-scraper-mcf`
  - Cloud Scheduler: Daily automation (9 PM & 9 AM SGT)
  - IAM Roles: storage.admin, bigquery.dataEditor, run.invoker, cloudscheduler.admin
- Deployment scripts: `/deployment/` 🟡 **Optional** (manual commands work, documented in docker-compose.yml)
- API: `/api/app.py` 🔲 (API Engineer - different agent)
- ETL: `/etl/` 🔲 (ETL Engineer - blocked on BigQuery integration)

# Implementation Guidelines

## BigQuery Best Practices
- Use `load_table_from_file()` for JSONL/Parquet
- Set `write_disposition`:
  - `WRITE_APPEND` for raw_jobs (keep all historical data)
  - `WRITE_TRUNCATE` or merge logic for cleaned_jobs
- Enable schema auto-detection for flexibility
- Use partitioning by date for query performance
- Add clustering on frequently filtered columns

## GCS Best Practices
- Use resumable uploads for files >5MB
- Set object lifecycle rules (e.g., delete raw data after 90 days)
- Use signed URLs for temporary access
- Compress JSONL files (`.jsonl.gz`) to save storage costs

## Docker Best Practices
- Multi-stage builds to minimize image size
- Pin all dependencies (use requirements.txt with versions)
- Run as non-root user
- Use `.dockerignore` to exclude `.venv`, `logs`, `data`
- Cache pip dependencies in separate layer

## Cloud Run Best Practices
- Set max instances to control costs
- Use `--cpu-boost` for startup performance
- Set appropriate timeout (scraper: 15min, API: 5min)
- Use service accounts with minimal IAM permissions
- Tag images with git commit SHA for traceability

# Implementation Priority

## Critical Path Items (Updated Order)

🔴 **Start Here (Foundation for Cloud Deployment):**
- Phase 3: Docker containerization (items 1-4)
  - Get scrapers running in containers first
  - Test locally before deploying to cloud
- Phase 4: Cloud Run deployment (items 5-8)
  - Deploy containerized scrapers to Cloud Run
  - Setup service accounts and IAM
- Phase 5: Cloud Scheduler orchestration (items 9-12)
  - Automate daily scraper execution
  - Test end-to-end triggers

🟡 **Next Priority (Enable Full Pipeline):**
- Phase 2: GCS integration (items 13-19)
  - Enable scrapers to upload to GCS
  - Trigger Cloud Function ETL automatically
- Phase 1: BigQuery integration (items 20-23)
  - Stream cleaned data to BigQuery
  - Required by Cloud Function ETL

🟢 **Final Polish:**
- Phase 2H: Cloud Function ETL (item 24)
  - Event-driven data cleaning
- Phase 4E-4F: Production features (items 25-26)
  - Health checks, monitoring, alerts
- Phase 6: Production readiness (items 27-32)
  - Testing, documentation, CI/CD

## Rationale for New Order

**Why Docker First?**
1. Scrapers already work locally - containerize them immediately
2. Validate container builds before setting up cloud infrastructure
3. Test locally with docker-compose before Cloud Run costs
4. Enables rapid iteration on deployment configuration

**Why GCS/BQ After Cloud Run?**
1. Get scrapers running in cloud first (even without GCS upload)
2. Cloud Run can store data locally in container temporarily
3. Add GCS upload as enhancement after baseline works
4. BigQuery streaming needed by ETL, not by initial scraper deployment

## Success Metrics

After completing all phases, you should have:
- ✅ Scrapers running daily on Cloud Scheduler (2 AM & 3 AM SGT)
- ✅ Raw data automatically uploaded to GCS
- ✅ Data loaded into BigQuery for ETL processing
- ✅ Local backups with 30-day retention
- ✅ Structured logging and monitoring dashboards
- ✅ Cost-optimized cloud resources (maximize free tier usage)
- ✅ Comprehensive documentation for deployment and maintenance

## Cost Optimization Strategy (Free Tier Focus)

**GCP Free Tier Limits (Always Free):**
- Cloud Run: 2 million requests/month, 360,000 vCPU-seconds, 180,000 GiB-seconds memory
- Cloud Scheduler: 3 jobs (free)
- GCS: 5 GB Standard Storage
- BigQuery: 10 GB storage, 1 TB queries/month
- Cloud Logging: 50 GB/month
- Cloud Monitoring: Free tier available

**Optimization Strategies:**
1. **Cloud Run (Scrapers):**
   - Use minimum CPU (0.5 vCPU for JobStreet, 1 vCPU for MCF)
   - Minimum memory: 512MB (JobStreet), 1GB (MCF)
   - Max instances: 1 (prevent concurrent runs)
   - CPU allocation: "CPU is only allocated during request processing"
   - Estimated: 2 runs × 15 min × 2 services = ~$0.50/month (well within free tier)

2. **Cloud Functions (ETL):**
   - Memory: 512MB (sufficient for ETL processing)
   - Timeout: 540s (9 minutes max)
   - 2 invocations/day × ~2 min = well within 2M invocations/month free tier
   - Event-driven (no polling costs)

3. **Cloud Scheduler:**
   - 2 jobs = FREE (within 3 job limit)

3. **GCS:**
   - Enable gzip compression on JSONL uploads (5-10x reduction)
   - Lifecycle policy: Delete raw data after 30 days (not 90)
   - Store only essential files
   - Estimated: 2-3 GB usage = FREE

4. **BigQuery:**
   - Partition tables by date to reduce query costs
   - Use clustering on frequently queried columns
   - Avoid SELECT * queries
   - Use views for repeated queries
   - Estimated: <5 GB storage, <100 GB queries = FREE

5. **Cloud Logging:**
   - Set log retention to 30 days (default)
   - Filter out verbose debug logs in production
   - Use structured logging (more efficient)
   - Estimated: <10 GB/month = FREE

6. **Avoid Paid Services:**
   - Skip Secret Manager (use environment variables instead)
   - Skip Artifact Registry scanning ($0.26/image)
   - Skip Cloud Build (build locally, push manually)
   - Use default VPC (no extra networking costs)

**Target: $0-2/month (mostly free tier)**

## Dependencies & Handoffs

**Provides to ETL Engineer Agent (Cloud Function):**
- ✅ Raw JSONL in GCS: `gs://{bucket}/raw/{site}/{timestamp}/dump.jsonl` - **WORKING**
- ✅ GCS finalize event trigger (automatic) - **READY**
- 🔲 BigQuery streaming API (`stream_rows_to_bq()`) - **IN PROGRESS (PRIORITY)**

**Receives from ETL Engineer Agent:**
- 🔲 Cleaned data in BigQuery: `cleaned_jobs` table (streamed directly) - **BLOCKED ON ABOVE**

**Provides to ML Agent:**
- 🔲 Cleaned data in BigQuery: `cleaned_jobs` table (no GCS Parquet needed) - **BLOCKED ON ETL**

**Provides to API Agent:**
- 🔲 Model storage in GCS: `gs://{bucket}/models/{model_name}/{version}/` - **GCS READY**
- 🔲 Cloud Run deployment template: `Dockerfile.api` (Phase 3E) - **NOT STARTED**

## Next Immediate Steps (UPDATED - December 2025)

### 🔴 CRITICAL PATH (Only Blocker Remaining)
1. ⏭️ **Phase 1B: Implement BigQuery Streaming API** 🔥 HIGHEST PRIORITY
   - `stream_rows_to_bq()` in `utils/bq.py`
   - `ensure_dataset()`, `ensure_table()` helpers
   - `load_jsonl_to_bq()` for testing
   - Test with existing JSONL data in `data/raw/`
   - **This is the ONLY blocker for ETL Cloud Function development**

### 🟡 NEXT PHASE (ETL Engineer Agent - Waiting on BigQuery)
2. **Hand off to ETL Engineer:** Cloud Function ETL implementation
   - Create `etl/cloud_function_main.py`
   - Deploy with GCS finalize trigger
   - Test: GCS upload → automatic ETL → BigQuery
   - **Can start immediately after Phase 1B completes**

### 🟢 PRODUCTION POLISH (Can Do in Parallel)
3. **Phase 4F:** Monitoring & alerting setup
   - Structured logging for Cloud Logging
   - Cloud Monitoring dashboard
   - Log-based alerts
4. **Phase 6:** Documentation updates
   - Deployment runbook (`deployment/RUNBOOK.md`)
   - Architecture diagram (`deployment/ARCHITECTURE.md`)
   - Cost analysis with actual numbers
5. **Phase 6A:** Integration testing
   - End-to-end pipeline tests
   - Failure scenario testing

**Current Milestone:** ~85% complete on Cloud Backend work  
**Achievement:** ✅ Full production infrastructure deployed and operational!  
**Remaining:** Only BigQuery integration blocks completion of entire data pipeline
