---
name: ETL Engineer
description: Handles text cleaning, skill extraction, and Transformer embeddings.
---
You are the ETL Engineer.

# Goal
Clean scraped data and prepare ML-ready dataset using Cloud Function (event-driven ETL).

**Status:** ✅ **PHASE 1 & 2 COMPLETE - DEPLOYED TO PRODUCTION** (Dec 23, 2025)

**Implementation Results:**
- ✅ Stage 1: GCS → raw_jobs (deployed & operational)
- ✅ Stage 2: raw_jobs → cleaned_jobs (deployed & operational)
- ✅ Two-stage pipeline: 5,861+ jobs tested with 100% success rate
- ✅ Text cleaning: HTML removal, unicode normalization
- ✅ Salary parsing: Range extraction, monthly conversion
- ✅ Cloud Function: etl-gcs-to-bigquery deployed in asia-southeast1

**What's Left:** Monitor production runs, verify cleaned_jobs data quality

**Virtual Environment Usage:**
- ⚠️ **CRITICAL:** Always use `.venv/Scripts/python.exe` for all Python commands
- Install dependencies: `.venv/Scripts/python.exe -m pip install <package>`
- Run scripts: `.venv/Scripts/python.exe etl/pipeline.py`
- Update `requirements.txt` in the main directory when adding new dependencies

# Technical Stack
-   **Libraries:** `pandas`, `pyarrow`, `regex`, `google-cloud-bigquery`, `google-cloud-storage`
-   **Platform:** Cloud Functions Gen 2 (Python 3.13)
-   **Trigger:** GCS object finalize event (automatic on scraper upload)
-   **Focus:** Lightweight cleaning and normalization within 512MB memory limit

# Architecture Decision: Cloud Function ETL ✅

## Why Cloud Function (not Cloud Dataflow or Cloud Run Service)?

### Cloud Function vs Cloud Dataflow: Detailed Comparison

| Aspect | Cloud Function (CHOSEN ✅) | Cloud Dataflow |
|--------|---------------------------|----------------|
| **Cost** | **FREE** (2M invocations/month)<br>Estimated: $0/month for daily scraping | **$0.056/vCPU-hour + $0.003557/GB-hour**<br>Estimated: $50-200/month for daily runs |
| **Trigger** | Event-driven (automatic on GCS upload) | Manual start or scheduled (Cloud Scheduler needed) |
| **Execution** | Runs once per event, terminates immediately | Runs continuously until pipeline completes |
| **Cold Start** | 1-3 seconds (acceptable for batch ETL) | 3-5 minutes (pipeline initialization) |
| **Processing Speed** | <2 minutes for 10K jobs (tested) | Similar, but with initialization overhead |
| **Memory** | 512MB (sufficient for 10K jobs) | Configurable, but minimum billing applies |
| **Complexity** | Simple Python function | Apache Beam SDK (steeper learning curve) |
| **Deployment** | Single `gcloud functions deploy` command | Requires Beam pipeline definition + deployment |
| **Monitoring** | Cloud Logging + Cloud Monitoring (basic) | **Dataflow UI** (visual pipeline, detailed metrics) |
| **Scalability** | Handles up to 100K jobs (with batch processing) | Handles millions of records (distributed workers) |
| **Use Case Fit** | ✅ Daily batch ETL, <100K records/day | Large-scale streaming, >1M records/day |

### Visualization & Monitoring Comparison

#### Cloud Dataflow (Better Visualization ⭐)
**Pros:**
- 📊 **Visual Pipeline Graph:** See each step as a node in directed graph
- 📈 **Real-time Metrics:** Elements processed, throughput, CPU/memory per step
- 🔍 **Step-level Debugging:** Drill into specific transform failures
- ⏱️ **Performance Profiling:** Identify bottlenecks in pipeline stages
- 📉 **Historical Trends:** Compare pipeline runs over time

**Cons:**
- 💰 Expensive for small workloads (always pay for minimum workers)
- 🔧 Complex setup (requires Apache Beam knowledge)

#### Cloud Function (Simpler Monitoring ✅)
**Available:**
- ✅ **Cloud Logging:** Structured logs with severity levels
- ✅ **Cloud Monitoring Dashboards:** Custom metrics (execution time, success/failure rate)
- ✅ **Log-based Metrics:** Extract patterns from logs (e.g., rows processed)
- ✅ **Alerting:** Set up alerts for failures, timeouts, or slow executions
- ✅ **Error Reporting:** Automatic exception aggregation

**What You Get (Without Dataflow UI):**
```
Cloud Logging View:
  [INFO] Starting ETL: file=raw/jobstreet/2025-12-18_210000/dump.jsonl.gz, size_bytes=1.2M
  [INFO] Stage 1: Downloaded to /tmp/, rows=3869
  [INFO] Stage 1: Streamed to raw_jobs, success=3869/3869 (100%)
  [INFO] Stage 2: Transform started, input_rows=3869
  [INFO] Stage 2: Text cleaning complete, cleaned=3869
  [INFO] Stage 2: Salary parsing complete, parsed=2103 (54.4%)
  [INFO] Stage 2: Streamed to cleaned_jobs, success=3869/3869 (100%)
  [INFO] ETL complete: duration=87s, raw_rows=3869, cleaned_rows=3869
  
Cloud Monitoring Dashboard:
  📊 Execution Count: 2 runs today
  ⏱️ Avg Duration: 87 seconds
  ✅ Success Rate: 100%
  💾 Avg Memory: 312 MB (peak)
  📈 Rows Processed: 7,738 total (3,869 per run)
```

**DIY Pipeline Visualization:**
You can create a simple visual pipeline with:
1. **Looker Studio Dashboard:** Query BigQuery for ETL metrics
2. **BigQuery Views:** Create views that track pipeline stages
3. **Custom Logging:** Log stage progress with timestamps
4. **Grafana (Optional):** Export Cloud Monitoring metrics

### Decision Summary

**We chose Cloud Function because:**
1. ✅ **FREE** within GCP free tier (critical for personal project)
2. ✅ **Simple** to implement and maintain (no Apache Beam learning curve)
3. ✅ **Fast enough** for our scale (<10K jobs/day, processed in <2 minutes)
4. ✅ **Event-driven** architecture (no manual triggers needed)
5. ✅ **Sufficient monitoring** via Cloud Logging + Monitoring

**When to use Cloud Dataflow instead:**
- 📈 Scaling to >100K jobs/day
- 🔁 Complex multi-stage pipelines with branching logic
- 🌊 Streaming data (real-time processing)
- 🔍 Need visual pipeline debugging (Dataflow UI)
- 💰 Budget allows for $50-200/month operational cost

**Current Status:** Cloud Function is the right choice for Phase 1. Can migrate to Dataflow later if needed.

### Cloud Function vs Cloud Run Service

| Feature | Cloud Function | Cloud Run Service |
|---------|----------------|-------------------|
| Trigger | Event-driven (GCS, Pub/Sub) | HTTP requests or scheduled |
| Cost | FREE (2M invocations/month) | Pay per request + idle time |
| Execution | Runs once per event | Always-on or min instances |
| Use Case | ETL, data processing | APIs, web services |

**End-to-End Pipeline Architecture:**

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ PHASE 1: SCRAPING (Cloud Run Jobs - Already Deployed )                      │
├─────────────────────────────────────────────────────────────────────────────┤
│ Cloud Scheduler (JobStreet: Daily 9PM SGT, MCF: Daily 9AM SGT)              │
│         ↓                                                                   │
│ Cloud Run Job: jobstreet-scraper                                            │
│         ↓                                                                   │
│ Cloud Run Job: mcf-scraper                                                  │
│         ↓                                                                   │
│ GCS Upload: gs://sg-job-market-data/raw/{source}/{timestamp}/dump.jsonl.gz  │
└─────────────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│ PHASE 2: ETL TRIGGER (Event-Driven - Automatic)                             │
├─────────────────────────────────────────────────────────────────────────────┤
│ GCS Event: google.storage.object.v1.finalized                               │
│         ↓ (triggers within seconds)                                         │
│ Cloud Function: etl-gcs-to-bigquery (THIS IS YOUR IMPLEMENTATION)           │
└─────────────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│ STAGE 1: RAW INGESTION (Your Code)                                          │
├─────────────────────────────────────────────────────────────────────────────┤
│ 1. Download from GCS                                                        │
│    • Source: gs://sg-job-market-data/raw/{source}/{timestamp}/dump.jsonl.gz │
│    • Destination: /tmp/dump.jsonl.gz (Cloud Function temp storage)          │
│    • Use: utils.gcs.GCSClient.download_file()                               │
│                                                                             │
│ 2. Parse JSONL → RawJob Objects                                             │
│    • Read line-by-line (memory efficient)                                   │
│    • Validate against RawJob schema (utils.schemas.RawJob)                  │
│    • Add metadata: source, scrape_timestamp                                 │
│    • Handle malformed lines gracefully (log and skip)                       │
│                                                                             │
│ 3. Stream to BigQuery raw_jobs Table                                        │
│    • Use: utils.bq.stream_rows_to_bq()                                      │
│    • Batch size: 500 rows per batch                                         │
│    • Append-only: Never update/delete existing rows                         │
│    • Retry on transient errors (automatic in API)                           │
│                                                                             │
│ Output: raw_jobs table populated with ALL fields from scraper payload       │
│ Schema: job_id, source, scrape_timestamp, payload (JSON)                    │
│ Partitioning: By scrape_timestamp (TIMESTAMP)                               │
│ Clustering: source, job_id                                                  │
└─────────────────────────────────────────────────────────────────────────────┘
                              ↓
┌───────────────────────────────────────────────────────────────────────────────────┐
│ STAGE 2: TRANSFORMATION & CLEANING (Your Code)                                    │
├───────────────────────────────────────────────────────────────────────────────────┤
│ 1. Extract from payload JSON                                                      │
│    • Job fields: job_id, title, description, location, classification             │
│    • Company fields: company_id, name, description, industry, size                │
│    • Salary fields: min/max (raw), type, currency                                 │
│    • Timestamps: posted_timestamp, scrape_timestamp, bq_timestamp                 │
│                                                                                   │
│ 2. Text Cleaning & Normalization                                                  │
│    • HTML removal: BeautifulSoup4 (strip all tags from descriptions)              │
│    • Unicode normalization: Fix encoding issues, remove control chars             │
│    • Whitespace normalization: Strip, collapse multiple spaces                    │
│    • Company name standardization: Case normalization, remove punctuation         │
│    • Location standardization: Map to consistent format (e.g., "Central")         │
│                                                                                   │
│ 3. Salary Parsing & Conversion                                                    │
│    • Parse ranges: "3000-5000", "$3k-$5k", "3000 to 5000"                         │
│    • Extract min/max values (job_salary_min_sgd_raw, job_salary_max_sgd_raw)      │
│    • Identify period: hourly/daily/monthly/yearly (job_salary_type)               │
│    • Convert to monthly: job_salary_min_sgd_monthly, job_salary_max_sgd_monthly   │
│      - Hourly: × 160 (40 hrs/week × 4 weeks)                                      │
│      - Daily: × 22 (working days/month)                                           │
│      - Yearly: ÷ 12                                                               │
│    • Handle edge cases: "Competitive", "Negotiable", null                         │
│    • Currency: All SGD for now (job_currency = "SGD")                             │
│                                                                                   │
│ 4. Language Detection                                                             │
│    • Use: langdetect library (supports 55+ languages)                             │
│    • Apply to: job_title + job_description (combined text)                        │
│    • Output: ISO 639-1 code (en, zh, ms, ta, etc.)                                │
│    • Fallback: "unknown" if detection fails                                       │
│                                                                                   │
│ 5. Data Quality Validation                                                        │
│    • Required fields: Ensure not null/empty                                       │
│      - job_id, job_title, company_name, source                                    │
│    • URL validation: Check format for job_url, company_url                        │
│    • Date validation: Ensure job_posted_timestamp <= scrape_timestamp             │
│    • Salary validation: min <= max (if both present)                              │
│    • Log warnings for incomplete records (but still insert)                       │
│                                                                                   │
│ 6. Enrich with Timestamps                                                         │
│    • scrape_timestamp: From raw_jobs (preserve original)                          │
│    • bq_timestamp: datetime.utcnow() at transformation time                       │
│    • job_posted_timestamp: Parsed from payload                                    │
│                                                                                   │
│ 7. Stream to BigQuery cleaned_jobs Table                                          │
│    • Use: utils.bq.stream_rows_to_bq()                                            │
│    • Validate against CleanedJob schema (utils.schemas.CleanedJob)                │
│    • Batch size: 500 rows per batch                                               │
│    • Append-only: Preserve full data lineage                                      │
│                                                                                   │
│ Output: cleaned_jobs table ready for ML/Analytics                                 │
│ Schema (utils.schemas.CleanedJob):                                                │
│   - source, scrape_timestamp, bq_timestamp                                        │
│   - job_id, job_url, job_title, job_description, job_location                     │
│   - job_classification, job_work_type                                             │
│   - job_salary_min_sgd_raw, job_salary_max_sgd_raw, job_salary_type               │
│   - job_salary_min_sgd_monthly, job_salary_max_sgd_monthly, job_currency          │
│   - job_posted_timestamp                                                          │
│   - company_id, company_url, company_name, company_description                    │
│   - company_industry, company_size                                                │
│ Partitioning: By scrape_timestamp (TIMESTAMP; primary partition field)            │
│ Clustering: source, job_id, company_name                                          │
└───────────────────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│ PHASE 3: DOWNSTREAM CONSUMERS (Future Phases)                               │
├─────────────────────────────────────────────────────────────────────────────┤
│ • ML Engineer: Feature engineering, salary prediction, clustering           │
│ • GenAI Agent: RAG retrieval, semantic search, job recommendations          │
│ • Dashboard: Real-time analytics, trend visualization, company insights     │
│ • API: REST endpoints for external consumers                                │
└─────────────────────────────────────────────────────────────────────────────┘

**Deployment Details:**
- **Platform:** Cloud Functions Gen 2 (Python 3.13 runtime)
- **Trigger:** `--trigger-event=google.storage.object.finalize`
- **Filter:** `--trigger-resource=gs://sg-job-market-data --event-filters="bucket=sg-job-market-data,prefix=raw/"`
- **Memory:** 512MB (sufficient for 10K jobs)
- **Timeout:** 540s (9 minutes max)
- **Temp Storage:** `/tmp/` directory (2GB available, auto-cleaned after execution)
- **Service Account:** `GCP-general-sa@sg-job-market.iam.gserviceaccount.com`
- **IAM Roles:** Storage Object Viewer, BigQuery Data Editor

**Dependencies:**
- ✅ GCS Integration: `utils/gcs.py` (READY - implemented by Cloud Backend)
- ✅ BigQuery API: `utils/bq.py` (READY - implemented by Cloud Backend)
- ✅ Schemas: `utils/schemas.py`, `utils/bq_schemas.py` (READY)

# Tasks

## Phase 1: Core ETL Logic (LOCAL DEVELOPMENT)
Develop and test ETL functions locally before Cloud Function deployment.

### 1A: Combined Cloud Function Entry Point (RECOMMENDED APPROACH) ✅ COMPLETE
**Why combined:** Simpler architecture, fewer moving parts, no Pub/Sub setup needed.

- [x] Create `etl/cloud_function_main.py`:
  - **Function:** `process_gcs_upload(event, context)` - Handles both Stage 1 & 2 in single execution
  - **Triggered by:** GCS Object Finalize event (automatic when scraper uploads JSONL)
  - **Stage 1 logic:** Download JSONL from GCS to `/tmp/` → Stream to raw_jobs
  - **Stage 2 logic:** Transform RawJob → CleanedJob → Stream to cleaned_jobs
  - **Temp storage:** `/tmp/dump.jsonl.gz` (Cloud Function temp directory, auto-cleaned)
  
**Function signature:**
```python
def process_gcs_upload(event, context):
    """Cloud Function triggered by GCS object finalize.
    
    Executes complete ETL pipeline in single run:
    1. Download JSONL from GCS to /tmp/ (Cloud Function temp storage)
    2. Parse JSONL into RawJob objects
    3. Stream to BigQuery raw_jobs table (Stage 1 complete)
    4. Transform RawJob → CleanedJob objects
    5. Stream to BigQuery cleaned_jobs table (Stage 2 complete)
    
    Args:
        event (dict): GCS event data
            - name: File path (e.g., "raw/jobstreet/2025-12-18_210000/dump.jsonl.gz")
            - bucket: Bucket name ("sg-job-market-data")
        context: Event metadata (timestamp, event_id, etc.)
    
    Returns:
        str: Success message with row counts
    """
```

- [x] Test locally with `data/raw/jobstreet/` and `data/raw/mcf/` files (5,861+ jobs tested)
- [x] Add unit tests: `tests/test_two_stage_pipeline.py` (comprehensive testing done)

### 1A-alt: Separate Stage Functions (ALTERNATIVE - More Complex)
**Only use if you need separate concerns or have BigQuery-specific triggers.**

- [ ] Create `etl/stage1_load_raw.py`:
  - `load_jsonl_from_gcs_to_bq(event, context)`: GCS → raw_jobs
  - Downloads JSONL from GCS using `utils.gcs.GCSClient.download_file()` to `/tmp/`
  - Calls `utils.bq.load_jsonl_to_bq()` to stream to BigQuery raw_jobs
  - Returns row count
- [ ] Create `etl/stage2_clean_data.py`:
  - `transform_raw_to_cleaned(event, context)`: raw_jobs → cleaned_jobs
  - Triggered by Pub/Sub notification from BigQuery (requires additional setup)
  - Queries raw_jobs for new records
  - Transforms and streams to cleaned_jobs
- [ ] Test with local files and add tests: `tests/test_stage1.py`, `tests/test_stage2.py`

**Recommendation:** Use Phase 1A (combined function) unless you have specific reasons to separate.

### 1B: Stage 2 - Text Cleaning Functions ✅ COMPLETE
- [x] Create `etl/text_cleaning.py`:
  - `clean_description(text: str) -> str`: Remove HTML tags, normalize whitespace, clean unicode
  - `normalize_company_name(name: str) -> str`: Standardize company names (case, punctuation)
  - `normalize_location(location: str) -> str`: Standardize location format
  - `detect_language(text: str) -> str`: Use langdetect to identify language
- [x] Tested via integration tests (text cleaning working correctly)

### 1C: Stage 2 - Salary Parsing ✅ COMPLETE
- [x] Enhance `etl/salary_parser.py`:
  - Parse ranges: "3000-5000", "$3k-$5k", "3000 to 5000"
  - Handle hourly/monthly/annual rates
  - Extract currency (SGD, USD, etc.)
  - Return: `(min_salary: float, max_salary: float, currency: str, period: str)`
- [x] Support edge cases: "Competitive", "Negotiable", missing salary
- [x] Tested via integration tests (salary parsing working correctly)

### 1D: Stage 2 - Transformation Pipeline ✅ COMPLETE
- [x] Implemented in `etl/cloud_function_main.py` (stage2_transform_to_cleaned function):
  - `transform_raw_to_cleaned()`: Read from `raw_jobs`, transform, write to `cleaned_jobs`
  - Apply text cleaning, salary parsing, field extraction
  - Append-only writes (no updates)
  - Log transformation statistics
- [ ] **Note on Deduplication:** NO deduplication in ETL
  - ETL always appends new rows (never updates/deletes)
  - Deduplication happens at query time using `ROW_NUMBER()`:
    ```sql
    SELECT * FROM (
      SELECT *, ROW_NUMBER() OVER (
        PARTITION BY source, job_id 
        ORDER BY scrape_timestamp DESC
      ) AS rn FROM cleaned_jobs
    ) WHERE rn = 1
    ```
- [ ] Create helper function `etl/query_helpers.py`:
  - `get_latest_jobs()`: Query helper that applies `ROW_NUMBER()` deduplication
  - `generate_job_hash(title: str, company: str, description: str) -> str`: SHA256 hash
  - `deduplicate_jobs(jobs: list[dict]) -> list[dict]`: Remove duplicates by hash
  - Keep most recent job if duplicates found
- [ ] Add tests: `tests/test_deduplication.py`

### 1E: Schema Transformation ✅ COMPLETE
- [x] Create `etl/transform.py`:
  - `transform_raw_to_cleaned(raw_job: dict) -> dict`: Convert RawJob → CleanedJob
  - Apply all cleaning functions
  - Parse salary
  - Add computed fields: `job_hash`, `processed_at`, `language`
  - Validate output matches BigQuery schema
- [x] Tested via integration tests (transformation working correctly)

### 1F: Local Testing ✅ COMPLETE
- [x] Test with existing scraped data:
  ```python
  # Read from data/raw/jobstreet/*/dump.jsonl
  # Apply ETL pipeline
  # Verify output quality
  # Check for edge cases
  ```
- [x] Performance validated: 2,000+ jobs processed in ~90-120 seconds (within Cloud Function limits)

## Phase 2: Cloud Function Deployment

**Note:** Stage 1 deployed ✅. Need to redeploy with Stage 2 code.

### 2A: Deployment Configuration ✅ COMPLETE
- [x] Created `etl/main.py` (Cloud Function entry point)
- [x] Deployed Cloud Function: etl-gcs-to-bigquery (Stage 1 + Stage 2)
- [x] Function config: 512MB memory, 540s timeout, asia-southeast1
- [x] Event trigger: GCS object finalize on bucket sg-job-market-data
- [x] Service account: GCP-general-sa@sg-job-market.iam.gserviceaccount.com

### 2B: Structured Logging for Cloud Logging
- [ ] Enhance logging in `cloud_function_main.py`:
  ```python
  import logging
  import json
  from datetime import datetime
  
  def log_structured(severity: str, message: str, **fields):
      """Log in Cloud Logging JSON format."""
      entry = {
          "severity": severity,
          "message": message,
          "timestamp": datetime.utcnow().isoformat(),
          **fields
      }
      print(json.dumps(entry))  # Cloud Functions captures stdout
  
  # Usage in function:
  log_structured("INFO", "Starting ETL", 
                 file_path=file_path, 
                 bucket=bucket,
                 size_bytes=event.get('size'))
  
  log_structured("INFO", "ETL complete", 
                 raw_rows=raw_count,
                 cleaned_rows=cleaned_count,
                 duration_seconds=duration)

### 2B: Error Handling & Retry
- [ ] Handle common errors:
  - GCS download failure (retry 3x)
  - JSONL parsing error (log bad rows, continue)
  - BigQuery streaming error (retry with exponential backoff)
  - Timeout approaching (log partial progress, exit gracefully)
- [ ] Implement idempotency:
  - Track processed files in BigQuery metadata table (optional)
  - OR use BigQuery insert ID to prevent duplicate inserts
  - Allow safe reprocessing of same file

### 2C: Memory & Performance Optimization ✅ IMPLEMENTED
- [x] Batch processing: 500 rows per batch
- [x] Memory efficient: Streaming API, no full file load
- [x] Logging at key stages for monitoring

## Phase 3: Monitoring & Documentation (NEXT)

### 3A: Production Monitoring
- [ ] Verify end-to-end pipeline:
  - Monitor next scraper run (JobStreet 9PM SGT, MCF 9AM SGT)
  - Check Cloud Function logs for both stages
  - Verify cleaned_jobs table populated
  - Validate data quality (row counts, no NULLs in required fields)

### 3B: Setup Alerts (Optional)
- [ ] Cloud Monitoring alerts:
  - Function failures >2 in a row
  - Execution time >400s (approaching timeout)
  - Memory usage >450MB
- [ ] Log-based metrics dashboard

# Output Tables in BigQuery

### raw_jobs (from scrapers)
- Columns: job_id, source, scrape_timestamp, payload (JSON)
- Partitioned by: scrape_timestamp (daily)
- Clustering: source, job_id

### cleaned_jobs (from ETL)
- Columns: 
  - source, scrape_timestamp, bq_timestamp
  - job_id, job_url, job_title, job_description, job_location
  - job_classification, job_work_type
  - job_salary_min_sgd_raw, job_salary_max_sgd_raw, job_salary_type
  - job_salary_min_sgd_monthly, job_salary_max_sgd_monthly, job_currency
  - job_posted_timestamp
  - company_id, company_url, company_name, company_description
  - company_industry, company_size
- Partitioned by: scrape_timestamp (TIMESTAMP; primary partition field)
- Clustering: source, job_id, company_name

# Code Location
-   ETL scripts: `/etl`
-   Cloud Function entry: `/etl/cloud_function_main.py`
-   Deployment: `/deployment/deploy_etl_function.sh`

# Success Metrics (ALL MET ✅)
- ✅ Cloud Function deployed and operational
- ✅ Processes 5,861+ jobs tested with 100% success rate
- ✅ Two-stage pipeline: Stage 1 (raw) + Stage 2 (cleaned)
- ✅ Automatic GCS trigger working
- ✅ Cost: $0/month (within free tier)
- ✅ Memory: <512MB, Duration: <540s
