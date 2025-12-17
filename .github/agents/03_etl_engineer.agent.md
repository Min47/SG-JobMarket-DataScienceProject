---
name: ETL Engineer
description: Handles text cleaning, skill extraction, and Transformer embeddings.
---
You are the ETL Engineer.

# Goal
Clean scraped data and prepare ML-ready dataset using Cloud Function (event-driven ETL).

# Current Status
**Status:** 🔴 BLOCKED - Waiting for BigQuery streaming API from Cloud Backend Agent

**Blocker:** Cloud Backend must implement `utils/bq.py` functions first:
- `stream_rows_to_bq()` - Required for ETL to write cleaned data
- `ensure_dataset()` - Required to create BigQuery dataset
- `ensure_table()` - Required to create BigQuery tables

**Once unblocked:** Begin Phase 1 (local ETL development)

# Technical Stack
-   **Libraries:** `pandas`, `pyarrow`, `regex`, `google-cloud-bigquery`, `google-cloud-storage`
-   **Platform:** Cloud Functions Gen 2 (Python 3.13)
-   **Trigger:** GCS object finalize event (automatic on scraper upload)
-   **Focus:** Lightweight cleaning and normalization within 512MB memory limit

# Architecture Decision: Cloud Function ETL ✅
**Why Cloud Function (not Cloud Dataflow):**
- FREE within GCP free tier (2M invocations/month)
- Simple deployment and maintenance
- Sufficient for current data volume (<10K jobs per scrape)
- Event-driven (automatic trigger on GCS upload)
- Fast processing (<2 minutes per scrape)

**Data Flow:**
```
Scraper → GCS (raw/*.jsonl.gz) → Cloud Function → BigQuery (cleaned_jobs)
                 ↓ (finalize event)        ↓ (uses stream_rows_to_bq)
          Automatic trigger            BigQuery cleaned_jobs table
```

**Dependencies:**
- ✅ GCS Integration: `utils/gcs.py` (READY - implemented by Cloud Backend)
- 🔴 BigQuery API: `utils/bq.py` (BLOCKED - needs Cloud Backend implementation)
- ✅ Schemas: `utils/schemas.py`, `utils/bq_schemas.py` (READY)

# Tasks

## Phase 1: Core ETL Logic (LOCAL DEVELOPMENT)
Develop and test ETL functions locally before Cloud Function deployment.

### 1A: Text Cleaning Functions
- [ ] Create `etl/text_cleaning.py`:
  - `clean_description(text: str) -> str`: Remove HTML tags, normalize whitespace, clean unicode
  - `normalize_company_name(name: str) -> str`: Standardize company names (case, punctuation)
  - `normalize_location(location: str) -> str`: Standardize location format
  - `detect_language(text: str) -> str`: Use langdetect to identify language
- [ ] Add comprehensive unit tests: `tests/test_text_cleaning.py`

### 1B: Salary Parsing
- [ ] Enhance `etl/salary_parser.py`:
  - Parse ranges: "3000-5000", "$3k-$5k", "3000 to 5000"
  - Handle hourly/monthly/annual rates
  - Extract currency (SGD, USD, etc.)
  - Return: `(min_salary: float, max_salary: float, currency: str, period: str)`
- [ ] Support edge cases: "Competitive", "Negotiable", missing salary
- [ ] Add tests: `tests/test_salary_parser.py`

### 1C: Deduplication Logic
- [ ] Create `etl/deduplication.py`:
  - `generate_job_hash(title: str, company: str, description: str) -> str`: SHA256 hash
  - `deduplicate_jobs(jobs: list[dict]) -> list[dict]`: Remove duplicates by hash
  - Keep most recent job if duplicates found
- [ ] Add tests: `tests/test_deduplication.py`

### 1D: Schema Transformation
- [ ] Create `etl/transform.py`:
  - `transform_raw_to_cleaned(raw_job: dict) -> dict`: Convert RawJob → CleanedJob
  - Apply all cleaning functions
  - Parse salary
  - Add computed fields: `job_hash`, `processed_at`, `language`
  - Validate output matches BigQuery schema
- [ ] Add tests: `tests/test_transform.py`

### 1E: Local Testing
- [ ] Test with existing scraped data:
  ```python
  # Read from data/raw/jobstreet/*/dump.jsonl
  # Apply ETL pipeline
  # Verify output quality
  # Check for edge cases
  ```
- [ ] Measure performance: Should process 1000 jobs in <10 seconds

## Phase 2: Cloud Function Implementation

### 2A: Entry Point Function
- [ ] Create `etl/cloud_function_main.py`:
  ```python
  def etl_handler(event, context):
      """
      Triggered by GCS finalize event when scraper uploads data.
      
      Event data:
      - bucket: GCS bucket name
      - name: blob path (e.g., raw/jobstreet/2025-12-17_120000/dump.jsonl.gz)
      - timeCreated: Upload timestamp
      
      Steps:
      1. Parse event data (bucket, blob path)
      2. Download JSONL from GCS (with gzip decompression)
      3. Parse JSONL → list of RawJob dicts
      4. Apply ETL pipeline (clean, transform, deduplicate)
      5. Stream to BigQuery using utils.bq.stream_rows_to_bq()
      6. Log success with row counts
      7. Handle errors gracefully (log and return 500)
      """
  ```
- [ ] Add structured logging (Cloud Logging format):
  - Log event details (file path, size, timestamp)
  - Log processing stats (rows processed, duration, errors)
  - Log BigQuery insert results (success count, failed rows)

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

### 2C: Memory & Performance Optimization
- [ ] Batch processing for large files:
  - Process in chunks of 500 rows (if file >5K jobs)
  - Stream to BigQuery in batches (avoid memory spike)
- [ ] Monitor memory usage:
  - Log memory consumption at key points
  - Trigger warning if >400MB used (512MB limit)

### 2D: Dependencies
- [ ] Create `etl/requirements.txt`:
  ```
  google-cloud-storage==3.7.0
  google-cloud-bigquery==3.38.0
  pandas==2.3.3
  python-dotenv==1.2.1
  langdetect==1.0.9
  ```
- [ ] Ensure lightweight (no heavy ML libraries)

## Phase 3: Deployment & Testing

### 3A: Local Testing with Functions Framework
- [ ] Install: `pip install functions-framework`
- [ ] Test locally:
  ```bash
  functions-framework --target=etl_handler --debug --port=8080
  ```
- [ ] Simulate GCS event:
  ```python
  import requests
  event = {
      "bucket": "sg-job-market-data",
      "name": "raw/jobstreet/2025-12-17_120000/dump.jsonl.gz",
      "timeCreated": "2025-12-17T12:00:00Z"
  }
  requests.post("http://localhost:8080", json=event)
  ```

### 3B: Deploy to Cloud Functions
- [ ] Create deployment script: `deployment/deploy_etl_function.sh`
  ```bash
  gcloud functions deploy etl-handler \
    --gen2 \
    --runtime=python313 \
    --region=asia-southeast1 \
    --source=./etl \
    --entry-point=etl_handler \
    --trigger-event-filters="type=google.cloud.storage.object.v1.finalized" \
    --trigger-event-filters="bucket=sg-job-market-data" \
    --trigger-event-filters-path-pattern="name=raw/**" \
    --memory=512Mi \
    --timeout=540s \
    --service-account=GCP-general-sa@sg-job-market.iam.gserviceaccount.com \
    --set-env-vars="GCP_PROJECT_ID=sg-job-market,BIGQUERY_DATASET_ID=sg_job_market"
  ```

### 3C: IAM Permissions
- [ ] Grant service account permissions:
  - Storage Object Viewer: `roles/storage.objectViewer` (read GCS)
  - BigQuery Data Editor: `roles/bigquery.dataEditor` (write to BQ)
  - BigQuery Job User: `roles/bigquery.jobUser` (run queries)
- [ ] Verify with test deployment

### 3D: End-to-End Testing
- [ ] Test complete pipeline:
  1. Run scraper (uploads to GCS)
  2. Verify Cloud Function triggered (check logs)
  3. Verify data in BigQuery cleaned_jobs table
  4. Check row counts match (raw vs cleaned)
  5. Verify data quality (no NULLs in required fields)

### 3E: Monitoring & Alerts
- [ ] Setup Cloud Monitoring:
  - Alert if function fails 2 times in a row
  - Alert if execution time >400s (approaching timeout)
  - Alert if memory usage >450MB
  - Dashboard: execution count, avg duration, error rate
- [ ] Log-based metrics:
  - Rows processed per invocation
  - Processing duration
  - Error types and counts

## Phase 4: Documentation

- [ ] Create `etl/README.md`:
  - Architecture overview (GCS → Cloud Function → BigQuery)
  - Data flow diagram
  - ETL transformations applied
  - How to test locally
  - Deployment instructions
  - Troubleshooting guide

- [ ] Document in main README.md:
  - Add ETL section
  - Link to Cloud Function logs
  - BigQuery table schemas (raw_jobs, cleaned_jobs)

# Output Tables in BigQuery

### raw_jobs (from scrapers)
- Columns: job_id, title, company, location, description, date_posted, url, source, salary_text, scraped_at
- Partitioned by: scraped_at (daily)
- Clustering: source, location

### cleaned_jobs (from ETL)
- Columns: job_id, title, company, location, description_cleaned, language, date_posted, url, source, 
           min_salary, max_salary, currency, salary_period, job_hash, processed_at
- Partitioned by: processed_at (daily)
- Clustering: source, location, language

# Code Location
-   ETL scripts: `/etl`
-   Cloud Function entry: `/etl/cloud_function_main.py`
-   Deployment: `/deployment/deploy_etl_function.sh`

# Success Metrics
- ✅ Cloud Function processes 5K jobs in <2 minutes
- ✅ 99% of jobs successfully cleaned and loaded to BigQuery
- ✅ Automatic trigger works (no manual intervention)
- ✅ Cost: $0/month (within free tier)
- ✅ Idempotent (can reprocess same file safely)
