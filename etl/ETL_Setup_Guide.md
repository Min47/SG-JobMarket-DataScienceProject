# ETL Pipeline Setup & Operations Guide

**Event-driven ETL: JSONL → BigQuery raw_jobs**

---

## Table of Contents

1. [Pipeline Flow: JSONL to raw_jobs](#pipeline-flow-jsonl-to-raw_jobs)
2. [Deployment](#deployment)
3. [Monitoring & Verification](#monitoring--verification)
4. [Troubleshooting](#troubleshooting)
5. [FAQ](#faq)

---

## Pipeline Flow: JSONL to raw_jobs

### Architecture Overview

```
Scraper (Cloud Run Job)
    ↓ uploads dump.jsonl.gz
GCS (gs://sg-job-market-data/raw/{source}/{timestamp}/)
    ↓ triggers (object.finalize event)
Cloud Function (etl-gcs-to-bigquery)
    ↓ downloads, parses, streams
BigQuery raw_jobs table
```

### Step-by-Step Flow

**1. Scraper Uploads File**
```python
# Path: raw/jobstreet/2025-12-19_210000/dump.jsonl.gz
# Format: Compressed JSONL (gzip)
# Size: ~1-2 MB
# Trigger: Completes → GCS finalize event fires
```

**2. Cloud Function Triggered**
```
T+0s:   GCS object finalized
T+1s:   Cloud Function starts (cold start: 1-3s)
T+2s:   Downloads JSONL to /tmp/
T+5s:   Parses JSONL line-by-line
T+10s:  Streams to BigQuery raw_jobs (batch: 500 rows)
T+60s:  Complete ✅
```

**3. Data in BigQuery**
```sql
-- Check latest data
SELECT source, scrape_timestamp, COUNT(*) as jobs
FROM raw_jobs
WHERE DATE(scrape_timestamp) = CURRENT_DATE()
GROUP BY source, scrape_timestamp
ORDER BY scrape_timestamp DESC;
```

### File Filtering (Code-Level)

**GCP Trigger:** Bucket-level filter only
```bash
--trigger-event-filters="bucket=sg-job-market-data"
# Triggers on ANY file in this bucket
```

**Code Validation:** File-level filter (etl/main.py)
```python
# 1. Path: raw/{source}/{timestamp}/...
# 2. Filename: dump.jsonl or dump.jsonl.gz
# 3. Source: jobstreet or mcf

# Examples:
# ✅ raw/jobstreet/2025-12-19_210000/dump.jsonl.gz → Process
# ✅ raw/mcf/2025-12-19_010000/dump.jsonl → Process
# ⚠️ raw/jobstreet/job_ids.txt → Skip (not dump.jsonl)
# ❌ processed/data.jsonl → Error (wrong path)
```

### Deduplication Strategy

**Write-time:** Append-only (no dedup)
```python
# ETL always inserts new rows
# Multiple runs = multiple rows in raw_jobs
```

**Query-time:** ROW_NUMBER() deduplication
```sql
SELECT * FROM (
  SELECT *, 
    ROW_NUMBER() OVER (
      PARTITION BY source, job_id, scrape_timestamp
      ORDER BY bq_timestamp DESC
    ) AS rn
  FROM raw_jobs
) WHERE rn = 1;
```

**Benefits:**
- Full data lineage preserved
- Safe retries (idempotent queries)
- Time-travel queries possible

---

## Deployment

### Prerequisites

```bash
# Set GCP project
gcloud config set project sg-job-market

# Enable APIs
gcloud services enable cloudfunctions.googleapis.com cloudbuild.googleapis.com bigquery.googleapis.com storage.googleapis.com eventarc.googleapis.com pubsub.googleapis.com run.googleapis.com

# Config set
gcloud config set run/region asia-southeast1
gcloud config set run/platform managed
gcloud config set eventarc/location asia-southeast1

# Grant IAM roles
gcloud projects add-iam-policy-binding sg-job-market --member="serviceAccount:GCP-general-sa@sg-job-market.iam.gserviceaccount.com" --role="roles/storage.objectViewer"

gcloud projects add-iam-policy-binding sg-job-market --member="serviceAccount:GCP-general-sa@sg-job-market.iam.gserviceaccount.com" --role="roles/bigquery.dataEditor"

gcloud projects add-iam-policy-binding sg-job-market --member="serviceAccount:GCP-general-sa@sg-job-market.iam.gserviceaccount.com" --role="roles/bigquery.jobUser"

gcloud projects add-iam-policy-binding sg-job-market --member="serviceAccount:GCP-general-sa@sg-job-market.iam.gserviceaccount.com" --role="roles/eventarc.eventReceiver"

# Grants the GCS service account permission to publish events to Pub/Sub
# This is required for GCS → Cloud Functions event triggers
# It's a one-time setup
gcloud projects add-iam-policy-binding sg-job-market --member="serviceAccount:service-692289959200@gs-project-accounts.iam.gserviceaccount.com" --role="roles/pubsub.publisher"
```

### Deploy Cloud Function

**Step 1: Prepare deployment package**
```powershell
# Run the deployment preparation script (Windows PowerShell)
.\deployment\prepare_deploy.ps1

# Or manually (if script fails):
mkdir .deploy_temp
cp etl/main.py .deploy_temp/main.py
cp -r etl/ .deploy_temp/etl/
cp -r utils/ .deploy_temp/utils/
cp requirements.txt .deploy_temp/requirements.txt
```

**Step 2: Deploy to GCP**
```bash
gcloud functions delete etl-gcs-to-bigquery --region=asia-southeast1 --gen2 --quiet

gcloud functions deploy etl-gcs-to-bigquery --gen2 --runtime=python313 --region=asia-southeast1 --source=.deploy_temp --entry-point=etl_gcs_to_bigquery --trigger-event-filters="type=google.cloud.storage.object.v1.finalized" --trigger-event-filters="bucket=sg-job-market-data" --memory=512MB --timeout=540s --service-account=GCP-general-sa@sg-job-market.iam.gserviceaccount.com --set-env-vars="GCP_PROJECT_ID=sg-job-market,BIGQUERY_DATASET_ID=sg_job_market,GCP_REGION=asia-southeast1,GCS_BUCKET=sg-job-market-data"
```

**Step 3: Cleanup**
```bash
rm -rf .deploy_temp
```

```powershell
Remove-Item -Path .deploy_temp -Recurse -Force
```

### Verify Deployment

```bash
# Check function status
gcloud functions describe etl-gcs-to-bigquery --region=asia-southeast1 --gen2

# Check logs
gcloud functions logs read etl-gcs-to-bigquery --region=asia-southeast1 --gen2 --limit=20

# Test trigger (upload file - without .gz since local file is uncompressed)
gcloud storage cp data/raw/jobstreet/2025-12-16_044220/dump.jsonl gs://sg-job-market-data/raw/jobstreet/2025-12-19_010203/dump.jsonl
gcloud storage cp data/raw/mcf/2025-12-16_045134/dump.jsonl gs://sg-job-market-data/raw/mcf/2025-12-19_020304/dump.jsonl
```


### Deployment Behavior & Costs

**Q: Does it start running immediately after deployment?**
- ✅ Yes, it's **event-driven** - automatically listens for GCS uploads
- ⏸️ **No background processes** - only runs when triggered (when scraper uploads JSONL)
- 🎯 Cold start: 1-3 seconds when first triggered after idle period

**Q: Is it costly to keep running?**
- 💰 **FREE** for typical usage (Cloud Functions free tier: 2M invocations/month, 400K GB-seconds)
- 📊 Expected usage: **2 scrapes/day × 30 days = 60 invocations/month** (0.003% of free tier)
- ⚡ Each execution: ~60s × 512MB = 0.0084 GB-seconds (0.000002% of free tier)
- 💵 **Estimated monthly cost: $0.00** (well within free tier limits)

**Q: What if I need to update? Will redeployment replace or create junk?**
- ♻️ **Replaces the function** - no junk files or duplicates
- 🔄 Zero-downtime deployment (old version serves requests until new one is ready)
- 📦 Old Cloud Build artifacts auto-deleted after 120 days (default GCP policy)
- 🧹 Each deployment creates new container image but GCP manages cleanup automatically

**Q: Why is Cloud Build slow during deployment?**
- ⏱️ Normal: 2-5 minutes for Python dependencies installation
- 🐳 Docker alternative (Cloud Run): Similar build time + more complexity (Dockerfile, docker build, push to Artifact Registry)
- ✅ Recommendation: **Stick with Cloud Functions** - simpler, auto-scales, same performance as Docker for this use case
- 🚀 If build time is critical: Use Cloud Build triggers with pre-built base images (advanced setup)

---

## Monitoring & Verification

### Daily Verification

**Option 1: Cloud Console (Recommended - Fast & Visual)**
```
# Check today's scrapes
https://console.cloud.google.com/bigquery?project=sg-job-market&ws=!1m5!1m4!4m3!1ssg-job-market!2ssg_job_market!3sraw_jobs

# Run this query in the editor:
SELECT source, scrape_timestamp, 
  COUNT(*) as job_count, 
  COUNT(DISTINCT job_id) as unique_jobs 
FROM `sg-job-market.sg_job_market.raw_jobs` 
WHERE DATE(scrape_timestamp) = CURRENT_DATE() 
GROUP BY source, scrape_timestamp 
ORDER BY scrape_timestamp DESC

# Check for duplicates (same query method as above):
SELECT source, job_id, scrape_timestamp, 
  COUNT(*) as duplicate_count 
FROM `sg-job-market.sg_job_market.raw_jobs` 
WHERE DATE(scrape_timestamp) = CURRENT_DATE() 
GROUP BY source, job_id, scrape_timestamp 
HAVING COUNT(*) > 1 
LIMIT 10
```

**Option 2: Cloud Logs Explorer (For Function Execution Logs)**
```
# View ETL function logs:
https://console.cloud.google.com/logs/query?project=sg-job-market

# Use this filter in the query editor:
resource.type="cloud_function"
resource.labels.function_name="etl-gcs-to-bigquery"
resource.labels.region="asia-southeast1"
timestamp>="2025-12-19T00:00:00Z"
```

**Why not use `bq` CLI?**
- ⚠️ Slow on Windows (5-10s due to Python cold start + auth refresh)
- 🌐 Cloud Console is instant, has syntax highlighting, and saves query history
- 📊 Better visualization (charts, schema explorer, query validator)

### Expected Schedule

| Source    | Scheduler Time | ETL Trigger Time   | Data Available    |
|-----------|---------------|--------------------|-------------------|
| JobStreet | 9:00 PM SGT   | ~9:05-9:10 PM SGT  | ~9:06-9:11 PM SGT |
| MCF       | 9:00 AM SGT   | ~9:05-9:10 AM SGT  | ~9:06-9:11 AM SGT |

### Cloud Monitoring

**Key Metrics:**
- `cloudfunctions.googleapis.com/function/execution_count` - Invocations per day
- `cloudfunctions.googleapis.com/function/execution_times` - Duration (should be <120s)
- `cloudfunctions.googleapis.com/function/error_count` - Failures (should be 0)
- `cloudfunctions.googleapis.com/function/user_memory_bytes` - Memory usage (<512MB)

**Alerts to Set Up:**
1. Error count > 2 in 10 minutes → Email admin
2. No execution in 25 hours → Email admin (missing scrape)
3. Memory usage > 450MB → Warning (approaching limit)

---

## Troubleshooting

### ETL Not Triggering

**Check:**
```bash
# 1. Function deployed?
gcloud functions describe etl-gcs-to-bigquery --region=asia-southeast1 --gen2

# 2. File path correct?
# Must be: raw/{source}/{timestamp}/dump.jsonl*

# 3. Service account permissions?
gcloud projects get-iam-policy sg-job-market \
  --flatten="bindings[].members" \
  --filter="bindings.members:GCP-general-sa@"
```

### Permission Denied

**Fix:**
```bash
# Grant missing roles
gcloud projects add-iam-policy-binding sg-job-market \
  --member="serviceAccount:GCP-general-sa@sg-job-market.iam.gserviceaccount.com" \
  --role="roles/bigquery.dataEditor"
```

### Timeout (600s limit)

**Solutions:**
```bash
# 1. Increase timeout (max 3600s)
gcloud functions deploy etl-gcs-to-bigquery --gen2 --runtime=python313 --region=asia-southeast1 --source=.deploy_temp --entry-point=etl_gcs_to_bigquery --trigger-event-filters="type=google.cloud.storage.object.v1.finalized" --trigger-event-filters="bucket=sg-job-market-data" --memory=512MB --timeout=3600s --service-account=GCP-general-sa@sg-job-market.iam.gserviceaccount.com --set-env-vars="GCP_PROJECT_ID=sg-job-market,BIGQUERY_DATASET_ID=sg_job_market,GCP_REGION=asia-southeast1,GCS_BUCKET=sg-job-market-data"

# 2. Reduce batch size in cloud_function_main.py
BATCH_SIZE = 250  # Default: 500
```

### Memory Limit Exceeded

**Solutions:**
```bash
# Increase memory to 1GB
gcloud functions deploy etl-gcs-to-bigquery --gen2 --runtime=python313 --region=asia-southeast1 --source=.deploy_temp --entry-point=etl_gcs_to_bigquery --trigger-event-filters="type=google.cloud.storage.object.v1.finalized" --trigger-event-filters="bucket=sg-job-market-data" --memory=1024MB --timeout=600s --service-account=GCP-general-sa@sg-job-market.iam.gserviceaccount.com --set-env-vars="GCP_PROJECT_ID=sg-job-market,BIGQUERY_DATASET_ID=sg_job_market,GCP_REGION=asia-southeast1,GCS_BUCKET=sg-job-market-data"
```

### Duplicate Rows

**Fix with query-time dedup:**
```sql
-- Always use this pattern for downstream queries
SELECT * FROM (
  SELECT *, 
    ROW_NUMBER() OVER (
      PARTITION BY source, job_id, scrape_timestamp
      ORDER BY bq_timestamp DESC
    ) AS rn
  FROM raw_jobs
) WHERE rn = 1;
```

---

## FAQ

### Q: How does file filtering work?

**A:** Two layers:
1. GCP trigger: Bucket-level only (`bucket=sg-job-market-data`)
2. Code validation: Path + filename check (in `etl/main.py`)

Code filters for:
- Path: `raw/{source}/{timestamp}/`
- Filename: `dump.jsonl` or `dump.jsonl.gz`
- Source: `jobstreet` or `mcf`

### Q: What if scraper runs multiple times per day?

**A:** Each run creates unique directory with timestamp. ETL processes each separately. Use query-time deduplication to get latest version of each job.

### Q: What if ETL fails?

**A:** GCP auto-retries up to 3 times. File remains in GCS (not deleted). Can manually reupload to trigger again. Deduplication handles multiple inserts.

### Q: How to manually trigger ETL?

**Option 1:** Reupload file to GCS (triggers event)
```bash
gsutil cp dump.jsonl.gz gs://sg-job-market-data/raw/jobstreet/2025-12-19_220000/
```

**Option 2:** Run locally
```python
from etl.main import etl_gcs_to_bigquery

event = {
    "bucket": "sg-job-market-data",
    "name": "raw/jobstreet/2025-12-19_210000/dump.jsonl.gz",
}
etl_gcs_to_bigquery(event, None)
```

### Q: What's the cost?

**A:** ~$0/month (within GCP free tier)
- Cloud Functions: 2M free invocations/month (we use ~60)
- BigQuery: 1 TB free queries/month (we use ~10 GB)
- GCS: 5 GB free storage/month (we use ~2 GB)

### Q: How to add new data source (e.g., LinkedIn)?

**A:** No code changes needed:
1. Scraper uploads to `raw/linkedin/{timestamp}/dump.jsonl.gz`
2. ETL auto-triggers and processes
3. Verify: `SELECT * FROM raw_jobs WHERE source = 'linkedin'`

### Q: How to redeploy without downtime?

**A:** Cloud Functions Gen 2 supports zero-downtime. Just redeploy:
```bash
# Create temp directory and copy files
mkdir .deploy_temp
cp etl/main.py .deploy_temp/main.py
cp -r etl/ .deploy_temp/etl/
cp -r utils/ .deploy_temp/utils/
cp requirements.txt .deploy_temp/requirements.txt

# Redeploy
gcloud functions deploy etl-gcs-to-bigquery --gen2 --runtime=python313 --region=asia-southeast1 --source=.deploy_temp --entry-point=etl_gcs_to_bigquery --trigger-event-filters="type=google.cloud.storage.object.v1.finalized" --trigger-event-filters="bucket=sg-job-market-data" --memory=512MB --timeout=600s --service-account=GCP-general-sa@sg-job-market.iam.gserviceaccount.com --set-env-vars="GCP_PROJECT_ID=sg-job-market,BIGQUERY_DATASET_ID=sg_job_market,GCP_REGION=asia-southeast1,GCS_BUCKET=sg-job-market-data"

# Cleanup
rm -rf .deploy_temp

# Old version handles in-flight requests
# New version receives new requests
# No events lost ✅
```

### Q: How to backfill historical data?

**Option 1:** Reupload old files with new timestamps
```bash
gsutil cp gs://.../old/dump.jsonl.gz gs://.../new_timestamp/dump.jsonl.gz
```

**Option 2:** Process locally then upload to BigQuery
```python
from etl.cloud_function_main import test_stage1_local
test_stage1_local(Path("data/raw/jobstreet/.../dump.jsonl"), "jobstreet", datetime.now())
```

---

## Next Steps

### Stage 2: Data Transformation (TODO)

Implement transformations in `etl/transform.py`:
1. Text cleaning (HTML removal, normalization)
2. Salary parsing (extract min/max, convert to monthly SGD)
3. Language detection (job descriptions)
4. Company name normalization
5. Location standardization

### Monitoring Enhancements

1. Create Looker Studio dashboard (scrape volumes, execution times)
2. Set up log-based metrics (rows processed, parse success rate)
3. Configure alerting policies (failures, missing scrapes)

### Cost Optimization

1. Compression: ✅ Already using gzip
2. Partitioning: ✅ Already partitioned by scrape_timestamp
3. Clustering: ✅ Already clustered by source, job_id
4. Lifecycle: Archive old data after 90 days

---

## References

- [Cloud Functions Docs](https://cloud.google.com/functions/docs)
- [BigQuery Streaming API](https://cloud.google.com/bigquery/docs/streaming-data-into-bigquery)
- [GCS Event Triggers](https://cloud.google.com/functions/docs/calling/storage)

---

**Last Updated:** December 19, 2025  
**Status:** Production-ready
