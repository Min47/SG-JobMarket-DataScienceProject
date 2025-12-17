"""
Comprehensive Two-Stage ETL Pipeline Test

This test simulates the complete data pipeline in GCP:
1. Stage 1 (Cloud Function): JSONL from GCS → BigQuery raw_jobs
2. Stage 2 (Cloud Function): BigQuery raw_jobs → ETL → BigQuery cleaned_jobs
3. Query: Deduplication using ROW_NUMBER() pattern

Architecture in GCP:
- Stage 1: Cloud Function triggered by GCS finalize event
  - Reads JSONL file uploaded by scraper
  - Calls load_jsonl_to_bq() to ingest into raw_jobs
  - Owner: ETL Agent (implements Cloud Function), Cloud Agent (provides API)
  
- Stage 2: Cloud Function triggered by raw_jobs insert
  - Reads from raw_jobs, performs data cleaning
  - Streams to cleaned_jobs using stream_rows_to_bq()
  - Owner: ETL Agent (full responsibility)

This test validates the Cloud Agent's APIs are ready for ETL Agent integration.
"""

import sys
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List
from google.cloud import bigquery

from utils.config import Settings
from utils.logging import configure_logging
from utils.bq import bq_client, load_jsonl_to_bq, stream_rows_to_bq

# Setup
logger = configure_logging(service_name="test_two_stage_pipeline")
settings = Settings.load()
client = bq_client(settings)


def find_latest_jsonl(source: str) -> Path:
    """Find the most recent JSONL dump for a given source."""
    raw_dir = Path("data/raw") / source
    
    if not raw_dir.exists():
        raise FileNotFoundError(f"No data directory for {source}")
    
    # Find all timestamp directories
    timestamp_dirs = [d for d in raw_dir.iterdir() if d.is_dir() and d.name != "smoke_test"]
    
    if not timestamp_dirs:
        raise FileNotFoundError(f"No data dumps found in {raw_dir}")
    
    # Sort by directory name (timestamp format: YYYY-MM-DD_HHMMSS)
    latest_dir = sorted(timestamp_dirs, reverse=True)[0]
    jsonl_file = latest_dir / "dump.jsonl"
    
    if not jsonl_file.exists():
        raise FileNotFoundError(f"No dump.jsonl found in {latest_dir}")
    
    return jsonl_file


def minimal_etl_transform(raw_row: Dict[str, Any]) -> Dict[str, Any]:
    """
    Minimal ETL transformation for testing purposes.
    
    NOTE: This is a placeholder for ETL Agent's real transformation logic.
    Real ETL will include:
    - Salary parsing and conversion
    - Company name normalization
    - Location extraction and standardization
    - Description HTML cleaning
    - Classification mapping
    - Duplicate detection
    
    For now, we just populate required fields with placeholder values.
    """
    from datetime import timezone
    
    # Parse payload JSON string
    if isinstance(raw_row['payload'], str):
        payload = json.loads(raw_row['payload'])
    else:
        payload = raw_row['payload']
    
    # Create cleaned_jobs row matching CleanedJob schema
    # This is a PLACEHOLDER - ETL Agent will implement full transformation
    cleaned_row = {
        # === Metadata fields ===
        "source": raw_row["source"],
        "scrape_timestamp": raw_row["scrape_timestamp"],
        "bq_timestamp": datetime.now(timezone.utc),  # Timezone-aware timestamp
        
        # === Job fields (ETL Agent will parse from payload) ===
        "job_id": raw_row["job_id"],
        "job_url": payload.get("url", payload.get("jobUrl", "https://example.com")),
        "job_title": payload.get("title", payload.get("jobTitle", "Unknown Title")),
        "job_description": payload.get("description", "No description"),  # ETL Agent will clean HTML
        "job_location": payload.get("location", "Singapore"),  # ETL Agent will standardize
        "job_classification": payload.get("category", "General"),  # ETL Agent will map
        "job_work_type": payload.get("employmentType", "Full-Time"),
        "job_salary_min_sgd_raw": None,  # ETL Agent will parse from salary string
        "job_salary_max_sgd_raw": None,  # ETL Agent will parse from salary string
        "job_salary_type": "Monthly",  # ETL Agent will parse
        "job_salary_min_sgd_monthly": None,  # ETL Agent will convert
        "job_salary_max_sgd_monthly": None,  # ETL Agent will convert
        "job_currency": "SGD",
        "job_posted_timestamp": datetime.now(timezone.utc),  # ETL Agent will parse from payload
        
        # === Employer fields (ETL Agent will parse from payload) ===
        "company_id": "unknown",  # ETL Agent will extract
        "company_url": "https://example.com",  # ETL Agent will extract
        "company_name": payload.get("company", payload.get("employerName", "Unknown Company")),
        "company_description": "No description",  # ETL Agent will extract
        "company_industry": "General",  # ETL Agent will extract
        "company_size": "Unknown",  # ETL Agent will extract/normalize
    }
    
    return cleaned_row


def main():
    logger.info("=" * 80)
    logger.info("TWO-STAGE ETL PIPELINE TEST")
    logger.info("=" * 80)
    logger.info(f"✓ Connected to project: {client.project}")
    logger.info(f"✓ Dataset: {settings.bigquery_dataset_id}")
    logger.info(f"✓ Region: {settings.gcp_region}")
    logger.info("")
    logger.info("This test simulates what the Cloud Functions will do:")
    logger.info("  Stage 1 (ETL Agent): GCS JSONL → BigQuery raw_jobs")
    logger.info("  Stage 2 (ETL Agent): BigQuery raw_jobs → ETL → BigQuery cleaned_jobs")
    
    # ========================================================================
    # STAGE 1: JSONL → raw_jobs (Simulates Cloud Function Stage 1)
    # ========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("STAGE 1: JSONL → raw_jobs (Cloud Function Simulation)")
    logger.info("=" * 80)
    logger.info("In GCP: Cloud Function triggered by GCS finalize event")
    logger.info("  - Reads JSONL uploaded by scraper to GCS")
    logger.info("  - Calls load_jsonl_to_bq() to ingest into raw_jobs")
    logger.info("  - Owner: ETL Agent (Cloud Function) + Cloud Agent (BigQuery API)")
    logger.info("")
    
    # Find latest JSONL files
    try:
        jobstreet_jsonl = find_latest_jsonl("jobstreet")
        logger.info(f"✓ Found JobStreet JSONL: {jobstreet_jsonl}")
    except FileNotFoundError as e:
        logger.warning(f"⚠ JobStreet data not found: {e}")
        jobstreet_jsonl = None
    
    try:
        mcf_jsonl = find_latest_jsonl("mcf")
        logger.info(f"✓ Found MCF JSONL: {mcf_jsonl}")
    except FileNotFoundError as e:
        logger.warning(f"⚠ MCF data not found: {e}")
        mcf_jsonl = None
    
    if not jobstreet_jsonl and not mcf_jsonl:
        logger.error("✗ No JSONL files found. Run scrapers first.")
        return 1
    
    # Load JSONL files into raw_jobs
    stage1_results = []
    
    if jobstreet_jsonl:
        logger.info(f"\n--- Loading JobStreet JSONL → raw_jobs ---")
        result = load_jsonl_to_bq(
            client,
            str(jobstreet_jsonl),
            settings.bigquery_dataset_id,
            "raw_jobs",
        )
        stage1_results.append(("JobStreet", result))
        logger.info(f"✓ JobStreet: {result['successful_rows']}/{result['total_rows']} rows inserted")
    
    if mcf_jsonl:
        logger.info(f"\n--- Loading MCF JSONL → raw_jobs ---")
        result = load_jsonl_to_bq(
            client,
            str(mcf_jsonl),
            settings.bigquery_dataset_id,
            "raw_jobs",
        )
        stage1_results.append(("MCF", result))
        logger.info(f"✓ MCF: {result['successful_rows']}/{result['total_rows']} rows inserted")
    
    # Summary
    total_stage1 = sum(r['successful_rows'] for _, r in stage1_results)
    logger.info(f"\n✓ STAGE 1 COMPLETE: {total_stage1} rows in raw_jobs")
    
    # ========================================================================
    # STAGE 2: raw_jobs → cleaned_jobs (Simulates Cloud Function Stage 2)
    # ========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("STAGE 2: raw_jobs → ETL → cleaned_jobs (Cloud Function Simulation)")
    logger.info("=" * 80)
    logger.info("In GCP: Cloud Function triggered by raw_jobs insert")
    logger.info("  - Queries raw_jobs for new records")
    logger.info("  - Performs data cleaning and transformation")
    logger.info("  - Streams to cleaned_jobs using stream_rows_to_bq()")
    logger.info("  - Owner: ETL Agent (full responsibility)")
    logger.info("")
    logger.info("NOTE: Using PLACEHOLDER transformation logic")
    logger.info("      ETL Agent will implement full salary parsing, deduplication, etc.")
    logger.info("")
    
    # Query raw_jobs to get records for transformation
    # In production, Cloud Function would query only new records
    # For testing, we'll get a sample from the most recent scrape
    query = f"""
    SELECT 
        job_id,
        source,
        scrape_timestamp,
        payload
    FROM `{client.project}.{settings.bigquery_dataset_id}.raw_jobs`
    WHERE scrape_timestamp >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 1 HOUR)
    ORDER BY scrape_timestamp DESC
    LIMIT 50
    """
    
    logger.info("--- Querying raw_jobs for sample records ---")
    query_job = client.query(query)
    raw_rows = list(query_job.result())
    logger.info(f"✓ Retrieved {len(raw_rows)} rows from raw_jobs")
    
    if not raw_rows:
        logger.warning("⚠ No rows found in raw_jobs. Stage 1 may have failed.")
        return 1
    
    # Transform rows (PLACEHOLDER for ETL Agent's logic)
    logger.info("\n--- Applying ETL transformation (PLACEHOLDER) ---")
    cleaned_rows = []
    for raw_row in raw_rows:
        try:
            cleaned_row = minimal_etl_transform(dict(raw_row))
            cleaned_rows.append(cleaned_row)
        except Exception as e:
            logger.warning(f"⚠ Failed to transform row {raw_row['job_id']}: {e}")
    
    logger.info(f"✓ Transformed {len(cleaned_rows)} rows")
    
    # Stream to cleaned_jobs
    logger.info("\n--- Streaming transformed data → cleaned_jobs ---")
    result = stream_rows_to_bq(
        client,
        settings.bigquery_dataset_id,
        "cleaned_jobs",
        cleaned_rows,
    )
    
    logger.info(f"✓ Streamed {result['successful_rows']}/{result['total_rows']} rows to cleaned_jobs")
    
    # ========================================================================
    # STAGE 3: Test Deduplication Query
    # ========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("STAGE 3: Test Deduplication Query Pattern")
    logger.info("=" * 80)
    logger.info("Testing ROW_NUMBER() pattern to get latest version of each job")
    logger.info("")
    
    dedup_query = f"""
    SELECT 
        source,
        job_id,
        job_title,
        company_name,
        scrape_timestamp,
        bq_timestamp
    FROM (
        SELECT *, 
            ROW_NUMBER() OVER (
                PARTITION BY source, job_id 
                ORDER BY scrape_timestamp DESC
            ) AS rn
        FROM `{client.project}.{settings.bigquery_dataset_id}.cleaned_jobs`
    )
    WHERE rn = 1
    ORDER BY source, scrape_timestamp DESC
    LIMIT 10
    """
    
    logger.info("--- Executing deduplication query ---")
    query_job = client.query(dedup_query)
    dedup_rows = list(query_job.result())
    
    logger.info(f"✓ Deduplication query returned {len(dedup_rows)} unique jobs")
    logger.info("\nSample deduplicated records:")
    for row in dedup_rows[:5]:
        logger.info(f"  - {row['source']:10s} | {row['job_id']:15s} | {row['job_title'][:50]:50s} | {row['scrape_timestamp']}")
    
    # ========================================================================
    # FINAL SUMMARY
    # ========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("TEST SUMMARY")
    logger.info("=" * 80)
    logger.info(f"✓ Stage 1 (JSONL → raw_jobs): {total_stage1} rows ingested")
    logger.info(f"✓ Stage 2 (raw_jobs → cleaned_jobs): {result['successful_rows']} rows transformed")
    logger.info(f"✓ Stage 3 (Deduplication): {len(dedup_rows)} unique jobs retrieved")
    logger.info("")
    logger.info("Architecture Responsibilities:")
    logger.info("  • Cloud Agent: Provides BigQuery APIs (load_jsonl_to_bq, stream_rows_to_bq)")
    logger.info("  • ETL Agent: Implements Cloud Functions for both stages")
    logger.info("")
    logger.info("Next Steps for ETL Agent:")
    logger.info("  1. Implement Cloud Function Stage 1 (GCS trigger → load_jsonl_to_bq)")
    logger.info("  2. Implement Cloud Function Stage 2 (raw_jobs → full ETL → cleaned_jobs)")
    logger.info("  3. Deploy Cloud Functions with proper triggers and IAM")
    logger.info("")
    logger.info("✓ TWO-STAGE PIPELINE TEST COMPLETE")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
