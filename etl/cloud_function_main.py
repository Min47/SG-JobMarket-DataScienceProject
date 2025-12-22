"""Cloud Function entry point for GCS-triggered ETL pipeline.

This module implements a two-stage ETL pipeline:
- Stage 1: JSONL from GCS → BigQuery raw_jobs (raw ingestion)
- Stage 2: raw_jobs → cleaned_jobs (transformation & cleaning)

Triggered automatically by GCS object finalize events when scrapers upload JSONL files.
"""

from __future__ import annotations

import gzip
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from google.cloud import bigquery

from utils.bq import bq_client, ensure_dataset, ensure_table, stream_rows_to_bq
from utils.bq_schemas import raw_jobs_schema, cleaned_jobs_schema
from utils.config import Settings
from utils.gcs import GCSClient, parse_gcs_uri
from utils.logging import configure_logging
from utils.schemas import RawJob, CleanedJob
from etl.transform import transform_raw_to_cleaned

logger = logging.getLogger(__name__)


# =============================================================================
# Stage 1: JSONL → raw_jobs (Raw Ingestion)
# =============================================================================

def stage1_load_raw(
    gcs_uri: str,
    source: str,
    scrape_timestamp: datetime,
    settings: Settings,
    local_file_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Stage 1: Load JSONL from GCS to BigQuery raw_jobs table.
    
    This function:
    1. Downloads JSONL file from GCS (if not already local)
    2. Parses JSONL line-by-line (memory efficient)
    3. Validates against RawJob schema
    4. Streams to BigQuery raw_jobs table
    
    Args:
        gcs_uri: GCS URI of the JSONL file (gs://bucket/path/to/dump.jsonl.gz)
        source: Source name ('jobstreet' or 'mcf')
        scrape_timestamp: Timestamp when job was scraped
        settings: Configuration settings
        local_file_path: Optional local file path (for testing without GCS download)
        
    Returns:
        Dict with stage 1 results:
            - downloaded_path: Path to downloaded file
            - total_lines: Total lines in JSONL
            - valid_rows: Number of valid rows
            - invalid_rows: Number of invalid/malformed rows
            - streamed_rows: Number of rows successfully streamed to BigQuery
            - duration_seconds: Stage duration
            
    Example:
        >>> settings = Settings.load()
        >>> result = stage1_load_raw(
        ...     gcs_uri="gs://sg-job-market-data/raw/jobstreet/2025-12-18_210000/dump.jsonl.gz",
        ...     source="jobstreet",
        ...     scrape_timestamp=datetime(2025, 12, 18, 21, 0, 0),
        ...     settings=settings
        ... )
        >>> print(f"Loaded {result['streamed_rows']} rows to raw_jobs")
    """
    stage_start = datetime.now(timezone.utc)
    
    logger.info(f"[Stage 1] Starting raw ingestion: source={source}, gcs_uri={gcs_uri}")
    
    # Step 1: Download from GCS (or use local file for testing)
    if local_file_path:
        downloaded_path = local_file_path
        logger.info(f"[Stage 1] Using local file: {downloaded_path}")
    else:
        # Determine file extension based on source file (preserve compression state)
        source_extension = ".jsonl.gz" if gcs_uri.endswith(".gz") else ".jsonl"
        tmp_path = Path("/tmp") / f"etl_{source}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}{source_extension}"
        logger.info(f"[Stage 1] Downloading from GCS to {tmp_path}")
        
        gcs_client = GCSClient(project_id=settings.gcp_project_id)
        downloaded_path = gcs_client.download_file(gcs_uri, tmp_path)
        logger.info(f"[Stage 1] Downloaded: {downloaded_path.stat().st_size / 1024 / 1024:.2f} MB")
    
    # Step 2: Parse JSONL → RawJob objects
    logger.info("[Stage 1] Parsing JSONL file...")
    
    raw_rows: List[Dict[str, Any]] = []
    total_lines = 0
    invalid_rows = 0
    
    # Handle both gzipped and plain JSONL
    if str(downloaded_path).endswith('.gz'):
        file_opener = gzip.open
    else:
        file_opener = open
    
    try:
        with file_opener(downloaded_path, 'rt', encoding='utf-8') as f:
            for line_num, line in enumerate(f, start=1):
                total_lines += 1
                line = line.strip()
                
                if not line:
                    continue
                
                try:
                    data = json.loads(line)
                    
                    # Scraper already outputs in RawJob format (job_id, source, scrape_timestamp, payload)
                    # Extract job_id from root level, not nested in payload
                    job_id = data.get("job_id")
                    if not job_id:
                        logger.warning(f"[Stage 1] Line {line_num}: Missing job_id, skipping")
                        invalid_rows += 1
                        continue
                    
                    # Extract payload (should already be a dict)
                    payload = data.get("payload")
                    if not isinstance(payload, dict):
                        logger.warning(f"[Stage 1] Line {line_num}: Invalid payload format, skipping")
                        invalid_rows += 1
                        continue
                    
                    # Create RawJob record (source and scrape_timestamp also at root)
                    # Note: BigQuery JSON fields must be serialized as strings for streaming insert
                    raw_row = {
                        "job_id": str(job_id),
                        "source": data.get("source", source),  # Use from data, fallback to argument
                        "scrape_timestamp": data.get("scrape_timestamp", scrape_timestamp.isoformat()),
                        "payload": json.dumps(payload),  # Serialize dict to JSON string for BigQuery
                    }
                    
                    raw_rows.append(raw_row)
                    
                except json.JSONDecodeError as e:
                    logger.warning(f"[Stage 1] Line {line_num}: JSON decode error - {e}")
                    invalid_rows += 1
                    continue
                except Exception as e:
                    logger.warning(f"[Stage 1] Line {line_num}: Unexpected error - {e}")
                    invalid_rows += 1
                    continue
        
        logger.info(
            f"[Stage 1] Parsed {total_lines} lines: "
            f"valid={len(raw_rows)}, invalid={invalid_rows}"
        )
    
    except Exception as e:
        logger.error(f"[Stage 1] Failed to parse JSONL: {e}")
        raise
    
    # Step 3: Ensure BigQuery table exists
    logger.info("[Stage 1] Ensuring BigQuery table exists...")
    client = bq_client(settings)
    
    ensure_dataset(
        client,
        settings.bigquery_dataset_id,
        location=settings.gcp_region,
        description="Singapore Job Market data warehouse"
    )
    
    ensure_table(
        client,
        settings.bigquery_dataset_id,
        "raw_jobs",
        raw_jobs_schema(),
        partition_field="scrape_timestamp",
        clustering_fields=["source", "job_id"],
        description="Raw scraped job data with TIMESTAMP partitioning"
    )
    
    # Step 4: Stream to BigQuery
    if raw_rows:
        logger.info(f"[Stage 1] Streaming {len(raw_rows)} rows to BigQuery raw_jobs...")
        
        stream_result = stream_rows_to_bq(
            client,
            settings.bigquery_dataset_id,
            "raw_jobs",
            raw_rows,
            batch_size=500
        )
        
        logger.info(
            f"[Stage 1] Streaming complete: "
            f"{stream_result['successful_rows']}/{stream_result['total_rows']} rows "
            f"({stream_result['successful_rows'] / stream_result['total_rows'] * 100:.1f}% success)"
        )
    else:
        logger.warning("[Stage 1] No valid rows to stream")
        stream_result = {
            "total_rows": 0,
            "successful_rows": 0,
            "failed_rows": 0,
            "error_details": []
        }
    
    # Cleanup temp file if downloaded from GCS
    if not local_file_path and downloaded_path.exists():
        try:
            downloaded_path.unlink()
            logger.info(f"[Stage 1] Cleaned up temp file: {downloaded_path}")
        except Exception as e:
            logger.warning(f"[Stage 1] Failed to cleanup temp file: {e}")
    
    stage_duration = (datetime.now(timezone.utc) - stage_start).total_seconds()
    
    result = {
        "downloaded_path": str(downloaded_path),
        "total_lines": total_lines,
        "valid_rows": len(raw_rows),
        "invalid_rows": invalid_rows,
        "streamed_rows": stream_result["successful_rows"],
        "failed_rows": stream_result["failed_rows"],
        "duration_seconds": stage_duration,
    }
    
    logger.info(
        f"[Stage 1] Complete: {result['streamed_rows']}/{result['valid_rows']} rows loaded "
        f"in {stage_duration:.1f}s"
    )
    
    return result


# =============================================================================
# Stage 2: raw_jobs → cleaned_jobs (Transformation & Cleaning)
# =============================================================================

def stage2_transform_to_cleaned(
    source: str,
    scrape_timestamp: datetime,
    settings: Settings,
    batch_size: int = 20,
) -> Dict[str, Any]:
    """Stage 2: Transform raw_jobs to cleaned_jobs.
    
    This function:
    1. Queries raw_jobs for the specific scrape run
    2. Transforms each RawJob → CleanedJob (text cleaning, salary parsing, etc.)
    3. Streams CleanedJob records to BigQuery cleaned_jobs table
    4. Processes in batches to manage memory
    
    Args:
        source: Source name ('jobstreet' or 'mcf')
        scrape_timestamp: Timestamp of the scrape run to process
        settings: Configuration settings
        batch_size: Number of rows to process per batch (default: 20)
        
    Returns:
        Result dictionary with statistics
        
    Example:
        >>> result = stage2_transform_to_cleaned(
        ...     source='jobstreet',
        ...     scrape_timestamp=datetime(2025, 12, 22, 21, 0, 0),
        ...     settings=Settings.load()
        ... )
        >>> print(f"Transformed {result['cleaned_rows']} rows")
    """
    start_time = datetime.now(timezone.utc)
    logger.info("")
    logger.info(f"[Stage 2] Starting transformation: source={source}, timestamp={scrape_timestamp}")
    
    # Ensure cleaned_jobs table exists
    dataset_id = settings.bigquery_dataset_id
    table_id = "cleaned_jobs"
    
    logger.info(f"[Stage 2] Ensuring table: {dataset_id}.{table_id}")
    ensure_dataset(dataset_id, settings)
    ensure_table(
        dataset_id=dataset_id,
        table_id=table_id,
        schema=cleaned_jobs_schema(),
        partition_field="scrape_timestamp",
        clustering_fields=["source", "job_id", "company_name"],
        settings=settings,
    )
    
    # Query raw_jobs for this specific scrape run
    query = f"""
    SELECT 
        job_id,
        source,
        scrape_timestamp,
        payload
    FROM `{settings.gcp_project_id}.{dataset_id}.raw_jobs`
    WHERE source = @source
      AND scrape_timestamp = @scrape_timestamp
    ORDER BY job_id
    """
    
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("source", "STRING", source),
            bigquery.ScalarQueryParameter("scrape_timestamp", "TIMESTAMP", scrape_timestamp),
        ]
    )
    
    logger.info(f"[Stage 2] Querying raw_jobs...")
    query_job = bq_client(settings).query(query, job_config=job_config)
    raw_rows = list(query_job.result())  # Fetch all (should be manageable for daily scrapes)
    
    total_raw_rows = len(raw_rows)
    logger.info(f"[Stage 2] Fetched {total_raw_rows} raw rows to transform")
    
    if total_raw_rows == 0:
        logger.warning(f"[Stage 2] No rows found for source={source}, timestamp={scrape_timestamp}")
        return {
            "source": source,
            "scrape_timestamp": scrape_timestamp,
            "raw_rows_fetched": 0,
            "transformed_rows": 0,
            "skipped_rows": 0,
            "cleaned_rows_streamed": 0,
            "duration_seconds": 0,
        }
    
    # Transform rows (batch processing)
    transformed_jobs = []
    skipped_count = 0
    
    for idx, row in enumerate(raw_rows, 1):
        # Build RawJob object
        raw_job = RawJob(
            job_id=row.job_id,
            source=row.source,
            scrape_timestamp=row.scrape_timestamp,
            payload=dict(row.payload) if row.payload else {},
        )
        
        # Transform
        cleaned_job = transform_raw_to_cleaned(raw_job)
        
        if cleaned_job:
            transformed_jobs.append(cleaned_job)
        else:
            skipped_count += 1
        
        # Log progress every 20 rows
        if idx % 20 == 0:
            logger.info(f"[Stage 2] Transformed {idx}/{total_raw_rows} rows...")
    
    transformed_count = len(transformed_jobs)
    logger.info(
        f"[Stage 2] Transformation complete: "
        f"{transformed_count} transformed, {skipped_count} skipped"
    )
    
    # Stream to BigQuery in batches
    if transformed_count == 0:
        logger.warning("[Stage 2] No valid cleaned rows to stream")
        return {
            "source": source,
            "scrape_timestamp": scrape_timestamp,
            "raw_rows_fetched": total_raw_rows,
            "transformed_rows": 0,
            "skipped_rows": skipped_count,
            "cleaned_rows_streamed": 0,
            "duration_seconds": (datetime.now(timezone.utc) - start_time).total_seconds(),
        }
    
    # Convert CleanedJob objects to dicts for streaming
    cleaned_rows = []
    for cleaned_job in transformed_jobs:
        # Convert dataclass to dict
        row_dict = {
            "source": cleaned_job.source,
            "scrape_timestamp": cleaned_job.scrape_timestamp.isoformat(),
            "bq_timestamp": cleaned_job.bq_timestamp.isoformat(),
            
            "job_id": cleaned_job.job_id,
            "job_url": cleaned_job.job_url,
            "job_title": cleaned_job.job_title,
            "job_description": cleaned_job.job_description,
            "job_location": cleaned_job.job_location,
            "job_classification": cleaned_job.job_classification,
            "job_work_type": cleaned_job.job_work_type,
            
            "job_salary_min_sgd_raw": cleaned_job.job_salary_min_sgd_raw,
            "job_salary_max_sgd_raw": cleaned_job.job_salary_max_sgd_raw,
            "job_salary_type": cleaned_job.job_salary_type,
            "job_salary_min_sgd_monthly": cleaned_job.job_salary_min_sgd_monthly,
            "job_salary_max_sgd_monthly": cleaned_job.job_salary_max_sgd_monthly,
            "job_currency": cleaned_job.job_currency,
            "job_posted_timestamp": cleaned_job.job_posted_timestamp.isoformat(),
            
            "company_id": cleaned_job.company_id,
            "company_url": cleaned_job.company_url,
            "company_name": cleaned_job.company_name,
            "company_description": cleaned_job.company_description,
            "company_industry": cleaned_job.company_industry,
            "company_size": cleaned_job.company_size,
        }
        cleaned_rows.append(row_dict)
    
    logger.info(f"[Stage 2] Streaming {len(cleaned_rows)} rows to cleaned_jobs...")
    
    streamed_count = stream_rows_to_bq(
        dataset_id=dataset_id,
        table_id=table_id,
        rows=cleaned_rows,
        batch_size=batch_size,
        settings=settings,
    )
    
    stage_duration = (datetime.now(timezone.utc) - start_time).total_seconds()
    
    logger.info(
        f"[Stage 2] Complete: {streamed_count}/{transformed_count} rows loaded "
        f"in {stage_duration:.1f}s"
    )
    
    return {
        "source": source,
        "scrape_timestamp": scrape_timestamp,
        "raw_rows_fetched": total_raw_rows,
        "transformed_rows": transformed_count,
        "skipped_rows": skipped_count,
        "cleaned_rows_streamed": streamed_count,
        "duration_seconds": stage_duration,
    }


# =============================================================================
# Cloud Function Entry Point (Future)
# =============================================================================

def process_gcs_upload(event: Dict[str, Any], context: Any) -> str:
    """Cloud Function triggered by GCS object finalize.
    
    This will be the main entry point when deployed to Cloud Functions.
    Executes complete ETL pipeline:
    1. Download JSONL from GCS to /tmp/
    2. Parse JSONL into RawJob objects
    3. Stream to BigQuery raw_jobs table (Stage 1)
    4. Transform RawJob → CleanedJob objects
    5. Stream to BigQuery cleaned_jobs table (Stage 2)
    
    Args:
        event: GCS event data
            - name: File path (e.g., "raw/jobstreet/2025-12-18_210000/dump.jsonl.gz")
            - bucket: Bucket name ("sg-job-market-data")
            - timeCreated: ISO 8601 timestamp
        context: Event metadata (event_id, timestamp, resource)
        
    Returns:
        Success message with row counts
        
    Example event:
        {
            "name": "raw/jobstreet/2025-12-18_210000/dump.jsonl.gz",
            "bucket": "sg-job-market-data",
            "timeCreated": "2025-12-18T21:00:00Z"
        }
    """
    logger.info(f"[ETL] Cloud Function triggered: event={event}")
    
    # TODO: Implement full pipeline (Stage 1 + Stage 2)
    # For now, just log the event
    
    file_name = event.get("name")
    bucket = event.get("bucket")
    time_created = event.get("timeCreated")
    
    logger.info(f"[ETL] Processing: gs://{bucket}/{file_name}")
    logger.info(f"[ETL] Time created: {time_created}")
    
    return f"ETL processing started for {file_name}"


# =============================================================================
# Local Testing Helper
# =============================================================================

def test_stage1_local(
    local_jsonl_path: Path,
    source: str,
    scrape_timestamp: datetime,
) -> Dict[str, Any]:
    """Test Stage 1 with a local JSONL file (without GCS).
    
    This is a convenience function for local development and testing.
    
    Args:
        local_jsonl_path: Path to local JSONL file
        source: Source name ('jobstreet' or 'mcf')
        scrape_timestamp: Timestamp when job was scraped
        
    Returns:
        Stage 1 results dict
        
    Example:
        >>> from pathlib import Path
        >>> from datetime import datetime
        >>> result = test_stage1_local(
        ...     local_jsonl_path=Path("data/raw/jobstreet/2025-12-15_121208/dump.jsonl"),
        ...     source="jobstreet",
        ...     scrape_timestamp=datetime(2025, 12, 15, 12, 12, 8)
        ... )
        >>> print(f"Loaded {result['streamed_rows']} rows")
    """
    logger.info(f"[Test] Testing Stage 1 with local file: {local_jsonl_path}")
    
    if not local_jsonl_path.exists():
        raise FileNotFoundError(f"Local file not found: {local_jsonl_path}")
    
    settings = Settings.load()
    
    # Build fake GCS URI (not used for local testing)
    gcs_uri = f"gs://{settings.gcs_bucket or 'test-bucket'}/raw/{source}/test/dump.jsonl"
    
    result = stage1_load_raw(
        gcs_uri=gcs_uri,
        source=source,
        scrape_timestamp=scrape_timestamp,
        settings=settings,
        local_file_path=local_jsonl_path,
    )
    
    return result


# =============================================================================
# CLI Entry Point
# =============================================================================

def main():
    """Command-line interface for testing Stage 1 locally."""
    import sys
    from pathlib import Path
    
    configure_logging(service_name="etl_stage1")
    
    if len(sys.argv) < 3:
        print("Usage: python -m etl.cloud_function_main <source> <jsonl_path>")
        print("Example: python -m etl.cloud_function_main jobstreet data/raw/jobstreet/2025-12-15_121208/dump.jsonl")
        sys.exit(1)
    
    source = sys.argv[1]
    jsonl_path = Path(sys.argv[2])
    
    if source not in ["jobstreet", "mcf"]:
        print(f"Error: Invalid source '{source}'. Must be 'jobstreet' or 'mcf'")
        sys.exit(1)
    
    if not jsonl_path.exists():
        print(f"Error: File not found: {jsonl_path}")
        sys.exit(1)
    
    # Extract timestamp from directory name (e.g., "2025-12-15_121208")
    timestamp_str = jsonl_path.parent.name
    try:
        scrape_timestamp = datetime.strptime(timestamp_str, "%Y-%m-%d_%H%M%S")
    except ValueError:
        print(f"Warning: Could not parse timestamp from directory name: {timestamp_str}")
        scrape_timestamp = datetime.now(timezone.utc)
    
    logger.info(f"Testing Stage 1: source={source}, file={jsonl_path}")
    
    result = test_stage1_local(
        local_jsonl_path=jsonl_path,
        source=source,
        scrape_timestamp=scrape_timestamp,
    )
    
    print("\n" + "=" * 80)
    print("STAGE 1 RESULTS")
    print("=" * 80)
    print(f"Source: {source}")
    print(f"File: {jsonl_path}")
    print(f"Total lines: {result['total_lines']}")
    print(f"Valid rows: {result['valid_rows']}")
    print(f"Invalid rows: {result['invalid_rows']}")
    print(f"Streamed to BigQuery: {result['streamed_rows']}")
    print(f"Failed: {result['failed_rows']}")
    print(f"Duration: {result['duration_seconds']:.1f}s")
    print(f"Success rate: {result['streamed_rows'] / result['valid_rows'] * 100:.1f}%" if result['valid_rows'] > 0 else "N/A")
    print("=" * 80)
    
    if result['streamed_rows'] < result['valid_rows']:
        print(f"\n⚠️  Warning: {result['failed_rows']} rows failed to stream")
        sys.exit(1)
    else:
        print(f"\n✓ Success: All {result['streamed_rows']} rows loaded to raw_jobs")


if __name__ == "__main__":
    main()
