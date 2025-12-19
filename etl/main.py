"""Cloud Function entry point for GCS-triggered ETL pipeline.

This file is required by Cloud Functions deployment.
The function name MUST match the --entry-point parameter.
"""

import logging
from datetime import datetime
from pathlib import Path

from etl.cloud_function_main import stage1_load_raw
from utils.config import Settings
from utils.logging import configure_logging

logger = logging.getLogger(__name__)


def etl_gcs_to_bigquery(event: dict, context) -> str:
    """Entry point called by Cloud Functions when JSONL file is uploaded to GCS.
    
    Triggered by: GCS object finalize event (google.cloud.storage.object.v1.finalized)
    Trigger filters:
      - bucket: sg-job-market-data
      - name pattern: raw/**/*.jsonl*
    
    Event structure:
      {
        "bucket": "sg-job-market-data",
        "name": "raw/jobstreet/2025-12-18_210000/dump.jsonl.gz",
        "size": "1234567",
        "timeCreated": "2025-12-18T21:00:45.123Z",
        "contentType": "application/gzip",
        ...
      }
    
    Args:
        event (dict): GCS event data
        context: Cloud Functions context (event_id, timestamp, resource)
    
    Returns:
        str: Success message with row counts
    """
    configure_logging(service_name="etl-gcs-to-bigquery")
    
    # Extract event details
    bucket = event.get("bucket")
    file_path = event.get("name")  # e.g., "raw/jobstreet/2025-12-18_210000/dump.jsonl.gz"
    size_bytes = int(event.get("size", 0))
    time_created = event.get("timeCreated")
    
    logger.info(
        f"[ETL] Triggered by GCS event: "
        f"gs://{bucket}/{file_path} ({size_bytes / 1024 / 1024:.2f} MB)"
    )
    
    # Validate file path pattern: raw/{source}/{timestamp}/dump.jsonl[.gz]
    path_parts = Path(file_path).parts
    if len(path_parts) < 3 or path_parts[0] != "raw":
        logger.error(f"[ETL] Invalid file path pattern: {file_path}")
        return f"ERROR: Invalid file path pattern: {file_path}"
    
    source = path_parts[1]  # 'jobstreet' or 'mcf'
    timestamp_dir = path_parts[2]  # e.g., '2025-12-18_210000'
    filename = Path(file_path).name  # e.g., 'dump.jsonl.gz'
    
    # Only process dump.jsonl or dump.jsonl.gz files
    if not filename.startswith("dump.jsonl"):
        logger.info(f"[ETL] Skipping non-dump file: {file_path}")
        return f"SKIP: Not a dump file: {filename}"
    
    if source not in ["jobstreet", "mcf"]:
        logger.error(f"[ETL] Unsupported source: {source}")
        return f"ERROR: Unsupported source: {source}"
    
    # Parse scrape timestamp from directory name
    try:
        scrape_timestamp = datetime.strptime(timestamp_dir, "%Y-%m-%d_%H%M%S")
    except ValueError:
        logger.warning(f"[ETL] Could not parse timestamp from {timestamp_dir}, using file creation time")
        scrape_timestamp = datetime.fromisoformat(time_created.replace('Z', '+00:00'))
    
    # Build GCS URI
    gcs_uri = f"gs://{bucket}/{file_path}"
    
    # Load settings
    settings = Settings.load()
    
    # Execute Stage 1: JSONL → raw_jobs
    try:
        result = stage1_load_raw(
            gcs_uri=gcs_uri,
            source=source,
            scrape_timestamp=scrape_timestamp,
            settings=settings,
            local_file_path=None,  # Download from GCS
        )
        
        success_rate = (
            result["streamed_rows"] / result["valid_rows"] * 100 
            if result["valid_rows"] > 0 else 0
        )
        
        logger.info(
            f"[ETL] Stage 1 complete: "
            f"{result['streamed_rows']}/{result['valid_rows']} rows "
            f"({success_rate:.1f}% success) in {result['duration_seconds']:.1f}s"
        )
        
        if result["failed_rows"] > 0:
            logger.warning(f"[ETL] {result['failed_rows']} rows failed to stream")
        
        return (
            f"✓ ETL complete: {result['streamed_rows']} rows loaded to raw_jobs "
            f"(source={source}, duration={result['duration_seconds']:.1f}s)"
        )
    
    except Exception as e:
        logger.error(f"[ETL] Stage 1 failed: {e}", exc_info=True)
        return f"ERROR: Stage 1 failed: {e}"
