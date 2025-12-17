"""BigQuery helpers for dataset and table management.

This module provides:
- Dataset creation and management
- Table creation with partitioning and clustering
- Schema validation and retrieval
- Streaming data inserts
- JSONL file loading

All operations are idempotent and use retry logic for reliability.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, List, Optional, Dict, Any

from google.cloud import bigquery
from google.cloud.exceptions import NotFound, Conflict
from google.api_core import retry

from utils.config import Settings
from utils.retry import RetryPolicy, retry_call


logger = logging.getLogger(__name__)


# =============================================================================
# Client Management
# =============================================================================

def bq_client(settings: Settings) -> bigquery.Client:
    """Create a BigQuery client for the configured project."""
    return bigquery.Client(project=settings.gcp_project_id)


# =============================================================================
# Phase 1A: Core Infrastructure
# =============================================================================

def ensure_dataset(
    client: bigquery.Client,
    dataset_id: str,
    location: str = "asia-southeast1",
    description: Optional[str] = None,
) -> bigquery.Dataset:
    """Ensure dataset exists, create if missing (idempotent).
    
    Args:
        client: BigQuery client instance
        dataset_id: Dataset ID (not full path, just the ID)
        location: Dataset location (default: asia-southeast1 for Singapore)
        description: Optional dataset description
        
    Returns:
        Dataset reference
        
    Raises:
        Exception: If dataset creation fails after retries
    """
    dataset_ref = f"{client.project}.{dataset_id}"
    
    def _ensure():
        try:
            # Try to get existing dataset
            dataset = client.get_dataset(dataset_ref)
            logger.info(f"[BQ] Dataset already exists: {dataset_ref}")
            return dataset
            
        except NotFound:
            # Dataset doesn't exist, create it
            logger.info(f"[BQ] Creating dataset: {dataset_ref} in {location}")
            
            dataset = bigquery.Dataset(dataset_ref)
            dataset.location = location
            
            if description:
                dataset.description = description
            
            # Create dataset
            dataset = client.create_dataset(dataset, exists_ok=True)
            logger.info(f"[BQ] ✓ Dataset created successfully: {dataset_ref}")
            return dataset
    
    return retry_call(
        _ensure,
        policy=RetryPolicy(max_attempts=3, base_delay_seconds=1.0),
        on_retry=lambda attempt, exc: logger.warning(
            f"[BQ] Retry {attempt} for ensure_dataset({dataset_id}): {exc}"
        ),
    )


def ensure_table(
    client: bigquery.Client,
    dataset_id: str,
    table_id: str,
    schema: List[bigquery.SchemaField],
    partition_field: Optional[str] = None,
    clustering_fields: Optional[List[str]] = None,
    description: Optional[str] = None,
) -> bigquery.Table:
    """Ensure table exists with schema, create if missing (idempotent).
    
    Args:
        client: BigQuery client instance
        dataset_id: Dataset ID
        table_id: Table ID
        schema: List of BigQuery SchemaField objects
        partition_field: Field name for date partitioning (optional)
        clustering_fields: List of field names for clustering (optional)
        description: Optional table description
        
    Returns:
        Table reference
        
    Raises:
        Exception: If table creation fails after retries
    """
    table_ref = f"{client.project}.{dataset_id}.{table_id}"
    
    def _ensure():
        try:
            # Try to get existing table
            table = client.get_table(table_ref)
            logger.info(f"[BQ] Table already exists: {table_ref}")
            return table
            
        except NotFound:
            # Table doesn't exist, create it
            logger.info(f"[BQ] Creating table: {table_ref}")
            
            table = bigquery.Table(table_ref, schema=schema)
            
            # Add partitioning if specified
            if partition_field:
                table.time_partitioning = bigquery.TimePartitioning(
                    type_=bigquery.TimePartitioningType.DAY,
                    field=partition_field,
                )
                logger.debug(f"[BQ]   Partitioning by: {partition_field}")
            
            # Add clustering if specified
            if clustering_fields:
                table.clustering_fields = clustering_fields
                logger.debug(f"[BQ]   Clustering by: {clustering_fields}")
            
            if description:
                table.description = description
            
            # Create table
            table = client.create_table(table, exists_ok=True)
            logger.info(
                f"[BQ] ✓ Table created successfully: {table_ref} "
                f"({len(schema)} fields)"
            )
            return table
    
    return retry_call(
        _ensure,
        policy=RetryPolicy(max_attempts=3, base_delay_seconds=1.0),
        on_retry=lambda attempt, exc: logger.warning(
            f"[BQ] Retry {attempt} for ensure_table({table_id}): {exc}"
        ),
    )


def get_table_schema(
    client: bigquery.Client,
    dataset_id: str,
    table_id: str,
) -> Optional[List[bigquery.SchemaField]]:
    """Retrieve existing table schema for validation.
    
    Args:
        client: BigQuery client instance
        dataset_id: Dataset ID
        table_id: Table ID
        
    Returns:
        List of SchemaField objects, or None if table doesn't exist
    """
    table_ref = f"{client.project}.{dataset_id}.{table_id}"
    
    def _get():
        try:
            table = client.get_table(table_ref)
            logger.debug(f"[BQ] Retrieved schema for: {table_ref} ({len(table.schema)} fields)")
            return table.schema
        except NotFound:
            logger.warning(f"[BQ] Table not found: {table_ref}")
            return None
    
    return retry_call(
        _get,
        policy=RetryPolicy(max_attempts=3, base_delay_seconds=1.0),
        on_retry=lambda attempt, exc: logger.warning(
            f"[BQ] Retry {attempt} for get_table_schema({table_id}): {exc}"
        ),
    )


def delete_table(
    client: bigquery.Client,
    dataset_id: str,
    table_id: str,
    not_found_ok: bool = True,
) -> bool:
    """Delete a table (helper for testing/cleanup).
    
    Args:
        client: BigQuery client instance
        dataset_id: Dataset ID
        table_id: Table ID
        not_found_ok: If True, don't raise error if table doesn't exist
        
    Returns:
        True if deleted, False if didn't exist (when not_found_ok=True)
        
    Raises:
        NotFound: If table doesn't exist and not_found_ok=False
    """
    table_ref = f"{client.project}.{dataset_id}.{table_id}"
    
    def _delete():
        try:
            client.delete_table(table_ref, not_found_ok=not_found_ok)
            logger.info(f"[BQ] ✓ Table deleted: {table_ref}")
            return True
        except NotFound:
            if not not_found_ok:
                raise
            logger.debug(f"[BQ] Table not found (already deleted): {table_ref}")
            return False
    
    return retry_call(
        _delete,
        policy=RetryPolicy(max_attempts=3, base_delay_seconds=1.0),
        on_retry=lambda attempt, exc: logger.warning(
            f"[BQ] Retry {attempt} for delete_table({table_id}): {exc}"
        ),
    )


# ==============================================================================
# Phase 1B: BigQuery Streaming API
# ==============================================================================

def stream_rows_to_bq(
    client: bigquery.Client,
    dataset_id: str,
    table_id: str,
    rows: List[Dict[str, Any]],
    batch_size: int = 500,
) -> Dict[str, Any]:
    """
    Stream data rows directly to BigQuery table using Streaming API.
    
    Efficiently inserts rows in batches with automatic retry on transient errors.
    Returns summary of successful and failed insertions.
    
    Args:
        client: BigQuery client instance
        dataset_id: Target dataset ID
        table_id: Target table ID
        rows: List of dictionaries representing rows to insert
        batch_size: Number of rows per batch (default: 500, max recommended)
    
    Returns:
        Dict with keys:
            - total_rows: Total number of rows attempted
            - successful_rows: Number of rows inserted successfully
            - failed_rows: Number of rows that failed
            - error_details: List of error details for failed rows
    
    Example:
        >>> rows = [
        ...     {"job_id": "123", "source": "jobstreet", "scrape_timestamp": "2025-12-18T01:00:00Z", "payload": "{}"},
        ...     {"job_id": "456", "source": "mcf", "scrape_timestamp": "2025-12-18T02:00:00Z", "payload": "{}"},
        ... ]
        >>> result = stream_rows_to_bq(client, "sg_job_market", "raw_jobs", rows)
        >>> print(f"Inserted {result['successful_rows']}/{result['total_rows']} rows")
    """
    table_ref = f"{client.project}.{dataset_id}.{table_id}"
    total_rows = len(rows)
    successful_rows = 0
    failed_rows = 0
    error_details = []
    
    logger.info(f"[BQ] Streaming {total_rows} rows to {table_ref} (batch_size={batch_size})")
    
    # Serialize datetime objects to ISO 8601 strings for BigQuery
    from datetime import datetime
    serialized_rows = []
    for row in rows:
        serialized_row = {}
        for key, value in row.items():
            if isinstance(value, datetime):
                serialized_row[key] = value.isoformat()
            else:
                serialized_row[key] = value
        serialized_rows.append(serialized_row)
    
    # Process rows in batches
    for i in range(0, total_rows, batch_size):
        batch = serialized_rows[i:i + batch_size]
        batch_num = (i // batch_size) + 1
        total_batches = (total_rows + batch_size - 1) // batch_size
        
        def _stream_batch():
            """Inner function for retry logic"""
            table = client.get_table(f"{dataset_id}.{table_id}")
            errors = client.insert_rows_json(table, batch)
            
            if errors:
                error_msg = f"Batch {batch_num}/{total_batches} had {len(errors)} errors"
                logger.warning(f"[BQ] {error_msg}")
                for error in errors[:3]:  # Log first 3 errors only
                    logger.warning(f"[BQ]   Error: {error}")
                return errors
            else:
                logger.debug(f"[BQ] ✓ Batch {batch_num}/{total_batches} inserted ({len(batch)} rows)")
                return []
        
        try:
            # Use retry logic for transient network errors
            batch_errors = retry_call(
                _stream_batch,
                policy=RetryPolicy(max_attempts=3, base_delay_seconds=1.0),
                on_retry=lambda attempt, exc: logger.warning(
                    f"[BQ] Retry {attempt} for batch {batch_num}: {exc}"
                ),
            )
            
            # Count successes and failures
            if batch_errors:
                failed_rows += len(batch_errors)
                error_details.extend(batch_errors)
            else:
                successful_rows += len(batch)
                
        except Exception as e:
            logger.error(f"[BQ] Failed to stream batch {batch_num}: {e}")
            failed_rows += len(batch)
            error_details.append({
                "batch": batch_num,
                "error": str(e),
                "row_count": len(batch)
            })
    
    success_rate = (successful_rows / total_rows * 100) if total_rows > 0 else 0
    logger.info(
        f"[BQ] ✓ Streaming complete: {successful_rows}/{total_rows} rows "
        f"({success_rate:.1f}% success rate)"
    )
    
    if failed_rows > 0:
        logger.warning(f"[BQ] ⚠ {failed_rows} rows failed to insert")
    
    return {
        "total_rows": total_rows,
        "successful_rows": successful_rows,
        "failed_rows": failed_rows,
        "error_details": error_details,
    }


def load_jsonl_to_bq(
    client: bigquery.Client,
    jsonl_path: str,
    dataset_id: str,
    table_id: str,
    transform_fn: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
    batch_size: int = 500,
) -> Dict[str, Any]:
    """
    Load local JSONL file to BigQuery table using streaming API.
    
    Reads JSONL file line-by-line, optionally transforms each row,
    then streams to BigQuery in batches.
    
    Args:
        jsonl_path: Path to local JSONL file (can be .jsonl or .jsonl.gz)
        dataset_id: Target dataset ID
        table_id: Target table ID
        transform_fn: Optional function to transform each row before insertion
                     Function signature: (row_dict) -> row_dict
        batch_size: Number of rows per batch (default: 500)
    
    Returns:
        Dict with streaming results (see stream_rows_to_bq)
    
    Example:
        >>> # Load raw_jobs data
        >>> result = load_jsonl_to_bq(
        ...     "data/raw/jobstreet/2025-12-15_121127/dump.jsonl",
        ...     "sg_job_market",
        ...     "raw_jobs"
        ... )
        
        >>> # Load with transformation
        >>> def transform(row):
        ...     row['scrape_timestamp'] = row['scrape_timestamp'].replace('Z', '+00:00')
        ...     return row
        >>> result = load_jsonl_to_bq(
        ...     "dump.jsonl",
        ...     "sg_job_market",
        ...     "raw_jobs",
        ...     transform_fn=transform
        ... )
    """
    import json
    import gzip
    from pathlib import Path
    
    path = Path(jsonl_path)
    
    if not path.exists():
        raise FileNotFoundError(f"JSONL file not found: {jsonl_path}")
    
    logger.info(f"[BQ] Loading JSONL file: {jsonl_path}")
    logger.info(f"[BQ] Target: {client.project}.{dataset_id}.{table_id}")
    
    rows = []
    line_count = 0
    error_count = 0
    
    # Determine if file is gzipped
    open_fn = gzip.open if path.suffix == '.gz' else open
    mode = 'rt' if path.suffix == '.gz' else 'r'
    
    try:
        with open_fn(path, mode, encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                
                try:
                    row = json.loads(line)
                    
                    # Convert dict payload to JSON string for BigQuery JSON columns
                    if 'payload' in row and isinstance(row['payload'], dict):
                        row['payload'] = json.dumps(row['payload'])
                    
                    # Convert ISO 8601 timestamp strings to datetime for TIMESTAMP columns
                    from dateutil import parser as dateparser
                    timestamp_fields = ['scrape_timestamp', 'bq_timestamp', 'job_posted_timestamp']
                    for field in timestamp_fields:
                        if field in row and isinstance(row[field], str):
                            try:
                                row[field] = dateparser.isoparse(row[field])
                            except:
                                pass  # Keep as string if parsing fails
                    
                    # Apply transformation if provided
                    if transform_fn:
                        row = transform_fn(row)
                    
                    rows.append(row)
                    line_count += 1
                    
                    # Progress logging every 100 rows
                    if line_count % 100 == 0:
                        logger.debug(f"[BQ] Parsed {line_count} rows...")
                    
                except json.JSONDecodeError as e:
                    logger.warning(f"[BQ] Invalid JSON on line {line_num}: {e}")
                    error_count += 1
                    continue
        
        logger.info(f"[BQ] Parsed {line_count} rows from JSONL file")
        if error_count > 0:
            logger.warning(f"[BQ] Skipped {error_count} invalid lines")
        
        # Stream to BigQuery
        if rows:
            result = stream_rows_to_bq(client, dataset_id, table_id, rows, batch_size)
            result['parse_errors'] = error_count
            return result
        else:
            logger.warning("[BQ] No valid rows found in JSONL file")
            return {
                "total_rows": 0,
                "successful_rows": 0,
                "failed_rows": 0,
                "error_details": [],
                "parse_errors": error_count,
            }
    
    except Exception as e:
        logger.error(f"[BQ] Failed to load JSONL file: {e}")
        raise


# =============================================================================
# Utility Functions
# =============================================================================

def recreate_tables(
    client: bigquery.Client,
    dataset_id: str,
) -> tuple[bigquery.Table, bigquery.Table]:
    """
    Recreate raw_jobs and cleaned_jobs tables with proper TIMESTAMP partitioning.
    
    ⚠️ WARNING: This DELETES existing tables and all data!
    
    Use this when:
    - Switching from STRING to TIMESTAMP partitioning
    - After schema changes in utils/schemas.py
    - Need to reset tables completely
    
    Args:
        client: BigQuery client instance
        dataset_id: Dataset ID
    
    Returns:
        Tuple of (raw_jobs_table, cleaned_jobs_table)
    
    Example:
        >>> from utils.config import Settings
        >>> settings = Settings.load()
        >>> client = bq_client(settings)
        >>> raw_table, cleaned_table = recreate_tables(client, settings.bigquery_dataset_id)
    """
    from utils.bq_schemas import raw_jobs_schema, cleaned_jobs_schema
    
    logger.warning("=" * 80)
    logger.warning("⚠️  RECREATING BIGQUERY TABLES (DELETES ALL DATA)")
    logger.warning("=" * 80)
    
    # Recreate raw_jobs
    logger.info("[BQ] Recreating raw_jobs table...")
    delete_table(client, dataset_id, "raw_jobs", not_found_ok=True)
    
    raw_schema = raw_jobs_schema()
    logger.info(f"[BQ]   Schema: {len(raw_schema)} fields")
    
    raw_table = ensure_table(
        client,
        dataset_id,
        "raw_jobs",
        raw_schema,
        partition_field="scrape_timestamp",
        clustering_fields=["source", "job_id"],
        description="Raw scraped job data from JobStreet and MCF with TIMESTAMP partitioning"
    )
    logger.info(f"[BQ] ✓ raw_jobs recreated: {raw_table.table_id}")
    
    # Recreate cleaned_jobs
    logger.info("[BQ] Recreating cleaned_jobs table...")
    delete_table(client, dataset_id, "cleaned_jobs", not_found_ok=True)
    
    cleaned_schema = cleaned_jobs_schema()
    logger.info(f"[BQ]   Schema: {len(cleaned_schema)} fields")
    
    cleaned_table = ensure_table(
        client,
        dataset_id,
        "cleaned_jobs",
        cleaned_schema,
        partition_field="scrape_timestamp",
        clustering_fields=["source", "job_id", "company_name"],
        description="Cleaned and transformed job data with TIMESTAMP partitioning"
    )
    logger.info(f"[BQ] ✓ cleaned_jobs recreated: {cleaned_table.table_id}")
    
    logger.warning("[BQ] ✓ Tables recreated successfully (all previous data deleted)")
    
    return raw_table, cleaned_table


# =============================================================================
# CLI Support
# =============================================================================

def _cli_main():
    """Command-line interface for BigQuery utilities."""
    import sys
    from utils.config import Settings
    from utils.logging import configure_logging
    
    if len(sys.argv) < 2:
        print("Usage: python -m utils.bq <command>")
        print("Commands:")
        print("  recreate-tables  - Recreate raw_jobs and cleaned_jobs (⚠️  DELETES DATA)")
        sys.exit(1)
    
    command = sys.argv[1]
    
    if command == "recreate-tables":
        logger_cli = configure_logging(service_name="bq_cli")
        settings = Settings.load()
        client = bq_client(settings)
        
        logger_cli.info(f"Connected to project: {client.project}")
        logger_cli.info(f"Dataset: {settings.bigquery_dataset_id}")
        
        # Confirmation prompt
        response = input("\n⚠️  This will DELETE all data in raw_jobs and cleaned_jobs. Continue? (yes/no): ")
        if response.lower() != "yes":
            logger_cli.info("Operation cancelled.")
            sys.exit(0)
        
        recreate_tables(client, settings.bigquery_dataset_id)
        logger_cli.info("✓ Done")
    else:
        print(f"Unknown command: {command}")
        sys.exit(1)


if __name__ == "__main__":
    _cli_main()
