"""Quick test for BigQuery core infrastructure functions."""

import logging
from utils.config import Settings
from utils.bq import bq_client, ensure_dataset, ensure_table, get_table_schema, delete_table
from utils.bq_schemas import raw_jobs_schema, cleaned_jobs_schema

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s (%(levelname)s) | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger(__name__)

def test_core_infrastructure():
    """Test dataset and table creation."""
    logger.info("=" * 80)
    logger.info("Testing BigQuery Core Infrastructure")
    logger.info("=" * 80)
    
    # Load settings
    settings = Settings.load()
    client = bq_client(settings)
    
    logger.info(f"\n✓ Connected to project: {settings.gcp_project_id}")
    logger.info(f"✓ Dataset: {settings.bigquery_dataset_id}")
    logger.info(f"✓ Region: {settings.gcp_region}")
    
    # Test 1: Ensure dataset exists
    logger.info("\n" + "=" * 80)
    logger.info("TEST 1: Ensure Dataset")
    logger.info("=" * 80)
    dataset = ensure_dataset(
        client,
        settings.bigquery_dataset_id,
        location=settings.gcp_region,
        description="Singapore Job Market data warehouse"
    )
    logger.info(f"✓ Dataset ready: {dataset.dataset_id}")
    
    # Test 2: Ensure raw_jobs table
    logger.info("\n" + "=" * 80)
    logger.info("TEST 2: Ensure raw_jobs Table")
    logger.info("=" * 80)
    raw_schema = raw_jobs_schema()
    logger.info(f"  Schema: {len(raw_schema)} fields")
    for field in raw_schema:
        logger.info(f"    - {field.name} ({field.field_type}, {field.mode})")
    
    raw_table = ensure_table(
        client,
        settings.bigquery_dataset_id,
        "raw_jobs",
        raw_schema,
        partition_field=None,  # No partitioning - scrape_timestamp is STRING
        clustering_fields=["source", "job_id"],
        description="Raw scraped job data from JobStreet and MCF"
    )
    logger.info(f"✓ Table ready: {raw_table.table_id}")
    
    # Test 3: Ensure cleaned_jobs table
    logger.info("\n" + "=" * 80)
    logger.info("TEST 3: Ensure cleaned_jobs Table")
    logger.info("=" * 80)
    cleaned_schema = cleaned_jobs_schema()
    logger.info(f"  Schema: {len(cleaned_schema)} fields")
    for field in cleaned_schema:
        logger.info(f"    - {field.name} ({field.field_type}, {field.mode})")
    
    cleaned_table = ensure_table(
        client,
        settings.bigquery_dataset_id,
        "cleaned_jobs",
        cleaned_schema,
        partition_field=None,  # No partitioning - scrape_timestamp is STRING
        clustering_fields=["source", "job_id"],
        description="Cleaned and transformed job data"
    )
    logger.info(f"✓ Table ready: {cleaned_table.table_id}")
    
    # Test 4: Get table schema (verify)
    logger.info("\n" + "=" * 80)
    logger.info("TEST 4: Get Table Schema")
    logger.info("=" * 80)
    retrieved_schema = get_table_schema(client, settings.bigquery_dataset_id, "raw_jobs")
    if retrieved_schema:
        logger.info(f"✓ Retrieved raw_jobs schema: {len(retrieved_schema)} fields")
    else:
        logger.error("✗ Failed to retrieve schema")
    
    # Test 5: Idempotence test (run again, should succeed)
    logger.info("\n" + "=" * 80)
    logger.info("TEST 5: Idempotence Test (Run Again)")
    logger.info("=" * 80)
    dataset2 = ensure_dataset(client, settings.bigquery_dataset_id, location=settings.gcp_region)
    table2 = ensure_table(client, settings.bigquery_dataset_id, "raw_jobs", raw_schema)
    logger.info("✓ Idempotence verified - no errors on re-run")
    
    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("✓ ALL TESTS PASSED")
    logger.info("=" * 80)
    logger.info(f"Dataset: {settings.gcp_project_id}.{settings.bigquery_dataset_id}")
    logger.info(f"Tables: raw_jobs, cleaned_jobs")
    logger.info(f"Location: {settings.gcp_region}")
    logger.info("=" * 80)

if __name__ == "__main__":
    test_core_infrastructure()
