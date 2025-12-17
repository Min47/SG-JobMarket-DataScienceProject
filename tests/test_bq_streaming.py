"""
Test BigQuery Streaming API (Phase 1B)

Tests stream_rows_to_bq() and load_jsonl_to_bq() functions with real data.
"""

import sys
from pathlib import Path
from utils.config import Settings
from utils.logging import configure_logging
from utils.bq import bq_client, stream_rows_to_bq, load_jsonl_to_bq, get_table_schema

# Setup
logger = configure_logging(service_name="test_bq_streaming")
settings = Settings.load()
client = bq_client(settings)

def main():
    logger.info("=" * 80)
    logger.info("Testing BigQuery Streaming API (Phase 1B)")
    logger.info("=" * 80)
    logger.info(f"✓ Connected to project: {client.project}")
    logger.info(f"✓ Dataset: {settings.bigquery_dataset_id}")
    logger.info(f"✓ Region: {settings.gcp_region}")
    
    # Test 1: Stream small batch of test rows
    logger.info("\n" + "=" * 80)
    logger.info("TEST 1: Stream Test Rows to raw_jobs")
    logger.info("=" * 80)
    
    test_rows = [
        {
            "job_id": "test_001",
            "source": "jobstreet",
            "scrape_timestamp": "2025-12-18T01:00:00Z",
            "payload": '{"title": "Test Job 1"}'
        },
        {
            "job_id": "test_002",
            "source": "mcf",
            "scrape_timestamp": "2025-12-18T02:00:00Z",
            "payload": '{"title": "Test Job 2"}'
        },
    ]
    
    result = stream_rows_to_bq(
        client,
        settings.bigquery_dataset_id,
        "raw_jobs",
        test_rows,
        batch_size=500
    )
    
    logger.info(f"✓ Stream result: {result['successful_rows']}/{result['total_rows']} rows")
    assert result['successful_rows'] == 2, "Should insert 2 test rows"
    assert result['failed_rows'] == 0, "Should have no failures"
    
    # Test 2: Load actual scraped JSONL file
    logger.info("\n" + "=" * 80)
    logger.info("TEST 2: Load JSONL File from JobStreet Scrape")
    logger.info("=" * 80)
    
    # Find most recent JobStreet scrape
    data_dir = Path("data/raw/jobstreet")
    scrape_dirs = sorted([d for d in data_dir.iterdir() if d.is_dir()], reverse=True)
    
    if not scrape_dirs:
        logger.warning("⚠ No JobStreet scrape data found, skipping Test 2")
    else:
        latest_scrape = scrape_dirs[0]
        jsonl_file = latest_scrape / "dump.jsonl"
        
        if not jsonl_file.exists():
            logger.warning(f"⚠ JSONL file not found: {jsonl_file}")
        else:
            logger.info(f"  Loading: {jsonl_file}")
            
            result = load_jsonl_to_bq(
                client,
                str(jsonl_file),
                settings.bigquery_dataset_id,
                "raw_jobs",
                batch_size=500
            )
            
            logger.info(f"✓ Loaded {result['successful_rows']}/{result['total_rows']} rows")
            if result['parse_errors'] > 0:
                logger.warning(f"⚠ {result['parse_errors']} parse errors")
            
            assert result['successful_rows'] > 0, "Should load at least 1 row"
    
    # Test 3: Load MCF scrape data
    logger.info("\n" + "=" * 80)
    logger.info("TEST 3: Load JSONL File from MCF Scrape")
    logger.info("=" * 80)
    
    data_dir = Path("data/raw/mcf")
    scrape_dirs = sorted([d for d in data_dir.iterdir() if d.is_dir()], reverse=True)
    
    if not scrape_dirs:
        logger.warning("⚠ No MCF scrape data found, skipping Test 3")
    else:
        latest_scrape = scrape_dirs[0]
        jsonl_file = latest_scrape / "dump.jsonl"
        
        if not jsonl_file.exists():
            logger.warning(f"⚠ JSONL file not found: {jsonl_file}")
        else:
            logger.info(f"  Loading: {jsonl_file}")
            
            result = load_jsonl_to_bq(
                client,
                str(jsonl_file),
                settings.bigquery_dataset_id,
                "raw_jobs",
                batch_size=500
            )
            
            logger.info(f"✓ Loaded {result['successful_rows']}/{result['total_rows']} rows")
            if result['parse_errors'] > 0:
                logger.warning(f"⚠ {result['parse_errors']} parse errors")
            
            assert result['successful_rows'] > 0, "Should load at least 1 row"
    
    # Test 4: Verify data in BigQuery
    logger.info("\n" + "=" * 80)
    logger.info("TEST 4: Query Data from BigQuery")
    logger.info("=" * 80)
    
    query = f"""
        SELECT 
            source,
            COUNT(*) as count,
            MIN(scrape_timestamp) as earliest,
            MAX(scrape_timestamp) as latest
        FROM `{client.project}.{settings.bigquery_dataset_id}.raw_jobs`
        GROUP BY source
        ORDER BY source
    """
    
    logger.info("  Running query...")
    query_job = client.query(query)
    results = list(query_job.result())
    
    logger.info(f"✓ Query returned {len(results)} source groups:")
    for row in results:
        logger.info(f"    - {row.source}: {row.count} jobs (from {row.earliest} to {row.latest})")
    
    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("✓ ALL TESTS PASSED")
    logger.info("=" * 80)
    logger.info(f"Dataset: {client.project}.{settings.bigquery_dataset_id}")
    logger.info(f"Table: raw_jobs")
    logger.info("=" * 80)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Test failed: {e}", exc_info=True)
        sys.exit(1)
