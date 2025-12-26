"""
Generate embeddings for job descriptions and store in BigQuery.

This script:
1. Queries cleaned_jobs from BigQuery
2. Generates embeddings using Sentence-BERT
3. Writes embeddings to job_embeddings table
4. Supports incremental updates (only new jobs)

**Prerequisites:**
    Run this FIRST to create the table:
    python -m nlp.setup_embeddings_table

Usage:
    python -m nlp.generate_embeddings --limit 1000
    python -m nlp.generate_embeddings --full
"""

from __future__ import annotations

import argparse
import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from nlp.setup_embeddings_table import create_embeddings_table

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# Configuration
PROJECT_ID = os.getenv("GCP_PROJECT_ID", "sg-job-market")
DATASET_ID = os.getenv("BQ_DATASET_ID", "sg_job_market")
SOURCE_TABLE = "cleaned_jobs"
TARGET_TABLE = "job_embeddings"


def get_jobs_to_embed(
    client: Any,
    limit: Optional[int] = None,
    only_new: bool = True,
    target_date: Optional[datetime] = None,
) -> List[Dict[str, Any]]:
    """
    Query jobs that need embeddings.

    Args:
        client: BigQuery client.
        limit: Maximum jobs to return.
        only_new: If True, only return jobs without embeddings.
        target_date: If provided, filter jobs scraped on this date (UTC).

    Returns:
        List of job dicts with job_id, source, title, description.
    """
    if only_new:
        # Only jobs not yet embedded (using latest version from append-only table)
        date_filter = f"AND DATE(scrape_timestamp) = '{target_date.strftime('%Y-%m-%d')}'" if target_date else ""
        
        query = f"""
        WITH latest_jobs AS (
            SELECT 
                job_id,
                source,
                job_title,
                job_description,
                ROW_NUMBER() OVER (
                    PARTITION BY source, job_id 
                    ORDER BY scrape_timestamp DESC
                ) AS rn
            FROM `{PROJECT_ID}.{DATASET_ID}.{SOURCE_TABLE}`
            WHERE 1=1 {date_filter}
        )
        SELECT 
            c.job_id,
            c.source,
            c.job_title,
            SUBSTR(c.job_description, 1, 2000) as job_description
        FROM latest_jobs c
        LEFT JOIN `{PROJECT_ID}.{DATASET_ID}.{TARGET_TABLE}` e
            ON c.job_id = e.job_id AND c.source = e.source
        WHERE c.rn = 1 AND e.job_id IS NULL
        """
    else:
        # Get all jobs (using latest version from append-only table)
        date_filter = f"WHERE DATE(scrape_timestamp) = '{target_date.strftime('%Y-%m-%d')}'" if target_date else ""
        
        query = f"""
        WITH latest_jobs AS (
            SELECT 
                job_id,
                source,
                job_title,
                job_description,
                ROW_NUMBER() OVER (
                    PARTITION BY source, job_id 
                    ORDER BY scrape_timestamp DESC
                ) AS rn
            FROM `{PROJECT_ID}.{DATASET_ID}.{SOURCE_TABLE}`
            {date_filter}
        )
        SELECT 
            job_id,
            source,
            job_title,
            SUBSTR(job_description, 1, 2000) as job_description
        FROM latest_jobs
        WHERE rn = 1
        """
    
    # Remove old date filter logic since it's now in the CTE
    # Add limit after WHERE clause
    if limit:
        query += f" LIMIT {limit}"

    date_info = f", target_date={target_date.strftime('%Y-%m-%d')}" if target_date else ""
    logger.info(f"Querying jobs to embed (only_new={only_new}, limit={limit}{date_info})")

    result = client.query(query).result()
    jobs = [dict(row) for row in result]
    logger.info(f"Found {len(jobs)} jobs to embed")
    return jobs


def write_embeddings_to_bq(
    client: Any,
    embeddings_data: List[Dict[str, Any]],
    batch_size: int = 500,
) -> int:
    """
    Write embeddings to BigQuery using streaming insert in batches.

    Args:
        client: BigQuery client.
        embeddings_data: List of dicts with job_id, source, embedding, model_name.
        batch_size: Number of rows to insert per batch (default 500).

    Returns:
        Number of rows inserted.
    """
    from google.cloud import bigquery

    table_id = f"{PROJECT_ID}.{DATASET_ID}.{TARGET_TABLE}"

    # Add timestamp
    timestamp = datetime.now(timezone.utc).isoformat()
    for row in embeddings_data:
        row["created_at"] = timestamp

    # Insert in batches to avoid timeouts
    total_inserted = 0
    for i in range(0, len(embeddings_data), batch_size):
        batch = embeddings_data[i:i + batch_size]
        
        # Add retry logic with timeout
        max_retries = 3
        for attempt in range(max_retries):
            try:
                # Insert with timeout (30 seconds per batch)
                import time
                start_time = time.time()
                errors = client.insert_rows_json(table_id, batch, timeout=30.0)
                elapsed = time.time() - start_time
                
                if errors:
                    logger.error(f"BigQuery insert errors for batch {i//batch_size + 1}: {errors[:5]}")
                    if attempt < max_retries - 1:
                        wait_time = 2 ** attempt  # Exponential backoff: 1s, 2s, 4s
                        logger.warning(f"Retrying in {wait_time}s... (attempt {attempt + 1}/{max_retries})")
                        time.sleep(wait_time)
                        continue
                    else:
                        raise RuntimeError(f"Failed to insert {len(errors)} rows in batch {i//batch_size + 1}")
                
                total_inserted += len(batch)
                logger.info(f"✅ Batch {i//batch_size + 1}: {len(batch)} embeddings in {elapsed:.1f}s ({total_inserted}/{len(embeddings_data)} total)")
                break  # Success, exit retry loop
                
            except Exception as e:
                logger.error(f"BigQuery insert exception for batch {i//batch_size + 1}: {e}")
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    logger.warning(f"Retrying in {wait_time}s... (attempt {attempt + 1}/{max_retries})")
                    time.sleep(wait_time)
                else:
                    raise

    logger.info(f"✅ Successfully inserted all {total_inserted} embeddings to BigQuery")
    return total_inserted


def generate_embeddings(
    limit: Optional[int] = None,
    batch_size: int = 32,
    only_new: bool = True,
    dry_run: bool = False,
    target_date: Optional[datetime] = None,
) -> Dict[str, Any]:
    """
    Main embedding generation pipeline.

    Args:
        limit: Maximum jobs to process.
        batch_size: Embedding batch size.
        only_new: Only embed jobs without existing embeddings.
        dry_run: If True, don't write to BigQuery.
        target_date: If provided, filter jobs scraped on this date (UTC).

    Returns:
        Dict with processing statistics.
    """
    from google.cloud import bigquery
    from nlp.embeddings import EmbeddingGenerator

    logger.info("=" * 50)
    logger.info("Starting embedding generation pipeline")
    date_info = f", target_date={target_date.strftime('%Y-%m-%d')}" if target_date else ""
    logger.info(f"  limit={limit}, batch_size={batch_size}, only_new={only_new}{date_info}")
    logger.info("=" * 50)

    # Initialize clients
    bq_client = bigquery.Client(project=PROJECT_ID)
    embedding_generator = EmbeddingGenerator()

    # Get jobs
    jobs = get_jobs_to_embed(bq_client, limit=limit, only_new=only_new, target_date=target_date)

    if not jobs:
        logger.info("No jobs to embed")
        return {"jobs_processed": 0, "status": "no_jobs"}

    # Filter out jobs with both empty title AND description
    jobs_before_filter = len(jobs)
    jobs = [
        j for j in jobs 
        if (j.get('job_title') or '').strip() or (j.get('job_description') or '').strip()
    ]
    if len(jobs) < jobs_before_filter:
        logger.warning(f"Filtered out {jobs_before_filter - len(jobs)} jobs with empty title AND description")

    if not jobs:
        logger.info("No valid jobs to embed after filtering")
        return {"jobs_processed": 0, "status": "no_valid_jobs"}

    # Prepare texts (combine title + description, truncate description to 1000 chars)
    texts = []
    for job in jobs:
        title = (job.get('job_title') or 'Unknown').strip()
        description = (job.get('job_description') or '').strip()[:1000]
        # Combine with period separator
        text = f"{title}. {description}" if description else title
        texts.append(text)
    
    logger.info(f"Prepared {len(texts)} texts for embedding")

    # Generate embeddings
    logger.info(f"Generating embeddings for {len(texts)} texts...")
    embeddings = embedding_generator.embed_texts(texts, batch_size=batch_size)

    # Prepare for BigQuery
    embeddings_data = []
    for job, embedding in zip(jobs, embeddings):
        embeddings_data.append({
            "job_id": job["job_id"],
            "source": job["source"],
            "embedding": embedding.tolist(),
            "model_name": embedding_generator.model_name,
        })
    
    # Clear memory after embeddings generation
    import gc
    del texts, embeddings
    gc.collect()
    logger.info("🧹 Cleared embedding tensors from memory")

    # Write to BigQuery
    if dry_run:
        logger.info(f"Dry run: would write {len(embeddings_data)} embeddings")
    else:
        write_embeddings_to_bq(bq_client, embeddings_data)

    result = {
        "jobs_processed": len(jobs),
        "embeddings_generated": len(embeddings_data),
        "embedding_dim": embeddings.shape[1],
        "model_name": embedding_generator.model_name,
        "status": "success",
    }
    
    if target_date:
        result["target_date"] = target_date.strftime("%Y-%m-%d")

    logger.info("=" * 50)
    logger.info(f"Complete: {result}")
    logger.info("=" * 50)

    return result


def main():
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(
        description="Generate job embeddings and store in BigQuery"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum jobs to process (default: all)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Embedding batch size (default: 32)",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Process all jobs (not just new ones)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Don't write to BigQuery",
    )
    parser.add_argument(
        "--create-table",
        action="store_true",
        help="Create job_embeddings table if not exists",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Create table if requested
    if args.create_table:
        from google.cloud import bigquery
        client = bigquery.Client(project=PROJECT_ID)
        create_embeddings_table(client)

    # Generate embeddings
    generate_embeddings(
        limit=args.limit,
        batch_size=args.batch_size,
        only_new=not args.full,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
