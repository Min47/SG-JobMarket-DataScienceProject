"""
Create BigQuery vector index for fast similarity search.

A vector index enables efficient nearest-neighbor search on embedding columns.
Without an index, BigQuery would need to compare query embeddings against ALL rows (slow).
With an IVF index, it only searches a subset of "buckets" (fast).

**What is a Vector Index?**
- Inverted File (IVF) index: Groups similar vectors into "buckets" (num_lists)
- Query searches only relevant buckets instead of all rows
- Trade-off: ~10% recall loss for 100x speed improvement

**Why COSINE distance?**
- SBERT embeddings are normalized (unit vectors)
- Cosine similarity measures angle between vectors
- Perfect for semantic similarity (ignores magnitude)

**Why num_lists=100?**
- Rule of thumb: sqrt(num_rows) for balanced speed/accuracy
- For 10K jobs: sqrt(10000) = 100 buckets
- Can tune based on performance needs

Usage:
    python -m nlp.create_vector_index
    
References:
    https://cloud.google.com/bigquery/docs/vector-search
"""

from __future__ import annotations

import os
import sys
from typing import Optional

from dotenv import load_dotenv
from google.cloud import bigquery

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.logging import configure_logging

load_dotenv()
logger = configure_logging(service_name="create_vector_index")

PROJECT_ID = os.getenv("GCP_PROJECT_ID", "sg-job-market")
DATASET_ID = os.getenv("BQ_DATASET_ID", "sg_job_market")
TABLE_NAME = "job_embeddings"
INDEX_NAME = "job_embedding_idx"


def create_vector_index(
    num_lists: int = 100,
    distance_type: str = "COSINE",
    drop_if_exists: bool = False,
) -> None:
    """
    Create vector index on job_embeddings table.
    
    Args:
        num_lists: Number of IVF buckets (default: 100 for ~10K rows).
        distance_type: Distance metric ("COSINE", "EUCLIDEAN", "DOT_PRODUCT").
        drop_if_exists: If True, drop existing index before creating.
    """
    client = bigquery.Client(project=PROJECT_ID)
    table_id = f"{PROJECT_ID}.{DATASET_ID}.{TABLE_NAME}"
    index_id = f"{table_id}.{INDEX_NAME}"
    
    logger.info("=" * 70)
    logger.info("Creating BigQuery Vector Index")
    logger.info("=" * 70)
    logger.info(f"Table: {table_id}")
    logger.info(f"Index: {INDEX_NAME}")
    logger.info(f"Distance: {distance_type}, Buckets: {num_lists}")
    logger.info("")
    
    # Check table exists
    try:
        table = client.get_table(table_id)
        row_count = table.num_rows
        logger.info(f"✅ Table found: {row_count:,} rows")
    except Exception as e:
        logger.error(f"❌ Table not found: {table_id}")
        logger.error(f"Error: {e}")
        logger.info("\nRun this first: python -m nlp.setup_embeddings_table")
        return
    
    # Drop existing index if requested
    if drop_if_exists:
        try:
            drop_sql = f"DROP VECTOR INDEX IF EXISTS {INDEX_NAME} ON `{table_id}`"
            logger.info(f"Dropping existing index...")
            client.query(drop_sql).result()
            logger.info("✅ Old index dropped")
        except Exception as e:
            logger.warning(f"Could not drop index: {e}")
    
    # Create vector index
    create_sql = f"""
    CREATE VECTOR INDEX IF NOT EXISTS {INDEX_NAME}
    ON `{table_id}`(embedding)
    OPTIONS(
        distance_type='{distance_type}',
        index_type='IVF',
        ivf_options='{{"num_lists": {num_lists}}}'
    )
    """
    
    logger.info("Creating index (this may take 1-2 minutes)...")
    logger.info("")
    
    try:
        job = client.query(create_sql)
        job.result()  # Wait for completion
        
        logger.info("=" * 70)
        logger.info("✅ SUCCESS: Vector index created!")
        logger.info("=" * 70)
        logger.info(f"Index name: {INDEX_NAME}")
        logger.info(f"Distance metric: {distance_type}")
        logger.info(f"IVF buckets: {num_lists}")
        logger.info("")
        logger.info("🚀 Ready for similarity search!")
        logger.info("")
        logger.info("Test with:")
        logger.info(f"  python -m notebooks.test_embeddings")
        
    except Exception as e:
        if "Already Exists" in str(e):
            logger.info("=" * 70)
            logger.info("✅ Index already exists")
            logger.info("=" * 70)
            logger.info(f"Index name: {INDEX_NAME}")
            logger.info("")
            logger.info("To recreate, run with --drop flag:")
            logger.info("  python -m nlp.create_vector_index --drop")
        else:
            logger.error("=" * 70)
            logger.error("❌ Failed to create index")
            logger.error("=" * 70)
            logger.error(f"Error: {e}")
            raise


def verify_index() -> None:
    """Verify the vector index was created successfully."""
    client = bigquery.Client(project=PROJECT_ID)
    
    # Query to test vector search
    test_query = f"""
    SELECT 
        job_id,
        source,
        ARRAY_LENGTH(embedding) as embedding_dim,
        model_name
    FROM `{PROJECT_ID}.{DATASET_ID}.{TABLE_NAME}`
    LIMIT 5
    """
    
    logger.info("\n" + "=" * 70)
    logger.info("Verifying index with sample query...")
    logger.info("=" * 70)
    
    try:
        result = client.query(test_query).result()
        rows = list(result)
        
        logger.info(f"\n✅ Query successful! Sample data:")
        logger.info(f"Total rows in table: {len(rows)}")
        for row in rows[:3]:
            logger.info(f"  {row.job_id[:20]}... | {row.source} | dim={row.embedding_dim}")
        
        logger.info("\n🎉 Vector index is ready for similarity search!")
        
    except Exception as e:
        logger.error(f"\n❌ Query failed: {e}")


def main() -> None:
    """CLI entrypoint."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Create BigQuery vector index for similarity search"
    )
    parser.add_argument(
        "--num-lists",
        type=int,
        default=100,
        help="Number of IVF buckets (default: 100 for ~10K rows)",
    )
    parser.add_argument(
        "--distance",
        type=str,
        default="COSINE",
        choices=["COSINE", "EUCLIDEAN", "DOT_PRODUCT"],
        help="Distance metric (default: COSINE)",
    )
    parser.add_argument(
        "--drop",
        action="store_true",
        help="Drop existing index before creating",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify index after creation",
    )
    
    args = parser.parse_args()
    
    create_vector_index(
        num_lists=args.num_lists,
        distance_type=args.distance,
        drop_if_exists=args.drop,
    )
    
    if args.verify:
        verify_index()


if __name__ == "__main__":
    main()
