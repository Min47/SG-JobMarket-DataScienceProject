"""
Create job_embeddings table in BigQuery.

This script creates the BigQuery table for storing job embeddings.
Run this once before generating embeddings.

Usage:
    python -m nlp.setup_embeddings_table
"""

from __future__ import annotations

import os
import sys

from dotenv import load_dotenv
from google.cloud import bigquery

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.logging import configure_logging
from utils.schemas import JobEmbedding
from utils.bq_schemas import job_embeddings_schema

load_dotenv()
logger = configure_logging(service_name="setup_embeddings_table")

PROJECT_ID = os.getenv("GCP_PROJECT_ID", "sg-job-market")
DATASET_ID = os.getenv("BQ_DATASET_ID", "sg_job_market")
TABLE_NAME = "job_embeddings"


def create_embeddings_table() -> None:
    """Create job_embeddings table in BigQuery."""
    client = bigquery.Client(project=PROJECT_ID)
    table_id = f"{PROJECT_ID}.{DATASET_ID}.{TABLE_NAME}"

    # Get schema from JobEmbedding dataclass
    schema = job_embeddings_schema()

    # Define table with partitioning and clustering
    table = bigquery.Table(table_id, schema=schema)

    # Partition by created_at (TIMESTAMP) for efficient querying
    table.time_partitioning = bigquery.TimePartitioning(
        type_=bigquery.TimePartitioningType.DAY,
        field="created_at",
    )

    # Cluster by source and job_id for fast lookups
    table.clustering_fields = ["source", "job_id"]

    # Create table (or skip if exists)
    try:
        table = client.create_table(table, exists_ok=True)
        logger.info(f"✅ Created table: {table_id}")
        logger.info(f"   Partitioned by: created_at (DAY)")
        logger.info(f"   Clustered by: source, job_id")
        logger.info(f"   Schema: {len(schema)} fields")

        # Print schema
        for field in schema:
            logger.info(f"      - {field.name}: {field.field_type} ({field.mode})")

    except Exception as e:
        logger.error(f"❌ Failed to create table: {e}")
        raise


def verify_table() -> None:
    """Verify table exists and show sample data."""
    client = bigquery.Client(project=PROJECT_ID)
    table_id = f"{PROJECT_ID}.{DATASET_ID}.{TABLE_NAME}"

    # Check if table exists
    try:
        table = client.get_table(table_id)
        logger.info(f"✅ Table exists: {table_id}")
        logger.info(f"   Rows: {table.num_rows}")
        logger.info(f"   Size: {table.num_bytes / (1024**2):.2f} MB")

        # Sample query
        if table.num_rows > 0:
            query = f"""
                SELECT 
                    job_id,
                    source,
                    model_name,
                    ARRAY_LENGTH(embedding) as embedding_dim,
                    created_at
                FROM `{table_id}`
                LIMIT 5
            """
            results = client.query(query).result()
            logger.info("   Sample rows:")
            for row in results:
                logger.info(f"      {dict(row)}")

    except Exception as e:
        logger.error(f"❌ Table does not exist: {e}")


def main():
    """Main entry point."""
    logger.info("=" * 60)
    logger.info("Setting up job_embeddings table in BigQuery")
    logger.info("=" * 60)

    # Create table
    create_embeddings_table()

    # Verify
    verify_table()

    logger.info("=" * 60)
    logger.info("✅ Setup complete!")
    logger.info("Next step: Run embedding generation")
    logger.info("   .venv/Scripts/python.exe -m nlp.generate_embeddings --limit 100")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
