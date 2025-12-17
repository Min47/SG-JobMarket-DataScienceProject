"""BigQuery helpers (placeholder).

This module will be expanded to:
- ensure dataset + tables exist
- load JSONL / Parquet into BigQuery
- provide CRUD helpers for smoke tests
"""

from __future__ import annotations

from google.cloud import bigquery

from utils.config import Settings


def bq_client(settings: Settings) -> bigquery.Client:
    """Create a BigQuery client for the configured project."""
    return bigquery.Client(project=settings.gcp_project_id)

