"""Configuration loader.

Loads required settings from environment variables and optional local `.env`.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Optional

from dotenv import load_dotenv


@dataclass(frozen=True, slots=True)
class Settings:
    """Project settings required by all pipeline components."""

    gcp_project_id: str
    bigquery_dataset_id: str
    gcp_region: str
    gcs_bucket: Optional[str] = None

    @staticmethod
    def load(*, env_file: str = ".env") -> "Settings":
        """Load settings from environment; raises ValueError if required vars are missing."""
        load_dotenv(env_file, override=False)

        gcp_project_id = os.getenv("GCP_PROJECT_ID", "").strip()
        bigquery_dataset_id = os.getenv("BIGQUERY_DATASET_ID", "").strip()
        gcp_region = os.getenv("GCP_REGION", "").strip()
        gcs_bucket = os.getenv("GCS_BUCKET", "").strip()

        missing = [
            name
            for name, value in (
                ("GCP_PROJECT_ID", gcp_project_id),
                ("BIGQUERY_DATASET_ID", bigquery_dataset_id),
                ("GCP_REGION", gcp_region),
                ("GCS_BUCKET", gcs_bucket),
            )
            if not value
        ]
        if missing:
            raise ValueError(f"Missing required env var(s): {', '.join(missing)}")

        return Settings(
            gcp_project_id=gcp_project_id,
            bigquery_dataset_id=bigquery_dataset_id,
            gcp_region=gcp_region,
            gcs_bucket=gcs_bucket,
        )

