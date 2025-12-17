"""BigQuery schema definitions.

**AUTO-GENERATED FROM DATACLASSES**
Do NOT manually edit schemas here. Update `utils.schemas` instead.

This module provides BigQuery schema functions that auto-generate
from the dataclass definitions in `utils.schemas`.
"""

from __future__ import annotations

from typing import List

from google.cloud import bigquery

from utils.schemas import RawJob, CleanedJob, _dataclass_to_bq_schema


def raw_jobs_schema() -> List[bigquery.SchemaField]:
    """Schema for the `raw_jobs` table.
    
    Auto-generated from `utils.schemas.RawJob` dataclass.
    """
    return _dataclass_to_bq_schema(RawJob)


def cleaned_jobs_schema() -> List[bigquery.SchemaField]:
    """Schema for the `cleaned_jobs` table.
    
    Auto-generated from `utils.schemas.CleanedJob` dataclass.
    """
    return _dataclass_to_bq_schema(CleanedJob)

