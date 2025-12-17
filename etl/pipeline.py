"""ETL orchestration (placeholder).

This module will:
- read raw JSONL from GCS
- clean and normalize fields
- write Parquet to GCS
- load into BigQuery
"""

from __future__ import annotations

from typing import Iterable, Iterator

from utils.schemas import CleanedJob, RawJob


def transform(raw_jobs: Iterable[RawJob]) -> Iterator[CleanedJob]:
    """Transform raw jobs into cleaned jobs (placeholder)."""
    _ = raw_jobs
    return iter(())

