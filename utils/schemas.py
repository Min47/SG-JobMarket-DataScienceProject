"""Schema contracts for pipeline records.

**SINGLE SOURCE OF TRUTH**: All schemas are defined here.
BigQuery schemas are auto-generated from these dataclasses.

To add/remove/modify fields:
1. Update the dataclass below
2. BigQuery schemas update automatically
3. Run migrations if tables already exist
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any, Dict, List, Optional, get_args, get_origin

from google.cloud import bigquery


def _python_type_to_bq_type(python_type: type) -> str:
    """Convert Python type annotation to BigQuery type."""
    # Handle Optional[X] by extracting X
    origin = get_origin(python_type)
    if origin is not None:
        args = get_args(python_type)
        if type(None) in args:
            # It's Optional[X], get X
            python_type = args[0] if args[0] is not type(None) else args[1]
    
    # Check for Dict type (for JSON columns)
    if origin is dict or str(python_type).startswith("typing.Dict"):
        return "JSON"
    
    # Map Python types to BigQuery types
    type_map = {
        str: "STRING",
        int: "INTEGER",
        float: "FLOAT",
        bool: "BOOLEAN",
        dict: "JSON",
        Dict: "JSON",
    }
    
    return type_map.get(python_type, "STRING")  # Default to STRING


def _dataclass_to_bq_schema(dataclass_type: type) -> List[bigquery.SchemaField]:
    """Auto-generate BigQuery schema from dataclass."""
    schema_fields = []
    
    for field in fields(dataclass_type):
        field_name = field.name
        field_type = field.type
        
        # Determine if field is nullable (Optional or has default)
        is_optional = (
            get_origin(field_type) is not None and 
            type(None) in get_args(field_type)
        ) or field.default is not dataclass.__class__
        
        mode = "NULLABLE" if is_optional else "REQUIRED"
        bq_type = _python_type_to_bq_type(field_type)
        
        schema_fields.append(
            bigquery.SchemaField(field_name, bq_type, mode=mode)
        )
    
    return schema_fields


@dataclass(frozen=True, slots=True)
class RawJob:
    """Raw job record produced by scrapers.
    
    This is the SINGLE SOURCE OF TRUTH for raw job schema.
    BigQuery `raw_jobs` table schema is auto-generated from this.
    
    To modify schema: Add/remove/change fields here only.
    """

    job_id: str
    source: str
    scrape_timestamp: str  # ISO 8601 timestamp
    payload: Dict[str, Any]  # Raw JSON payload from source


@dataclass(frozen=True, slots=True)
class CleanedJob:
    """Cleaned job record produced by ETL.
    
    This is the SINGLE SOURCE OF TRUTH for cleaned job schema.
    BigQuery `cleaned_jobs` table schema is auto-generated from this.
    
    To modify schema: Add/remove/change fields here only.
    """

    job_id: str
    title: str
    company: str
    location: str
    description_clean: str
    url: str
    source: str
    date_posted: str
    salary_min_monthly_sgd: Optional[float]
    salary_max_monthly_sgd: Optional[float]
    currency: str = "SGD"

