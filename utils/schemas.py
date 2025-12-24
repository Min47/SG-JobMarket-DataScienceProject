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
from datetime import datetime
from typing import Any, Dict, List, Optional, get_args, get_origin

from google.cloud import bigquery


def _python_type_to_bq_type(python_type: type) -> str:
    """Convert Python type annotation to BigQuery type."""
    # Get origin first to check type structure
    origin = get_origin(python_type)
    
    # Handle Optional[X] by extracting X
    if origin is not None:
        args = get_args(python_type)
        if type(None) in args:
            # It's Optional[X], get the non-None type
            python_type = args[0] if args[0] is not type(None) else args[1]
            # Re-get origin after unwrapping Optional
            origin = get_origin(python_type)
    
    # Check for List type (for ARRAY columns like embeddings)
    if origin is list:
        args = get_args(python_type)
        if args and args[0] == float:
            return "FLOAT64"  # Will be wrapped in ARRAY by _dataclass_to_bq_schema
    
    # Check for Dict type (for JSON columns)
    if origin is dict:
        return "JSON"
    
    # Map Python types to BigQuery types
    type_map = {
        str: "STRING",
        int: "INTEGER",
        float: "FLOAT",
        bool: "BOOLEAN",
        dict: "JSON",
        datetime: "TIMESTAMP",
    }
    
    return type_map.get(python_type, "STRING")  # Default to STRING


def _dataclass_to_bq_schema(dataclass_type: type) -> List[bigquery.SchemaField]:
    """Auto-generate BigQuery schema from dataclass."""
    import typing
    from datetime import datetime as dt_class
    schema_fields = []
    
    for field in fields(dataclass_type):
        field_name = field.name
        field_type = field.type
        
        # Handle string annotations from `from __future__ import annotations`
        if isinstance(field_type, str):
            # Evaluate string annotation to get actual type
            try:
                # Create namespace with typing and datetime
                namespace = {**typing.__dict__, 'datetime': dt_class}
                field_type = eval(field_type, namespace, {})
            except:
                # If evaluation fails, keep as string type
                pass
        
        # Determine if field is nullable (Optional or has default)
        is_optional = (
            get_origin(field_type) is not None and 
            type(None) in get_args(field_type)
        ) or field.default is not dataclass.__class__
        
        mode = "NULLABLE" if is_optional else "REQUIRED"
        bq_type = _python_type_to_bq_type(field_type)
        
        # Special handling for List[float] (embeddings) - use REPEATED mode
        origin = get_origin(field_type)
        if origin is list:
            mode = "REPEATED"
            # bq_type already returns FLOAT64 for List[float]
        
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
    scrape_timestamp: datetime  # Timestamp of when job was scraped
    payload: Dict[str, Any]  # Raw JSON payload from source


@dataclass(frozen=True, slots=True)
class CleanedJob:
    """Cleaned job record produced by ETL.
    
    This is the SINGLE SOURCE OF TRUTH for cleaned job schema.
    BigQuery `cleaned_jobs` table schema is auto-generated from this.
    
    To modify schema: Add/remove/change fields here only.
    """

    # =====
    # Metadata fields
    source: str
    scrape_timestamp: datetime  # Timestamp of when job was scraped
    bq_timestamp: datetime  # BigQuery ingestion timestamp


    # =====
    # Job fields (all will be taken under "payload" > "raw" parent json)
    job_id: str 
    # JobStreet: "job" > "id"
    # MCF: "uuid"

    job_url: str
    # JobStreet: "job" > "shareLink"
    # MCF: "metadata" > "jobDetailsUrl"

    job_title: str
    # JobStreet: "job" > "title"
    # MCF: "title"

    job_description: str # Need to clean HTML tags
    # JobStreet: "job" > "content"
    # MCF: "description"

    job_location: str
    # JobStreet: "job" > "tracking" > "locationInfo" > "location"
    # MCF: "address" > "districts" > first element > "location"

    job_classification: str
    # JobStreet: "job" > "tracking" > "classificationInfo" > "classification" ("job" > "tracking" > "classificationInfo" > "subClassification" by appending after "classification" with a " - " separator if exists)
    # MCF: "categories" > first element > "category"

    job_work_type: str
    # JobStreet: "job" > "workTypes" > "label"
    # MCF: "employmentTypes" > first element > "employmentType"

    job_salary_min_sgd_raw: Optional[float]
    # JobStreet: parsed from "job" > "salary" > "label" (Don't convert to monthly, take raw. Need to extract min value if range)
    # MCF: parsed from "salary" > "minimum" (Don't convert to monthly, take raw.)

    job_salary_max_sgd_raw: Optional[float]
    # JobStreet: parsed from "job" > "salary" > "label" (Don't convert to monthly, take raw. Need to extract max value if range)
    # MCF: parsed from "salary" > "maximum" (Don't convert to monthly, take raw.)

    job_salary_type: str
    # JobStreet: parsed from "job" > "salary" > "label" (e.g., "per month", "per year")
    # MCF: "salary" > "type" > "salaryType" (e.g., "Monthly")

    job_salary_min_sgd_monthly: Optional[float]
    # Converted to monthly SGD equivalent

    job_salary_max_sgd_monthly: Optional[float]
    # Converted to monthly SGD equivalent

    job_currency: str
    # All use SGD temporarily

    job_posted_timestamp: datetime
    # JobStreet: "job" > "listedAt" > "dateTimeUtc"
    # MCF: "metadata" > "updatedAt"


    # =====
    # Employer fields
    company_id: str
    # JobStreet: "companyProfile" > "id"
    # MCF: "postedCompany" > "uen"

    company_url: str
    # JobStreet: "companySearchUrl"
    # MCF: "postedCompany" > "_links" > "self" > "href"

    company_name: str
    # JobStreet: "job" > "advertiser" > "name"
    # MCF: "postedCompany" > "name"

    company_description: str
    # JobStreet: "companyProfile" > "overview" > "description" > "paragraphs" (joined with double newlines)
    # MCF: "postedCompany" > "description"

    company_industry: str
    # JobStreet: "companyProfile" > "overview" > "industry"
    # MCF: Not available

    company_size: str
    # JobStreet: "companyProfile" > "overview" > "size" > "description"
    # MCF: "postedCompany" > "employeeCount"


@dataclass(frozen=True, slots=True)
class JobEmbedding:
    """Job embedding record produced by embedding generator.
    
    This is the SINGLE SOURCE OF TRUTH for job embedding schema.
    BigQuery `job_embeddings` table schema is auto-generated from this.
    
    To modify schema: Add/remove/change fields here only.
    """

    job_id: str
    source: str
    model_name: str  # e.g., "sentence-transformers/all-MiniLM-L6-v2"
    embedding: List[float]  # EMBEDDING_DIM-dimensional vector, 384 floats, array<FLOAT64>
    created_at: datetime  # Timestamp of when embedding was created