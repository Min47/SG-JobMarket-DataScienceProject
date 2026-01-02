"""Statistical analysis tools for job market intelligence.

Tools:
- aggregate_stats: Compute salary statistics and job counts by category
"""

import json
import logging
from typing import Optional, Literal

from langchain.tools import tool
from pydantic import BaseModel, Field

from utils.config import Settings
from google.cloud import bigquery

logger = logging.getLogger(__name__)


# =============================================================================
# Tool 3: aggregate_stats
# =============================================================================

class AggregateStatsInput(BaseModel):
    """Input schema for aggregate_stats tool.
    
    Validates grouping and filtering parameters for statistical analysis.
    """
    group_by: Literal["classification", "work_type", "location"] = Field(
        description="Field to group statistics by: 'classification' (job category), 'work_type' (Full Time/Part Time), or 'location' (city/district)",
        examples=["classification", "work_type", "location"]
    )
    classification: Optional[str] = Field(
        default=None,
        description="Filter by job classification/category (optional)",
        examples=["Information & Communication Technology", "Banking & Financial Services"]
    )
    location: Optional[str] = Field(
        default=None,
        description="Filter by location (optional)",
        examples=["Singapore", "Central"]
    )
    work_type: Optional[str] = Field(
        default=None,
        description="Filter by employment type (optional)",
        examples=["Full Time", "Part Time", "Contract"]
    )
    limit: int = Field(
        default=10,
        ge=1,
        le=50,
        description="Maximum number of groups to return"
    )


@tool(args_schema=AggregateStatsInput)
def aggregate_stats(
    group_by: str = "classification",
    classification: Optional[str] = None,
    location: Optional[str] = None,
    work_type: Optional[str] = None,
    limit: int = 10,
) -> str:
    """Compute salary statistics and job counts grouped by category.
    
    Analyzes the job market data to provide:
    - Job counts per group
    - Average salary (min, max, median)
    - Salary ranges (25th, 75th percentile)
    - Distribution insights
    
    Use this when users ask:
    - "What's the average salary for X jobs?"
    - "How many Y positions are available?"
    - "Compare salaries across different job categories"
    - "Show me salary ranges by location"
    
    Args:
        group_by: Field to group by ('classification', 'work_type', 'location')
        classification: Filter by job category (optional)
        location: Filter by location (optional)
        work_type: Filter by employment type (optional)
        limit: Max number of groups to return (1-50, default: 10)
        
    Returns:
        JSON string with:
        - success: bool
        - stats: List of group statistics (count, avg_salary, etc.)
        - summary: Overall statistics across all groups
        - filters: Applied filters
        
    Example:
        >>> aggregate_stats(group_by="classification", limit=5)
        '{"success": true, "stats": [{"group": "IT", "count": 1234, "avg_salary": 6500}, ...]}'
    """
    logger.info(f"[Tool: aggregate_stats] group_by={group_by}, filters: classification={classification}, location={location}, work_type={work_type}")
    
    try:
        # Validate group_by
        if group_by not in ["classification", "work_type", "location"]:
            raise ValueError(f"Invalid group_by: {group_by}. Must be 'classification', 'work_type', or 'location'")
        
        # Load settings
        settings = Settings.load()
        client = bigquery.Client(project=settings.gcp_project_id)
        
        # Build WHERE clause for filters
        where_clauses = ["job_salary_min_sgd_monthly IS NOT NULL"]  # Only jobs with salary data
        query_params = []
        
        if classification:
            where_clauses.append(f"job_classification = @classification")
            query_params.append(bigquery.ScalarQueryParameter("classification", "STRING", classification))
        
        if location:
            where_clauses.append(f"job_location LIKE @location")
            query_params.append(bigquery.ScalarQueryParameter("location", "STRING", f"%{location}%"))
        
        if work_type:
            where_clauses.append(f"job_work_type = @work_type")
            query_params.append(bigquery.ScalarQueryParameter("work_type", "STRING", work_type))
        
        where_clause = " AND ".join(where_clauses)
        
        # Map group_by field
        group_field_map = {
            "classification": "job_classification",
            "work_type": "job_work_type",
            "location": "job_location"
        }
        group_field = group_field_map[group_by]
        
        # Build aggregation query
        query = f"""
        SELECT
            {group_field} AS group_name,
            COUNT(*) AS job_count,
            AVG(job_salary_min_sgd_monthly) AS avg_min_salary,
            AVG(job_salary_max_sgd_monthly) AS avg_max_salary,
            AVG((job_salary_min_sgd_monthly + job_salary_max_sgd_monthly) / 2) AS avg_mid_salary,
            MIN(job_salary_min_sgd_monthly) AS min_salary,
            MAX(job_salary_max_sgd_monthly) AS max_salary,
            APPROX_QUANTILES(job_salary_min_sgd_monthly, 4)[OFFSET(1)] AS p25_salary,
            APPROX_QUANTILES(job_salary_min_sgd_monthly, 4)[OFFSET(2)] AS median_salary,
            APPROX_QUANTILES(job_salary_min_sgd_monthly, 4)[OFFSET(3)] AS p75_salary
        FROM `{settings.gcp_project_id}.{settings.bigquery_dataset_id}.cleaned_jobs`
        WHERE {where_clause}
        GROUP BY {group_field}
        ORDER BY job_count DESC
        LIMIT @limit
        """
        
        query_params.append(bigquery.ScalarQueryParameter("limit", "INT64", limit))
        
        job_config = bigquery.QueryJobConfig(query_parameters=query_params)
        query_job = client.query(query, job_config=job_config)
        results = list(query_job.result(timeout=30))  # 30s timeout to prevent hanging
        
        # Format results
        stats = []
        total_jobs = 0
        
        for row in results:
            group_stats = {
                "group": row.group_name,
                "job_count": row.job_count,
                "avg_min_salary": round(row.avg_min_salary, 2) if row.avg_min_salary else None,
                "avg_max_salary": round(row.avg_max_salary, 2) if row.avg_max_salary else None,
                "avg_mid_salary": round(row.avg_mid_salary, 2) if row.avg_mid_salary else None,
                "min_salary": round(row.min_salary, 2) if row.min_salary else None,
                "max_salary": round(row.max_salary, 2) if row.max_salary else None,
                "p25_salary": round(row.p25_salary, 2) if row.p25_salary else None,
                "median_salary": round(row.median_salary, 2) if row.median_salary else None,
                "p75_salary": round(row.p75_salary, 2) if row.p75_salary else None,
            }
            stats.append(group_stats)
            total_jobs += row.job_count
        
        # Compute overall summary
        if stats:
            all_mid_salaries = [s["avg_mid_salary"] for s in stats if s["avg_mid_salary"]]
            summary = {
                "total_groups": len(stats),
                "total_jobs": total_jobs,
                "overall_avg_salary": round(sum(all_mid_salaries) / len(all_mid_salaries), 2) if all_mid_salaries else None,
                "salary_range": {
                    "min": min(s["min_salary"] for s in stats if s["min_salary"]),
                    "max": max(s["max_salary"] for s in stats if s["max_salary"])
                } if any(s["min_salary"] for s in stats) else None
            }
        else:
            summary = {
                "total_groups": 0,
                "total_jobs": 0,
                "overall_avg_salary": None,
                "salary_range": None
            }
        
        result = {
            "success": True,
            "stats": stats,
            "summary": summary,
            "filters": {
                "group_by": group_by,
                "classification": classification,
                "location": location,
                "work_type": work_type
            }
        }
        
        logger.info(f"[Tool: aggregate_stats] Success: {len(stats)} groups, {total_jobs} total jobs")
        return json.dumps(result, default=str)
        
    except Exception as e:
        logger.error(f"[Tool: aggregate_stats] Error: {e}")
        error_result = {
            "success": False,
            "stats": [],
            "summary": {},
            "error": str(e)
        }
        return json.dumps(error_result)
