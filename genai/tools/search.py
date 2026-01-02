"""Search and retrieval tools for job market agent.

Tools:
- search_jobs: Enhanced job search with filters and hybrid scoring
- get_job_details: Fetch complete job information by ID
"""

import json
import logging
from typing import Optional, Dict, Any

from langchain.tools import tool
from pydantic import BaseModel, Field

from genai.rag import retrieve_jobs
from genai.tools._validation import JobFilters
from utils.config import Settings
from google.cloud import bigquery

logger = logging.getLogger(__name__)


# =============================================================================
# Tool 1: search_jobs
# =============================================================================

class SearchJobsInput(BaseModel):
    """Input schema for search_jobs tool.
    
    Validates user query and optional filters for job search.
    """
    query: str = Field(
        description="Natural language job search query (e.g., 'python developer', 'data scientist')",
        min_length=2,
        max_length=500,
        examples=["python developer", "data scientist with ML experience", "marketing manager"]
    )
    location: Optional[str] = Field(
        default=None,
        description="Filter by location (e.g., 'Singapore', 'Central')",
    )
    min_salary: Optional[float] = Field(
        default=None,
        ge=0,
        le=100000,
        description="Minimum monthly salary in SGD"
    )
    max_salary: Optional[float] = Field(
        default=None,
        ge=0,
        le=100000,
        description="Maximum monthly salary in SGD"
    )
    work_type: Optional[str] = Field(
        default=None,
        description="Employment type (e.g., 'Full Time', 'Contract')"
    )
    max_results: int = Field(
        default=10,
        ge=1,
        le=50,
        description="Maximum number of results to return"
    )


@tool(args_schema=SearchJobsInput)
def search_jobs(
    query: str,
    location: Optional[str] = None,
    min_salary: Optional[float] = None,
    max_salary: Optional[float] = None,
    work_type: Optional[str] = None,
    max_results: int = 10,
) -> str:
    """Search for job postings in Singapore using natural language queries.
    
    This tool performs semantic search across 6,000+ job listings using:
    - Vector embeddings for semantic matching
    - Keyword matching for exact term hits
    - Hybrid scoring (70% vector + 30% keyword)
    
    Use this when users ask:
    - "Find me jobs in X"
    - "What are the available Y positions?"
    - "Show me Z roles with salary > $N"
    
    Args:
        query: Natural language job search (e.g., "python developer")
        location: Filter by city/district (optional)
        min_salary: Minimum monthly salary in SGD (optional)
        max_salary: Maximum monthly salary in SGD (optional)
        work_type: Employment type filter (optional)
        max_results: Number of results (1-50, default: 10)
        
    Returns:
        JSON string with:
        - success: bool
        - count: int (number of results)
        - jobs: List of job objects with title, company, salary, etc.
        - query_info: Search metadata
        
    Example:
        >>> search_jobs("data scientist", min_salary=5000, max_results=5)
        '{"success": true, "count": 5, "jobs": [...]}'
    """
    logger.info(f"[Tool: search_jobs] Query: '{query}', Filters: location={location}, salary={min_salary}-{max_salary}")
    
    try:
        # Build filters dict
        filters = {}
        if location:
            filters["location"] = location
        if min_salary:
            filters["min_salary"] = min_salary
        if max_salary:
            filters["max_salary"] = max_salary
        if work_type:
            filters["work_type"] = work_type
        
        # Load settings
        settings = Settings.load()
        
        # Call RAG retrieve_jobs function
        jobs = retrieve_jobs(
            query=query,
            top_k=max_results,
            filters=filters if filters else None,
            settings=settings
        )
        
        # Format response
        result = {
            "success": True,
            "count": len(jobs),
            "jobs": jobs,
            "query_info": {
                "query": query,
                "filters": filters,
                "max_results": max_results
            }
        }
        
        logger.info(f"[Tool: search_jobs] Success: {len(jobs)} jobs found")
        return json.dumps(result, default=str)
        
    except Exception as e:
        logger.error(f"[Tool: search_jobs] Error: {e}")
        error_result = {
            "success": False,
            "count": 0,
            "jobs": [],
            "error": str(e)
        }
        return json.dumps(error_result)


# =============================================================================
# Tool 2: get_job_details
# =============================================================================

class GetJobDetailsInput(BaseModel):
    """Input schema for get_job_details tool.
    
    Validates job_id and source for fetching full job record.
    """
    job_id: str = Field(
        description="Unique job identifier",
        min_length=1,
        examples=["12345678", "abc-def-123"]
    )
    source: str = Field(
        description="Job source platform ('jobstreet', 'JobStreet', 'mcf', or 'MCF')",
        pattern="^(jobstreet|JobStreet|mcf|MCF)$",
        examples=["jobstreet", "JobStreet", "mcf", "MCF"]
    )


@tool(args_schema=GetJobDetailsInput)
def get_job_details(job_id: str, source: str) -> str:
    """Fetch complete details for a specific job posting.
    
    Retrieves the full job record from BigQuery, including:
    - Complete job description (no truncation)
    - Detailed company information
    - Full salary information
    - Application URL
    - Posting timestamp
    
    Use this when users ask:
    - "Tell me more about job X"
    - "Show me the full details for this position"
    - "What's the description for job ID Y?"
    
    Args:
        job_id: Unique job identifier from search results
        source: Platform source ('jobstreet' or 'mcf')
        
    Returns:
        JSON string with:
        - success: bool
        - job: Complete job object (if found)
        - error: Error message (if failed)
        
    Example:
        >>> get_job_details("12345678", "jobstreet")
        '{"success": true, "job": {...}}'
    """
    logger.info(f"[Tool: get_job_details] Fetching job_id={job_id}, source={source}")
    
    try:
        # Normalize source to match database values (JobStreet, MCF)
        source_lower = source.lower()
        if source_lower not in ["jobstreet", "mcf"]:
            raise ValueError(f"Invalid source: {source}. Must be 'jobstreet' or 'mcf'")
        
        source_normalized = "JobStreet" if source_lower == "jobstreet" else "MCF"
        
        # Load settings and create BigQuery client
        settings = Settings.load()
        client = bigquery.Client(project=settings.gcp_project_id)
        
        # Query for specific job
        query = f"""
        SELECT
            job_id,
            source,
            job_title,
            job_description,
            job_location,
            job_classification,
            job_work_type,
            job_salary_min_sgd_monthly,
            job_salary_max_sgd_monthly,
            job_salary_type,
            job_url,
            job_posted_timestamp,
            company_name,
            company_description,
            company_industry,
            company_size,
            company_url
        FROM `{settings.gcp_project_id}.{settings.bigquery_dataset_id}.cleaned_jobs`
        WHERE job_id = @job_id
          AND source = @source
        LIMIT 1
        """
        
        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("job_id", "STRING", job_id),
                bigquery.ScalarQueryParameter("source", "STRING", source_normalized),
            ]
        )
        
        query_job = client.query(query, job_config=job_config)
        results = list(query_job.result(timeout=30))  # 30s timeout to prevent hanging
        
        if not results:
            result = {
                "success": False,
                "job": None,
                "error": f"Job not found: {job_id} from {source}"
            }
            logger.warning(f"[Tool: get_job_details] Job not found")
            return json.dumps(result)
        
        # Convert to dict
        row = results[0]
        job = dict(row.items())
        
        result = {
            "success": True,
            "job": job
        }
        
        logger.info(f"[Tool: get_job_details] Success: Found job '{job.get('job_title')}'")
        return json.dumps(result, default=str)
        
    except Exception as e:
        logger.error(f"[Tool: get_job_details] Error: {e}")
        error_result = {
            "success": False,
            "job": None,
            "error": str(e)
        }
        return json.dumps(error_result)
