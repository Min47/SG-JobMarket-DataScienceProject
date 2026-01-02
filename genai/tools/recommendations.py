"""Job recommendation tools using semantic similarity.

Tools:
- find_similar_jobs: Recommend jobs based on embedding similarity
"""

import json
import logging
from typing import Optional

from langchain.tools import tool
from pydantic import BaseModel, Field

from utils.config import Settings
from google.cloud import bigquery

logger = logging.getLogger(__name__)


# =============================================================================
# Tool 4: find_similar_jobs
# =============================================================================

class FindSimilarJobsInput(BaseModel):
    """Input schema for find_similar_jobs tool.
    
    Validates job_id, source, and result count for similarity search.
    """
    job_id: str = Field(
        description="Job ID to find similar jobs for",
        min_length=1,
        examples=["12345678", "abc-def-123"]
    )
    source: str = Field(
        description="Job source platform ('jobstreet', 'JobStreet', 'mcf', or 'MCF')",
        pattern="^(jobstreet|JobStreet|mcf|MCF)$",
        examples=["jobstreet", "JobStreet", "mcf", "MCF"]
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of similar jobs to return"
    )
    min_similarity: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Minimum similarity score (0.0-1.0, higher = more similar)"
    )


@tool(args_schema=FindSimilarJobsInput)
def find_similar_jobs(
    job_id: str,
    source: str,
    top_k: int = 5,
    min_similarity: float = 0.7
) -> str:
    """Find jobs similar to a given job using semantic embeddings.
    
    Uses BigQuery vector search to find jobs with similar:
    - Job titles and descriptions
    - Required skills and qualifications
    - Company type and industry
    
    Useful for:
    - "Show me jobs like this one"
    - "What are similar positions?"
    - "Recommend alternative roles"
    
    The similarity score ranges from 0.0 to 1.0:
    - 1.0 = Identical job
    - 0.9-0.95 = Very similar (same role, different company)
    - 0.8-0.9 = Similar (related roles in same field)
    - 0.7-0.8 = Somewhat similar (overlapping skills)
    - <0.7 = Different roles
    
    Args:
        job_id: ID of the reference job
        source: Platform source ('jobstreet' or 'mcf')
        top_k: Number of similar jobs to return (1-20, default: 5)
        min_similarity: Minimum similarity threshold (0.0-1.0, default: 0.7)
        
    Returns:
        JSON string with:
        - success: bool
        - reference_job: Original job details
        - similar_jobs: List of similar jobs with similarity scores
        - count: Number of similar jobs found
        
    Example:
        >>> find_similar_jobs("12345678", "jobstreet", top_k=5)
        '{"success": true, "similar_jobs": [{"job_title": "...", "similarity": 0.92}, ...]}'
    """
    logger.info(f"[Tool: find_similar_jobs] job_id={job_id}, source={source}, top_k={top_k}, min_similarity={min_similarity}")
    
    try:
        # Normalize source to match database values (JobStreet, MCF)
        source_lower = source.lower()
        if source_lower not in ["jobstreet", "mcf"]:
            raise ValueError(f"Invalid source: {source}. Must be 'jobstreet' or 'mcf'")
        
        source_normalized = "JobStreet" if source_lower == "jobstreet" else "MCF"
        
        # Load settings
        settings = Settings.load()
        client = bigquery.Client(project=settings.gcp_project_id)
        
        # Step 1: Get the reference job's embedding
        reference_query = f"""
        SELECT
            e.embedding,
            c.job_title,
            c.company_name,
            c.job_classification
        FROM `{settings.gcp_project_id}.{settings.bigquery_dataset_id}.job_embeddings` e
        JOIN `{settings.gcp_project_id}.{settings.bigquery_dataset_id}.cleaned_jobs` c
          ON e.job_id = c.job_id AND e.source = c.source
        WHERE e.job_id = @job_id
          AND e.source = @source
        LIMIT 1
        """
        
        ref_job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("job_id", "STRING", job_id),
                bigquery.ScalarQueryParameter("source", "STRING", source_normalized),
            ]
        )
        
        ref_query_job = client.query(reference_query, job_config=ref_job_config)
        ref_results = list(ref_query_job.result(timeout=30))  # 30s timeout
        
        if not ref_results:
            result = {
                "success": False,
                "reference_job": None,
                "similar_jobs": [],
                "error": f"Reference job not found: {job_id} from {source}"
            }
            logger.warning(f"[Tool: find_similar_jobs] Reference job not found")
            return json.dumps(result)
        
        ref_row = ref_results[0]
        ref_embedding = ref_row.embedding
        ref_job = {
            "job_id": job_id,
            "source": source,
            "job_title": ref_row.job_title,
            "company_name": ref_row.company_name,
            "job_classification": ref_row.job_classification
        }
        
        # Step 2: Find similar jobs using vector search
        # Convert min_similarity to max distance (cosine distance = 1 - similarity)
        max_distance = 1.0 - min_similarity
        
        similarity_query = f"""
        SELECT
            base.job_id,
            base.source,
            base.job_title,
            base.company_name,
            base.job_location,
            base.job_classification,
            base.job_work_type,
            base.job_salary_min_sgd_monthly,
            base.job_salary_max_sgd_monthly,
            base.job_url,
            vs.distance AS vector_distance,
            (1.0 - vs.distance) AS similarity_score
        FROM VECTOR_SEARCH(
            TABLE `{settings.gcp_project_id}.{settings.bigquery_dataset_id}.job_embeddings`,
            'embedding',
            (SELECT @embedding AS embedding),
            top_k => @top_k_plus_one,
            distance_type => 'COSINE'
        ) AS vs
        JOIN `{settings.gcp_project_id}.{settings.bigquery_dataset_id}.cleaned_jobs` base
          ON base.job_id = vs.base.job_id
          AND base.source = vs.base.source
        WHERE vs.distance <= @max_distance
          AND NOT (base.job_id = @exclude_job_id AND base.source = @exclude_source)
        ORDER BY vs.distance ASC
        LIMIT @top_k
        """
        
        sim_job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ArrayQueryParameter("embedding", "FLOAT64", ref_embedding),
                bigquery.ScalarQueryParameter("top_k_plus_one", "INT64", top_k + 1),  # +1 to account for self
                bigquery.ScalarQueryParameter("max_distance", "FLOAT64", max_distance),
                bigquery.ScalarQueryParameter("exclude_job_id", "STRING", job_id),
                bigquery.ScalarQueryParameter("exclude_source", "STRING", source_normalized),
                bigquery.ScalarQueryParameter("top_k", "INT64", top_k),
            ]
        )
        
        sim_query_job = client.query(similarity_query, job_config=sim_job_config)
        sim_results = list(sim_query_job.result(timeout=30))  # 30s timeout
        
        # Format similar jobs
        similar_jobs = []
        for row in sim_results:
            job = {
                "job_id": row.job_id,
                "source": row.source,
                "job_title": row.job_title,
                "company_name": row.company_name,
                "job_location": row.job_location,
                "job_classification": row.job_classification,
                "job_work_type": row.job_work_type,
                "job_salary_min_sgd_monthly": row.job_salary_min_sgd_monthly,
                "job_salary_max_sgd_monthly": row.job_salary_max_sgd_monthly,
                "job_url": row.job_url,
                "similarity_score": round(row.similarity_score, 4),
                "vector_distance": round(row.vector_distance, 4)
            }
            similar_jobs.append(job)
        
        result = {
            "success": True,
            "reference_job": ref_job,
            "similar_jobs": similar_jobs,
            "count": len(similar_jobs),
            "search_params": {
                "top_k": top_k,
                "min_similarity": min_similarity
            }
        }
        
        logger.info(f"[Tool: find_similar_jobs] Success: Found {len(similar_jobs)} similar jobs")
        return json.dumps(result, default=str)
        
    except Exception as e:
        logger.error(f"[Tool: find_similar_jobs] Error: {e}")
        error_result = {
            "success": False,
            "reference_job": None,
            "similar_jobs": [],
            "count": 0,
            "error": str(e)
        }
        return json.dumps(error_result)
