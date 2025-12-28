"""Tool adapters for LangGraph agent.

This module exports LangChain-compatible tools that extend the agent's capabilities:
- search_jobs: Enhanced job search with filters
- get_job_details: Fetch detailed job information by ID
- aggregate_stats: Salary statistics and job counts
- find_similar_jobs: Job recommendations based on similarity

All tools include:
- Pydantic schema validation
- Error handling with graceful fallbacks
- Timeout protection (30s max)
- Structured JSON output
"""

from genai.tools.search import search_jobs, get_job_details
from genai.tools.stats import aggregate_stats
from genai.tools.recommendations import find_similar_jobs

__all__ = [
    "search_jobs",
    "get_job_details",
    "aggregate_stats",
    "find_similar_jobs",
]
