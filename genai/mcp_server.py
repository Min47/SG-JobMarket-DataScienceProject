"""MCP Server for Job Market Intelligence.

This module exposes job market data as tools to external AI assistants
via the Model Context Protocol (MCP). Compatible with:
- Cursor IDE (stdio transport)
- Windsurf (stdio transport)
- Remote clients (HTTP transport)

MCP Tools (4 total):
1. search_jobs → Vector search with filters
2. get_job_details → Fetch full job info by ID
3. aggregate_stats → Salary statistics by classification
4. find_similar_jobs → Semantic similarity search

Installation:
    pip install mcp>=1.0.0

Usage (stdio mode for Cursor IDE):
    python -m genai.mcp_server
    
Usage (HTTP mode for remote access):
    python -m genai.mcp_server --transport http --port 8001

Configuration (Cursor IDE):
    Add to Cursor settings mcp.json:
    {
      "mcpServers": {
        "sg-job-market": {
          "command": "python",
          "args": ["-m", "genai.mcp_server"],
          "cwd": "/path/to/SG_Job_Market"
        }
      }
    }
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from typing import Dict, List, Any, Optional
from pathlib import Path

# MCP SDK imports
try:
    from mcp.server import Server
    from mcp.server.stdio import stdio_server
    from mcp.types import Tool, TextContent
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False
    print("ERROR: MCP SDK not installed. Run: pip install mcp>=1.0.0", file=sys.stderr)
    sys.exit(1)

# Import our existing tools (reuse, don't duplicate)
from genai.tools import search_jobs, get_job_details, aggregate_stats, find_similar_jobs
from utils.config import Settings
from utils.logging import configure_logging
from google.cloud import bigquery

logger = configure_logging(service_name="mcp-server")


# =============================================================================
# MCP Server Setup
# =============================================================================

# Create MCP server instance
server = Server("sg-job-market")

# Global BigQuery client (initialized once, reused for all queries)
_bigquery_client: Optional[bigquery.Client] = None
_settings: Optional[Settings] = None


def get_bigquery_client() -> bigquery.Client:
    """Get or create singleton BigQuery client."""
    global _bigquery_client, _settings
    if _bigquery_client is None:
        if _settings is None:
            _settings = Settings.load()
        _bigquery_client = bigquery.Client(project=_settings.gcp_project_id)
        logger.info("[MCP] BigQuery client initialized")
    return _bigquery_client


# =============================================================================
# Tool Implementation Functions
# =============================================================================

async def search_jobs_tool(
    query: str,
    location: Optional[str] = None,
    min_salary: Optional[int] = None,
    max_salary: Optional[int] = None,
    work_type: Optional[str] = None,
    classification: Optional[str] = None,
    top_k: int = 10,
) -> str:
    """Search for jobs in Singapore job market using semantic search.
    
    This tool searches across 6,775+ jobs from JobStreet and MyCareersFuture
    using SBERT embeddings for semantic matching. Returns jobs ranked by
    relevance score.
    
    Args:
        query: Natural language search query (e.g., "python developer", "data scientist")
        location: Filter by location (e.g., "Central", "West")
        min_salary: Minimum monthly salary in SGD (e.g., 5000)
        max_salary: Maximum monthly salary in SGD (e.g., 10000)
        work_type: Filter by work type ("Full Time", "Part Time", "Contract", "Permanent")
        classification: Filter by job classification (e.g., "Information & Communication Technology")
        top_k: Number of results to return (1-50, default 10)
        
    Returns:
        JSON string with search results:
        {
          "success": true,
          "count": 5,
          "jobs": [
            {
              "job_id": "123",
              "source": "JobStreet",
              "job_title": "Python Developer",
              "company_name": "Tech Corp",
              "job_location": "Central",
              "job_salary_min_sgd_monthly": 6000,
              "job_salary_max_sgd_monthly": 8000,
              "similarity_score": 0.87,
              ...
            }
          ]
        }
        
    Example:
        search_jobs_tool("machine learning engineer", min_salary=7000, top_k=5)
    """
    try:
        logger.info(f"[MCP] search_jobs_tool called: query='{query}', top_k={top_k}")
        
        # Call existing search_jobs function in thread pool (non-blocking)
        result_json = await asyncio.to_thread(
            search_jobs.invoke,
            {
                "query": query,
                "location": location,
                "min_salary": min_salary,
                "max_salary": max_salary,
                "work_type": work_type,
                "classification": classification,
                "max_results": top_k,
            }
        )
        
        # Parse to verify it's valid JSON
        result = json.loads(result_json)
        logger.info(f"[MCP] search_jobs_tool returned {result.get('count', 0)} jobs")
        
        return result_json
        
    except Exception as e:
        logger.error(f"[MCP] search_jobs_tool error: {e}", exc_info=True)
        return json.dumps({
            "success": False,
            "error": str(e),
            "count": 0,
            "jobs": []
        })


# =============================================================================
# Tool 2: Get Job Details
# =============================================================================

async def get_job_details_tool(
    job_id: str,
    source: str,
) -> str:
    """Get detailed information about a specific job posting.
    
    Fetches complete job information including full description, requirements,
    company details, and salary information.
    
    Args:
        job_id: Unique job identifier (from search results)
        source: Job source - "JobStreet" or "MCF" (case-insensitive)
        
    Returns:
        JSON string with job details:
        {
          "success": true,
          "job": {
            "job_id": "123",
            "source": "JobStreet",
            "job_title": "Senior Python Developer",
            "job_description": "Full job description...",
            "company_name": "Tech Corp",
            "company_industry": "Information Technology",
            "job_location": "Central",
            "job_work_type": "Full Time",
            "job_salary_min_sgd_monthly": 8000,
            "job_salary_max_sgd_monthly": 12000,
            "job_posted_timestamp": "2025-12-15T10:30:00Z",
            "job_url": "https://...",
            ...
          }
        }
        
    Example:
        get_job_details_tool("123456", "JobStreet")
    """
    try:
        logger.info(f"[MCP] get_job_details_tool called: job_id={job_id}, source={source}")
        
        # Call existing get_job_details function in thread pool (non-blocking)
        result_json = await asyncio.to_thread(
            get_job_details.invoke,
            {
                "job_id": job_id,
                "source": source,
            }
        )
        
        result = json.loads(result_json)
        logger.info(f"[MCP] get_job_details_tool: found={result.get('success', False)}")
        
        return result_json
        
    except Exception as e:
        logger.error(f"[MCP] get_job_details_tool error: {e}", exc_info=True)
        return json.dumps({
            "success": False,
            "error": str(e),
            "job": None
        })


# =============================================================================
# Tool 3: Aggregate Statistics
# =============================================================================

async def aggregate_stats_tool(
    group_by: str = "classification",
    location: Optional[str] = None,
    work_type: Optional[str] = None,
    source: Optional[str] = None,
) -> str:
    """Get salary statistics and job counts grouped by classification or location.
    
    Provides aggregate insights like average salaries, job counts, and
    salary ranges across different categories.
    
    Args:
        group_by: Grouping field - "classification", "location", or "work_type"
        location: Filter by location (optional)
        work_type: Filter by work type (optional)
        source: Filter by source - "JobStreet" or "MCF" (optional)
        
    Returns:
        JSON string with statistics:
        {
          "success": true,
          "group_by": "classification",
          "count": 15,
          "stats": [
            {
              "category": "Information & Communication Technology",
              "job_count": 2543,
              "avg_salary_min": 5200,
              "avg_salary_max": 7800,
              "median_salary_min": 5000,
              "median_salary_max": 7500,
              "min_salary": 3000,
              "max_salary": 15000
            },
            ...
          ]
        }
        
    Example:
        aggregate_stats_tool("classification", location="Central")
    """
    try:
        logger.info(f"[MCP] aggregate_stats_tool called: group_by={group_by}")
        
        # Call existing aggregate_stats function in thread pool (non-blocking)
        result_json = await asyncio.to_thread(
            aggregate_stats.invoke,
            {
                "group_by": group_by,
                "location": location,
                "work_type": work_type,
                "source": source,
            }
        )
        
        result = json.loads(result_json)
        logger.info(f"[MCP] aggregate_stats_tool returned {result.get('count', 0)} groups")
        
        return result_json
        
    except Exception as e:
        logger.error(f"[MCP] aggregate_stats_tool error: {e}", exc_info=True)
        return json.dumps({
            "success": False,
            "error": str(e),
            "group_by": group_by,
            "count": 0,
            "stats": []
        })


# =============================================================================
# Tool 4: Find Similar Jobs
# =============================================================================

async def find_similar_jobs_tool(
    job_id: str,
    source: str,
    top_k: int = 5,
) -> str:
    """Find jobs similar to a given job using semantic embeddings.
    
    Uses SBERT embeddings to find jobs with similar titles and descriptions.
    Useful for job recommendations and exploring related opportunities.
    
    Args:
        job_id: Reference job ID to find similar jobs for
        source: Source of reference job - "JobStreet" or "MCF"
        top_k: Number of similar jobs to return (1-20, default 5)
        
    Returns:
        JSON string with similar jobs:
        {
          "success": true,
          "reference_job": {
            "job_id": "123",
            "job_title": "Data Scientist",
            "company_name": "Analytics Corp"
          },
          "count": 5,
          "similar_jobs": [
            {
              "job_id": "456",
              "source": "JobStreet",
              "job_title": "Machine Learning Engineer",
              "company_name": "AI Startup",
              "similarity_score": 0.89,
              ...
            },
            ...
          ]
        }
        
    Example:
        find_similar_jobs_tool("123456", "JobStreet", top_k=5)
    """
    try:
        logger.info(f"[MCP] find_similar_jobs_tool called: job_id={job_id}, top_k={top_k}")
        
        # Call existing find_similar_jobs function in thread pool (non-blocking)
        result_json = await asyncio.to_thread(
            find_similar_jobs.invoke,
            {
                "job_id": job_id,
                "source": source,
                "top_k": top_k,
            }
        )
        
        result = json.loads(result_json)
        logger.info(f"[MCP] find_similar_jobs_tool returned {result.get('count', 0)} similar jobs")
        
        return result_json
        
    except Exception as e:
        logger.error(f"[MCP] find_similar_jobs_tool error: {e}", exc_info=True)
        return json.dumps({
            "success": False,
            "error": str(e),
            "reference_job": None,
            "count": 0,
            "similar_jobs": []
        })


# =============================================================================
# MCP Handler Registration
# =============================================================================

@server.list_tools()
async def handle_list_tools() -> list[Tool]:
    """List all available tools for MCP clients."""
    return [
        Tool(
            name="search_jobs_tool",
            description="Search for jobs using semantic search with optional filters",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query for jobs (e.g., 'Python developer', 'Data scientist with ML')"
                    },
                    "location": {
                        "type": "string",
                        "description": "Filter by job location (optional)"
                    },
                    "min_salary": {
                        "type": "integer",
                        "description": "Minimum monthly salary in SGD (optional)"
                    },
                    "max_salary": {
                        "type": "integer",
                        "description": "Maximum monthly salary in SGD (optional)"
                    },
                    "work_type": {
                        "type": "string",
                        "description": "Filter by work type: Full Time, Part Time, Contract, Temporary (optional)"
                    },
                    "classification": {
                        "type": "string",
                        "description": "Filter by job classification/category (optional)"
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Maximum number of results to return (default: 10, max: 50)"
                    }
                },
                "required": ["query"]
            }
        ),
        Tool(
            name="get_job_details_tool",
            description="Get detailed information about a specific job posting by ID",
            inputSchema={
                "type": "object",
                "properties": {
                    "job_id": {
                        "type": "string",
                        "description": "The unique job ID"
                    },
                    "source": {
                        "type": "string",
                        "description": "The job source: 'JobStreet' or 'MCF'",
                        "enum": ["JobStreet", "MCF"]
                    }
                },
                "required": ["job_id", "source"]
            }
        ),
        Tool(
            name="aggregate_stats_tool",
            description="Get salary statistics grouped by classification, location, or work type",
            inputSchema={
                "type": "object",
                "properties": {
                    "group_by": {
                        "type": "string",
                        "description": "Group statistics by: classification, location, or work_type",
                        "enum": ["classification", "location", "work_type"]
                    },
                    "location": {
                        "type": "string",
                        "description": "Filter by location before aggregating (optional)"
                    },
                    "classification": {
                        "type": "string",
                        "description": "Filter by classification before aggregating (optional)"
                    },
                    "work_type": {
                        "type": "string",
                        "description": "Filter by work type before aggregating (optional)"
                    }
                },
                "required": ["group_by"]
            }
        ),
        Tool(
            name="find_similar_jobs_tool",
            description="Find semantically similar jobs to a given job posting",
            inputSchema={
                "type": "object",
                "properties": {
                    "job_id": {
                        "type": "string",
                        "description": "The reference job ID to find similar jobs for"
                    },
                    "source": {
                        "type": "string",
                        "description": "The job source: 'JobStreet' or 'MCF'",
                        "enum": ["JobStreet", "MCF"]
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Number of similar jobs to return (default: 5, max: 20)"
                    }
                },
                "required": ["job_id", "source"]
            }
        )
    ]


@server.call_tool()
async def handle_call_tool(name: str, arguments: dict) -> list[TextContent]:
    """Handle tool invocation requests from MCP clients."""
    logger.info(f"Tool called: {name} with args: {arguments}")
    
    try:
        # Route to appropriate tool function
        if name == "search_jobs_tool":
            result = await search_jobs_tool(
                query=arguments["query"],
                location=arguments.get("location"),
                min_salary=arguments.get("min_salary"),
                max_salary=arguments.get("max_salary"),
                work_type=arguments.get("work_type"),
                classification=arguments.get("classification"),
                top_k=arguments.get("top_k", 10),
            )
        elif name == "get_job_details_tool":
            result = await get_job_details_tool(
                job_id=arguments["job_id"],
                source=arguments["source"],
            )
        elif name == "aggregate_stats_tool":
            result = await aggregate_stats_tool(
                group_by=arguments["group_by"],
                location=arguments.get("location"),
                classification=arguments.get("classification"),
                work_type=arguments.get("work_type"),
            )
        elif name == "find_similar_jobs_tool":
            result = await find_similar_jobs_tool(
                job_id=arguments["job_id"],
                source=arguments["source"],
                top_k=arguments.get("top_k", 5),
            )
        else:
            error_result = json.dumps({
                "success": False,
                "error": f"Unknown tool: {name}"
            })
            return [TextContent(type="text", text=error_result)]
        
        # Truncate large responses (GitHub Copilot may have size limits)
        MAX_RESPONSE_SIZE = 50000  # 50KB limit
        if len(result) > MAX_RESPONSE_SIZE:
            logger.warning(f"[MCP] Response too large ({len(result)} bytes), truncating...")
            result_obj = json.loads(result)
            
            # Truncate job descriptions if present
            if "job" in result_obj and result_obj["job"]:
                if "job_description" in result_obj["job"]:
                    desc = result_obj["job"]["job_description"]
                    if len(desc) > 2000:
                        result_obj["job"]["job_description"] = desc[:2000] + "... [truncated]"
            
            if "jobs" in result_obj:
                for job in result_obj["jobs"]:
                    if "job_description" in job and len(job["job_description"]) > 500:
                        job["job_description"] = job["job_description"][:500] + "... [truncated]"
            
            result = json.dumps(result_obj)
            logger.info(f"[MCP] Truncated response to {len(result)} bytes")
        
        # Return result as TextContent
        logger.info(f"[MCP] Returning response ({len(result)} bytes)")
        return [TextContent(type="text", text=result)]
        
    except Exception as e:
        logger.error(f"Error executing tool {name}: {e}", exc_info=True)
        error_result = json.dumps({
            "success": False,
            "error": str(e)
        })
        return [TextContent(type="text", text=error_result)]


# =============================================================================
# Server Lifecycle
# =============================================================================

async def main():
    """Run MCP server with stdio transport (default for Claude Desktop)."""
    logger.info("=" * 80)
    logger.info("SG Job Market - MCP Server Starting")
    logger.info("=" * 80)
    logger.info("Server Name: sg-job-market")
    logger.info("Registered Tools: 4")
    logger.info("  1. search_jobs_tool")
    logger.info("  2. get_job_details_tool")
    logger.info("  3. aggregate_stats_tool")
    logger.info("  4. find_similar_jobs_tool")
    logger.info("Transport: stdio (for Claude Desktop/Cursor)")
    logger.info("=" * 80)
    
    # Verify settings can be loaded
    try:
        settings = Settings.load()
        logger.info(f"✓ Settings loaded: project={settings.gcp_project_id}, dataset={settings.bigquery_dataset_id}")
    except Exception as e:
        logger.error(f"✗ Failed to load settings: {e}")
        logger.error("Make sure .env file exists with required variables")
        sys.exit(1)
    
    # Run server with stdio transport
    async with stdio_server() as (read_stream, write_stream):
        logger.info("✓ MCP Server ready - listening for tool calls...")
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options()
        )


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("\n✓ MCP Server stopped by user")
    except Exception as e:
        logger.error(f"✗ MCP Server error: {e}", exc_info=True)
        sys.exit(1)
