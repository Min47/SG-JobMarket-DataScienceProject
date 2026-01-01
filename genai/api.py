"""FastAPI Service for Job Market Intelligence.

This module exposes the GenAI agent and tools as REST API endpoints:
- POST /v1/chat → Conversational agent with LangGraph orchestration
- POST /v1/search → Direct vector search (bypasses agent)
- GET /v1/jobs/{job_id} → Fetch job details
- GET /v1/jobs/{job_id}/similar → Find similar jobs
- POST /v1/stats → Aggregate salary statistics
- GET /health → Health check for monitoring
- GET /docs → OpenAPI documentation (Swagger UI)

Middleware:
- Rate limiting (100 req/min per IP)
- CORS for dashboard access
- Request logging with structured JSON
- Automatic request validation (Pydantic)

Deployment:
- Docker containerization (Dockerfile.api)
- Cloud Run deployment (asia-southeast1)
- Auto-scaling (0-10 instances)
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional

from fastapi import FastAPI, HTTPException, Request, Query, Path
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from utils.config import Settings
from utils.logging import configure_logging

# Import agent and tools
from genai.agent import JobMarketAgent
from genai.rag import retrieve_jobs, generate_answer
from genai.tools import search_jobs, get_job_details, aggregate_stats, find_similar_jobs

# Import guardrails
from genai.guardrails import InputGuardrails, OutputGuardrails

logger = configure_logging(service_name="genai-api")

# =============================================================================
# Request/Response Models
# =============================================================================

class ChatRequest(BaseModel):
    """Request schema for conversational agent endpoint."""
    
    message: str = Field(
        ...,
        min_length=3,
        max_length=1000,
        description="User's natural language query",
        examples=["Find me accountants jobs with salary > $3000"]
    )
    conversation_id: Optional[str] = Field(
        default=None,
        description="Optional conversation ID for multi-turn context",
        examples=["550e8400-e29b-41d4-a716-446655440000"]
    )
    filters: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional filters: location, min_salary, max_salary, work_type",
        examples=[{"location": "Central", "min_salary": 3000}]
    )


class ChatResponse(BaseModel):
    """Response schema for conversational agent endpoint."""
    
    answer: str = Field(description="Natural language response from agent")
    sources: List[Dict[str, str]] = Field(
        description="Job sources cited in answer",
        examples=[[{"job_id": "123", "job_title": "Software Engineer", "company": "TechCo", "url": "..."}]]
    )
    conversation_id: str = Field(description="Conversation ID for follow-up queries")
    metadata: Dict[str, Any] = Field(
        description="Processing metadata: retrieved_count, relevance_score, timing, etc."
    )


class SearchRequest(BaseModel):
    """Request schema for direct vector search endpoint."""
    
    query: str = Field(
        ...,
        min_length=3,
        max_length=500,
        description="Search query for jobs",
        examples=["python backend developer"]
    )
    top_k: int = Field(
        default=10,
        ge=1,
        le=50,
        description="Number of results to return"
    )
    filters: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional filters: location, min_salary, max_salary, work_type"
    )


class SearchResponse(BaseModel):
    """Response schema for direct vector search endpoint."""
    
    jobs: List[Dict[str, Any]] = Field(description="List of matching jobs")
    count: int = Field(description="Number of jobs returned")
    query: str = Field(description="Original query")
    processing_time_ms: int = Field(description="Processing time in milliseconds")


class StatsRequest(BaseModel):
    """Request schema for aggregate statistics endpoint."""
    
    group_by: str = Field(
        ...,
        pattern="^(classification|work_type|location)$",
        description="Field to group by: classification, work_type, or location",
        examples=["classification"]
    )
    filters: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional filters to apply before aggregation"
    )
    limit: int = Field(
        default=10,
        ge=1,
        le=50,
        description="Maximum number of groups to return"
    )


class StatsResponse(BaseModel):
    """Response schema for aggregate statistics endpoint."""
    
    stats: List[Dict[str, Any]] = Field(description="Statistics per group")
    summary: Dict[str, Any] = Field(description="Overall summary statistics")
    group_by: str = Field(description="Field used for grouping")


class HealthResponse(BaseModel):
    """Response schema for health check endpoint."""
    
    status: str = Field(description="Overall health status: healthy, degraded, unhealthy")
    version: str = Field(description="API version")
    timestamp: str = Field(description="Current server timestamp (ISO 8601)")
    services: Dict[str, str] = Field(description="Status of dependent services")


class ErrorResponse(BaseModel):
    """Standard error response schema."""
    
    error: str = Field(description="Error message")
    detail: Optional[str] = Field(default=None, description="Detailed error information")
    request_id: Optional[str] = Field(default=None, description="Request tracking ID")


# =============================================================================
# FastAPI App Initialization
# =============================================================================

# Rate limiter configuration
limiter = Limiter(key_func=get_remote_address)

# Create FastAPI app
app = FastAPI(
    title="SG Job Market Intelligence API",
    description="Production-grade REST API for job search, salary analytics, and AI-powered recommendations",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

# Add rate limiter to app
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# CORS configuration (allow dashboard access)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",  # Local React/Next.js
        "http://localhost:8501",  # Local Streamlit
        "https://dashboard.sg-job-market.com",  # Production dashboard (if deployed)
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load settings
settings = Settings.load()

# Initialize agent (lazy loading)
_agent: Optional[JobMarketAgent] = None


def get_agent() -> JobMarketAgent:
    """Get or create singleton JobMarketAgent instance."""
    global _agent
    if _agent is None:
        logger.info("[API] Initializing JobMarketAgent...")
        _agent = JobMarketAgent(settings=settings)
    return _agent


# Initialize guardrails
input_guards = InputGuardrails()
output_guards = OutputGuardrails()
logger.info("[API] Guardrails initialized (PII, injection, hallucination detection)")


# =============================================================================
# Middleware: Request Logging
# =============================================================================

@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all incoming requests with timing and metadata."""
    request_id = str(uuid.uuid4())
    start_time = time.time()
    
    # Attach request_id to request state
    request.state.request_id = request_id
    
    logger.info(
        f"[API Request] {request.method} {request.url.path}",
        extra={
            "request_id": request_id,
            "method": request.method,
            "path": request.url.path,
            "client_ip": get_remote_address(request),
        }
    )
    
    # Process request
    try:
        response = await call_next(request)
        processing_time_ms = int((time.time() - start_time) * 1000)
        
        # Add custom headers
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Processing-Time-MS"] = str(processing_time_ms)
        
        logger.info(
            f"[API Response] {request.method} {request.url.path} → {response.status_code}",
            extra={
                "request_id": request_id,
                "status_code": response.status_code,
                "processing_time_ms": processing_time_ms,
            }
        )
        
        return response
        
    except Exception as e:
        processing_time_ms = int((time.time() - start_time) * 1000)
        logger.error(
            f"[API Error] {request.method} {request.url.path} → {str(e)}",
            extra={
                "request_id": request_id,
                "error": str(e),
                "processing_time_ms": processing_time_ms,
            },
            exc_info=True
        )
        raise


# =============================================================================
# Endpoint 1: Conversational Agent (POST /v1/chat)
# =============================================================================

@app.post(
    "/v1/chat",
    response_model=ChatResponse,
    tags=["Agent"],
    summary="Conversational job search with AI agent",
    description=(
        "Full agentic workflow with multi-step reasoning, query rewriting, "
        "and LangGraph orchestration. Use for natural language queries."
    ),
)
@limiter.limit("10/minute")  # Lower limit for compute-intensive agent
async def chat_endpoint(request: Request, chat_request: ChatRequest) -> ChatResponse:
    """Conversational interface to the job search agent.
    
    This endpoint:
    1. Processes user's natural language query
    2. Orchestrates retrieval → grading → generation pipeline
    3. Handles query rewriting if results are poor
    4. Returns natural language answer with job citations
    
    Example:
        POST /v1/chat
        {
            "message": "Find me data scientist jobs in fintech with salary > $10k",
            "filters": {"location": "Central"}
        }
    """
    logger.info(f"[Chat Endpoint] Query: {chat_request.message[:100]}...")
    
    try:
        # === GUARDRAILS: Validate input ===
        validation_result = input_guards.validate(chat_request.message)
        if not validation_result.passed:
            logger.warning(
                f"[Chat Endpoint] Input blocked by guardrails: {validation_result.reason}",
                extra={"violations": validation_result.violations}
            )
            raise HTTPException(
                status_code=400,
                detail={
                    "error": validation_result.reason,
                    "violations": validation_result.violations,
                    "severity": validation_result.severity.value,
                }
            )
        
        # Get or create conversation ID
        conversation_id = chat_request.conversation_id or str(uuid.uuid4())
        
        # Get agent instance
        agent = get_agent()
        
        # Run agent with query
        result = agent.run(
            query=chat_request.message,
            filters=chat_request.filters or {}
        )
        
        # Extract response components
        # Note: agent.run() returns final_state["final_answer"] directly,
        # which already has "answer" and "sources" at top level
        answer = result.get("answer", "No answer generated")
        sources = result.get("sources", [])
        
        # === GUARDRAILS: Validate output ===
        # Get context jobs for hallucination check
        context_jobs = result.get("graded_jobs", [])
        output_validation = output_guards.validate(
            response={"answer": answer, "sources": sources},
            context_jobs=context_jobs
        )
        
        if not output_validation.passed:
            logger.warning(
                f"[Chat Endpoint] Output validation warning: {output_validation.reason}",
                extra={"violations": output_validation.violations}
            )
            # For output issues, log warning but don't block (WARNING severity)
            # unless it's BLOCKED severity
            if output_validation.severity.value == "blocked":
                raise HTTPException(
                    status_code=500,
                    detail="Response failed safety checks. Please try a different query."
                )
        
        # Build metadata (extract from result["metadata"] which contains execution stats)
        metadata = {
            "conversation_id": conversation_id,
            "retrieved_count": result.get("metadata", {}).get("retrieved_count", 0),
            "graded_count": result.get("metadata", {}).get("graded_count", 0),
            "average_relevance_score": result.get("metadata", {}).get("average_relevance_score", 0.0),
            "rewrite_count": result.get("metadata", {}).get("rewrite_count", 0),
            "original_query": result.get("metadata", {}).get("original_query", chat_request.message),
            "final_query": result.get("metadata", {}).get("final_query", chat_request.message),
        }
        
        return ChatResponse(
            answer=answer,
            sources=sources,
            conversation_id=conversation_id,
            metadata=metadata
        )
        
    except HTTPException:
        # Let HTTPExceptions propagate (guardrail blocks, etc.)
        raise
    except Exception as e:
        logger.error(f"[Chat Endpoint] Error: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Agent execution failed: {str(e)}"
        )


# =============================================================================
# Endpoint 2: Direct Vector Search (POST /v1/search)
# =============================================================================

@app.post(
    "/v1/search",
    response_model=SearchResponse,
    tags=["Search"],
    summary="Direct vector similarity search",
    description="Fast vector search without agent reasoning. Use for quick job listings.",
)
@limiter.limit("50/minute")  # Higher limit for fast searches
async def search_endpoint(request: Request, search_request: SearchRequest) -> SearchResponse:
    """Direct vector search bypassing agent orchestration.
    
    This endpoint:
    1. Generates embedding for query
    2. Performs BigQuery VECTOR_SEARCH
    3. Applies optional filters
    4. Returns raw job results with relevance scores
    
    Example:
        POST /v1/search
        {
            "query": "python backend developer",
            "top_k": 10,
            "filters": {"min_salary": 5000}
        }
    """
    logger.info(f"[Search Endpoint] Query: {search_request.query[:100]}...")
    start_time = time.time()
    
    try:
        # === GUARDRAILS: Validate input ===
        validation_result = input_guards.validate(search_request.query)
        if not validation_result.passed:
            logger.warning(
                f"[Search Endpoint] Input blocked: {validation_result.reason}",
                extra={"violations": validation_result.violations}
            )
            raise HTTPException(
                status_code=400,
                detail={
                    "error": validation_result.reason,
                    "violations": validation_result.violations,
                }
            )
        
        # Call retrieve_jobs directly (bypasses agent)
        jobs = retrieve_jobs(
            query=search_request.query,
            top_k=search_request.top_k,
            filters=search_request.filters or {},
            settings=settings
        )
        
        processing_time_ms = int((time.time() - start_time) * 1000)
        
        return SearchResponse(
            jobs=jobs,
            count=len(jobs),
            query=search_request.query,
            processing_time_ms=processing_time_ms
        )
        
    except HTTPException:
        # Let HTTPExceptions propagate (guardrail blocks, etc.)
        raise
    except Exception as e:
        logger.error(f"[Search Endpoint] Error: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Search failed: {str(e)}"
        )


# =============================================================================
# Endpoint 3: Get Job Details (GET /v1/jobs/{job_id})
# =============================================================================

@app.get(
    "/v1/jobs/{job_id}",
    response_model=Dict[str, Any],
    tags=["Jobs"],
    summary="Get complete job details by ID",
    description="Fetch full job information including description, salary, and company info.",
)
@limiter.limit("100/minute")
async def get_job_endpoint(
    request: Request,
    job_id: str = Path(..., description="Unique job identifier"),
    source: str = Query(..., pattern="^(jobstreet|JobStreet|mcf|MCF)$", description="Job source platform")
) -> Dict[str, Any]:
    """Fetch complete details for a specific job.
    
    Example:
        GET /v1/jobs/89329928?source=JobStreet
    """
    logger.info(f"[Get Job Endpoint] job_id={job_id}, source={source}")
    
    try:
        # Call tool directly
        result_str = get_job_details.invoke({"job_id": job_id, "source": source})
        result = json.loads(result_str)
        
        if not result.get("success"):
            raise HTTPException(
                status_code=404,
                detail=result.get("error", f"Job {job_id} not found")
            )
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[Get Job Endpoint] Error: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch job: {str(e)}"
        )


# =============================================================================
# Endpoint 4: Find Similar Jobs (GET /v1/jobs/{job_id}/similar)
# =============================================================================

@app.get(
    "/v1/jobs/{job_id}/similar",
    response_model=Dict[str, Any],
    tags=["Jobs"],
    summary="Find semantically similar jobs",
    description="Use vector similarity to recommend jobs similar to a given job.",
)
@limiter.limit("50/minute")
async def similar_jobs_endpoint(
    request: Request,
    job_id: str = Path(..., description="Reference job ID"),
    source: str = Query(..., pattern="^(jobstreet|JobStreet|mcf|MCF)$", description="Job source"),
    top_k: int = Query(5, ge=1, le=20, description="Number of similar jobs to return"),
    min_similarity: float = Query(0.7, ge=0.0, le=1.0, description="Minimum similarity threshold")
) -> Dict[str, Any]:
    """Find jobs similar to a reference job using vector embeddings.
    
    Example:
        GET /v1/jobs/89329928/similar?source=JobStreet&top_k=5&min_similarity=0.75
    """
    logger.info(f"[Similar Jobs Endpoint] job_id={job_id}, source={source}, top_k={top_k}")
    
    try:
        # Call tool directly
        result_str = find_similar_jobs.invoke({
            "job_id": job_id,
            "source": source,
            "top_k": top_k,
            "min_similarity": min_similarity
        })
        result = json.loads(result_str)
        
        if not result.get("success"):
            raise HTTPException(
                status_code=404,
                detail=result.get("error", f"Reference job {job_id} not found")
            )
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[Similar Jobs Endpoint] Error: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Similarity search failed: {str(e)}"
        )


# =============================================================================
# Endpoint 5: Aggregate Statistics (POST /v1/stats)
# =============================================================================

@app.post(
    "/v1/stats",
    response_model=StatsResponse,
    tags=["Analytics"],
    summary="Aggregate salary statistics",
    description="Compute salary statistics grouped by classification, work type, or location.",
)
@limiter.limit("30/minute")
async def stats_endpoint(request: Request, stats_request: StatsRequest) -> StatsResponse:
    """Aggregate salary statistics by category.
    
    Example:
        POST /v1/stats
        {
            "group_by": "classification",
            "filters": {"classification": "Information & Communication Technology"},
            "limit": 10
        }
    """
    logger.info(f"[Stats Endpoint] group_by={stats_request.group_by}")
    
    try:
        # Call tool directly
        tool_input = {
            "group_by": stats_request.group_by,
            "limit": stats_request.limit
        }
        # Add filters if provided
        if stats_request.filters:
            tool_input.update(stats_request.filters)
        
        result_str = aggregate_stats.invoke(tool_input)
        result = json.loads(result_str)
        
        if not result.get("success"):
            raise HTTPException(
                status_code=400,
                detail=result.get("error", "Statistics computation failed")
            )
        
        return StatsResponse(
            stats=result.get("stats", []),
            summary=result.get("summary", {}),
            group_by=stats_request.group_by
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[Stats Endpoint] Error: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Statistics computation failed: {str(e)}"
        )


# =============================================================================
# Endpoint 6: Health Check (GET /health)
# =============================================================================

@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["System"],
    summary="Health check for monitoring",
    description="Returns service health status and dependent service availability.",
)
async def health_endpoint(request: Request) -> HealthResponse:
    """Health check endpoint for Kubernetes/Cloud Run probes.
    
    Checks:
    - BigQuery connectivity
    - Vertex AI availability
    - Embedding model loaded
    
    Example:
        GET /health
    """
    services = {}
    overall_status = "healthy"
    
    # Check BigQuery
    try:
        from google.cloud import bigquery
        client = bigquery.Client(project=settings.gcp_project_id)
        # Simple query to check connectivity
        query = f"SELECT COUNT(*) FROM `{settings.gcp_project_id}.{settings.bigquery_dataset_id}.cleaned_jobs` LIMIT 1"
        client.query(query).result()
        services["bigquery"] = "ok"
    except Exception as e:
        services["bigquery"] = f"error: {str(e)[:50]}"
        overall_status = "degraded"
    
    # Check Vertex AI (just check if initialized)
    try:
        from google.cloud import aiplatform
        aiplatform.init(project=settings.gcp_project_id, location=settings.gcp_region)
        services["vertex_ai"] = "ok"
    except Exception as e:
        services["vertex_ai"] = f"error: {str(e)[:50]}"
        overall_status = "degraded"
    
    # Check embedding model
    try:
        from nlp.embeddings import EmbeddingGenerator
        generator = EmbeddingGenerator()
        # Test embedding generation
        test_embedding = generator.embed_texts(["test"], show_progress=False)
        if test_embedding.size > 0:
            services["embeddings"] = "ok"
        else:
            services["embeddings"] = "error: empty embedding"
            overall_status = "degraded"
    except Exception as e:
        services["embeddings"] = f"error: {str(e)[:50]}"
        overall_status = "degraded"
    
    return HealthResponse(
        status=overall_status,
        version="1.0.0",
        timestamp=datetime.now(timezone.utc).isoformat(),
        services=services
    )


# =============================================================================
# Root Endpoint (GET /)
# =============================================================================

@app.get("/", tags=["System"])
async def root():
    """API root with navigation links."""
    return {
        "service": "SG Job Market Intelligence API",
        "version": "1.0.0",
        "documentation": "/docs",
        "redoc": "/redoc",
        "health": "/health",
        "endpoints": {
            "chat": "POST /v1/chat",
            "search": "POST /v1/search",
            "job_details": "GET /v1/jobs/{job_id}",
            "similar_jobs": "GET /v1/jobs/{job_id}/similar",
            "statistics": "POST /v1/stats",
        }
    }


# =============================================================================
# Exception Handlers
# =============================================================================

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions with consistent error format."""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code,
            "request_id": getattr(request.state, "request_id", None),
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle unexpected exceptions with logging."""
    logger.error(f"[API] Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": str(exc),
            "request_id": getattr(request.state, "request_id", None),
        }
    )


# =============================================================================
# Main (for local development)
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    
    logger.info("[API] Starting FastAPI server in development mode...")
    logger.info("[API] OpenAPI docs: http://localhost:8000/docs")
    logger.info("[API] ReDoc: http://localhost:8000/redoc")
    
    uvicorn.run(
        "genai.api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,  # Auto-reload on code changes
        log_level="info"
    )
