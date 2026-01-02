"""Observability & Monitoring for GenAI API.

This module provides distributed tracing, metrics, and structured logging:
- OpenTelemetry tracing for request flows (API → Agent → RAG → LLM)
- Prometheus metrics for monitoring (latency, errors, cost)
- Structured JSON logging with trace correlation

Architecture:
    Request → [TRACE] → Agent → [TRACE] → RAG → [TRACE] → LLM
                ↓              ↓             ↓            ↓
            [METRICS]      [METRICS]     [METRICS]   [METRICS]
                ↓              ↓             ↓            ↓
            [LOGS]         [LOGS]        [LOGS]      [LOGS]

Usage:
    # Initialize observability (call once at startup)
    >>> init_observability(service_name="genai-api")
    
    # Add tracing to functions
    >>> @trace_function("retrieve_jobs")
    >>> def retrieve_jobs(query: str) -> List[Dict]:
    ...     with add_span_attributes({"query_length": len(query)}):
    ...         results = vector_search(query)
    ...         RETRIEVAL_LATENCY.observe(0.5)
    ...         return results
    
    # Metrics in endpoints
    >>> REQUEST_COUNT.labels(endpoint="/chat", status="200").inc()
    >>> REQUEST_LATENCY.labels(endpoint="/chat").observe(1.2)
"""

from __future__ import annotations

import logging
import time
from functools import wraps
from typing import Dict, Any, Optional, Callable
from contextlib import contextmanager

# OpenTelemetry imports
from opentelemetry import trace, metrics
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.sdk.resources import Resource
from opentelemetry.exporter.cloud_trace import CloudTraceSpanExporter
from opentelemetry.exporter.cloud_monitoring import CloudMonitoringMetricsExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

# Prometheus imports
from prometheus_client import Counter, Histogram, Gauge, Info

logger = logging.getLogger(__name__)


# =============================================================================
# Global Configuration
# =============================================================================

# OpenTelemetry tracer and meter (initialized in init_observability)
_TRACER: Optional[trace.Tracer] = None
_METER: Optional[metrics.Meter] = None


# =============================================================================
# Prometheus Metrics (always enabled)
# =============================================================================

# Request metrics
REQUEST_COUNT = Counter(
    "genai_requests_total",
    "Total number of requests",
    ["endpoint", "status_code", "method"],
)

REQUEST_LATENCY = Histogram(
    "genai_request_duration_seconds",
    "Request latency in seconds",
    ["endpoint", "method"],
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0],
)

# LLM metrics
LLM_CALL_COUNT = Counter(
    "genai_llm_calls_total",
    "Total number of LLM calls",
    ["provider", "model", "operation"],  # e.g., ("vertexai", "gemini-2.5-flash", "generate")
)

LLM_TOKEN_COUNT = Counter(
    "genai_llm_tokens_total",
    "Total tokens used",
    ["provider", "model", "token_type"],  # token_type: "input", "output"
)

LLM_COST = Counter(
    "genai_llm_cost_usd_total",
    "Total cost in USD",
    ["provider", "model"],
)

LLM_LATENCY = Histogram(
    "genai_llm_duration_seconds",
    "LLM call latency in seconds",
    ["provider", "model", "operation"],
    buckets=[0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 30.0, 60.0],
)

# RAG pipeline metrics
RETRIEVAL_LATENCY = Histogram(
    "genai_retrieval_duration_seconds",
    "Vector search latency",
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0],
)

RETRIEVAL_COUNT = Counter(
    "genai_retrieval_total",
    "Total retrievals",
    ["status"],  # status: "success", "failure", "empty"
)

GRADING_LATENCY = Histogram(
    "genai_grading_duration_seconds",
    "Document grading latency",
    buckets=[0.5, 1.0, 2.0, 5.0, 10.0],
)

AVERAGE_RELEVANCE_SCORE = Gauge(
    "genai_average_relevance_score",
    "Average relevance score from grading",
)

REWRITE_COUNT = Counter(
    "genai_query_rewrites_total",
    "Total query rewrites",
    ["reason"],  # reason: "low_score", "few_results"
)

# Agent metrics
AGENT_EXECUTION_LATENCY = Histogram(
    "genai_agent_duration_seconds",
    "Agent execution time",
    buckets=[1.0, 5.0, 10.0, 20.0, 30.0, 60.0, 120.0],
)

AGENT_STEP_COUNT = Counter(
    "genai_agent_steps_total",
    "Total agent steps executed",
    ["step_name"],  # step_name: "retrieve", "grade", "generate", "rewrite"
)

# Guardrails metrics
GUARDRAIL_BLOCKS = Counter(
    "genai_guardrail_blocks_total",
    "Total requests blocked by guardrails",
    ["guard_type"],  # guard_type: "pii", "injection", "hallucination"
)

# System metrics
ACTIVE_REQUESTS = Gauge(
    "genai_active_requests",
    "Number of requests currently being processed",
)

# API info (version, build, etc.)
API_INFO = Info(
    "genai_api_info",
    "API version and build information",
)


# =============================================================================
# OpenTelemetry Initialization
# =============================================================================

def init_observability(
    service_name: str = "genai-api",
    gcp_project_id: Optional[str] = None,
    enable_cloud_trace: bool = True,
    enable_cloud_monitoring: bool = True,
) -> None:
    """Initialize OpenTelemetry tracing and metrics.
    
    Sets up:
    - Cloud Trace for distributed tracing
    - Cloud Monitoring for metrics export
    - Prometheus metrics (always enabled)
    - Structured logging with trace IDs
    
    Args:
        service_name: Name of the service (for trace/metric labels)
        gcp_project_id: GCP project ID (auto-detected if None)
        enable_cloud_trace: Export traces to Cloud Trace
        enable_cloud_monitoring: Export metrics to Cloud Monitoring
    """
    global _TRACER, _METER
    
    # Resource attributes (service name, version, etc.)
    resource = Resource.create({
        "service.name": service_name,
        "service.version": "1.0.0",
    })
    
    # =========================================================================
    # Tracing Setup
    # =========================================================================
    
    if enable_cloud_trace:
        try:
            # Cloud Trace exporter
            trace_exporter = CloudTraceSpanExporter(project_id=gcp_project_id)
            span_processor = BatchSpanProcessor(trace_exporter)
            
            # Tracer provider
            tracer_provider = TracerProvider(resource=resource)
            tracer_provider.add_span_processor(span_processor)
            trace.set_tracer_provider(tracer_provider)
            
            _TRACER = trace.get_tracer(__name__)
            logger.info(f"[Observability] Cloud Trace enabled for {service_name}")
        except Exception as e:
            logger.warning(f"[Observability] Cloud Trace setup failed: {e}. Tracing disabled.")
            _TRACER = trace.get_tracer(__name__)  # Use no-op tracer
    else:
        _TRACER = trace.get_tracer(__name__)
        logger.info("[Observability] Cloud Trace disabled (local mode)")
    
    # =========================================================================
    # Metrics Setup
    # =========================================================================
    
    if enable_cloud_monitoring:
        try:
            # Cloud Monitoring exporter
            metric_exporter = CloudMonitoringMetricsExporter(project_id=gcp_project_id)
            metric_reader = PeriodicExportingMetricReader(
                metric_exporter,
                export_interval_millis=60000,  # Export every 60 seconds
            )
            
            # Meter provider
            meter_provider = MeterProvider(
                resource=resource,
                metric_readers=[metric_reader],
            )
            metrics.set_meter_provider(meter_provider)
            
            _METER = metrics.get_meter(__name__)
            logger.info(f"[Observability] Cloud Monitoring enabled for {service_name}")
        except Exception as e:
            logger.warning(f"[Observability] Cloud Monitoring setup failed: {e}. Using Prometheus only.")
            _METER = metrics.get_meter(__name__)  # Use no-op meter
    else:
        _METER = metrics.get_meter(__name__)
        logger.info("[Observability] Cloud Monitoring disabled (Prometheus only)")
    
    # =========================================================================
    # Set API Info
    # =========================================================================
    
    API_INFO.info({
        "version": "1.0.0",
        "service": service_name,
        "build_date": "2026-01-02",
    })
    
    logger.info("[Observability] Initialization complete")


# =============================================================================
# Tracing Decorators & Context Managers
# =============================================================================

def trace_function(span_name: str, attributes: Optional[Dict[str, Any]] = None):
    """Decorator to trace a function with OpenTelemetry.
    
    Creates a span for the function execution and automatically:
    - Records function name, duration, and outcome
    - Adds custom attributes (if provided)
    - Links errors to the span
    
    Args:
        span_name: Name of the span (e.g., "retrieve_jobs")
        attributes: Additional span attributes
        
    Example:
        >>> @trace_function("vector_search", {"db": "bigquery"})
        >>> def search(query: str) -> List[Dict]:
        ...     return results
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            if _TRACER is None:
                return func(*args, **kwargs)
            
            with _TRACER.start_as_current_span(span_name) as span:
                # Add custom attributes
                if attributes:
                    for key, value in attributes.items():
                        span.set_attribute(key, value)
                
                # Add function args as attributes (only if small)
                if args and len(str(args)) < 1000:
                    span.set_attribute("function.args", str(args))
                if kwargs and len(str(kwargs)) < 1000:
                    span.set_attribute("function.kwargs", str(kwargs))
                
                try:
                    result = func(*args, **kwargs)
                    span.set_attribute("function.status", "success")
                    return result
                except Exception as e:
                    span.set_attribute("function.status", "error")
                    span.set_attribute("error.type", type(e).__name__)
                    span.set_attribute("error.message", str(e))
                    span.record_exception(e)
                    raise
        
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            if _TRACER is None:
                return await func(*args, **kwargs)
            
            with _TRACER.start_as_current_span(span_name) as span:
                # Add custom attributes
                if attributes:
                    for key, value in attributes.items():
                        span.set_attribute(key, value)
                
                try:
                    result = await func(*args, **kwargs)
                    span.set_attribute("function.status", "success")
                    return result
                except Exception as e:
                    span.set_attribute("function.status", "error")
                    span.set_attribute("error.type", type(e).__name__)
                    span.set_attribute("error.message", str(e))
                    span.record_exception(e)
                    raise
        
        # Return appropriate wrapper
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


@contextmanager
def trace_span(span_name: str, attributes: Optional[Dict[str, Any]] = None):
    """Context manager to create a trace span.
    
    Use for tracing code blocks that aren't functions.
    
    Args:
        span_name: Name of the span
        attributes: Span attributes
        
    Example:
        >>> with trace_span("batch_processing", {"batch_size": 100}):
        ...     for item in batch:
        ...         process(item)
    """
    if _TRACER is None:
        yield
        return
    
    with _TRACER.start_as_current_span(span_name) as span:
        if attributes:
            for key, value in attributes.items():
                span.set_attribute(key, value)
        yield span


def add_span_attributes(attributes: Dict[str, Any]) -> None:
    """Add attributes to the current active span.
    
    Args:
        attributes: Key-value pairs to add
        
    Example:
        >>> add_span_attributes({
        ...     "query_length": len(query),
        ...     "top_k": 10,
        ... })
    """
    current_span = trace.get_current_span()
    if current_span:
        for key, value in attributes.items():
            current_span.set_attribute(key, value)


# =============================================================================
# Metrics Helpers
# =============================================================================

@contextmanager
def track_request_metrics(endpoint: str, method: str = "POST"):
    """Context manager to automatically track request metrics.
    
    Tracks:
    - Request count (with status code)
    - Request latency
    - Active requests (in-flight)
    
    Args:
        endpoint: API endpoint (e.g., "/v1/chat")
        method: HTTP method (e.g., "POST")
        
    Example:
        >>> with track_request_metrics("/v1/chat"):
        ...     response = agent.run(query)
    """
    ACTIVE_REQUESTS.inc()
    start_time = time.time()
    status_code = "200"  # Default to success
    
    try:
        yield
    except Exception as e:
        status_code = "500"
        raise
    finally:
        duration = time.time() - start_time
        REQUEST_LATENCY.labels(endpoint=endpoint, method=method).observe(duration)
        REQUEST_COUNT.labels(endpoint=endpoint, status_code=status_code, method=method).inc()
        ACTIVE_REQUESTS.dec()


def track_llm_call(
    provider: str,
    model: str,
    operation: str,
    duration: float,
    input_tokens: int,
    output_tokens: int,
    cost_usd: float,
) -> None:
    """Record LLM call metrics.
    
    Args:
        provider: LLM provider ("vertexai", "ollama")
        model: Model name ("gemini-2.5-flash", "qwen3-vl:4b")
        operation: Operation type ("generate", "grade", "rewrite")
        duration: Call duration in seconds
        input_tokens: Number of input tokens
        output_tokens: Number of output tokens
        cost_usd: Cost in USD
        
    Example:
        >>> track_llm_call(
        ...     provider="vertexai",
        ...     model="gemini-2.5-flash",
        ...     operation="generate",
        ...     duration=2.5,
        ...     input_tokens=1000,
        ...     output_tokens=500,
        ...     cost_usd=0.0015,
        ... )
    """
    LLM_CALL_COUNT.labels(provider=provider, model=model, operation=operation).inc()
    LLM_LATENCY.labels(provider=provider, model=model, operation=operation).observe(duration)
    LLM_TOKEN_COUNT.labels(provider=provider, model=model, token_type="input").inc(input_tokens)
    LLM_TOKEN_COUNT.labels(provider=provider, model=model, token_type="output").inc(output_tokens)
    LLM_COST.labels(provider=provider, model=model).inc(cost_usd)


def track_retrieval(duration: float, result_count: int, status: str = "success") -> None:
    """Record retrieval metrics.
    
    Args:
        duration: Retrieval duration in seconds
        result_count: Number of jobs retrieved
        status: "success", "failure", or "empty"
    """
    RETRIEVAL_LATENCY.observe(duration)
    RETRIEVAL_COUNT.labels(status=status).inc()


def track_grading(duration: float, average_score: float) -> None:
    """Record grading metrics.
    
    Args:
        duration: Grading duration in seconds
        average_score: Average relevance score (0-10)
    """
    GRADING_LATENCY.observe(duration)
    AVERAGE_RELEVANCE_SCORE.set(average_score)


def track_agent_execution(duration: float, step_counts: Dict[str, int]) -> None:
    """Record agent execution metrics.
    
    Args:
        duration: Total agent execution time
        step_counts: Dict of step_name -> count (e.g., {"retrieve": 2, "generate": 1})
    """
    AGENT_EXECUTION_LATENCY.observe(duration)
    for step_name, count in step_counts.items():
        AGENT_STEP_COUNT.labels(step_name=step_name).inc(count)


# =============================================================================
# FastAPI Integration
# =============================================================================

def instrument_fastapi(app) -> None:
    """Automatically instrument FastAPI with OpenTelemetry.
    
    Adds automatic tracing to all FastAPI endpoints.
    
    Args:
        app: FastAPI application instance
        
    Example:
        >>> from fastapi import FastAPI
        >>> app = FastAPI()
        >>> instrument_fastapi(app)
    """
    try:
        FastAPIInstrumentor.instrument_app(app)
        logger.info("[Observability] FastAPI instrumented with OpenTelemetry")
    except Exception as e:
        logger.warning(f"[Observability] FastAPI instrumentation failed: {e}")


# =============================================================================
# Prometheus Metrics Endpoint
# =============================================================================

def get_metrics_handler():
    """Return handler for /metrics endpoint (Prometheus format).
    
    Example:
        >>> from prometheus_client import generate_latest
        >>> @app.get("/metrics")
        >>> def metrics():
        ...     return Response(generate_latest(), media_type="text/plain")
    """
    from prometheus_client import generate_latest
    return generate_latest
