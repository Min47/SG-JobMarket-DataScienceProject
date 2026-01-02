"""RAG Pipeline for Job Market Intelligence.

This module implements Retrieval-Augmented Generation using:
- BigQuery Vector Search for semantic job retrieval
- Vertex AI Gemini Pro for answer generation
- Hybrid search (vector + keyword) for optimal results
"""

from __future__ import annotations

import logging
import time
from typing import Dict, List, Any, Optional
from datetime import datetime, timezone

from google.cloud import bigquery
import numpy as np
import json

from utils.config import Settings
from nlp.embeddings import EmbeddingGenerator
from genai.gateway import ModelGateway, GenerationConfig
from genai.observability import (
    trace_function,
    trace_span,
    add_span_attributes,
    track_retrieval,
    track_grading,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Query Embedding
# =============================================================================

# Global singleton instances
_EMBEDDING_GENERATOR: Optional[EmbeddingGenerator] = None
_MODEL_GATEWAY: Optional[ModelGateway] = None


def _get_embedding_generator() -> EmbeddingGenerator:
    """Get or create singleton EmbeddingGenerator instance.
    
    Returns:
        Shared EmbeddingGenerator instance
    """
    global _EMBEDDING_GENERATOR
    if _EMBEDDING_GENERATOR is None:
        logger.info("[RAG] Initializing EmbeddingGenerator")
        _EMBEDDING_GENERATOR = EmbeddingGenerator()  # Uses default all-MiniLM-L6-v2
    return _EMBEDDING_GENERATOR


def _get_model_gateway() -> ModelGateway:
    """Get or create singleton ModelGateway instance.
    
    Returns:
        Shared ModelGateway instance with all providers
    """
    global _MODEL_GATEWAY
    if _MODEL_GATEWAY is None:
        logger.info("[RAG] Initializing ModelGateway")
        _MODEL_GATEWAY = ModelGateway()
    return _MODEL_GATEWAY


def embed_query(
    query: str,
    settings: Optional[Settings] = None,
) -> List[float]:
    """Generate 384-dim embedding for user query using all-MiniLM-L6-v2.
    
    Reuses the EmbeddingGenerator from nlp.embeddings module to avoid
    code duplication and ensure consistent embedding generation.
    
    Args:
        query: User's natural language query (max 512 tokens)
        settings: Configuration settings (unused, kept for API compatibility)
        
    Returns:
        384-dimensional L2-normalized float vector
        
    Raises:
        ValueError: If query is empty or too short
        
    Example:
        >>> embedding = embed_query("data scientist jobs in fintech")
        >>> len(embedding)
        384
        >>> isinstance(embedding[0], float)
        True
    """
    # Input validation
    if not query or not query.strip():
        raise ValueError("Query cannot be empty")
    
    query = query.strip()
    
    if len(query) < 3:
        raise ValueError(f"Query too short (min 3 chars): '{query}'")
    
    logger.info(f"[RAG] Generating embedding for query: {query[:100]}...")
    
    # Truncate very long queries
    if len(query) > 1000:  # ~500 tokens
        query = query[:1000]
        logger.warning("[RAG] Query truncated to 1000 chars")
    
    # Get shared embedding generator
    generator = _get_embedding_generator()
    
    # Generate embedding using existing NLP module
    # Note: We need to manually normalize for cosine similarity in BigQuery
    embeddings = generator.embed_texts([query], show_progress=False)
    
    if embeddings.size == 0:
        raise ValueError("Failed to generate embedding")
    
    # Extract single embedding
    embedding_array = embeddings[0]  # Shape: (384,)
    
    # Normalize for cosine similarity (L2 norm = 1)
    norm = np.linalg.norm(embedding_array)
    if norm > 0:
        embedding_array = embedding_array / norm
    
    logger.info(f"[RAG] Generated {len(embedding_array)}-dim embedding (norm={norm:.4f}, normalized)")
    
    # Convert to list for JSON serialization
    return embedding_array.tolist()


# =============================================================================
# Vector Search & Retrieval
# =============================================================================

@trace_function("retrieve_jobs", {"operation": "vector_search"})
def retrieve_jobs(
    query: str,
    top_k: int = 10,
    filters: Optional[Dict[str, Any]] = None,
    hybrid_weight: float = 0.7,
    settings: Optional[Settings] = None,
) -> List[Dict[str, Any]]:
    """Retrieve relevant jobs using BigQuery Vector Search with hybrid scoring.
    
    Combines:
    - Vector similarity (embedding cosine distance)
    - Keyword matching (for exact term matches)
    
    Args:
        query: User's natural language query
        top_k: Number of results to return (default: 10)
        filters: Optional filters:
            - location: str (e.g., "Singapore", "Central")
            - min_salary: float (monthly SGD)
            - max_salary: float (monthly SGD)
            - work_type: str (e.g., "Full Time", "Contract")
            - classification: str (e.g., "Information & Communication Technology")
        hybrid_weight: Weight for vector score (0.0-1.0, default 0.7)
            - 0.7 = 70% vector + 30% keyword
            - 1.0 = 100% vector only
        settings: Configuration settings
        
    Returns:
        List of job dictionaries with metadata and relevance scores:
        - job_id: str
        - source: str
        - job_title: str
        - company_name: str
        - job_location: str
        - job_classification: str
        - job_work_type: str
        - job_salary_min_sgd_monthly: Optional[float]
        - job_salary_max_sgd_monthly: Optional[float]
        - job_description: str (truncated to 500 chars)
        - vector_distance: float (0.0-2.0, lower = more similar)
        - keyword_score: float (0.0-1.0, higher = more matches)
        - hybrid_score: float (weighted combination)
        
    Example:
        >>> jobs = retrieve_jobs(
        ...     "data scientist with python experience",
        ...     top_k=5,
        ...     filters={"min_salary": 5000, "location": "Singapore"}
        ... )
        >>> for job in jobs:
        ...     print(f"{job['job_title']} at {job['company_name']} - Score: {job['hybrid_score']:.3f}")
    """
    logger.info(f"[RAG] Retrieving jobs for query: '{query}' (top_k={top_k}, filters={filters})")
    
    if not settings:
        settings = Settings.load()
    
    # Step 1: Generate query embedding
    try:
        query_embedding = embed_query(query, settings)
    except Exception as e:
        logger.error(f"[RAG] Failed to generate query embedding: {e}")
        return []
    
    # Step 2: Build BigQuery vector search query
    client = bigquery.Client(project=settings.gcp_project_id)
    
    # Construct filter WHERE clauses
    filter_clauses = []
    if filters:
        if "location" in filters:
            # Partial match on location (e.g., "Singapore" matches "Singapore, Central")
            filter_clauses.append(f"c.job_location LIKE '%{filters['location']}%'")
        
        if "min_salary" in filters:
            filter_clauses.append(f"c.job_salary_min_sgd_monthly >= {filters['min_salary']}")
        
        if "max_salary" in filters:
            filter_clauses.append(f"c.job_salary_max_sgd_monthly <= {filters['max_salary']}")
        
        if "work_type" in filters:
            filter_clauses.append(f"c.job_work_type = '{filters['work_type']}'")
        
        if "classification" in filters:
            filter_clauses.append(f"c.job_classification LIKE '%{filters['classification']}%'")
    
    where_clause = " AND " + " AND ".join(filter_clauses) if filter_clauses else ""
    
    # Keyword scoring: Count query terms in title + description
    query_terms = query.lower().split()
    keyword_conditions = " + ".join([
        f"(CASE WHEN LOWER(c.job_title) LIKE '%{term}%' OR LOWER(c.job_description) LIKE '%{term}%' THEN 1 ELSE 0 END)"
        for term in query_terms[:10]  # Limit to first 10 terms
    ])
    keyword_score_expr = f"({keyword_conditions}) / {len(query_terms)}" if query_terms else "0"
    
    # BigQuery SQL with VECTOR_SEARCH function
    # VECTOR_SEARCH returns: query (RECORD), base (RECORD), distance (FLOAT)
    # Access base table columns via base.column_name, distance directly
    # cleaned_jobs is append-only, so use ROW_NUMBER() to get latest version
    sql = f"""
    WITH latest_jobs AS (
        SELECT
            *,
            ROW_NUMBER() OVER (PARTITION BY job_id, source ORDER BY scrape_timestamp DESC) AS rn
        FROM `{settings.gcp_project_id}.{settings.bigquery_dataset_id}.cleaned_jobs`
    )
    SELECT
        c.job_id,
        c.source,
        c.job_title,
        c.company_name,
        c.job_location,
        c.job_classification,
        c.job_work_type,
        c.job_salary_min_sgd_monthly,
        c.job_salary_max_sgd_monthly,
        SUBSTR(c.job_description, 1, 500) AS job_description_preview,
        c.job_url,
        vs.distance AS vector_distance,
        {keyword_score_expr} AS keyword_score,
        ({hybrid_weight} * (1.0 - vs.distance / 2.0) + {1.0 - hybrid_weight} * {keyword_score_expr}) AS hybrid_score
    FROM VECTOR_SEARCH(
        (SELECT * FROM `{settings.gcp_project_id}.{settings.bigquery_dataset_id}.job_embeddings`),
        'embedding',
        (SELECT {query_embedding} AS embedding),
        distance_type => 'COSINE',
        top_k => {top_k * 3}  -- Retrieve 3x for filtering
    ) AS vs
    JOIN latest_jobs c
        ON vs.base.job_id = c.job_id AND vs.base.source = c.source AND c.rn = 1
    WHERE TRUE {where_clause}
    ORDER BY hybrid_score DESC
    LIMIT {top_k}
    """
    
    logger.info(f"[RAG] Executing BigQuery vector search...")
    logger.debug(f"[RAG] Full SQL query:\n{sql}")
    # print(f"Full SQL query:\n{sql}")
    
    # Execute query with metrics tracking
    start_time = time.time()
    
    try:
        query_job = client.query(sql, project=settings.gcp_project_id)
        results = list(query_job.result())
        
        duration = time.time() - start_time
        
        # Track retrieval metrics
        if len(results) == 0:
            track_retrieval(duration, 0, status="empty")
        else:
            track_retrieval(duration, len(results), status="success")
        
        # Add span attributes for tracing
        add_span_attributes({
            "query_length": len(query),
            "top_k": top_k,
            "result_count": len(results),
            "duration_ms": int(duration * 1000),
            "has_filters": filters is not None,
        })
        
        logger.info(f"[RAG] Retrieved {len(results)} jobs from BigQuery in {duration:.2f}s")
        
        # Convert to dictionaries
        jobs = []
        for row in results:
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
                "job_description": row.job_description_preview,
                "job_url": row.job_url,
                "vector_distance": float(row.vector_distance),
                "keyword_score": float(row.keyword_score),
                "hybrid_score": float(row.hybrid_score),
            }
            jobs.append(job)
        
        if jobs:
            logger.info(
                f"[RAG] Top result: '{jobs[0]['job_title']}' "
                f"(score={jobs[0]['hybrid_score']:.3f})"
            )
        
        return jobs
    
    except Exception as e:
        duration = time.time() - start_time
        track_retrieval(duration, 0, status="failure")
        logger.error(f"[RAG] BigQuery vector search failed: {e}")
        logger.exception(e)
        return []


@trace_function("grade_documents", {"operation": "relevance_grading"})
def grade_documents(
    query: str,
    documents: List[Dict[str, Any]],
    threshold: float = 5.0,
    settings: Optional[Settings] = None,
) -> List[Dict[str, Any]]:
    """Grade retrieved documents for relevance using Gemini Pro.
    
    Uses Vertex AI Gemini to score each job's relevance to the user query
    on a scale of 0-10. Documents below the threshold are filtered out.
    Results are re-ranked by LLM grade (descending).
    
    Args:
        query: Original user query
        documents: Retrieved job documents from retrieve_jobs()
        threshold: Minimum relevance score (0-10) to keep document (default: 5.0)
        settings: Configuration settings
        
    Returns:
        Filtered and re-ranked documents with added fields:
        - relevance_score (float): LLM-assigned score 0-10
        - relevance_explanation (str): Brief explanation of score
        
    Raises:
        ValueError: If documents list is empty
        
    Example:
        >>> docs = retrieve_jobs("data scientist python", top_k=10)
        >>> graded = grade_documents("data scientist python", docs, threshold=6.0)
        >>> print(f"Kept {len(graded)}/{len(docs)} relevant jobs")
        >>> graded[0]['relevance_score']
        8.5
    """
    if not documents:
        logger.warning("[RAG] No documents to grade")
        return []
    
    logger.info(f"[RAG] Grading {len(documents)} documents for relevance (threshold={threshold})")
    
    if not settings:
        settings = Settings.load()
    
    # Get ModelGateway for LLM calls
    gateway = _get_model_gateway()
    
    # Grade each document
    graded_docs = []  # Docs that pass threshold
    all_graded_docs = []  # ALL docs with scores (for statistics)

    # Execute query with metrics tracking
    start_time = time.time()
    
    for i, doc in enumerate(documents, 1):
        try:
            # Construct grading prompt (very short for faster response)
            # Escape quotes to avoid JSON issues
            query_safe = query.replace('"', '\\"').replace("'", "\\'")
            desc = doc.get('job_description', 'N/A')[:200].replace('"', '\\"').replace("'", "\\'")
            title = doc.get('job_title', 'N/A').replace('"', '\\"')
            company = doc.get('company_name', 'N/A').replace('"', '\\"')
            classification = doc.get('job_classification', 'N/A').replace('"', '\\"')
            
            prompt = f"""Rate relevance 0-10.
Query: {query_safe}
Job: {title} at {company}
Type: {classification}
Desc: {desc}

CRITICAL: Explanation (brief reason) should under 1000 words.

Respond ONLY with valid JSON: {{"score": 8.5, "explanation": "brief reason"}}"""
            
            # Call LLM via gateway (with automatic fallback)
            config = GenerationConfig(
                temperature=0.0,  # Deterministic scoring
                max_tokens=8192,   # Short responses
            )
            
            result = gateway.generate(
                prompt,
                model="auto",  # Auto-select best provider
                config=config,
                fallback=True,  # Enable fallback
            )
            
            response_text = result.text.strip()
            
            # Sometimes Gemini adds prefixes even with response_mime_type - clean them
            if response_text.startswith("Here is"):
                # Extract JSON after the prefix
                lines = response_text.split('\n')
                for i, line in enumerate(lines):
                    if line.strip().startswith('{'):
                        response_text = '\n'.join(lines[i:])
                        break
            
            # Remove markdown code blocks if present
            if '```json' in response_text:
                parts = response_text.split('```json')
                if len(parts) > 1:
                    response_text = parts[1].split('```')[0].strip()
            elif '```' in response_text:
                parts = response_text.split('```')
                if len(parts) >= 2:
                    response_text = parts[1].strip()
            
            try:
                grading = json.loads(response_text)
                score = float(grading.get('score', 5.0))
                explanation = grading.get('explanation', 'No explanation provided')
            except json.JSONDecodeError as e:
                logger.warning(f"[RAG] Failed to parse Gemini response for doc {i}: {e}")
                logger.warning(f"[RAG] Full raw response: {response_text}")
                # Fallback: Try to extract score with regex
                import re
                score_match = re.search(r'"score"\s*:\s*(\d+\.?\d*)', response_text)
                if score_match:
                    score = float(score_match.group(1))
                    explanation = "Incomplete response (JSON parse error)"
                    logger.info(f"[RAG] Extracted score {score} via regex fallback")
                else:
                    # Last resort: Assign neutral score
                    score = 5.0
                    explanation = f"Failed to parse response: {response_text[:100]}"
                    logger.warning(f"[RAG] Using fallback score 5.0 for doc {i}")
            
            # Validate score range
            score = max(0.0, min(10.0, score))
            
            # Add grading fields to document
            doc['relevance_score'] = score
            doc['relevance_explanation'] = explanation
            
            # Store in all_graded for average calculation (BEFORE filtering)
            all_graded_docs.append(doc)
            
            # Filter by threshold for final results
            if score >= threshold:
                graded_docs.append(doc)
                logger.debug(
                    f"[RAG] Doc {i}/{len(documents)}: '{doc.get('job_title', 'N/A')}' "
                    f"- Score: {score:.1f} ✓ (kept)"
                )
            else:
                logger.debug(
                    f"[RAG] Doc {i}/{len(documents)}: '{doc.get('job_title', 'N/A')}' "
                    f"- Score: {score:.1f} ✗ (filtered)"
                )
        
        except json.JSONDecodeError as e:
            logger.warning(f"[RAG] Failed to parse Gemini response for doc {i}: {e}")
            logger.warning(f"[RAG] Full raw response: {response_text}")
            # Assign neutral score on error
            doc['relevance_score'] = 6.0
            doc['relevance_explanation'] = "Grading error - assigned neutral score"
            if 6.0 >= threshold:
                graded_docs.append(doc)
        
        except Exception as e:
            logger.warning(f"[RAG] Error grading doc {i}: {e}")
            # Assign neutral score on error
            doc['relevance_score'] = 6.0
            doc['relevance_explanation'] = f"Grading error: {str(e)[:50]}"
            if 6.0 >= threshold:
                graded_docs.append(doc)
    
    # Calculate statistics on ALL retrieved documents (before filtering)
    avg_score_all = (
        sum(doc['relevance_score'] for doc in all_graded_docs) / len(all_graded_docs)
        if all_graded_docs else 0.0
    )
    
    # Track grading metrics
    grading_duration = time.time() - start_time
    track_grading(grading_duration, avg_score_all)
    
    # Add span attributes
    add_span_attributes({
        "document_count": len(documents),
        "passed_threshold": len(graded_docs),
        "average_score": round(avg_score_all, 2),
        "threshold": threshold,
        "duration_ms": int(grading_duration * 1000),
    })
    
    # Re-rank by relevance score (descending)
    graded_docs.sort(key=lambda x: x.get('relevance_score', 0), reverse=True)
    
    logger.info(
        f"[RAG] Kept {len(graded_docs)}/{len(documents)} documents after grading "
        f"(threshold={threshold}, avg_score_all={avg_score_all:.1f}/10, time={grading_duration:.1f}s)"
    )
    
    if graded_docs:
        top_score = graded_docs[0].get('relevance_score', 0)
        logger.info(
            f"[RAG] Top result: '{graded_docs[0].get('job_title', 'N/A')}' "
            f"(relevance={top_score:.1f}/10)"
        )
    else:
        logger.warning(
            f"[RAG] No documents passed threshold={threshold} "
            f"(avg_score_all={avg_score_all:.1f}/10)"
        )
    
    return graded_docs


# =============================================================================
# Answer Generation
# =============================================================================

@trace_function("generate_answer", {"operation": "answer_generation"})
def generate_answer(
    query: str,
    context_jobs: List[Dict[str, Any]],
    max_context_jobs: int = 5,
    settings: Optional[Settings] = None,
) -> Dict[str, Any]:
    """Generate natural language answer using Gemini Pro with job context.
    
    Uses the top-K most relevant jobs (sorted by relevance_score from grade_documents)
    as context to generate a comprehensive answer. Includes citations to source jobs.
    
    Args:
        query: User's original question
        context_jobs: Graded and filtered jobs (output from grade_documents)
        max_context_jobs: Maximum number of jobs to include in context (default: 5)
        settings: Configuration settings
        
    Returns:
        Dictionary with:
        - answer: Generated natural language response (markdown)
        - sources: List of job citations used
        - metadata: Token usage, latency, etc.
        
    Raises:
        ValueError: If context_jobs is empty
        RuntimeError: If Gemini API call fails
        
    Example:
        >>> jobs = retrieve_jobs("python developer", top_k=10)
        >>> graded = grade_documents("python developer", jobs)
        >>> result = generate_answer("What Python jobs are available?", graded)
        >>> print(result['answer'])
    """
    if not context_jobs:
        logger.warning("[RAG] No context jobs provided for answer generation")
        return {
            "answer": (
                "I couldn't find any jobs matching your search criteria. "
                "This might be because:\n"
                "- The filters are too restrictive (try relaxing location or salary requirements)\n"
                "- The query terms don't match available jobs\n"
                "- Try a broader search without filters first."
            ),
            "sources": [],
            "metadata": {
                "error": "No context",
                "suggestion": "Try removing filters or using broader search terms"
            }
        }
    
    if not settings:
        settings = Settings.load()
    
    logger.info(f"[RAG] Generating answer for query: {query[:100]}... using {len(context_jobs)} jobs")
    
    # Get ModelGateway for LLM calls
    gateway = _get_model_gateway()
    
    # Limit context to top-K most relevant jobs
    top_jobs = context_jobs[:max_context_jobs]
    logger.info(f"[RAG] Using top {len(top_jobs)} jobs as context")
    
    # Format job context for prompt
    context_text = _format_job_context(top_jobs)
    
    # Construct prompt
    prompt = f"""You are a Singapore job market expert assistant. Answer the user's question based on the provided job listings.

User Question: "{query}"

Available Job Listings:
{context_text}

Instructions:
1. Provide a clear, comprehensive answer based on the job listings above
2. Include specific details: job titles, companies, salary ranges, requirements
3. Cite jobs by number [1], [2], etc. when mentioning specific information
4. If asked about salaries, provide ranges and mention currency (SGD)
5. If asked about requirements, summarize common skills/qualifications
6. If the question cannot be fully answered with the provided jobs, acknowledge this
7. **IMPORTANT: Keep your response under 1000 characters (approximately 3-4 paragraphs)**
8. Use markdown formatting for better readability

Your Answer:"""
    
    # Call LLM via gateway
    try:
        start_time = datetime.now(timezone.utc)
        
        # Configuration: Limit output to 1024 tokens (~700-800 words)
        # Note: Token limits help control response length and reduce costs
        config = GenerationConfig(
            temperature=0.3,  # Slightly creative but factual
            max_tokens=8192,
            top_p=0.9,
            top_k=40,
        )
        
        result = gateway.generate(
            prompt,
            model="auto",  # Auto-select best provider
            config=config,
            fallback=True,
        )
        
        end_time = datetime.now(timezone.utc)
        latency_ms = int((end_time - start_time).total_seconds() * 1000)
        
        # Extract and truncate answer (safety limit: 4096 chars)
        answer_text = result.text.strip()
        if len(answer_text) > 4096:
            logger.warning(f"[RAG] Truncating answer from {len(answer_text)} to 4096 chars")
            answer_text = answer_text[:4093] + "..."
        logger.info(
            f"[RAG] Generated answer ({len(answer_text)} chars, {latency_ms}ms) "
            f"using {result.provider}"
        )
        
        # Add span attributes for tracing
        add_span_attributes({
            "context_job_count": len(top_jobs),
            "answer_length": len(answer_text),
            "duration_ms": latency_ms,
            "model": f"{result.provider}/{result.model}",
        })
        
        # Extract sources (jobs cited in answer)
        sources = _extract_sources(top_jobs)
        
        return {
            "answer": answer_text,
            "sources": sources,
            "metadata": {
                "num_context_jobs": len(top_jobs),
                "latency_ms": latency_ms,
                "model": f"{result.provider}/{result.model}",
                "cost_usd": result.cost,
                "query": query,
            }
        }
        
    except Exception as e:
        logger.error(f"[RAG] Error generating answer: {e}")
        raise RuntimeError(f"Failed to generate answer: {e}")


def _format_job_context(jobs: List[Dict[str, Any]]) -> str:
    """Format job listings into structured context for LLM prompt.
    
    Args:
        jobs: List of job dictionaries (from grade_documents output)
        
    Returns:
        Formatted string with numbered job listings
    """
    context_lines = []
    
    for i, job in enumerate(jobs, 1):
        # Extract key fields
        title = job.get('job_title', 'N/A')
        company = job.get('company_name', 'N/A')
        location = job.get('job_location', 'N/A')
        classification = job.get('job_classification', 'N/A')
        work_type = job.get('job_work_type', 'N/A')
        
        # Salary info
        salary_min = job.get('job_salary_min_sgd_monthly')
        salary_max = job.get('job_salary_max_sgd_monthly')
        
        if salary_min and salary_max:
            salary_str = f"SGD ${salary_min:,.0f} - ${salary_max:,.0f}/month"
        elif salary_min:
            salary_str = f"From SGD ${salary_min:,.0f}/month"
        elif salary_max:
            salary_str = f"Up to SGD ${salary_max:,.0f}/month"
        else:
            salary_str = "Not disclosed"
        
        # Description snippet (first 300 chars)
        description = job.get('job_description', '')
        desc_snippet = description[:300] + "..." if len(description) > 300 else description
        
        # Relevance score (if available from grading)
        relevance = job.get('relevance_score')
        relevance_str = f" (Relevance: {relevance:.1f}/10)" if relevance else ""
        
        # Format job entry
        job_entry = f"""[{i}] **{title}**{relevance_str}
   Company: {company}
   Location: {location}
   Type: {work_type} | {classification}
   Salary: {salary_str}
   Description: {desc_snippet}
"""
        context_lines.append(job_entry)
    
    return "\n".join(context_lines)


def _extract_sources(jobs: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """Extract source citations from jobs used in context.
    
    Args:
        jobs: List of job dictionaries
        
    Returns:
        List of source dictionaries with job_id, title, company, url
    """
    sources = []
    
    for i, job in enumerate(jobs, 1):
        sources.append({
            "number": str(i),  # Convert to string for Pydantic validation
            "job_id": job.get('job_id', 'N/A'),
            "job_title": job.get('job_title', 'N/A'),
            "company_name": job.get('company_name', 'N/A'),
            "job_url": job.get('job_url', '#'),
        })
    
    return sources


# =============================================================================
# RAG Pipeline Orchestration
# =============================================================================

def rag_pipeline(
    query: str,
    top_k: int = 10,
    filters: Optional[Dict[str, Any]] = None,
    settings: Optional[Settings] = None,
) -> Dict[str, Any]:
    """Complete RAG pipeline: Retrieve → Grade → Generate.
    
    This is the main entry point for RAG queries.
    
    Args:
        query: User's natural language question
        top_k: Number of jobs to retrieve
        filters: Optional filters for retrieval
        settings: Configuration settings
        
    Returns:
        Dict with 'answer', 'sources', and 'metadata'
        
    Example:
        >>> result = rag_pipeline("What are the top paying data science jobs?")
        >>> print(result['answer'])
        >>> print(f"Based on {len(result['sources'])} jobs")
    """
    logger.info(f"[RAG Pipeline] Starting for query: {query}")
    
    # Step 1: Retrieve
    retrieved_jobs = retrieve_jobs(query, top_k=top_k, filters=filters, settings=settings)
    logger.info(f"[RAG Pipeline] Retrieved {len(retrieved_jobs)} jobs")
    
    # Step 2: Grade
    relevant_jobs = grade_documents(query, retrieved_jobs)
    logger.info(f"[RAG Pipeline] {len(relevant_jobs)} jobs passed relevance filter")
    
    # Step 3: Generate
    answer = generate_answer(query, relevant_jobs, settings=settings)
    
    return {
        "answer": answer,
        "sources": relevant_jobs,
        "metadata": {
            "query": query,
            "retrieved_count": len(retrieved_jobs),
            "relevant_count": len(relevant_jobs),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    }