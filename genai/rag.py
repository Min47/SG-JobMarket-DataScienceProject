"""RAG Pipeline for Job Market Intelligence.

This module implements Retrieval-Augmented Generation using:
- BigQuery Vector Search for semantic job retrieval
- Vertex AI Gemini Pro for answer generation
- Hybrid search (vector + keyword) for optimal results
"""

from __future__ import annotations

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timezone

from google.cloud import bigquery
from google.cloud import aiplatform
import vertexai
from vertexai.generative_models import GenerativeModel
import numpy as np
import json

from utils.config import Settings
from nlp.embeddings import EmbeddingGenerator

logger = logging.getLogger(__name__)


# =============================================================================
# Query Embedding (Reuses NLP module)
# =============================================================================

# Global embedding generator instance (singleton pattern)
_EMBEDDING_GENERATOR: Optional[EmbeddingGenerator] = None


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
    
    # Execute query
    try:
        query_job = client.query(sql, project=settings.gcp_project_id)
        results = list(query_job.result())
        
        logger.info(f"[RAG] Retrieved {len(results)} jobs from BigQuery")
        
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
        logger.error(f"[RAG] BigQuery vector search failed: {e}")
        logger.exception(e)
        return []


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
    
    # Initialize Vertex AI
    try:
        vertexai.init(project=settings.gcp_project_id, location=settings.gcp_region)
        model = GenerativeModel("gemini-2.5-flash")  # Fast model for grading
        logger.debug("[RAG] Initialized Vertex AI Gemini")
    except Exception as e:
        logger.error(f"[RAG] Failed to initialize Vertex AI: {e}")
        # Fallback: Return all documents with neutral score
        for doc in documents:
            doc['relevance_score'] = 7.0
            doc['relevance_explanation'] = "Grading unavailable (LLM init failed)"
        return documents
    
    # Grade each document
    graded_docs = []
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

Respond ONLY with JSON: {{"score": 8.5, "explanation": "reason"}}"""
            
            # Call Gemini
            response = model.generate_content(
                prompt,
                generation_config={
                    "temperature": 0.0,  # Zero temp for deterministic scoring
                    "max_output_tokens": 65535,  # Large buffer for extended thinking models
                    "candidate_count": 1, # How many generations to produce
                }
            )
            
            # Parse response (should be pure JSON now)
            response_text = response.text.strip()
            
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
            
            grading = json.loads(response_text) if response_text else {"score": 5.0, "explanation": "Empty response"}
            score = float(grading.get('score', 5.0))
            explanation = grading.get('explanation', 'No explanation provided')
            
            # Validate score range
            score = max(0.0, min(10.0, score))
            
            # Add grading fields to document
            doc['relevance_score'] = score
            doc['relevance_explanation'] = explanation
            
            # Filter by threshold
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
    
    # Re-rank by relevance score (descending)
    graded_docs.sort(key=lambda x: x.get('relevance_score', 0), reverse=True)
    
    logger.info(
        f"[RAG] Kept {len(graded_docs)}/{len(documents)} documents after grading "
        f"(threshold={threshold})"
    )
    
    if graded_docs:
        top_score = graded_docs[0].get('relevance_score', 0)
        logger.info(
            f"[RAG] Top result: '{graded_docs[0].get('job_title', 'N/A')}' "
            f"(relevance={top_score:.1f}/10)"
        )
    
    return graded_docs


# =============================================================================
# Answer Generation
# =============================================================================

def generate_answer(
    query: str,
    context_jobs: List[Dict[str, Any]],
    settings: Optional[Settings] = None,
) -> str:
    """Generate natural language answer using Gemini Pro.
    
    TODO: Implement answer generation
    - Construct prompt with query + retrieved job context
    - Call Vertex AI Gemini Pro API
    - Format response with job details, salary insights, trends
    - Include citations/sources (job IDs, companies)
    
    Args:
        query: User's question
        context_jobs: Retrieved jobs to use as context
        settings: Configuration settings
        
    Returns:
        Generated answer as markdown-formatted string
        
    Example:
        >>> jobs = retrieve_jobs("software engineer salary Singapore")
        >>> answer = generate_answer("What's the average salary?", jobs)
        >>> print(answer)
    """
    logger.info(f"[RAG] Generating answer for query: {query}")
    
    # TODO: Replace with Vertex AI Gemini call
    # Placeholder response
    return (
        f"**Answer to: {query}**\n\n"
        f"Based on {len(context_jobs)} relevant jobs, here's what I found:\n\n"
        f"*[Placeholder: Gemini Pro response will be generated here]*\n\n"
        f"**Sources:** {len(context_jobs)} jobs analyzed from JobStreet and MyCareersFuture."
    )


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


# =============================================================================
# Vector Index Management (Future)
# =============================================================================

def create_vector_index(
    table_name: str = "cleaned_jobs",
    embedding_column: str = "job_description_embedding",
    settings: Optional[Settings] = None,
) -> None:
    """Create BigQuery Vector Index for semantic search.
    
    TODO: Implement vector index creation
    - Check if embeddings column exists in table
    - Create vector index using CREATE VECTOR INDEX SQL
    - Configure distance metric (cosine, euclidean, dot_product)
    - Set index options (num_leaves, distance_type)
    
    Args:
        table_name: Name of the table to index
        embedding_column: Column containing embeddings
        settings: Configuration settings
    """
    logger.info(f"[RAG] Creating vector index on {table_name}.{embedding_column}")
    
    # TODO: Implement using BigQuery SQL
    # Example SQL:
    # CREATE VECTOR INDEX job_embeddings_idx
    # ON `project.dataset.cleaned_jobs`(job_description_embedding)
    # OPTIONS(distance_type='COSINE', index_type='IVF');
    
    raise NotImplementedError("Vector index creation not yet implemented")
