"""LangGraph Agent for Job Market Intelligence.

This module implements an agentic workflow using LangGraph for:
- Multi-step reasoning and planning
- Adaptive query refinement based on retrieval quality
- Conversational memory and context management
- Autonomous orchestration of RAG pipeline (retrieve → grade → generate)

Architecture:
    START → retrieve → grade → [decision] → generate → END
                          ↓ (low scores)
                        rewrite → retrieve (retry up to 2x)
"""

from __future__ import annotations

import logging
from typing import Dict, List, Any, Optional, TypedDict, Annotated, Literal
from operator import add

from langgraph.graph import StateGraph, END, START
from langgraph.graph.message import add_messages

from utils.config import Settings

logger = logging.getLogger(__name__)


# =============================================================================
# Agent State Management
# =============================================================================

class AgentState(TypedDict):
    """State for the LangGraph agent workflow.
    
    This TypedDict defines all data that flows through the graph nodes.
    LangGraph automatically manages state updates between nodes.
    
    Fields:
        messages: Conversation history (uses add_messages for merging)
        query: Current user query string
        original_query: Original user query (preserved for context)
        retrieved_jobs: Raw jobs from vector search (with scores)
        graded_jobs: Jobs after LLM relevance scoring
        final_answer: Generated natural language response
        rewrite_count: Number of query rewrites attempted (max 2)
        average_relevance_score: Mean score from grading step
        metadata: Additional context (timings, model info, filters)
    """
    # Conversation context
    messages: Annotated[List[Dict[str, str]], add_messages]
    query: str
    original_query: str
    
    # Pipeline data
    retrieved_jobs: List[Dict[str, Any]]
    graded_jobs: List[Dict[str, Any]]
    final_answer: Optional[Dict[str, Any]]  # {answer, sources, metadata}
    
    # Decision tracking
    rewrite_count: int
    average_relevance_score: float
    
    # Additional context
    metadata: Dict[str, Any]


# =============================================================================
# Decision Functions (Conditional Edges)
# =============================================================================

def should_rewrite(state: AgentState) -> Literal["rewrite", "generate"]:
    """Decide whether to rewrite query or proceed to generation.
    
    Decision logic:
    - If average relevance score < 6.0 AND rewrite_count < 2 → rewrite
    - Otherwise → generate (either good results or max retries reached)
    
    Args:
        state: Current agent state with grading results
        
    Returns:
        Next node to execute: "rewrite" or "generate"
    """
    avg_score = state.get("average_relevance_score", 0.0)
    rewrite_count = state.get("rewrite_count", 0)
    
    # Check if we should rewrite
    if avg_score < 6.0 and rewrite_count < 2:
        logger.info(
            f"[Agent Decision] Low relevance (avg={avg_score:.2f}), "
            f"retry #{rewrite_count + 1} → REWRITE"
        )
        return "rewrite"
    
    # Otherwise proceed to generation
    if avg_score < 6.0:
        logger.warning(
            f"[Agent Decision] Low relevance (avg={avg_score:.2f}) but "
            f"max retries reached ({rewrite_count}) → GENERATE anyway"
        )
    else:
        logger.info(
            f"[Agent Decision] Good relevance (avg={avg_score:.2f}) → GENERATE"
        )
    
    return "generate"


# =============================================================================
# Agent Nodes (Graph Components)
# =============================================================================

def retrieve_node(state: AgentState) -> AgentState:
    """Retrieve relevant jobs from vector store using BigQuery Vector Search.
    
    This node:
    1. Takes the current query from state
    2. Generates embedding using all-MiniLM-L6-v2
    3. Performs BigQuery VECTOR_SEARCH with hybrid scoring
    4. Returns top-K jobs with relevance scores
    5. Updates state with retrieved jobs
    
    The hybrid search combines:
    - 70% vector similarity (semantic meaning)
    - 30% keyword matching (exact terms)
    
    Args:
        state: Current agent state with 'query' field
        
    Returns:
        Updated state with 'retrieved_jobs' populated
        
    Example:
        Input state: {"query": "python developer"}
        Output state: {"retrieved_jobs": [10 jobs with scores]}
    """
    from genai.rag import retrieve_jobs
    
    query = state["query"]
    filters = state["metadata"].get("filters", {})
    
    logger.info(f"[Agent Node: Retrieve] Query: '{query[:100]}...', Filters: {filters}")
    
    try:
        # Call RAG retrieve_jobs with hybrid search
        jobs = retrieve_jobs(
            query=query,
            top_k=10,
            filters=filters,
            hybrid_weight=0.7,  # 70% vector + 30% keyword
        )
        
        state["retrieved_jobs"] = jobs
        state["metadata"]["retrieval_count"] = len(jobs)
        
        # Handle empty results gracefully
        if not jobs:
            logger.warning(
                f"[Agent Node: Retrieve] Retrieved 0 jobs. "
                f"Filters may be too restrictive: {filters}"
            )
        else:
            logger.info(
                f"[Agent Node: Retrieve] Retrieved {len(jobs)} jobs, "
                f"distances: {min(j['vector_distance'] for j in jobs):.3f}-"
                f"{max(j['vector_distance'] for j in jobs):.3f}"
            )
        
    except Exception as e:
        logger.error(f"[Agent Node: Retrieve] Failed: {e}")
        state["retrieved_jobs"] = []
        state["metadata"]["retrieval_count"] = 0
        state["metadata"]["retrieval_error"] = str(e)
    
    return state


def grade_node(state: AgentState) -> AgentState:
    """Grade documents for relevance using Gemini LLM.
    
    This node:
    1. Takes retrieved jobs from state
    2. Uses Gemini 2.5 Flash to score each job 0-10 for relevance
    3. Filters out jobs below threshold (5.0)
    4. Computes average relevance score for decision making
    5. Updates state with graded jobs and average score
    
    The grading prompt asks Gemini:
    "On a scale of 0-10, how relevant is this job to the query?"
    
    Example:
    - Query: "python developer"
    - Job: "Python Developer at Google" → Score: 10/10 (exact match)
    - Job: "Java Developer" → Score: 3/10 (wrong language) → filtered out
    
    Args:
        state: Current agent state with 'retrieved_jobs'
        
    Returns:
        Updated state with 'graded_jobs' and 'average_relevance_score'
    """
    from genai.rag import grade_documents
    
    query = state["query"]
    retrieved = state["retrieved_jobs"]
    
    logger.info(f"[Agent Node: Grade] Grading {len(retrieved)} jobs for query: '{query[:50]}...'")
    
    if not retrieved:
        logger.warning("[Agent Node: Grade] No jobs to grade, skipping")
        state["graded_jobs"] = []
        state["average_relevance_score"] = 0.0
        state["metadata"]["graded_count"] = 0
        return state
    
    try:
        # Call RAG grade_documents with Gemini scoring
        graded = grade_documents(
            query=query,
            documents=retrieved,
            threshold=5.0,  # Filter jobs below 5/10
        )
        
        # Compute average score for decision making
        if graded:
            avg_score = sum(job.get("relevance_score", 0.0) for job in graded) / len(graded)
        else:
            avg_score = 0.0
        
        state["graded_jobs"] = graded
        state["average_relevance_score"] = avg_score
        state["metadata"]["graded_count"] = len(graded)
        
        logger.info(
            f"[Agent Node: Grade] Graded {len(retrieved)} → {len(graded)} jobs passed, "
            f"avg score: {avg_score:.2f}/10"
        )
        
        # Log score distribution
        if graded:
            scores = [job.get("relevance_score", 0.0) for job in graded]
            logger.info(
                f"[Agent Node: Grade] Score range: {min(scores):.1f}-{max(scores):.1f}"
            )
        
    except Exception as e:
        logger.error(f"[Agent Node: Grade] Failed: {e}")
        # Fallback: pass through ungraded jobs with default score
        state["graded_jobs"] = retrieved
        state["average_relevance_score"] = 6.0  # Neutral score
        state["metadata"]["graded_count"] = len(retrieved)
        state["metadata"]["grading_error"] = str(e)
    
    return state


def generate_node(state: AgentState) -> AgentState:
    """Generate final natural language answer using Gemini.
    
    This node:
    1. Takes graded jobs from state (already filtered and scored)
    2. Formats jobs into context with citations [1], [2], [3]
    3. Uses Gemini 2.5 Flash to generate natural language answer
    4. Includes job details: title, company, salary, requirements
    5. Provides source citations for transparency
    6. Updates state with final answer
    
    The generation prompt includes:
    - System role: "Singapore job market expert assistant"
    - Job context with numbered listings
    - Instructions for citations, salary formatting, conciseness
    
    Example output:
    "Based on 5 relevant positions, here are Python developer jobs:
    
    OPTIMUM SOLUTIONS is hiring a Contract Python Developer [1] with
    salary SGD $5,000-$7,000/month. Requirements include 2-3 years
    Python experience and SQL knowledge.
    
    INNOVATIQ TECHNOLOGIES has a Full-time Python Developer [2] role
    focused on Security Operations automation..."
    
    Args:
        state: Current agent state with 'graded_jobs'
        
    Returns:
        Updated state with 'final_answer' = {answer, sources, metadata}
    """
    from genai.rag import generate_answer
    
    query = state["query"]
    graded = state["graded_jobs"]
    
    logger.info(
        f"[Agent Node: Generate] Generating answer with {len(graded)} context jobs, "
        f"avg score: {state['average_relevance_score']:.2f}"
    )
    
    try:
        # Call RAG generate_answer with Gemini Pro
        result = generate_answer(
            query=query,
            context_jobs=graded,
            max_context_jobs=5,  # Limit to top 5 for token efficiency
        )
        
        state["final_answer"] = result
        
        logger.info(
            f"[Agent Node: Generate] Generated answer: "
            f"{len(result['answer'])} chars, {len(result['sources'])} sources"
        )
        
    except Exception as e:
        logger.error(f"[Agent Node: Generate] Failed: {e}")
        # Fallback: return error message
        state["final_answer"] = {
            "answer": f"Sorry, I encountered an error generating the answer: {str(e)}",
            "sources": [],
            "metadata": {"error": str(e)}
        }
    
    return state


def rewrite_node(state: AgentState) -> AgentState:
    """Rewrite query using LLM to improve retrieval results.
    
    This node is triggered when grading scores are low (avg < 6.0).
    It uses Gemini to reformulate the query for better results.
    
    Query rewriting strategies:
    1. Add domain keywords ("software engineer" → "software engineer python sql")
    2. Expand abbreviations ("ML engineer" → "machine learning engineer")
    3. Add location context ("developer" → "developer jobs in Singapore")
    4. Clarify ambiguous terms ("data" → "data analyst OR data scientist")
    5. Simplify overly specific queries
    
    Example transformations:
    - "Find me a ML job" → "machine learning engineer jobs Singapore"
    - "Python stuff" → "Python developer software engineer"
    - "High paying tech" → "senior software engineer high salary"
    
    The rewrite preserves original intent while improving retrieval quality.
    Max 2 rewrites to prevent infinite loops.
    
    Args:
        state: Current agent state with low-scoring results
        
    Returns:
        Updated state with:
        - 'query': Rewritten query string
        - 'rewrite_count': Incremented counter
        - 'original_query': Preserved for reference
    """
    import vertexai
    from vertexai.generative_models import GenerativeModel
    
    original_query = state["query"]
    rewrite_count = state["rewrite_count"]
    
    logger.info(
        f"[Agent Node: Rewrite] Attempt #{rewrite_count + 1}, "
        f"Original: '{original_query}'"
    )
    
    try:
        # Initialize Vertex AI
        settings = Settings.load()
        vertexai.init(project=settings.gcp_project_id, location=settings.gcp_region)
        model = GenerativeModel("gemini-2.5-flash")
        
        # Construct rewrite prompt
        prompt = f"""You are a job search query optimizer for Singapore job market.

Original query: "{original_query}"

The original query returned low-relevance results (avg score < 6/10).
Rewrite this query to improve job search results.

Guidelines:
1. Add relevant job market keywords (e.g., "engineer", "developer", "analyst")
2. Expand abbreviations ("ML" → "machine learning", "DS" → "data science")
3. Keep Singapore context when relevant
4. Be specific about role types (junior, senior, manager)
5. Include common skill keywords when implied
6. Keep it concise (5-10 words)

Return ONLY the rewritten query, no explanation.

Rewritten query:"""
        
        # Generate rewrite
        response = model.generate_content(
            prompt,
            generation_config={
                "temperature": 0.7,  # Some creativity for alternatives
                "max_output_tokens": 100,
                "top_p": 0.9,
            }
        )
        
        rewritten_query = response.text.strip()
        
        # Clean up common artifacts
        rewritten_query = rewritten_query.strip('"').strip("'").strip()
        
        # Validate rewrite
        if len(rewritten_query) < 3:
            logger.warning(f"[Agent Node: Rewrite] Invalid rewrite, using original")
            rewritten_query = original_query
        
        # Update state
        state["query"] = rewritten_query
        state["rewrite_count"] = rewrite_count + 1
        
        logger.info(
            f"[Agent Node: Rewrite] Rewritten: '{rewritten_query}' "
            f"({len(rewritten_query)} chars)"
        )
        
    except Exception as e:
        logger.error(f"[Agent Node: Rewrite] Failed: {e}")
        # Fallback: keep original query but increment counter
        state["rewrite_count"] = rewrite_count + 1
        state["metadata"]["rewrite_error"] = str(e)
    
    return state


# =============================================================================
# Agent Graph Construction
# =============================================================================

def build_agent_graph(settings: Optional[Settings] = None) -> StateGraph:
    """Build LangGraph StateGraph for job market agent.
    
    Constructs the complete workflow with nodes and conditional edges:
    
    Graph Flow:
        1. START → retrieve_node
        2. retrieve_node → grade_node
        3. grade_node → should_rewrite() decision:
           - If avg_score < 6.0 AND retries < 2 → rewrite_node → retrieve_node
           - Otherwise → generate_node → END
    
    Args:
        settings: Configuration settings (passed to nodes)
        
    Returns:
        Compiled StateGraph ready for execution
        
    Example:
        >>> graph = build_agent_graph()
        >>> result = graph.invoke({
        ...     "query": "data scientist jobs",
        ...     "messages": [],
        ...     "rewrite_count": 0,
        ...     "metadata": {}
        ... })
        >>> print(result['final_answer']['answer'])
    """
    if not settings:
        settings = Settings.load()
    
    logger.info("[Agent] Building LangGraph StateGraph...")
    
    # Create graph with AgentState schema
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("grade", grade_node)
    workflow.add_node("generate", generate_node)
    workflow.add_node("rewrite", rewrite_node)
    
    # Define edges
    workflow.add_edge(START, "retrieve")  # Entry point
    workflow.add_edge("retrieve", "grade")  # Always grade after retrieval
    
    # Conditional edge: decide rewrite or generate based on scores
    workflow.add_conditional_edges(
        "grade",
        should_rewrite,
        {
            "rewrite": "rewrite", # If return "rewrite", then go to rewrite node
            "generate": "generate", # If return "generate", then go to generate node
        }
    )
    
    workflow.add_edge("rewrite", "retrieve")  # Retry loop
    workflow.add_edge("generate", END)  # Exit point
    
    # Compile graph
    compiled_graph = workflow.compile()
    
    logger.info(
        "[Agent] Graph compiled successfully with nodes: "
        "retrieve, grade, generate, rewrite"
    )
    
    return compiled_graph


# =============================================================================
# Agent Class (High-Level Interface)
# =============================================================================

class JobMarketAgent:
    """High-level interface for the LangGraph job market agent.
    
    This agent orchestrates multi-step reasoning for job search queries,
    automatically handling retrieval, grading, query refinement, and
    answer generation.
    
    Features:
    - Autonomous workflow execution
    - Adaptive query refinement (up to 2 retries)
    - Conversation memory (last 5 turns)
    - Detailed execution metadata
    
    Example:
        >>> agent = JobMarketAgent()
        >>> result = agent.run("Find data science jobs with >10k salary")
        >>> print(result['answer'])
        "Based on 8 relevant positions..."
        >>> print(f"Sources: {len(result['sources'])}")
        Sources: 5
    """
    
    def __init__(self, settings: Optional[Settings] = None):
        """Initialize the agent.
        
        Args:
            settings: Configuration settings (GCP project, region, etc.)
        """
        self.settings = settings or Settings.load()
        self.graph = build_agent_graph(self.settings)
        logger.info("[JobMarketAgent] Initialized with compiled StateGraph")
    
    def run(
        self,
        query: str,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        filters: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Run the agent on a user query.
        
        Executes the full LangGraph workflow:
        1. Initialize state with query and history
        2. Run graph (retrieve → grade → [rewrite?] → generate)
        3. Extract final answer from state
        
        Args:
            query: User's question (e.g., "What are data science salaries?")
            conversation_history: Previous messages for context (optional)
            filters: Additional search filters (location, salary, etc.)
            
        Returns:
            Dict with:
                - answer: str (natural language response)
                - sources: List[Dict] (cited jobs with metadata)
                - metadata: Dict (execution stats, model info, etc.)
        
        Raises:
            RuntimeError: If graph execution fails
        """
        logger.info(f"[JobMarketAgent] Running query: {query[:100]}...")
        
        # Initialize state
        initial_state = {
            "query": query,
            "original_query": query,
            "messages": conversation_history or [],
            "retrieved_jobs": [],
            "graded_jobs": [],
            "final_answer": None,
            "rewrite_count": 0,
            "average_relevance_score": 0.0,
            "metadata": {
                "filters": filters or {},
                "settings": {
                    "gcp_project": self.settings.gcp_project_id,
                    "gcp_region": self.settings.gcp_region,
                }
            }
        }
        
        try:
            # Execute graph
            final_state = self.graph.invoke(initial_state)
            
            # Extract answer
            if not final_state.get("final_answer"):
                raise RuntimeError("Graph completed but no answer generated")
            
            result = final_state["final_answer"]
            
            # Add execution metadata (preserve filters from initial state)
            result["metadata"]["rewrite_count"] = final_state["rewrite_count"]
            result["metadata"]["average_relevance_score"] = final_state["average_relevance_score"]
            result["metadata"]["retrieved_count"] = len(final_state["retrieved_jobs"])
            result["metadata"]["graded_count"] = len(final_state["graded_jobs"])
            result["metadata"]["filters"] = final_state.get("metadata", {}).get("filters", {})
            
            logger.info(
                f"[JobMarketAgent] Completed successfully: "
                f"{len(result['sources'])} sources, "
                f"{final_state['rewrite_count']} rewrites"
            )
            
            return result
            
        except Exception as e:
            logger.error(f"[JobMarketAgent] Execution failed: {e}")
            raise RuntimeError(f"Agent execution failed: {e}")
    
    def stream(self, query: str, filters: Optional[Dict[str, Any]] = None):
        """Stream agent execution for real-time UI updates.
        
        Yields intermediate steps as the graph executes, allowing
        UI to show progress (e.g., "Retrieving jobs...", "Grading...").
        
        Args:
            query: User's question
            filters: Additional search filters (optional)
            
        Yields:
            Dict with step updates:
                - step: str (node name: "retrieve", "grade", "generate")
                - message: str (human-readable status)
                - data: Dict (intermediate results)
        """
        logger.info(f"[JobMarketAgent] Streaming query: {query[:100]}...")
        
        # Initialize state
        initial_state = {
            "query": query,
            "original_query": query,
            "messages": [],
            "retrieved_jobs": [],
            "graded_jobs": [],
            "final_answer": None,
            "rewrite_count": 0,
            "average_relevance_score": 0.0,
            "metadata": {"filters": filters or {}}
        }
        
        # Stream graph execution
        for output in self.graph.stream(initial_state):
            # LangGraph stream yields dict with node names as keys
            # Filter out special nodes like __start__, __end__
            node_names = [k for k in output.keys() if not k.startswith("__")]
            
            if not node_names:
                continue  # Skip special nodes
            
            node_name = node_names[0]
            state = output[node_name]
            
            yield {
                "node": node_name,
                "state": state,
                "message": f"Executing {node_name} node...",
            }
