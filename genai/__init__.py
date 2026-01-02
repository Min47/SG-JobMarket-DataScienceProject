"""GenAI module for RAG, LangGraph agents, and MCP server.

This module provides:
- RAG pipeline using BigQuery Vector Search + Gemini Pro
- LangGraph orchestration for multi-step reasoning
- MCP Server for external tool access (Claude/Cursor integration)
"""

from genai.rag import retrieve_jobs, generate_answer
from genai.agent import JobMarketAgent

__all__ = [
    "retrieve_jobs",
    "generate_answer",
    "JobMarketAgent",
]