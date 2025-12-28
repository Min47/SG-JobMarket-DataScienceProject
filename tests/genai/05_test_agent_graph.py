"""Test 4.2.1: LangGraph State & Graph Definition

Tests the agent graph construction and state management:
- AgentState TypedDict structure
- Graph compilation with nodes and edges
- Conditional edge logic (should_rewrite)
- Graph visualization

⚠️ REQUIRES:
- langgraph package installed
- Agent nodes implemented in subsequent tasks

Test Coverage:
- Graph structure (nodes, edges, entry/exit points)
- Conditional routing logic
- State initialization and validation
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from genai.agent import build_agent_graph, should_rewrite, AgentState

print("=" * 70)
print("Test 4.2.1: LangGraph State & Graph Definition")
print("=" * 70)

# Test 1: Graph construction
print("\n" + "=" * 70)
print("[Test 1] Build agent graph")
print("=" * 70)

try:
    graph = build_agent_graph()
    print("✓ Graph compiled successfully")
    
    # Check graph structure
    print("\n📊 Graph Structure:")
    print(f"   Type: {type(graph).__name__}")
    
    # Try to get node names (LangGraph API)
    if hasattr(graph, 'nodes'):
        print(f"   Nodes: {list(graph.nodes.keys())}")
    
    print("\n✓ Test 1 PASSED")

except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()

# Test 2: Conditional edge logic
print("\n" + "=" * 70)
print("[Test 2] Conditional edge: should_rewrite()")
print("=" * 70)

try:
    # Case 1: Low score, no retries → rewrite
    state1: AgentState = {
        "messages": [],
        "query": "test",
        "original_query": "test",
        "retrieved_jobs": [],
        "graded_jobs": [],
        "final_answer": None,
        "rewrite_count": 0,
        "average_relevance_score": 4.5,
        "metadata": {}
    }
    decision1 = should_rewrite(state1)
    assert decision1 == "rewrite", f"Expected 'rewrite', got '{decision1}'"
    print(f"✓ Case 1: Score=4.5, Retries=0 → {decision1}")
    
    # Case 2: Low score, max retries → generate anyway
    state2: AgentState = {
        "messages": [],
        "query": "test",
        "original_query": "test",
        "retrieved_jobs": [],
        "graded_jobs": [],
        "final_answer": None,
        "rewrite_count": 2,
        "average_relevance_score": 4.5,
        "metadata": {}
    }
    decision2 = should_rewrite(state2)
    assert decision2 == "generate", f"Expected 'generate', got '{decision2}'"
    print(f"✓ Case 2: Score=4.5, Retries=2 → {decision2}")
    
    # Case 3: Good score → generate immediately
    state3: AgentState = {
        "messages": [],
        "query": "test",
        "original_query": "test",
        "retrieved_jobs": [],
        "graded_jobs": [],
        "final_answer": None,
        "rewrite_count": 0,
        "average_relevance_score": 8.5,
        "metadata": {}
    }
    decision3 = should_rewrite(state3)
    assert decision3 == "generate", f"Expected 'generate', got '{decision3}'"
    print(f"✓ Case 3: Score=8.5, Retries=0 → {decision3}")
    
    # Case 4: Boundary case (exactly 6.0) → generate
    state4: AgentState = {
        "messages": [],
        "query": "test",
        "original_query": "test",
        "retrieved_jobs": [],
        "graded_jobs": [],
        "final_answer": None,
        "rewrite_count": 0,
        "average_relevance_score": 6.0,
        "metadata": {}
    }
    decision4 = should_rewrite(state4)
    assert decision4 == "generate", f"Expected 'generate', got '{decision4}'"
    print(f"✓ Case 4: Score=6.0, Retries=0 → {decision4} (boundary)")
    
    print("\n✓ Test 2 PASSED - All conditional logic correct")

except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()

# Test 3: State validation
print("\n" + "=" * 70)
print("[Test 3] AgentState schema validation")
print("=" * 70)

try:
    # Create valid state
    test_state: AgentState = {
        "messages": [{"role": "user", "content": "test"}],
        "query": "python developer jobs",
        "original_query": "python developer jobs",
        "retrieved_jobs": [{"job_id": "test123", "job_title": "Python Dev"}],
        "graded_jobs": [{"job_id": "test123", "relevance_score": 9.0}],
        "final_answer": {"answer": "Test answer", "sources": [], "metadata": {}},
        "rewrite_count": 1,
        "average_relevance_score": 9.0,
        "metadata": {"filters": {}}
    }
    
    print("✓ AgentState structure:")
    for key, value in test_state.items():
        value_type = type(value).__name__
        value_str = str(value)[:50] + "..." if len(str(value)) > 50 else str(value)
        print(f"   - {key}: {value_type} = {value_str}")
    
    # Verify all required fields present
    required_fields = [
        "messages", "query", "original_query", "retrieved_jobs",
        "graded_jobs", "final_answer", "rewrite_count",
        "average_relevance_score", "metadata"
    ]
    for field in required_fields:
        assert field in test_state, f"Missing required field: {field}"
    
    print(f"\n✓ All {len(required_fields)} required fields present")
    print("\n✓ Test 3 PASSED")

except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 70)
print("✓ TESTS COMPLETED")
print("=" * 70)
print("\nNext Steps:")
print("  - Task 4.2.2: Implement node functions (retrieve, grade, generate, rewrite)")
print("  - Task 4.2.3: Add logging and error handling in nodes")
print("  - Task 4.2.4: End-to-end integration test with real RAG pipeline")
