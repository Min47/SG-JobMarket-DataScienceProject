"""Test 4.2.2: Agent Node Implementations

Tests the individual node functions and their integration with RAG pipeline:
- retrieve_node: Vector search with hybrid scoring
- grade_node: Gemini relevance scoring and filtering
- generate_node: Natural language answer generation
- rewrite_node: Query reformulation with LLM

⚠️ REQUIRES:
- GCP credentials with Vertex AI API enabled
- BigQuery dataset with embeddings and cleaned_jobs
- Completed Tasks 4.1.x (RAG pipeline)

Test Coverage:
- Node execution with real data
- State updates and metadata tracking
- Error handling and fallbacks
- End-to-end workflow simulation
"""

import sys
from pathlib import Path
from typing import Dict, Any

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from genai.agent import retrieve_node, grade_node, generate_node, rewrite_node, AgentState
from utils.config import Settings

print("=" * 70)
print("Test 4.2.2: Agent Node Implementations")
print("=" * 70)

# Load settings
try:
    settings = Settings.load()
    print(f"\n✓ GCP Project: {settings.gcp_project_id}")
    print(f"✓ Region: {settings.gcp_region}")
except ValueError as e:
    print(f"\n✗ Configuration error: {e}")
    exit(1)

# Test 1: retrieve_node
print("\n" + "=" * 70)
print("[Test 1] retrieve_node - Vector search with hybrid scoring")
print("=" * 70)

try:
    # Initialize state
    state: AgentState = {
        "messages": [],
        "query": "python developer jobs",
        "original_query": "python developer jobs",
        "retrieved_jobs": [],
        "graded_jobs": [],
        "final_answer": None,
        "rewrite_count": 0,
        "average_relevance_score": 0.0,
        "metadata": {"filters": {}}
    }
    
    print(f"\n📊 Input query: '{state['query']}'")
    
    # Execute retrieve node
    state = retrieve_node(state)
    
    # Validate results
    assert "retrieved_jobs" in state, "Missing retrieved_jobs in state"
    assert len(state["retrieved_jobs"]) > 0, "No jobs retrieved"
    
    retrieved_count = len(state["retrieved_jobs"])
    print(f"✓ Retrieved {retrieved_count} jobs")
    
    # Check job structure
    sample_job = state["retrieved_jobs"][0]
    required_fields = ["job_id", "job_title", "company_name", "vector_distance"]
    for field in required_fields:
        assert field in sample_job, f"Missing field: {field}"
    
    print(f"✓ Job structure valid")
    
    # Show sample
    print(f"\n📋 Sample job:")
    print(f"   Title: {sample_job['job_title']}")
    print(f"   Company: {sample_job['company_name']}")
    print(f"   Distance: {sample_job['vector_distance']:.4f}")
    
    # Check metadata
    assert state["metadata"]["retrieval_count"] == retrieved_count
    print(f"✓ Metadata updated: retrieval_count={retrieved_count}")
    
    print("\n✓ Test 1 PASSED")

except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()

# Test 2: grade_node
print("\n" + "=" * 70)
print("[Test 2] grade_node - Gemini relevance scoring")
print("=" * 70)

try:
    # State should have retrieved_jobs from Test 1
    print(f"\n🤖 Grading {len(state['retrieved_jobs'])} jobs...")
    
    # Execute grade node
    state = grade_node(state)
    
    # Validate results
    assert "graded_jobs" in state, "Missing graded_jobs in state"
    assert "average_relevance_score" in state, "Missing average_relevance_score"
    
    graded_count = len(state["graded_jobs"])
    avg_score = state["average_relevance_score"]
    
    print(f"✓ Graded: {len(state['retrieved_jobs'])} → {graded_count} jobs passed")
    print(f"✓ Average score: {avg_score:.2f}/10")
    
    # Check score structure
    if graded_count > 0:
        sample_graded = state["graded_jobs"][0]
        assert "relevance_score" in sample_graded, "Missing relevance_score"
        assert "relevance_explanation" in sample_graded, "Missing relevance_explanation"
        
        print(f"\n📊 Sample graded job:")
        print(f"   Title: {sample_graded['job_title']}")
        print(f"   Score: {sample_graded['relevance_score']:.1f}/10")
        print(f"   Reason: {sample_graded['relevance_explanation'][:100]}...")
    
    # Check metadata
    assert state["metadata"]["graded_count"] == graded_count
    print(f"✓ Metadata updated: graded_count={graded_count}")
    
    print("\n✓ Test 2 PASSED")

except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()

# Test 3: generate_node
print("\n" + "=" * 70)
print("[Test 3] generate_node - Natural language answer")
print("=" * 70)

try:
    # State should have graded_jobs from Test 2
    print(f"\n✨ Generating answer from {len(state['graded_jobs'])} context jobs...")
    
    # Execute generate node
    state = generate_node(state)
    
    # Validate results
    assert "final_answer" in state, "Missing final_answer in state"
    assert state["final_answer"] is not None, "final_answer is None"
    
    answer_dict = state["final_answer"]
    assert "answer" in answer_dict, "Missing 'answer' field"
    assert "sources" in answer_dict, "Missing 'sources' field"
    assert "metadata" in answer_dict, "Missing 'metadata' field"
    
    answer_text = answer_dict["answer"]
    sources = answer_dict["sources"]
    
    print(f"✓ Answer generated: {len(answer_text)} chars")
    print(f"✓ Sources: {len(sources)} jobs cited")
    
    print(f"\n📝 Generated answer (first 300 chars):")
    print(f"{answer_text[:300]}...")
    
    print(f"\n📚 Citations:")
    for src in sources[:3]:
        print(f"   [{src['number']}] {src['job_title']} - {src['company_name']}")
    
    print("\n✓ Test 3 PASSED")

except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()

# Test 4: rewrite_node
print("\n" + "=" * 70)
print("[Test 4] rewrite_node - Query reformulation")
print("=" * 70)

try:
    # Create new state with vague query
    rewrite_state: AgentState = {
        "messages": [],
        "query": "ML jobs",  # Vague query
        "original_query": "ML jobs",
        "retrieved_jobs": [],
        "graded_jobs": [],
        "final_answer": None,
        "rewrite_count": 0,
        "average_relevance_score": 4.5,  # Low score triggers rewrite
        "metadata": {}
    }
    
    original_query = rewrite_state["query"]
    print(f"\n🔄 Original query: '{original_query}'")
    print(f"   (Low score: {rewrite_state['average_relevance_score']:.1f}/10)")
    
    # Execute rewrite node
    rewrite_state = rewrite_node(rewrite_state)
    
    # Validate results
    rewritten_query = rewrite_state["query"]
    rewrite_count = rewrite_state["rewrite_count"]
    
    print(f"✓ Rewritten query: '{rewritten_query}'")
    print(f"✓ Rewrite count: {rewrite_count}")
    
    # Check that query changed (unless error)
    if "rewrite_error" not in rewrite_state["metadata"]:
        assert rewritten_query != original_query, "Query should be different"
        assert len(rewritten_query) >= 3, "Rewritten query too short"
        print(f"✓ Query successfully reformulated")
    else:
        print(f"⚠️  Rewrite failed, kept original: {rewrite_state['metadata']['rewrite_error']}")
    
    print("\n✓ Test 4 PASSED")

except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()

# Test 5: End-to-end workflow simulation
print("\n" + "=" * 70)
print("[Test 5] End-to-end workflow simulation")
print("=" * 70)

try:
    # Simulate full graph execution manually
    workflow_state: AgentState = {
        "messages": [],
        "query": "data scientist salary range",
        "original_query": "data scientist salary range",
        "retrieved_jobs": [],
        "graded_jobs": [],
        "final_answer": None,
        "rewrite_count": 0,
        "average_relevance_score": 0.0,
        "metadata": {"filters": {"min_salary": 5000}}
    }
    
    print(f"\n🚀 Starting workflow with query: '{workflow_state['query']}'")
    print(f"   Filters: {workflow_state['metadata']['filters']}")
    
    # Step 1: Retrieve
    print("\n[Step 1] Retrieving jobs...")
    workflow_state = retrieve_node(workflow_state)
    print(f"   → {len(workflow_state['retrieved_jobs'])} jobs retrieved")
    
    # Step 2: Grade
    print("\n[Step 2] Grading relevance...")
    workflow_state = grade_node(workflow_state)
    print(f"   → {len(workflow_state['graded_jobs'])} jobs passed (avg: {workflow_state['average_relevance_score']:.2f}/10)")
    
    # Step 3: Decision (simulate)
    if workflow_state['average_relevance_score'] < 6.0 and workflow_state['rewrite_count'] < 2:
        print(f"\n[Decision] Low score → Rewrite query")
        workflow_state = rewrite_node(workflow_state)
        print(f"   → New query: '{workflow_state['query']}'")
        # In real graph, would loop back to retrieve
    else:
        print(f"\n[Decision] Good score → Generate answer")
    
    # Step 4: Generate
    print("\n[Step 3] Generating answer...")
    workflow_state = generate_node(workflow_state)
    print(f"   → Answer: {len(workflow_state['final_answer']['answer'])} chars")
    print(f"   → Sources: {len(workflow_state['final_answer']['sources'])} jobs")
    
    print(f"\n✅ Workflow completed successfully!")
    print(f"\n📊 Final state summary:")
    print(f"   - Retrieved: {len(workflow_state['retrieved_jobs'])} jobs")
    print(f"   - Graded: {len(workflow_state['graded_jobs'])} jobs (avg: {workflow_state['average_relevance_score']:.2f}/10)")
    print(f"   - Answer: {len(workflow_state['final_answer']['answer'])} chars")
    print(f"   - Rewrites: {workflow_state['rewrite_count']}")
    
    print("\n✓ Test 5 PASSED")

except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 70)
print("✓ ALL TESTS COMPLETED")
print("=" * 70)
print("\nNext Steps:")
print("  - Task 4.2.3: Full graph execution test with JobMarketAgent.run()")
print("  - Test edge cases: empty results, errors, max retries")
print("  - Performance benchmarking: measure latency per node")
print("  - Task 4.3: Implement tool adapters for extended functionality")
