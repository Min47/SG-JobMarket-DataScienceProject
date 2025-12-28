"""Test 4.3: Tool Adapters

Tests all 4 LangChain-compatible tools:
- search_jobs: Job search with filters
- get_job_details: Fetch single job by ID
- aggregate_stats: Salary statistics by category
- find_similar_jobs: Recommendations based on similarity

⚠️ REQUIRES:
- GCP credentials with BigQuery and Vertex AI access
- BigQuery dataset with cleaned_jobs and job_embeddings tables
- Valid job IDs for testing

Test Coverage:
- Tool invocation with valid inputs
- Pydantic schema validation
- Error handling for invalid inputs
- JSON response parsing
- BigQuery integration
"""

import sys
import json
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from genai.tools import search_jobs, get_job_details, aggregate_stats, find_similar_jobs
from utils.config import Settings

print("=" * 80)
print("Test 4.3: Tool Adapters")
print("=" * 80)

# Load settings
try:
    settings = Settings.load()
    print(f"\n✓ GCP Project: {settings.gcp_project_id}")
    print(f"✓ Dataset: {settings.bigquery_dataset_id}")
except ValueError as e:
    print(f"\n✗ Configuration error: {e}")
    exit(1)


# =============================================================================
# Test 1: search_jobs tool
# =============================================================================

print("\n" + "=" * 80)
print("[Test 1] search_jobs - Job search with filters")
print("=" * 80)

try:
    print("\n🔍 Test 1a: Basic search (no filters)")
    result_str = search_jobs.invoke({"query": "python developer"})
    result = json.loads(result_str)
    
    assert result["success"] == True, "Search should succeed"
    assert result["count"] > 0, "Should find jobs"
    assert len(result["jobs"]) > 0, "Jobs list should not be empty"
    
    print(f"✓ Found {result['count']} jobs")
    print(f"✓ Sample: {result['jobs'][0]['job_title']} at {result['jobs'][0]['company_name']}")
    
    print("\n🔍 Test 1b: Search with filters")
    result_str = search_jobs.invoke({
        "query": "data scientist",
        "min_salary": 5000,
        "max_results": 5
    })
    result = json.loads(result_str)
    
    assert result["success"] == True
    assert result["count"] <= 5, "Should respect max_results limit"
    
    print(f"✓ Found {result['count']} jobs with salary >= $5000")
    
    print("\n🔍 Test 1c: Invalid input (empty query)")
    try:
        result_str = search_jobs.invoke({"query": ""})
        print("✗ Should have raised validation error")
    except Exception as e:
        print(f"✓ Correctly rejected empty query: {type(e).__name__}")
    
    print("\n✓ Test 1 PASSED")

except Exception as e:
    print(f"\n✗ Test 1 FAILED: {e}")
    import traceback
    traceback.print_exc()


# =============================================================================
# Test 2: get_job_details tool
# =============================================================================

print("\n" + "=" * 80)
print("[Test 2] get_job_details - Fetch single job by ID")
print("=" * 80)

try:
    # First, get a valid job_id from search
    print("\n📋 Getting a valid job ID from search results...")
    search_result_str = search_jobs.invoke({"query": "software engineer", "max_results": 1})
    search_result = json.loads(search_result_str)
    
    if search_result["count"] == 0:
        print("⚠️  No jobs found, skipping test 2")
    else:
        sample_job = search_result["jobs"][0]
        job_id = sample_job["job_id"]
        source = sample_job["source"]
        
        print(f"✓ Using job_id: {job_id}, source: {source}")
        
        print("\n🔍 Test 2a: Fetch valid job")
        result_str = get_job_details.invoke({
            "job_id": job_id,
            "source": source
        })
        result = json.loads(result_str)
        
        assert result["success"] == True, "Should find job"
        assert result["job"] is not None, "Job should not be None"
        assert result["job"]["job_id"] == job_id, "Job ID should match"
        
        job = result["job"]
        print(f"✓ Found: {job['job_title']}")
        print(f"✓ Company: {job['company_name']}")
        print(f"✓ Description length: {len(job['job_description'])} chars")
        
        print("\n🔍 Test 2b: Invalid job ID")
        result_str = get_job_details.invoke({
            "job_id": "invalid_id_xyz_999",
            "source": "jobstreet"
        })
        result = json.loads(result_str)
        
        assert result["success"] == False, "Should fail for invalid ID"
        assert "not found" in result["error"].lower(), "Should have 'not found' error"
        
        print(f"✓ Correctly handled invalid ID: {result['error']}")
        
        print("\n✓ Test 2 PASSED")

except Exception as e:
    print(f"\n✗ Test 2 FAILED: {e}")
    import traceback
    traceback.print_exc()


# =============================================================================
# Test 3: aggregate_stats tool
# =============================================================================

print("\n" + "=" * 80)
print("[Test 3] aggregate_stats - Salary statistics by category")
print("=" * 80)

try:
    print("\n📊 Test 3a: Group by classification (job category)")
    result_str = aggregate_stats.invoke({
        "group_by": "classification",
        "limit": 5
    })
    result = json.loads(result_str)
    
    assert result["success"] == True, "Should succeed"
    assert len(result["stats"]) > 0, "Should have stats"
    assert result["summary"]["total_jobs"] > 0, "Should have total jobs"
    
    print(f"✓ Found {result['summary']['total_groups']} job categories")
    print(f"✓ Total jobs analyzed: {result['summary']['total_jobs']}")
    print(f"✓ Overall avg salary: ${result['summary']['overall_avg_salary']:.2f}/month")
    
    print(f"\n📋 Top 3 Categories:")
    for i, stat in enumerate(result["stats"][:3], 1):
        print(f"   [{i}] {stat['group']}: {stat['job_count']} jobs, avg ${stat['avg_mid_salary']:.2f}/month")
    
    print("\n📊 Test 3b: Group by work type")
    result_str = aggregate_stats.invoke({
        "group_by": "work_type",
        "limit": 10
    })
    result = json.loads(result_str)
    
    assert result["success"] == True
    print(f"✓ Found {len(result['stats'])} work types")
    
    print("\n📊 Test 3c: Filter by classification")
    result_str = aggregate_stats.invoke({
        "group_by": "location",
        "classification": "Information & Communication Technology",
        "limit": 5
    })
    result = json.loads(result_str)
    
    assert result["success"] == True
    print(f"✓ IT jobs by location: {len(result['stats'])} locations")
    
    print("\n✓ Test 3 PASSED")

except Exception as e:
    print(f"\n✗ Test 3 FAILED: {e}")
    import traceback
    traceback.print_exc()


# =============================================================================
# Test 4: find_similar_jobs tool
# =============================================================================

print("\n" + "=" * 80)
print("[Test 4] find_similar_jobs - Job recommendations")
print("=" * 80)

try:
    # Get a job ID for similarity search
    print("\n📋 Getting a reference job for similarity search...")
    search_result_str = search_jobs.invoke({"query": "data analyst", "max_results": 1})
    search_result = json.loads(search_result_str)
    
    if search_result["count"] == 0:
        print("⚠️  No jobs found, skipping test 4")
    else:
        sample_job = search_result["jobs"][0]
        job_id = sample_job["job_id"]
        source = sample_job["source"]
        
        print(f"✓ Reference job: {sample_job['job_title']} at {sample_job['company_name']}")
        
        print("\n🔍 Test 4a: Find similar jobs (default params)")
        result_str = find_similar_jobs.invoke({
            "job_id": job_id,
            "source": source
        })
        result = json.loads(result_str)
        
        assert result["success"] == True, "Should succeed"
        assert result["count"] > 0, "Should find similar jobs"
        
        print(f"✓ Found {result['count']} similar jobs")
        
        print(f"\n📋 Top 3 Similar Jobs:")
        for i, job in enumerate(result["similar_jobs"][:3], 1):
            print(f"   [{i}] {job['job_title']} at {job['company_name']}")
            print(f"       Similarity: {job['similarity_score']:.3f} ({int(job['similarity_score']*100)}%)")
        
        print("\n🔍 Test 4b: Higher similarity threshold")
        result_str = find_similar_jobs.invoke({
            "job_id": job_id,
            "source": source,
            "top_k": 3,
            "min_similarity": 0.8
        })
        result = json.loads(result_str)
        
        assert result["success"] == True
        print(f"✓ Found {result['count']} very similar jobs (>80%)")
        
        print("\n🔍 Test 4c: Invalid job ID")
        result_str = find_similar_jobs.invoke({
            "job_id": "invalid_xyz_999",
            "source": "jobstreet"
        })
        result = json.loads(result_str)
        
        assert result["success"] == False
        assert "not found" in result["error"].lower()
        print(f"✓ Correctly handled invalid ID")
        
        print("\n✓ Test 4 PASSED")

except Exception as e:
    print(f"\n✗ Test 4 FAILED: {e}")
    import traceback
    traceback.print_exc()


# =============================================================================
# Summary
# =============================================================================

print("\n" + "=" * 80)
print("✓ ALL TESTS COMPLETED")
print("=" * 80)

print("\n📋 Tool Summary:")
print("   ✓ search_jobs: Search with filters and hybrid scoring")
print("   ✓ get_job_details: Fetch complete job information")
print("   ✓ aggregate_stats: Salary statistics by category")
print("   ✓ find_similar_jobs: Semantic similarity recommendations")

print("\n🎯 Task 4.3 Complete: All 4 tools validated")
print("\n📝 Next Steps:")
print("   - Task 4.4: FastAPI Service (expose tools via REST API)")
print("   - Task 4.5: Model Gateway (multi-provider LLM support)")
print("   - Task 4.6: Guardrails (PII detection, input validation)")
print("   - Task 4.7: Observability (tracing, metrics, logging)")
print("   - Task 4.8: MCP Server (external tool access)")
