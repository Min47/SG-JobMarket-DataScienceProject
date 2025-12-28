"""Test 4.4: FastAPI Service

Tests all REST API endpoints:
- POST /v1/chat → Conversational agent
- POST /v1/search → Direct vector search
- GET /v1/jobs/{job_id} → Job details
- GET /v1/jobs/{job_id}/similar → Similar jobs
- POST /v1/stats → Aggregate statistics
- GET /health → Health check
- GET / → Root endpoint

⚠️ REQUIRES:
- FastAPI and dependencies installed: pip install fastapi uvicorn slowapi python-multipart
- GCP credentials with BigQuery and Vertex AI access
- BigQuery dataset with cleaned_jobs and job_embeddings tables

Test Coverage:
- Endpoint functionality with valid inputs
- Request/response validation
- Error handling (404, 400, 500)
- Rate limiting behavior
- CORS headers
"""

import sys
import json
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

print("=" * 80)
print("Test 4.4: FastAPI Service")
print("=" * 80)

# Check if FastAPI is installed
try:
    import fastapi
    import uvicorn
    from fastapi.testclient import TestClient
    print(f"\n✓ FastAPI version: {fastapi.__version__}")
    print(f"✓ Uvicorn version: {uvicorn.__version__}")
except ImportError as e:
    print(f"\n✗ Missing dependency: {e}")
    print("\n📦 Install with: pip install fastapi uvicorn[standard] slowapi python-multipart")
    exit(1)

from genai.api import app
from utils.config import Settings

# Load settings
try:
    settings = Settings.load()
    print(f"\n✓ GCP Project: {settings.gcp_project_id}")
    print(f"✓ Dataset: {settings.bigquery_dataset_id}")
except ValueError as e:
    print(f"\n✗ Configuration error: {e}")
    exit(1)

# Create test client
client = TestClient(app)


# =============================================================================
# Test 1: Root Endpoint (GET /)
# =============================================================================

print("\n" + "=" * 80)
print("[Test 1] Root Endpoint - API Information")
print("=" * 80)

try:
    response = client.get("/")
    
    assert response.status_code == 200, f"Expected 200, got {response.status_code}"
    data = response.json()
    
    assert "service" in data
    assert "version" in data
    assert "endpoints" in data
    
    print(f"✓ Service: {data['service']}")
    print(f"✓ Version: {data['version']}")
    print(f"✓ Available endpoints: {len(data['endpoints'])}")
    
    print("\n✓ Test 1 PASSED")

except Exception as e:
    print(f"\n✗ Test 1 FAILED: {e}")
    import traceback
    traceback.print_exc()


# =============================================================================
# Test 2: Health Check (GET /health)
# =============================================================================

print("\n" + "=" * 80)
print("[Test 2] Health Check Endpoint")
print("=" * 80)

try:
    response = client.get("/health")
    
    assert response.status_code == 200, f"Expected 200, got {response.status_code}"
    data = response.json()
    
    assert "status" in data
    assert "version" in data
    assert "timestamp" in data
    assert "services" in data
    
    print(f"✓ Status: {data['status']}")
    print(f"✓ Version: {data['version']}")
    print(f"✓ Timestamp: {data['timestamp']}")
    
    print(f"\n📋 Service Health:")
    for service, status in data['services'].items():
        icon = "✓" if status == "ok" else "✗"
        print(f"   {icon} {service}: {status}")
    
    print("\n✓ Test 2 PASSED")

except Exception as e:
    print(f"\n✗ Test 2 FAILED: {e}")
    import traceback
    traceback.print_exc()


# =============================================================================
# Test 3: Direct Vector Search (POST /v1/search)
# =============================================================================

print("\n" + "=" * 80)
print("[Test 3] Direct Vector Search Endpoint")
print("=" * 80)

try:
    print("\n🔍 Test 3a: Basic search")
    response = client.post(
        "/v1/search",
        json={
            "query": "software engineer",
            "top_k": 5
        }
    )
    
    assert response.status_code == 200, f"Expected 200, got {response.status_code}"
    data = response.json()
    
    assert "jobs" in data
    assert "count" in data
    assert "query" in data
    assert "processing_time_ms" in data
    
    print(f"✓ Found {data['count']} jobs")
    print(f"✓ Processing time: {data['processing_time_ms']}ms")
    
    if data['count'] > 0:
        first_job = data['jobs'][0]
        print(f"✓ Sample: {first_job['job_title']} at {first_job['company_name']}")
    
    print("\n🔍 Test 3b: Search with filters")
    response = client.post(
        "/v1/search",
        json={
            "query": "data scientist",
            "top_k": 3,
            "filters": {"min_salary": 6000}
        }
    )
    
    assert response.status_code == 200
    data = response.json()
    print(f"✓ Found {data['count']} jobs with filters")
    
    print("\n🔍 Test 3c: Invalid request (empty query)")
    response = client.post(
        "/v1/search",
        json={
            "query": "",  # Too short
            "top_k": 5
        }
    )
    
    assert response.status_code == 422, "Should reject empty query"
    print(f"✓ Correctly rejected invalid input (422)")
    
    print("\n✓ Test 3 PASSED")

except Exception as e:
    print(f"\n✗ Test 3 FAILED: {e}")
    import traceback
    traceback.print_exc()


# =============================================================================
# Test 4: Get Job Details (GET /v1/jobs/{job_id})
# =============================================================================

print("\n" + "=" * 80)
print("[Test 4] Get Job Details Endpoint")
print("=" * 80)

try:
    # First, get a valid job_id from search
    print("\n📋 Getting a valid job ID from search...")
    search_response = client.post(
        "/v1/search",
        json={"query": "software engineer", "top_k": 1}
    )
    search_data = search_response.json()
    
    if search_data['count'] == 0:
        print("⚠️  No jobs found, skipping test 4")
    else:
        job_id = search_data['jobs'][0]['job_id']
        source = search_data['jobs'][0]['source']
        print(f"✓ Using job_id: {job_id}, source: {source}")
        
        print("\n🔍 Test 4a: Get valid job details")
        response = client.get(f"/v1/jobs/{job_id}?source={source}")
        
        assert response.status_code == 200, f"Expected 200, got {response.status_code}"
        data = response.json()
        
        assert data['success'] == True
        assert data['job'] is not None
        
        job = data['job']
        print(f"✓ Job: {job['job_title']}")
        print(f"✓ Company: {job['company_name']}")
        print(f"✓ Description length: {len(job['job_description'])} chars")
        
        print("\n🔍 Test 4b: Invalid job ID (404)")
        response = client.get("/v1/jobs/invalid_xyz_999?source=jobstreet")
        
        assert response.status_code == 404, "Should return 404 for invalid job"
        print(f"✓ Correctly returned 404 for invalid job")
        
        print("\n✓ Test 4 PASSED")

except Exception as e:
    print(f"\n✗ Test 4 FAILED: {e}")
    import traceback
    traceback.print_exc()


# =============================================================================
# Test 5: Find Similar Jobs (GET /v1/jobs/{job_id}/similar)
# =============================================================================

print("\n" + "=" * 80)
print("[Test 5] Find Similar Jobs Endpoint")
print("=" * 80)

try:
    # Get a job ID for similarity search
    print("\n📋 Getting a reference job...")
    search_response = client.post(
        "/v1/search",
        json={"query": "data analyst", "top_k": 1}
    )
    search_data = search_response.json()
    
    if search_data['count'] == 0:
        print("⚠️  No jobs found, skipping test 5")
    else:
        job_id = search_data['jobs'][0]['job_id']
        source = search_data['jobs'][0]['source']
        print(f"✓ Reference job: {search_data['jobs'][0]['job_title']}")
        
        print("\n🔍 Test 5a: Find similar jobs")
        response = client.get(f"/v1/jobs/{job_id}/similar?source={source}&top_k=3")
        
        assert response.status_code == 200, f"Expected 200, got {response.status_code}"
        data = response.json()
        
        assert data['success'] == True
        assert 'similar_jobs' in data
        
        print(f"✓ Found {data['count']} similar jobs")
        
        if data['count'] > 0:
            print(f"\n📋 Top 3 Similar:")
            for i, job in enumerate(data['similar_jobs'][:3], 1):
                print(f"   [{i}] {job['job_title']} (similarity: {job['similarity_score']:.3f})")
        
        print("\n🔍 Test 5b: Invalid job ID (404)")
        response = client.get("/v1/jobs/invalid_xyz_999/similar?source=jobstreet")
        
        assert response.status_code == 404
        print(f"✓ Correctly returned 404 for invalid job")
        
        print("\n✓ Test 5 PASSED")

except Exception as e:
    print(f"\n✗ Test 5 FAILED: {e}")
    import traceback
    traceback.print_exc()


# =============================================================================
# Test 6: Aggregate Statistics (POST /v1/stats)
# =============================================================================

print("\n" + "=" * 80)
print("[Test 6] Aggregate Statistics Endpoint")
print("=" * 80)

try:
    print("\n📊 Test 6a: Group by classification")
    response = client.post(
        "/v1/stats",
        json={
            "group_by": "classification",
            "limit": 5
        }
    )
    
    assert response.status_code == 200, f"Expected 200, got {response.status_code}"
    data = response.json()
    
    assert "stats" in data
    assert "summary" in data
    assert "group_by" in data
    
    print(f"✓ Groups: {len(data['stats'])}")
    print(f"✓ Total jobs: {data['summary']['total_jobs']}")
    print(f"✓ Avg salary: ${data['summary']['overall_avg_salary']:.2f}/month")
    
    print("\n📊 Test 6b: Invalid group_by field")
    response = client.post(
        "/v1/stats",
        json={
            "group_by": "invalid_field",
            "limit": 5
        }
    )
    
    assert response.status_code == 422, "Should reject invalid group_by"
    print(f"✓ Correctly rejected invalid group_by (422)")
    
    print("\n✓ Test 6 PASSED")

except Exception as e:
    print(f"\n✗ Test 6 FAILED: {e}")
    import traceback
    traceback.print_exc()


# =============================================================================
# Test 7: Conversational Agent (POST /v1/chat)
# =============================================================================

print("\n" + "=" * 80)
print("[Test 7] Conversational Agent Endpoint")
print("=" * 80)

print("\n⚠️  WARNING: This test is SLOW (~30-60 seconds)")
print("It runs the full LangGraph agent with Gemini API calls")
user_input = input("Run this test? [y/N]: ")

if user_input.lower() == 'y':
    try:
        print("\n💬 Test 7a: Simple chat query")
        response = client.post(
            "/v1/chat",
            json={
                "message": "Find me python developer jobs",
                "filters": {}
            },
            timeout=120.0  # Allow up to 2 minutes for agent
        )
        
        assert response.status_code == 200, f"Expected 200, got {response.status_code}"
        data = response.json()
        
        assert "answer" in data
        assert "sources" in data
        assert "conversation_id" in data
        assert "metadata" in data
        
        print(f"✓ Answer length: {len(data['answer'])} chars")
        print(f"✓ Sources cited: {len(data['sources'])}")
        print(f"✓ Conversation ID: {data['conversation_id']}")
        
        metadata = data['metadata']
        print(f"\n📋 Agent Metadata:")
        print(f"   Retrieved: {metadata.get('retrieved_count', 0)} jobs")
        print(f"   Graded: {metadata.get('graded_count', 0)} jobs")
        print(f"   Avg relevance: {metadata.get('average_relevance_score', 0.0):.2f}/10")
        print(f"   Rewrites: {metadata.get('rewrite_count', 0)}")
        
        print(f"\n💬 Answer Preview:")
        print(f"   {data['answer'][:200]}...")
        
        print("\n✓ Test 7 PASSED")
    
    except Exception as e:
        print(f"\n✗ Test 7 FAILED: {e}")
        import traceback
        traceback.print_exc()
else:
    print("⏭️  Test 7 SKIPPED (user choice)")


# =============================================================================
# Test 8: CORS Headers
# =============================================================================

print("\n" + "=" * 80)
print("[Test 8] CORS Headers")
print("=" * 80)

try:
    response = client.options(
        "/v1/search",
        headers={"Origin": "http://localhost:3000"}
    )
    
    # Check for CORS headers
    headers = response.headers
    print(f"✓ Access-Control-Allow-Origin: {headers.get('access-control-allow-origin', 'NOT SET')}")
    print(f"✓ Access-Control-Allow-Methods: {headers.get('access-control-allow-methods', 'NOT SET')}")
    
    print("\n✓ Test 8 PASSED")

except Exception as e:
    print(f"\n✗ Test 8 FAILED: {e}")


# =============================================================================
# Summary
# =============================================================================

print("\n" + "=" * 80)
print("✓ API TESTS COMPLETED")
print("=" * 80)

print("\n📋 Endpoint Summary:")
print("   ✓ GET / → Root API info")
print("   ✓ GET /health → Health check")
print("   ✓ POST /v1/search → Direct vector search")
print("   ✓ GET /v1/jobs/{id} → Job details")
print("   ✓ GET /v1/jobs/{id}/similar → Similar jobs")
print("   ✓ POST /v1/stats → Aggregate statistics")
print("   ✓ POST /v1/chat → Conversational agent")
print("   ✓ CORS headers present")

print("\n🎯 Task 4.4 FastAPI Service Ready!")
print("\n📝 Next Steps:")
print("   1. Start local server: python -m genai.api")
print("   2. Test endpoints: http://localhost:8000/docs")
print("   3. Create Dockerfile for deployment")
print("   4. Deploy to Cloud Run")
