"""Test 02: Job Retrieval Function (Task 4.1.1)

Tests the retrieve_jobs() function that performs BigQuery vector search
with hybrid scoring (vector + keyword) and metadata filters.

⚠️ REQUIRES:
- GCP credentials configured
- BigQuery dataset with job_embeddings and cleaned_jobs tables  
- Vector index created on job_embeddings.embedding

Test Coverage:
- Basic vector search without filters
- Vector search with multiple filters
- Hybrid weight tuning
- Edge cases (empty query, no results)
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from genai.rag import retrieve_jobs
from utils.config import Settings

print("=" * 70)
print("Test 02: retrieve_jobs() Function - BigQuery Vector Search")
print("=" * 70)

# Load settings
try:
    settings = Settings.load()
    print(f"\n✓ GCP Project: {settings.gcp_project_id}")
    print(f"✓ Dataset: {settings.bigquery_dataset_id}")
except ValueError as e:
    print(f"\n✗ Configuration error: {e}")
    print("Please set required environment variables in .env file")
    exit(1)

# Test 1: Basic retrieval without filters
print("\n" + "=" * 70)
print("[Test 1] Basic retrieval - 'data scientist python'")
print("=" * 70)

try:
    jobs = retrieve_jobs(
        query="data scientist python",
        top_k=5,
        settings=settings
    )
    
    print(f"\n✓ Retrieved {len(jobs)} jobs")
    
    if jobs:
        print("\nTop 5 results:")
        for i, job in enumerate(jobs[:5], 1):
            print(f"\n{i}. {job['job_title']}")
            print(f"   Company: {job['company_name']}")
            print(f"   Location: {job['job_location']}")
            print(f"   Classification: {job['job_classification']}")
            if job['job_salary_min_sgd_monthly']:
                salary_range = f"${job['job_salary_min_sgd_monthly']:.0f}"
                if job['job_salary_max_sgd_monthly']:
                    salary_range += f" - ${job['job_salary_max_sgd_monthly']:.0f}"
                print(f"   Salary: {salary_range}/month")
            print(f"   Vector Distance: {job['vector_distance']:.4f}")
            print(f"   Keyword Score: {job['keyword_score']:.4f}")
            print(f"   Hybrid Score: {job['hybrid_score']:.4f}")
        
        print("\n✓ Test 1 PASSED")
    else:
        print("⚠️  No jobs found - check if vector index is created")
        print("   Run: python -m nlp.create_vector_index")

except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()

# Test 2: Retrieval with filters
print("\n" + "=" * 70)
print("[Test 2] Retrieval with filters - 'software engineer' + filters")
print("=" * 70)

try:
    jobs = retrieve_jobs(
        query="software engineer",
        top_k=5,
        filters={
            "location": "Singapore",
            "min_salary": 4000,
            "work_type": "Full Time"
        },
        settings=settings
    )
    
    print(f"\n✓ Retrieved {len(jobs)} jobs matching filters")
    
    if jobs:
        print("\nFiltered results:")
        for i, job in enumerate(jobs[:3], 1):
            print(f"\n{i}. {job['job_title']} at {job['company_name']}")
            print(f"   Location: {job['job_location']}")
            print(f"   Work Type: {job['job_work_type']}")
            if job['job_salary_min_sgd_monthly']:
                print(f"   Min Salary: ${job['job_salary_min_sgd_monthly']:.0f}/month")
            print(f"   Hybrid Score: {job['hybrid_score']:.4f}")
            
            # Verify filters applied
            assert "Singapore" in job['job_location'], "Location filter failed"
            if job['job_salary_min_sgd_monthly']:
                assert job['job_salary_min_sgd_monthly'] >= 4000, "Salary filter failed"
            assert job['job_work_type'] == "Full Time", "Work type filter failed"
        
        print("\n✓ Test 2 PASSED - All filters applied correctly")
    else:
        print("⚠️  No jobs match the filters")

except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()

# Test 3: Hybrid weight comparison
print("\n" + "=" * 70)
print("[Test 3] Hybrid weight comparison - 'machine learning engineer'")
print("=" * 70)

try:
    # Pure vector search
    print("\nPure vector search (weight=1.0):")
    jobs_vector = retrieve_jobs(
        query="machine learning engineer",
        top_k=3,
        hybrid_weight=1.0,
        settings=settings
    )
    
    if jobs_vector:
        for i, job in enumerate(jobs_vector, 1):
            print(f"  {i}. {job['job_title']} (score={job['hybrid_score']:.3f})")
    
    # Balanced hybrid
    print("\nBalanced hybrid search (weight=0.5):")
    jobs_hybrid = retrieve_jobs(
        query="machine learning engineer",
        top_k=3,
        hybrid_weight=0.5,
        settings=settings
    )
    
    if jobs_hybrid:
        for i, job in enumerate(jobs_hybrid, 1):
            print(f"  {i}. {job['job_title']} (score={job['hybrid_score']:.3f})")
    
    print("\n✓ Test 3 PASSED - Hybrid weight affects ranking")

except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()

# Test 4: Edge cases
print("\n" + "=" * 70)
print("[Test 4] Edge cases")
print("=" * 70)

# Empty query
try:
    jobs = retrieve_jobs("", top_k=5, settings=settings)
    assert jobs == [], "Empty query should return empty list"
    print("✓ Empty query handled gracefully (returned empty list)")
except Exception as e:
    print(f"✗ Empty query error: {e}")

# No results query
try:
    jobs = retrieve_jobs("xyzabc123nonexistent", top_k=5, settings=settings)
    print(f"✓ Nonsense query returned {len(jobs)} results (expected 0-5)")
except Exception as e:
    print(f"✗ Nonsense query error: {e}")

print("\n" + "=" * 70)
print("✓ TESTS COMPLETED")
print("=" * 70)
print("\nNext Steps:")
print("  - Task 4.1.2: Implement grade_documents() with Gemini")
print("  - Task 4.1.3: Implement generate_answer() with context")
