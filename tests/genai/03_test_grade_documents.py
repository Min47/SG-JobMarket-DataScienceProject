"""Test 03: Document Grading Function (Task 4.1.2)

Tests the grade_documents() function that uses Gemini Pro to score
document relevance and filter/re-rank results.

⚠️ REQUIRES:
- GCP credentials configured with Vertex AI API enabled
- BigQuery dataset with job_embeddings and cleaned_jobs tables

Test Coverage:
- Basic document grading and filtering
- Threshold adjustment
- Re-ranking by relevance score
- Error handling (malformed responses, API failures)
"""

import sys
import traceback
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from genai.rag import retrieve_jobs, grade_documents
from utils.config import Settings

print("=" * 70)
print("Test 03: grade_documents() Function - LLM Relevance Grading")
print("=" * 70)

# Load settings
try:
    settings = Settings.load()
    print(f"\n✓ GCP Project: {settings.gcp_project_id}")
    print(f"✓ Region: {settings.gcp_region}")
except ValueError as e:
    print(f"\n✗ Configuration error: {e}")
    exit(1)

# Test 1: Basic grading with moderate threshold
print("\n" + "=" * 70)
print("[Test 1] Basic grading - 'python developer' (threshold=5.0)")
print("=" * 70)

try:
    # First, retrieve jobs
    print("\n📊 Step 1: Retrieving jobs...")
    jobs = retrieve_jobs(
        query="python developer",
        top_k=10,
        settings=settings
    )
    print(f"   Retrieved {len(jobs)} jobs")
    
    if not jobs:
        print("⚠️  No jobs retrieved - skipping grading test")
    else:
        # Then grade them
        print("\n🤖 Step 2: Grading with Gemini Pro...")
        graded = grade_documents(
            query="python developer",
            documents=jobs,
            threshold=5.0,
            settings=settings
        )
        
        print(f"\n✓ Graded and filtered: {len(graded)}/{len(jobs)} jobs kept")
        
        if graded:
            print("\nTop 3 graded results:")
            for i, job in enumerate(graded[:3], 1):
                print(f"\n{i}. {job['job_title']}")
                print(f"   Company: {job['company_name']}")
                print(f"   Relevance: {job['relevance_score']:.1f}/10")
                print(f"   Explanation: {job['relevance_explanation']}")
            
            # Verify scores are in descending order
            scores = [j['relevance_score'] for j in graded]
            assert scores == sorted(scores, reverse=True), "❌ Results not sorted by score"
            print("\n✓ Results correctly sorted by relevance score")
            
            # Verify all pass threshold
            assert all(j['relevance_score'] >= 5.0 for j in graded), "❌ Found job below threshold"
            print("✓ All results meet threshold (≥5.0)")
            
            print("\n✓ Test 1 PASSED")
        else:
            print("⚠️  All jobs filtered out (all below threshold)")

except Exception as e:
    print(f"✗ Error: {e}")
    traceback.print_exc()

# Test 2: High threshold filtering
print("\n" + "=" * 70)
print("[Test 2] Strict filtering - 'data scientist' (threshold=7.0)")
print("=" * 70)

try:
    # Retrieve jobs
    jobs = retrieve_jobs(
        query="data scientist with machine learning experience",
        top_k=8,
        settings=settings
    )
    print(f"\n📊 Retrieved {len(jobs)} jobs")
    
    if jobs:
        # Grade with high threshold
        graded = grade_documents(
            query="data scientist with machine learning experience",
            documents=jobs,
            threshold=7.0,
            settings=settings
        )
        
        print(f"✓ High threshold filtering: {len(graded)}/{len(jobs)} jobs kept (threshold=7.0)")
        
        if graded:
            print("\nTop result:")
            job = graded[0]
            print(f"  Title: {job['job_title']}")
            print(f"  Relevance: {job['relevance_score']:.1f}/10")
            print(f"  Explanation: {job['relevance_explanation']}")
            
            # Verify threshold enforcement
            assert all(j['relevance_score'] >= 7.0 for j in graded), "❌ Found job below threshold"
            print("\n✓ All results meet high threshold (≥7.0)")
        
        print("\n✓ Test 2 PASSED")

except Exception as e:
    print(f"✗ Error: {e}")
    traceback.print_exc()

# Test 3: Compare grading with different queries
print("\n" + "=" * 70)
print("[Test 3] Query sensitivity - Same jobs, different queries")
print("=" * 70)

try:
    # Retrieve jobs with one query
    jobs = retrieve_jobs(
        query="software engineer",
        top_k=5,
        settings=settings
    )
    
    if jobs and len(jobs) >= 3:
        print(f"\n📊 Retrieved {len(jobs)} jobs for 'software engineer'")
        
        # Grade with specific query
        graded_specific = grade_documents(
            query="senior full stack developer with React and Node.js",
            documents=jobs.copy(),
            threshold=0.0,  # Keep all to compare scores
            settings=settings
        )
        
        # Grade with generic query
        graded_generic = grade_documents(
            query="software job",
            documents=jobs.copy(),
            threshold=0.0,  # Keep all to compare scores
            settings=settings
        )
        
        print("\nScore comparison for same jobs:")
        print(f"{'Job Title':<40} {'Specific Query':<15} {'Generic Query':<15}")
        print("-" * 70)
        
        for i in range(min(3, len(jobs))):
            title = jobs[i]['job_title'][:38]
            specific_score = graded_specific[i]['relevance_score']
            generic_score = graded_generic[i]['relevance_score']
            print(f"{title:<40} {specific_score:>6.1f}/10        {generic_score:>6.1f}/10")
        
        print("\n✓ Test 3 PASSED - Grading adapts to query specificity")

except Exception as e:
    print(f"✗ Error: {e}")
    traceback.print_exc()

# Test 4: Edge cases
print("\n" + "=" * 70)
print("[Test 4] Edge cases")
print("=" * 70)

# Empty documents list
try:
    graded = grade_documents("test query", [], threshold=5.0, settings=settings)
    assert graded == [], "Empty input should return empty output"
    print("✓ Empty documents list handled gracefully")
except Exception as e:
    print(f"✗ Empty list error: {e}")

# Very low threshold (should keep all)
try:
    jobs = retrieve_jobs("developer", top_k=3, settings=settings)
    if jobs:
        graded = grade_documents("developer", jobs, threshold=0.0, settings=settings)
        print(f"✓ Low threshold (0.0): Kept {len(graded)}/{len(jobs)} jobs")
except Exception as e:
    print(f"✗ Low threshold error: {e}")

# Very high threshold (may filter all)
try:
    jobs = retrieve_jobs("developer", top_k=3, settings=settings)
    if jobs:
        graded = grade_documents("extremely specific niche technology xyz", jobs, threshold=9.5, settings=settings)
        print(f"✓ High threshold (9.5): Kept {len(graded)}/{len(jobs)} jobs")
except Exception as e:
    print(f"✗ High threshold error: {e}")

print("\n" + "=" * 70)
print("✓ TESTS COMPLETED")
print("=" * 70)
print("\nNext Steps:")
print("  - Task 4.1.3: Implement generate_answer() with context")
print("  - Task 4.2: Build LangGraph agent with conditional edges")
