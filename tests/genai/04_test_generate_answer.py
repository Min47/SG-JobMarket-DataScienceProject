"""Test 04: Answer Generation Function (Task 4.1.3)

Tests the generate_answer() function that uses Gemini Pro to generate
natural language answers based on graded job context.

⚠️ REQUIRES:
- GCP credentials configured with Vertex AI API enabled
- BigQuery dataset with job_embeddings and cleaned_jobs tables
- Completed Tasks 4.1.1 and 4.1.2

Test Coverage:
- Basic answer generation with context
- Citation handling
- Empty context handling
- Different query types (salary, requirements, companies)
- Response quality (structure, citations, accuracy)
"""

import sys
import traceback
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from genai.rag import retrieve_jobs, grade_documents, generate_answer
from utils.config import Settings

print("=" * 70)
print("Test 04: generate_answer() Function - Answer Generation")
print("=" * 70)

# Load settings
try:
    settings = Settings.load()
    print(f"\n✓ GCP Project: {settings.gcp_project_id}")
    print(f"✓ Region: {settings.gcp_region}")
except ValueError as e:
    print(f"\n✗ Configuration error: {e}")
    exit(1)

# Test 1: Basic answer generation
print("\n" + "=" * 70)
print("[Test 1] Basic answer generation - 'python developer jobs'")
print("=" * 70)

try:
    print("\n📊 Step 1: Retrieving jobs...")
    jobs = retrieve_jobs("python developer jobs", top_k=10, settings=settings)
    print(f"   Retrieved {len(jobs)} jobs")
    
    print("\n🤖 Step 2: Grading documents...")
    graded = grade_documents("python developer jobs", jobs, threshold=5.0, settings=settings)
    print(f"   Graded and filtered: {len(graded)} jobs")
    
    if graded:
        print("\n✨ Step 3: Generating answer with Gemini Flash...")
        result = generate_answer("What Python developer jobs are available?", graded, settings=settings)
        
        print(f"\n{'=' * 70}")
        print("GENERATED ANSWER:")
        print(f"{'=' * 70}")
        print(result['answer'])
        print(f"\n{'=' * 70}")
        
        # Verify response structure
        assert 'answer' in result, "❌ Missing 'answer' field"
        assert 'sources' in result, "❌ Missing 'sources' field"
        assert 'metadata' in result, "❌ Missing 'metadata' field"
        print(f"\n✓ Response structure valid")
        
        # Verify answer content
        assert len(result['answer']) > 100, "❌ Answer too short"
        print(f"✓ Answer length: {len(result['answer'])} chars")
        
        # Verify sources
        assert len(result['sources']) > 0, "❌ No sources provided"
        print(f"✓ Sources: {len(result['sources'])} jobs cited")
        
        # Show sources
        print("\nCITED SOURCES:")
        for src in result['sources'][:3]:
            print(f"  [{src['number']}] {src['job_title']} - {src['company_name']}")
        
        # Show metadata
        print(f"\nMETADATA:")
        print(f"  Model: {result['metadata']['model']}")
        print(f"  Latency: {result['metadata']['latency_ms']}ms")
        print(f"  Context jobs: {result['metadata']['num_context_jobs']}")
        
        print("\n✓ Test 1 PASSED")
    else:
        print("⚠️  No jobs passed grading threshold")

except Exception as e:
    print(f"✗ Error: {e}")
    traceback.print_exc()

# Test 2: Salary-focused query
print("\n" + "=" * 70)
print("[Test 2] Salary query - 'data scientist salary range'")
print("=" * 70)

try:
    print("\n📊 Retrieving and grading...")
    jobs = retrieve_jobs("data scientist", top_k=8, settings=settings)
    graded = grade_documents("data scientist salary", jobs, threshold=5.0, settings=settings)
    
    if graded:
        print(f"✓ {len(graded)} relevant jobs found")
        
        print("\n✨ Generating salary analysis...")
        result = generate_answer(
            "What is the salary range for data scientist positions?",
            graded,
            settings=settings
        )
        
        print(f"\n{'=' * 70}")
        print("SALARY ANALYSIS:")
        print(f"{'=' * 70}")
        print(result['answer'])
        print(f"\n{'=' * 70}")
        
        # Check for salary mentions
        answer_lower = result['answer'].lower()
        assert 'sgd' in answer_lower or 'salary' in answer_lower, "❌ No salary information"
        print("\n✓ Answer contains salary information")
        
        print("\n✓ Test 2 PASSED")

except Exception as e:
    print(f"✗ Error: {e}")
    traceback.print_exc()

# Test 3: Requirements/skills query
print("\n" + "=" * 70)
print("[Test 3] Skills query - 'software engineer requirements'")
print("=" * 70)

try:
    print("\n📊 Retrieving and grading...")
    jobs = retrieve_jobs("software engineer", top_k=8, settings=settings)
    graded = grade_documents("software engineer", jobs, threshold=5.0, settings=settings)
    
    if graded:
        print(f"✓ {len(graded)} relevant jobs found")
        
        print("\n✨ Generating requirements summary...")
        result = generate_answer(
            "What are the common requirements for software engineer positions?",
            graded,
            settings=settings
        )
        
        print(f"\n{'=' * 70}")
        print("REQUIREMENTS SUMMARY:")
        print(f"{'=' * 70}")
        print(result['answer'])
        print(f"\n{'=' * 70}")
        
        # Check for technical content
        answer_lower = result['answer'].lower()
        technical_keywords = ['skill', 'experience', 'require', 'qualification', 'language', 'framework']
        has_technical = any(kw in answer_lower for kw in technical_keywords)
        assert has_technical, "❌ No technical requirements mentioned"
        print("\n✓ Answer contains technical requirements")
        
        print("\n✓ Test 3 PASSED")

except Exception as e:
    print(f"✗ Error: {e}")
    traceback.print_exc()

# Test 4: Empty context handling
print("\n" + "=" * 70)
print("[Test 4] Edge case - Empty context")
print("=" * 70)

try:
    result = generate_answer("test query", [], settings=settings)
    
    assert result['answer'], "❌ No answer for empty context"
    assert len(result['sources']) == 0, "❌ Should have no sources"
    assert 'error' in result['metadata'], "❌ Should indicate error in metadata"
    
    print(f"✓ Empty context handled gracefully")
    print(f"  Answer: {result['answer'][:100]}...")
    
    print("\n✓ Test 4 PASSED")

except Exception as e:
    print(f"✗ Error: {e}")
    traceback.print_exc()

# Test 5: Citation verification
print("\n" + "=" * 70)
print("[Test 5] Citation format verification")
print("=" * 70)

try:
    jobs = retrieve_jobs("developer", top_k=5, settings=settings)
    graded = grade_documents("developer", jobs, threshold=0.0, settings=settings)
    
    if graded:
        result = generate_answer("List these jobs", graded, max_context_jobs=3, settings=settings)
        
        # Verify sources have required fields
        for src in result['sources']:
            assert 'number' in src, "❌ Source missing number"
            assert 'job_id' in src, "❌ Source missing job_id"
            assert 'job_title' in src, "❌ Source missing job_title"
            assert 'company_name' in src, "❌ Source missing company_name"
            assert 'job_url' in src, "❌ Source missing job_url"
        
        print(f"✓ All {len(result['sources'])} sources have required fields")
        
        # Verify max_context_jobs limit
        assert len(result['sources']) <= 3, "❌ Exceeded max_context_jobs limit"
        print(f"✓ Respected max_context_jobs=3 limit")
        
        print("\n✓ Test 5 PASSED")

except Exception as e:
    print(f"✗ Error: {e}")
    traceback.print_exc()

print("\n" + "=" * 70)
print("✓ TESTS COMPLETED")
print("=" * 70)
print("\nNext Steps:")
print("  - Task 4.2.1: Define LangGraph StateGraph")
print("  - Task 4.2.2: Implement agent nodes (retrieve, grade, generate, rewrite)")
print("  - Task 4.2.3: Add conditional edges and orchestration")
