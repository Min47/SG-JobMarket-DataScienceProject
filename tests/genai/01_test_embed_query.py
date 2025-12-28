"""Test 01: Query Embedding Function (Task 4.1.1)

Tests the embed_query() function that converts user queries into 384-dim vectors
for semantic search in BigQuery.

Test Coverage:
- Basic embedding generation
- L2 normalization verification
- Consistency (same query = same embedding)
- Edge cases (empty, short, long queries)
- Special characters handling
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from genai.rag import embed_query
from nlp.embeddings import EmbeddingGenerator

_EMBEDDING_DIM = EmbeddingGenerator.EMBEDDING_DIM  # 384

print("=" * 60)
print("Test 01: embed_query() Function")
print("=" * 60)

# Test 1: Basic query embedding
print("\n[Test 1] Basic query embedding")
query = "data scientist machine learning python"
embedding = embed_query(query)
print(f"✓ Query: '{query}'")
print(f"✓ Embedding dimension: {len(embedding)}")
print(f"✓ Sample values: {embedding[:5]}")
assert len(embedding) == _EMBEDDING_DIM, f"Expected {_EMBEDDING_DIM} dims, got {len(embedding)}"

# Test 2: L2 normalization check
print("\n[Test 2] L2 normalization check")
norm = sum(x**2 for x in embedding) ** 0.5
print(f"✓ L2 norm: {norm:.6f} (should be ~1.0)")
assert 0.99 < norm < 1.01, f"Normalization failed: {norm}"
print("✓ Normalization verified")

# Test 3: Consistency
print("\n[Test 3] Consistency check")
embedding1 = embed_query(query)
embedding2 = embed_query(query)
is_same = embedding1 == embedding2
print(f"✓ Same query produces same embedding: {is_same}")
assert is_same, "Embeddings not consistent!"

# Test 4: Different queries
print("\n[Test 4] Different queries produce different embeddings")
query2 = "software engineer java backend"
embedding3 = embed_query(query2)
is_different = embedding != embedding3
print(f"✓ Query 1: '{query}'")
print(f"✓ Query 2: '{query2}'")
print(f"✓ Embeddings are different: {is_different}")
assert is_different, "Different queries should produce different embeddings!"

# Test 5: Edge cases
print("\n[Test 5] Edge cases")
try:
    embed_query("")
    print("✗ Empty query should raise ValueError")
    assert False
except ValueError as e:
    print(f"✓ Empty query raises ValueError: {e}")

try:
    embed_query("ab")
    print("✗ Short query should raise ValueError")
    assert False
except ValueError as e:
    print(f"✓ Short query raises ValueError: {e}")

# Test 6: Special characters
print("\n[Test 6] Special characters")
special_query = "C++ developer (full-time) with 5+ years @Singapore"
embedding4 = embed_query(special_query)
print(f"✓ Query with special chars: '{special_query}'")
print(f"✓ Embedding generated: {len(embedding4)} dims")
assert len(embedding4) == _EMBEDDING_DIM

# Test 7: Long query truncation
print("\n[Test 7] Long query truncation")
long_query = "data scientist " * 200  # ~3000 chars
embedding5 = embed_query(long_query)
print(f"✓ Long query ({len(long_query)} chars) handled")
print(f"✓ Embedding generated: {len(embedding5)} dims")
assert len(embedding5) == _EMBEDDING_DIM

# Test 8: Singleton pattern verification
print("\n[Test 8] Singleton pattern (generator reuse)")
from genai.rag import _get_embedding_generator
gen1 = _get_embedding_generator()
gen2 = _get_embedding_generator()
is_same_instance = gen1 is gen2
print(f"✓ Same EmbeddingGenerator instance: {is_same_instance}")
assert is_same_instance, "Singleton pattern not working!"

print("\n" + "=" * 60)
print("✓ ALL TESTS PASSED!")
print("=" * 60)
print("\nNext: Run 02_test_retrieve_jobs.py for integration test")
