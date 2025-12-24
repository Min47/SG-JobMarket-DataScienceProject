# 🔍 Vector Index Explained

## What is a Vector Index?

A **vector index** is a data structure that enables **fast nearest-neighbor search** on high-dimensional embeddings (like our 384-dimensional SBERT vectors).

### Without Index (Slow ❌)
```
Query: "Data Scientist jobs"
  ↓ Generate embedding: [0.23, -0.45, 0.67, ..., 0.33]
  ↓ Compare with ALL 10,000 jobs:
    Job 1: cosine_similarity(query, job1_embedding)
    Job 2: cosine_similarity(query, job2_embedding)
    ...
    Job 10,000: cosine_similarity(query, job10000_embedding)
  ↓ Sort by similarity
  ↓ Return top 10

Duration: ~5 seconds (10,000 comparisons)
```

### With IVF Index (Fast ✅)
```
Query: "Data Scientist jobs"
  ↓ Generate embedding: [0.23, -0.45, 0.67, ..., 0.33]
  ↓ Find nearest buckets (2-3 out of 100)
  ↓ Compare with ~300 jobs (not 10,000!):
    Bucket 42: Compare with 150 jobs
    Bucket 87: Compare with 150 jobs
  ↓ Sort by similarity
  ↓ Return top 10

Duration: ~50ms (300 comparisons)
Speedup: 100x faster! ⚡
```

---

## How IVF (Inverted File) Index Works

### Step 1: Clustering (Index Creation Time)
During index creation, BigQuery runs KMeans clustering on ALL embeddings:

```python
# Pseudocode of what BigQuery does internally
embeddings = load_all_embeddings()  # 10,000 jobs × 384 dims
kmeans = KMeans(n_clusters=100)  # Create 100 buckets
cluster_labels = kmeans.fit_predict(embeddings)

# Build inverted file structure
inverted_file = {
    0: [job_123, job_456, job_789, ...],  # Jobs in bucket 0
    1: [job_234, job_567, ...],           # Jobs in bucket 1
    ...
    99: [job_999, job_1234, ...]          # Jobs in bucket 99
}
```

**Result:** 10,000 jobs divided into 100 buckets (~100 jobs per bucket)

### Step 2: Query Time Search
When you search, BigQuery:

1. **Find nearest cluster centers** to your query embedding
2. **Retrieve jobs** from those 2-3 buckets only
3. **Compare** query against ~300 jobs instead of 10,000
4. **Return** top-K results

**Trade-off:**
- **Recall:** ~90% (may miss some relevant results in distant buckets)
- **Speed:** 100x faster
- **Acceptable for most applications** ✅

---

## Why COSINE Distance?

Our embeddings are **normalized unit vectors** (norm = 1.0), which means:

```python
# SBERT automatically normalizes
embedding = sbert.encode("Data Scientist")
norm = np.linalg.norm(embedding)  # Always 1.0

# Cosine similarity between unit vectors
cos_sim(a, b) = dot(a, b) / (||a|| × ||b||)
              = dot(a, b) / (1.0 × 1.0)
              = dot(a, b)  # Just dot product!
```

**Why use COSINE over EUCLIDEAN?**

| Distance | Formula | Best For |
|----------|---------|----------|
| **COSINE** ✅ | `1 - cos_similarity(a, b)` | Semantic similarity (direction matters, not magnitude) |
| EUCLIDEAN | `sqrt(sum((a - b)²))` | Spatial distance (magnitude matters) |
| DOT_PRODUCT | `sum(a × b)` | Unnormalized vectors |

**Example:**
```
Job A: "Senior Data Scientist"    → embedding_A (norm=1.0)
Job B: "Lead Data Scientist"      → embedding_B (norm=1.0)
Job C: "Restaurant Manager"       → embedding_C (norm=1.0)

Cosine Similarity:
  sim(A, B) = 0.92  ← High (similar roles)
  sim(A, C) = 0.15  ← Low (different roles)
```

For normalized vectors, cosine similarity ranges from:
- **1.0** = identical direction (perfect match)
- **0.0** = orthogonal (unrelated)
- **-1.0** = opposite direction (antonyms)

---

## Why num_lists=100?

**Rule of thumb:** `num_lists ≈ sqrt(num_rows)`

| Dataset Size | Recommended num_lists |
|--------------|----------------------|
| 1,000 jobs | 30-50 |
| 10,000 jobs | 100 ✅ |
| 100,000 jobs | 300-500 |
| 1,000,000 jobs | 1,000 |

**Trade-off:**
- **Too few buckets (e.g., 10):** Each bucket has 1,000 jobs → still slow
- **Too many buckets (e.g., 1,000):** Each bucket has 10 jobs → may need to search many buckets → slower
- **Just right (100):** Each bucket has ~100 jobs, search 2-3 buckets → fast ⚡

---

## Real-World Performance

### Without Index
```sql
-- Naive similarity search (no index)
SELECT 
    job_id,
    job_title,
    (1 - COSINE_DISTANCE(embedding, query_embedding)) AS similarity
FROM job_embeddings
ORDER BY similarity DESC
LIMIT 10;

-- Performance: 5 seconds (scans all 10,000 rows)
```

### With Index
```sql
-- Same query, but uses index automatically
SELECT 
    job_id,
    job_title,
    (1 - COSINE_DISTANCE(embedding, query_embedding)) AS similarity
FROM job_embeddings
ORDER BY similarity DESC
LIMIT 10;

-- Performance: 50ms (scans ~300 rows in relevant buckets)
```

**BigQuery automatically uses the index!** No need to change your SQL.

---

## Index Creation Command

```bash
.venv/Scripts/python.exe -m nlp.create_vector_index
```

**What it does:**
1. Checks `job_embeddings` table exists
2. Creates `job_embedding_idx` on the `embedding` column
3. Configures:
   - Distance: COSINE
   - Index type: IVF
   - Buckets: 100
4. Takes 1-2 minutes (runs KMeans clustering)

**Output:**
```
✅ SUCCESS: Vector index created!
Index name: job_embedding_idx
Distance metric: COSINE
IVF buckets: 100
🚀 Ready for similarity search!
```

---

## When to Rebuild Index

### Automatic Reindexing
BigQuery automatically updates the index when you INSERT new embeddings. No action needed!

### Manual Rebuild (Optional)
Rebuild if:
- Data grew significantly (1K → 100K rows)
- Want to tune `num_lists` for new data size

```bash
# Drop and recreate with new parameters
.venv/Scripts/python.exe -m nlp.create_vector_index --drop --num-lists 300
```

---

## Testing the Index

Use the test notebook to verify:

```bash
jupyter notebook notebooks/test_embeddings.ipynb
```

**Test 1: Similarity Search**
```python
query = "Senior Data Scientist with Python"
results = similarity_search(query, top_k=10)
# Should return: Data Scientist, ML Engineer, Data Analyst roles
```

**Test 2: Query Performance**
```python
import time
start = time.time()
results = similarity_search(query)
duration = time.time() - start
print(f"Query took: {duration*1000:.0f}ms")
# Expected: <100ms with index, ~5s without
```

---

## FAQ

### Q: Do I need to recreate the index when adding new jobs?
**A:** No! BigQuery auto-updates the index on INSERT. Just run `generate_embeddings.py` daily.

### Q: What if I change num_lists?
**A:** Run `create_vector_index.py --drop` to recreate. Test different values to find optimal speed/recall trade-off.

### Q: Can I use other distance metrics?
**A:** Yes, but COSINE is best for SBERT embeddings:
- EUCLIDEAN: Works, but less interpretable
- DOT_PRODUCT: Only if embeddings are NOT normalized

### Q: How much does it cost?
**A:** Index creation is free! Query costs are same as regular SELECT queries (~$5/TB scanned, covered by free tier).

### Q: What's the recall loss?
**A:** ~10% for IVF with 100 buckets. You might miss 1-2 relevant results out of top-100, but top-10 are usually correct.

---

## Summary

| Aspect | Value | Why |
|--------|-------|-----|
| **Index Type** | IVF (Inverted File) | Fast approximate search |
| **Distance** | COSINE | Perfect for normalized embeddings |
| **Buckets** | 100 | Optimal for 10K rows (sqrt rule) |
| **Speed Gain** | 100x | 5s → 50ms per query |
| **Recall Loss** | ~10% | Acceptable trade-off |
| **Cost** | FREE | Covered by BigQuery free tier |

**Next Steps:**
1. ✅ Run `create_vector_index.py`
2. ✅ Test in notebook
3. ✅ Use in RAG pipeline (Phase 4)
