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

## Vector Index Algorithms Comparison

### Overview: IVF vs HNSW vs LSH

There are several algorithms for approximate nearest-neighbor search. Here's a comparison:

| Algorithm | Full Name | How It Works | Speed | Accuracy | Memory | Best For |
|-----------|-----------|--------------|-------|----------|--------|----------|
| **IVF** ✅ | Inverted File | KMeans clustering into buckets | Fast | 90-95% | Low | **Our use case** (7K-100K vectors) |
| **HNSW** | Hierarchical Navigable Small World | Multi-layer graph navigation | **Fastest** | 95-99% | **High** | >100K vectors, need max accuracy |
| **LSH** | Locality-Sensitive Hashing | Random projection hashing | Medium | 80-90% | Low | >1M vectors, prioritize speed |
| **Flat** | Brute Force | Compare all vectors | **Slow** | 100% | Low | <10K vectors, need perfect recall |

### Detailed Algorithm Breakdown

#### 1. IVF (Inverted File) - **OUR CHOICE** ✅

**How it works:**
```python
# Index Creation (One-Time)
1. Run KMeans on all embeddings → 100 cluster centers
2. Assign each job to nearest cluster
3. Store: {cluster_id: [job_ids]}

# Query Time
1. Find 2-3 nearest cluster centers to query
2. Search only jobs in those clusters (~300 jobs)
3. Return top-K most similar
```

**Pros:**
- ✅ Simple to understand and tune
- ✅ Low memory overhead
- ✅ Automatically updates when new vectors added
- ✅ Good balance: 90-95% recall, 100x speedup
- ✅ Built into BigQuery (no external library needed)

**Cons:**
- ⚠️ Slightly lower accuracy than HNSW (5-10% recall loss)
- ⚠️ Clustering quality depends on num_lists parameter

**Perfect for:**
- 7K-100K vectors (our daily job count)
- Daily incremental updates
- Cloud-based (BigQuery) deployment

---

#### 2. HNSW (Hierarchical Navigable Small World)

**How it works:**
```python
# Index Creation
1. Build multi-layer graph (like skip list)
2. Each vector connects to ~16 neighbors per layer
3. Higher layers: sparse, long-range connections
4. Lower layers: dense, local connections

# Query Time
1. Start at top layer, navigate to nearest neighbor
2. Drop to next layer, refine search
3. Repeat until bottom layer
4. Return top-K results
```

**Pros:**
- ✅ **Highest accuracy** (95-99% recall)
- ✅ **Fastest query time** (<10ms for 1M vectors)
- ✅ Works well for high-dimensional data

**Cons:**
- ❌ **High memory** (16 edges × num_layers × num_vectors)
- ❌ **Not natively supported in BigQuery** (need external library like FAISS/Annoy)
- ❌ **Rebuild expensive** when adding many vectors
- ❌ Complex to tune (M, efConstruction, efSearch parameters)

**When to use:**
- >100K vectors
- Need >95% recall
- Have dedicated vector DB (Pinecone, Weaviate, Milvus)
- Infrequent updates (not daily)

**Example memory:**
```
100K vectors × 384 dims × 4 bytes = 150 MB (embeddings)
100K vectors × 16 edges × 5 layers × 4 bytes = 320 MB (graph)
Total: ~470 MB

1M vectors → ~4.7 GB (may not fit in memory)
```

---

#### 3. LSH (Locality-Sensitive Hashing)

**How it works:**
```python
# Index Creation
1. Create random projection matrices (hash functions)
2. Hash each vector → binary code (e.g., 10101011)
3. Group vectors with similar hash codes into buckets

# Query Time
1. Hash query vector
2. Search buckets with matching/similar hash codes
3. Return top-K results
```

**Pros:**
- ✅ **Very fast for massive scale** (>1M vectors)
- ✅ Low memory overhead
- ✅ Sub-linear query time O(n^(1/c))

**Cons:**
- ❌ **Lower accuracy** (80-90% recall)
- ❌ Requires many hash functions for good accuracy
- ❌ Not great for high-dimensional data (curse of dimensionality)
- ❌ Sensitive to hash function choice

**When to use:**
- >1M vectors
- Can tolerate 10-20% recall loss
- Need extremely fast queries
- Text/image duplicate detection

---

#### 4. Flat (Brute Force) - Baseline

**How it works:**
```python
# No index needed
# Query Time: Compare with ALL vectors
similarities = cosine_similarity(query, all_embeddings)
top_k = np.argsort(similarities)[-k:]
```

**Pros:**
- ✅ **100% recall** (perfect accuracy)
- ✅ Simple, no tuning
- ✅ No index overhead

**Cons:**
- ❌ **Slow** (5 seconds for 10K vectors)
- ❌ O(n) complexity - doesn't scale

**When to use:**
- <5K vectors
- Need perfect recall
- Prototyping/baseline

---

### Recommendation for Our Project ✅

**Current Scale: 7K jobs/day → 200K jobs/year**

| Algorithm | Speed | Accuracy | Complexity | Verdict |
|-----------|-------|----------|------------|---------|
| **IVF (BigQuery)** | ⚡⚡⚡ (50ms) | ⭐⭐⭐⭐ (90%) | Simple | ✅ **BEST CHOICE** |
| HNSW (FAISS) | ⚡⚡⚡⚡ (10ms) | ⭐⭐⭐⭐⭐ (95%) | Complex | Overkill, requires external library |
| LSH | ⚡⚡⚡⚡⚡ (5ms) | ⭐⭐⭐ (85%) | Medium | Too inaccurate for job matching |
| Flat | ⚡ (5s) | ⭐⭐⭐⭐⭐ (100%) | Simple | Too slow for production |

**Why IVF wins for us:**
1. ✅ **Native BigQuery support** - No external services needed
2. ✅ **Automatic updates** - Add new jobs daily without rebuild
3. ✅ **Good enough accuracy** - 90% recall is fine for job recommendations
4. ✅ **Scales to 1M+ rows** - Future-proof for 5 years
5. ✅ **Cost-effective** - No additional infrastructure

**When to switch to HNSW:**
- If we reach >500K jobs AND need >95% recall
- If query latency becomes critical (<10ms required)
- Willing to manage vector DB (Pinecone/Weaviate)

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

## Daily Operations: How Incremental Updates Work

### Understanding Cumulative Data Growth

**Question:** "We scrape 7K jobs daily. After embedding and indexing, do we get 200 buckets?"

**Answer:** No! The index **automatically adapts** to total data size, not daily additions.

### Timeline Breakdown

```
Day 1 (Initial Deployment):
  Scraped jobs: 7,000
  Embeddings table: 7,000 rows
  Vector index created: 100 buckets (sqrt(7000) ≈ 84 → rounded to 100)
  Bucket distribution: ~70 jobs per bucket

Day 30:
  Scraped jobs: 7,000 new
  Embeddings table: 7,000 + 7,000 = 14,000 rows (cumulative)
  Vector index: STILL 100 buckets (auto-updated by BigQuery)
  Bucket distribution: ~140 jobs per bucket

Day 365 (1 year):
  Scraped jobs: 7,000 new
  Embeddings table: 7,000 × 365 = 2,555,000 rows (cumulative)
  Vector index: STILL 100 buckets (unless we manually recreate)
  Bucket distribution: ~25,550 jobs per bucket ⚠️ Getting large!
```

### Maintenance Schedule (Our Project)

| Time | Total Jobs | Current Buckets | Jobs/Bucket | Action |
|------|-----------|-----------------|-------------|--------|
| Day 1 | 7K | 100 | 70 | ✅ Create initial index |
| Month 1 | 210K | 100 | 2,100 | ✅ No action (still fast) |
| Month 6 | 1.26M | 100 | 12,600 | ⚠️ Consider recreating with 1,000 buckets |
| Year 1 | 2.55M | 100 | 25,550 | ❌ **Must recreate** with 1,600 buckets |

**Quarterly Review (Recommended):**
```bash
# Check current stats
.venv/Scripts/python.exe -c "
from google.cloud import bigquery
client = bigquery.Client(project='sg-job-market')
query = 'SELECT COUNT(*) as total FROM job_embeddings'
total = list(client.query(query).result())[0].total
optimal_buckets = int(total ** 0.5)
print(f'Total jobs: {total:,}')
print(f'Current buckets: 100')
print(f'Optimal buckets: {optimal_buckets}')
if optimal_buckets > 200:
    print('⚠️ Consider recreating index!')
"
```

### Daily Workflow (Automated)

```bash
# Step 1: Scraper runs (Cloud Scheduler 2 AM SGT)
# Scrapes ~7K new jobs → GCS

# Step 2: ETL runs (Cloud Function auto-triggered)
# GCS → raw_jobs → cleaned_jobs (BigQuery)

# Step 3: Embedding generation (Cloud Scheduler 4 AM SGT)
.venv/Scripts/python.exe -m nlp.generate_embeddings
# Only embeds NEW jobs (incremental, uses LEFT JOIN)
# Inserts to job_embeddings

# Step 4: Vector index auto-updates (BigQuery)
# No manual action needed!
# New embeddings assigned to nearest buckets

# Step 5: Quarterly review (every 3 months)
# Check if index needs recreation with more buckets
```

**No daily index recreation needed!** ✅

---

## Tuning Parameters Deep Dive

### 1. num_lists (Number of Buckets)

**What it controls:** How many clusters (buckets) to divide your embeddings into.

**Formula:** `num_lists = sqrt(total_rows)`

**Examples:**
```
1,000 jobs    → sqrt(1000)    ≈ 32   → use 30-50
10,000 jobs   → sqrt(10000)   = 100  → use 100 ✅ (our initial)
100,000 jobs  → sqrt(100000)  ≈ 316  → use 300-500
1,000,000 jobs → sqrt(1000000) = 1000 → use 1000-1500
```

**Impact on Performance:**

| num_lists | Buckets | Jobs/Bucket (for 10K) | Search Space | Query Time | Recall |
|-----------|---------|----------------------|--------------|------------|--------|
| 10 | 10 | 1,000 | 3,000 jobs (3 buckets) | **Slow** (500ms) | 95% |
| 50 | 50 | 200 | 600 jobs | Medium (100ms) | 92% |
| **100** ✅ | **100** | **100** | **300 jobs** | **Fast (50ms)** | **90%** |
| 500 | 500 | 20 | 60 jobs | **Fastest (10ms)** | 80% |
| 1000 | 1000 | 10 | 30 jobs | Fastest (5ms) | **70%** ⚠️ |

**Visual Breakdown:**
```
num_lists = 100 (OPTIMAL ✅)
├── Bucket 0: [Job1, Job45, Job234, ...] (100 jobs)
├── Bucket 1: [Job2, Job56, Job345, ...] (100 jobs)
├── ...
└── Bucket 99: [Job99, Job999, Job8888, ...] (100 jobs)

Query: "Data Scientist"
  → Find nearest buckets: [Bucket 42, Bucket 87, Bucket 93]
  → Search ~300 jobs instead of 10,000
  → 33x speedup, 90% recall ✅

num_lists = 10 (TOO FEW ❌)
├── Bucket 0: [1000 jobs] ← Too large!
├── ...
└── Bucket 9: [1000 jobs]

Query: "Data Scientist"
  → Find nearest buckets: [Bucket 4, Bucket 7, Bucket 9]
  → Search ~3,000 jobs (still slow)
  → Only 3x speedup ❌

num_lists = 1000 (TOO MANY ❌)
├── Bucket 0: [10 jobs]
├── ...
└── Bucket 999: [10 jobs]

Query: "Data Scientist"
  → Need to search 20+ buckets to find enough candidates
  → Search overhead increases
  → Recall drops to 70% ❌
```

**How to tune:**

```python
# Experimentation script
num_lists_options = [50, 100, 200, 500]

for n in num_lists_options:
    # Create index
    create_vector_index(num_lists=n)
    
    # Test query performance
    queries = ["Data Scientist", "Software Engineer", "Marketing Manager"]
    
    for query in queries:
        # Measure latency and recall
        results = similarity_search(query, top_k=10)
        # Compare with ground truth (flat search)
        
    # Log results
    print(f"num_lists={n}: avg_latency={latency}ms, avg_recall={recall}%")

# Choose best trade-off (usually sqrt rule)
```

### 2. nprobe (Number of Buckets to Search) - **Advanced**

**Note:** BigQuery doesn't expose this parameter, but it's important to understand.

**What it controls:** How many buckets to search during query.

**Default:** BigQuery auto-tunes (usually 2-5 buckets)

**Trade-off:**
```
nprobe = 1:  Search 1 bucket  → Fastest (20ms), Low recall (70%)
nprobe = 3:  Search 3 buckets → Fast (50ms), Good recall (90%) ✅
nprobe = 10: Search 10 buckets → Medium (150ms), High recall (95%)
nprobe = 50: Search 50 buckets → Slow (500ms), Near-perfect (99%)
```

**If you need to tune (external FAISS library):**
```python
import faiss

index = faiss.IndexIVFFlat(quantizer, d, num_lists)
index.nprobe = 3  # Search 3 nearest buckets (default)

# Increase for higher recall
index.nprobe = 10  # Search 10 buckets (slower, more accurate)
```

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
