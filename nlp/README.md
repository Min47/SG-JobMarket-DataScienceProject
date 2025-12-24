# 🚀 NLP Pipeline Deployment Guide

Complete sequence for setting up embeddings infrastructure from scratch.

---

## Prerequisites
- ✅ BigQuery dataset `sg_job_market` created
- ✅ `cleaned_jobs` table populated with data
- ✅ Virtual environment activated: `.venv/Scripts/activate`
- ✅ GCP credentials configured: `GOOGLE_APPLICATION_CREDENTIALS` set

---

## Phase 1: Initial Setup (One-Time)

### Step 1: Create Embeddings Table
Creates `job_embeddings` table in BigQuery with proper schema (FLOAT64 REPEATED).

```bash
.venv/Scripts/python.exe -m nlp.setup_embeddings_table
```

**What it does:**
- Creates table with columns: job_id, source, embedding, model_name, created_at
- Sets up partitioning by `created_at` (DAY) for efficient queries
- Sets up clustering by `source, job_id` for fast lookups
- Schema auto-generated from `JobEmbedding` dataclass in `utils/schemas.py`

**Expected output:**
```
✅ Created table sg-job-market.sg_job_market.job_embeddings
Schema:
  - job_id: STRING (NULLABLE)
  - source: STRING (NULLABLE)
  - model_name: STRING (NULLABLE)
  - embedding: FLOAT64 (REPEATED)  ← 384 dimensions
  - created_at: TIMESTAMP (NULLABLE)
Partitioned by: created_at (DAY)
Clustered by: source, job_id
Rows: 0, Size: 0.00 MB
```

---

## Phase 2: Generate Embeddings

### Step 2a: Test on Small Sample (Recommended First)
Test embedding generation on 10 jobs to verify pipeline works.

```bash
.venv/Scripts/python.exe -m nlp.generate_embeddings --limit 10
```

**What it does:**
- Queries 10 jobs from `cleaned_jobs` table
- Combines job_title + job_description (truncated to 1000 chars)
- Generates 384-dim embeddings using Sentence-BERT (all-MiniLM-L6-v2)
- Inserts to BigQuery `job_embeddings` table

**Expected duration:** ~10 seconds

### Step 2b: Test on Medium Sample
Test on 100 jobs to verify stability.

```bash
.venv/Scripts/python.exe -m nlp.generate_embeddings --limit 100
```

**Expected duration:** ~20 seconds

### Step 2c: Full Generation (All Jobs)
Generate embeddings for ALL jobs in cleaned_jobs table.

```bash
.venv/Scripts/python.exe -m nlp.generate_embeddings
```

**What it does:**
- Queries all jobs WITHOUT embeddings (incremental, idempotent)
- Generates embeddings in batches of 32 texts
- Inserts to BigQuery in batches of 500 rows (prevents timeout)
- Supports re-running (only processes new jobs)

**Expected duration:** 
- 5-10 minutes for ~10K jobs (CPU)
- 1-2 minutes for ~10K jobs (GPU, if available)

**Expected output:**
```
✅ Generated embeddings with shape: (6775, 384)
✅ Inserted batch 1: 500 embeddings (500/6775 total)
...
✅ Successfully inserted all 6775 embeddings to BigQuery
Complete: {'jobs_processed': 6775, 'embeddings_generated': 6775, 'embedding_dim': 384, 'model_name': 'all-MiniLM-L6-v2', 'status': 'success'}
```

---

## Phase 3: Create Vector Index (Enables Fast Search)

### Step 3: Create IVF Vector Index
Creates an Inverted File (IVF) index for fast nearest-neighbor search.

```bash
.venv/Scripts/python.exe -m nlp.create_vector_index
```

**What it does:**
- Creates `job_embedding_idx` on `job_embeddings.embedding` column
- Uses COSINE distance (perfect for normalized SBERT embeddings)
- Creates 100 IVF buckets (optimal for ~10K rows)
- Enables ~100x faster similarity search vs full table scan

**Why we need this:**
- **Without index:** Query compares against ALL 10K embeddings (slow, ~5s)
- **With index:** Query only searches relevant buckets (~50ms)
- Trade-off: ~10% recall loss for 100x speed improvement

**Expected duration:** 1-2 minutes

**Expected output:**
```
✅ SUCCESS: Vector index created!
Index name: job_embedding_idx
Distance metric: COSINE
IVF buckets: 100
🚀 Ready for similarity search!
```

**Optional flags:**
```bash
# Drop and recreate index
.venv/Scripts/python.exe -m nlp.create_vector_index --drop

# Verify after creation
.venv/Scripts/python.exe -m nlp.create_vector_index --verify

# Custom bucket count (for different data sizes)
.venv/Scripts/python.exe -m nlp.create_vector_index --num-lists 200
```

---

## Phase 4: Test & Verify

### Step 4: Run Test Notebook
Open and run `notebooks/test_embeddings.ipynb` to verify:
- ✅ Embeddings loaded correctly from BigQuery
- ✅ Similarity search returns relevant results
- ✅ Embedding quality metrics (normalized vectors, proper value range)
- ✅ Visualization (PCA 2D plot by job category)

```bash
# Launch Jupyter
jupyter notebook notebooks/test_embeddings.ipynb
```

**What it tests:**
1. Load embeddings with JOIN to cleaned_jobs
2. Semantic search: "Senior Data Scientist with Python"
3. Quality checks: vector norms, value distribution
4. PCA visualization colored by job_classification
5. Intra-category similarity analysis

---

## 🔄 Daily Operations (After Initial Setup)

### Incremental Embedding Generation (Daily)
Run daily after scraper adds new jobs. Only processes jobs without embeddings.

```bash
.venv/Scripts/python.exe -m nlp.generate_embeddings
```

**Why it's fast:**
- Queries: `LEFT JOIN job_embeddings WHERE e.job_id IS NULL`
- Only embeds NEW jobs since last run
- Idempotent: safe to run multiple times

---

## 📊 Monitoring & Verification

### Check Embedding Coverage
```bash
.venv/Scripts/python.exe -c "from google.cloud import bigquery; client = bigquery.Client(project='sg-job-market'); q1 = 'SELECT COUNT(*) as total FROM \`sg-job-market.sg_job_market.job_embeddings\`'; q2 = 'SELECT COUNT(*) as total FROM \`sg-job-market.sg_job_market.cleaned_jobs\`'; emb = list(client.query(q1).result())[0].total; jobs = list(client.query(q2).result())[0].total; print(f'Embeddings: {emb}'); print(f'Cleaned jobs: {jobs}'); print(f'Coverage: {emb/jobs*100:.1f}%')"
```

### Verify Embedding Quality
```bash
.venv/Scripts/python.exe -c "from google.cloud import bigquery; import numpy as np; client = bigquery.Client(project='sg-job-market'); query = 'SELECT embedding FROM \`sg-job-market.sg_job_market.job_embeddings\` LIMIT 100'; rows = list(client.query(query).result()); embeddings = np.array([row.embedding for row in rows]); print(f'Shape: {embeddings.shape}'); print(f'Range: [{embeddings.min():.3f}, {embeddings.max():.3f}]'); norms = np.linalg.norm(embeddings, axis=1); print(f'Norm: {norms.mean():.3f} ± {norms.std():.3f} (expected: 1.0)')"
```

---

## 🐛 Troubleshooting

### Issue: "Table not found"
**Solution:** Run Step 1 first: `python -m nlp.setup_embeddings_table`

### Issue: "Model loading timeout"
**Solution:** First run downloads ~90MB model from HuggingFace. Wait 2-3 minutes.

### Issue: "Connection timeout during insert"
**Solution:** Already fixed! Code inserts in batches of 500 rows. If still fails, reduce batch size in `write_embeddings_to_bq()`.

### Issue: "Invalid array specified for created_at"
**Solution:** Already fixed! Was trailing comma bug: `timestamp = datetime.now().isoformat(),` → removed comma.

### Issue: "Index already exists"
**Solution:** Run with `--drop` flag: `python -m nlp.create_vector_index --drop`

---

## 📝 Quick Reference

| Command | Purpose | Duration |
|---------|---------|----------|
| `setup_embeddings_table` | Create BigQuery table (one-time) | <5s |
| `generate_embeddings --limit 10` | Test on 10 jobs | ~10s |
| `generate_embeddings` | Generate all embeddings | 5-10 min |
| `create_vector_index` | Create search index | 1-2 min |
| `test_embeddings.ipynb` | Verify quality | Interactive |

---

## 🎯 Success Criteria

After completing all steps, you should have:
- ✅ `job_embeddings` table in BigQuery with 6,775+ rows
- ✅ 100% coverage (embeddings for all cleaned jobs)
- ✅ Vector index `job_embedding_idx` created
- ✅ Embedding norms ≈ 1.0 (normalized vectors)
- ✅ Similarity search returns relevant results (verified in notebook)

**Next phase:** Train ML models (Phase 3B - Feature Engineering)
