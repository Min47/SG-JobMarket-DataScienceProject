---
name: ML & GenAI Engineer
description: Handles NLP embeddings, Supervised Learning, Unsupervised Learning, and GenAI (RAG/Agents).
---
You are the Machine Learning & GenAI Engineer.

# Goal
Generate embeddings, train ML models, and build Agentic RAG workflows for job market intelligence.

**Status:** 🔲 **READY TO START** (Dec 23, 2025)

**Dependencies Met:**
- ✅ ETL Pipeline deployed (cleaned_jobs data available in BigQuery)
- ✅ BigQuery streaming API ready (for writing embeddings back)
- ✅ GenAI folder scaffolded (`/genai/` with placeholders)
- ✅ ML/NLP folders exist (`/ml/`, `/nlp/` with placeholders)

**What's Next:** Phase 3A - NLP Embeddings Generation

**Virtual Environment Usage:**
- ⚠️ **CRITICAL:** Always use `.venv/Scripts/python.exe` for all Python commands
- Install dependencies: `.venv/Scripts/python.exe -m pip install <package>`
- Run training: `.venv/Scripts/python.exe -m ml.train`
- Update `requirements.txt` when adding new packages

---

# Strategic Decision: Manual Coding vs Vertex AI AutoML

## Why Build From Scratch First?

| Aspect | Manual Coding (Our Approach ✅) | Vertex AI AutoML |
|--------|--------------------------------|------------------|
| **Learning Value** | ⭐⭐⭐⭐⭐ Deep understanding | ⭐⭐ Black box |
| **Learning Result** | Can explain internals | "I used AutoML" |
| **Customization** | Full control | Limited options |
| **Cost** | FREE (local training) | $$$$ (training + hosting) |
| **Production Scale** | Requires more work | Easy deployment |

## Hybrid Approach (Recommended)

```
Phase 1: BUILD FROM SCRATCH (Learning & Portfolio)
├── Implement embeddings manually (understand transformers)
├── Train LightGBM yourself (understand gradient boosting)
├── Build clustering from sklearn (understand unsupervised learning)
└── Learn WHY each decision works

Phase 2: COMPARE WITH VERTEX AI (Validation)
├── Try Vertex AI AutoML on same data
├── Compare metrics (your model vs AutoML)
├── Understand when AutoML is better/worse
└── Document tradeoffs

Phase 3: PRODUCTION (Real World)
├── Use Vertex AI for model serving (scalability)
├── But keep custom model logic for flexibility
└── Best of both worlds
```

---

# Technical Stack

| Category | Libraries | Purpose |
|----------|-----------|---------|
| **NLP** | `sentence-transformers`, `spacy` | Embeddings, tokenization |
| **ML** | `scikit-learn`, `lightgbm`, `xgboost` | Training & evaluation |
| **Data** | `pandas`, `numpy`, `pyarrow` | Data manipulation |
| **Visualization** | `matplotlib`, `seaborn`, `plotly` | Charts & EDA |
| **BigQuery** | `google-cloud-bigquery` | Data I/O, Vector Search |
| **GenAI** | `langchain`, `langgraph`, `google-cloud-aiplatform` | RAG, Agents |
| **Experiment Tracking** | `mlflow` (optional) or BigQuery logging | Model versioning |

---

# Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ PHASE 3A: NLP EMBEDDINGS                                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│ BigQuery: cleaned_jobs                                                      │
│         ↓ (query job_description, job_title)                                │
│ Sentence-BERT: all-MiniLM-L6-v2 (384 dimensions)                            │
│         ↓ (batch embedding generation)                                      │
│ BigQuery: job_embeddings table (job_id, embedding ARRAY<FLOAT64>)           │
│         ↓ (create vector index)                                             │
│ BigQuery Vector Search: Ready for similarity queries                        │
└─────────────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│ PHASE 3B: FEATURE ENGINEERING                                               │
├─────────────────────────────────────────────────────────────────────────────┤
│ Input: cleaned_jobs + job_embeddings                                        │
│         ↓                                                                   │
│ Numerical: salary_min, salary_max (log transform, imputation)               │
│ Categorical: location, work_type, classification (one-hot/label encoding)   │
│ Text: title + description embeddings (384-dim vector)                       │
│ Temporal: days_since_posted, is_weekend_post                                │
│         ↓                                                                   │
│ Output: ml_features table (job_id + all features)                           │
└─────────────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│ PHASE 3C: MODEL TRAINING                                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│ SUPERVISED                          │ UNSUPERVISED                          │
│ ─────────────────────────────────── │ ──────────────────────────────────────│
│ Salary Prediction (Regression)      │ Job Clustering (KMeans)               │
│   • Target: salary_mid_monthly      │   • Input: embeddings + features      │
│   • Models: LightGBM, XGBoost       │   • Output: cluster_id (0-9)          │
│   • Metric: RMSE, MAE, R²           │   • Metric: Silhouette, Inertia       │
│                                     │                                       │
│ Role Classification (Multi-class)   │ Dimensionality Reduction (PCA/UMAP)   │
│   • Target: job_classification      │   • Input: 384-dim embeddings         │
│   • Models: LightGBM, LogReg        │   • Output: 2D/3D for visualization   │
│   • Metric: F1-macro, Accuracy      │   • Purpose: Cluster visualization    │
└─────────────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│ PHASE 3D: MODEL ARTIFACTS                                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│ Local: /models/{model_name}/{version}/                                      │
│   • model.joblib (serialized model)                                         │
│   • config.json (hyperparameters)                                           │
│   • metrics.json (evaluation results)                                       │
│                                                                             │
│ GCS: gs://sg-job-market-data/models/{model_name}/{version}/                 │
│   • Same structure, for production deployment                               │
│                                                                             │
│ BigQuery: ml_predictions table (job_id, predicted_salary, cluster_id, etc.) │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

# Phase 3A: NLP Embeddings Generation

**Goal:** Generate semantic embeddings for all job descriptions to enable similarity search and clustering.

## 3A.0: Conceptual Understanding

### What are Embeddings?
Embeddings convert text → dense numerical vectors where **similar meanings = similar vectors**.

```
Raw Text                              Embedding (384 floats)
─────────────────                     ────────────────────────
"Senior Data Scientist"        →      [0.23, -0.45, 0.67, ...]
"Data Scientist Lead"          →      [0.22, -0.44, 0.69, ...]  ← Similar!
"Restaurant Manager"           →      [-0.78, 0.91, -0.12, ...] ← Different!
```

### Why Sentence-BERT (SBERT) and Not BM25?

| Aspect | Sentence-BERT (SBERT) ✅ | BM25 |
|--------|--------------------------|------|
| **Type** | Dense embeddings (neural network) | Sparse (term frequency-inverse doc frequency) |
| **Output** | 384 floats per document | Inverted index (word → documents) |
| **Similarity** | Cosine similarity in vector space | TF-IDF scoring |
| **Semantic Understanding** | ✅ "Software Engineer" ≈ "Developer" | ❌ Exact word match only |
| **Clustering Support** | ✅ KMeans needs dense vectors | ❌ Cannot cluster |
| **Storage** | ~1.5KB per job (384 × 4 bytes) | Variable, often larger |
| **Speed** | Slower to generate (neural forward pass) | Faster (counting) |
| **Use Case** | Semantic search, clustering, ML features | Keyword search |

**Why SBERT for our project:**
1. **Job descriptions are semantic** - "Python Developer" and "Python Engineer" should match
2. **Clustering requires vectors** - KMeans requires dense numerical input
3. **ML features** - Embeddings become input features for salary prediction
4. **Industry standard** - Every modern search/recommendation system uses embeddings

**Advanced: Hybrid BM25 + SBERT (for future RAG):**
```python
def hybrid_search(query):
    # Step 1: BM25 retrieves top 100 candidates (fast, broad recall)
    candidates = bm25_search(query, top_k=100)
    
    # Step 2: SBERT reranks by semantic similarity (precise)
    query_embedding = sbert.encode(query)
    reranked = sort_by_cosine_similarity(candidates, query_embedding)
    return reranked[:10]
```

### Deep Dive: Hybrid BM25 + SBERT Search

#### Understanding BM25 (Best Match 25)

BM25 is a **sparse retrieval** algorithm based on term frequency.

**Formula (simplified):**
```
BM25(query, document) = Σ IDF(word) × TF(word, document) × (k1 + 1) / (TF + k1)

Where:
- IDF(word) = log((N - df + 0.5) / (df + 0.5))
  - N = total documents
  - df = document frequency (how many docs contain this word)
- TF(word, doc) = term frequency (how often word appears in doc)
- k1 = tuning parameter (usually 1.2-2.0)
```

**Example: Why BM25 Finds Related Jobs**

**Query:** "Python Developer"

**Job 1:** "Python Developer with 3 years experience..."
**Job 2:** "Software Engineer proficient in Python..."
**Job 3:** "Restaurant Manager..."

**BM25 Scoring:**

```python
# Step 1: Tokenize query
query_terms = ["Python", "Developer"]

# Step 2: Calculate IDF for each term (rare words get higher scores)
IDF("Python") = log((10000 - 1500 + 0.5) / (1500 + 0.5)) = 1.74  ← Common in tech

IDF("Developer") = log((10000 - 2000 + 0.5) / (2000 + 0.5)) = 1.38  ← Also common

# Step 3: Calculate BM25 for each job
Job 1 Score:
  "Python" appears 2 times → TF = 2
  "Developer" appears 3 times → TF = 3
  BM25 = (1.74 × f(2)) + (1.38 × f(3)) = 4.2

Job 2 Score:
  "Python" appears 1 time → TF = 1
  "Developer" NOT in doc → TF = 0
  BM25 = (1.74 × f(1)) + (1.38 × 0) = 1.5

Job 3 Score:
  No query terms → BM25 = 0
```

**Result:** Job 1 ranks higher even though Job 2 says "Software Engineer" instead of "Developer".

#### Why BM25 Alone Is Not Enough

```
Query: "machine learning jobs"

BM25 Results:
1. "Machine Learning Engineer" (exact match) ✅
2. "Data Scientist with ML experience" (contains "ML") ✅
3. "AI Researcher specializing in neural networks" ❌ MISSED!
   - No words "machine", "learning", or "ML"
   - But semantically VERY relevant

4. "Looking for machine learning internship" ⚠️ FALSE POSITIVE
   - Contains "machine learning" but it's an internship posting
```

#### Hybrid Approach: BM25 + SBERT in Detail

```python
def hybrid_search(query: str, top_k: int = 10):
    """
    Combine BM25 (fast, broad) with SBERT (semantic, precise).
    """
    # Phase 1: BM25 retrieves 100 candidates (0.5 seconds)
    bm25_candidates = bm25_index.search(query, top_k=100)
    
    # Phase 2: SBERT reranks candidates (1 second)
    candidate_embeddings = get_embeddings(bm25_candidates)
    query_embedding = embed_text(query)
    similarities = cosine_similarity(query_embedding, candidate_embeddings)
    reranked = sort_by_similarity(bm25_candidates, similarities)
    
    return reranked[:top_k]
```

**Example Flow:**

```
User Query: "AI jobs in Singapore"

PHASE 1: BM25 (Fast Keyword Retrieval)
────────────────────────────────────────
Retrieves 100 jobs containing:
- "AI" (75 jobs)
- "artificial intelligence" (30 jobs)
- "Singapore" (100 jobs)

Top 10 BM25 Results:
1. "AI Engineer, Singapore" (score: 8.5)
2. "Data Scientist - AI, Singapore" (score: 7.8)
3. "Singapore AI Research" (score: 7.2)
4. "Machine Learning Singapore" (score: 5.1) ← Different words!
5. "Singapore Software Engineer" (score: 4.8) ← Less relevant
...

PHASE 2: SBERT (Semantic Reranking)
────────────────────────────────────────
Calculate semantic similarity:

query_emb = embed("AI jobs in Singapore")
# → [0.21, -0.45, 0.67, ..., 0.33]

job_1_emb = embed("AI Engineer, Singapore...")
# → [0.22, -0.44, 0.65, ..., 0.31]  ← Very similar!
cosine_sim = 0.92

job_4_emb = embed("Machine Learning Singapore...")
# → [0.20, -0.43, 0.64, ..., 0.29]  ← Also similar!
cosine_sim = 0.89  ← Boosted from rank 4!

job_5_emb = embed("Singapore Software Engineer...")
# → [-0.15, 0.32, -0.21, ..., -0.08]  ← Not similar
cosine_sim = 0.35  ← Demoted!

Final Reranked Results:
1. "AI Engineer, Singapore" (similarity: 0.92) ✅
2. "Data Scientist - AI, Singapore" (similarity: 0.90) ✅
3. "Machine Learning Singapore" (similarity: 0.89) ⬆️ Jumped from #4
4. "Deep Learning Engineer" (similarity: 0.87) ⬆️ Jumped from #12
5. "AI Research Scientist" (similarity: 0.85) ✅
```

**Why This Works:**
1. **BM25 ensures recall** - Won't miss jobs with exact keywords
2. **SBERT adds semantic understanding** - Finds "ML Engineer" when you search "AI jobs"
3. **Fast** - BM25 narrows 10K jobs → 100, SBERT only reranks 100

### Why 384 Dimensions?

It's determined by the **model architecture**, not our choice:

| Model | Dimensions | Architecture | Speed |
|-------|------------|--------------|-------|
| all-MiniLM-L6-v2 ✅ | 384 | 6-layer transformer, 384 hidden units | ⚡ Fast |
| all-mpnet-base-v2 | 768 | 12-layer transformer, 768 hidden units | Medium |
| OpenAI text-embedding-3-small | 1536 | Larger architecture | Slow (API) |

**Tradeoff:** More dimensions = more semantic information but more storage/computation.
**384 is the sweet spot** for most use cases (good quality, fast, small storage).

### Why BigQuery for Vector Storage (Not ChromaDB/Pinecone)?

| Aspect | BigQuery Vector Search ✅ | ChromaDB | Pinecone |
|--------|--------------------------|----------|----------|
| **Type** | Data warehouse + vectors | Vector-only DB | Vector-only DB |
| **Cost** | $5/TB scanned (free tier!) | Free (local) | $70/month+ |
| **Scalability** | Billions of rows | Millions | Billions |
| **Query Speed** | ~100ms (with index) | ~10ms | ~10ms |
| **Integration** | Already using BQ ✅ | Separate service | Separate service |
| **Append-Only** | ✅ Perfect fit | ✅ Supports | ✅ Supports |
| **SQL Analytics** | ✅ JOIN with job data | ❌ Vectors only | ❌ Vectors only |

**Why BigQuery for us:**
1. **Already using BQ** - No new infrastructure to manage
2. **JOIN capability** - `SELECT * FROM jobs JOIN embeddings` in one query
3. **Cost efficient** - Free tier covers our volume (~10K jobs)
4. **Append-only fits our model** - We don't update embeddings, just add new ones

**When to use ChromaDB:** Local dev/testing, RAG with sub-10ms latency needs.

### Deep Dive: BigQuery SQL JOINs with Embeddings

**Architecture:**
```
Traditional Approach (Separate Databases):
┌─────────────────┐           ┌─────────────────┐
│   BigQuery      │           │   Pinecone      │
│   (Job Data)    │           │   (Embeddings)  │
│                 │           │                 │
│ job_id          │           │ job_id          │
│ job_title       │  NO JOIN  │ embedding[384]  │
│ salary          │           │ model_name      │
└─────────────────┘           └─────────────────┘
         ↓                           ↓
    Query both, combine in Python (slow!)


Our Approach (BigQuery for Both):
┌───────────────────────────────────────────┐
│            BigQuery                       │
│                                           │
│  ┌──────────────┐      ┌───────────────┐  │
│  │ cleaned_jobs │      │ job_embeddings│  │
│  │              │ JOIN │               │  │
│  │ job_id       │──────│ job_id        │  │
│  │ salary       │      │ embedding[384]│  │
│  └──────────────┘      └───────────────┘  │
└───────────────────────────────────────────┘
         ↓
    Single SQL query (fast!)
```

**Example 1: Find High-Paying Jobs Similar to "Data Scientist"**

```sql
-- Step 1: Get embedding for "Data Scientist" query
DECLARE query_embedding ARRAY<FLOAT64>;
SET query_embedding = (
  SELECT embedding 
  FROM job_embeddings 
  WHERE job_id = 'reference-data-scientist-job'
);

-- Step 2: Find similar high-paying jobs
SELECT 
  c.job_id,
  c.job_title,
  c.job_salary_max_sgd_monthly AS salary,
  c.company_name,
  c.job_location,
  -- Calculate cosine similarity using embeddings
  (1 - COSINE_DISTANCE(e.embedding, query_embedding)) AS similarity_score
FROM 
  cleaned_jobs c
  JOIN job_embeddings e 
    ON c.job_id = e.job_id AND c.source = e.source
WHERE 
  c.job_salary_max_sgd_monthly > 7000  -- High paying only
  AND (1 - COSINE_DISTANCE(e.embedding, query_embedding)) > 0.7  -- Similar
ORDER BY 
  similarity_score DESC
LIMIT 10;
```

**Output:**
| job_title | salary | similarity_score | location |
|-----------|--------|------------------|----------|
| Senior Data Scientist | $9,000 | 0.92 | Singapore - CBD |
| ML Engineer | $8,500 | 0.87 | Singapore - West |
| Data Science Lead | $10,000 | 0.85 | Singapore - Central |

**Example 2: Cluster Analysis with Salary Insights**

```sql
-- Get average salary per cluster with job titles
SELECT 
  cl.cluster_id,
  cl.cluster_name,
  COUNT(*) AS num_jobs,
  AVG(c.job_salary_mid_sgd_monthly) AS avg_salary,
  APPROX_TOP_COUNT(c.job_title, 5) AS top_titles
FROM 
  ml_predictions cl
  JOIN cleaned_jobs c 
    ON cl.job_id = c.job_id AND cl.source = c.source
  JOIN job_embeddings e
    ON c.job_id = e.job_id AND c.source = e.source
GROUP BY 
  cl.cluster_id, cl.cluster_name
ORDER BY 
  avg_salary DESC;
```

**Output:**
| cluster_name | num_jobs | avg_salary | top_titles |
|--------------|----------|------------|------------|
| Tech/Software | 2,300 | $7,200 | Software Engineer, Developer, Tech Lead |
| Finance | 850 | $6,800 | Financial Analyst, Accountant |
| Healthcare | 650 | $5,500 | Nurse, Medical Officer |

**Example 3: Recommendation System**

```sql
-- User viewed job_id = 'abc123'
-- Find 10 similar jobs they might like
WITH user_viewed_embedding AS (
  SELECT embedding
  FROM job_embeddings
  WHERE job_id = 'abc123'
)

SELECT 
  c.job_id,
  c.job_title,
  c.company_name,
  c.job_location,
  c.job_salary_max_sgd_monthly,
  (1 - COSINE_DISTANCE(e.embedding, uve.embedding)) AS match_score
FROM 
  cleaned_jobs c
  JOIN job_embeddings e ON c.job_id = e.job_id
  CROSS JOIN user_viewed_embedding uve
WHERE 
  c.job_id != 'abc123'  -- Don't recommend same job
  AND c.job_classification = (
    SELECT job_classification FROM cleaned_jobs WHERE job_id = 'abc123'
  )
ORDER BY 
  match_score DESC
LIMIT 10;
```

**Cost Comparison:**

**BigQuery:**
- Storage: $0.02/GB/month (embeddings: 10K jobs × 1.5KB = 15MB = **$0.0003/month**)
- Queries: $5/TB scanned (with free tier: **FREE**)

**Pinecone:**
- Starter: $70/month for 100K vectors
- For 10K jobs: **$7/month minimum**

## 3A.1: Embedding Model Selection

| Model | Dimensions | Speed | Quality | Use Case |
|-------|------------|-------|---------|----------|
| `all-MiniLM-L6-v2` ✅ | 384 | Fast | Good | **CHOSEN** - Best balance |
| `all-mpnet-base-v2` | 768 | Medium | Better | If quality is critical |
| `text-embedding-004` (Vertex AI) | 768 | API call | Best | Production with budget |

**Decision:** Use `all-MiniLM-L6-v2` for initial implementation (free, local, fast).
Can upgrade to Vertex AI embeddings later for production.

## 3A.2: Implementation Tasks

### Task 3A.2.1: Create Embedding Pipeline
- [ ] Create `nlp/embeddings.py`:
  ```python
  # Core functions to implement:
  def load_model(model_name: str = "all-MiniLM-L6-v2") -> SentenceTransformer
  def embed_texts(texts: List[str], batch_size: int = 32) -> np.ndarray
  def embed_jobs_from_bq(limit: int = None) -> Dict[str, Any]
  ```
- [ ] Add batched processing (32-64 texts per batch, GPU-aware)
- [ ] Add progress bar with tqdm
- [ ] Handle empty/null descriptions gracefully
- [ ] Log embedding statistics (min, max, mean values)

### Task 3A.2.2: BigQuery Schema for Embeddings
- [ ] Update `utils/schemas.py` with `JobEmbedding` dataclass:
  ```python
  @dataclass
  class JobEmbedding:
      job_id: str
      source: str
      embedding: List[float]  # 384 dimensions
      model_name: str
      created_at: datetime
  ```
- [ ] Create `job_embeddings` table in BigQuery
- [ ] Partition by `created_at`, cluster by `source, job_id`

### Task 3A.2.3: Embedding Generation Script
- [ ] Create `nlp/generate_embeddings.py`:
  - Query `cleaned_jobs` for job_id, job_title, job_description
  - Combine: `f"{title}. {description[:1000]}"` (truncate for efficiency)
  - Generate embeddings in batches
  - Stream to BigQuery `job_embeddings` table
  - Support incremental updates (only embed new jobs)
- [ ] Add CLI: `.venv/Scripts/python.exe -m nlp.generate_embeddings --limit 1000`
- [ ] Add tests: `tests/test_embeddings.py`

### Task 3A.2.4: BigQuery Vector Index
- [ ] Create vector index for similarity search:
  ```sql
  CREATE VECTOR INDEX job_embedding_idx
  ON `sg-job-market.sg_job_market.job_embeddings`(embedding)
  OPTIONS(distance_type='COSINE', index_type='IVF', ivf_options='{"num_lists": 100}');
  ```
- [ ] Create similarity search function:
  ```python
  def find_similar_jobs(query_embedding: List[float], top_k: int = 10) -> List[Dict]
  ```
- [ ] Test with sample queries

**Acceptance Criteria:**
- [ ] All cleaned_jobs have embeddings in BigQuery
- [ ] Vector index created and queryable
- [ ] Similar job search returns relevant results
- [ ] Processing time: <5 minutes for 10K jobs

---

# Phase 3B: Feature Engineering

**Goal:** Create ML-ready features from cleaned jobs and embeddings.

## 3B.0: Conceptual Understanding

### What is Feature Engineering?
**Feature Engineering = Converting raw data → numbers that ML models can understand**

```
Raw Job Posting                         ML Features (Numbers)
─────────────────                       ─────────────────────
"Senior Data Scientist at               salary_min: 8000
Google, 8000-12000 SGD,          →      salary_max: 12000
Full-time, Singapore,                   is_fulltime: 1
5 years experience..."                  location_singapore: 1
                                        description_length: 2341
                                        embedding[0:384]: [0.23, ...]
```

### Why is Feature Engineering Important?
- **ML models only understand numbers** - Can't feed raw text directly
- **Good features = good predictions** - "Garbage in, garbage out"
- **Domain knowledge encoded** - salary_range might indicate job level
- **80% of ML work** - Data scientists spend most time here, not on model tuning

### What Each Feature Type Captures

| Feature Type | Examples | What it Captures | Why Useful |
|-------------|----------|------------------|------------|
| **Numerical** | salary_min, description_length | Direct measurements | Continuous relationships |
| **Categorical** | location, work_type | Group membership | Different baselines per group |
| **Temporal** | days_since_posted | Time patterns | Fresh jobs might pay differently |
| **Embeddings** | 384-dim vector | Semantic meaning of text | Role/industry context |

### Why 384 Embedding Dimensions in Features?
The 384 comes from our chosen SBERT model (all-MiniLM-L6-v2). Each dimension is a **learned semantic feature** - we can't interpret individual dimensions, but together they capture meaning.

**For ML, we might reduce to 10-50 dimensions using PCA:**
- 384 raw dimensions can cause overfitting with small datasets
- PCA preserves most information in fewer dimensions
- Faster training and inference

### Should ml_features Be a Table or View?

**Recommendation: VIEW for computed features, TABLE only for embeddings**

```sql
-- VIEW (computed on-the-fly, always fresh)
CREATE VIEW vw_ml_features AS
SELECT 
  job_id,
  (salary_min + salary_max) / 2 AS salary_mid,  -- Cheap to compute
  LENGTH(job_description) AS desc_length,        -- Cheap to compute
FROM cleaned_jobs;

-- TABLE (for expensive pre-computed data)
CREATE TABLE job_embeddings (
  job_id STRING,
  embedding ARRAY<FLOAT64>  -- Expensive to compute, store once
);
```

**Why:**
- Views = always up-to-date, no sync issues
- Tables = faster but can become stale
- Embeddings are expensive (neural network) → store in table
- Simple SQL features are cheap → compute in view

## 3B.1: Feature Categories

### Numerical Features
| Feature | Source | Transformation |
|---------|--------|----------------|
| `salary_min_monthly` | cleaned_jobs | Log transform, impute median |
| `salary_max_monthly` | cleaned_jobs | Log transform, impute median |
| `salary_mid_monthly` | Derived | `(min + max) / 2` |
| `salary_range` | Derived | `max - min` |
| `days_since_posted` | cleaned_jobs | `NOW() - job_posted_timestamp` |
| `description_length` | cleaned_jobs | `LEN(job_description)` |
| `title_length` | cleaned_jobs | `LEN(job_title)` |

### Categorical Features
| Feature | Source | Encoding |
|---------|--------|----------|
| `source` | cleaned_jobs | One-hot (2 categories) |
| `job_location` | cleaned_jobs | One-hot or target encoding |
| `job_classification` | cleaned_jobs | Label encoding (for target) |
| `job_work_type` | cleaned_jobs | One-hot |
| `company_industry` | cleaned_jobs | One-hot or target encoding |
| `company_size` | cleaned_jobs | Ordinal encoding |

### Embedding Features
| Feature | Source | Shape |
|---------|--------|-------|
| `embedding` | job_embeddings | 384 floats |
| `embedding_pca_2d` | Derived | 2 floats (for visualization) |
| `embedding_pca_10d` | Derived | 10 floats (for ML) |

## 3B.2: Implementation Tasks

### Task 3B.2.1: Feature Engineering Module
- [ ] Create `ml/features.py`:
  ```python
  def extract_numerical_features(df: pd.DataFrame) -> pd.DataFrame
  def extract_categorical_features(df: pd.DataFrame) -> pd.DataFrame
  def create_feature_matrix(df: pd.DataFrame, embeddings: np.ndarray) -> pd.DataFrame
  ```
- [ ] Handle missing values (imputation strategies)
- [ ] Create feature pipeline (sklearn Pipeline)
- [ ] Save feature transformers (for inference)

### Task 3B.2.2: Feature Storage
- [ ] Create BigQuery view `vw_ml_features`:
  ```sql
  CREATE VIEW vw_ml_features AS
  SELECT 
    c.job_id,
    c.source,
    c.job_title,
    c.job_classification,
    c.job_location,
    c.job_work_type,
    c.job_salary_min_sgd_monthly,
    c.job_salary_max_sgd_monthly,
    (c.job_salary_min_sgd_monthly + c.job_salary_max_sgd_monthly) / 2 AS salary_mid,
    TIMESTAMP_DIFF(CURRENT_TIMESTAMP(), c.job_posted_timestamp, DAY) AS days_since_posted,
    LENGTH(c.job_description) AS description_length,
    e.embedding
  FROM cleaned_jobs c
  JOIN job_embeddings e ON c.job_id = e.job_id AND c.source = e.source
  WHERE c.job_salary_min_sgd_monthly IS NOT NULL
  ```

### Task 3B.2.3: Data Splitting Strategy
- [ ] Implement time-based split (not random):
  - Train: Jobs posted before cutoff date
  - Validation: Jobs posted after cutoff
  - Test: Most recent 10% of jobs
- [ ] Stratify by job_classification and salary_range
- [ ] Document split ratios and date ranges

**Acceptance Criteria:**
- [ ] Feature matrix created with all features
- [ ] No data leakage between train/val/test
- [ ] Feature importance analysis completed
- [ ] Documentation of all feature transformations

---

# Phase 3C: Model Training

**Goal:** Train and evaluate salary prediction and clustering models.

## 3C.0: Conceptual Understanding

### Why LightGBM Over XGBoost or Others?

| Aspect | LightGBM ✅ | XGBoost | Random Forest | Linear Regression |
|--------|------------|---------|---------------|-------------------|
| **Speed** | ⚡ Fastest | Medium | Slowest | ⚡ Fastest |
| **Memory** | Low | Medium | High | Lowest |
| **Accuracy** | Excellent | Excellent | Good | Poor (non-linear) |
| **Handles Categorical** | ✅ Native | ❌ Need encoding | ❌ Need encoding | ❌ Need encoding |
| **Handles Missing** | ✅ Native | ✅ Native | ❌ Need imputation | ❌ Need imputation |
| **Tree Growth** | Leaf-wise (deeper) | Level-wise (balanced) | Level-wise | N/A |
| **Overfitting Risk** | Higher | Lower | Lowest | Lowest |
| **Hyperparameter Sensitivity** | More sensitive | More forgiving | Least sensitive | None |
| **Interpretability** | Feature importance | Feature importance | Feature importance | Coefficients |

**Why LightGBM for our project:**
1. **Native categorical handling** - job_location, work_type don't need one-hot encoding
2. **Speed** - Fast iteration during development and hyperparameter tuning
3. **Industry standard** - Used at Microsoft, Alibaba, most Kaggle winners
4. **Good with embeddings** - Handles high-dimensional features well

**Learning Result:** "LightGBM uses leaf-wise tree growth which creates deeper, more specialized trees faster than XGBoost's level-wise approach. For our tabular data with many categorical features like job_location and work_type, LightGBM's native categorical handling avoids the curse of dimensionality from one-hot encoding (Singapore alone has 50+ neighborhoods). The tradeoff is higher overfitting risk, which we mitigate with early stopping, cross-validation, and regularization parameters like min_child_samples."

### Deep Dive: Leaf-Wise vs Level-Wise Tree Growth

**Visual Comparison:**

```
LEVEL-WISE TREE GROWTH (XGBoost, Random Forest)
================================================
Split all nodes at the same depth first

Level 0:           [Root]
                  /      \
Level 1:      [Node A]  [Node B]     ← Split BOTH nodes before going deeper
                /  \      /  \
Level 2:    [C] [D]   [E] [F]        ← Then split all 4 nodes

Advantage: Balanced tree, less overfitting
Disadvantage: Wastes splits on nodes that don't matter much


LEAF-WISE TREE GROWTH (LightGBM)
=================================
Split the leaf that reduces loss the MOST

Step 1:            [Root]
                  /      \
Step 2:      [Node A]  [Node B]      ← Node A reduces loss more
                /  \         
Step 3:    [C] [D]  [Node B]         ← Split Node A first
              /  \
Step 4:   [E] [F] [D] [Node B]       ← Keep splitting best leaves

Advantage: Deeper trees, better accuracy, faster training
Disadvantage: Can overfit if not careful
```

**Real-World Example with Salary Data:**

```
Leaf-Wise Approach (LightGBM):
Root: Split by location (Singapore vs Rest)
├── Singapore (avg salary: $5,000)
│   ├── Split by industry (Tech: $7,000 vs Non-Tech: $3,000) ← Big gain!
│   │   ├── Split Tech by years_exp (Senior vs Junior) ← Keep going deeper
│   │   │   ├── Senior: $9,000
│   │   │   └── Junior: $5,000
│   └── Non-Tech stays as leaf (not much to gain)
└── Rest stays as leaf (few samples, not worth splitting)

Focuses on splitting where it matters most
```

**Native Categorical Handling:**

```python
# XGBoost requires one-hot encoding
location = "Singapore - Downtown"
# Becomes: location_sg_downtown=1, location_sg_jurong=0, ... (50+ columns!)

# LightGBM handles categorical natively
model.fit(X, y, categorical_feature=['job_location', 'job_work_type'])
# Stores as a single column with smart split logic
```

**For Singapore job data:**
- `job_location`: 50+ unique neighborhoods → 50 columns (XGBoost) vs 1 column (LightGBM)
- One-hot encoding explosion: 100+ dummy columns → LightGBM keeps them as 3 columns

**Speed Comparison on Our Data:**

| Dataset Size | LightGBM | XGBoost | Speedup |
|--------------|----------|---------|---------|
| 1,000 jobs | 5 seconds | 12 seconds | 2.4x |
| 10,000 jobs | 30 seconds | 90 seconds | 3x |
| 100,000 jobs | 5 minutes | 20 minutes | 4x |

### Understanding Evaluation Metrics

#### For Classification: The Confusion Matrix

```
                        Predicted
                    Positive    Negative
                  ┌─────────────┬───────────┐
Actual Positive   │    TP       │    FN     │
                  │  (Hit!)     │ (Missed!) │
                  ├─────────────┼───────────┤
Actual Negative   │    FP       │    TN     │
                  │(False Alarm)│ (Correct) │
                  └─────────────┴───────────┘

TP = True Positive  → Predicted IT job, Actually IT job ✅
TN = True Negative  → Predicted NOT IT, Actually NOT IT ✅
FP = False Positive → Predicted IT job, Actually Finance ❌ (Type I Error)
FN = False Negative → Predicted Finance, Actually IT ❌ (Type II Error)
```

**Derived Metrics:**

| Metric | Formula | Plain English | When to Use |
|--------|---------|---------------|-------------|
| **Accuracy** | (TP+TN) / All | "% of all predictions correct" | Balanced classes only |
| **Precision** | TP / (TP+FP) | "When I say positive, how often am I right?" | Cost of FP is high (spam filter) |
| **Recall** | TP / (TP+FN) | "Of all actual positives, how many did I find?" | Cost of FN is high (cancer detection) |
| **F1 Score** | 2×P×R / (P+R) | "Balance of precision and recall" | Imbalanced classes |
| **F1 Macro** | Average F1 across all classes | "Equal weight to all classes" | Multi-class, care about minority |
| **F1 Weighted** | Weighted average by class size | "Proportional to class frequency" | Multi-class, care about majority |

**Example:**
- 100 IT jobs, 10 Finance jobs
- Model predicts all as IT → Accuracy = 91% (misleading!)
- F1 Macro = 0.45 (honest, shows Finance performance is bad)

#### For Regression: Error Metrics

| Metric | Formula | Plain English | Interpretation |
|--------|---------|---------------|----------------|
| **RMSE** | √(Σ(y-ŷ)²/n) | "Average error, penalizing big mistakes" | RMSE=$1500 → typical error is $1500 |
| **MAE** | Σ\|y-ŷ\|/n | "Average absolute error" | MAE=$1000 → average miss by $1000 |
| **R²** | 1 - (SS_res/SS_tot) | "% of variance explained" | R²=0.7 → model explains 70% of salary variation |
| **MAPE** | Σ(\|y-ŷ\|/y)/n × 100 | "Average % error" | MAPE=10% → typically off by 10% |

**Our Targets:**
- RMSE < $1,500 → "Predictions typically within $1,500 of actual salary"
- R² > 0.7 → "Model explains 70%+ of why salaries differ"
- F1 Macro > 0.6 → "Balanced performance across all job categories"

### Deep Dive: RMSE and R² Explained

#### RMSE (Root Mean Square Error)

**Formula:** `RMSE = √(Σ(actual - predicted)² / n)`

**Step-by-Step Example:**

| Job | Actual Salary | Predicted Salary | Error | Squared Error |
|-----|---------------|------------------|-------|---------------|
| 1 | $5,000 | $5,200 | +$200 | 40,000 |
| 2 | $7,000 | $6,500 | -$500 | 250,000 |
| 3 | $4,500 | $4,600 | +$100 | 10,000 |
| 4 | $9,000 | $8,200 | -$800 | 640,000 |
| 5 | $6,000 | $6,100 | +$100 | 10,000 |

```python
squared_errors = [40000, 250,000, 10000, 640000, 10000]
mean_squared_error = sum(squared_errors) / 5 = 190,000
RMSE = √190,000 = $436
```

**What Does RMSE = $436 Mean?**
- On average, predictions are **off by $436**
- **Interpretation:** "Typical prediction error is around $436"

**Why Square the Errors?**
1. **Penalizes large errors** - Being $1,000 off is worse than being $100 off 10 times
2. **Makes math work** - Negatives cancel out without squaring

**Our Target: RMSE < $1,500**
```
If RMSE = $1,200:
- User sees: "This job probably pays $5,000 ± $1,200"
- Acceptable for job hunting ✅

If RMSE = $3,000:
- User sees: "This job probably pays $2,000-$8,000"
- NOT useful! Range too wide ❌
```

#### R² (R-Squared / Coefficient of Determination)

R² answers: **"How much of the salary variation does my model explain?"**

**Formula:** `R² = 1 - (SS_residual / SS_total)`

**Example Calculation:**

```python
# Dataset: 5 jobs
actual_salaries = [5000, 7000, 4500, 9000, 6000]
mean_salary = 6300

# Baseline: Always predict the mean
baseline_predictions = [6300, 6300, 6300, 6300, 6300]

# Your model's predictions
model_predictions = [5200, 6500, 4600, 8200, 6100]

# Calculate SS_total (baseline error)
SS_total = (5000-6300)² + (7000-6300)² + ... = 12,800,000

# Calculate SS_residual (your model's error)
SS_residual = (5000-5200)² + (7000-6500)² + ... = 950,000

# Calculate R²
R² = 1 - (950,000 / 12,800,000) = 0.926 (92.6%)
```

**What Does R² = 0.926 Mean?**
1. "My model explains 92.6% of salary variation"
2. "My model reduces prediction error by 92.6% vs just guessing the average"
3. "92.6% of why salaries differ is captured by my features"

**Visual Explanation:**

```
If you just guess the mean ($6,300) every time:
       Actual: $5,000  vs  Guess: $6,300  → Error: $1,300
       Actual: $9,000  vs  Guess: $6,300  → Error: $2,700
       Total Error: $12,800,000

Your model predictions:
       Actual: $5,000  vs  Predicted: $5,200  → Error: $200  ✅
       Actual: $9,000  vs  Predicted: $8,200  → Error: $800  ✅
       Total Error: $950,000

Error Reduction: 92.6%  ← This is R²!
```

**R² Values Interpretation:**

| R² Value | Meaning | Action |
|----------|---------|--------|
| **R² > 0.9** | Excellent! Model explains 90%+ of variation | Ship to production |
| **R² = 0.7-0.9** | Good. Explains most variation | ✅ Our target |
| **R² = 0.5-0.7** | Moderate. Missing important features | Add more features |
| **R² < 0.5** | Poor. Model barely better than guessing mean | Rethink approach |

**Summary**

> **RMSE tells the average dollar error** - with RMSE = $1,200, we can tell users 'this job likely pays $5,000 ± $1,200'. 
>
> **R² tells us if we're capturing the right patterns** - with R² = 0.75, we know our features (location, title embeddings, work type) explain 75% of why salaries differ. The remaining 25% might be explained by factors we don't have, like years of experience or company size. 
>
> If R² is low (<0.5) even with acceptable RMSE, it signals we need to extract more features from the job descriptions, perhaps using NLP to identify required skills, tech stack, or seniority level."

### Understanding Clustering Metrics

#### Why KMeans Over Other Algorithms?

| Algorithm | KMeans ✅ | DBSCAN | Hierarchical | GMM |
|-----------|----------|--------|--------------|-----|
| **Requires k?** | ✅ Must specify | ❌ Auto-detects | ❌ Dendrogram | ✅ Must specify |
| **Cluster Shape** | Spherical only | Any shape | Any shape | Elliptical |
| **Speed** | ⚡ O(n×k×i) | O(n²) | O(n³) | Medium |
| **Outlier Handling** | ❌ Assigns all points | ✅ Labels as noise | ❌ Assigns all | ✅ Low probability |
| **Scalability** | ✅ Millions | Medium | ❌ Thousands | Medium |
| **Interpretability** | ✅ Clear centers | Medium | ✅ Tree structure | Probabilities |

**Why KMeans for job embeddings:**
1. **SBERT creates spherical-ish clusters** - Embeddings are normalized, work well with cosine/euclidean
2. **Scalable** - Works on 100K+ jobs easily
3. **Interpretable** - Each cluster has a "centroid" (average job in that cluster)
4. **Simple to explain** - "Jobs closest to this center belong to this cluster"

#### Silhouette Score Explained

```
For each data point i:
  a(i) = average distance to OTHER points in SAME cluster (cohesion)
  b(i) = average distance to points in NEAREST OTHER cluster (separation)
  
  silhouette(i) = (b(i) - a(i)) / max(a(i), b(i))
  
  Overall silhouette = average across all points
```

**Interpretation:**

| Score | Meaning |
|-------|---------|
| +1.0 | Perfect! Point is far from other clusters, close to own cluster |
| +0.5 to +1.0 | Strong clustering structure |
| +0.3 to +0.5 | Reasonable clustering ← Our target |
| 0.0 | Point is on boundary between clusters |
| -1.0 | Wrong cluster! Point is closer to another cluster |

**Learning Result:** "Silhouette score measures how similar a point is to its own cluster compared to other clusters. A score of 0.4 means points are reasonably well-clustered but there's some overlap - which is expected for job postings since a 'Data Engineer' might legitimately belong to both 'Tech' and 'Data Science' clusters."

## 3C.1: Salary Prediction (Regression)

### Model Comparison

| Model | Pros | Cons | Priority |
|-------|------|------|----------|
| **LightGBM** ✅ | Fast, handles categorical, good default | Requires tuning | P0 |
| **XGBoost** | Robust, well-documented | Slower than LightGBM | P1 |
| **Random Forest** | Interpretable, no tuning needed | Memory heavy | P2 |
| **Linear Regression** | Baseline, interpretable | Poor with non-linear | Baseline |

### Task 3C.1.1: Baseline Models
- [ ] Create `ml/salary_predictor.py`:
  ```python
  class SalaryPredictor:
      def __init__(self, model_type: str = "lightgbm")
      def train(self, X: pd.DataFrame, y: pd.Series) -> None
      def predict(self, X: pd.DataFrame) -> np.ndarray
      def evaluate(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]
      def save(self, path: Path) -> None
      def load(self, path: Path) -> None
  ```
- [ ] Implement baseline: Linear Regression
- [ ] Implement LightGBM with default params
- [ ] Implement XGBoost for comparison

### Task 3C.1.2: Hyperparameter Tuning
- [ ] Use Optuna or RandomizedSearchCV
- [ ] Key hyperparameters for LightGBM:
  - `num_leaves`: [31, 50, 100]
  - `learning_rate`: [0.01, 0.05, 0.1]
  - `n_estimators`: [100, 500, 1000]
  - `min_child_samples`: [20, 50, 100]
- [ ] Log all experiments to `ml/experiments/`

### Task 3C.1.3: Evaluation Metrics
- [ ] Implement evaluation suite:
  ```python
  def evaluate_regression(y_true, y_pred) -> Dict[str, float]:
      return {
          "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
          "mae": mean_absolute_error(y_true, y_pred),
          "r2": r2_score(y_true, y_pred),
          "mape": mean_absolute_percentage_error(y_true, y_pred),
      }
  ```
- [ ] Create residual plots
- [ ] Analyze errors by salary range and job type

**Target Metrics:**
- RMSE: < $1,500 SGD
- MAE: < $1,000 SGD
- R²: > 0.7

## 3C.2: Job Role Classification (Multi-class)

### Task 3C.2.1: Classification Pipeline
- [ ] Create `ml/role_classifier.py`:
  ```python
  class RoleClassifier:
      def __init__(self, model_type: str = "lightgbm")
      def train(self, X: pd.DataFrame, y: pd.Series) -> None
      def predict(self, X: pd.DataFrame) -> np.ndarray
      def predict_proba(self, X: pd.DataFrame) -> np.ndarray
      def evaluate(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]
  ```
- [ ] Handle class imbalance (SMOTE or class weights)
- [ ] Implement top-k accuracy for multi-class

### Task 3C.2.2: Evaluation Metrics
- [ ] Implement classification metrics:
  ```python
  def evaluate_classification(y_true, y_pred) -> Dict[str, float]:
      return {
          "accuracy": accuracy_score(y_true, y_pred),
          "f1_macro": f1_score(y_true, y_pred, average='macro'),
          "f1_weighted": f1_score(y_true, y_pred, average='weighted'),
      }
  ```
- [ ] Create confusion matrix visualization
- [ ] Per-class precision/recall analysis

**Target Metrics:**
- F1 Macro: > 0.6
- Top-3 Accuracy: > 0.85

## 3C.3: Job Clustering (Unsupervised)

### Task 3C.3.1: Clustering Pipeline
- [ ] Create `ml/clustering.py`:
  ```python
  class JobClusterer:
      def __init__(self, n_clusters: int = 10)
      def fit(self, embeddings: np.ndarray) -> None
      def predict(self, embeddings: np.ndarray) -> np.ndarray
      def get_cluster_centers(self) -> np.ndarray
      def get_cluster_labels(self) -> Dict[int, str]  # Human-readable names
  ```
- [ ] Implement KMeans with elbow method
- [ ] Try DBSCAN for density-based clustering
- [ ] Implement cluster labeling (top keywords per cluster)

### Task 3C.3.2: Dimensionality Reduction
- [ ] Implement PCA for feature reduction
- [ ] Implement UMAP for visualization
- [ ] Create 2D/3D scatter plots with cluster colors

### Task 3C.3.3: Cluster Analysis
- [ ] Generate cluster summaries:
  - Cluster size distribution
  - Average salary per cluster
  - Top job titles per cluster
  - Top companies per cluster
- [ ] Name clusters (e.g., "Tech/Software", "Finance/Banking", "Healthcare")

**Target Metrics:**
- Silhouette Score: > 0.3
- Cluster sizes: Relatively balanced (no cluster < 5% of data)

---

# Phase 3D: Model Artifacts & Deployment

**Goal:** Save, version, and deploy trained models.

## 3D.0: Conceptual Understanding

### What Do We Deploy? Where?

**Deployment Flow:**
```
LOCAL DEVELOPMENT                     PRODUCTION
─────────────────                     ──────────────────────────
Train model locally          →        Save to GCS (versioned)
Evaluate metrics                      gs://bucket/models/salary_predictor/v1/
                                              ↓
                                      Load in Cloud Function/FastAPI
                                              ↓
                                      Serve predictions via API
                                              ↓
                                      Write results to BigQuery
```

**Deployment Options:**

| Option | Use Case | Cost | Latency | Our Choice |
|--------|----------|------|---------|------------|
| **Cloud Function** | Batch predictions | Free tier | 1-5s cold start | ✅ Daily batch |
| **FastAPI on Cloud Run** | Real-time API | ~$5/month | 100ms warm | ✅ API layer |
| **Vertex AI Endpoint** | Production serving | $50+/month | 50ms | ❌ Too expensive for now |

**Our Strategy:**
- **Batch predictions:** Cloud Function runs daily, processes new jobs, writes to BigQuery
- **Real-time API:** FastAPI loads model from GCS, serves predictions on-demand
- **Vertex AI:** Use for monitoring dashboards, not hosting (cost reasons)

### What Do We Schedule Daily?

**NOT model training!** Training is occasional (weekly/monthly).

**Daily Schedule:**

| Time | Task | What Happens |
|------|------|--------------|
| 6:00 AM | Scraping | Cloud Run scrapes new jobs → GCS |
| 6:30 AM | ETL | Cloud Function processes → BigQuery |
| 7:00 AM | **Embedding Generation** | Generate embeddings for NEW jobs only |
| 7:30 AM | **Batch Predictions** | Predict salary/cluster for NEW jobs only |

**Model Retraining Schedule:**

| Strategy | When to Retrain | Pros | Cons |
|----------|-----------------|------|------|
| **Calendar-based** | Every Sunday | Simple, predictable | May retrain unnecessarily |
| **Performance-based** ✅ | When accuracy drops | Efficient | Needs monitoring |
| **Continuous** | Every new batch | Always fresh | Expensive, risky |

**Our Approach (Performance-based):**
```python
# Pseudo-code for monitoring
current_accuracy = evaluate_on_holdout_set()
if current_accuracy < threshold:
    trigger_retraining()
    alert_team()
```

### What Model Artifacts Do We Save?

```
/models/salary_predictor/v1/
├── model.joblib          # Serialized LightGBM model
├── config.json           # Hyperparameters used
├── metrics.json          # Evaluation results (RMSE, R², etc.)
├── feature_names.json    # Column names in expected order
└── training_metadata.json # Date, data version, etc.
```

**Why version models?**
- Rollback if new model performs worse
- A/B testing between versions
- Audit trail for compliance

## 3D.1: Model Serialization

### Task 3D.1.1: Model Registry Structure
- [ ] Create directory structure:
  ```
  /models/
    salary_predictor/
      v1/
        model.joblib
        config.json
        metrics.json
        feature_names.json
    role_classifier/
      v1/
        model.joblib
        config.json
        metrics.json
        label_encoder.joblib
    clustering/
      v1/
        model.joblib
        config.json
        metrics.json
        cluster_labels.json
  ```
- [ ] Implement `ml/registry.py`:
  ```python
  def save_model(model, name: str, version: str, metrics: Dict) -> Path
  def load_model(name: str, version: str = "latest") -> Any
  def list_models() -> List[Dict]
  def get_model_metrics(name: str, version: str) -> Dict
  ```

### Task 3D.1.2: GCS Upload
- [ ] Upload models to GCS: `gs://sg-job-market-data/models/`
- [ ] Implement model download for inference
- [ ] Version management (keep last 5 versions)

## 3D.2: Batch Predictions

### Task 3D.2.1: Prediction Pipeline
- [ ] Create `ml/predict.py`:
  ```python
  def predict_salary_batch(job_ids: List[str]) -> Dict[str, float]
  def predict_cluster_batch(job_ids: List[str]) -> Dict[str, int]
  def predict_all_new_jobs() -> Dict[str, Any]
  ```
- [ ] Write predictions to BigQuery `ml_predictions` table
- [ ] Schedule daily predictions (Cloud Scheduler → Cloud Function)

### Task 3D.2.2: BigQuery Predictions Table
- [ ] Create schema:
  ```python
  @dataclass
  class MLPrediction:
      job_id: str
      source: str
      predicted_salary: float
      salary_confidence: float
      predicted_cluster: int
      cluster_name: str
      predicted_at: datetime
  ```

---

# .py vs .ipynb: When to Use Each?

## Quick Answer

| Aspect | .py Scripts ✅ (Our Production Code) | .ipynb Notebooks |
|--------|-------------------------------------|------------------|
| **Use Case** | Production pipelines, APIs, deployment | Exploration, prototyping, analysis |
| **Version Control** | ✅ Clean diffs, easy to review | ❌ JSON format, messy diffs |
| **Testing** | ✅ Easy with pytest | ❌ Hard to test |
| **CI/CD** | ✅ Runs in pipelines | ❌ Requires papermill/nbconvert |
| **Code Quality** | ✅ Linters, formatters work well | ❌ Limited tool support |
| **Collaboration** | ✅ Multiple devs can work on same file | ❌ Merge conflicts common |
| **Debugging** | ✅ Standard debuggers | ❌ Cell execution order issues |
| **Documentation** | Docstrings + separate docs | ✅ Inline markdown + plots |
| **Reproducibility** | ✅ Deterministic execution | ❌ Hidden state from cell order |

## Why Our Project Uses .py for Production

**Our Architecture:**
```
notebooks/              ← .ipynb for EDA & experimentation
├── eda_salary.ipynb
├── cluster_viz.ipynb
└── prototype_embeddings.ipynb

nlp/                    ← .py for production code
├── embeddings.py       ✅ Deployed to Cloud Functions
├── generate_embeddings.py  ✅ CLI for batch jobs

ml/                     ← .py for training pipelines
├── features.py         ✅ Reusable in API
├── salary_predictor.py ✅ Can import in FastAPI
└── train.py            ✅ Runs in scheduled jobs
```

**Reasons:**

1. **Cloud Deployment** - Cloud Functions require .py modules, not notebooks
2. **Import-ability** - `from ml.salary_predictor import SalaryPredictor` only works with .py
3. **Testing** - `pytest tests/test_embeddings.py` needs .py files
4. **Version Control** - `.py` diffs are readable, `.ipynb` diffs are JSON noise
5. **Code Reuse** - Same `EmbeddingGenerator` class used by CLI, API, and Cloud Function

## When We DO Use Notebooks

**Exploratory Data Analysis (EDA):**
```python
# notebooks/eda_salary.ipynb
import pandas as pd
import matplotlib.pyplot as plt

# Quick analysis
df = pd.read_gbq("SELECT * FROM cleaned_jobs LIMIT 1000")
df.describe()
df['salary_mid'].hist()
plt.show()

# Once satisfied, move to .py:
# ml/features.py → extract_numerical_features()
```

**Prototype Testing:**
```python
# notebooks/prototype_embeddings.ipynb
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")
texts = ["Data Scientist", "Software Engineer"]
embeddings = model.encode(texts)

# Visualize
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
reduced = pca.fit_transform(embeddings)
plt.scatter(reduced[:, 0], reduced[:, 1])

# Once working, refactor to .py:
# nlp/embeddings.py → EmbeddingGenerator class
```

**Visualization & Reporting:**
```python
# notebooks/cluster_viz.ipynb
# Generate interactive Plotly charts for stakeholders
# Export as HTML for dashboards
```

## Best Practice Workflow

```
EXPLORATION (Notebooks)           PRODUCTION (Scripts)
──────────────────────            ────────────────────
1. notebooks/prototype.ipynb  →   2. Refactor to ml/features.py
   - Try different approaches        - Clean API
   - Visualize results               - Docstrings
   - Test hyperparameters            - Error handling

3. notebooks/eda.ipynb        →   4. Production pipeline
   - Analyze data patterns           - ml/train.py
   - Identify issues                 - Scheduled jobs
   - Share insights                  - Logging & monitoring
```

## Interview Answer

> **"Why .py instead of .ipynb for production ML?"**
> 
> "Notebooks are great for exploration—I use them for EDA, prototyping embeddings, and visualizing clusters. But for production pipelines, I use .py modules because:
> 1. **Cloud Functions require .py** - Can't deploy notebooks directly
> 2. **Version control** - .py diffs are readable, notebooks are JSON
> 3. **Testing** - pytest works seamlessly with .py, notebooks need papermill
> 4. **Reproducibility** - .py scripts execute top-to-bottom deterministically, notebooks have hidden state from cell order
> 5. **Code reuse** - Same `EmbeddingGenerator` class is imported by CLI, API, and Cloud Function
> 
> My workflow: prototype in notebooks, refactor to .py, write tests, then deploy. Best of both worlds."

---

# Phase 4: GenAI & RAG (Agentic Retrieval-Augmented Generation)

**Goal:** Build intelligent job market assistant using LangChain, LangGraph, and Gemini Pro.

**Status:** 🔲 **PENDING** (After Phase 3C completes)

**Dependencies:**
- ✅ BigQuery with cleaned_jobs and embeddings (from Phase 3A)
- ✅ Vector search capability (from Phase 3A)
- 🔲 Trained ML models (Phase 3C)
- 🔲 FastAPI serving predictions (Phase 3D)

---

## 4.0: RAG Fundamentals

### What is RAG (Retrieval-Augmented Generation)?

**Traditional LLM Problem:**
```
User: "What skills are most in-demand for Data Scientists in Singapore?"

GPT-4: "Based on my training data (up to 2023), common skills include Python,
       machine learning, SQL..."  ← GENERIC, OUTDATED
```

**RAG Solution:**
```
User: "What skills are most in-demand for Data Scientists in Singapore?"

System Flow:
1. Embed query → [0.21, -0.45, 0.67, ...]
2. Vector search → Find 10 most similar jobs from our BigQuery
3. Extract skills from job descriptions → ["Python", "PyTorch", "BigQuery", "MLOps"]
4. Feed to LLM with context:
   "Based on these 10 recent Data Scientist jobs in Singapore:
    Job 1: Requires Python, PyTorch, BigQuery...
    Job 2: Requires Python, TensorFlow, Docker...
    ...
    
    What skills are most in-demand?"

Gemini Pro: "Based on the current Singapore job market data, the most in-demand
            skills are:
            1. Python (100% of jobs)
            2. PyTorch/TensorFlow (80%)
            3. BigQuery/Cloud (70%)..."  ← ACCURATE, UP-TO-DATE
```

**RAG = Retrieval (Vector Search) + Augmentation (Add Context) + Generation (LLM)**

### Why RAG for Job Market Intelligence?

| Without RAG | With RAG ✅ |
|-------------|------------|
| LLM answers from training data (outdated) | LLM answers from YOUR fresh data |
| Generic advice | Singapore-specific insights |
| Can't answer "what jobs are available" | Can list actual job postings |
| Hallucinations | Grounded in real data |

**Our RAG Use Cases:**
1. **Job Recommendations:** "Find me Data Scientist jobs paying >$7K/month in CBD"
2. **Salary Insights:** "What's the typical salary for Backend Engineers in Singapore?"
3. **Skill Trends:** "What new skills are Finance companies looking for?"
4. **Career Advice:** "How do I transition from Data Analyst to Data Scientist?"

---

## 4.1: RAG Architecture (LangChain + LangGraph)

### Why LangChain?

**LangChain = Framework for building LLM applications**

**What it provides:**
- **Chains:** Link LLM calls with retrieval steps
- **Memory:** Maintain conversation context
- **Tools:** Let LLM call external functions (e.g., BigQuery queries)
- **Vector Store Integration:** Connect to BigQuery Vector Search
- **Prompt Templates:** Reusable prompt structures

### Why LangGraph?

**LangGraph = State machine for agentic workflows (built on LangChain)**

**What it solves:**
```
Simple Chain (LangChain):
User Query → Retrieve → LLM → Response
             ↓
          (Always retrieves, even if not needed)

Agentic Flow (LangGraph):
User Query → Agent → Decide: Do I need to search?
                      ├── YES → Retrieve → LLM → Response
                      └── NO  → LLM → Response

Example:
"What is machine learning?" → NO retrieval needed (general knowledge)
"Data Scientist jobs in Jurong?" → YES retrieval needed (specific data)
```

**LangGraph Features:**
- **State Management:** Track conversation context across turns
- **Conditional Routing:** Agent decides next step dynamically
- **Tool Calling:** Agent can invoke multiple tools (search, predict salary, etc.)
- **Cycles:** Agent can retry or ask follow-up questions

### Our RAG Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ USER INPUT                                                                  │
│ "Find me Data Scientist jobs in Singapore paying over $8K/month"            │
└─────────────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│ LANGGRAPH AGENT (State Machine)                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│ State: { query: "...", retrieved_jobs: [], predictions: [], chat_history }  │
│                                                                             │
│ Node 1: QUERY CLASSIFIER                                                    │
│   → Is this a search query, analysis question, or chat?                     │
│   → Decide: route to [SEARCH] or [ANALYSIS] or [CHAT]                       │
│                                                                             │
│ Node 2: SEARCH AGENT (if search needed)                                     │
│   → Extract search criteria (role, location, salary)                        │
│   → Call Tool: BigQuery Vector Search                                       │
│   → Call Tool: Salary Predictor (if missing)                                │
│   → Store retrieved_jobs in state                                           │
│                                                                             │
│ Node 3: ANALYSIS AGENT (if analysis needed)                                 │
│   → Call Tool: Aggregate SQL queries (avg salary, top skills)               │
│   → Store analysis_results in state                                         │
│                                                                             │
│ Node 4: RESPONSE GENERATOR                                                  │
│   → Build prompt with retrieved context                                     │
│   → Call Gemini Pro with grounded data                                      │
│   → Return formatted response to user                                       │
└─────────────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│ TOOLS (Agent can invoke these)                                              │
├─────────────────────────────────────────────────────────────────────────────┤
│ Tool 1: vector_search(query: str, filters: dict) → List[Job]                │
│   → Embeds query with SBERT                                                 │
│   → Searches BigQuery job_embeddings                                        │
│   → Returns top-k similar jobs                                              │
│                                                                             │
│ Tool 2: predict_salary(job_id: str) → float                                 │
│   → Loads LightGBM model from GCS                                           │
│   → Returns predicted salary if missing                                     │
│                                                                             │
│ Tool 3: get_skill_trends(role: str, months: int) → List[str]                │
│   → Queries BigQuery for recent jobs                                        │
│   → Extracts skills with NER/regex                                          │
│   → Returns top-10 skills by frequency                                      │
│                                                                             │
│ Tool 4: aggregate_salary_stats(role: str, location: str) → Dict             │
│   → SQL: AVG(salary), PERCENTILE(salary, 0.5), COUNT(*)                     │
│   → Returns salary statistics                                               │
└─────────────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│ GEMINI PRO (LLM)                                                            │
├─────────────────────────────────────────────────────────────────────────────┤
│ Prompt:                                                                     │
│ """                                                                         │
│ You are a Singapore job market expert.                                      │
│                                                                             │
│ User Query: "Data Scientist jobs in Singapore paying over $8K/month"        │
│                                                                             │
│ Retrieved Jobs:                                                             │
│ 1. Senior Data Scientist at Google - $10,000/month - Singapore CBD          │
│ 2. ML Engineer at Grab - $9,500/month - Singapore West                      │
│ 3. Data Scientist at DBS - $8,500/month - Marina Bay                        │
│ ...                                                                         │
│                                                                             │
│ Based on this data, provide a helpful response to the user.                 │
│ """                                                                         │
│                                                                             │
│ Response: "I found 3 Data Scientist positions paying over $8K/month:        │
│            1. Google is hiring a Senior Data Scientist for $10K/month..."   │
└─────────────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│ USER INTERFACE                                                              │
│ • FastAPI: /chat endpoint (REST API)                                        │
│ • MCP Server: Expose as tools to Claude/Cursor                              │
│ • Streamlit: Chat interface with job cards                                  │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 4.2: LangChain vs LangGraph Deep Dive

### LangChain: Linear Chains

**Simple RAG with LangChain:**
```python
from langchain.chains import RetrievalQA
from langchain_google_vertexai import VertexAI
from langchain.vectorstores import BigQueryVectorSearch

# Setup
llm = VertexAI(model_name="gemini-pro")
vectorstore = BigQueryVectorSearch(
    project_id="sg-job-market",
    dataset_id="sg_job_market",
    table_name="job_embeddings"
)

# Create chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever(search_kwargs={"k": 5}),
    return_source_documents=True
)

# Use
response = qa_chain("Data Scientist jobs in Singapore?")
print(response["result"])
```

**Limitations:**
- ❌ Always retrieves, even for general questions
- ❌ Can't combine multiple tools (search + salary prediction)
- ❌ No conditional logic
- ❌ No state management across turns

### LangGraph: Agentic Workflows

**Agentic RAG with LangGraph:**
```python
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage

# Define state
class RAGState(TypedDict):
    messages: List[BaseMessage]
    query: str
    needs_search: bool
    retrieved_jobs: List[dict]
    predictions: List[float]
    final_response: str

# Define nodes
def classify_query(state: RAGState) -> RAGState:
    """Decide if we need to search or just chat."""
    query = state["query"]
    # Use LLM to classify
    classification = llm.invoke(
        f"Is this a job search query or general question? '{query}'"
    )
    state["needs_search"] = "search" in classification.lower()
    return state

def search_jobs(state: RAGState) -> RAGState:
    """Retrieve relevant jobs from BigQuery."""
    query = state["query"]
    embeddings = embed_query(query)
    jobs = bigquery_vector_search(embeddings, top_k=10)
    state["retrieved_jobs"] = jobs
    return state

def predict_missing_salaries(state: RAGState) -> RAGState:
    """Predict salaries for jobs missing salary data."""
    jobs = state["retrieved_jobs"]
    for job in jobs:
        if not job.get("salary"):
            job["salary_predicted"] = salary_model.predict(job)
    return state

def generate_response(state: RAGState) -> RAGState:
    """Generate final response with context."""
    jobs = state["retrieved_jobs"]
    prompt = build_prompt(state["query"], jobs)
    response = llm.invoke(prompt)
    state["final_response"] = response.content
    return state

# Build graph
workflow = StateGraph(RAGState)

# Add nodes
workflow.add_node("classify", classify_query)
workflow.add_node("search", search_jobs)
workflow.add_node("predict", predict_missing_salaries)
workflow.add_node("respond", generate_response)

# Add edges (conditional routing)
workflow.set_entry_point("classify")
workflow.add_conditional_edges(
    "classify",
    lambda state: "search" if state["needs_search"] else "respond",
    {
        "search": "search",
        "respond": "respond"
    }
)
workflow.add_edge("search", "predict")
workflow.add_edge("predict", "respond")
workflow.add_edge("respond", END)

# Compile
app = workflow.compile()

# Use
result = app.invoke({"query": "Data Scientist jobs in Singapore?"})
print(result["final_response"])
```

**Advantages:**
- ✅ Conditional logic (only search if needed)
- ✅ Multi-tool orchestration (search → predict → respond)
- ✅ State management (track across conversation turns)
- ✅ Debuggable (visualize state at each step)

---

## 4.3: Implementation Tasks

### Task 4.3.1: BigQuery Vector Search Integration
- [ ] Create `genai/retriever.py`:
  ```python
  class BigQueryRetriever:
      def __init__(self, project_id: str, dataset_id: str)
      def search(self, query: str, top_k: int = 10, filters: dict = None) -> List[Dict]
      def search_by_embedding(self, embedding: List[float], top_k: int = 10) -> List[Dict]
  ```
- [ ] Support filters (salary_min, location, work_type)
- [ ] Return job data + similarity scores

### Task 4.3.2: LangChain Tools
- [ ] Create `genai/tools.py`:
  ```python
  @tool
  def search_jobs(query: str, filters: dict) -> List[Dict]:
      """Search for jobs matching query."""
      
  @tool
  def predict_salary(job_id: str) -> float:
      """Predict salary for job without salary data."""
      
  @tool
  def get_salary_stats(role: str, location: str) -> Dict:
      """Get salary statistics for role/location."""
      
  @tool
  def extract_skills(job_ids: List[str]) -> List[str]:
      """Extract top skills from job descriptions."""
  ```

### Task 4.3.3: LangGraph Agent
- [ ] Create `genai/agent.py`:
  ```python
  class JobMarketAgent:
      def __init__(self, llm, tools: List[Tool])
      def build_graph(self) -> StateGraph
      def invoke(self, query: str) -> str
      def stream(self, query: str) -> Iterator[str]
  ```
- [ ] Implement nodes: classify, search, analyze, respond
- [ ] Add conversation memory (last 5 turns)

### Task 4.3.4: Prompt Engineering
- [ ] Create `genai/prompts.py`:
  ```python
  SYSTEM_PROMPT = """
  You are a Singapore job market expert assistant.
  
  Your knowledge is grounded in real-time data from JobStreet and MyCareersFuture.
  
  Guidelines:
  - Use retrieved job data to answer questions
  - Cite specific jobs when possible
  - If no relevant jobs found, say so
  - Provide salary ranges in SGD
  - Mention job locations
  """
  
  SEARCH_PROMPT = """
  Based on these {num_jobs} jobs:
  {job_list}
  
  User Question: {query}
  
  Provide a helpful response.
  """
  ```

### Task 4.3.5: FastAPI Integration
- [ ] Create `/chat` endpoint:
  ```python
  @app.post("/chat")
  async def chat(request: ChatRequest) -> ChatResponse:
      response = agent.invoke(request.message)
      return ChatResponse(message=response)
  ```
- [ ] Add streaming endpoint for real-time responses
- [ ] Add conversation history management

### Task 4.3.6: MCP Server
- [ ] Create `genai/mcp_server.py`:
  ```python
  # Expose job search as MCP tool for Claude/Cursor
  @mcp.tool()
  def search_singapore_jobs(query: str) -> List[Dict]:
      """Search Singapore job market."""
      
  @mcp.tool()
  def get_job_salary_insights(role: str) -> Dict:
      """Get salary insights for role."""
  ```
- [ ] Allow external AI assistants to query our data

---

## 4.4: Evaluation & Testing

### RAG Evaluation Metrics

| Metric | What it Measures | Target |
|--------|------------------|--------|
| **Retrieval Precision** | % of retrieved jobs that are relevant | >0.8 |
| **Retrieval Recall** | % of relevant jobs that were retrieved | >0.7 |
| **Answer Relevance** | Does LLM answer match user intent? | >0.8 (human eval) |
| **Faithfulness** | Does answer only use retrieved context? | >0.9 |
| **Latency** | Time from query to response | <3 seconds |

### Testing Strategy

**Unit Tests:**
- [ ] Test retriever with known queries
- [ ] Test tools individually
- [ ] Test prompt templates

**Integration Tests:**
- [ ] Test full agent flow
- [ ] Test with conversation history
- [ ] Test tool chaining

**Human Evaluation:**
- [ ] Create test set of 50 questions
- [ ] Rate answers: Relevant, Accurate, Helpful
- [ ] Compare to baseline (no RAG)

---

# Testing Strategy

## Unit Tests
- [ ] `tests/test_embeddings.py` - Embedding generation
- [ ] `tests/test_features.py` - Feature engineering
- [ ] `tests/test_salary_predictor.py` - Regression model
- [ ] `tests/test_clustering.py` - Clustering model
- [ ] `tests/test_retriever.py` - BigQuery vector search
- [ ] `tests/test_rag_tools.py` - LangChain tools
- [ ] `tests/test_agent.py` - LangGraph agent

## Integration Tests
- [ ] `tests/test_ml_pipeline.py` - End-to-end ML workflow
- [ ] `tests/test_bq_ml_integration.py` - BigQuery read/write
- [ ] `tests/test_rag_pipeline.py` - End-to-end RAG workflow

## Model Validation
- [ ] Cross-validation with 5 folds
- [ ] Time-based validation (train on past, test on recent)
- [ ] A/B testing framework (for future online evaluation)
- [ ] RAG answer quality evaluation (human eval + automated metrics)

---

# Dependencies to Add

```txt
# Add to requirements.txt

# Phase 3: ML/NLP
sentence-transformers==2.2.2
scikit-learn==1.3.2
lightgbm==4.2.0
xgboost==2.0.3
optuna==3.5.0
umap-learn==0.5.5
matplotlib==3.8.2
seaborn==0.13.0
plotly==5.18.0
joblib==1.3.2

# Phase 4: GenAI/RAG
langchain==0.1.0
langgraph==0.0.20
google-cloud-aiplatform==1.38.0
langchain-google-vertexai==0.1.0
```

---

# Code Location Summary

| Module | Purpose | Key Files |
|--------|---------|-----------|
| `/nlp/` | Embeddings | `embeddings.py`, `generate_embeddings.py` |
| `/ml/` | Training | `features.py`, `salary_predictor.py`, `role_classifier.py`, `clustering.py`, `registry.py`, `predict.py` |
| `/models/` | Artifacts | `{model_name}/{version}/model.joblib` |
| **Phase 3: ML/NLP** | | | |
| 1 | Install ML dependencies | 5 min | None |
| 2 | Generate embeddings (3A) | 30 min | cleaned_jobs data |
| 3 | Create vector index | 10 min | Embeddings |
| 4 | Feature engineering (3B) | 1 hr | Embeddings |
| 5 | Train salary predictor (3C.1) | 1 hr | Features |
| 6 | Train clustering (3C.3) | 30 min | Embeddings |
| 7 | Train role classifier (3C.2) | 30 min | Features |
| 8 | Model registry & GCS (3D) | 30 min | All models |
| 9 | Batch predictions | 30 min | Registry |
| 10 | Documentation & tests | 1 hr | All |
| **Phase 4: GenAI/RAG** | | | |
| 11 | Install GenAI dependencies | 5 min | Phase 3 complete |
| 12 | BigQuery retriever (4.3.1) | 1 hr | Vector index |
| 13 | LangChain tools (4.3.2) | 1 hr | Models deployed |
| 14 | LangGraph agent (4.3.3) | 2 hr | Tools ready |
| 15 | Prompt engineering (4.3.4) | 1 hr | Agent ready |
| 16 | FastAPI /chat endpoint (4.3.5) | 1 hr | Agent ready |
| 17 | MCP Server (4.3.6) | 1 hr | FastAPI ready |
| 18 | RAG testing & evaluation (4.4) | 2 hr | All |

**Total Estimated Time:** 
- Phase 3: 6-8 hours
- Phase 4: 9-11 hours
- **Overall: 15-19 hours**hr | Embeddings |
| 5 | Train salary predictor (3C.1) | 1 hr | Features |
| 6 | Train clustering (3C.3) | 30 min | Embeddings |
| 7 | Train role classifier (3C.2) | 30 min | Features |
| 8 | Model registry & GCS (3D) | 30 min | All models |
| 9 | Batch predictions | 30 min | Registry |
| 10 | Documentation & tests | 1 hr | All |

**Total Estimated Time:** 6-8 hours

---

# Success Criteria

**Phase 3 Complete When:**
- [ ] All cleaned_jobs have embeddings in BigQuery
- [ ] Vector similarity search returns relevant results
- [ ] Salary prediction RMSE < $1,500 SGD
- [ ] Clustering produces 8-12 meaningful clusters
- [ ] Models saved to GCS with versioning
- [ ] Predictions written to BigQuery
- [ ] All tests passing

---

# 🎯 Final Deliverables Summary

## What You'll Have Built

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ SINGAPORE JOB MARKET INTELLIGENCE PLATFORM                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│ 1. SALARY PREDICTOR                                                         │
│    Model: LightGBM regression                                               │
│    Input: Job title + description + location + work type + embeddings       │
│    Output: Predicted monthly salary (SGD)                                   │
│    Accuracy: RMSE < $1,500, R² > 0.7                                        │
│    Use Case: "How much should this job pay?"                                │
│                                                                             │
│ 2. JOB SIMILARITY SEARCH                                                    │
│    Model: Sentence-BERT (all-MiniLM-L6-v2) + BigQuery Vector Search         │
│    Input: Job description or search query                                   │
│    Output: Top-10 semantically similar jobs                                 │
│    Use Case: "Find jobs similar to this one"                                │
│                                                                             │
│ 3. JOB CLUSTERING                                                           │
│    Model: KMeans on 384-dim SBERT embeddings                                │
│    Input: All job embeddings                                                │
│    Output: 8-12 clusters with human-readable labels                         │
│    Metrics: Silhouette Score > 0.3                                          │
│    Use Case: "What job categories exist in Singapore market?"               │
│                                                                             │
│ 4. ROLE CLASSIFIER                                                          │
│    Model: LightGBM multi-class classification                               │
│    Input: Job title + description embeddings                                │
│    Output: Job category (IT, Finance, Healthcare, etc.)                     │
│    Accuracy: F1 Macro > 0.6                                                 │
│    Use Case: "What category does this job belong to?"                       │
│                                                                             │
│ 5. RAG CHATBOT (Phase 4 - GenAI)                                            │
│    Model: Gemini Pro + LangChain + SBERT embeddings                         │
│    Input: Natural language questions                                        │
│    Output: Answers grounded in real Singapore job data                      │
│    Use Case: "What skills are most in demand for Data Scientists in SG?"    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Technical Skills Demonstrated

| Category | Skills | Evidence |
|----------|--------|----------|
| **NLP** | Embeddings, Transformers, Semantic Search | SBERT implementation, vector indexing |
| **ML** | Regression, Classification, Clustering | LightGBM, KMeans, evaluation metrics |
| **Data Engineering** | ETL, BigQuery, Streaming | Cloud Functions, append-only design |
| **MLOps** | Model versioning, Batch inference | GCS registry, scheduled predictions |
| **Cloud** | GCP services | Cloud Run, Cloud Functions, BigQuery |
| **Software Engineering** | Clean code, Testing | Modular design, unit/integration tests |