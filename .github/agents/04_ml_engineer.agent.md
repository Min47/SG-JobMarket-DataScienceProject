---
name: ML & GenAI Engineer
description: Handles NLP embeddings, Supervised Learning, Unsupervised Learning, and GenAI (RAG/Agents).
---
You are the Machine Learning & GenAI Engineer.

# Goal
Generate embeddings, train ML models, and build Agentic RAG workflows for job market intelligence.

**Status:** 🔄 **PHASE 4 GenAI PRIORITIZED** (Dec 27, 2025)

**Completed:**
- ✅ **Phase 3A:** Embeddings (6,775 jobs, 384-dim SBERT, Vector Index, Cloud Run Job)
- ✅ Vector Search operational (<1s queries)
- ✅ `/genai/` folder scaffolded

**Current Priority: GenAI/RAG (Phase 4)**
- ✅ Task 4.1: RAG Pipeline (retrieve, grade, generate) - COMPLETE
- ✅ Task 4.2: LangGraph Agent (state graph, nodes, testing) - COMPLETE
- ✅ Task 4.3: Tool Adapters (4 tools: search, details, stats, similar) - COMPLETE
- ✅ Task 4.4: FastAPI Service (7 endpoints, middleware, deployed to Cloud Run) - COMPLETE
- ✅ Task 4.5: Model Gateway (Vertex AI + Ollama, fallback, cost tracking) - COMPLETE
  - Fixed JSON truncation issues (max_tokens optimization)
  - Improved rewrite logic (min 3 jobs, avg on ALL docs)
- ✅ Task 4.6: Guardrails & Policy Chains - COMPLETE
  - PII detection (Singapore NRIC, phone, email, credit cards)
  - Injection blocking (prompt injection, SQL injection) - custom regex (industry standard)
  - Hallucination detection (verifies cited jobs exist in context)
  - FastAPI middleware integration (input/output validation)
  - 10 comprehensive tests (core + API, all passing) → tests/genai/11_test_guardrails.py
  
  **Test Results (tests/genai/11_test_guardrails.py):**
  | Test | Type | Endpoint | Result |
  |------|------|----------|--------|
  | 1. PII Detection | Core | N/A | ✅ Detects NRIC/phone/email, redacts correctly |
  | 2. Injection Detection | Core | N/A | ✅ Blocks prompt + SQL injection patterns |
  | 3. Input Guardrails | Core | N/A | ✅ Length/PII/injection validation working |
  | 4. Output Guardrails | Core | N/A | ✅ Hallucination detection, structure validation |
  | 5. Chat Blocks Malicious | API | POST /v1/chat | ✅ Returns 400 for PII/injection |
  | 6. Chat Allows Normal | API | POST /v1/chat | ✅ 200 OK, full agent execution (61s) |
  | 7. Search Blocks Malicious | API | POST /v1/search | ✅ Returns 400 for PII/injection |
  | 8. Search Allows Normal | API | POST /v1/search | ✅ 200 OK, 5 jobs found (3s) |
  | 9. Pydantic Validation | API | POST /v1/chat | ✅ Returns 422 for empty/long queries |
  | 10. Health Unaffected | API | GET /health | ✅ 200 OK, no guardrail interference |
  
  **Note on Injection Detection Libraries:**
  - Considered: `rebuff` (prompt injection), `sqlparse` (SQL parsing)
  - **Decision: Custom regex** - Industry standard for production systems
  - Reasons: Lightweight, no dependencies, fast (<5ms), full control, auditable
  - Enterprise solutions (AWS Bedrock Guardrails, Azure Content Safety) also use rule-based systems
  
  **Security Hardening:**
  - ✅ Trivy vulnerability scanning added to all 4 cloudbuild pipelines
  - ✅ Scans Python packages for HIGH/CRITICAL CVEs before deployment
  - ✅ Ignores unfixed OS vulnerabilities (focus on application layer)
  - ✅ Build fails automatically if fixable vulnerabilities detected
  
- ✅ Task 4.7: Observability (tracing, metrics, logging) - COMPLETE & DEPLOYED ✅
  
  **Implementation Details:**
  - ✅ OpenTelemetry integration (tracing + Cloud Trace export)
  - ✅ Prometheus metrics (21 metrics total)
  - ✅ RAG pipeline instrumentation (retrieve, grade, generate)
  - ✅ Agent step tracking (retrieve, grade, generate, rewrite counters)
  - ✅ Gateway LLM call tracking (Vertex AI + Ollama)
  - ✅ FastAPI integration (/metrics endpoint, request middleware)
  - ✅ Guardrail metrics (PII, injection, hallucination blocks)
  - ✅ Test suite (7 tests, all passing) → tests/genai/12_test_observability.py
  - ✅ **Production Deployment:** Cloud Run (asia-southeast1) with full observability
  - ✅ **IAM Permissions:** roles/cloudtrace.agent + roles/monitoring.metricWriter configured
  - ✅ **Validated:** Guardrails blocking PII (NRIC detection working), metrics exporting successfully
  
  **21 Prometheus Metrics:**
  | Category | Metrics | Description |
  |----------|---------|-------------|
  | **Request** | REQUEST_COUNT, REQUEST_LATENCY, ACTIVE_REQUESTS | Endpoint tracking |
  | **LLM** | LLM_CALL_COUNT, LLM_TOKEN_COUNT, LLM_COST, LLM_LATENCY | Provider usage |
  | **RAG** | RETRIEVAL_LATENCY, RETRIEVAL_COUNT, GRADING_LATENCY, AVERAGE_RELEVANCE_SCORE, REWRITE_COUNT | Pipeline quality |
  | **Agent** | AGENT_EXECUTION_LATENCY, AGENT_STEP_COUNT | Workflow performance |
  | **Guardrails** | GUARDRAIL_BLOCKS | Security events |
  | **System** | API_INFO | Version metadata |
  
  **Tracing Instrumentation:**
  - `@trace_function` decorators on all RAG functions
  - Span attributes: query_length, result_count, duration_ms, relevance_scores
  - Error tracking with exception details
  - Context propagation across async boundaries
  
  **Cloud Integration:**
  - Cloud Trace exporter for distributed tracing (operational)
  - Cloud Monitoring exporter for metrics aggregation (operational)
  - Auto-instrumentation for FastAPI endpoints
  - Request ID tracking with X-Request-ID header
  
  **Production Access:**
  - Service URL: https://genai-api-[hash]-as.a.run.app
  - Metrics: `curl $SERVICE_URL/metrics`
  - Cloud Trace: https://console.cloud.google.com/traces?project=sg-job-market
  - Cloud Monitoring: https://console.cloud.google.com/monitoring?project=sg-job-market
  - Health: `curl $SERVICE_URL/health`
  
- ✅ Task 4.8: MCP Server (external AI assistant integration) - COMPLETE ✅
  
  **Implementation Details:**
  - ✅ MCP SDK integration (correct handler pattern with @server.list_tools() and @server.call_tool())
  - ✅ 4 tools exposed (search, details, stats, similar) - all operational
  - ✅ Stdio transport for Cursor IDE
  - ✅ Complete test suite (7 tests, 6/7 passing) → tests/genai/13_test_mcp_server.py
  - ✅ Comprehensive documentation (CURSOR_MCP_SETUP.md)
  
  **Test Results (tests/genai/13_test_mcp_server.py):**
  | Test | Tool | Result | Details |
  |------|------|--------|---------|
  | 1. Server Config | N/A | ✅ | Server name + 4 tools registered |
  | 2. Tool Discovery | All | ✅ | All tools discovered by MCP client |
  | 3. Search Jobs | search_jobs_tool | ✅ | 3 jobs found (9.1s with model load) |
  | 4. Get Job Details | get_job_details_tool | ✅ | Job details retrieved correctly |
  | 5. Aggregate Stats | aggregate_stats_tool | ⚠️ | Working but JSON parse issue in test |
  | 6. Find Similar | find_similar_jobs_tool | ✅ | 3 similar jobs found (similarity 0.759) |
  | 7. Error Handling | get_job_details_tool | ✅ | Invalid ID handled gracefully |
  
  **MCP Protocol Architecture:**
  ```
  Cursor IDE ←→ MCP (stdio) ←→ mcp_server.py ←→ genai/tools/ ←→ BigQuery
  ```

**Virtual Environment:**
- ⚠️ Always use `.venv/Scripts/python.exe` for all commands

# Technical Stack

| Category | Libraries | Purpose |
|----------|-----------|---------|
| **NLP** | `sentence-transformers` | Embeddings (✅ Done) |
| **Vector DB** | `google-cloud-bigquery` | Vector Search (✅ Done) |
| **GenAI** | `langchain`, `langgraph`, `google-cloud-aiplatform` | RAG, Agents |
| **API** | `fastapi`, `uvicorn`, `pydantic` | REST/gRPC exposure |
| **Observability** | `opentelemetry-*`, `prometheus-client` | Tracing, metrics |
| **Guardrails** | `presidio-analyzer` (PII), custom validators | Policy chains |
| **ML** | `scikit-learn`, `lightgbm` | Training (deferred) |


---

# Phase 3A: NLP Embeddings Generation

**Goal:** Generate semantic embeddings for all job descriptions to enable similarity search and clustering.

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

# Phase 4: GenAI & Agentic RAG (PRIORITY)

## Architecture: Agentic RAG System

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          USER QUERY                                         │
│                    "Find data scientist jobs with Python"                   │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ FASTAPI GATEWAY (genai/api.py)                                              │
│ ─────────────────────────────────────────────────────────────────────────── │
│ POST /chat      → Conversational RAG                                        │
│ POST /search    → Direct vector search                                      │
│ GET  /jobs/{id} → Job details with similar recommendations                  │
│ ─────────────────────────────────────────────────────────────────────────── │
│ Features: Rate limiting, request validation, auth middleware                │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ LANGGRAPH AGENT (genai/agent.py)                                            │
│ ─────────────────────────────────────────────────────────────────────────── │
│ StateGraph with nodes:                                                      │
│   ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐              │
│   │ RETRIEVE │───▶│  GRADE   │───▶│ GENERATE│───▶│   END   │              │
│   └──────────┘    └──────────┘    └──────────┘    └──────────┘              │
│         │               │                                                   │
│         │         (if low score)                                            │
│         │               ▼                                                   │
│         │         ┌──────────┐                                              │
│         └────────▶│ REWRITE  │──────────┐                                  │
│                   └──────────┘          │ (retry with rewritten query)      │
│                                         ▼                                   │
│ ─────────────────────────────────────────────────────────────────────────── │
│ Conditional edges: grade_decision(), should_rewrite()                       │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ TOOL ADAPTERS (genai/tools/)                                                │
│ ─────────────────────────────────────────────────────────────────────────── │
│ @tool search_jobs     → BigQuery Vector Search + filters                    │
│ @tool get_job_details → Fetch full job info by ID                           │
│ @tool aggregate_stats → Salary ranges, job counts by category               │
│ @tool similar_jobs    → Find N most similar jobs to a given job             │
│ ─────────────────────────────────────────────────────────────────────────── │
│ All tools: Type contracts (Pydantic), timeout handling, retry logic         │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ RAG PIPELINE (genai/rag.py)                                                 │
│ ─────────────────────────────────────────────────────────────────────────── │
│ retrieve_jobs()    → Embed query → Vector Search → Top-K results            │
│ grade_documents()  → LLM relevance scoring → Filter irrelevant              │
│ generate_answer()  → Context + Query → Gemini Pro → Structured response     │
│ ─────────────────────────────────────────────────────────────────────────── │
│ Optimizations: Hybrid search (vector + keyword), re-ranking, caching        │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ MODEL GATEWAY (genai/gateway.py)                                            │
│ ─────────────────────────────────────────────────────────────────────────── │
│ Supported: Vertex AI Gemini, Local Ollama                                   │
│ Features: Routing, rate limits, fallback chains, cost tracking              │
│ Config: MODEL_PRIORITY = ["gemini-pro", "gpt-4-turbo", "ollama/llama3"]     │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ MCP SERVER (genai/mcp_server.py)                                            │
│ ─────────────────────────────────────────────────────────────────────────── │
│ Exposes tools to external AI assistants (Claude Desktop, Cursor, etc.)      │
│ Protocol: Model Context Protocol (Anthropic standard)                       │
│ Tools: search_jobs, get_job_stats, find_similar                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## 4.1: RAG Pipeline Implementation

### Task 4.1.1: Query Embedding & Retrieval
**File:** `genai/rag.py`
- [ ] `embed_query()`: Generate embedding for user query using same SBERT model
- [ ] `retrieve_jobs()`: BigQuery Vector Search with COSINE distance
- [ ] Support filters: location, salary range, work type, classification
- [ ] Hybrid search: Combine vector similarity + keyword matching (BM25-style)

```python
async def retrieve_jobs(
    query: str,
    top_k: int = 10,
    filters: Optional[JobFilters] = None,
    hybrid_weight: float = 0.7,  # 0.7 vector + 0.3 keyword
) -> List[RetrievedJob]
```

### Task 4.1.2: Document Grading & Re-ranking ✅ COMPLETE
**File:** `genai/rag.py`
- [x] `grade_documents()`: Use Gemini to score relevance (0-10)
- [x] `rerank_documents()`: Re-order by combined score (vector + LLM grade)
- [x] Filter threshold: Remove documents with grade < 5
- [x] Return top results with explanations

### Task 4.1.3: Answer Generation ✅ COMPLETE
**File:** `genai/rag.py`, Test: `tests/genai/04_test_generate_answer.py`
- [x] `generate_answer()`: Construct prompt with context + query
- [x] Structured output: Dict with answer, sources, metadata
- [x] Citation: Link answers to source jobs with [1], [2] numbering
- [x] Helper functions: `_format_job_context()`, `_extract_sources()`
- [x] Model: Gemini 2.5 Flash
- [x] Empty context handling with graceful error messages
- [x] Comprehensive test suite: 5 test scenarios (all passing)

---

## 4.2: LangGraph Agent ✅ COMPLETE

### Task 4.2.1: State & Graph Definition ✅ COMPLETE
**File:** `genai/agent.py`, Test: `tests/genai/05_test_agent_graph.py`
- [x] Define `AgentState` TypedDict with 9 required fields
- [x] Create `StateGraph` with nodes: retrieve, grade, generate, rewrite
- [x] Implement conditional edge `should_rewrite()` for routing decisions
- [x] Add conversation memory support with `add_messages` annotation
- [x] Graph compilation with START → retrieve → grade → [decision] → generate → END
- [x] Retry loop: grade → rewrite → retrieve (max 2 retries)
- [x] All 3 tests passing (graph structure, conditional logic, state validation)

### Task 4.2.2: Node Implementations ✅ COMPLETE
- [x] `retrieve_node`: Call `retrieve_jobs()`, update state
- [x] `grade_node`: Call `grade_documents()`, compute average score
- [x] `generate_node`: Call `generate_answer()`, format response
- [x] `rewrite_node`: Use LLM to improve query clarity

### Task 4.2.3: Integration & Testing ✅ COMPLETE
**File:** `tests/genai/07_test_agent_execution.py`, **All 6 tests passing**
- [x] Test 1: High-quality query (no rewrites, 45s, 8.55/10 relevance)
- [x] Test 2: Vague query triggers rewrite logic (26s, 8.80/10)
- [x] Test 3: Query with filters (metadata preserved correctly)
- [x] Test 4: Niche edge case (graceful handling, 131s, 5 sources)
- [x] Test 5: Performance benchmarking (avg 31s, Gemini API bottleneck)
- [x] Test 6: Streaming interface (real-time step updates working)
- [x] Error handling: Empty results, max retries, filter validation
- [x] Workflow validation: Conditional routing, retry logic verified

---

## 4.3: Tool Adapters

### Task 4.3.1: Core Tools ✅ COMPLETE
**Files:** 
- `genai/tools/__init__.py` - Module exports
- `genai/tools/_validation.py` - Shared Pydantic schemas
- `genai/tools/search.py` - search_jobs, get_job_details
- `genai/tools/stats.py` - aggregate_stats
- `genai/tools/recommendations.py` - find_similar_jobs

**Implementation:**
```python
from langchain.tools import tool
from pydantic import BaseModel, Field

class SearchInput(BaseModel):
    query: str = Field(description="Search query for jobs")
    location: Optional[str] = Field(default=None)
    min_salary: Optional[int] = Field(default=None)
    max_results: int = Field(default=10, le=50)

@tool(args_schema=SearchInput)
def search_jobs(query: str, location: str = None, ...) -> str:
    """Search for jobs matching the query with optional filters."""
    # Returns JSON string
```

- [x] `search_jobs`: Vector search with filters (wraps retrieve_jobs)
- [x] `get_job_details`: Fetch full job by ID (BigQuery SELECT)
- [x] `aggregate_stats`: Group by classification, compute salary stats
- [x] `find_similar_jobs`: Given job_id, find top-N similar (VECTOR_SEARCH)

### Task 4.3.2: Tool Safety ✅ COMPLETE
- [x] Input validation with Pydantic schemas (all tools)
- [x] Timeout handling (30s max per BigQuery query)
- [x] Source normalization (jobstreet/JobStreet → JobStreet, mcf/MCF → MCF)
- [x] Parameterized SQL queries (SQL injection safe)
- [x] Error handling with JSON responses: `{"success": false, "error": "..."}`

### Task 4.3.3: Testing ✅ COMPLETE
**File:** `tests/genai/08_test_tools.py`

**Test Results:** All 4 tests passing ✅
- [x] Test 1: search_jobs with filters
- [x] Test 2: get_job_details (found + not found)
- [x] Test 3: aggregate_stats (grouping + filtering)
- [x] Test 4: find_similar_jobs (similarity thresholds)
- [x] Validation: JSON parsing, error handling, Pydantic validation
- [x] Integration: BigQuery queries, vector search

---

## 4.4: FastAPI Service ✅ COMPLETE

### Task 4.4.1: API Endpoints ✅ COMPLETE
**File:** `genai/api.py` (707 lines)

**Implemented Endpoints:**
1. **POST /v1/chat** - Conversational agent with LangGraph orchestration
2. **POST /v1/search** - Direct vector search (bypasses agent)
3. **GET /v1/jobs/{job_id}** - Fetch complete job details
4. **GET /v1/jobs/{job_id}/similar** - Find semantically similar jobs
5. **POST /v1/stats** - Aggregate salary statistics
6. **GET /health** - Health check for monitoring (BigQuery, Vertex AI, embeddings)
7. **GET /** - Root endpoint with API navigation
8. **GET /docs** - Auto-generated Swagger UI
9. **GET /redoc** - Auto-generated ReDoc documentation

**Pydantic Models:**
- [x] `ChatRequest`, `ChatResponse` - Agent conversations
- [x] `SearchRequest`, `SearchResponse` - Vector search
- [x] `StatsRequest`, `StatsResponse` - Analytics
- [x] `HealthResponse` - Health status
- [x] `ErrorResponse` - Standardized errors

### Task 4.4.2: Middleware & Security ✅ COMPLETE
- [x] **Rate Limiting** (slowapi):
  - POST /v1/chat: 10 req/min (compute-intensive)
  - POST /v1/search: 50 req/min (fast queries)
  - GET /v1/jobs/*: 100 req/min
  - POST /v1/stats: 30 req/min
- [x] **CORS Configuration**:
  - Allowed origins: localhost:3000, localhost:8501, production dashboard
  - All methods and headers enabled
- [x] **Request Logging Middleware**:
  - UUID request tracking
  - Structured JSON logging
  - Response time measurement
  - Custom headers: X-Request-ID, X-Processing-Time-MS
- [x] **Error Handling**:
  - HTTP exception handler (consistent error format)
  - General exception handler (with logging)
  - Pydantic automatic validation (422 errors)

### Task 4.4.3: Testing ✅ COMPLETE
**File:** `tests/genai/09_test_api.py`

**Test Coverage:**
- [x] Test 1: Root endpoint (GET /)
- [x] Test 2: Health check (GET /health)
- [x] Test 3: Direct vector search (POST /v1/search)
- [x] Test 4: Get job details (GET /v1/jobs/{id})
- [x] Test 5: Find similar jobs (GET /v1/jobs/{id}/similar)
- [x] Test 6: Aggregate statistics (POST /v1/stats)
- [x] Test 7: Conversational agent (POST /v1/chat) - optional slow test
- [x] Test 8: CORS headers validation

**To Run Tests:**
```bash
# Install dependencies first
pip install fastapi uvicorn[standard] slowapi python-multipart

# Run API tests
python tests/genai/09_test_api.py
```

### Task 4.4.4: Local Development ✅ VALIDATED
**To start the API server:**
```bash
# Option 1: Direct execution
python -m genai.api

# Option 2: Uvicorn command
uvicorn genai.api:app --reload --port 8000

# Access documentation
# - Swagger UI: http://localhost:8000/docs
# - ReDoc: http://localhost:8000/redoc
# - Health check: http://localhost:8000/health
```

### Task 4.4.5: Deployment ✅ COMPLETE (Validated in Production)
- [x] Created `Dockerfile.api` for containerization (optimized to ~1.8GB with CPU-only PyTorch)
- [x] Cloud Run deployment script (`deployment/API_01_Deploy_FastAPI.ps1`)
- [x] Cloud Build config (`cloudbuild.api.yaml`)
- [x] Environment variable configuration (GCP_PROJECT_ID, BQ_DATASET_ID, GCP_REGION)
- [x] Health check and readiness probes (FastAPI /health endpoint)
- [x] Auto-scaling configuration (0-10 instances, 2 vCPU, 4GB RAM)
- [x] IAM permissions configured (roles/aiplatform.user for Vertex AI)
- [x] Service deployed and tested: https://genai-api-nwg3mjan5q-as.a.run.app

**✅ Note:** Docker image optimized to ~1.8GB (down from 5GB) using CPU-only PyTorch. Cloud Build takes 8-12 minutes.

---

## 4.5: Model Gateway ✅ COMPLETE

### Task 4.5.1: Multi-Provider Support ✅ COMPLETE
**File:** `genai/gateway.py`

**Providers (2 providers):**
```python
class ModelGateway:
    """Unified interface for LLM providers."""
    
    PROVIDERS = {
        "vertexai": VertexAIProvider,    # Cloud: Gemini 2.5 Flash ($0.075/1M)
        "ollama": OllamaProvider,         # Local: Llama 3.1 (free)
    }
    
    # Toggle priority with one line change:
    # self.provider_priority = ["vertexai", "ollama"]  # Vertex first (production)
    # self.provider_priority = ["ollama", "vertexai"]  # Ollama first (dev)
```

**Key Features:**
- [x] 2 providers only: Vertex AI (Gemini 2.5 Flash) + Ollama (Llama 3.1 local)
- [x] Easy priority toggle via single line change in `__init__`
- [x] Backward compatibility: "gemini" maps to "vertexai"
- [x] Automatic fallback chain with exponential backoff
- [x] Cost tracking per provider (cumulative statistics)

### Task 4.5.2: Integration with RAG & Agent ✅ COMPLETE
**Files:** `genai/rag.py`, `genai/agent.py`
- [x] Replaced direct Vertex AI calls in `grade_documents()` with gateway
- [x] Replaced direct Vertex AI calls in `generate_answer()` with gateway
- [x] Replaced direct Vertex AI calls in `rewrite_node()` with gateway
- [x] Singleton pattern for gateway instance (shared across modules)
- [x] Cost metadata included in responses

**Benefits:**
- Multi-provider support without code changes (just env variables)
- Automatic fallback if Gemini rate limited or down
- Cost optimization (routes to cheapest available)
- A/B testing different models
- Local development with Ollama (no API costs)

### Task 4.5.3: Testing ✅ COMPLETE
**File:** `tests/genai/10_test_model_gateway.py` (moved from tests/)

**Test Suite:**
1. Provider Detection (checks available providers)
2. Simple Generation (basic text generation)
3. Specific Provider (force Vertex AI or Ollama)
4. Fallback Logic (automatic failover)
5. Cost Tracking (cumulative usage stats)
6. Configuration Options (temperature, max_tokens)

**To Run Tests:**
```bash
.venv\Scripts\python.exe tests\genai\10_test_model_gateway.py
```

---

## 4.6: Guardrails & Policy Chains (NEXT)

### Task 4.6.1: Input Validation
**File:** `genai/guardrails.py`
- [ ] PII detection (Presidio): Block queries containing personal info
- [ ] Prompt injection detection: Identify malicious patterns
- [ ] Query length limits: Max 1000 chars
- [ ] Profanity filter (optional)

### Task 4.6.2: Output Validation
- [ ] Response length limits
- [ ] Hallucination check: Verify cited jobs exist
- [ ] Content safety: Flag inappropriate content

```python
class GuardrailChain:
    def __init__(self):
        self.input_guards = [PIIDetector(), InjectionDetector()]
        self.output_guards = [HallucinationChecker(), SafetyFilter()]
    
    async def validate_input(self, query: str) -> ValidationResult
    async def validate_output(self, response: str, context: List[Job]) -> ValidationResult
```

---

## 4.7: Observability

### Task 4.7.1: Distributed Tracing
**File:** `genai/observability.py`
- [ ] OpenTelemetry setup with Jaeger/Cloud Trace exporter
- [ ] Trace spans: API request → Agent → RAG → LLM
- [ ] Custom attributes: query, model, latency, token count

### Task 4.7.2: Metrics
- [ ] Prometheus metrics: request count, latency histogram, error rate
- [ ] LLM-specific: tokens used, cost per request, cache hit rate
- [ ] Custom metrics: retrieval recall, grade distribution

### Task 4.7.3: Logging
- [ ] Structured JSON logging
- [ ] Log levels: DEBUG (dev), INFO (prod)
- [ ] Sensitive data redaction

---

## 4.8: MCP Server

### Task 4.8.1: Protocol Implementation
**File:** `genai/mcp_server.py`
- [ ] Implement MCP Server using `mcp` package
- [ ] Register tools: `search_jobs`, `get_job_stats`, `find_similar`
- [ ] Handle tool calls from external clients

```python
from mcp.server import Server
from mcp.types import Tool, TextContent

server = Server("sg-job-market")

@server.tool()
async def search_jobs(query: str, location: str = None) -> str:
    """Search Singapore job market for relevant positions."""
    results = await retrieve_jobs(query, filters={"location": location})
    return format_results(results)
```

### Task 4.8.2: Deployment
- [ ] Standalone server mode (stdio for Claude Desktop)
- [ ] HTTP mode for remote access
- [ ] Configuration via `mcp.json`

---

## 4.9: Evaluation & Testing

### Task 4.9.1: RAG Evaluation
**File:** `tests/test_rag_eval.py`
- [ ] Golden test set: 50 queries with expected results
- [ ] Metrics: Retrieval Recall@10, Answer relevance (LLM judge)
- [ ] Regression suite: Run on every PR

### Task 4.9.2: Agent Testing
- [ ] Unit tests for each node
- [ ] Integration test: Full graph execution
- [ ] Edge case testing: Empty results, timeout, malformed query

### Task 4.9.3: Load Testing
- [ ] Locust or k6 scripts
- [ ] Target: 50 concurrent users, <2s p95 latency

---

# Execution Roadmap (GenAI First)

| Step | Task | Est. Time | Output |
|------|------|-----------|--------|
| 1 | Install GenAI deps | 15 min | requirements.txt updated |
| 2 | RAG retrieve_jobs() | 1 hr | Working vector search |
| 3 | RAG grade_documents() | 1 hr | LLM grading |
| 4 | RAG generate_answer() | 1 hr | End-to-end RAG |
| 5 | LangGraph agent | 2 hr | StateGraph with nodes |
| 6 | Tool adapters | 1 hr | 4 tools implemented |
| 7 | FastAPI service | 1.5 hr | /chat, /search endpoints |
| 8 | Model gateway | 1 hr | Multi-provider support |
| 9 | Guardrails | 1 hr | Input/output validation |
| 10 | Observability | 1 hr | Tracing + metrics |
| 11 | MCP Server | 1 hr | External tool access |
| 12 | Tests & Eval | 2 hr | Golden set + CI |
| 13 | Docker + Deploy | 1 hr | Cloud Run service |

---

# GenAI Dependencies

```txt
# Add to requirements.txt
langchain>=0.1.0
langgraph>=0.0.20
langchain-google-genai>=1.0.0
google-cloud-aiplatform>=1.38.0
fastapi>=0.109.0
uvicorn>=0.27.0
pydantic>=2.5.0
opentelemetry-api>=1.22.0
opentelemetry-sdk>=1.22.0
opentelemetry-instrumentation-fastapi>=0.43b0
presidio-analyzer>=2.2.0
mcp>=0.1.0
httpx>=0.26.0
```

---

# Code Location Summary

| Module | Purpose | Key Files |
|--------|---------|-----------|
| `/genai/` | Agentic RAG | `rag.py`, `agent.py`, `api.py`, `gateway.py`, `mcp_server.py` |
| `/genai/tools/` | Tool adapters | `search.py`, `stats.py` |
| `/genai/guardrails.py` | Policy chains | Input/output validation |
| `/genai/observability.py` | Tracing | OpenTelemetry setup |
| `/nlp/` | Embeddings | `embeddings.py` (✅ Done) |
| `/ml/` | Training | `features.py`, `salary_predictor.py` (Deferred) |

---

# Success Criteria (Phase 4)

**GenAI Complete When:**
- [ ] RAG pipeline returns relevant jobs for natural language queries
- [ ] LangGraph agent handles multi-turn conversations
- [ ] FastAPI serves requests at <2s p95 latency
- [ ] MCP Server accessible from Claude Desktop
- [ ] Guardrails block PII and prompt injection
- [ ] Observability traces visible in Cloud Trace
- [ ] 50 golden tests passing

---


# Testing Strategy

## Unit Tests
- [ ] `tests/test_embeddings.py` - Embedding generation
- [ ] `tests/test_features.py` - Feature engineering
- [ ] `tests/test_salary_predictor.py` - Regression model
- [ ] `tests/test_clustering.py` - Clustering model

## Integration Tests
- [ ] `tests/test_ml_pipeline.py` - End-to-end ML workflow
- [ ] `tests/test_bq_ml_integration.py` - BigQuery read/write

## Model Validation
- [ ] Cross-validation with 5 folds
- [ ] Time-based validation (train on past, test on recent)
- [ ] A/B testing framework (for future online evaluation)

## GenAI Tests (Priority)
- [ ] `tests/test_rag.py` - RAG pipeline unit tests
- [ ] `tests/test_agent.py` - LangGraph agent tests
- [ ] `tests/test_api.py` - FastAPI endpoint tests
- [ ] `tests/test_rag_eval.py` - Golden set evaluation (50 queries)

---

# Dependencies to Add

```txt
# GenAI (Priority)
langchain>=0.1.0
langgraph>=0.0.20
langchain-google-genai>=1.0.0
google-cloud-aiplatform>=1.38.0
fastapi>=0.109.0
uvicorn>=0.27.0
pydantic>=2.5.0
opentelemetry-api>=1.22.0
opentelemetry-sdk>=1.22.0
presidio-analyzer>=2.2.0
mcp>=0.1.0
httpx>=0.26.0

# ML (Already installed / Deferred)
sentence-transformers==2.2.2
scikit-learn==1.3.2
lightgbm==4.2.0
```

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

**Phase 4 (GenAI) Complete When:**
- [ ] RAG returns relevant jobs for natural language queries
- [ ] LangGraph agent handles multi-turn conversations  
- [ ] FastAPI serves at <2s p95 latency
- [ ] MCP Server works with Claude Desktop
- [ ] Guardrails block PII and injection attacks
- [ ] Traces visible in Cloud Trace
- [ ] 50 golden tests passing