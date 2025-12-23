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

**What's Next:** Phase 1A - NLP Embeddings Generation

**Virtual Environment Usage:**
- ⚠️ **CRITICAL:** Always use `.venv/Scripts/python.exe` for all Python commands
- Install dependencies: `.venv/Scripts/python.exe -m pip install <package>`
- Run training: `.venv/Scripts/python.exe -m ml.train`
- Update `requirements.txt` when adding new packages

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
│ Sentence-BERT: all-MiniLM-L6-v2 (384 dimensions)                           │
│         ↓ (batch embedding generation)                                      │
│ BigQuery: job_embeddings table (job_id, embedding ARRAY<FLOAT64>)          │
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
│ ─────────────────────────────────── │ ─────────────────────────────────────│
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

---

# Dependencies to Add

```txt
# Add to requirements.txt
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
```

---

# Code Location Summary

| Module | Purpose | Key Files |
|--------|---------|-----------|
| `/nlp/` | Embeddings | `embeddings.py`, `generate_embeddings.py` |
| `/ml/` | Training | `features.py`, `salary_predictor.py`, `role_classifier.py`, `clustering.py`, `registry.py`, `predict.py` |
| `/models/` | Artifacts | `{model_name}/{version}/model.joblib` |
| `/tests/` | Testing | `test_embeddings.py`, `test_ml_pipeline.py` |

---

# Execution Order

| Step | Task | Est. Time | Dependencies |
|------|------|-----------|--------------|
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