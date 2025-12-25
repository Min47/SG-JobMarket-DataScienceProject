# Phase 3B: Feature Engineering - Detailed Breakdown

**Goal:** Transform cleaned_jobs + embeddings → ML-ready dataset for training models

**Status:** 🟡 **IN PROGRESS**
- ✅ `ml/features.py` skeleton exists (FeatureEngineer class)
- 🔲 BigQuery view `vw_ml_features` not created yet
- 🔲 Feature validation script not implemented
- 🔲 Feature statistics/EDA not done

---

## 📋 Current Code Status

### Implemented ✅
**File:** `ml/features.py`
- `FeatureConfig` dataclass with feature lists
- `FeatureEngineer` class with 3 methods:
  - `extract_numerical_features()`: Salary, text length, temporal
  - `extract_categorical_features()`: One-hot encoding
  - `prepare_training_data()`: Combines all features + embeddings
- `create_train_test_split()`: Time-based split function

### Not Implemented 🔲
1. **BigQuery View:** `vw_ml_features` (SQL aggregation of features)
2. **Data Loading:** Function to query BigQuery → pandas
3. **PCA Reduction:** Embedding dimensionality reduction
4. **Feature Scaling:** StandardScaler for numerical features
5. **Missing Value Handling:** Salary imputation strategy
6. **Feature Validation:** Check for NaN, inf, data types
7. **Feature Statistics:** Distribution analysis, correlation matrix

---

## 🎯 Phase 3B Roadmap (5 Tasks)

### Task 3B.1: Create BigQuery Feature View ⏱️ 30 min

**Why a VIEW instead of TABLE?**
- Computed features (salary_mid, days_since_posted) are cheap to calculate
- Always up-to-date (no sync issues)
- Storage savings
- Only embeddings need TABLE storage (expensive to compute)

**Implementation:**

**File:** `ml/setup_features_view.py`

```python
"""
Create vw_ml_features view in BigQuery.

Combines cleaned_jobs + job_embeddings with query-time deduplication
and derived features.

Usage:
    python -m ml.setup_features_view
"""

from google.cloud import bigquery
import os
from dotenv import load_dotenv

load_dotenv()

PROJECT_ID = os.getenv('GCP_PROJECT_ID', 'sg-job-market')
DATASET_ID = os.getenv('BQ_DATASET_ID', 'sg_job_market')

VIEW_SQL = f"""
CREATE OR REPLACE VIEW `{PROJECT_ID}.{DATASET_ID}.vw_ml_features` AS
WITH latest_jobs AS (
  SELECT 
    job_id,
    source,
    job_title,
    job_description,
    job_classification,
    job_location,
    job_work_type,
    job_salary_min_sgd_monthly,
    job_salary_max_sgd_monthly,
    job_posted_timestamp,
    company_name,
    company_industry,
    company_size,
    scrape_timestamp,
    ROW_NUMBER() OVER (
      PARTITION BY source, job_id 
      ORDER BY scrape_timestamp DESC
    ) AS rn
  FROM `{PROJECT_ID}.{DATASET_ID}.cleaned_jobs`
)
SELECT 
  c.job_id,
  c.source,
  
  -- Text features (for model input)
  c.job_title,
  c.job_description,
  c.job_classification,
  
  -- Categorical features
  c.job_location,
  c.job_work_type,
  c.company_industry,
  c.company_size,
  
  -- Numerical features (derived)
  c.job_salary_min_sgd_monthly,
  c.job_salary_max_sgd_monthly,
  (c.job_salary_min_sgd_monthly + c.job_salary_max_sgd_monthly) / 2 AS salary_mid_monthly,
  c.job_salary_max_sgd_monthly - c.job_salary_min_sgd_monthly AS salary_range,
  LENGTH(c.job_description) AS description_length,
  LENGTH(c.job_title) AS title_length,
  TIMESTAMP_DIFF(CURRENT_TIMESTAMP(), c.job_posted_timestamp, DAY) AS days_since_posted,
  
  -- Embeddings (384 dimensions)
  e.embedding,
  e.model_name,
  
  -- Metadata
  c.job_posted_timestamp,
  c.scrape_timestamp
  
FROM latest_jobs c
LEFT JOIN `{PROJECT_ID}.{DATASET_ID}.job_embeddings` e 
  ON c.job_id = e.job_id AND c.source = e.source
WHERE c.rn = 1
"""

def create_view():
    """Create or replace vw_ml_features view."""
    client = bigquery.Client(project=PROJECT_ID)
    
    print("Creating vw_ml_features view...")
    print(f"Project: {PROJECT_ID}")
    print(f"Dataset: {DATASET_ID}")
    
    query_job = client.query(VIEW_SQL)
    query_job.result()  # Wait for completion
    
    print("\n✅ View created successfully!")
    print(f"View: {PROJECT_ID}.{DATASET_ID}.vw_ml_features")
    
    # Verify
    verify_query = f"""
    SELECT COUNT(*) as total_rows
    FROM `{PROJECT_ID}.{DATASET_ID}.vw_ml_features`
    """
    result = client.query(verify_query).result()
    row_count = list(result)[0]['total_rows']
    
    print(f"✅ View contains {row_count:,} rows")
    
    # Show sample
    sample_query = f"""
    SELECT 
      job_id,
      job_title,
      salary_mid_monthly,
      description_length,
      days_since_posted
    FROM `{PROJECT_ID}.{DATASET_ID}.vw_ml_features`
    WHERE salary_mid_monthly IS NOT NULL
    LIMIT 5
    """
    print("\n📋 Sample rows:")
    for row in client.query(sample_query).result():
        print(f"  {row.job_title[:50]:50} | ${row.salary_mid_monthly:,.0f} | {row.description_length:,} chars | {row.days_since_posted} days")

if __name__ == "__main__":
    create_view()
```

**Run:**
```bash
.venv/Scripts/python.exe -m ml.setup_features_view
```

**Expected Output:**
```
Creating vw_ml_features view...
✅ View created successfully!
✅ View contains 6,775 rows
📋 Sample rows:
  Senior Data Analyst                               | $7,500 | 2,341 chars | 5 days
  Backend Engineer                                  | $6,000 | 1,823 chars | 2 days
```

---

### Task 3B.2: Data Loading Function ⏱️ 20 min

**File:** Add to `ml/features.py`

```python
# Add at top
from google.cloud import bigquery
from utils.config import Settings

# Add as method to FeatureEngineer class
def load_from_bigquery(
    self,
    project_id: str,
    dataset_id: str,
    limit: Optional[int] = None,
    filters: Optional[Dict[str, Any]] = None,
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Load features from BigQuery vw_ml_features.
    
    Args:
        project_id: GCP project ID
        dataset_id: BigQuery dataset ID
        limit: Optional row limit for testing
        filters: Optional WHERE clause filters (e.g., {"salary_mid_monthly": "> 3000"})
    
    Returns:
        Tuple of (DataFrame with job data, embeddings matrix)
    """
    client = bigquery.Client(project=project_id)
    
    # Build query
    query = f"""
    SELECT *
    FROM `{project_id}.{dataset_id}.vw_ml_features`
    WHERE embedding IS NOT NULL  -- Only jobs with embeddings
    """
    
    # Add filters
    if filters:
        for col, condition in filters.items():
            query += f" AND {col} {condition}"
    
    # Add limit
    if limit:
        query += f" LIMIT {limit}"
    
    logger.info(f"Loading features from BigQuery (limit={limit})...")
    df = client.query(query).to_dataframe()
    
    # Extract embeddings as numpy array
    embeddings = np.vstack(df['embedding'].values)
    
    # Drop embedding column from df (will be added back as separate features)
    df = df.drop(columns=['embedding', 'model_name'])
    
    logger.info(f"✅ Loaded {len(df):,} rows, embeddings shape: {embeddings.shape}")
    return df, embeddings
```

**Usage:**
```python
from ml.features import FeatureEngineer

fe = FeatureEngineer()
df, embeddings = fe.load_from_bigquery(
    project_id="sg-job-market",
    dataset_id="sg_job_market",
    limit=1000,  # Test with 1K rows first
    filters={"salary_mid_monthly": "IS NOT NULL"}
)

X, y = fe.prepare_training_data(df, embeddings, target="salary_mid_monthly")
```

---

### Task 3B.3: PCA Dimensionality Reduction ⏱️ 30 min

**Why PCA?**
- 384 embedding dimensions can cause overfitting with small datasets
- PCA preserves 90-95% of variance in 10-50 dimensions
- Faster training and inference

**File:** Add to `ml/features.py`

```python
# Add at top
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Add to FeatureEngineer class
def fit_pca(
    self,
    embeddings: np.ndarray,
    n_components: int = 50,
    explained_variance_threshold: float = 0.95,
) -> None:
    """
    Fit PCA on embeddings for dimensionality reduction.
    
    Args:
        embeddings: Embedding matrix (n_samples x 384)
        n_components: Number of PCA components (or None for threshold)
        explained_variance_threshold: If n_components is None, use this threshold
    """
    if n_components is None:
        # Find components that explain threshold% of variance
        pca = PCA(n_components=explained_variance_threshold)
    else:
        pca = PCA(n_components=n_components)
    
    logger.info(f"Fitting PCA with n_components={n_components}...")
    pca.fit(embeddings)
    
    self._pca = pca
    
    explained = pca.explained_variance_ratio_.sum()
    logger.info(f"✅ PCA fitted: {pca.n_components_} components explain {explained:.1%} variance")

def transform_embeddings(
    self,
    embeddings: np.ndarray,
) -> np.ndarray:
    """
    Transform embeddings using fitted PCA.
    
    Args:
        embeddings: Embedding matrix (n_samples x 384)
    
    Returns:
        Reduced embeddings (n_samples x n_components)
    """
    if self._pca is None:
        logger.warning("PCA not fitted yet, returning original embeddings")
        return embeddings
    
    reduced = self._pca.transform(embeddings)
    logger.info(f"Transformed embeddings: {embeddings.shape} → {reduced.shape}")
    return reduced
```

**Update `prepare_training_data()` method:**

```python
# Inside prepare_training_data(), replace PCA TODO with:
if embeddings is not None and self.config.include_embeddings:
    # Optional PCA reduction
    if self.config.embedding_pca_components:
        if self._pca is None:
            # Fit PCA on training data
            self.fit_pca(embeddings, n_components=self.config.embedding_pca_components)
        embeddings = self.transform_embeddings(embeddings)
    
    emb_cols = [f"emb_{i}" for i in range(embeddings.shape[1])]
    emb_df = pd.DataFrame(embeddings, index=df.index, columns=emb_cols)
    X = pd.concat([X, emb_df], axis=1)
    logger.info(f"Added {len(emb_cols)} embedding features")
```

---

### Task 3B.4: Feature Validation & Statistics ⏱️ 30 min

**File:** `ml/validate_features.py`

```python
"""
Validate and analyze features from vw_ml_features.

Checks for:
- Missing values
- Infinite values
- Data type consistency
- Feature distributions
- Correlation analysis

Usage:
    python -m ml.validate_features --limit 5000
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from google.cloud import bigquery
from dotenv import load_dotenv

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.logging import configure_logging

load_dotenv()
logger = configure_logging(service_name="validate_features")

PROJECT_ID = os.getenv('GCP_PROJECT_ID', 'sg-job-market')
DATASET_ID = os.getenv('BQ_DATASET_ID', 'sg_job_market')


def load_features(limit: int = None) -> pd.DataFrame:
    """Load features from BigQuery."""
    client = bigquery.Client(project=PROJECT_ID)
    
    query = f"""
    SELECT *
    FROM `{PROJECT_ID}.{DATASET_ID}.vw_ml_features`
    """
    if limit:
        query += f" LIMIT {limit}"
    
    logger.info(f"Loading features (limit={limit})...")
    df = client.query(query).to_dataframe()
    logger.info(f"✅ Loaded {len(df):,} rows")
    return df


def check_missing_values(df: pd.DataFrame) -> None:
    """Report missing values."""
    logger.info("\n" + "=" * 70)
    logger.info("MISSING VALUES")
    logger.info("=" * 70)
    
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    
    missing_df = pd.DataFrame({
        'missing_count': missing,
        'missing_pct': missing_pct
    }).sort_values('missing_pct', ascending=False)
    
    # Show only columns with missing values
    missing_df = missing_df[missing_df['missing_count'] > 0]
    
    if len(missing_df) == 0:
        logger.info("✅ No missing values found!")
    else:
        logger.info(f"\n{missing_df.to_string()}")


def check_infinite_values(df: pd.DataFrame) -> None:
    """Check for infinite values in numerical columns."""
    logger.info("\n" + "=" * 70)
    logger.info("INFINITE VALUES")
    logger.info("=" * 70)
    
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    inf_counts = {}
    
    for col in numerical_cols:
        inf_count = np.isinf(df[col]).sum()
        if inf_count > 0:
            inf_counts[col] = inf_count
    
    if len(inf_counts) == 0:
        logger.info("✅ No infinite values found!")
    else:
        for col, count in inf_counts.items():
            logger.info(f"⚠️  {col}: {count} infinite values")


def analyze_distributions(df: pd.DataFrame) -> None:
    """Analyze numerical feature distributions."""
    logger.info("\n" + "=" * 70)
    logger.info("NUMERICAL FEATURE DISTRIBUTIONS")
    logger.info("=" * 70)
    
    numerical_cols = [
        'salary_mid_monthly',
        'salary_range',
        'description_length',
        'title_length',
        'days_since_posted'
    ]
    
    for col in numerical_cols:
        if col not in df.columns:
            continue
        
        data = df[col].dropna()
        logger.info(f"\n{col}:")
        logger.info(f"  Count: {len(data):,}")
        logger.info(f"  Mean: {data.mean():.2f}")
        logger.info(f"  Median: {data.median():.2f}")
        logger.info(f"  Std: {data.std():.2f}")
        logger.info(f"  Min: {data.min():.2f}")
        logger.info(f"  Max: {data.max():.2f}")
        logger.info(f"  25%: {data.quantile(0.25):.2f}")
        logger.info(f"  75%: {data.quantile(0.75):.2f}")


def analyze_categorical(df: pd.DataFrame) -> None:
    """Analyze categorical feature distributions."""
    logger.info("\n" + "=" * 70)
    logger.info("CATEGORICAL FEATURE DISTRIBUTIONS")
    logger.info("=" * 70)
    
    categorical_cols = [
        'source',
        'job_classification',
        'job_location',
        'job_work_type',
        'company_industry',
        'company_size'
    ]
    
    for col in categorical_cols:
        if col not in df.columns:
            continue
        
        logger.info(f"\n{col}:")
        value_counts = df[col].value_counts()
        logger.info(f"  Unique values: {len(value_counts)}")
        logger.info(f"  Top 5:")
        for val, count in value_counts.head(5).items():
            pct = count / len(df) * 100
            logger.info(f"    {val}: {count:,} ({pct:.1f}%)")


def correlation_analysis(df: pd.DataFrame) -> None:
    """Compute correlation matrix for numerical features."""
    logger.info("\n" + "=" * 70)
    logger.info("CORRELATION ANALYSIS")
    logger.info("=" * 70)
    
    numerical_cols = [
        'salary_mid_monthly',
        'salary_range',
        'description_length',
        'title_length',
        'days_since_posted'
    ]
    
    # Filter to available columns
    available_cols = [c for c in numerical_cols if c in df.columns]
    numerical_df = df[available_cols].dropna()
    
    corr_matrix = numerical_df.corr()
    logger.info(f"\n{corr_matrix.to_string()}")
    
    # Visualize
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", center=0)
    plt.title("Feature Correlation Matrix")
    plt.tight_layout()
    plt.savefig("feature_correlation.png")
    logger.info("\n✅ Saved correlation heatmap: feature_correlation.png")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Validate ML features")
    parser.add_argument("--limit", type=int, default=None, help="Row limit for testing")
    args = parser.parse_args()
    
    # Load data
    df = load_features(limit=args.limit)
    
    # Run checks
    check_missing_values(df)
    check_infinite_values(df)
    analyze_distributions(df)
    analyze_categorical(df)
    correlation_analysis(df)
    
    logger.info("\n" + "=" * 70)
    logger.info("✅ VALIDATION COMPLETE")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
```

**Run:**
```bash
.venv/Scripts/python.exe -m ml.validate_features --limit 5000
```

---

### Task 3B.5: Test Notebook ⏱️ 20 min

**File:** `notebooks/ml_test_features.ipynb`

```python
# Cell 1: Setup
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from dotenv import load_dotenv

sys.path.insert(0, os.path.abspath('..'))
from ml.features import FeatureEngineer, FeatureConfig

load_dotenv()

PROJECT_ID = os.getenv('GCP_PROJECT_ID', 'sg-job-market')
DATASET_ID = os.getenv('BQ_DATASET_ID', 'sg_job_market')

print("✅ Setup complete")

# Cell 2: Load features
fe = FeatureEngineer()
df, embeddings = fe.load_from_bigquery(
    project_id=PROJECT_ID,
    dataset_id=DATASET_ID,
    limit=1000,
    filters={"salary_mid_monthly": "IS NOT NULL"}
)

print(f"Data shape: {df.shape}")
print(f"Embeddings shape: {embeddings.shape}")
df.head()

# Cell 3: Prepare training data (no PCA)
X, y = fe.prepare_training_data(df, embeddings, target="salary_mid_monthly")

print(f"Feature matrix: {X.shape}")
print(f"Target: {y.shape}")
print(f"Feature names: {list(X.columns[:10])}")

# Cell 4: Prepare with PCA
config = FeatureConfig(embedding_pca_components=50)
fe_pca = FeatureEngineer(config=config)

X_pca, y_pca = fe_pca.prepare_training_data(df, embeddings, target="salary_mid_monthly")

print(f"Feature matrix with PCA: {X_pca.shape}")
print(f"Reduced from 384 → 50 embedding dimensions")

# Cell 5: Train/test split
from ml.features import create_train_test_split

train_df, test_df = create_train_test_split(
    df,
    test_size=0.2,
    time_column="job_posted_timestamp"
)

print(f"Train set: {len(train_df)}")
print(f"Test set: {len(test_df)}")

# Cell 6: Visualize feature distributions
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

axes[0, 0].hist(y, bins=50)
axes[0, 0].set_title('Salary Distribution')
axes[0, 0].set_xlabel('Salary (SGD/month)')

axes[0, 1].hist(df['description_length'].dropna(), bins=50)
axes[0, 1].set_title('Description Length Distribution')

axes[1, 0].hist(df['days_since_posted'].dropna(), bins=50)
axes[1, 0].set_title('Days Since Posted')

axes[1, 1].bar(df['source'].value_counts().index, df['source'].value_counts().values)
axes[1, 1].set_title('Jobs by Source')

plt.tight_layout()
plt.show()
```

---

## ✅ Acceptance Criteria

Phase 3B complete when:
- [ ] `vw_ml_features` view created in BigQuery
- [ ] `load_from_bigquery()` method implemented
- [ ] PCA dimensionality reduction working
- [ ] Feature validation script runs without errors
- [ ] Test notebook executes successfully
- [ ] Feature matrix shape: (n_jobs, ~450 features) without PCA
- [ ] Feature matrix shape: (n_jobs, ~100 features) with PCA
- [ ] No NaN or inf values in final feature matrix
- [ ] Documentation updated in 04_ml_engineer.agent.md

---

## 🔄 Next Steps (Phase 3C)

After Phase 3B completion:
1. Train salary predictor (LightGBM regression)
2. Train role classifier (LightGBM multi-class)
3. Train job clusterer (KMeans on embeddings)
4. Evaluate models and save artifacts

**Estimated Time for Phase 3B:** 2-3 hours
