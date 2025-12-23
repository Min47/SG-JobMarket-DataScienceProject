"""
Feature engineering module for ML models.

Transforms cleaned job data into ML-ready features including:
- Numerical features (salary, description length, etc.)
- Categorical encoding (location, work type, etc.)
- Embedding features (from job_embeddings table)

Usage:
    from ml.features import FeatureEngineer
    
    fe = FeatureEngineer()
    X, y = fe.prepare_training_data(target="salary_mid")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class FeatureConfig:
    """Configuration for feature engineering."""

    # Numerical features to extract
    numerical_features: List[str] = None
    # Categorical features to one-hot encode
    categorical_features: List[str] = None
    # Whether to include embeddings
    include_embeddings: bool = True
    # PCA components for embeddings (None = use all)
    embedding_pca_components: Optional[int] = None
    # Fill missing salaries with median
    impute_salary: bool = True

    def __post_init__(self):
        if self.numerical_features is None:
            self.numerical_features = [
                "job_salary_min_sgd_monthly",
                "job_salary_max_sgd_monthly",
                "description_length",
                "title_length",
                "days_since_posted",
            ]
        if self.categorical_features is None:
            self.categorical_features = [
                "source",
                "job_location",
                "job_work_type",
            ]


class FeatureEngineer:
    """Transform job data into ML features."""

    def __init__(self, config: Optional[FeatureConfig] = None):
        """
        Initialize feature engineer.

        Args:
            config: Feature configuration. Uses defaults if None.
        """
        self.config = config or FeatureConfig()
        self._label_encoders: Dict[str, Any] = {}
        self._scaler = None
        self._pca = None
        logger.info("FeatureEngineer initialized")

    def extract_numerical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract and transform numerical features.

        Args:
            df: DataFrame with job data.

        Returns:
            DataFrame with numerical features only.
        """
        features = pd.DataFrame(index=df.index)

        # Salary features
        if "job_salary_min_sgd_monthly" in df.columns:
            features["salary_min"] = df["job_salary_min_sgd_monthly"]
            features["salary_max"] = df["job_salary_max_sgd_monthly"]
            features["salary_mid"] = (features["salary_min"] + features["salary_max"]) / 2
            features["salary_range"] = features["salary_max"] - features["salary_min"]

            # Log transform (handle zeros)
            features["salary_min_log"] = np.log1p(features["salary_min"])
            features["salary_max_log"] = np.log1p(features["salary_max"])

        # Text length features
        if "job_description" in df.columns:
            features["description_length"] = df["job_description"].fillna("").str.len()
        if "job_title" in df.columns:
            features["title_length"] = df["job_title"].fillna("").str.len()

        # Temporal features
        if "job_posted_timestamp" in df.columns:
            posted = pd.to_datetime(df["job_posted_timestamp"])
            now = pd.Timestamp.now(tz="UTC")
            features["days_since_posted"] = (now - posted).dt.days

        logger.info(f"Extracted {len(features.columns)} numerical features")
        return features

    def extract_categorical_features(
        self,
        df: pd.DataFrame,
        fit: bool = True,
    ) -> pd.DataFrame:
        """
        Extract and encode categorical features.

        Args:
            df: DataFrame with job data.
            fit: If True, fit encoders. If False, use existing encoders.

        Returns:
            DataFrame with one-hot encoded categorical features.
        """
        features = pd.DataFrame(index=df.index)

        for col in self.config.categorical_features:
            if col not in df.columns:
                logger.warning(f"Categorical column {col} not found in data")
                continue

            # One-hot encode
            dummies = pd.get_dummies(
                df[col].fillna("unknown"),
                prefix=col,
                dtype=int,
            )
            features = pd.concat([features, dummies], axis=1)

        logger.info(f"Extracted {len(features.columns)} categorical features")
        return features

    def prepare_training_data(
        self,
        df: pd.DataFrame,
        embeddings: Optional[np.ndarray] = None,
        target: str = "salary_mid",
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare full feature matrix and target for training.

        Args:
            df: DataFrame with cleaned job data.
            embeddings: Optional embedding matrix (n_jobs x embedding_dim).
            target: Target column name.

        Returns:
            Tuple of (X features DataFrame, y target Series).
        """
        logger.info(f"Preparing training data with target={target}")

        # Extract features
        numerical = self.extract_numerical_features(df)
        categorical = self.extract_categorical_features(df)

        # Combine
        X = pd.concat([numerical, categorical], axis=1)

        # Add embeddings if provided
        if embeddings is not None and self.config.include_embeddings:
            emb_cols = [f"emb_{i}" for i in range(embeddings.shape[1])]
            emb_df = pd.DataFrame(embeddings, index=df.index, columns=emb_cols)

            # Optional PCA reduction
            if self.config.embedding_pca_components:
                # TODO: Implement PCA
                pass

            X = pd.concat([X, emb_df], axis=1)
            logger.info(f"Added {len(emb_cols)} embedding features")

        # Extract target
        if target in numerical.columns:
            y = numerical[target]
        elif target in df.columns:
            y = df[target]
        else:
            raise ValueError(f"Target column {target} not found")

        # Remove target from features if present
        if target in X.columns:
            X = X.drop(columns=[target])

        # Filter to rows with valid target
        valid_idx = ~y.isna()
        X = X.loc[valid_idx]
        y = y.loc[valid_idx]

        logger.info(f"Final feature matrix: {X.shape}, target: {len(y)}")
        return X, y

    def get_feature_names(self) -> List[str]:
        """Return list of feature names after fitting."""
        # TODO: Implement after fitting
        return []


def create_train_test_split(
    df: pd.DataFrame,
    test_size: float = 0.2,
    time_column: str = "job_posted_timestamp",
    stratify_column: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Create time-based train/test split (no data leakage).

    Args:
        df: Full DataFrame.
        test_size: Fraction for test set (most recent).
        time_column: Column with timestamp for ordering.
        stratify_column: Optional column to stratify by.

    Returns:
        Tuple of (train_df, test_df).
    """
    # Sort by time
    df_sorted = df.sort_values(time_column)

    # Split point
    split_idx = int(len(df_sorted) * (1 - test_size))

    train_df = df_sorted.iloc[:split_idx]
    test_df = df_sorted.iloc[split_idx:]

    logger.info(f"Train set: {len(train_df)}, Test set: {len(test_df)}")
    return train_df, test_df
