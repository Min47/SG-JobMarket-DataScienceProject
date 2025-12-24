"""
ML training entrypoints.

Provides CLI and programmatic access to train all models:
- Salary Predictor (LightGBM regression)
- Role Classifier (LightGBM multi-class)
- Job Clusterer (KMeans)

Usage:
    CLI: python -m ml.train --model salary --limit 1000
    Code: from ml.train import train_all; train_all()
"""

from __future__ import annotations

import argparse
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# Model registry path
MODELS_DIR = Path(__file__).parent.parent / "models"


def train_salary_predictor(
    data_limit: Optional[int] = None,
    save_model: bool = True,
) -> Dict[str, Any]:
    """
    Train salary prediction model (LightGBM regression).

    Args:
        data_limit: Limit number of training samples (for testing).
        save_model: Whether to save model to disk.

    Returns:
        Dict with model, metrics, and path.
    """
    logger.info("Training salary predictor...")

    # TODO: Implement after features.py is ready
    # 1. Load features from BigQuery vw_ml_features
    # 2. Filter to jobs with salary data
    # 3. Train LightGBM regressor
    # 4. Evaluate and log metrics
    # 5. Save model artifacts

    metrics = {
        "rmse": 0.0,
        "mae": 0.0,
        "r2": 0.0,
        "training_samples": 0,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "status": "NOT_IMPLEMENTED",
    }

    logger.warning("SalaryPredictor training not yet implemented")
    return {"model": None, "metrics": metrics, "path": None}


def train_role_classifier(
    data_limit: Optional[int] = None,
    save_model: bool = True,
) -> Dict[str, Any]:
    """
    Train role classification model (LightGBM multi-class).

    Args:
        data_limit: Limit number of training samples.
        save_model: Whether to save model to disk.

    Returns:
        Dict with model, metrics, and path.
    """
    logger.info("Training role classifier...")

    # TODO: Implement after features.py is ready
    # 1. Load features from BigQuery
    # 2. Use job_classification as target
    # 3. Train LightGBM classifier with class weights
    # 4. Evaluate F1 macro, confusion matrix
    # 5. Save model artifacts

    metrics = {
        "accuracy": 0.0,
        "f1_macro": 0.0,
        "f1_weighted": 0.0,
        "num_classes": 0,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "status": "NOT_IMPLEMENTED",
    }

    logger.warning("RoleClassifier training not yet implemented")
    return {"model": None, "metrics": metrics, "path": None}


def train_job_clusterer(
    n_clusters: int = 10,
    data_limit: Optional[int] = None,
    save_model: bool = True,
) -> Dict[str, Any]:
    """
    Train job clustering model (KMeans on embeddings).

    Args:
        n_clusters: Number of clusters to create.
        data_limit: Limit number of samples.
        save_model: Whether to save model to disk.

    Returns:
        Dict with model, metrics, cluster_labels, and path.
    """
    logger.info(f"Training job clusterer (n_clusters={n_clusters})...")

    # TODO: Implement after embeddings are generated
    # 1. Load embeddings from BigQuery job_embeddings
    # 2. Run elbow method to validate n_clusters
    # 3. Train KMeans
    # 4. Generate cluster labels (top keywords)
    # 5. Save model artifacts

    metrics = {
        "silhouette_score": 0.0,
        "inertia": 0.0,
        "n_clusters": n_clusters,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "status": "NOT_IMPLEMENTED",
    }

    logger.warning("JobClusterer training not yet implemented")
    return {"model": None, "metrics": metrics, "cluster_labels": {}, "path": None}


def train_all(
    data_limit: Optional[int] = None,
    save_models: bool = True,
) -> Dict[str, Any]:
    """
    Train all models in sequence.

    Args:
        data_limit: Limit training samples per model.
        save_models: Whether to save models to disk.

    Returns:
        Dict with results for each model.
    """
    logger.info("=" * 50)
    logger.info("Starting full ML training pipeline")
    logger.info("=" * 50)

    results = {}

    # Train each model
    results["salary_predictor"] = train_salary_predictor(data_limit, save_models)
    results["role_classifier"] = train_role_classifier(data_limit, save_models)
    results["job_clusterer"] = train_job_clusterer(data_limit=data_limit, save_model=save_models)

    # Summary
    logger.info("=" * 50)
    logger.info("Training complete. Summary:")
    for name, result in results.items():
        status = result["metrics"].get("status", "UNKNOWN")
        logger.info(f"  {name}: {status}")
    logger.info("=" * 50)

    return results


def main() -> None:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(description="Train ML models for SG Job Market")
    parser.add_argument(
        "--model",
        choices=["salary", "classifier", "clustering", "all"],
        default="all",
        help="Which model to train (default: all)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit training samples (for testing)",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Don't save models to disk",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    save = not args.no_save

    if args.model == "salary":
        train_salary_predictor(args.limit, save)
    elif args.model == "classifier":
        train_role_classifier(args.limit, save)
    elif args.model == "clustering":
        train_job_clusterer(data_limit=args.limit, save_model=save)
    else:
        train_all(args.limit, save)


if __name__ == "__main__":
    main()

