"""ML package.

Training pipelines, feature engineering, evaluation, and model registry.

Modules:
- train: CLI and programmatic training entrypoints
- features: Feature engineering and data preparation
- salary_predictor: LightGBM salary regression model
- clustering: KMeans job clustering

Usage:
    from ml.salary_predictor import SalaryPredictor
    from ml.clustering import JobClusterer
    from ml.features import FeatureEngineer
    from ml.train import train_all
"""

from ml.train import train_all, train_salary_predictor, train_role_classifier, train_job_clusterer

__all__ = [
    "train_all",
    "train_salary_predictor",
    "train_role_classifier",
    "train_job_clusterer",
]

