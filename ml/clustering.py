"""
Job clustering module using KMeans.

Groups jobs by embedding similarity into meaningful clusters.

Usage:
    from ml.clustering import JobClusterer
    
    clusterer = JobClusterer(n_clusters=10)
    clusterer.fit(embeddings)
    labels = clusterer.predict(new_embeddings)
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class JobClusterer:
    """KMeans-based job clustering on embeddings."""

    def __init__(
        self,
        n_clusters: int = 10,
        random_state: int = 42,
    ):
        """
        Initialize job clusterer.

        Args:
            n_clusters: Number of clusters to create.
            random_state: Random seed for reproducibility.
        """
        self.n_clusters = n_clusters
        self.random_state = random_state
        self._model = None
        self._cluster_labels: Dict[int, str] = {}
        self._metrics: Dict[str, float] = {}
        self._trained_at: Optional[datetime] = None
        logger.info(f"JobClusterer initialized with n_clusters={n_clusters}")

    def fit(
        self,
        embeddings: np.ndarray,
        job_titles: Optional[List[str]] = None,
    ) -> Dict[str, float]:
        """
        Fit clustering model to embeddings.

        Args:
            embeddings: Job embeddings (n_samples x embedding_dim).
            job_titles: Optional job titles for cluster labeling.

        Returns:
            Dict with clustering metrics.
        """
        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score

        logger.info(f"Fitting KMeans on {len(embeddings)} embeddings")

        # Fit KMeans
        self._model = KMeans(
            n_clusters=self.n_clusters,
            random_state=self.random_state,
            n_init=10,
        )
        labels = self._model.fit_predict(embeddings)

        self._trained_at = datetime.utcnow()

        # Calculate metrics
        silhouette = silhouette_score(embeddings, labels)
        inertia = self._model.inertia_

        self._metrics = {
            "silhouette_score": round(silhouette, 4),
            "inertia": round(inertia, 2),
            "n_clusters": self.n_clusters,
            "n_samples": len(embeddings),
        }

        # Cluster size distribution
        cluster_sizes = Counter(labels)
        for i in range(self.n_clusters):
            self._metrics[f"cluster_{i}_size"] = cluster_sizes.get(i, 0)

        # Generate cluster labels if titles provided
        if job_titles is not None:
            self._generate_cluster_labels(labels, job_titles)

        logger.info(f"Clustering complete. Silhouette: {silhouette:.4f}")
        return self._metrics

    def predict(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Assign cluster labels to new embeddings.

        Args:
            embeddings: New job embeddings.

        Returns:
            Array of cluster IDs.
        """
        if self._model is None:
            raise ValueError("Model not fitted. Call fit() first.")
        return self._model.predict(embeddings)

    def get_cluster_centers(self) -> np.ndarray:
        """Get cluster centroid embeddings."""
        if self._model is None:
            raise ValueError("Model not fitted.")
        return self._model.cluster_centers_

    def _generate_cluster_labels(
        self,
        labels: np.ndarray,
        job_titles: List[str],
    ) -> None:
        """
        Generate human-readable cluster labels from top job titles.

        Args:
            labels: Cluster assignments.
            job_titles: Job titles for each sample.
        """
        for cluster_id in range(self.n_clusters):
            # Get titles in this cluster
            cluster_mask = labels == cluster_id
            cluster_titles = [
                t for t, m in zip(job_titles, cluster_mask) if m and t
            ]

            if not cluster_titles:
                self._cluster_labels[cluster_id] = f"Cluster {cluster_id}"
                continue

            # Find most common words (simple approach)
            words = []
            for title in cluster_titles[:100]:  # Sample first 100
                words.extend(title.lower().split())

            # Filter common words
            stopwords = {"the", "a", "an", "and", "or", "in", "at", "for", "to", "of", "-"}
            word_counts = Counter(w for w in words if w not in stopwords and len(w) > 2)

            # Top 2-3 keywords
            top_words = [w for w, _ in word_counts.most_common(3)]
            label = " / ".join(top_words).title() if top_words else f"Cluster {cluster_id}"
            self._cluster_labels[cluster_id] = label

        logger.info(f"Generated {len(self._cluster_labels)} cluster labels")

    def get_cluster_labels(self) -> Dict[int, str]:
        """Get human-readable cluster labels."""
        return self._cluster_labels.copy()

    def analyze_clusters(
        self,
        df: pd.DataFrame,
        labels: np.ndarray,
    ) -> pd.DataFrame:
        """
        Analyze cluster characteristics.

        Args:
            df: Job data DataFrame.
            labels: Cluster assignments.

        Returns:
            DataFrame with cluster statistics.
        """
        df_with_cluster = df.copy()
        df_with_cluster["cluster_id"] = labels
        df_with_cluster["cluster_name"] = df_with_cluster["cluster_id"].map(
            self._cluster_labels
        )

        # Aggregate stats per cluster
        stats = []
        for cluster_id in range(self.n_clusters):
            cluster_df = df_with_cluster[df_with_cluster["cluster_id"] == cluster_id]

            stat = {
                "cluster_id": cluster_id,
                "cluster_name": self._cluster_labels.get(cluster_id, f"Cluster {cluster_id}"),
                "count": len(cluster_df),
                "pct": round(len(cluster_df) / len(df) * 100, 1),
            }

            # Salary stats if available
            if "job_salary_min_sgd_monthly" in cluster_df.columns:
                salary = cluster_df["job_salary_min_sgd_monthly"].dropna()
                if len(salary) > 0:
                    stat["avg_salary"] = round(salary.mean(), 0)
                    stat["median_salary"] = round(salary.median(), 0)

            # Top location
            if "job_location" in cluster_df.columns:
                top_loc = cluster_df["job_location"].mode()
                stat["top_location"] = top_loc.iloc[0] if len(top_loc) > 0 else "N/A"

            stats.append(stat)

        return pd.DataFrame(stats)

    def find_optimal_clusters(
        self,
        embeddings: np.ndarray,
        k_range: Tuple[int, int] = (5, 20),
    ) -> Dict[int, float]:
        """
        Find optimal number of clusters using elbow method.

        Args:
            embeddings: Job embeddings.
            k_range: Range of k values to try (min, max).

        Returns:
            Dict mapping k to inertia score.
        """
        from sklearn.cluster import KMeans

        logger.info(f"Finding optimal clusters in range {k_range}")

        results = {}
        for k in range(k_range[0], k_range[1] + 1):
            kmeans = KMeans(n_clusters=k, random_state=self.random_state, n_init=10)
            kmeans.fit(embeddings)
            results[k] = kmeans.inertia_
            logger.debug(f"k={k}, inertia={kmeans.inertia_:.2f}")

        return results

    def save(self, path: Union[str, Path]) -> None:
        """Save model to disk."""
        import joblib

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save model
        joblib.dump(self._model, path / "model.joblib")

        # Save config
        config = {
            "n_clusters": self.n_clusters,
            "random_state": self.random_state,
            "cluster_labels": self._cluster_labels,
            "trained_at": self._trained_at.isoformat() if self._trained_at else None,
        }
        with open(path / "config.json", "w") as f:
            json.dump(config, f, indent=2)

        # Save metrics
        with open(path / "metrics.json", "w") as f:
            json.dump(self._metrics, f, indent=2)

        logger.info(f"Clusterer saved to {path}")

    @classmethod
    def load(cls, path: Union[str, Path]) -> "JobClusterer":
        """Load model from disk."""
        import joblib

        path = Path(path)

        with open(path / "config.json") as f:
            config = json.load(f)

        clusterer = cls(
            n_clusters=config["n_clusters"],
            random_state=config.get("random_state", 42),
        )
        clusterer._model = joblib.load(path / "model.joblib")
        clusterer._cluster_labels = {
            int(k): v for k, v in config.get("cluster_labels", {}).items()
        }

        if config.get("trained_at"):
            clusterer._trained_at = datetime.fromisoformat(config["trained_at"])

        metrics_path = path / "metrics.json"
        if metrics_path.exists():
            with open(metrics_path) as f:
                clusterer._metrics = json.load(f)

        logger.info(f"Clusterer loaded from {path}")
        return clusterer


def reduce_dimensions(
    embeddings: np.ndarray,
    method: str = "pca",
    n_components: int = 2,
) -> np.ndarray:
    """
    Reduce embedding dimensions for visualization.

    Args:
        embeddings: High-dimensional embeddings.
        method: "pca" or "umap".
        n_components: Output dimensions (2 or 3).

    Returns:
        Reduced embeddings.
    """
    logger.info(f"Reducing dimensions with {method} to {n_components}D")

    if method == "pca":
        from sklearn.decomposition import PCA
        reducer = PCA(n_components=n_components, random_state=42)
    elif method == "umap":
        try:
            import umap
            reducer = umap.UMAP(n_components=n_components, random_state=42)
        except ImportError:
            raise ImportError("umap-learn not installed. Run: pip install umap-learn")
    else:
        raise ValueError(f"Unknown method: {method}")

    return reducer.fit_transform(embeddings)
