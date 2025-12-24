"""
Salary prediction model using LightGBM.

Predicts monthly SGD salary based on job features and embeddings.

Usage:
    from ml.salary_predictor import SalaryPredictor
    
    model = SalaryPredictor()
    model.train(X_train, y_train)
    predictions = model.predict(X_test)
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class SalaryPredictor:
    """LightGBM-based salary prediction model."""

    DEFAULT_PARAMS = {
        "objective": "regression",
        "metric": "rmse",
        "boosting_type": "gbdt",
        "num_leaves": 31,
        "learning_rate": 0.05,
        "n_estimators": 500,
        "min_child_samples": 20,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42,
        "verbose": -1,
    }

    def __init__(
        self,
        model_type: str = "lightgbm",
        params: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize salary predictor.

        Args:
            model_type: Model type ("lightgbm", "xgboost", "linear").
            params: Model hyperparameters. Uses defaults if None.
        """
        self.model_type = model_type
        self.params = params or self.DEFAULT_PARAMS.copy()
        self._model = None
        self._feature_names: list[str] = []
        self._metrics: Dict[str, float] = {}
        self._trained_at: Optional[datetime] = None
        logger.info(f"SalaryPredictor initialized with model_type={model_type}")

    def train(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
    ) -> Dict[str, float]:
        """
        Train the salary prediction model.

        Args:
            X: Training features.
            y: Training targets (monthly salary).
            X_val: Optional validation features.
            y_val: Optional validation targets.

        Returns:
            Dict with training metrics.
        """
        logger.info(f"Training {self.model_type} on {len(X)} samples")

        self._feature_names = list(X.columns)

        if self.model_type == "lightgbm":
            self._train_lightgbm(X, y, X_val, y_val)
        elif self.model_type == "xgboost":
            self._train_xgboost(X, y, X_val, y_val)
        elif self.model_type == "linear":
            self._train_linear(X, y)
        else:
            raise ValueError(f"Unknown model_type: {self.model_type}")

        self._trained_at = datetime.now(timezone.utc)

        # Evaluate on training data
        train_pred = self.predict(X)
        self._metrics = self._calculate_metrics(y, train_pred, prefix="train_")

        # Evaluate on validation if provided
        if X_val is not None and y_val is not None:
            val_pred = self.predict(X_val)
            val_metrics = self._calculate_metrics(y_val, val_pred, prefix="val_")
            self._metrics.update(val_metrics)

        logger.info(f"Training complete. RMSE: {self._metrics.get('train_rmse', 'N/A')}")
        return self._metrics

    def _train_lightgbm(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        X_val: Optional[pd.DataFrame],
        y_val: Optional[pd.Series],
    ) -> None:
        """Train LightGBM model."""
        try:
            import lightgbm as lgb
        except ImportError:
            raise ImportError("lightgbm not installed. Run: pip install lightgbm")

        callbacks = []
        eval_set = None

        if X_val is not None and y_val is not None:
            eval_set = [(X_val, y_val)]
            callbacks.append(lgb.early_stopping(stopping_rounds=50, verbose=False))

        self._model = lgb.LGBMRegressor(**self.params)
        self._model.fit(
            X, y,
            eval_set=eval_set,
            callbacks=callbacks if callbacks else None,
        )

    def _train_xgboost(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        X_val: Optional[pd.DataFrame],
        y_val: Optional[pd.Series],
    ) -> None:
        """Train XGBoost model."""
        try:
            import xgboost as xgb
        except ImportError:
            raise ImportError("xgboost not installed. Run: pip install xgboost")

        xgb_params = {
            "objective": "reg:squarederror",
            "n_estimators": self.params.get("n_estimators", 500),
            "learning_rate": self.params.get("learning_rate", 0.05),
            "max_depth": 6,
            "random_state": 42,
        }

        self._model = xgb.XGBRegressor(**xgb_params)

        eval_set = [(X, y)]
        if X_val is not None:
            eval_set.append((X_val, y_val))

        self._model.fit(
            X, y,
            eval_set=eval_set,
            verbose=False,
        )

    def _train_linear(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Train linear regression baseline."""
        from sklearn.linear_model import Ridge

        self._model = Ridge(alpha=1.0)
        self._model.fit(X, y)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict salary for given features.

        Args:
            X: Features DataFrame.

        Returns:
            Array of predicted salaries.
        """
        if self._model is None:
            raise ValueError("Model not trained. Call train() first.")
        return self._model.predict(X)

    def _calculate_metrics(
        self,
        y_true: pd.Series,
        y_pred: np.ndarray,
        prefix: str = "",
    ) -> Dict[str, float]:
        """Calculate regression metrics."""
        from sklearn.metrics import (
            mean_absolute_error,
            mean_squared_error,
            r2_score,
        )

        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)

        # MAPE (avoid division by zero)
        y_true_arr = np.array(y_true)
        mask = y_true_arr != 0
        mape = np.mean(np.abs((y_true_arr[mask] - y_pred[mask]) / y_true_arr[mask])) * 100

        return {
            f"{prefix}rmse": round(rmse, 2),
            f"{prefix}mae": round(mae, 2),
            f"{prefix}r2": round(r2, 4),
            f"{prefix}mape": round(mape, 2),
        }

    def get_feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        """
        Get feature importance from trained model.

        Args:
            top_n: Number of top features to return.

        Returns:
            DataFrame with feature names and importance scores.
        """
        if self._model is None:
            raise ValueError("Model not trained.")

        if hasattr(self._model, "feature_importances_"):
            importance = self._model.feature_importances_
        else:
            return pd.DataFrame()

        df = pd.DataFrame({
            "feature": self._feature_names,
            "importance": importance,
        })
        return df.nlargest(top_n, "importance").reset_index(drop=True)

    def save(self, path: Union[str, Path]) -> None:
        """
        Save model to disk.

        Args:
            path: Directory path to save model artifacts.
        """
        import joblib

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save model
        joblib.dump(self._model, path / "model.joblib")

        # Save config
        config = {
            "model_type": self.model_type,
            "params": self.params,
            "feature_names": self._feature_names,
            "trained_at": self._trained_at.isoformat() if self._trained_at else None,
        }
        with open(path / "config.json", "w") as f:
            json.dump(config, f, indent=2)

        # Save metrics
        with open(path / "metrics.json", "w") as f:
            json.dump(self._metrics, f, indent=2)

        logger.info(f"Model saved to {path}")

    @classmethod
    def load(cls, path: Union[str, Path]) -> "SalaryPredictor":
        """
        Load model from disk.

        Args:
            path: Directory path with model artifacts.

        Returns:
            Loaded SalaryPredictor instance.
        """
        import joblib

        path = Path(path)

        # Load config
        with open(path / "config.json") as f:
            config = json.load(f)

        # Create instance
        predictor = cls(
            model_type=config["model_type"],
            params=config.get("params"),
        )
        predictor._model = joblib.load(path / "model.joblib")
        predictor._feature_names = config.get("feature_names", [])

        if config.get("trained_at"):
            predictor._trained_at = datetime.fromisoformat(config["trained_at"])

        # Load metrics
        metrics_path = path / "metrics.json"
        if metrics_path.exists():
            with open(metrics_path) as f:
                predictor._metrics = json.load(f)

        logger.info(f"Model loaded from {path}")
        return predictor
