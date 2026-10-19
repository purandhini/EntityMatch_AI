"""Model training module with F0.5 threshold optimization and cross-validation."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
try:
    import joblib
    HAS_JOBLIB = True
except ImportError:
    HAS_JOBLIB = False
    import pickle

from ..config import TrainingConfig, DEFAULT_CONFIG
from .evaluation import EvaluationMetrics

# Model import fallbacks
try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import StratifiedKFold
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


class ModelTrainer:
    """
    Trains gradient boosting or linear classifiers for entity resolution
    and tunes the decision threshold for the competition F0.5 metric.
    """

    def __init__(self, config: Optional[TrainingConfig] = None):
        self.config = config or DEFAULT_CONFIG.training
        self.model: Optional[Any] = None
        self.feature_names: List[str] = []
        self.optimal_threshold: float = 0.5
        self.training_metadata: Dict[str, Any] = {}

    def _init_estimator(self):
        """Initializes the underlying classifier according to configuration."""
        model_type = self.config.model_type.lower()
        if model_type == "lightgbm":
            if HAS_LIGHTGBM:
                return lgb.LGBMClassifier(**self.config.lightgbm_params)
            elif HAS_SKLEARN:
                print("Notice: lightgbm not installed; falling back to sklearn.ensemble.RandomForestClassifier")
                return RandomForestClassifier(n_estimators=100, random_state=self.config.random_state)
            else:
                raise ImportError("Neither lightgbm nor scikit-learn is installed. Please install requirements.txt.")
        elif model_type == "logistic_regression":
            if HAS_SKLEARN:
                return LogisticRegression(max_iter=1000, random_state=self.config.random_state)
            raise ImportError("scikit-learn is required for logistic_regression.")
        else:
            if HAS_SKLEARN:
                return RandomForestClassifier(n_estimators=100, random_state=self.config.random_state)
            raise ImportError("scikit-learn is required for random_forest.")

    def train_with_cv(
        self,
        X: pd.DataFrame,
        y: Union[pd.Series, np.ndarray],
    ) -> Dict[str, Any]:
        """
        Runs stratified K-Fold cross validation, computes out-of-fold probabilities,
        finds the threshold maximizing F0.5, and fits the final model on all data.

        Args:
            X: Feature matrix DataFrame.
            y: Binary target labels (1 = match, 0 = non-match).

        Returns:
            Dictionary containing cross-validation metrics, optimal threshold, and feature importances.
        """
        self.feature_names = list(X.columns)
        X_mat = X.values.astype(np.float32)
        y_vec = np.asarray(y).astype(int)

        n_samples = len(y_vec)
        oof_probs = np.zeros(n_samples, dtype=float)

        if HAS_SKLEARN and self.config.cv_folds > 1:
            skf = StratifiedKFold(
                n_splits=self.config.cv_folds,
                shuffle=True,
                random_state=self.config.random_state,
            )
            for fold, (train_idx, val_idx) in enumerate(skf.split(X_mat, y_vec)):
                fold_model = self._init_estimator()
                fold_model.fit(X_mat[train_idx], y_vec[train_idx])
                oof_probs[val_idx] = fold_model.predict_proba(X_mat[val_idx])[:, 1]
        else:
            # Simple single-split fallback if scikit-learn CV is unavailable
            split_idx = int(n_samples * 0.8)
            fold_model = self._init_estimator()
            fold_model.fit(X_mat[:split_idx], y_vec[:split_idx])
            oof_probs[split_idx:] = fold_model.predict_proba(X_mat[split_idx:])[:, 1]
            oof_probs[:split_idx] = fold_model.predict_proba(X_mat[:split_idx])[:, 1]

        # Optimize threshold for F_beta (default F0.5)
        opt_thresh, best_score, cv_metrics = EvaluationMetrics.find_optimal_threshold(
            y_true=y_vec,
            y_probs=oof_probs,
            beta=self.config.f_beta,
            steps=self.config.threshold_search_steps,
        )
        self.optimal_threshold = opt_thresh

        # Fit final model on all available training data
        self.model = self._init_estimator()
        self.model.fit(X_mat, y_vec)

        self.training_metadata = {
            "model_type": self.config.model_type,
            "optimal_threshold": self.optimal_threshold,
            "f_beta_target": self.config.f_beta,
            "cv_metrics": cv_metrics,
            "sample_count": n_samples,
            "feature_count": len(self.feature_names),
        }

        return self.training_metadata

    def save_model(self, file_path: Union[str, Path]):
        """Saves trained model, threshold, and feature metadata bundle."""
        if self.model is None:
            raise ValueError("Cannot save an unfitted model. Train the model first.")
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        bundle = {
            "model": self.model,
            "feature_names": self.feature_names,
            "optimal_threshold": self.optimal_threshold,
            "training_metadata": self.training_metadata,
        }
        if HAS_JOBLIB:
            joblib.dump(bundle, path)
        else:
            with open(path, "wb") as f:
                pickle.dump(bundle, f)

    @classmethod
    def load_model(cls, file_path: Union[str, Path]) -> "ModelTrainer":
        """Loads a saved model bundle."""
        path = Path(file_path)
        if not path.is_file():
            raise FileNotFoundError(f"Model file not found at: {path.resolve()}")
        if HAS_JOBLIB:
            bundle = joblib.load(path)
        else:
            with open(path, "rb") as f:
                bundle = pickle.load(f)
        instance = cls()
        instance.model = bundle["model"]
        instance.feature_names = bundle["feature_names"]
        instance.optimal_threshold = bundle["optimal_threshold"]
        instance.training_metadata = bundle.get("training_metadata", {})
        return instance
