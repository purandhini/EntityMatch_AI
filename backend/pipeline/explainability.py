"""Explainability module for feature importance and local pair match interpretations."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import numpy as np
import pandas as pd

from .train import ModelTrainer

# Optional SHAP integration
try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False


class MatchExplainer:
    """
    Provides global feature importance and pair-level local explanations
    for entity resolution match decisions.
    """

    def __init__(self, trainer: ModelTrainer):
        self.trainer = trainer
        self.model = trainer.model
        self.feature_names = trainer.feature_names
        self._shap_explainer = None

    def get_global_feature_importance(self) -> pd.DataFrame:
        """
        Returns ranked global feature importances from the trained model.
        """
        if self.model is None:
            raise ValueError("Model is not fitted. Cannot extract feature importances.")

        importances = None
        if hasattr(self.model, "feature_importances_"):
            importances = self.model.feature_importances_
        elif hasattr(self.model, "coef_"):
            importances = np.abs(self.model.coef_[0])
        else:
            importances = np.ones(len(self.feature_names))

        df_imp = pd.DataFrame({
            "feature": self.feature_names,
            "importance": importances,
        }).sort_values(by="importance", ascending=False).reset_index(drop=True)

        # Normalize to percentage
        total = df_imp["importance"].sum()
        df_imp["relative_weight_%"] = (
            (df_imp["importance"] / total * 100).round(2) if total > 0 else 0.0
        )
        return df_imp

    def explain_pair(
        self,
        features_dict: Dict[str, float],
        top_k: int = 5,
    ) -> Dict[str, Any]:
        """
        Generates a local explanation for a specific entity pair.

        Args:
            features_dict: Dictionary of feature name -> computed similarity score.
            top_k: Number of most impactful features to highlight.

        Returns:
            Dictionary containing feature values, contribution direction, and text rationale.
        """
        if self.model is None:
            raise ValueError("Model is not fitted. Cannot explain predictions.")

        feat_vector = [features_dict.get(fn, 0.0) for fn in self.feature_names]
        X_sample = np.array([feat_vector], dtype=np.float32)

        # 1. SHAP-based local attribution if available
        if HAS_SHAP:
            try:
                if self._shap_explainer is None:
                    self._shap_explainer = shap.Explainer(self.model)
                shap_values = self._shap_explainer(X_sample)
                vals = shap_values.values[0]
                if len(vals.shape) > 1:
                    vals = vals[:, 1]  # positive class
                contributions = dict(zip(self.feature_names, vals.tolist()))
            except Exception:
                contributions = self._heuristic_contributions(feat_vector)
        else:
            contributions = self._heuristic_contributions(feat_vector)

        # Sort top positive and negative drivers
        sorted_contrib = sorted(
            contributions.items(), key=lambda x: abs(x[1]), reverse=True
        )[:top_k]

        top_drivers = []
        for feat, weight in sorted_contrib:
            raw_val = features_dict.get(feat, 0.0)
            effect = "supports match" if weight > 0 else "opposes match"
            top_drivers.append({
                "feature": feat,
                "value": round(raw_val, 4),
                "impact": round(weight, 4),
                "direction": effect,
            })

        # Generate human-readable narrative
        narrative = self._generate_narrative(features_dict, top_drivers)

        return {
            "top_drivers": top_drivers,
            "explanation_summary": narrative,
        }

    def _heuristic_contributions(self, feat_vector: List[float]) -> Dict[str, float]:
        """Calculates surrogate feature impact when SHAP is unavailable."""
        global_imp = self.get_global_feature_importance().set_index("feature")["importance"].to_dict()
        contributions = {}
        for fn, val in zip(self.feature_names, feat_vector):
            weight = global_imp.get(fn, 1.0)
            # Center similarity around neutral threshold 0.5
            impact = (val - 0.5) * weight
            contributions[fn] = float(impact)
        return contributions

    def _generate_narrative(
        self,
        features: Dict[str, float],
        top_drivers: List[Dict[str, Any]],
    ) -> str:
        """Constructs an interpretable sentence summarizing the pair match factors."""
        name_sim = features.get("name_jaro_winkler", features.get("name_levenshtein", 0.0))
        phone_match = features.get("phone_exact_match", 0.0)
        web_match = features.get("website_exact_match", 0.0)
        zip_match = features.get("zip_exact_match", 0.0)

        clauses = []
        if name_sim >= 0.85:
            clauses.append(f"strong company name resemblance ({name_sim:.2f} Jaro-Winkler)")
        elif name_sim < 0.5:
            clauses.append(f"low name similarity ({name_sim:.2f})")

        if phone_match == 1.0:
            clauses.append("identical telephone numbers")
        if web_match == 1.0:
            clauses.append("matching website domains")
        if zip_match == 1.0:
            clauses.append("identical postal codes")

        if not clauses:
            clauses.append(f"top factor: {top_drivers[0]['feature']} ({top_drivers[0]['direction']})")

        return "Match decision influenced by " + ", and ".join(clauses) + "."

    def export_importance_report(self, output_path: Union[str, Path]):
        """Exports global feature importance table to CSV / TSV."""
        df_imp = self.get_global_feature_importance()
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        df_imp.to_csv(path, sep="\t", index=False)
        print(f"Feature importance report saved to: {path.resolve()}")
