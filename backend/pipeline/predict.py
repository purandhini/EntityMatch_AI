"""Inference and batch prediction module."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

from ..config import DEFAULT_CONFIG
from .feature_engineering import PairwiseFeatureExtractor
from .preprocessing import EntityPreprocessor
from .ingestion import SchemaValidator, canonicalize_pairwise_dataframe
from .linkage import build_pairwise_dataframe_from_candidates, prepare_entity_table
from .blocking import EntityBlocker
from .train import ModelTrainer


class EntityPredictor:
    """Runs batch or single-pair inference using a trained ModelTrainer bundle."""

    def __init__(
        self,
        trainer: Optional[ModelTrainer] = None,
        model_path: Optional[Union[str, Path]] = None,
    ):
        if trainer is not None:
            self.trainer = trainer
        elif model_path is not None:
            self.trainer = ModelTrainer.load_model(model_path)
        else:
            self.trainer = None

        self.preprocessor = EntityPreprocessor()
        self.feature_extractor = PairwiseFeatureExtractor()

    def predict_features(
        self,
        X: pd.DataFrame,
        threshold: Optional[float] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Runs prediction on pre-computed feature matrix.

        Returns:
            Tuple of (probabilities, binary_predictions).
        """
        if self.trainer is None or self.trainer.model is None:
            raise ValueError("No model loaded. Please load or train a model before predicting.")

        # Ensure column order matches training feature set
        expected_cols = self.trainer.feature_names
        missing_cols = [c for c in expected_cols if c not in X.columns]
        if missing_cols:
            raise ValueError(f"Feature matrix is missing expected features: {missing_cols}")

        X_ordered = X[expected_cols].values.astype(np.float32)
        probs = self.trainer.model.predict_proba(X_ordered)[:, 1]

        active_threshold = threshold if threshold is not None else self.trainer.optimal_threshold
        preds = (probs >= active_threshold).astype(int)

        return probs, preds

    def predict_pair(
        self,
        record_left: Dict[str, Any],
        record_right: Dict[str, Any],
        threshold: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Runs inference for a single entity record pair."""
        rec_l_clean = self.preprocessor.preprocess_record(record_left)
        rec_r_clean = self.preprocessor.preprocess_record(record_right)

        features = self.feature_extractor.extract_pair_features(rec_l_clean, rec_r_clean)
        features_df = pd.DataFrame([features])

        probs, preds = self.predict_features(features_df, threshold=threshold)
        return {
            "match_probability": round(float(probs[0]), 6),
            "is_match": int(preds[0]),
            "threshold_applied": threshold or self.trainer.optimal_threshold,
        }

    def predict_pairs_df(
        self,
        pairs_df: pd.DataFrame,
        id_1_col: str = "id_1",
        id_2_col: str = "id_2",
        threshold: Optional[float] = None,
    ) -> pd.DataFrame:
        """
        Runs end-to-end inference on a DataFrame of test pairs.

        Returns:
            DataFrame with id_1, id_2, match_probability, and is_match.
        """
        # Apply the same schema detection + preprocessing used by training/evaluation.
        schema = SchemaValidator().inspect_headers(list(pairs_df.columns))
        pairs_canonical = canonicalize_pairwise_dataframe(pairs_df, schema=schema)
        pairs_clean = self.preprocessor.preprocess_dataframe(
            pairs_canonical,
            name_cols=["name_1", "name_2"],
            address_cols=["address_1", "address_2"],
            phone_cols=["phone_1", "phone_2"],
            website_cols=["website_1", "website_2"],
            postal_cols=["postal_code_1", "postal_code_2"],
        )

        # Feature extraction now sees the same canonical cleaned fields as training.
        features_df = self.feature_extractor.extract_features_from_pair_df(pairs_clean)
        probs, preds = self.predict_features(features_df, threshold=threshold)

        results = pd.DataFrame({
            id_1_col: pairs_df[id_1_col],
            id_2_col: pairs_df[id_2_col],
            "match_probability": np.round(probs, 6),
            "is_match": preds,
        })
        return results


    def generate_entity_candidates(
        self,
        entity_df: pd.DataFrame,
        blocker: Optional[EntityBlocker] = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Preprocess a single entity table, run blocking, and materialize pair rows."""
        prepared = prepare_entity_table(entity_df, preprocessor=self.preprocessor)
        active_blocker = blocker or EntityBlocker()
        # Pass country_col when the prepared table carries the column.
        country_col = "country" if "country" in prepared.columns else None
        candidates = active_blocker.generate_candidate_pairs_from_tables(
            prepared,
            id_col_left="id",
            name_col="name",
            address_col="address",
            postal_col="postal_code",
            phone_col="phone",
            website_col="website",
            country_col=country_col,
        )
        pair_df = build_pairwise_dataframe_from_candidates(prepared, candidates)
        candidate_df = pd.DataFrame(candidates, columns=["id_1", "id_2"])
        return pair_df, candidate_df

    def predict_entity_table(
        self,
        entity_df: pd.DataFrame,
        threshold: Optional[float] = None,
        blocker: Optional[EntityBlocker] = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Run blocking + ML prediction for an unlabeled single entity table."""
        pair_df, candidate_df = self.generate_entity_candidates(entity_df, blocker=blocker)
        if pair_df.empty:
            results = pd.DataFrame(columns=["id_1", "id_2", "match_probability", "is_match"])
            return results, candidate_df
        results = self.predict_pairs_df(pair_df, threshold=threshold)
        return results, candidate_df

    def save_predictions_tsv(
        self,
        results_df: pd.DataFrame,
        output_path: Union[str, Path],
    ):
        """Saves prediction DataFrame to TSV format for challenge submission."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        results_df.to_csv(path, sep="\t", index=False)
        print(f"Predictions saved to: {path.resolve()}")
