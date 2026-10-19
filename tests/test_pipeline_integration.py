"""Integration tests for train/predict schema and preprocessing consistency."""

import sys
import unittest
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.pipeline.ingestion import SchemaValidator, canonicalize_pairwise_dataframe
from backend.pipeline.preprocessing import EntityPreprocessor
from backend.pipeline.feature_engineering import PairwiseFeatureExtractor


class TestPairwisePipelineIntegration(unittest.TestCase):
    def test_arbitrary_pairwise_headers_are_canonicalized(self):
        df = pd.DataFrame([
            {
                "record_id_left": "L1",
                "record_id_right": "R1",
                "company_left": "ACME Corporation",
                "company_right": "Acme Inc.",
                "street_left": "100 North Main Street",
                "street_right": "100 N Main St.",
                "label": "1",
            }
        ])
        schema = SchemaValidator().inspect_headers(list(df.columns))
        canonical = canonicalize_pairwise_dataframe(df, schema)

        self.assertEqual(canonical.loc[0, "id_1"], "L1")
        self.assertEqual(canonical.loc[0, "id_2"], "R1")
        self.assertEqual(canonical.loc[0, "name_1"], "ACME Corporation")
        self.assertEqual(canonical.loc[0, "name_2"], "Acme Inc.")
        self.assertEqual(canonical.loc[0, "address_1"], "100 North Main Street")
        self.assertEqual(canonical.loc[0, "label"], "1")

    def test_prediction_features_use_preprocessed_fields(self):
        df = pd.DataFrame([
            {
                "id_1": "L1", "id_2": "R1",
                "name_1": "ACME Corporation", "name_2": "Acme Inc.",
                "address_1": "100 North Main Street", "address_2": "100 N Main St.",
            }
        ])
        preprocessor = EntityPreprocessor(strip_legal_suffixes=True)
        clean = preprocessor.preprocess_dataframe(
            df,
            name_cols=["name_1", "name_2"],
            address_cols=["address_1", "address_2"],
        )
        features = PairwiseFeatureExtractor().extract_features_from_pair_df(clean)

        self.assertEqual(clean.loc[0, "name_1_clean"], "acme")
        self.assertEqual(clean.loc[0, "name_2_clean"], "acme")
        self.assertEqual(features.loc[0, "name_exact_match"], 1.0)


if __name__ == "__main__":
    unittest.main()

class TestBlockingPipelineIntegration(unittest.TestCase):
    def test_single_table_is_materialized_into_pairwise_candidates(self):
        from backend.pipeline.linkage import prepare_entity_table, build_pairwise_dataframe_from_candidates
        from backend.pipeline.blocking import EntityBlocker

        df = pd.DataFrame([
            {"id": "1", "company": "ACME Corporation", "street": "100 North Main Street"},
            {"id": "2", "company": "Acme Inc.", "street": "100 N Main St."},
            {"id": "3", "company": "Zebra Foods", "street": "900 Oak Road"},
        ])
        prepared = prepare_entity_table(df)
        pairs = EntityBlocker().generate_candidate_pairs_from_tables(
            prepared, id_col_left="id", name_col="name", address_col="address"
        )
        self.assertIn(("1", "2"), pairs)
        pair_df = build_pairwise_dataframe_from_candidates(prepared, pairs)
        self.assertIn("name_1", pair_df.columns)
        self.assertIn("name_2", pair_df.columns)
        self.assertEqual(len(pair_df), len(pairs))

    def test_single_table_without_id_gets_stable_row_ids(self):
        from backend.pipeline.linkage import prepare_entity_table
        df = pd.DataFrame([
            {"name": "Alpha Co", "address": "10 Main St"},
            {"name": "Alpha Company", "address": "10 Main Street"},
        ])
        prepared = prepare_entity_table(df)
        self.assertEqual(prepared["id"].tolist(), ["0", "1"])


if __name__ == "__main__":
    unittest.main()
