"""Unit tests verifying pipeline strengthening, caching, 29-feature interface, and 3-source flow."""

import sys
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.pipeline.preprocessing import EntityPreprocessor
from backend.pipeline.blocking import EntityBlocker
from backend.pipeline.feature_engineering import (
    PairwiseFeatureExtractor,
    pure_jaro_winkler,
    pure_levenshtein_ratio,
    token_sort_ratio,
    token_set_ratio,
    n_gram_jaccard,
    HAS_RAPIDFUZZ,
)
from backend.pipeline.predict import EntityPredictor
import run_test_prediction as rtp


class TestModelAndFeatureIntegrity(unittest.TestCase):
    """Verifies the exact 29-feature interface and model artifacts."""

    def setUp(self):
        self.model_path = PROJECT_ROOT / "models" / "entitymatch_model.joblib"
        self.assertTrue(self.model_path.exists(), f"Model file missing at {self.model_path}")
        self.model_data = joblib.load(self.model_path)
        self.extractor = PairwiseFeatureExtractor()

    def test_model_artifact_metadata(self):
        self.assertIn("model", self.model_data)
        self.assertIn("feature_names", self.model_data)
        self.assertIn("optimal_threshold", self.model_data)
        self.assertEqual(len(self.model_data["feature_names"]), 29)
        self.assertAlmostEqual(self.model_data["optimal_threshold"], 0.8712, places=4)

    def test_feature_extractor_returns_exact_29_features_in_order(self):
        rec_left = {
            "name": "Target Store",
            "name_clean": "target store",
            "address": "1000 Nicollet Mall",
            "address_clean": "1000 nicollet mall",
            "city": "Minneapolis",
            "state": "MN",
            "postal_code": "55403",
            "phone": "6123046073",
            "website": "target.com",
        }
        rec_right = {
            "name": "Target Stores Inc",
            "name_clean": "target stores inc",
            "address": "1000 Nicollet Mall Ste 100",
            "address_clean": "1000 nicollet mall ste 100",
            "city": "minneapolis",
            "state": "mn",
            "postal_code": "55403",
            "phone": "6123046073",
            "website": "target.com",
        }
        feats = self.extractor.extract_pair_features(rec_left, rec_right)
        feat_keys = list(feats.keys())
        model_feats = self.model_data["feature_names"]

        self.assertEqual(len(feat_keys), 29)
        self.assertEqual(feat_keys, model_feats)

    def test_extract_features_from_pair_df_order_and_types(self):
        pair_df = pd.DataFrame([
            {
                "name_1": "Target", "name_2": "Target Inc",
                "address_1": "1000 Nicollet", "address_2": "1000 Nicollet Mall",
                "city_1": "Minneapolis", "city_2": "Minneapolis",
                "state_1": "MN", "state_2": "MN",
                "postal_code_1": "55403", "postal_code_2": "55403",
                "phone_1": "6120001111", "phone_2": "6120001111",
                "website_1": "target.com", "website_2": "target.com",
            },
            {
                "name_1": "Best Buy", "name_2": "Walmart",
                "address_1": "7601 Penn Ave S", "address_2": "702 SW 8th St",
                "city_1": "Richfield", "city_2": "Bentonville",
                "state_1": "MN", "state_2": "AR",
                "postal_code_1": "55423", "postal_code_2": "72716",
                "phone_1": "", "phone_2": "",
                "website_1": "", "website_2": "",
            }
        ])
        feat_df = self.extractor.extract_features_from_pair_df(pair_df)
        self.assertEqual(feat_df.shape, (2, 29))
        self.assertEqual(list(feat_df.columns), self.model_data["feature_names"])
        self.assertFalse(feat_df.isna().any().any(), "Features should never contain NaN values")

    def test_nan_and_empty_robustness(self):
        """Verifies that missing or NaN fields do not trigger false positive matches."""
        rec_empty = {
            "name": np.nan, "address": None, "city": "", "state": np.nan,
            "postal_code": None, "phone": np.nan, "website": None
        }
        feats = self.extractor.extract_pair_features(rec_empty, rec_empty)
        self.assertEqual(feats["phone_both_present"], 0.0)
        self.assertEqual(feats["website_both_present"], 0.0)
        self.assertEqual(feats["name_both_missing"], 1.0)
        self.assertEqual(feats["address_both_missing"], 1.0)
        self.assertEqual(feats["phone_exact_match"], 0.0)
        self.assertEqual(feats["city_exact_match"], 0.0)

    def test_rapidfuzz_fallback_equivalence(self):
        """Verifies that RapidFuzz and pure-Python fallbacks produce identical results."""
        import backend.pipeline.feature_engineering as fe
        test_pairs = [
            ("martha", "marhta"),
            ("dwayne", "duane"),
            ("dixon", "dicksonx"),
            ("apple inc", "apple corporation"),
            ("100 main st", "100 main street"),
            ("", "something"),
            ("same", "same"),
            ("abc def", "def abc"),
            ("international business machines", "business machines international"),
            ("microsoft corporation", "micro soft corp"),
        ]

        fe.HAS_RAPIDFUZZ = False
        pure_jw = [fe.pure_jaro_winkler(s1, s2) for s1, s2 in test_pairs]
        pure_lev = [fe.pure_levenshtein_ratio(s1, s2) for s1, s2 in test_pairs]
        pure_sort = [fe.token_sort_ratio(s1, s2) for s1, s2 in test_pairs]
        pure_set = [fe.token_set_ratio(s1, s2) for s1, s2 in test_pairs]

        fe.HAS_RAPIDFUZZ = True
        rf_jw = [fe.pure_jaro_winkler(s1, s2) for s1, s2 in test_pairs]
        rf_lev = [fe.pure_levenshtein_ratio(s1, s2) for s1, s2 in test_pairs]
        rf_sort = [fe.token_sort_ratio(s1, s2) for s1, s2 in test_pairs]
        rf_set = [fe.token_set_ratio(s1, s2) for s1, s2 in test_pairs]

        for i in range(len(test_pairs)):
            self.assertAlmostEqual(pure_jw[i], rf_jw[i], places=5)
            self.assertAlmostEqual(pure_lev[i], rf_lev[i], places=5)
            self.assertAlmostEqual(pure_sort[i], rf_sort[i], places=5)
            self.assertAlmostEqual(pure_set[i], rf_set[i], places=5)


class TestEntityBlockerCaching(unittest.TestCase):
    """Verifies that EntityBlocker caches left table indexing across streaming chunks."""

    def setUp(self):
        self.blocker = EntityBlocker()
        self.df_left = pd.DataFrame([
            {"id": "L1", "name": "Apple Store", "name_clean": "apple store", "address": "100 Infinite Loop", "postal_code": "95014"},
            {"id": "L2", "name": "Google HQ", "name_clean": "google hq", "address": "1600 Amphitheatre Pkwy", "postal_code": "94043"},
        ])
        self.df_right_chunk1 = pd.DataFrame([
            {"id": "R1", "name": "Apple Store Inc", "name_clean": "apple store inc", "address": "100 Infinite Loop", "postal_code": "95014"},
        ])
        self.df_right_chunk2 = pd.DataFrame([
            {"id": "R2", "name": "Google Mountain View", "name_clean": "google mountain view", "address": "1600 Amphitheatre", "postal_code": "94043"},
        ])

    def test_left_cache_reuse_and_clear(self):
        self.assertEqual(len(self.blocker._left_cache), 0)

        # First chunk streaming builds cache
        pairs1 = list(self.blocker.stream_candidate_pairs(self.df_left, self.df_right_chunk1))
        self.assertGreater(len(self.blocker._left_cache), 0)
        self.assertIn(("L1", "R1"), pairs1)

        cache_size_after_chunk1 = len(self.blocker._left_cache)

        # Second chunk streaming reuses cache without expanding key set
        pairs2 = list(self.blocker.stream_candidate_pairs(self.df_left, self.df_right_chunk2))
        self.assertIn(("L2", "R2"), pairs2)
        self.assertEqual(len(self.blocker._left_cache), cache_size_after_chunk1)

        # clear_cache resets properly
        self.blocker.clear_cache()
        self.assertEqual(len(self.blocker._left_cache), 0)


class TestThreeSourceMultiLinkage(unittest.TestCase):
    """Smoke test ensuring 3-source processing handles all 3 pairs without loss."""

    def test_make_pair_df_and_smoke_prediction(self):
        s1 = pd.DataFrame([
            {"id": "s1_1", "name": "Acme Tools", "name_clean": "acme tools", "address": "123 Main St", "address_clean": "123 main st", "city": "Denver", "state": "CO", "postal_code": "80202", "phone": "3035550100", "website": "acmetools.com"},
        ])
        s2 = pd.DataFrame([
            {"id": "s2_1", "name": "Acme Tools Co", "name_clean": "acme tools co", "address": "123 Main St Ste 4", "address_clean": "123 main st ste 4", "city": "Denver", "state": "CO", "postal_code": "80202", "phone": "3035550100", "website": "acmetools.com"},
        ])
        pair_df = rtp.make_pair_df(s1, s2)
        self.assertIn("name_1", pair_df.columns)
        self.assertIn("name_2", pair_df.columns)
        self.assertIn("name_1_clean", pair_df.columns)
        self.assertIn("name_2_clean", pair_df.columns)

        predictor = EntityPredictor(model_path=rtp.MODEL_PATH)
        feat_df = predictor.feature_extractor.extract_features_from_pair_df(pair_df)
        probs, preds = predictor.predict_features(feat_df, threshold=rtp.THRESHOLD)

        self.assertEqual(len(probs), 1)
        self.assertGreaterEqual(probs[0], 0.0)
        self.assertLessEqual(probs[0], 1.0)
        self.assertIn(preds[0], (0, 1))

    def test_full_three_source_flow_no_overwrite(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)

            s1_data = pd.DataFrame([
                {"entity_id": "1", "business_name": "Acme Corp", "business_address": "100 Main St", "city": "Denver", "state": "CO", "postal_code": "80202", "phone": "3035550100", "website": "acmecorp.com"},
            ])
            s2_data = pd.DataFrame([
                {"entity_id": "1", "business_name": "Acme Corporation", "business_address": "100 Main Street", "city": "Denver", "state": "CO", "postal_code": "80202", "phone": "3035550100", "website": "acmecorp.com"},
            ])
            s3_data = pd.DataFrame([
                {"entity_id": "1", "business_name": "Acme Corp Inc", "business_address": "100 Main St Ste 10", "city": "Denver", "state": "CO", "postal_code": "80202", "phone": "3035550100", "website": "acmecorp.com"},
            ])

            p1, p2, p3 = tmp_path / "s1.tsv", tmp_path / "s2.tsv", tmp_path / "s3.tsv"
            s1_data.to_csv(p1, sep="\t", index=False)
            s2_data.to_csv(p2, sep="\t", index=False)
            s3_data.to_csv(p3, sep="\t", index=False)

            out_pred = tmp_path / "matching_results.tsv"
            out_cand = tmp_path / "candidate_pairs.tsv"

            preprocessor = EntityPreprocessor()
            blocker = EntityBlocker()
            predictor = EntityPredictor(model_path=rtp.MODEL_PATH)

            s1 = rtp.preprocess(rtp.load_source(p1, "source1"), preprocessor)
            rtp.process_source_pair("source1", s1, "source2", p2, preprocessor, blocker, predictor, out_pred, out_cand)
            rtp.process_source_pair("source1", s1, "source3", p3, preprocessor, blocker, predictor, out_pred, out_cand)

            s2 = rtp.preprocess(rtp.load_source(p2, "source2"), preprocessor)
            rtp.process_source_pair("source2", s2, "source3", p3, preprocessor, blocker, predictor, out_pred, out_cand)

            pred_df = pd.read_csv(out_pred, sep="\t")
            cand_df = pd.read_csv(out_cand, sep="\t")

            self.assertEqual(list(pred_df.columns), ["id_1", "id_2", "match_probability", "is_match"])
            self.assertEqual(list(cand_df.columns), ["id_1", "id_2"])

            pairs = set(zip(pred_df["id_1"].str.split("_").str[0], pred_df["id_2"].str.split("_").str[0]))
            self.assertIn(("source1", "source2"), pairs)
            self.assertIn(("source1", "source3"), pairs)
            self.assertIn(("source2", "source3"), pairs)
            self.assertEqual(len(pred_df), 3)


if __name__ == "__main__":
    unittest.main()
