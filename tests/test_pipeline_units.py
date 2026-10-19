"""Unit tests for EntityMatch AI modular pipeline components."""

import sys
import unittest
from pathlib import Path
import numpy as np
import pandas as pd

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
    n_gram_jaccard,
)
from backend.pipeline.evaluation import EvaluationMetrics


class TestPreprocessing(unittest.TestCase):
    """Verifies entity normalization logic."""

    def setUp(self):
        self.preprocessor = EntityPreprocessor(strip_legal_suffixes=True)

    def test_company_legal_suffix_stripping(self):
        cleaned1 = self.preprocessor.clean_company_name("Acme Corporation")
        cleaned2 = self.preprocessor.clean_company_name("Acme Inc.")
        cleaned3 = self.preprocessor.clean_company_name("Acme LLC")
        self.assertEqual(cleaned1, "acme")
        self.assertEqual(cleaned2, "acme")
        self.assertEqual(cleaned3, "acme")

    def test_address_abbreviations(self):
        addr = self.preprocessor.clean_address("100 North Main Street, Suite 400")
        self.assertIn("100 n main st", addr)
        self.assertIn("ste 400", addr)

    def test_phone_normalization(self):
        phone1 = self.preprocessor.clean_phone("+1 (800) 555-0199")
        phone2 = self.preprocessor.clean_phone("800-555-0199")
        self.assertEqual(phone1, "8005550199")
        self.assertEqual(phone2, "8005550199")

    def test_website_canonicalization(self):
        url1 = self.preprocessor.clean_website("https://www.acme-tech.com/contact?src=web#top")
        url2 = self.preprocessor.clean_website("http://acme-tech.com/")
        self.assertEqual(url1, "acme-tech.com")
        self.assertEqual(url2, "acme-tech.com")

    def test_postal_code_cleaning(self):
        zip_us = self.preprocessor.clean_postal_code("90210-1234")
        self.assertEqual(zip_us, "90210")


class TestFeatureEngineering(unittest.TestCase):
    """Verifies pairwise similarity calculation algorithms."""

    def test_jaro_winkler_metric(self):
        # Identical strings
        self.assertAlmostEqual(pure_jaro_winkler("acme", "acme"), 1.0)
        # Empty string
        self.assertAlmostEqual(pure_jaro_winkler("", "acme"), 0.0)
        # Close strings
        score = pure_jaro_winkler("martha", "marhta")
        self.assertGreater(score, 0.90)

    def test_levenshtein_ratio(self):
        self.assertAlmostEqual(pure_levenshtein_ratio("acme corp", "acme corp"), 1.0)
        self.assertAlmostEqual(pure_levenshtein_ratio("abcd", "wxyz"), 0.0)

    def test_token_sort_ratio(self):
        # Inverted token order should yield 1.0 similarity
        score = token_sort_ratio("Google Alphabet", "Alphabet Google")
        self.assertAlmostEqual(score, 1.0, places=2)

    def test_ngram_jaccard(self):
        score = n_gram_jaccard("apple", "apples", n=3)
        self.assertGreater(score, 0.5)

    def test_pair_feature_vector_structure(self):
        extractor = PairwiseFeatureExtractor()
        rec_l = {
            "name_clean": "acme technologies",
            "address_clean": "100 market st",
            "city": "San Francisco",
            "state": "CA",
            "postal_code": "94105",
            "phone_clean": "4155551234",
            "website_clean": "acme.com",
        }
        rec_r = {
            "name_clean": "acme tech",
            "address_clean": "100 market street",
            "city": "San Francisco",
            "state": "ca",
            "postal_code": "94105",
            "phone_clean": "4155551234",
            "website_clean": "acme.com",
        }
        feats = extractor.extract_pair_features(rec_l, rec_r)
        self.assertIn("name_jaro_winkler", feats)
        self.assertIn("phone_exact_match", feats)
        self.assertEqual(feats["phone_exact_match"], 1.0)
        self.assertEqual(feats["city_exact_match"], 1.0)
        self.assertGreater(feats["name_jaro_winkler"], 0.8)


class TestBlocking(unittest.TestCase):
    """Verifies candidate pair generation reducing quadratic complexity."""

    def test_blocking_generates_candidates(self):
        df = pd.DataFrame([
            {"id": "1", "name": "Apple Inc",           "name_clean": "apple inc",           "postal_code": "95014"},
            {"id": "2", "name": "Apple Computer",      "name_clean": "apple computer",      "postal_code": "95014"},
            {"id": "3", "name": "Microsoft Corp",      "name_clean": "microsoft corp",      "postal_code": "98052"},
            {"id": "4", "name": "Micro-soft Corporation", "name_clean": "microsoft corporation", "postal_code": "98052"},
        ])
        blocker = EntityBlocker()
        pairs = blocker.generate_candidate_pairs_from_tables(
            df, id_col_left="id", name_col="name", postal_col="postal_code"
        )
        self.assertIn(("1", "2"), pairs)
        self.assertIn(("3", "4"), pairs)
        # Apple and Microsoft share no tokens and different postal codes
        self.assertNotIn(("1", "3"), pairs)


class TestEvaluationMetrics(unittest.TestCase):
    """Verifies F0.5 mathematical properties and threshold sweep."""

    def test_f05_mathematical_precision_weighting(self):
        # F0.5 with Precision=1.0, Recall=0.5
        # F0.5 = 1.25 * (1.0 * 0.5) / (0.25 * 1.0 + 0.5) = 0.625 / 0.75 = 0.833333
        f05 = EvaluationMetrics.calculate_f_beta(precision=1.0, recall=0.5, beta=0.5)
        self.assertAlmostEqual(f05, 5 / 6, places=4)

        # F1 with same P & R: 2 * (1.0 * 0.5) / (1.5) = 1.0 / 1.5 = 0.666667
        f1 = EvaluationMetrics.calculate_f_beta(precision=1.0, recall=0.5, beta=1.0)
        self.assertAlmostEqual(f1, 2 / 3, places=4)

        # F0.5 must be strictly greater than F1 when Precision > Recall
        self.assertGreater(f05, f1)

    def test_threshold_optimization(self):
        y_true = np.array([0, 0, 0, 1, 1, 1, 1])
        y_probs = np.array([0.1, 0.2, 0.45, 0.6, 0.7, 0.85, 0.9])

        opt_th, best_f, metrics = EvaluationMetrics.find_optimal_threshold(
            y_true, y_probs, beta=0.5
        )
        self.assertGreaterEqual(opt_th, 0.45)
        self.assertLessEqual(opt_th, 0.65)
        self.assertGreater(best_f, 0.8)


if __name__ == "__main__":
    unittest.main()
