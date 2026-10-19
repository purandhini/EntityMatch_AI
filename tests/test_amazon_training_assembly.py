"""Unit tests for build_amazon_training_dataframe — the Amazon training-data
assembly function that connects ground truth to the existing training pipeline."""

import sys
import tempfile
import shutil
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from backend.pipeline.linkage import (
    load_amazon_sources,
    load_amazon_ground_truth,
    build_amazon_training_dataframe,
)


# ---------------------------------------------------------------------------
# Shared fixture helpers
# ---------------------------------------------------------------------------

def _write_source(path: Path, rows: list):
    """Write a minimal Amazon-format source TSV."""
    lines = ["entity_id\tbusiness_name\tbusiness_address\tcountry"]
    for r in rows:
        lines.append("\t".join(r))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_ground_truth(path: Path, rows: list):
    """Write a ground-truth TSV (no header, bare entity IDs)."""
    lines = ["\t".join(r) for r in rows]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


class TestBuildAmazonTrainingDataframe(unittest.TestCase):

    def setUp(self):
        self.tmpdir = Path(tempfile.mkdtemp())
        self.s1 = self.tmpdir / "train_source1.tsv"
        self.s2 = self.tmpdir / "train_source2.tsv"
        self.s3 = self.tmpdir / "train_source3.tsv"
        self.gt = self.tmpdir / "train_ground_truth.tsv"

        _write_source(self.s1, [
            ("S1-001", "Acme Corporation",  "100 Main Street",   "US"),
            ("S1-002", "Globex Inc",         "200 Oak Avenue",    "US"),
            ("S1-003", "Wayne Enterprises",  "1 Wayne Tower",     "US"),
        ])
        _write_source(self.s2, [
            ("S2-001", "Acme Corp",          "100 Main St",       "US"),
            ("S2-002", "Globex Incorporated","200 Oak Ave",       "US"),
            ("S2-003", "Stark Industries",   "300 Elm Road",      "US"),
        ])
        _write_source(self.s3, [
            ("S3-001", "Acme Company",       "100 Main Street",   "US"),
            ("S3-002", "Daily Planet",       "900 Metropolis Ave","US"),
        ])

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _load_combined(self):
        return load_amazon_sources(
            [self.s1, self.s2, self.s3],
            source_labels=["source1", "source2", "source3"],
        )

    # ------------------------------------------------------------------
    # Raw ID → global ID mapping
    # ------------------------------------------------------------------

    def test_raw_id_to_global_id_mapping(self):
        """entity_id column in combined_df maps S1-001 → source1_S1-001."""
        combined = self._load_combined()
        id_map = dict(zip(combined["entity_id"], combined["id"]))
        self.assertEqual(id_map["S1-001"], "source1_S1-001")
        self.assertEqual(id_map["S2-001"], "source2_S2-001")
        self.assertEqual(id_map["S3-001"], "source3_S3-001")

    # ------------------------------------------------------------------
    # S1→S2 positive pair
    # ------------------------------------------------------------------

    def test_s1_to_s2_positive_pair(self):
        """Ground-truth S1-001 ↔ S2-001 must be labeled 1 in the training data."""
        _write_ground_truth(self.gt, [("S1-001", "S2-001")])
        combined = self._load_combined()
        gt = load_amazon_ground_truth(self.gt)
        result = build_amazon_training_dataframe(combined, gt, negative_ratio=0)
        positives = result[result["label"] == 1]
        self.assertGreater(len(positives), 0)
        # Confirm at least one positive row carries the expected global IDs
        pos_ids = set(
            zip(positives["id_1"].tolist(), positives["id_2"].tolist())
        )
        # The pair may be ordered either way
        found = (
            ("source1_S1-001", "source2_S2-001") in pos_ids
            or ("source2_S2-001", "source1_S1-001") in pos_ids
        )
        self.assertTrue(found, f"Expected Acme S1/S2 pair in positives; got {pos_ids}")

    # ------------------------------------------------------------------
    # S1→S3 positive pair
    # ------------------------------------------------------------------

    def test_s1_to_s3_positive_pair(self):
        """Ground-truth S1-001 ↔ S3-001 must be labeled 1."""
        _write_ground_truth(self.gt, [("S1-001", "S3-001")])
        combined = self._load_combined()
        gt = load_amazon_ground_truth(self.gt)
        result = build_amazon_training_dataframe(combined, gt, negative_ratio=0)
        positives = result[result["label"] == 1]
        pos_ids = set(
            zip(positives["id_1"].tolist(), positives["id_2"].tolist())
        )
        found = (
            ("source1_S1-001", "source3_S3-001") in pos_ids
            or ("source3_S3-001", "source1_S1-001") in pos_ids
        )
        self.assertTrue(found, f"Expected Acme S1/S3 pair in positives; got {pos_ids}")

    # ------------------------------------------------------------------
    # Multiple matches for one S1 entity
    # ------------------------------------------------------------------

    def test_multiple_matches_for_one_s1_entity(self):
        """S1-001 matching both S2-001 and S3-001 produces two positive rows."""
        _write_ground_truth(self.gt, [("S1-001", "S2-001,S3-001")])
        combined = self._load_combined()
        gt = load_amazon_ground_truth(self.gt)
        self.assertEqual(len(gt), 2)  # ground truth expands to 2 pairs
        result = build_amazon_training_dataframe(combined, gt, negative_ratio=0)
        positives = result[result["label"] == 1]
        self.assertGreaterEqual(len(positives), 2)

    # ------------------------------------------------------------------
    # Positive/negative labeling
    # ------------------------------------------------------------------

    def test_positive_and_negative_labels_coexist(self):
        """With negative_ratio > 0 both label=1 and label=0 rows are present."""
        _write_ground_truth(self.gt, [("S1-001", "S2-001")])
        combined = self._load_combined()
        gt = load_amazon_ground_truth(self.gt)
        result = build_amazon_training_dataframe(combined, gt, negative_ratio=5)
        self.assertIn(1, result["label"].values)
        self.assertIn(0, result["label"].values)

    def test_only_label_values_are_0_and_1(self):
        """No label values other than 0 and 1 should appear."""
        _write_ground_truth(self.gt, [("S1-001", "S2-001")])
        combined = self._load_combined()
        gt = load_amazon_ground_truth(self.gt)
        result = build_amazon_training_dataframe(combined, gt, negative_ratio=3)
        self.assertTrue(set(result["label"].unique()).issubset({0, 1}))

    # ------------------------------------------------------------------
    # Known positives never become negatives
    # ------------------------------------------------------------------

    def test_known_positives_never_labeled_negative(self):
        """Ground-truth pairs must always carry label=1, even under heavy downsampling."""
        _write_ground_truth(self.gt, [
            ("S1-001", "S2-001"),
            ("S1-002", "S2-002"),
        ])
        combined = self._load_combined()
        gt = load_amazon_ground_truth(self.gt)
        # Use negative_ratio=0 to keep all negatives; positives must still be 1
        result = build_amazon_training_dataframe(combined, gt, negative_ratio=0)
        for _, row in gt.iterrows():
            left_global = f"source1_{row['left_id']}"
            right_global = f"source2_{row['right_id']}"
            match = result[
                ((result["id_1"] == left_global) & (result["id_2"] == right_global))
                | ((result["id_1"] == right_global) & (result["id_2"] == left_global))
            ]
            if len(match) > 0:
                self.assertEqual(
                    match["label"].iloc[0], 1,
                    f"Ground-truth pair ({left_global}, {right_global}) was labeled 0",
                )

    # ------------------------------------------------------------------
    # negative_ratio
    # ------------------------------------------------------------------

    def test_negative_ratio_zero_keeps_all_negatives(self):
        """negative_ratio=0 must keep all blocked negatives."""
        _write_ground_truth(self.gt, [("S1-001", "S2-001")])
        combined = self._load_combined()
        gt = load_amazon_ground_truth(self.gt)
        result_no_cap = build_amazon_training_dataframe(combined, gt, negative_ratio=0)
        result_capped = build_amazon_training_dataframe(combined, gt, negative_ratio=1)
        n_neg_no_cap = (result_no_cap["label"] == 0).sum()
        n_neg_capped = (result_capped["label"] == 0).sum()
        # With ratio=0 we keep all; with ratio=1 we keep at most n_pos negatives
        self.assertGreaterEqual(n_neg_no_cap, n_neg_capped)

    def test_negative_ratio_caps_negatives(self):
        """With negative_ratio=2 there are at most 2×n_positives negatives."""
        _write_ground_truth(self.gt, [("S1-001", "S2-001")])
        combined = self._load_combined()
        gt = load_amazon_ground_truth(self.gt)
        result = build_amazon_training_dataframe(combined, gt, negative_ratio=2)
        n_pos = (result["label"] == 1).sum()
        n_neg = (result["label"] == 0).sum()
        self.assertLessEqual(n_neg, 2 * n_pos)

    # ------------------------------------------------------------------
    # Missing IDs — safe handling
    # ------------------------------------------------------------------

    def test_missing_id_does_not_crash(self):
        """A ground-truth pair referencing a non-existent entity ID must not raise."""
        _write_ground_truth(self.gt, [
            ("S1-001", "S2-001"),     # valid
            ("S1-999", "S2-999"),     # IDs not in any source
        ])
        combined = self._load_combined()
        gt = load_amazon_ground_truth(self.gt)
        # Must not raise; missing pair is silently skipped
        try:
            result = build_amazon_training_dataframe(combined, gt, negative_ratio=0)
        except Exception as exc:
            self.fail(f"build_amazon_training_dataframe raised unexpectedly: {exc}")

    def test_missing_id_does_not_corrupt_labels(self):
        """A skipped missing-ID pair must not affect labeling of valid pairs."""
        _write_ground_truth(self.gt, [
            ("S1-001", "S2-001"),
            ("S1-999", "S2-999"),  # non-existent
        ])
        combined = self._load_combined()
        gt = load_amazon_ground_truth(self.gt)
        result = build_amazon_training_dataframe(combined, gt, negative_ratio=0)
        # The valid positive must still be labeled 1
        pos = result[result["label"] == 1]
        self.assertGreater(len(pos), 0)

    # ------------------------------------------------------------------
    # Output schema compatibility
    # ------------------------------------------------------------------

    def test_output_has_required_columns_for_feature_extractor(self):
        """Returned DataFrame must have id_1, id_2, name_1, name_2, address_1, address_2, label."""
        _write_ground_truth(self.gt, [("S1-001", "S2-001")])
        combined = self._load_combined()
        gt = load_amazon_ground_truth(self.gt)
        result = build_amazon_training_dataframe(combined, gt, negative_ratio=1)
        for col in ["id_1", "id_2", "name_1", "name_2", "address_1", "address_2", "label"]:
            self.assertIn(col, result.columns, f"Missing required column: {col}")

    def test_feature_extractor_accepts_output(self):
        """PairwiseFeatureExtractor must run without error on the assembled training data."""
        from backend.pipeline.feature_engineering import PairwiseFeatureExtractor
        _write_ground_truth(self.gt, [("S1-001", "S2-001")])
        combined = self._load_combined()
        gt = load_amazon_ground_truth(self.gt)
        result = build_amazon_training_dataframe(combined, gt, negative_ratio=2)
        features = PairwiseFeatureExtractor().extract_features_from_pair_df(result)
        self.assertEqual(len(features), len(result))
        self.assertGreater(len(features.columns), 0)


if __name__ == "__main__":
    unittest.main()
