"""Focused tests for the reservoir-sampled negative generation in
build_amazon_training_dataframe()."""

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
from backend.config import BlockingConfig
from backend.pipeline.blocking import EntityBlocker


def _write_source(path, rows):
    lines = ["entity_id\tbusiness_name\tbusiness_address\tcountry"]
    for r in rows:
        lines.append("\t".join(r))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_gt(path, rows):
    path.write_text("\n".join("\t".join(r) for r in rows) + "\n", encoding="utf-8")


class TestReservoirNegatives(unittest.TestCase):

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        s1 = self.tmp / "s1.tsv"
        s2 = self.tmp / "s2.tsv"
        self.gt_path = self.tmp / "gt.tsv"

        # 10 entities per source with similar names → blocking generates many candidates
        _write_source(s1, [
            (f"S1-{i:03d}", f"Acme Company {i}", f"{i*10} Main St", "US")
            for i in range(10)
        ])
        _write_source(s2, [
            (f"S2-{i:03d}", f"Acme Corp {i}", f"{i*10} Main St", "US")
            for i in range(10)
        ])
        self.combined = load_amazon_sources(
            [s1, s2], source_labels=["source1", "source2"]
        )
        # Only 3 of the 10 pairs are GT positives
        _write_gt(self.gt_path, [
            ("S1-000", "S2-000"),
            ("S1-001", "S2-001"),
            ("S1-002", "S2-002"),
        ])
        self.gt = load_amazon_ground_truth(self.gt_path)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    # ------------------------------------------------------------------
    # Known positives are never labeled as negatives
    # ------------------------------------------------------------------
    def test_known_positives_never_in_negatives(self):
        result = build_amazon_training_dataframe(
            self.combined, self.gt, negative_ratio=0
        )
        negatives = result[result["label"] == 0]
        pos_ids = {
            tuple(sorted(["source1_S1-000", "source2_S2-000"])),
            tuple(sorted(["source1_S1-001", "source2_S2-001"])),
            tuple(sorted(["source1_S1-002", "source2_S2-002"])),
        }
        for _, row in negatives.iterrows():
            key = tuple(sorted([row["id_1"], row["id_2"]]))
            self.assertNotIn(key, pos_ids,
                f"GT positive pair appeared as a negative: {key}")

    # ------------------------------------------------------------------
    # negative_ratio=1 caps negatives at n_positives
    # ------------------------------------------------------------------
    def test_negative_ratio_1_caps_negatives(self):
        result = build_amazon_training_dataframe(
            self.combined, self.gt, negative_ratio=1
        )
        n_pos = (result["label"] == 1).sum()
        n_neg = (result["label"] == 0).sum()
        self.assertLessEqual(n_neg, n_pos)

    # ------------------------------------------------------------------
    # negative_ratio=0 keeps all negatives (reservoir = entire candidate set)
    # ------------------------------------------------------------------
    def test_negative_ratio_0_keeps_all_negatives(self):
        result_all = build_amazon_training_dataframe(
            self.combined, self.gt, negative_ratio=0
        )
        result_capped = build_amazon_training_dataframe(
            self.combined, self.gt, negative_ratio=1
        )
        n_neg_all   = (result_all["label"]   == 0).sum()
        n_neg_capped = (result_capped["label"] == 0).sum()
        self.assertGreaterEqual(n_neg_all, n_neg_capped)

    # ------------------------------------------------------------------
    # Reservoir target = negative_ratio × n_positives
    # ------------------------------------------------------------------
    def test_reservoir_target_size(self):
        for ratio in [1, 2, 3]:
            result = build_amazon_training_dataframe(
                self.combined, self.gt, negative_ratio=ratio
            )
            n_pos = (result["label"] == 1).sum()
            n_neg = (result["label"] == 0).sum()
            self.assertLessEqual(n_neg, ratio * n_pos,
                f"ratio={ratio}: n_neg={n_neg} > {ratio}×{n_pos}")

    # ------------------------------------------------------------------
    # Reproducibility: same random_state → same negative set
    # ------------------------------------------------------------------
    def test_same_random_state_reproducible(self):
        r1 = build_amazon_training_dataframe(
            self.combined, self.gt, negative_ratio=1, random_state=99
        )
        r2 = build_amazon_training_dataframe(
            self.combined, self.gt, negative_ratio=1, random_state=99
        )
        neg1 = frozenset(zip(
            r1[r1["label"]==0]["id_1"], r1[r1["label"]==0]["id_2"]
        ))
        neg2 = frozenset(zip(
            r2[r2["label"]==0]["id_1"], r2[r2["label"]==0]["id_2"]
        ))
        self.assertEqual(neg1, neg2)

    # ------------------------------------------------------------------
    # Different random_state can produce different negatives
    # ------------------------------------------------------------------
    def test_different_random_states_may_differ(self):
        sets = []
        for seed in range(5):
            r = build_amazon_training_dataframe(
                self.combined, self.gt, negative_ratio=2, random_state=seed
            )
            neg = frozenset(zip(
                r[r["label"]==0]["id_1"], r[r["label"]==0]["id_2"]
            ))
            sets.append(neg)
        self.assertGreater(len(set(sets)), 1)

    # ------------------------------------------------------------------
    # Output schema still has required columns
    # ------------------------------------------------------------------
    def test_output_schema_intact(self):
        result = build_amazon_training_dataframe(
            self.combined, self.gt, negative_ratio=1
        )
        for col in ["id_1", "id_2", "name_1", "name_2",
                    "address_1", "address_2", "label"]:
            self.assertIn(col, result.columns)

    # ------------------------------------------------------------------
    # Only label values 0 and 1
    # ------------------------------------------------------------------
    def test_only_binary_labels(self):
        result = build_amazon_training_dataframe(
            self.combined, self.gt, negative_ratio=2
        )
        self.assertTrue(set(result["label"].unique()).issubset({0, 1}))

    # ------------------------------------------------------------------
    # positive_sample_size still works with reservoir path
    # ------------------------------------------------------------------
    def test_positive_sample_size_with_reservoir(self):
        result = build_amazon_training_dataframe(
            self.combined, self.gt,
            negative_ratio=1,
            positive_sample_size=2,
            random_state=42,
        )
        n_pos = (result["label"] == 1).sum()
        self.assertLessEqual(n_pos, 2)

    # ------------------------------------------------------------------
    # Feature extractor accepts reservoir output
    # ------------------------------------------------------------------
    def test_feature_extractor_accepts_output(self):
        from backend.pipeline.feature_engineering import PairwiseFeatureExtractor
        result = build_amazon_training_dataframe(
            self.combined, self.gt, negative_ratio=1
        )
        features = PairwiseFeatureExtractor().extract_features_from_pair_df(result)
        self.assertEqual(len(features), len(result))
        self.assertGreater(len(features.columns), 0)


if __name__ == "__main__":
    unittest.main()
