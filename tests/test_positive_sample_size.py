"""Focused tests for the positive_sample_size parameter of
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


def _write_source(path, rows):
    lines = ["entity_id\tbusiness_name\tbusiness_address\tcountry"]
    for r in rows:
        lines.append("\t".join(r))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_gt(path, rows):
    path.write_text("\n".join("\t".join(r) for r in rows) + "\n", encoding="utf-8")


class TestPositiveSampleSize(unittest.TestCase):

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        s1 = self.tmp / "s1.tsv"
        s2 = self.tmp / "s2.tsv"
        self.gt_path = self.tmp / "gt.tsv"

        # 10 entities per source, all with matching names (guaranteed blocking hits)
        _write_source(s1, [
            (f"S1-{i:03d}", f"Alpha Company {i}", f"{i*10} Main St", "US")
            for i in range(10)
        ])
        _write_source(s2, [
            (f"S2-{i:03d}", f"Alpha Corp {i}", f"{i*10} Main St", "US")
            for i in range(10)
        ])
        self.combined = load_amazon_sources(
            [s1, s2], source_labels=["source1", "source2"]
        )
        # 10 GT positive pairs
        _write_gt(self.gt_path, [
            (f"S1-{i:03d}", f"S2-{i:03d}") for i in range(10)
        ])
        self.gt = load_amazon_ground_truth(self.gt_path)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    # ------------------------------------------------------------------
    # Default: positive_sample_size=None uses all GT rows
    # ------------------------------------------------------------------
    def test_default_none_uses_all_gt_pairs(self):
        result = build_amazon_training_dataframe(
            self.combined, self.gt, negative_ratio=0
        )
        n_pos = (result["label"] == 1).sum()
        # All 10 positives should be present (some may be outside blocking candidates
        # but the count is bounded by len(gt))
        self.assertGreater(n_pos, 0)
        self.assertLessEqual(n_pos, len(self.gt))

    # ------------------------------------------------------------------
    # Sampling reduces the number of positives used
    # ------------------------------------------------------------------
    def test_sample_size_limits_positives(self):
        result_full = build_amazon_training_dataframe(
            self.combined, self.gt, negative_ratio=0
        )
        result_sampled = build_amazon_training_dataframe(
            self.combined, self.gt, negative_ratio=0,
            positive_sample_size=3,
        )
        n_pos_full = (result_full["label"] == 1).sum()
        n_pos_sampled = (result_sampled["label"] == 1).sum()
        self.assertLessEqual(n_pos_sampled, n_pos_full)
        self.assertLessEqual(n_pos_sampled, 3)

    # ------------------------------------------------------------------
    # sample_size >= len(gt) is a no-op
    # ------------------------------------------------------------------
    def test_sample_size_larger_than_gt_is_noop(self):
        result_full = build_amazon_training_dataframe(
            self.combined, self.gt, negative_ratio=0
        )
        result_noop = build_amazon_training_dataframe(
            self.combined, self.gt, negative_ratio=0,
            positive_sample_size=1000,  # larger than the 10-row GT
        )
        self.assertEqual(
            (result_full["label"] == 1).sum(),
            (result_noop["label"] == 1).sum(),
        )

    # ------------------------------------------------------------------
    # Reproducibility: same random_state gives same rows
    # ------------------------------------------------------------------
    def test_same_random_state_is_reproducible(self):
        r1 = build_amazon_training_dataframe(
            self.combined, self.gt, negative_ratio=0,
            positive_sample_size=4, random_state=7,
        )
        r2 = build_amazon_training_dataframe(
            self.combined, self.gt, negative_ratio=0,
            positive_sample_size=4, random_state=7,
        )
        pos1 = set(zip(r1[r1["label"]==1]["id_1"], r1[r1["label"]==1]["id_2"]))
        pos2 = set(zip(r2[r2["label"]==1]["id_1"], r2[r2["label"]==1]["id_2"]))
        self.assertEqual(pos1, pos2)

    # ------------------------------------------------------------------
    # Different random_state may give different rows
    # ------------------------------------------------------------------
    def test_different_random_states_differ(self):
        results = []
        for seed in range(5):
            r = build_amazon_training_dataframe(
                self.combined, self.gt, negative_ratio=0,
                positive_sample_size=3, random_state=seed,
            )
            pos = frozenset(zip(r[r["label"]==1]["id_1"], r[r["label"]==1]["id_2"]))
            results.append(pos)
        # At least two runs should differ (with 10 GT pairs sampled to 3)
        self.assertGreater(len(set(results)), 1)

    # ------------------------------------------------------------------
    # negative_ratio still respected after sampling
    # ------------------------------------------------------------------
    def test_negative_ratio_respected_after_sampling(self):
        result = build_amazon_training_dataframe(
            self.combined, self.gt,
            negative_ratio=2,
            positive_sample_size=3,
        )
        n_pos = (result["label"] == 1).sum()
        n_neg = (result["label"] == 0).sum()
        self.assertLessEqual(n_neg, 2 * n_pos)

    # ------------------------------------------------------------------
    # Output schema unchanged
    # ------------------------------------------------------------------
    def test_output_schema_unchanged_with_sampling(self):
        result = build_amazon_training_dataframe(
            self.combined, self.gt,
            negative_ratio=1,
            positive_sample_size=3,
        )
        for col in ["id_1", "id_2", "name_1", "name_2", "address_1", "address_2", "label"]:
            self.assertIn(col, result.columns, f"Missing column: {col}")

    # ------------------------------------------------------------------
    # sample_size=1 edge case
    # ------------------------------------------------------------------
    def test_sample_size_one(self):
        result = build_amazon_training_dataframe(
            self.combined, self.gt, negative_ratio=0,
            positive_sample_size=1,
        )
        n_pos = (result["label"] == 1).sum()
        self.assertLessEqual(n_pos, 1)


if __name__ == "__main__":
    unittest.main()
