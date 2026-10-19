"""Unit tests for estimate_blocking_recall_from_gt."""

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
    estimate_blocking_recall_from_gt,
)
from backend.config import BlockingConfig


def _write_source(path, rows):
    lines = ["entity_id\tbusiness_name\tbusiness_address\tcountry"]
    for r in rows:
        lines.append("\t".join(r))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_gt(path, rows):
    path.write_text("\n".join("\t".join(r) for r in rows) + "\n", encoding="utf-8")


class TestEstimateBlockingRecall(unittest.TestCase):

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.s1 = self.tmp / "s1.tsv"
        self.s2 = self.tmp / "s2.tsv"
        self.gt = self.tmp / "gt.tsv"

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _combined(self, s1_rows, s2_rows):
        _write_source(self.s1, s1_rows)
        _write_source(self.s2, s2_rows)
        return load_amazon_sources(
            [self.s1, self.s2], source_labels=["source1", "source2"]
        )

    # ------------------------------------------------------------------
    # Return-value structure
    # ------------------------------------------------------------------
    def test_returns_all_required_keys(self):
        combined = self._combined(
            [("S1-001", "Acme Corp", "100 Main St", "US")],
            [("S2-001", "Acme Inc",  "100 Main St", "US")],
        )
        _write_gt(self.gt, [("S1-001", "S2-001")])
        gt = load_amazon_ground_truth(self.gt)
        result = estimate_blocking_recall_from_gt(combined, gt)
        for key in [
            "total_gt_pairs", "gt_pairs_found", "gt_pairs_missed",
            "estimated_recall", "sampled", "sample_size_used",
            "gt_pairs_skipped", "strategy_hits",
        ]:
            self.assertIn(key, result, f"Missing key: {key}")

    # ------------------------------------------------------------------
    # Token-prefix: shared name tokens → found
    # ------------------------------------------------------------------
    def test_shared_name_prefix_found(self):
        combined = self._combined(
            [("S1-001", "Acme Corporation", "100 Main St", "US")],
            [("S2-001", "Acme Corp",        "200 Oak Ave", "US")],
        )
        _write_gt(self.gt, [("S1-001", "S2-001")])
        gt = load_amazon_ground_truth(self.gt)
        cfg = BlockingConfig(
            use_token_prefix_blocking=True,
            use_sorted_neighborhood_blocking=False,
            use_address_street_number_blocking=False,
            use_exact_attribute_blocking=False,
        )
        result = estimate_blocking_recall_from_gt(combined, gt, config=cfg)
        self.assertEqual(result["gt_pairs_found"], 1)
        self.assertEqual(result["estimated_recall"], 1.0)
        self.assertGreater(result["strategy_hits"]["token_prefix"], 0)

    # ------------------------------------------------------------------
    # No shared tokens → missed by prefix-only blocking
    # ------------------------------------------------------------------
    def test_no_shared_prefix_missed(self):
        combined = self._combined(
            [("S1-001", "Xenon Industries", "42 Quantum Rd", "US")],
            [("S2-001", "Yttrium Solutions","99 Plasma Ave", "DE")],
        )
        _write_gt(self.gt, [("S1-001", "S2-001")])
        gt = load_amazon_ground_truth(self.gt)
        cfg = BlockingConfig(
            use_token_prefix_blocking=True,
            use_sorted_neighborhood_blocking=False,
            use_address_street_number_blocking=False,
            use_exact_attribute_blocking=False,
        )
        result = estimate_blocking_recall_from_gt(combined, gt, config=cfg)
        self.assertEqual(result["gt_pairs_missed"], 1)
        self.assertEqual(result["estimated_recall"], 0.0)

    # ------------------------------------------------------------------
    # Street-number blocking
    # ------------------------------------------------------------------
    def test_street_number_strategy_fires(self):
        combined = self._combined(
            [("S1-001", "Alpha Corp",  "200 Main Street", "US")],
            [("S2-001", "Alpha Inc",   "200 Oak Avenue",  "US")],
        )
        _write_gt(self.gt, [("S1-001", "S2-001")])
        gt = load_amazon_ground_truth(self.gt)
        cfg = BlockingConfig(
            use_token_prefix_blocking=False,
            use_sorted_neighborhood_blocking=False,
            use_address_street_number_blocking=True,
            use_exact_attribute_blocking=False,
        )
        result = estimate_blocking_recall_from_gt(combined, gt, config=cfg)
        self.assertEqual(result["gt_pairs_found"], 1)
        self.assertGreater(result["strategy_hits"]["street_number"], 0)

    # ------------------------------------------------------------------
    # Exact-attribute (country) blocking
    # ------------------------------------------------------------------
    def test_exact_country_strategy_fires(self):
        combined = self._combined(
            [("S1-001", "Xenon Corp", "1 Nowhere Rd", "USA")],
            [("S2-001", "Yttrium Inc","2 Nowhere Ave","USA")],
        )
        _write_gt(self.gt, [("S1-001", "S2-001")])
        gt = load_amazon_ground_truth(self.gt)
        cfg = BlockingConfig(
            use_token_prefix_blocking=False,
            use_sorted_neighborhood_blocking=False,
            use_address_street_number_blocking=False,
            use_exact_attribute_blocking=True,
        )
        result = estimate_blocking_recall_from_gt(combined, gt, config=cfg)
        self.assertEqual(result["gt_pairs_found"], 1)
        self.assertGreater(result["strategy_hits"]["exact_attribute"], 0)

    # ------------------------------------------------------------------
    # Mixed: some found, some missed
    # ------------------------------------------------------------------
    def test_partial_recall(self):
        _write_source(self.s1, [
            ("S1-001", "Acme Corporation", "100 Main St", "US"),
            ("S1-002", "Xenon Industries", "42 Quantum Rd", "FR"),
        ])
        _write_source(self.s2, [
            ("S2-001", "Acme Corp",        "100 Main St", "US"),
            ("S2-002", "Yttrium Solutions","99 Plasma Ave","DE"),
        ])
        combined = load_amazon_sources(
            [self.s1, self.s2], source_labels=["source1", "source2"]
        )
        _write_gt(self.gt, [("S1-001", "S2-001"), ("S1-002", "S2-002")])
        gt = load_amazon_ground_truth(self.gt)
        cfg = BlockingConfig(
            use_token_prefix_blocking=True,
            use_sorted_neighborhood_blocking=False,
            use_address_street_number_blocking=False,
            use_exact_attribute_blocking=False,
        )
        result = estimate_blocking_recall_from_gt(combined, gt, config=cfg)
        self.assertEqual(result["total_gt_pairs"], 2)
        self.assertLess(result["estimated_recall"], 1.0)
        self.assertGreater(result["gt_pairs_missed"], 0)

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------
    def test_sample_size_limits_checked_pairs(self):
        _write_source(self.s1, [
            (f"S1-{i:03d}", f"Company {i}", f"{i} Main St", "US")
            for i in range(20)
        ])
        _write_source(self.s2, [
            (f"S2-{i:03d}", f"Company {i}", f"{i} Main St", "US")
            for i in range(20)
        ])
        combined = load_amazon_sources(
            [self.s1, self.s2], source_labels=["source1", "source2"]
        )
        gt_rows = [(f"S1-{i:03d}", f"S2-{i:03d}") for i in range(20)]
        _write_gt(self.gt, gt_rows)
        gt = load_amazon_ground_truth(self.gt)
        result = estimate_blocking_recall_from_gt(
            combined, gt, sample_size=5, random_state=42
        )
        self.assertEqual(result["sample_size_used"], 5)
        self.assertTrue(result["sampled"])

    def test_no_sampling_checks_all_pairs(self):
        combined = self._combined(
            [("S1-001", "Acme Corp", "100 Main", "US"),
             ("S1-002", "Beta Corp", "200 Oak",  "US")],
            [("S2-001", "Acme Inc",  "100 Main", "US"),
             ("S2-002", "Beta Inc",  "200 Oak",  "US")],
        )
        _write_gt(self.gt, [("S1-001", "S2-001"), ("S1-002", "S2-002")])
        gt = load_amazon_ground_truth(self.gt)
        result = estimate_blocking_recall_from_gt(combined, gt)
        self.assertFalse(result["sampled"])
        self.assertEqual(result["sample_size_used"], 2)

    # ------------------------------------------------------------------
    # Empty GT
    # ------------------------------------------------------------------
    def test_empty_gt_returns_zero(self):
        combined = self._combined(
            [("S1-001", "Acme Corp", "100 Main", "US")],
            [("S2-001", "Acme Inc",  "100 Main", "US")],
        )
        gt = pd.DataFrame(columns=["left_id", "right_id", "label"])
        result = estimate_blocking_recall_from_gt(combined, gt)
        self.assertEqual(result["total_gt_pairs"], 0)
        self.assertEqual(result["estimated_recall"], 0.0)

    # ------------------------------------------------------------------
    # Missing IDs handled gracefully
    # ------------------------------------------------------------------
    def test_missing_ids_skipped_not_crash(self):
        combined = self._combined(
            [("S1-001", "Acme Corp", "100 Main", "US")],
            [("S2-001", "Acme Inc",  "100 Main", "US")],
        )
        _write_gt(self.gt, [("S1-001", "S2-001"), ("S1-999", "S2-999")])
        gt = load_amazon_ground_truth(self.gt)
        try:
            result = estimate_blocking_recall_from_gt(combined, gt)
        except Exception as exc:
            self.fail(f"Raised unexpectedly: {exc}")
        self.assertEqual(result["gt_pairs_skipped"], 1)

    # ------------------------------------------------------------------
    # Counts internally consistent
    # ------------------------------------------------------------------
    def test_found_plus_missed_equals_total(self):
        combined = self._combined(
            [("S1-001", "Acme Corp",   "100 Main", "US"),
             ("S1-002", "Xenon Corp",  "1 Unknown","FR")],
            [("S2-001", "Acme Inc",    "100 Main", "US"),
             ("S2-002", "Yttrium Inc", "2 Unknown","DE")],
        )
        _write_gt(self.gt, [("S1-001", "S2-001"), ("S1-002", "S2-002")])
        gt = load_amazon_ground_truth(self.gt)
        cfg = BlockingConfig(
            use_token_prefix_blocking=True,
            use_sorted_neighborhood_blocking=False,
            use_address_street_number_blocking=False,
            use_exact_attribute_blocking=False,
        )
        result = estimate_blocking_recall_from_gt(combined, gt, config=cfg)
        self.assertEqual(
            result["gt_pairs_found"] + result["gt_pairs_missed"],
            result["total_gt_pairs"],
        )


if __name__ == "__main__":
    unittest.main()
