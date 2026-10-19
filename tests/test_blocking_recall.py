"""Unit tests for evaluate_blocking_recall."""

import sys
import tempfile
import shutil
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.pipeline.linkage import (
    load_amazon_sources,
    load_amazon_ground_truth,
    evaluate_blocking_recall,
)


def _write_source(path, rows):
    lines = ["entity_id\tbusiness_name\tbusiness_address\tcountry"]
    for r in rows:
        lines.append("\t".join(r))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_gt(path, rows):
    path.write_text("\n".join("\t".join(r) for r in rows) + "\n", encoding="utf-8")


class TestEvaluateBlockingRecall(unittest.TestCase):

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.s1 = self.tmp / "s1.tsv"
        self.s2 = self.tmp / "s2.tsv"
        self.s3 = self.tmp / "s3.tsv"
        self.gt = self.tmp / "gt.tsv"

        _write_source(self.s1, [
            ("S1-001", "Acme Corporation", "100 Main Street", "US"),
            ("S1-002", "Globex Inc",        "200 Oak Avenue",  "US"),
        ])
        _write_source(self.s2, [
            ("S2-001", "Acme Corp",          "100 Main St",    "US"),
            ("S2-002", "Completely Different","999 Nowhere Rd","US"),
        ])
        _write_source(self.s3, [
            ("S3-001", "Acme Company",       "100 Main Street","US"),
        ])

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _combined(self, sources=None, labels=None):
        sources = sources or [self.s1, self.s2, self.s3]
        labels  = labels  or ["source1", "source2", "source3"]
        return load_amazon_sources(sources, source_labels=labels)

    # ------------------------------------------------------------------
    # Return-value structure
    # ------------------------------------------------------------------
    def test_returns_all_required_keys(self):
        _write_gt(self.gt, [("S1-001", "S2-001")])
        result = evaluate_blocking_recall(
            self._combined(), load_amazon_ground_truth(self.gt)
        )
        for key in [
            "total_gt_pairs", "gt_pairs_in_candidates", "gt_pairs_missed",
            "blocking_recall", "total_candidates", "reduction_ratio", "gt_pairs_skipped",
        ]:
            self.assertIn(key, result, f"Missing key: {key}")

    # ------------------------------------------------------------------
    # All positives found
    # ------------------------------------------------------------------
    def test_all_positives_found_recall_is_1(self):
        """Acme S1-S2 share name tokens — blocking must find the pair."""
        _write_gt(self.gt, [("S1-001", "S2-001")])
        result = evaluate_blocking_recall(
            self._combined(), load_amazon_ground_truth(self.gt)
        )
        self.assertEqual(result["total_gt_pairs"], 1)
        self.assertEqual(result["gt_pairs_in_candidates"], 1)
        self.assertEqual(result["gt_pairs_missed"], 0)
        self.assertEqual(result["blocking_recall"], 1.0)

    # ------------------------------------------------------------------
    # S1-S3 positive pair
    # ------------------------------------------------------------------
    def test_s1_s3_pair_found(self):
        """Acme Corporation (S1) vs Acme Company (S3) must be a candidate."""
        _write_gt(self.gt, [("S1-001", "S3-001")])
        result = evaluate_blocking_recall(
            self._combined(), load_amazon_ground_truth(self.gt)
        )
        self.assertEqual(result["gt_pairs_in_candidates"], 1)
        self.assertEqual(result["blocking_recall"], 1.0)

    # ------------------------------------------------------------------
    # S1-S2 and S1-S3 together
    # ------------------------------------------------------------------
    def test_s1_s2_and_s1_s3_both_found(self):
        _write_gt(self.gt, [("S1-001", "S2-001,S3-001")])
        gt = load_amazon_ground_truth(self.gt)
        self.assertEqual(len(gt), 2)
        result = evaluate_blocking_recall(self._combined(), gt)
        self.assertEqual(result["total_gt_pairs"], 2)
        self.assertEqual(result["gt_pairs_in_candidates"], 2)
        self.assertEqual(result["blocking_recall"], 1.0)

    # ------------------------------------------------------------------
    # Some positives missed
    # ------------------------------------------------------------------
    def test_missed_pair_lowers_recall(self):
        """A pair with completely different names will be missed by blocking."""
        # S1-002 (Globex Inc) vs S2-002 (Completely Different) — no shared tokens
        _write_gt(self.gt, [
            ("S1-001", "S2-001"),   # should be found
            ("S1-002", "S2-002"),   # should be missed
        ])
        result = evaluate_blocking_recall(
            self._combined(), load_amazon_ground_truth(self.gt)
        )
        self.assertEqual(result["total_gt_pairs"], 2)
        self.assertLess(result["blocking_recall"], 1.0)
        self.assertGreater(result["gt_pairs_missed"], 0)
        self.assertEqual(
            result["gt_pairs_in_candidates"] + result["gt_pairs_missed"],
            result["total_gt_pairs"],
        )

    # ------------------------------------------------------------------
    # No ground-truth pairs
    # ------------------------------------------------------------------
    def test_empty_ground_truth_returns_zero_recall(self):
        """An empty ground-truth DataFrame (no rows) returns zero recall without error."""
        import pandas as pd
        # Build an empty ground-truth DataFrame directly — writing an empty file
        # causes pandas to raise EmptyDataError, which is expected behaviour.
        gt = pd.DataFrame(columns=["left_id", "right_id", "label"])
        result = evaluate_blocking_recall(self._combined(), gt)
        self.assertEqual(result["total_gt_pairs"], 0)
        self.assertEqual(result["blocking_recall"], 0.0)
        self.assertEqual(result["gt_pairs_missed"], 0)

    def test_missed_pair_lowers_recall(self):
        """A GT pair whose names share no tokens must be missed by prefix-only blocking."""
        import tempfile, shutil
        from backend.config import BlockingConfig
        from backend.pipeline.blocking import EntityBlocker
        tmp2 = Path(tempfile.mkdtemp())
        try:
            s1b = tmp2 / "s1b.tsv"
            s2b = tmp2 / "s2b.tsv"
            gtb = tmp2 / "gtb.tsv"
            _write_source(s1b, [
                ("S1-001", "Acme Corporation", "100 Main Street", "US"),
                ("S1-002", "Xenon Industries",  "42 Quantum Road",  "US"),
            ])
            _write_source(s2b, [
                ("S2-001", "Acme Corp",          "100 Main St",     "US"),
                ("S2-002", "Yttrium Solutions",  "99 Plasma Avenue","DE"),
            ])
            _write_gt(gtb, [
                ("S1-001", "S2-001"),  # similar names → found by prefix
                ("S1-002", "S2-002"),  # no shared tokens → missed by prefix
            ])
            combined2 = load_amazon_sources(
                [s1b, s2b], source_labels=["source1", "source2"]
            )
            gt2 = load_amazon_ground_truth(gtb)
            # Use only token-prefix blocking so SNM cannot bridge the unrelated pair
            prefix_only = EntityBlocker(BlockingConfig(
                use_token_prefix_blocking=True,
                use_sorted_neighborhood_blocking=False,
                use_address_street_number_blocking=False,
                use_exact_attribute_blocking=False,
            ))
            result = evaluate_blocking_recall(combined2, gt2, blocker=prefix_only)
            self.assertEqual(result["total_gt_pairs"], 2)
            self.assertLess(result["blocking_recall"], 1.0)
            self.assertGreater(result["gt_pairs_missed"], 0)
        finally:
            shutil.rmtree(tmp2, ignore_errors=True)

    # ------------------------------------------------------------------
    # Missing IDs are counted, not crashing
    # ------------------------------------------------------------------
    def test_missing_ids_counted_in_skipped(self):
        _write_gt(self.gt, [
            ("S1-001", "S2-001"),
            ("S1-999", "S2-999"),   # not in any source
        ])
        result = evaluate_blocking_recall(
            self._combined(), load_amazon_ground_truth(self.gt)
        )
        self.assertEqual(result["gt_pairs_skipped"], 1)

    # ------------------------------------------------------------------
    # Counts are internally consistent
    # ------------------------------------------------------------------
    def test_found_plus_missed_equals_total(self):
        _write_gt(self.gt, [("S1-001", "S2-001"), ("S1-002", "S2-002")])
        result = evaluate_blocking_recall(
            self._combined(), load_amazon_ground_truth(self.gt)
        )
        self.assertEqual(
            result["gt_pairs_in_candidates"] + result["gt_pairs_missed"],
            result["total_gt_pairs"],
        )

    def test_reduction_ratio_between_0_and_1(self):
        _write_gt(self.gt, [("S1-001", "S2-001")])
        result = evaluate_blocking_recall(
            self._combined(), load_amazon_ground_truth(self.gt)
        )
        self.assertGreaterEqual(result["reduction_ratio"], 0.0)
        self.assertLessEqual(result["reduction_ratio"], 1.0)


if __name__ == "__main__":
    unittest.main()
