"""Focused tests for EntityBlocker.stream_candidate_pairs and its use
in the reservoir-based training path."""

import sys
import tempfile
import shutil
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from backend.config import BlockingConfig
from backend.pipeline.blocking import EntityBlocker
from backend.pipeline.linkage import (
    load_amazon_sources,
    load_amazon_ground_truth,
    build_amazon_training_dataframe,
)


def make_df(*rows):
    return pd.DataFrame(list(rows))


def _write_source(path, rows):
    lines = ["entity_id\tbusiness_name\tbusiness_address\tcountry"]
    for r in rows:
        lines.append("\t".join(r))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_gt(path, rows):
    path.write_text("\n".join("\t".join(r) for r in rows) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# stream_candidate_pairs unit tests
# ---------------------------------------------------------------------------

class TestStreamCandidatePairs(unittest.TestCase):

    def _blocker(self, **kwargs):
        return EntityBlocker(BlockingConfig(**kwargs))

    def test_returns_generator(self):
        """stream_candidate_pairs must return a generator, not a list."""
        import types
        df = make_df(
            {"id": "1", "name": "Acme Corp",  "name_clean": "acme corp"},
            {"id": "2", "name": "Acme Inc",   "name_clean": "acme inc"},
        )
        blocker = self._blocker(
            use_token_prefix_blocking=True,
            use_sorted_neighborhood_blocking=False,
            use_address_street_number_blocking=False,
            use_exact_attribute_blocking=False,
        )
        result = blocker.stream_candidate_pairs(df, id_col_left="id", name_col="name")
        self.assertIsInstance(result, types.GeneratorType)

    def test_stream_yields_same_pairs_as_list_method(self):
        """Pairs from stream must be a superset of the list method's output
        (stream may include duplicates across strategies)."""
        df = make_df(
            {"id": "1", "name": "Acme Corp",   "name_clean": "acme corp",   "postal_code": "94105"},
            {"id": "2", "name": "Acme Inc",    "name_clean": "acme inc",    "postal_code": "94105"},
            {"id": "3", "name": "Globex Corp", "name_clean": "globex corp", "postal_code": "10001"},
        )
        blocker = self._blocker(
            use_token_prefix_blocking=True,
            use_sorted_neighborhood_blocking=False,
            use_address_street_number_blocking=False,
            use_exact_attribute_blocking=True,
        )
        list_pairs = set(blocker.generate_candidate_pairs_from_tables(
            df, id_col_left="id", name_col="name", postal_col="postal_code"
        ))
        stream_pairs = set(blocker.stream_candidate_pairs(
            df, id_col_left="id", name_col="name", postal_col="postal_code"
        ))
        # Every pair from the list method must appear in the stream
        for p in list_pairs:
            self.assertIn(p, stream_pairs,
                f"List-method pair {p} missing from stream")

    def test_stream_consumed_incrementally(self):
        """Pairs can be consumed one at a time without materialising all."""
        df = make_df(
            {"id": "1", "name": "Acme Corp",  "name_clean": "acme corp"},
            {"id": "2", "name": "Acme Inc",   "name_clean": "acme inc"},
            {"id": "3", "name": "Acme Ltd",   "name_clean": "acme ltd"},
        )
        blocker = self._blocker(
            use_token_prefix_blocking=True,
            use_sorted_neighborhood_blocking=False,
            use_address_street_number_blocking=False,
            use_exact_attribute_blocking=False,
        )
        gen = blocker.stream_candidate_pairs(df, id_col_left="id", name_col="name")
        first = next(gen)  # consumes only one pair
        self.assertIsInstance(first, tuple)
        self.assertEqual(len(first), 2)

    def test_stream_two_table_linkage(self):
        """stream_candidate_pairs works in two-table (linkage) mode."""
        left = make_df(
            {"id_l": "L1", "name": "Apple Inc",  "name_clean": "apple inc"},
        )
        right = make_df(
            {"id_r": "R1", "name": "Apple Corp", "name_clean": "apple corp"},
            {"id_r": "R2", "name": "Meta Platforms", "name_clean": "meta platforms"},
        )
        blocker = self._blocker(
            use_token_prefix_blocking=True,
            use_sorted_neighborhood_blocking=False,
            use_address_street_number_blocking=False,
            use_exact_attribute_blocking=False,
        )
        pairs = list(blocker.stream_candidate_pairs(
            left, right, id_col_left="id_l", id_col_right="id_r", name_col="name"
        ))
        self.assertIn(("L1", "R1"), pairs)

    def test_stream_empty_when_all_strategies_disabled(self):
        """No pairs are yielded when all strategies are disabled."""
        df = make_df(
            {"id": "1", "name": "Acme Corp", "name_clean": "acme corp"},
            {"id": "2", "name": "Acme Inc",  "name_clean": "acme inc"},
        )
        blocker = self._blocker(
            use_token_prefix_blocking=False,
            use_sorted_neighborhood_blocking=False,
            use_address_street_number_blocking=False,
            use_exact_attribute_blocking=False,
        )
        pairs = list(blocker.stream_candidate_pairs(df, id_col_left="id", name_col="name"))
        self.assertEqual(pairs, [])

    def test_stream_does_not_hold_full_set(self):
        """Consuming the first pair via next() does not force all pairs to exist."""
        # This is a structural test: if stream were a list, next() on a list
        # would still have required full computation. With a generator, we can
        # call next() and stop — verifying the interface is lazy.
        import types
        df = make_df(
            *[{"id": str(i), "name": f"Company {i}", "name_clean": f"company {i}"}
              for i in range(100)]
        )
        blocker = self._blocker(
            use_token_prefix_blocking=True,
            use_sorted_neighborhood_blocking=False,
            use_address_street_number_blocking=False,
            use_exact_attribute_blocking=False,
        )
        gen = blocker.stream_candidate_pairs(df, id_col_left="id", name_col="name")
        self.assertIsInstance(gen, types.GeneratorType)
        # Consume only one pair — should not raise StopIteration for a 100-entity table
        pair = next(gen, None)
        self.assertIsNotNone(pair)


# ---------------------------------------------------------------------------
# Integration: reservoir path uses streaming (no full materialisation)
# ---------------------------------------------------------------------------

class TestReservoirUsesStreaming(unittest.TestCase):

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        s1 = self.tmp / "s1.tsv"
        s2 = self.tmp / "s2.tsv"
        self.gt_path = self.tmp / "gt.tsv"
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
        _write_gt(self.gt_path, [("S1-000", "S2-000"), ("S1-001", "S2-001")])
        self.gt = load_amazon_ground_truth(self.gt_path)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_training_completes_without_materialising_all_candidates(self):
        """build_amazon_training_dataframe must complete and return a valid DataFrame."""
        result = build_amazon_training_dataframe(
            self.combined, self.gt, negative_ratio=1
        )
        self.assertGreater(len(result), 0)
        self.assertIn("label", result.columns)

    def test_positives_never_in_negatives_with_streaming(self):
        """Streaming path must still exclude GT positives from the reservoir."""
        result = build_amazon_training_dataframe(
            self.combined, self.gt, negative_ratio=0
        )
        negatives = result[result["label"] == 0]
        pos_keys = {
            tuple(sorted(["source1_S1-000", "source2_S2-000"])),
            tuple(sorted(["source1_S1-001", "source2_S2-001"])),
        }
        for _, row in negatives.iterrows():
            key = tuple(sorted([row["id_1"], row["id_2"]]))
            self.assertNotIn(key, pos_keys,
                f"GT positive appeared as negative: {key}")

    def test_negative_ratio_respected_in_streaming_path(self):
        result = build_amazon_training_dataframe(
            self.combined, self.gt, negative_ratio=2
        )
        n_pos = (result["label"] == 1).sum()
        n_neg = (result["label"] == 0).sum()
        self.assertLessEqual(n_neg, 2 * n_pos)

    def test_reproducible_with_streaming(self):
        r1 = build_amazon_training_dataframe(
            self.combined, self.gt, negative_ratio=1, random_state=7
        )
        r2 = build_amazon_training_dataframe(
            self.combined, self.gt, negative_ratio=1, random_state=7
        )
        neg1 = frozenset(zip(r1[r1["label"]==0]["id_1"], r1[r1["label"]==0]["id_2"]))
        neg2 = frozenset(zip(r2[r2["label"]==0]["id_1"], r2[r2["label"]==0]["id_2"]))
        self.assertEqual(neg1, neg2)


if __name__ == "__main__":
    unittest.main()
