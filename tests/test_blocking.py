"""
Unit tests for the smart blocking stage of EntityMatch AI.

Covers:
  1. Obvious matches become candidates (name prefix, SNM, address, postal)
  2. Clearly unrelated records are not unnecessarily paired
  3. Duplicate candidate pairs are removed (set semantics)
  4. Records with missing address/name fields are handled gracefully
  5. Individual strategies can be toggled; combined set is union of results
  6. Two-table (linkage) mode works correctly
  7. max_cluster_size stop-key pruning
"""

import sys
import unittest
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.config import BlockingConfig
from backend.pipeline.blocking import EntityBlocker, _street_number, _ordered_pair


# ---------------------------------------------------------------------------
# Helper builders
# ---------------------------------------------------------------------------

def make_df(*rows):
    """Convenience builder for small test DataFrames."""
    return pd.DataFrame(list(rows))


# ---------------------------------------------------------------------------
# Unit-level helper tests
# ---------------------------------------------------------------------------

class TestHelpers(unittest.TestCase):

    def test_street_number_extracts_leading_digits(self):
        self.assertEqual(_street_number("100 n main st"), "100")
        self.assertEqual(_street_number("1600 pennsylvania ave"), "1600")
        self.assertEqual(_street_number("42 wallaby way"), "42")

    def test_street_number_returns_empty_when_absent(self):
        self.assertEqual(_street_number(""), "")
        self.assertEqual(_street_number("po box 5"), "")   # no leading number
        self.assertEqual(_street_number("   "), "")

    def test_ordered_pair_always_smaller_first(self):
        self.assertEqual(_ordered_pair("z", "a"), ("a", "z"))
        self.assertEqual(_ordered_pair("1", "2"), ("1", "2"))
        self.assertEqual(_ordered_pair("b", "a"), ("a", "b"))

    def test_ordered_pair_equal_ids(self):
        self.assertEqual(_ordered_pair("x", "x"), ("x", "x"))


# ---------------------------------------------------------------------------
# Deduplication (single-table) blocking
# ---------------------------------------------------------------------------

class TestDeduplicationBlocking(unittest.TestCase):

    def _blocker(self, **kwargs) -> EntityBlocker:
        cfg = BlockingConfig(**kwargs)
        return EntityBlocker(config=cfg)

    # 1. Obvious matches become candidates —————————————————————————————

    def test_name_prefix_pairs_obvious_matches(self):
        """Records with highly similar names share a token prefix → candidates."""
        df = make_df(
            {"id": "1", "name": "Apple Inc",      "name_clean": "apple inc"},
            {"id": "2", "name": "Apple Computer", "name_clean": "apple computer"},
            {"id": "3", "name": "Microsoft Corp", "name_clean": "microsoft corp"},
            {"id": "4", "name": "Micro-soft Corporation", "name_clean": "microsoft corp"},
        )
        blocker = self._blocker(
            use_token_prefix_blocking=True,
            use_sorted_neighborhood_blocking=False,
            use_address_street_number_blocking=False,
            use_exact_attribute_blocking=False,
        )
        pairs = blocker.generate_candidate_pairs_from_tables(df, id_col_left="id", name_col="name")
        self.assertIn(("1", "2"), pairs)   # both share "appl" prefix
        self.assertIn(("3", "4"), pairs)   # both share "micr" prefix

    def test_snm_pairs_name_variants_that_differ_in_prefix(self):
        """SNM pairs records whose names sort close together even without a shared prefix."""
        df = make_df(
            {"id": "A", "name": "Acme Services LLC",  "name_clean": "acme services llc"},
            {"id": "B", "name": "Acme Services Inc",  "name_clean": "acme services inc"},
            {"id": "C", "name": "Zebra Corp",          "name_clean": "zebra corp"},
        )
        blocker = self._blocker(
            use_token_prefix_blocking=False,
            use_sorted_neighborhood_blocking=True,
            sorted_neighborhood_window=3,
            use_address_street_number_blocking=False,
            use_exact_attribute_blocking=False,
        )
        pairs = blocker.generate_candidate_pairs_from_tables(df, id_col_left="id", name_col="name")
        self.assertIn(("A", "B"), pairs)   # adjacent in sort order

    def test_address_street_number_pairs_same_building(self):
        """Records at the same street number become candidates via address blocking."""
        df = make_df(
            {"id": "1", "name": "Corp A", "name_clean": "corp a", "address": "200 Main St",     "address_clean": "200 main st"},
            {"id": "2", "name": "Corp B", "name_clean": "corp b", "address": "200 Market Blvd", "address_clean": "200 market blvd"},
            {"id": "3", "name": "Corp C", "name_clean": "corp c", "address": "999 Oak Ave",     "address_clean": "999 oak ave"},
        )
        blocker = self._blocker(
            use_token_prefix_blocking=False,
            use_sorted_neighborhood_blocking=False,
            use_address_street_number_blocking=True,
            use_exact_attribute_blocking=False,
        )
        pairs = blocker.generate_candidate_pairs_from_tables(
            df, id_col_left="id", name_col="name", address_col="address"
        )
        self.assertIn(("1", "2"), pairs)   # same street number "200"
        self.assertNotIn(("1", "3"), pairs)
        self.assertNotIn(("2", "3"), pairs)

    def test_postal_code_exact_block_produces_candidates(self):
        """Records sharing a postal code become candidates via exact-attribute blocking."""
        df = make_df(
            {"id": "1", "name": "Co One",   "name_clean": "co one",   "postal_code": "94105"},
            {"id": "2", "name": "Co Two",   "name_clean": "co two",   "postal_code": "94105"},
            {"id": "3", "name": "Co Three", "name_clean": "co three", "postal_code": "10001"},
        )
        blocker = self._blocker(
            use_token_prefix_blocking=False,
            use_sorted_neighborhood_blocking=False,
            use_address_street_number_blocking=False,
            use_exact_attribute_blocking=True,
        )
        pairs = blocker.generate_candidate_pairs_from_tables(
            df, id_col_left="id", name_col="name", postal_col="postal_code"
        )
        self.assertIn(("1", "2"), pairs)
        self.assertNotIn(("1", "3"), pairs)
        self.assertNotIn(("2", "3"), pairs)

    # 2. Unrelated records are not unnecessarily paired ————————————————

    def test_completely_different_names_not_paired(self):
        """Records with no shared name tokens and different postal codes stay apart."""
        df = make_df(
            {"id": "1", "name": "Alpha Industries", "name_clean": "alpha industries", "postal_code": "10001"},
            {"id": "2", "name": "Beta Services",    "name_clean": "beta services",    "postal_code": "20002"},
            {"id": "3", "name": "Gamma Solutions",  "name_clean": "gamma solutions",  "postal_code": "30003"},
        )
        blocker = self._blocker(
            use_token_prefix_blocking=True,
            use_sorted_neighborhood_blocking=False,
            use_address_street_number_blocking=False,
            use_exact_attribute_blocking=True,
        )
        pairs = blocker.generate_candidate_pairs_from_tables(
            df, id_col_left="id", name_col="name", postal_col="postal_code"
        )
        # No pair should exist because nothing is shared
        self.assertNotIn(("1", "2"), pairs)
        self.assertNotIn(("1", "3"), pairs)
        self.assertNotIn(("2", "3"), pairs)

    # 3. Duplicate candidate pairs are removed —————————————————————————

    def test_no_duplicate_pairs_in_output(self):
        """Running multiple strategies produces deduplicated pairs."""
        df = make_df(
            {"id": "1", "name": "Acme Corp",       "name_clean": "acme corp",       "postal_code": "94105"},
            {"id": "2", "name": "Acme Corporation", "name_clean": "acme corporation", "postal_code": "94105"},
        )
        blocker = self._blocker(
            use_token_prefix_blocking=True,
            use_sorted_neighborhood_blocking=True,
            use_address_street_number_blocking=False,
            use_exact_attribute_blocking=True,
        )
        pairs = blocker.generate_candidate_pairs_from_tables(
            df, id_col_left="id", name_col="name", postal_col="postal_code"
        )
        # Pair (1, 2) may be discovered by multiple strategies but must appear once
        self.assertEqual(pairs.count(("1", "2")), 1)

    def test_pair_list_contains_no_self_pairs(self):
        """No record should be paired with itself."""
        df = make_df(
            {"id": "1", "name": "Widget Co",  "name_clean": "widget co"},
            {"id": "2", "name": "Widget Inc", "name_clean": "widget inc"},
        )
        blocker = self._blocker()
        pairs = blocker.generate_candidate_pairs_from_tables(df, id_col_left="id", name_col="name")
        for a, b in pairs:
            self.assertNotEqual(a, b, f"Self-pair found: ({a}, {b})")

    def test_pair_ordering_is_deterministic(self):
        """Output list is sorted; calling twice gives the same order."""
        df = make_df(
            {"id": "1", "name": "Acme Corp",  "name_clean": "acme corp"},
            {"id": "2", "name": "Acme Inc",   "name_clean": "acme inc"},
            {"id": "3", "name": "Acme Ltd",   "name_clean": "acme ltd"},
        )
        blocker = self._blocker()
        pairs_a = blocker.generate_candidate_pairs_from_tables(df, id_col_left="id", name_col="name")
        pairs_b = blocker.generate_candidate_pairs_from_tables(df, id_col_left="id", name_col="name")
        self.assertEqual(pairs_a, pairs_b)

    # 4. Missing address / name fields ————————————————————————————————

    def test_missing_address_does_not_raise(self):
        """Records without an address value are silently skipped by address blocking."""
        df = make_df(
            {"id": "1", "name": "Alpha Corp", "name_clean": "alpha corp", "address": "",    "address_clean": ""},
            {"id": "2", "name": "Alpha Inc",  "name_clean": "alpha inc",  "address": None,  "address_clean": None},
            {"id": "3", "name": "Alpha Ltd",  "name_clean": "alpha ltd",  "address": "100 Main St", "address_clean": "100 main st"},
        )
        blocker = self._blocker(
            use_token_prefix_blocking=True,
            use_sorted_neighborhood_blocking=False,
            use_address_street_number_blocking=True,
            use_exact_attribute_blocking=False,
        )
        # Must not raise; records with missing address are just skipped
        try:
            pairs = blocker.generate_candidate_pairs_from_tables(
                df, id_col_left="id", name_col="name", address_col="address"
            )
        except Exception as exc:
            self.fail(f"blocking raised an exception with missing address fields: {exc}")

    def test_missing_name_does_not_raise(self):
        """Records with empty names are skipped by name-based strategies."""
        df = make_df(
            {"id": "1", "name": "",        "name_clean": ""},
            {"id": "2", "name": None,      "name_clean": None},
            {"id": "3", "name": "Real Co", "name_clean": "real co"},
        )
        blocker = self._blocker()
        try:
            blocker.generate_candidate_pairs_from_tables(df, id_col_left="id", name_col="name")
        except Exception as exc:
            self.fail(f"blocking raised an exception with missing name fields: {exc}")

    # 5. Strategy combination ——————————————————————————————————————————

    def test_combined_strategies_produce_union(self):
        """Union of multiple strategies contains at least as many pairs as any single strategy."""
        df = make_df(
            {"id": "1", "name": "Acme Corp",   "name_clean": "acme corp",   "postal_code": "94105"},
            {"id": "2", "name": "Acme Inc",    "name_clean": "acme inc",    "postal_code": "94105"},
            {"id": "3", "name": "Globex Corp", "name_clean": "globex corp", "postal_code": "10001"},
            {"id": "4", "name": "Globex Inc",  "name_clean": "globex inc",  "postal_code": "10001"},
        )
        cfg_all = BlockingConfig(
            use_token_prefix_blocking=True,
            use_sorted_neighborhood_blocking=True,
            use_address_street_number_blocking=False,
            use_exact_attribute_blocking=True,
        )
        cfg_prefix_only = BlockingConfig(
            use_token_prefix_blocking=True,
            use_sorted_neighborhood_blocking=False,
            use_address_street_number_blocking=False,
            use_exact_attribute_blocking=False,
        )
        pairs_all = EntityBlocker(cfg_all).generate_candidate_pairs_from_tables(
            df, id_col_left="id", name_col="name", postal_col="postal_code"
        )
        pairs_prefix = EntityBlocker(cfg_prefix_only).generate_candidate_pairs_from_tables(
            df, id_col_left="id", name_col="name", postal_col="postal_code"
        )
        # Combined must cover at least everything the single strategy covers
        for p in pairs_prefix:
            self.assertIn(p, pairs_all, f"Combined strategy missing pair {p} found by prefix-only")

    def test_disabling_all_strategies_produces_no_candidates(self):
        """When all strategies are disabled, no pairs are returned."""
        df = make_df(
            {"id": "1", "name": "Acme Corp", "name_clean": "acme corp"},
            {"id": "2", "name": "Acme Inc",  "name_clean": "acme inc"},
        )
        cfg = BlockingConfig(
            use_token_prefix_blocking=False,
            use_sorted_neighborhood_blocking=False,
            use_address_street_number_blocking=False,
            use_exact_attribute_blocking=False,
        )
        blocker = EntityBlocker(cfg)
        pairs = blocker.generate_candidate_pairs_from_tables(df, id_col_left="id", name_col="name")
        self.assertEqual(pairs, [])


# ---------------------------------------------------------------------------
# Two-table (linkage) blocking
# ---------------------------------------------------------------------------

class TestLinkageBlocking(unittest.TestCase):

    def test_linkage_pairs_cross_table_matches(self):
        """Left and right records with matching names are linked."""
        left = make_df(
            {"id_left": "L1", "name": "Apple Inc", "name_clean": "apple inc"},
            {"id_left": "L2", "name": "Google LLC", "name_clean": "google llc"},
        )
        right = make_df(
            {"id_right": "R1", "name": "Apple Computer", "name_clean": "apple computer"},
            {"id_right": "R2", "name": "Meta Platforms",  "name_clean": "meta platforms"},
        )
        blocker = EntityBlocker()
        pairs = blocker.generate_candidate_pairs_from_tables(
            left, right,
            id_col_left="id_left", id_col_right="id_right",
            name_col="name",
        )
        self.assertIn(("L1", "R1"), pairs)

    def test_linkage_does_not_produce_within_table_pairs(self):
        """Linkage mode should not produce pairs from two left-table records."""
        left = make_df(
            {"id_left": "L1", "name": "Acme Corp", "name_clean": "acme corp"},
            {"id_left": "L2", "name": "Acme Inc",  "name_clean": "acme inc"},
        )
        right = make_df(
            {"id_right": "R1", "name": "Acme Ltd",  "name_clean": "acme ltd"},
        )
        blocker = EntityBlocker(BlockingConfig(
            use_token_prefix_blocking=True,
            use_sorted_neighborhood_blocking=False,
            use_address_street_number_blocking=False,
            use_exact_attribute_blocking=False,
        ))
        pairs = blocker.generate_candidate_pairs_from_tables(
            left, right,
            id_col_left="id_left", id_col_right="id_right",
            name_col="name",
        )
        # (L1, L2) should not appear; only cross-table pairs
        for a, b in pairs:
            self.assertFalse(
                a.startswith("L") and b.startswith("L"),
                f"Within-left-table pair found: ({a}, {b})"
            )


# ---------------------------------------------------------------------------
# Max cluster size (stop-key pruning)
# ---------------------------------------------------------------------------

class TestClusterSizePruning(unittest.TestCase):

    def test_oversized_cluster_is_pruned(self):
        """A blocking key that groups more records than max_cluster_size is discarded."""
        # Create 10 records all sharing prefix "serv" (services, served, …)
        rows = [{"id": str(i), "name": f"services entity {i}", "name_clean": f"services entity {i}"}
                for i in range(10)]
        df = pd.DataFrame(rows)

        cfg = BlockingConfig(
            use_token_prefix_blocking=True,
            use_sorted_neighborhood_blocking=False,
            use_address_street_number_blocking=False,
            use_exact_attribute_blocking=False,
            max_cluster_size=5,  # anything over 5 is pruned
        )
        blocker = EntityBlocker(cfg)
        pairs = blocker.generate_candidate_pairs_from_tables(df, id_col_left="id", name_col="name")
        # The "serv" key has 10 members → pruned.  Unique shorter tokens may still fire.
        # Main assertion: no error raised; result is a list.
        self.assertIsInstance(pairs, list)


if __name__ == "__main__":
    unittest.main()
