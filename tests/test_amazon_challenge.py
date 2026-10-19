"""Tests for Amazon-challenge support: business_address alias, country column,
multi-source loading, and cross-source candidate generation."""

import sys
import tempfile
import shutil
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from backend.pipeline.ingestion import (
    DataIngestionPipeline,
    SchemaValidator,
    FIELD_ALIASES,
)
from backend.pipeline.linkage import (
    CANONICAL_ENTITY_FIELDS,
    load_amazon_sources,
    load_amazon_ground_truth,
    generate_cross_source_candidates,
    build_pairwise_dataframe_from_candidates,
)
from backend.pipeline.blocking import EntityBlocker


# ---------------------------------------------------------------------------
# Fix 1 — business_address alias
# ---------------------------------------------------------------------------

class TestBusinessAddressAlias(unittest.TestCase):

    def test_business_address_in_field_aliases(self):
        """'business_address' must appear in the address alias list."""
        self.assertIn("business_address", FIELD_ALIASES["address"])

    def test_schema_validator_maps_business_address(self):
        """SchemaValidator resolves 'business_address' to the canonical 'address'."""
        validator = SchemaValidator()
        schema = validator.inspect_headers(
            ["entity_id", "business_name", "business_address", "country"]
        )
        self.assertEqual(schema.column_mapping.get("address"), "business_address")

    def test_schema_validator_maps_business_name(self):
        """SchemaValidator resolves 'business_name' to the canonical 'name'."""
        validator = SchemaValidator()
        schema = validator.inspect_headers(
            ["entity_id", "business_name", "business_address", "country"]
        )
        self.assertEqual(schema.column_mapping.get("name"), "business_name")

    def test_ingestion_succeeds_with_amazon_columns(self):
        """Full ingestion pipeline accepts the Amazon column schema without error."""
        tmpdir = Path(tempfile.mkdtemp())
        try:
            raw_dir = tmpdir / "raw"
            proc_dir = tmpdir / "processed"
            raw_dir.mkdir(parents=True)
            proc_dir.mkdir(parents=True)

            tsv = raw_dir / "train_source1.tsv"
            tsv.write_text(
                "entity_id\tbusiness_name\tbusiness_address\tcountry\n"
                "1\tAcme Corp\t100 Main St\tUS\n"
                "2\tGlobex Inc\t200 Market Ave\tUS\n",
                encoding="utf-8",
            )
            pipeline = DataIngestionPipeline(raw_dir=raw_dir, processed_dir=proc_dir)
            result = pipeline.ingest_file(tsv)
            self.assertEqual(result["status"], "success")
            self.assertEqual(result["total_records"], 2)
            mapping = result["detected_column_mapping"]
            self.assertEqual(mapping["address"], "business_address")
            self.assertEqual(mapping["name"], "business_name")
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Fix 2 — country column preservation
# ---------------------------------------------------------------------------

class TestCountryColumn(unittest.TestCase):

    def test_country_in_field_aliases(self):
        """'country' canonical key must exist in FIELD_ALIASES."""
        self.assertIn("country", FIELD_ALIASES)

    def test_country_in_canonical_entity_fields(self):
        """'country' must be in CANONICAL_ENTITY_FIELDS for linkage."""
        self.assertIn("country", CANONICAL_ENTITY_FIELDS)

    def test_country_preserved_in_load_amazon_sources(self):
        """load_amazon_sources retains the country column after canonicalization."""
        tmpdir = Path(tempfile.mkdtemp())
        try:
            p = tmpdir / "src1.tsv"
            p.write_text(
                "entity_id\tbusiness_name\tbusiness_address\tcountry\n"
                "1\tAcme Corp\t100 Main St\tUS\n"
                "2\tBeta Ltd\t200 Oak Ave\tGB\n",
                encoding="utf-8",
            )
            combined = load_amazon_sources([p], source_labels=["source1"])
            self.assertIn("country", combined.columns)
            self.assertEqual(combined.loc[0, "country"], "US")
            self.assertEqual(combined.loc[1, "country"], "GB")
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_country_preserved_in_pair_rows(self):
        """build_pairwise_dataframe_from_candidates includes country_1 and country_2."""
        df = pd.DataFrame([
            {"id": "1", "name": "Acme Corp", "address": "100 Main St", "country": "US"},
            {"id": "2", "name": "Acme Inc",  "address": "100 Main St", "country": "US"},
        ])
        candidates = [("1", "2")]
        pair_df = build_pairwise_dataframe_from_candidates(df, candidates)
        self.assertIn("country_1", pair_df.columns)
        self.assertIn("country_2", pair_df.columns)
        self.assertEqual(pair_df.loc[0, "country_1"], "US")

    def test_country_missing_does_not_break_older_datasets(self):
        """Datasets without country still ingest and process without error."""
        tmpdir = Path(tempfile.mkdtemp())
        try:
            raw_dir = tmpdir / "raw"
            proc_dir = tmpdir / "processed"
            raw_dir.mkdir(parents=True)
            proc_dir.mkdir(parents=True)

            tsv = raw_dir / "legacy.tsv"
            tsv.write_text(
                "id\tname\taddress\n"
                "1\tWayne Enterprises\t100 Gotham Way\n",
                encoding="utf-8",
            )
            pipeline = DataIngestionPipeline(raw_dir=raw_dir, processed_dir=proc_dir)
            result = pipeline.ingest_file(tsv)
            self.assertEqual(result["status"], "success")
            self.assertNotIn("country", result["detected_column_mapping"])
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_country_blocking_key_works(self):
        """country column is accepted as a blocking key in EntityBlocker."""
        df = pd.DataFrame([
            {"id": "1", "name": "Alpha Corp", "country": "US"},
            {"id": "2", "name": "Alpha Inc",  "country": "US"},
            {"id": "3", "name": "Beta GmbH",  "country": "DE"},
        ])
        blocker = EntityBlocker()
        # Should not raise; country is passed as exact-attribute key
        pairs = blocker.generate_candidate_pairs_from_tables(
            df,
            id_col_left="id",
            name_col="name",
            country_col="country",
        )
        self.assertIsInstance(pairs, list)
        # Entities sharing country "US" may become candidates via exact-attribute block
        self.assertIn(("1", "2"), pairs)
        # Entity 3 shares no country with 1 or 2
        self.assertNotIn(("1", "3"), pairs)
        self.assertNotIn(("2", "3"), pairs)


# ---------------------------------------------------------------------------
# Fix 3 — multi-source loading and cross-source candidate generation
# ---------------------------------------------------------------------------

class TestMultiSourceLoading(unittest.TestCase):

    def _write_source(self, path: Path, rows: list):
        lines = ["entity_id\tbusiness_name\tbusiness_address\tcountry"]
        for r in rows:
            lines.append("\t".join(r))
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def setUp(self):
        self.tmpdir = Path(tempfile.mkdtemp())
        self.s1 = self.tmpdir / "train_source1.tsv"
        self.s2 = self.tmpdir / "train_source2.tsv"
        self.s3 = self.tmpdir / "train_source3.tsv"
        self._write_source(self.s1, [
            ("101", "Acme Corporation", "100 Main St", "US"),
            ("102", "Globex Inc",        "200 Oak Ave",  "US"),
        ])
        self._write_source(self.s2, [
            ("201", "Acme Corp",         "100 Main Street", "US"),
            ("202", "Cyberdyne Systems", "300 Elm Road",    "US"),
        ])
        self._write_source(self.s3, [
            ("301", "Wayne Enterprises", "1 Wayne Tower", "US"),
        ])

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_load_three_sources_combined_row_count(self):
        """Combined table has rows from all three sources."""
        combined = load_amazon_sources(
            [self.s1, self.s2, self.s3],
            source_labels=["source1", "source2", "source3"],
        )
        self.assertEqual(len(combined), 5)

    def test_source_column_assigned_correctly(self):
        """Each row carries the correct source label."""
        combined = load_amazon_sources(
            [self.s1, self.s2, self.s3],
            source_labels=["source1", "source2", "source3"],
        )
        self.assertEqual(list(combined["source"].unique()), ["source1", "source2", "source3"])
        self.assertEqual(combined[combined["source"] == "source1"].shape[0], 2)
        self.assertEqual(combined[combined["source"] == "source3"].shape[0], 1)

    def test_entity_id_preserved(self):
        """Original entity_id values are retained in the entity_id column."""
        combined = load_amazon_sources([self.s1], source_labels=["source1"])
        self.assertIn("entity_id", combined.columns)
        entity_ids = combined["entity_id"].tolist()
        self.assertIn("101", entity_ids)
        self.assertIn("102", entity_ids)

    def test_global_ids_are_unique(self):
        """id column is unique across all sources."""
        combined = load_amazon_sources(
            [self.s1, self.s2, self.s3],
            source_labels=["source1", "source2", "source3"],
        )
        self.assertEqual(len(combined["id"].unique()), len(combined))

    def test_global_id_format(self):
        """id column uses <source>_<entity_id> format."""
        combined = load_amazon_sources([self.s1], source_labels=["source1"])
        self.assertIn("source1_101", combined["id"].tolist())

    def test_name_column_canonicalized(self):
        """business_name is mapped to canonical 'name' column."""
        combined = load_amazon_sources([self.s1], source_labels=["source1"])
        self.assertIn("name", combined.columns)
        self.assertEqual(combined.loc[0, "name"], "Acme Corporation")

    def test_address_column_canonicalized(self):
        """business_address is mapped to canonical 'address' column."""
        combined = load_amazon_sources([self.s1], source_labels=["source1"])
        self.assertIn("address", combined.columns)


class TestCrossSourceCandidateGeneration(unittest.TestCase):

    def _write_source(self, path: Path, rows: list):
        lines = ["entity_id\tbusiness_name\tbusiness_address\tcountry"]
        for r in rows:
            lines.append("\t".join(r))
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def setUp(self):
        self.tmpdir = Path(tempfile.mkdtemp())
        self.s1 = self.tmpdir / "train_source1.tsv"
        self.s2 = self.tmpdir / "train_source2.tsv"
        self._write_source(self.s1, [
            ("101", "Acme Corporation", "100 Main St",    "US"),
            ("102", "Globex Inc",       "200 Oak Ave",    "US"),
        ])
        self._write_source(self.s2, [
            ("201", "Acme Corp",        "100 Main Street", "US"),
            ("202", "Stark Industries", "300 Elm Road",    "US"),
        ])

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_cross_source_candidates_generated(self):
        """Acme Corporation (s1) and Acme Corp (s2) must be candidate pairs."""
        combined = load_amazon_sources(
            [self.s1, self.s2],
            source_labels=["source1", "source2"],
        )
        pair_df, candidate_df = generate_cross_source_candidates(combined)
        self.assertGreater(len(candidate_df), 0)
        # Verify Acme pair is a candidate
        ids = set(map(tuple, candidate_df[["id_1", "id_2"]].values.tolist()))
        acme_pair = tuple(sorted(["source1_101", "source2_201"]))
        self.assertIn(acme_pair, ids)

    def test_no_within_source_pairs(self):
        """generate_cross_source_candidates must not produce within-source pairs."""
        combined = load_amazon_sources(
            [self.s1, self.s2],
            source_labels=["source1", "source2"],
        )
        _, candidate_df = generate_cross_source_candidates(combined)
        for _, row in candidate_df.iterrows():
            src_a = row["id_1"].split("_")[0]
            src_b = row["id_2"].split("_")[0]
            self.assertNotEqual(
                src_a, src_b,
                f"Within-source pair found: {row['id_1']} vs {row['id_2']}",
            )

    def test_pair_df_has_required_columns(self):
        """Materialized pair DataFrame must have name_1, name_2, address_1, address_2."""
        combined = load_amazon_sources(
            [self.s1, self.s2],
            source_labels=["source1", "source2"],
        )
        pair_df, _ = generate_cross_source_candidates(combined)
        for col in ["id_1", "id_2", "name_1", "name_2", "address_1", "address_2"]:
            self.assertIn(col, pair_df.columns, f"Missing column: {col}")

    def test_single_source_raises_value_error(self):
        """Cross-source blocking with only one source must raise ValueError."""
        combined = load_amazon_sources([self.s1], source_labels=["source1"])
        with self.assertRaises(ValueError):
            generate_cross_source_candidates(combined)


class TestChunkedLoading(unittest.TestCase):
    """Prove that chunked reading produces the same result as full-file reading."""

    def _write_source(self, path: Path, n_rows: int):
        lines = ["entity_id\tbusiness_name\tbusiness_address\tcountry"]
        for i in range(n_rows):
            lines.append(f"{i}\tCompany {i}\t{i} Main St\tUS")
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def setUp(self):
        self.tmpdir = Path(tempfile.mkdtemp())
        self.src = self.tmpdir / "large_source.tsv"
        self._write_source(self.src, n_rows=250)  # 250 rows, split across chunks

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_chunked_and_full_produce_same_row_count(self):
        """chunksize=100 produces the same number of rows as chunksize=None."""
        chunked = load_amazon_sources(
            [self.src], source_labels=["s1"], chunksize=100
        )
        full = load_amazon_sources(
            [self.src], source_labels=["s1"], chunksize=None
        )
        self.assertEqual(len(chunked), len(full))
        self.assertEqual(len(chunked), 250)

    def test_chunked_global_ids_are_unique(self):
        """Chunked loading must still produce unique global IDs across chunk boundaries."""
        chunked = load_amazon_sources(
            [self.src], source_labels=["s1"], chunksize=100
        )
        self.assertEqual(len(chunked["id"].unique()), 250)

    def test_chunked_preserves_entity_ids(self):
        """entity_id column is correctly preserved across chunk boundaries."""
        chunked = load_amazon_sources(
            [self.src], source_labels=["s1"], chunksize=100
        )
        expected_ids = {str(i) for i in range(250)}
        self.assertEqual(set(chunked["entity_id"].tolist()), expected_ids)

    def test_chunked_same_columns_as_full(self):
        """Chunked output has the same column set as full-read output."""
        chunked = load_amazon_sources(
            [self.src], source_labels=["s1"], chunksize=100
        )
        full = load_amazon_sources(
            [self.src], source_labels=["s1"], chunksize=None
        )
        self.assertEqual(sorted(chunked.columns.tolist()), sorted(full.columns.tolist()))

    def test_small_chunksize_one_row_per_chunk(self):
        """Extreme chunksize=1 still produces correct results."""
        chunked = load_amazon_sources(
            [self.src], source_labels=["s1"], chunksize=1
        )
        self.assertEqual(len(chunked), 250)
        self.assertEqual(len(chunked["id"].unique()), 250)

    def test_two_sources_chunked(self):
        """Chunked loading of two sources preserves source labels correctly."""
        src2 = self.tmpdir / "large_source2.tsv"
        self._write_source(src2, n_rows=150)
        combined = load_amazon_sources(
            [self.src, src2], source_labels=["s1", "s2"], chunksize=100
        )
        self.assertEqual(len(combined), 400)
        self.assertEqual(combined[combined["source"] == "s1"].shape[0], 250)
        self.assertEqual(combined[combined["source"] == "s2"].shape[0], 150)


class TestGroundTruthLoading(unittest.TestCase):

    def test_ground_truth_parsed_to_pairs(self):
        """load_amazon_ground_truth expands comma-separated matches into rows."""
        tmpdir = Path(tempfile.mkdtemp())
        try:
            gt = tmpdir / "train_ground_truth.tsv"
            # Format: source1_entity_id <TAB> matched_entity_ids (comma-sep)
            gt.write_text("101\t201,202\n102\t203\n", encoding="utf-8")
            result = load_amazon_ground_truth(gt)
            self.assertEqual(len(result), 3)
            self.assertIn("left_id",  result.columns)
            self.assertIn("right_id", result.columns)
            self.assertIn("label",    result.columns)
            # All labels should be 1
            self.assertTrue((result["label"] == 1).all())
            pairs = set(zip(result["left_id"], result["right_id"]))
            self.assertIn(("101", "201"), pairs)
            self.assertIn(("101", "202"), pairs)
            self.assertIn(("102", "203"), pairs)
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_ground_truth_skips_header_if_present(self):
        """load_amazon_ground_truth skips a text header row."""
        tmpdir = Path(tempfile.mkdtemp())
        try:
            gt = tmpdir / "gt_with_header.tsv"
            gt.write_text(
                "source1_entity_id\tmatched_entity_ids\n101\t201\n",
                encoding="utf-8",
            )
            result = load_amazon_ground_truth(gt)
            self.assertEqual(len(result), 1)
            self.assertEqual(result.loc[0, "left_id"], "101")
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    # ------------------------------------------------------------------
    # Regression tests for the header-skip bug with hyphenated IDs
    # ------------------------------------------------------------------

    def test_hyphenated_ids_first_row_not_dropped(self):
        """First data row with IDs like S1-965667 must NOT be treated as a header."""
        tmpdir = Path(tempfile.mkdtemp())
        try:
            gt = tmpdir / "gt_hyphen.tsv"
            # No header — first row is a real data row with hyphenated IDs
            gt.write_text(
                "S1-965667\tS2-123456,S3-789012\n"
                "S1-111111\tS2-222222\n",
                encoding="utf-8",
            )
            result = load_amazon_ground_truth(gt)
            # Both rows must be present (3 pairs total)
            self.assertEqual(len(result), 3)
            pairs = set(zip(result["left_id"], result["right_id"]))
            self.assertIn(("S1-965667", "S2-123456"), pairs)
            self.assertIn(("S1-965667", "S3-789012"), pairs)
            self.assertIn(("S1-111111", "S2-222222"), pairs)
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_hyphenated_ids_with_header_row_skips_only_header(self):
        """With a real header + hyphenated IDs, only the header line is skipped."""
        tmpdir = Path(tempfile.mkdtemp())
        try:
            gt = tmpdir / "gt_header_hyphen.tsv"
            gt.write_text(
                "source1_entity_id\tmatched_entity_ids\n"
                "S1-965667\tS2-123456\n"
                "S1-111111\tS2-222222\n",
                encoding="utf-8",
            )
            result = load_amazon_ground_truth(gt)
            self.assertEqual(len(result), 2)
            pairs = set(zip(result["left_id"], result["right_id"]))
            self.assertIn(("S1-965667", "S2-123456"), pairs)
            self.assertIn(("S1-111111", "S2-222222"), pairs)
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_multiple_matches_per_entity_hyphenated(self):
        """Multiple comma-separated right-side IDs are each emitted as a separate pair."""
        tmpdir = Path(tempfile.mkdtemp())
        try:
            gt = tmpdir / "gt_multi.tsv"
            gt.write_text(
                "S1-001\tS2-010,S2-011,S3-020\n",
                encoding="utf-8",
            )
            result = load_amazon_ground_truth(gt)
            self.assertEqual(len(result), 3)
            pairs = set(zip(result["left_id"], result["right_id"]))
            self.assertIn(("S1-001", "S2-010"), pairs)
            self.assertIn(("S1-001", "S2-011"), pairs)
            self.assertIn(("S1-001", "S3-020"), pairs)
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
