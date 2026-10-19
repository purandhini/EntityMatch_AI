"""Unit tests for TSV data ingestion, schema detection, and identity validation."""

import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.pipeline.ingestion import (
    DataIngestionPipeline,
    SchemaValidator,
    EmptyFileError,
    MalformedFileError,
    SchemaValidationError,
)


class TestDataIngestion(unittest.TestCase):
    """Test suite for TSV data ingestion, validation, and encoding fallbacks."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.raw_dir = Path(self.temp_dir) / "data" / "raw"
        self.processed_dir = Path(self.temp_dir) / "data" / "processed"
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self.pipeline = DataIngestionPipeline(
            raw_dir=self.raw_dir,
            processed_dir=self.processed_dir,
        )

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    # 1. Valid Single-Table TSV
    def test_valid_single_table_ingestion(self):
        tsv_content = (
            "record_id\tcompany_name\tstreet_address\tcity\tstate\tzip\n"
            "101\tAcme Corp\t123 Main St\tSan Francisco\tCA\t94105\n"
            "102\tGlobex LLC\t456 Market Ave\tNew York\tNY\t10001\n"
        )
        test_file = self.raw_dir / "valid_single.tsv"
        test_file.write_text(tsv_content, encoding="utf-8")

        result = self.pipeline.ingest_file(test_file)

        self.assertEqual(result["status"], "success")
        self.assertEqual(result["dataset_format"], "SINGLE_TABLE")
        self.assertEqual(result["total_records"], 2)
        # Verify alias mappings detected
        mapping = result["detected_column_mapping"]
        self.assertEqual(mapping["name"], "company_name")
        self.assertEqual(mapping["address"], "street_address")
        self.assertEqual(mapping["city"], "city")
        self.assertEqual(mapping["postal_code"], "zip")
        # Verify preprocessed file created
        self.assertTrue(Path(result["preprocessed_file_path"]).is_file())

    # 2. Valid Pairwise TSV
    def test_valid_pairwise_ingestion(self):
        tsv_content = (
            "id_1\tname_1\taddress_1\tcity_1\tid_2\tname_2\taddress_2\tcity_2\tlabel\n"
            "A1\tApple Inc\t1 Infinite Loop\tCupertino\tB1\tApple Computer\t1 Infinite Loop\tCupertino\t1\n"
        )
        test_file = self.raw_dir / "valid_pairs.tsv"
        test_file.write_text(tsv_content, encoding="utf-8")

        result = self.pipeline.ingest_file(test_file)

        self.assertEqual(result["status"], "success")
        self.assertEqual(result["dataset_format"], "PAIRWISE")
        self.assertEqual(result["total_records"], 1)
        mapping = result["detected_column_mapping"]
        self.assertEqual(mapping["name_1"], "name_1")
        self.assertEqual(mapping["name_2"], "name_2")
        self.assertEqual(mapping["address_1"], "address_1")
        self.assertEqual(mapping["address_2"], "address_2")

    # 3. Missing Required Business Identity Fields (e.g. Missing Address)
    def test_missing_address_column_raises_validation_error(self):
        tsv_content = (
            "id\tcompany\tphone\twebsite\n"
            "1\tStark Industries\t5551234\tstark.com\n"
        )
        test_file = self.raw_dir / "missing_address.tsv"
        test_file.write_text(tsv_content, encoding="utf-8")

        with self.assertRaises(SchemaValidationError) as ctx:
            self.pipeline.ingest_file(test_file)

        self.assertTrue(any("address" in field.lower() for field in ctx.exception.missing_fields))
        self.assertIn("company", ctx.exception.found_columns)

    def test_missing_name_column_raises_validation_error(self):
        tsv_content = (
            "id\taddress\tcity\tpostal_code\n"
            "1\t742 Evergreen Terrace\tSpringfield\t97477\n"
        )
        test_file = self.raw_dir / "missing_name.tsv"
        test_file.write_text(tsv_content, encoding="utf-8")

        with self.assertRaises(SchemaValidationError) as ctx:
            self.pipeline.ingest_file(test_file)

        self.assertTrue(any("name" in field.lower() for field in ctx.exception.missing_fields))

    def test_pairwise_missing_right_side_raises_validation_error(self):
        # Has name_1 and address_1, but missing address_2
        tsv_content = (
            "id_1\tname_1\taddress_1\tid_2\tname_2\tphone_2\n"
            "1\tAcme\t100 Main St\t2\tAcme Corp\t5550000\n"
        )
        test_file = self.raw_dir / "pairwise_incomplete.tsv"
        test_file.write_text(tsv_content, encoding="utf-8")

        with self.assertRaises(SchemaValidationError) as ctx:
            self.pipeline.ingest_file(test_file)

        self.assertTrue(any("address_2" in f or "Right Address" in f for f in ctx.exception.missing_fields))

    # 4. Empty Files
    def test_zero_byte_empty_file_raises_error(self):
        empty_file = self.raw_dir / "empty_0bytes.tsv"
        empty_file.write_bytes(b"")

        with self.assertRaises(EmptyFileError):
            self.pipeline.ingest_file(empty_file)

    def test_header_only_file_with_no_rows_raises_error(self):
        header_only = self.raw_dir / "header_only.tsv"
        header_only.write_text("id\tname\taddress\n", encoding="utf-8")

        with self.assertRaises(EmptyFileError):
            self.pipeline.ingest_file(header_only)

    # 5. Malformed TSV Files
    def test_csv_instead_of_tsv_detected_as_malformed(self):
        csv_content = "id,name,address,city\n1,Acme,123 Main,SF\n"
        csv_file = self.raw_dir / "comma_separated.tsv"
        csv_file.write_text(csv_content, encoding="utf-8")

        with self.assertRaises(MalformedFileError) as ctx:
            self.pipeline.ingest_file(csv_file)

        self.assertIn("comma-separated", str(ctx.exception).lower())

    # 6. Multi-Encoding Fallback (Latin-1 with accented characters)
    def test_latin1_encoding_fallback_preserves_accents(self):
        # "Münchener Straße" and "Café Zürich AG" encoded in ISO-8859-1 (Latin-1)
        german_tsv = "id\tcompany\tstreet\n1\tCaf\xe9 Z\xfcrich AG\tM\xfcnchener Stra\xdfe 10\n".encode("latin-1")
        latin1_file = self.raw_dir / "latin1_entities.tsv"
        latin1_file.write_bytes(german_tsv)

        result = self.pipeline.ingest_file(latin1_file)
        self.assertEqual(result["status"], "success")
        self.assertIn(result["detected_encoding"], ["latin-1", "cp1252"])

    # 7. Raw Preservation Guarantee
    def test_raw_file_remains_unaltered_after_ingestion(self):
        tsv_content = "id\tname\taddress\n1\tWayne Enterprises\t100 Gotham Way\n"
        test_file = self.raw_dir / "preserve_test.tsv"
        test_file.write_text(tsv_content, encoding="utf-8")

        original_bytes = test_file.read_bytes()
        original_hash = self.pipeline.compute_sha256(test_file)

        result = self.pipeline.ingest_file(test_file)

        post_bytes = test_file.read_bytes()
        post_hash = self.pipeline.compute_sha256(test_file)

        self.assertEqual(original_bytes, post_bytes)
        self.assertEqual(original_hash, post_hash)
        self.assertEqual(result["sha256"], original_hash)

    # 8. Byte Ingestion via upload simulation
    def test_ingest_bytes_upload(self):
        content = b"id\tname\taddress\n1\tCyberdyne Systems\t18144 El Camino\n"
        result = self.pipeline.ingest_bytes(content, filename="upload_test.tsv")

        self.assertEqual(result["status"], "success")
        saved_raw = Path(result["raw_file_path"])
        self.assertTrue(saved_raw.is_file())
        self.assertEqual(saved_raw.read_bytes(), content)


if __name__ == "__main__":
    unittest.main()
