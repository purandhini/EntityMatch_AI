"""Unit tests for robust business entity text preprocessing and normalization."""

import sys
import unittest
from pathlib import Path
import pandas as pd

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.pipeline.preprocessing import EntityPreprocessor


class TestEntityPreprocessing(unittest.TestCase):
    """Test suite covering Unicode, legal suffixes, addresses, punctuation, and DataFrame preservation."""

    def setUp(self):
        self.preprocessor_canonical = EntityPreprocessor(strip_legal_suffixes=False)
        self.preprocessor_stripped = EntityPreprocessor(strip_legal_suffixes=True)

    # 1. Unicode & Diacritic Normalization
    def test_unicode_accents_and_diacritics(self):
        # French accents
        cleaned = self.preprocessor_canonical.clean_company_name("Café des Artistes")
        self.assertEqual(cleaned, "cafe des artistes")

        # German umlauts and eszett
        cleaned_de = self.preprocessor_canonical.clean_company_name("Großmann & Söhne")
        self.assertEqual(cleaned_de, "grossmann and sohne")

        # Spanish tilde
        cleaned_es = self.preprocessor_canonical.clean_company_name("Peña & Compañía")
        self.assertEqual(cleaned_es, "pena and compania")

        # Scandinavian characters
        cleaned_no = self.preprocessor_canonical.clean_company_name("Møller & Søn A/S")
        self.assertIn("moller and son", cleaned_no)

    def test_unicode_special_glyphs_and_whitespace(self):
        # Non-breaking space, zero-width space, typographic quotes, em-dash
        raw_text = "“Acme\u00a0Technologies”—‘Global\u202fHoldings’"
        cleaned = self.preprocessor_canonical.normalize_text(raw_text)
        self.assertEqual(cleaned, "acme technologies global holdings")

    # 2. Punctuation & Whitespace Normalization
    def test_punctuation_and_ampersand(self):
        # Ampersand expansion
        cleaned_amp = self.preprocessor_canonical.clean_company_name("Johnson & Johnson")
        self.assertEqual(cleaned_amp, "johnson and johnson")

        # Punctuation does not fuse words together
        cleaned_punct = self.preprocessor_canonical.clean_company_name("Apple,Inc.")
        self.assertEqual(cleaned_punct, "apple inc")

        # Slashes and parentheses
        cleaned_slash = self.preprocessor_canonical.clean_company_name("AT&T (WarnerMedia / Discovery)")
        self.assertEqual(cleaned_slash, "at and t warnermedia discovery")

    def test_whitespace_collapsing(self):
        messy_space = "   Federal   Express   Corporation   \t\n  "
        cleaned = self.preprocessor_canonical.clean_company_name(messy_space)
        self.assertEqual(cleaned, "federal express corp")

    # 3. Business Legal Suffixes
    def test_legal_suffixes_canonicalization(self):
        cases = [
            ("Apex Corporation", "apex corp"),
            ("Apex Incorporated", "apex inc"),
            ("Apex Limited Liability Company", "apex llc"),
            ("Apex L.L.C.", "apex llc"),
            ("Apex Limited", "apex ltd"),
            ("Apex Company", "apex co"),
            ("Apex Public Limited Company", "apex plc"),
            ("Apex GmbH", "apex gmbh"),
            ("Apex AG", "apex ag"),
            ("Apex Proprietary Limited", "apex pty ltd"),
            ("Apex S.A.S.", "apex sas"),
        ]
        for raw, expected in cases:
            with self.subTest(raw=raw):
                self.assertEqual(self.preprocessor_canonical.clean_company_name(raw), expected)

    def test_legal_suffixes_stripping(self):
        cases = [
            ("Apex Corporation", "apex"),
            ("Apex Inc.", "apex"),
            ("Apex LLC", "apex"),
            ("Apex Limited", "apex"),
            ("Apex GmbH", "apex"),
            ("Apex AG", "apex"),
            ("Apex Pty Ltd", "apex"),
        ]
        for raw, expected in cases:
            with self.subTest(raw=raw):
                self.assertEqual(self.preprocessor_stripped.clean_company_name(raw), expected)

    # 4. Address Abbreviations
    def test_address_street_abbreviations(self):
        cases = [
            ("100 Main Street", "100 main st"),
            ("200 Fifth Avenue", "200 fifth ave"),
            ("300 Sunset Boulevard", "300 sunset blvd"),
            ("400 Mountain Road", "400 mountain rd"),
            ("500 Ocean Drive", "500 ocean dr"),
            ("600 Country Lane", "600 country ln"),
            ("700 Rolling Court", "700 rolling ct"),
            ("800 Research Parkway", "800 research pkwy"),
            ("900 Interstate Highway", "900 interstate hwy"),
        ]
        for raw, expected in cases:
            with self.subTest(raw=raw):
                self.assertEqual(self.preprocessor_canonical.clean_address(raw), expected)

    def test_address_secondary_units_and_directions(self):
        addr = "100 North Michigan Avenue, Suite 400, Building B"
        cleaned = self.preprocessor_canonical.clean_address(addr)
        self.assertEqual(cleaned, "100 n michigan ave ste 400 bldg b")

        addr_hash = "500 South West Way # 12B"
        cleaned_hash = self.preprocessor_canonical.clean_address(addr_hash)
        self.assertEqual(cleaned_hash, "500 s w way ste 12b")

    # 5. Phone and URL Normalization
    def test_phone_normalization(self):
        self.assertEqual(self.preprocessor_canonical.clean_phone("+1 (800) 555-0199"), "8005550199")
        self.assertEqual(self.preprocessor_canonical.clean_phone("800.555.0199"), "8005550199")
        self.assertEqual(self.preprocessor_canonical.clean_phone("020 7946 0991"), "02079460991")

    def test_website_canonicalization(self):
        self.assertEqual(
            self.preprocessor_canonical.clean_website("https://www.google.com/search?q=test#top"),
            "google.com",
        )
        self.assertEqual(
            self.preprocessor_canonical.clean_website("http://sub.domain.co.uk:8080/path/to/page"),
            "sub.domain.co.uk",
        )

    # 6. Column Preservation in DataFrames
    def test_dataframe_preserves_original_columns(self):
        raw_df = pd.DataFrame([
            {
                "id": "1",
                "name": "Café Müller GmbH",
                "address": "100 North Main Street, Suite 5",
                "phone": "+1 (800) 555-1234",
                "website": "https://www.mueller-cafe.de/home",
            }
        ])

        # Run preprocessing
        processed_df = self.preprocessor_canonical.preprocess_dataframe(
            raw_df,
            name_cols=["name"],
            address_cols=["address"],
            phone_cols=["phone"],
            website_cols=["website"],
        )

        # Original columns must be 100% identical and preserved
        self.assertEqual(processed_df["name"].iloc[0], "Café Müller GmbH")
        self.assertEqual(processed_df["address"].iloc[0], "100 North Main Street, Suite 5")
        self.assertEqual(processed_df["phone"].iloc[0], "+1 (800) 555-1234")
        self.assertEqual(processed_df["website"].iloc[0], "https://www.mueller-cafe.de/home")

        # New normalized columns must exist and match expected values
        self.assertEqual(processed_df["name_clean"].iloc[0], "cafe muller gmbh")
        self.assertEqual(processed_df["address_clean"].iloc[0], "100 n main st ste 5")
        self.assertEqual(processed_df["phone_clean"].iloc[0], "8005551234")
        self.assertEqual(processed_df["website_clean"].iloc[0], "mueller-cafe.de")

    # 7. Single Record Preservation
    def test_preprocess_record_preservation(self):
        rec = {
            "company_name": "Wayne Enterprises, Inc.",
            "street": "100 Gotham Boulevard",
        }
        res = self.preprocessor_canonical.preprocess_record(rec)
        # Original preserved
        self.assertEqual(res["company_name"], "Wayne Enterprises, Inc.")
        self.assertEqual(res["street"], "100 Gotham Boulevard")
        # Clean added
        self.assertEqual(res["company_name_clean"], "wayne enterprises inc")
        self.assertEqual(res["street_clean"], "100 gotham blvd")


if __name__ == "__main__":
    unittest.main()
