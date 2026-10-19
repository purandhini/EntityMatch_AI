"""Robust business entity text preprocessing and normalization module."""

import re
import string
import unicodedata
from typing import Any, Dict, List, Optional
import pandas as pd


class EntityPreprocessor:
    """
    Standardizes business entity names, addresses, phones, and websites.

    Performs Unicode normalization, punctuation hygiene, whitespace collapsing,
    legal suffix canonicalization, and street abbreviation standardization
    while preserving original columns untouched.
    """

    # Direct character replacements for ligatures and special glyphs.
    # Applied BEFORE NFKD decomposition so that ligatures are expanded correctly.
    UNICODE_CHAR_REPLACEMENTS = {
        "\u00df": "ss",  # ß (eszett / sharp-s)
        "\u00e6": "ae",  # æ
        "\u0153": "oe",  # œ
        "\u00f8": "o",   # ø
        "\u00f0": "d",   # ð
        "\u00fe": "th",  # þ
        "\u00c6": "ae",  # Æ (uppercase)
        "\u0152": "oe",  # Œ (uppercase)
        "\u00d8": "o",   # Ø (uppercase)
        "\u00de": "th",  # Þ (uppercase)
        "\u00df": "ss",  # ß (repeated for safety)
        "\u00a0": " ",   # Non-breaking space
        "\u200b": " ",   # Zero-width space
        "\u202f": " ",   # Narrow no-break space
        "\u2009": " ",   # Thin space
        "\u2013": "-",   # En dash
        "\u2014": "-",   # Em dash
        "\u2212": "-",   # Minus sign
        "\u2010": "-",   # Hyphen
        "\u2011": "-",   # Non-breaking hyphen
        "\u2018": "'",   # Left single quotation mark
        "\u2019": "'",   # Right single quotation mark
        "\u201c": '"',   # Left double quotation mark
        "\u201d": '"',   # Right double quotation mark
        "\u00ab": '"',   # «
        "\u00bb": '"',   # »
    }

    # Ordered legal suffix canonicalization rules.
    # IMPORTANT: multi-word compound patterns MUST come before their component words,
    # otherwise the single-word patterns consume the tokens first.
    # Each entry is (raw_regex_string, canonical_replacement).
    # Regexes are matched against the *lowercased + punctuation-normalised* text
    # where dots are already spaces, so we additionally list the de-dotted form.
    LEGAL_SUFFIX_MAP_ORDERED = [
        # --- Multi-word compounds first ---
        (r"\bpublic limited company\b", "plc"),
        (r"\bpublic limited co\b", "plc"),
        (r"\blimited liability company\b", "llc"),
        (r"\blimited liability co\b", "llc"),
        (r"\bproprietary limited\b", "pty ltd"),
        (r"\bproprietary ltd\b", "pty ltd"),
        (r"\bpty limited\b", "pty ltd"),
        (r"\bpty ltd\b", "pty ltd"),
        (r"\bpty\b", "pty"),
        (r"\bsociedad anonima\b", "sa"),
        (r"\bsociete anonyme\b", "sa"),
        (r"\bges m b h\b", "gmbh"),   # de-dotted form of Ges.m.b.H.
        # --- Dotted abbreviations (de-dotted form after normalize_text) ---
        # L.L.C. -> "l l c"  after punctuation stripping
        (r"\bl l c\b", "llc"),
        # P.L.C. -> "p l c"
        (r"\bp l c\b", "plc"),
        # S.A.S. -> "s a s"
        (r"\bs a s\b", "sas"),
        # S.A. -> "s a"
        (r"\bs a\b", "sa"),
        # B.V. -> "b v"
        (r"\bb v\b", "bv"),
        # N.V. -> "n v"
        (r"\bn v\b", "nv"),
        # A.G. -> "a g"
        (r"\ba g\b", "ag"),
        # Ges.m.b.H -> handled above
        # --- Single-word suffixes ---
        (r"\bcorporation\b", "corp"),
        (r"\bcorp\b", "corp"),
        (r"\bincorporated\b", "inc"),
        (r"\binc\b", "inc"),
        (r"\bllc\b", "llc"),
        (r"\bltd\b", "ltd"),
        (r"\blimited\b", "ltd"),
        (r"\bcompany\b", "co"),
        (r"\bplc\b", "plc"),
        (r"\bgmbh\b", "gmbh"),
        (r"\bag\b", "ag"),
        (r"\bsas\b", "sas"),
        (r"\b\bsa\b\b", "sa"),
        (r"\bbv\b", "bv"),
        (r"\bnv\b", "nv"),
    ]

    # Patterns to completely strip legal suffixes when requested.
    # Again: multi-word patterns before single-word ones to avoid partial matches.
    LEGAL_SUFFIX_STRIP_PATTERNS = [
        r"\bpublic limited company\b",
        r"\bpublic limited co\b",
        r"\blimited liability company\b",
        r"\blimited liability co\b",
        r"\bproprietary limited\b",
        r"\bproprietary ltd\b",
        r"\bpty limited\b",
        r"\bpty ltd\b",
        r"\bpty\b",
        r"\bges m b h\b",
        r"\bl l c\b",
        r"\bp l c\b",
        r"\bs a s\b",
        r"\bs a\b",
        r"\bb v\b",
        r"\bn v\b",
        r"\ba g\b",
        r"\bcorporation\b",
        r"\bcorp\b",
        r"\bincorporated\b",
        r"\binc\b",
        r"\bllc\b",
        r"\bltd\b",
        r"\blimited\b",
        r"\bcompany\b",
        r"\bco\b",
        r"\bplc\b",
        r"\bgmbh\b",
        r"\bag\b",
        r"\bsas\b",
        r"\bsa\b",
        r"\bbv\b",
        r"\bnv\b",
        r"\bholdings\b",
        r"\bholding\b",
        r"\bgroup\b",
        r"\bservices\b",
        r"\benterprises\b",
        r"\binternational\b",
        r"\bintl\b",
    ]

    # Common street address abbreviations.
    # The '#' pattern is handled separately in clean_address() BEFORE
    # normalize_text() strips the '#' symbol.
    ADDRESS_ABBREVIATIONS = {
        # Thoroughfare types
        r"\bstreet\b": "st",
        r"\bavenue\b": "ave",
        r"\bboulevard\b": "blvd",
        r"\broad\b": "rd",
        r"\bdrive\b": "dr",
        r"\blane\b": "ln",
        r"\bcourt\b": "ct",
        r"\bcircle\b": "cir",
        r"\bplace\b": "pl",
        r"\bsquare\b": "sq",
        r"\bterrace\b": "ter",
        r"\bparkway\b": "pkwy",
        r"\bhighway\b": "hwy",
        r"\bexpressway\b": "expy",
        r"\bturnpike\b": "tpke",
        r"\btrail\b": "trl",
        # Occupancy / secondary units
        r"\bsuite\b": "ste",
        r"\bapartment\b": "apt",
        r"\bbuilding\b": "bldg",
        r"\bfloor\b": "fl",
        r"\broom\b": "rm",
        r"\bdepartment\b": "dept",
        # Cardinal directions
        r"\bnorth\b": "n",
        r"\bsouth\b": "s",
        r"\beast\b": "e",
        r"\bwest\b": "w",
        r"\bnortheast\b": "ne",
        r"\bnorthwest\b": "nw",
        r"\bsoutheast\b": "se",
        r"\bsouthwest\b": "sw",
    }

    def __init__(self, strip_legal_suffixes: bool = False):
        self.strip_legal_suffixes = strip_legal_suffixes

        # Precompile ordered suffix canonicalization patterns
        self._compiled_suffix_map = [
            (re.compile(pat, re.IGNORECASE), repl)
            for pat, repl in self.LEGAL_SUFFIX_MAP_ORDERED
        ]
        # Precompile strip patterns
        self._compiled_suffix_strip = [
            re.compile(pat, re.IGNORECASE)
            for pat in self.LEGAL_SUFFIX_STRIP_PATTERNS
        ]
        # Precompile address abbreviation patterns
        self._compiled_addr_regex = [
            (re.compile(pat, re.IGNORECASE), repl)
            for pat, repl in self.ADDRESS_ABBREVIATIONS.items()
        ]
        # Pattern used to convert # <unit> -> ste <unit> in raw address strings
        self._hash_unit_re = re.compile(r"#\s*", re.IGNORECASE)

    # ------------------------------------------------------------------
    # Core Unicode normalization
    # ------------------------------------------------------------------

    def normalize_unicode(self, text: str) -> str:
        """
        Expands ligatures/special characters, then decomposes and strips
        diacritics/accents via NFKD normalization.
        """
        if not text:
            return ""

        # 1. Apply direct replacements (ligatures, special whitespace, dashes, quotes)
        for char, replacement in self.UNICODE_CHAR_REPLACEMENTS.items():
            if char in text:
                text = text.replace(char, replacement)

        # 2. NFKD decomposition to remove combining marks (é -> e, ü -> u, ñ -> n …)
        decomposed = unicodedata.normalize("NFKD", text)
        stripped = "".join(c for c in decomposed if not unicodedata.combining(c))
        return stripped

    def normalize_text(self, text: Optional[str]) -> str:
        """
        General text cleaning pipeline:
          1. Unicode normalization and diacritic removal
          2. Lowercase
          3. Ampersand expansion  (& -> and)
          4. Punctuation -> space
          5. Whitespace collapsing
        """
        if text is None:
            return ""
        # Guard against pandas NA / NaN
        try:
            if pd.isna(text):
                return ""
        except (TypeError, ValueError):
            pass

        raw_str = str(text)
        if not raw_str.strip():
            return ""

        # 1. Unicode decomposition
        norm = self.normalize_unicode(raw_str)

        # 2. Lowercase
        norm = norm.lower()

        # 3. Ampersand -> " and " (before punctuation removal so "&" is still present)
        norm = re.sub(r"\s*&\s*", " and ", norm)

        # 4. Replace punctuation with space (prevents token fusion e.g. "Apple,Inc." -> "apple inc")
        norm = re.sub(r"[" + re.escape(string.punctuation) + r"]", " ", norm)

        # 5. Collapse whitespace
        norm = re.sub(r"\s+", " ", norm).strip()

        return norm

    # ------------------------------------------------------------------
    # Field-specific cleaners
    # ------------------------------------------------------------------

    def clean_company_name(self, name: Optional[str]) -> str:
        """
        Normalizes company names.

        Pipeline:
          1. normalize_text()   – unicode, lowercase, ampersand, punctuation, spaces
          2a. If strip_legal_suffixes=False: canonicalize each suffix (e.g. "Corporation" -> "corp").
          2b. If strip_legal_suffixes=True:  remove all legal suffix tokens.
          3. Final whitespace collapse.

        Note: dotted abbreviations (L.L.C., S.A.S., A.G., etc.) are handled because
        after normalize_text() the dots become spaces, producing the de-dotted form
        (e.g. "L.L.C." -> "l l c"), and the suffix patterns explicitly match those
        spaced forms before matching individual single-letter tokens.
        """
        cleaned = self.normalize_text(name)
        if not cleaned:
            return ""

        if self.strip_legal_suffixes:
            for regex in self._compiled_suffix_strip:
                cleaned = regex.sub(" ", cleaned)
        else:
            for regex, repl in self._compiled_suffix_map:
                cleaned = regex.sub(repl, cleaned)

        return re.sub(r"\s+", " ", cleaned).strip()

    def clean_address(self, address: Optional[str]) -> str:
        """
        Normalizes street addresses.

        Pipeline:
          1. Convert '#<number>' to 'ste <number>' BEFORE punctuation stripping.
          2. normalize_text()
          3. Apply street-type, unit-designator, and directional abbreviations.
          4. Final whitespace collapse.
        """
        if address is None:
            return ""
        try:
            if pd.isna(address):
                return ""
        except (TypeError, ValueError):
            pass

        raw = str(address)
        if not raw.strip():
            return ""

        # Step 1: Replace '#<unit>' with 'ste <unit>' before punctuation stripping
        raw = self._hash_unit_re.sub("ste ", raw)

        # Step 2: General text normalization
        cleaned = self.normalize_text(raw)

        # Step 3: Apply address abbreviations
        for regex, replacement in self._compiled_addr_regex:
            cleaned = regex.sub(replacement, cleaned)

        return re.sub(r"\s+", " ", cleaned).strip()

    def clean_phone(self, phone: Optional[str]) -> str:
        """
        Extracts numeric digits for invariant comparison.
        Strips US country code '1' when the result would be 11 digits starting with 1.
        """
        if phone is None:
            return ""
        try:
            if pd.isna(phone):
                return ""
        except (TypeError, ValueError):
            pass
        digits = re.sub(r"\D", "", str(phone))
        if len(digits) == 11 and digits.startswith("1"):
            digits = digits[1:]
        return digits

    def clean_website(self, url: Optional[str]) -> str:
        """
        Canonicalizes website URLs to bare domain:
          - Strips protocol (http://, https://)
          - Strips www / www2 prefix
          - Strips port, path, query, and fragment
        """
        if url is None:
            return ""
        try:
            if pd.isna(url):
                return ""
        except (TypeError, ValueError):
            pass
        u = str(url).lower().strip()
        u = re.sub(r"^https?://", "", u)
        u = re.sub(r"^www\d*\.", "", u)
        u = u.split("?")[0].split("#")[0]
        u = u.split("/")[0]
        u = u.split(":")[0]
        return u.strip()

    def clean_postal_code(self, postal_code: Optional[str]) -> str:
        """Cleans postal codes: US 5-digit ZIP or alphanumeric international codes."""
        if postal_code is None:
            return ""
        try:
            if pd.isna(postal_code):
                return ""
        except (TypeError, ValueError):
            pass
        code = str(postal_code).strip().upper()
        match = re.match(r"^(\d{5})(-\d{4})?$", code)
        if match:
            return match.group(1)
        return re.sub(r"[^A-Z0-9]", "", code)

    # ------------------------------------------------------------------
    # Record and DataFrame interfaces
    # ------------------------------------------------------------------

    def preprocess_record(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """
        Preprocesses a single entity record dict.
        Returns a new dict with all original keys preserved and '_clean' variants added.
        """
        processed = dict(record)  # shallow copy preserves original values

        for key, value in record.items():
            k_lower = key.lower()
            if "name" in k_lower or "company" in k_lower or "firm" in k_lower:
                processed[f"{key}_clean"] = self.clean_company_name(value)
            elif "address" in k_lower or "street" in k_lower or "location" in k_lower:
                processed[f"{key}_clean"] = self.clean_address(value)
            elif "phone" in k_lower or "tel" in k_lower:
                processed[f"{key}_clean"] = self.clean_phone(value)
            elif "web" in k_lower or "url" in k_lower:
                processed[f"{key}_clean"] = self.clean_website(value)
            elif "zip" in k_lower or "postal" in k_lower:
                processed[f"{key}_clean"] = self.clean_postal_code(value)
            elif "city" in k_lower or "state" in k_lower:
                processed[f"{key}_clean"] = self.normalize_text(value)

        return processed

    def preprocess_dataframe(
        self,
        df: pd.DataFrame,
        name_cols: Optional[List[str]] = None,
        address_cols: Optional[List[str]] = None,
        phone_cols: Optional[List[str]] = None,
        website_cols: Optional[List[str]] = None,
        postal_cols: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Returns a copy of *df* with original columns untouched and new '_clean' columns appended.
        Does not modify the input DataFrame.
        """
        df_out = df.copy()

        for col in (name_cols or []):
            if col in df_out.columns:
                df_out[f"{col}_clean"] = df_out[col].apply(self.clean_company_name)

        for col in (address_cols or []):
            if col in df_out.columns:
                df_out[f"{col}_clean"] = df_out[col].apply(self.clean_address)

        for col in (phone_cols or []):
            if col in df_out.columns:
                df_out[f"{col}_clean"] = df_out[col].apply(self.clean_phone)

        for col in (website_cols or []):
            if col in df_out.columns:
                df_out[f"{col}_clean"] = df_out[col].apply(self.clean_website)

        for col in (postal_cols or []):
            if col in df_out.columns:
                df_out[f"{col}_clean"] = df_out[col].apply(self.clean_postal_code)

        return df_out
