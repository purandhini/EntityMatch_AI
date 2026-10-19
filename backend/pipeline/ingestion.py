"""Data ingestion, schema detection, and validation layer for TSV challenge files."""

import csv
import hashlib
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import pandas as pd

from ..config import DEFAULT_CONFIG
from .preprocessing import EntityPreprocessor


class IngestionError(Exception):
    """Base exception for data ingestion failures."""
    pass


class EmptyFileError(IngestionError):
    """Raised when an uploaded or specified file has 0 bytes or no data rows."""
    pass


class MalformedFileError(IngestionError):
    """Raised when a file cannot be parsed as tabular TSV data."""
    pass


class SchemaValidationError(IngestionError):
    """Raised when mandatory business identity fields are missing from TSV headers."""

    def __init__(self, message: str, missing_fields: List[str], found_columns: List[str]):
        super().__init__(message)
        self.missing_fields = missing_fields
        self.found_columns = found_columns


# Canonical field aliases commonly found in entity resolution challenge datasets
FIELD_ALIASES: Dict[str, List[str]] = {
    "name": [
        "name", "company", "company_name", "business_name", "entity_name",
        "legal_name", "firm", "vendor", "account_name", "org_name", "organization"
    ],
    "address": [
        "address", "street", "street_address", "addr", "location",
        "address_line_1", "address_line1", "street_1", "address1",
        "business_address",
    ],
    "country": [
        "country", "country_code", "nation", "country_name",
    ],
    "city": ["city", "municipality", "town", "locality"],
    "state": ["state", "province", "region", "administrative_area"],
    "postal_code": ["postal_code", "zip", "zip_code", "postcode", "postal"],
    "phone": ["phone", "telephone", "tel", "phone_number", "contact_number", "mobile"],
    "website": ["website", "url", "web", "domain", "homepage", "web_url"],
    "id": ["id", "record_id", "entity_id", "uid", "uuid", "key"],
    "label": ["label", "is_match", "match", "target", "ground_truth", "class"],
}


@dataclass
class DetectedSchema:
    """Metadata describing the detected schema and column alignments."""
    dataset_format: str  # "PAIRWISE" or "SINGLE_TABLE"
    column_mapping: Dict[str, str] = field(default_factory=dict)
    all_columns: List[str] = field(default_factory=list)
    pair_suffixes: Optional[Tuple[str, str]] = None
    warnings: List[str] = field(default_factory=list)


class SchemaValidator:
    """Inspects headers, auto-detects column semantics, and validates business identity fields."""

    PAIRWISE_SUFFIX_PAIRS = [
        ("_1", "_2"),
        ("_left", "_right"),
        ("_a", "_b"),
        ("1", "2"),
    ]

    def inspect_headers(self, columns: List[str]) -> DetectedSchema:
        """
        Analyzes a list of column names, determines format, and maps aliases.
        """
        clean_cols = [str(c).strip() for c in columns]
        lower_cols = [c.lower() for c in clean_cols]
        col_lookup = dict(zip(lower_cols, clean_cols))

        # Check for pairwise column conventions
        detected_suffix: Optional[Tuple[str, str]] = None
        for s1, s2 in self.PAIRWISE_SUFFIX_PAIRS:
            s1_matches = [c for c in lower_cols if c.endswith(s1)]
            s2_matches = [c for c in lower_cols if c.endswith(s2)]
            if len(s1_matches) >= 1 and len(s2_matches) >= 1:
                detected_suffix = (s1, s2)
                break

        if detected_suffix:
            return self._inspect_pairwise_schema(clean_cols, lower_cols, col_lookup, detected_suffix)
        else:
            return self._inspect_single_table_schema(clean_cols, lower_cols, col_lookup)

    def _inspect_single_table_schema(
        self,
        clean_cols: List[str],
        lower_cols: List[str],
        col_lookup: Dict[str, str],
    ) -> DetectedSchema:
        mapping: Dict[str, str] = {}
        for canonical, aliases in FIELD_ALIASES.items():
            for alias in aliases:
                if alias in col_lookup:
                    mapping[canonical] = col_lookup[alias]
                    break

        return DetectedSchema(
            dataset_format="SINGLE_TABLE",
            column_mapping=mapping,
            all_columns=clean_cols,
        )

    def _inspect_pairwise_schema(
        self,
        clean_cols: List[str],
        lower_cols: List[str],
        col_lookup: Dict[str, str],
        suffixes: Tuple[str, str],
    ) -> DetectedSchema:
        s1, s2 = suffixes
        mapping: Dict[str, str] = {}

        for canonical, aliases in FIELD_ALIASES.items():
            # Check left attribute
            for alias in aliases:
                key = f"{alias}{s1}"
                if key in col_lookup:
                    mapping[f"{canonical}{s1}"] = col_lookup[key]
                    break
            # Check right attribute
            for alias in aliases:
                key = f"{alias}{s2}"
                if key in col_lookup:
                    mapping[f"{canonical}{s2}"] = col_lookup[key]
                    break

        # Check for non-suffixed label or id columns
        for special_field in ["label", "id"]:
            for alias in FIELD_ALIASES[special_field]:
                if alias in col_lookup and alias not in mapping:
                    mapping[special_field] = col_lookup[alias]
                    break

        return DetectedSchema(
            dataset_format="PAIRWISE",
            column_mapping=mapping,
            all_columns=clean_cols,
            pair_suffixes=suffixes,
        )

    def validate_schema(self, schema: DetectedSchema) -> None:
        """
        Validates presence of required business identity fields:
        - At minimum, Business Name and Address must be resolved.
        - For pairwise files, both left and right sides must have Name and Address.

        Raises:
            SchemaValidationError: If required fields cannot be found.
        """
        missing_requirements: List[str] = []

        if schema.dataset_format == "SINGLE_TABLE":
            if "name" not in schema.column_mapping:
                missing_requirements.append("Business Name (aliases: name, company, company_name, business_name, legal_name)")
            if "address" not in schema.column_mapping:
                missing_requirements.append("Address (aliases: address, street, street_address, addr, location)")
        else:
            s1, s2 = schema.pair_suffixes  # type: ignore
            if f"name{s1}" not in schema.column_mapping:
                missing_requirements.append(f"Left Business Name (e.g. name{s1}, company{s1})")
            if f"name{s2}" not in schema.column_mapping:
                missing_requirements.append(f"Right Business Name (e.g. name{s2}, company{s2})")
            if f"address{s1}" not in schema.column_mapping:
                missing_requirements.append(f"Left Address (e.g. address{s1}, street{s1})")
            if f"address{s2}" not in schema.column_mapping:
                missing_requirements.append(f"Right Address (e.g. address{s2}, street{s2})")

        if missing_requirements:
            msg = (
                f"Schema validation failed: Missing required business identity fields:\n"
                f"  - " + "\n  - ".join(missing_requirements) + "\n"
                f"Found columns in TSV: {schema.all_columns}\n"
                f"Detected mappings: {schema.column_mapping}"
            )
            raise SchemaValidationError(
                message=msg,
                missing_fields=missing_requirements,
                found_columns=schema.all_columns,
            )


def canonicalize_pairwise_dataframe(df: pd.DataFrame, schema: Optional[DetectedSchema] = None) -> pd.DataFrame:
    """Return a pairwise dataframe using the pipeline's canonical column names.

    The source dataframe is not modified. Arbitrary detected pairwise aliases such as
    ``company_left``/``company_right`` are mapped to canonical names such as
    ``name_1``/``name_2`` so preprocessing and feature extraction use the same schema
    during training, evaluation, and prediction.
    """
    validator = SchemaValidator()
    detected = schema or validator.inspect_headers(list(df.columns))
    if detected.dataset_format != "PAIRWISE" or not detected.pair_suffixes:
        raise SchemaValidationError(
            "Pairwise pipeline requires a pairwise TSV with left/right identity columns.",
            missing_fields=["pairwise schema"],
            found_columns=list(df.columns),
        )
    validator.validate_schema(detected)

    s1, s2 = detected.pair_suffixes
    out = df.copy()

    # Canonicalize the fields consumed by PairwiseFeatureExtractor.
    for canonical in ["name", "address", "city", "state", "postal_code", "phone", "website"]:
        for source_suffix, target_suffix in ((s1, "_1"), (s2, "_2")):
            source = detected.column_mapping.get(f"{canonical}{source_suffix}")
            if source and source != f"{canonical}{target_suffix}":
                out[f"{canonical}{target_suffix}"] = out[source]

    # Keep the detected label available under its canonical name.
    if "label" in detected.column_mapping:
        source = detected.column_mapping["label"]
        if source != "label":
            out["label"] = out[source]

    # Canonicalize pair identifiers when the source schema provides them.
    for source_suffix, target_suffix in ((s1, "_1"), (s2, "_2")):
        source = detected.column_mapping.get(f"id{source_suffix}")
        if source:
            out[f"id{target_suffix}"] = out[source]

    return out


class DataIngestionPipeline:
    """
    Coordinates ingestion of raw TSV files:
    - Multi-encoding reader fallback (utf-8, utf-8-sig, cp1252, latin-1)
    - Raw preservation in data/raw/
    - Schema inspection & identity validation
    - Preprocessing integration
    """

    SUPPORTED_ENCODINGS = ["utf-8", "utf-8-sig", "cp1252", "latin-1"]

    def __init__(
        self,
        raw_dir: Optional[Path] = None,
        processed_dir: Optional[Path] = None,
    ):
        self.raw_dir = Path(raw_dir or DEFAULT_CONFIG.raw_data_dir)
        self.processed_dir = Path(processed_dir or DEFAULT_CONFIG.processed_data_dir)
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)

        self.validator = SchemaValidator()
        self.preprocessor = EntityPreprocessor(strip_legal_suffixes=True)

    @staticmethod
    def compute_sha256(file_path: Path) -> str:
        """Calculates SHA256 checksum of raw file for provenance tracking."""
        hasher = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                hasher.update(chunk)
        return hasher.hexdigest()

    def read_tsv_with_fallback(
        self, file_path: Path, sep: str = "\t"
    ) -> Tuple[pd.DataFrame, str]:
        """
        Reads TSV with fallback through supported encodings and validates non-emptiness.
        """
        if not file_path.is_file():
            raise FileNotFoundError(f"File not found: {file_path.resolve()}")

        size_bytes = file_path.stat().st_size
        if size_bytes == 0:
            raise EmptyFileError(f"TSV file is completely empty (0 bytes): {file_path.name}")

        last_error = None
        for enc in self.SUPPORTED_ENCODINGS:
            try:
                # First check first line for tabs
                with open(file_path, "r", encoding=enc) as f:
                    first_line = f.readline()
                    if not first_line:
                        raise EmptyFileError(f"File contains no data: {file_path.name}")

                df = pd.read_csv(
                    file_path,
                    sep=sep,
                    encoding=enc,
                    dtype=str,
                    on_bad_lines="error",
                    low_memory=False,
                )
                if df.empty or len(df.columns) == 0:
                    raise EmptyFileError(f"TSV contains headers but 0 data records: {file_path.name}")

                # If only 1 column detected and commas or tabs are present, warn if delimiter is suspicious
                if len(df.columns) == 1 and "\t" not in df.columns[0] and "," in df.columns[0]:
                    raise MalformedFileError(
                        f"File '{file_path.name}' appears to be comma-separated rather than tab-separated (TSV)."
                    )

                return df, enc
            except (UnicodeDecodeError, UnicodeError) as e:
                last_error = e
                continue
            except EmptyFileError:
                raise
            except MalformedFileError:
                raise
            except (csv.Error, pd.errors.ParserError) as pe:
                raise MalformedFileError(f"Malformed TSV content in {file_path.name}: {pe}") from pe
            except Exception as ex:
                last_error = ex
                continue

        raise MalformedFileError(
            f"Failed to parse TSV file '{file_path.name}' using supported encodings {self.SUPPORTED_ENCODINGS}. "
            f"Last error: {last_error}"
        )

    def ingest_file(
        self,
        file_path: Union[str, Path],
        copy_to_raw: bool = False,
        save_preprocessed: bool = True,
    ) -> Dict[str, Any]:
        """
        Main ingestion entrypoint for a local TSV file.

        Args:
            file_path: Path to the TSV file.
            copy_to_raw: If True and file is not already in data/raw/, creates a pristine copy.
            save_preprocessed: If True, writes normalized data to data/processed/.

        Returns:
            Dictionary containing schema metadata, validation results, and paths.
        """
        src_path = Path(file_path).resolve()
        if not src_path.is_file():
            raise FileNotFoundError(f"TSV file not found: {src_path}")

        # Ensure raw file is preserved
        target_raw = src_path
        if copy_to_raw and src_path.parent.resolve() != self.raw_dir.resolve():
            target_raw = self.raw_dir / src_path.name
            if target_raw.resolve() != src_path:
                with open(src_path, "rb") as sf, open(target_raw, "wb") as df_raw:
                    df_raw.write(sf.read())

        sha256 = self.compute_sha256(target_raw)

        # 1. Read TSV
        df, detected_enc = self.read_tsv_with_fallback(target_raw)

        # 2. Inspect headers & validate business identity fields
        schema = self.validator.inspect_headers(list(df.columns))
        self.validator.validate_schema(schema)

        # 3. Clean and normalize missing values
        df = df.fillna("").astype(str)

        # 4. Integrate with EntityPreprocessor
        processed_path = None
        df_clean = self._preprocess_by_schema(df, schema)

        if save_preprocessed:
            clean_stem = target_raw.stem
            out_filename = f"{clean_stem}_preprocessed.tsv"
            processed_path = self.processed_dir / out_filename
            df_clean.to_csv(processed_path, sep="\t", index=False)

        return {
            "status": "success",
            "raw_file_path": str(target_raw),
            "file_size_bytes": target_raw.stat().st_size,
            "sha256": sha256,
            "detected_encoding": detected_enc,
            "dataset_format": schema.dataset_format,
            "detected_column_mapping": schema.column_mapping,
            "total_records": len(df),
            "columns": list(df.columns),
            "preprocessed_file_path": str(processed_path) if processed_path else None,
            "warnings": schema.warnings,
        }

    def ingest_bytes(
        self,
        file_bytes: bytes,
        filename: str,
        save_preprocessed: bool = True,
    ) -> Dict[str, Any]:
        """
        Ingests TSV content directly from raw bytes (e.g. web upload).
        Saves raw bytes into data/raw/ unaltered before processing.
        """
        if not file_bytes or len(file_bytes) == 0:
            raise EmptyFileError(f"Uploaded file '{filename}' is empty (0 bytes).")

        # Sanitize filename
        safe_name = os.path.basename(filename).strip() or "uploaded_dataset.tsv"
        if not safe_name.lower().endswith(".tsv") and not safe_name.lower().endswith(".txt"):
            safe_name += ".tsv"

        raw_destination = self.raw_dir / safe_name

        # Preserve pristine upload
        with open(raw_destination, "wb") as f:
            f.write(file_bytes)

        # Continue with normal file ingestion
        return self.ingest_file(
            raw_destination,
            copy_to_raw=False,
            save_preprocessed=save_preprocessed,
        )

    def _preprocess_by_schema(
        self, df: pd.DataFrame, schema: DetectedSchema
    ) -> pd.DataFrame:
        """Applies EntityPreprocessor to resolved identity columns."""
        mapping = schema.column_mapping

        name_cols: List[str] = []
        addr_cols: List[str] = []
        phone_cols: List[str] = []
        web_cols: List[str] = []
        zip_cols: List[str] = []

        for key, actual_col in mapping.items():
            if key.startswith("name"):
                name_cols.append(actual_col)
            elif key.startswith("address"):
                addr_cols.append(actual_col)
            elif key.startswith("phone"):
                phone_cols.append(actual_col)
            elif key.startswith("website"):
                web_cols.append(actual_col)
            elif key.startswith("postal_code"):
                zip_cols.append(actual_col)

        return self.preprocessor.preprocess_dataframe(
            df,
            name_cols=name_cols,
            address_cols=addr_cols,
            phone_cols=phone_cols,
            website_cols=web_cols,
            postal_cols=zip_cols,
        )
