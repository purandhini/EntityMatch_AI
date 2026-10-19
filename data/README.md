# EntityMatch AI - Data Directory

This directory is prepared for your Business Entity Resolution ML challenge data.

> [!NOTE]
> Currently, this directory contains no data files. When your challenge dataset is ready, place your TSV files here.

---

## Directory Organization

```text
data/
├── raw/             <-- Put your official challenge TSV files here
├── processed/       <-- Cache for preprocessed tables, candidate pairs, and feature matrices
└── README.md        <-- This guide
```

---

## Supported Challenge Formats

The pipeline accommodates the two primary data formats used in entity resolution challenges:

### Format A: Pairwise Records (`train.tsv` / `test.tsv`)
In this format, each row represents a candidate pair with columns for both entities:
- `id_1`, `id_2` (Unique record identifiers)
- `name_1`, `name_2` (Company or business entity names)
- `address_1`, `address_2` (Street addresses)
- `city_1`, `city_2`
- `state_1`, `state_2`
- `postal_code_1`, `postal_code_2`
- `phone_1`, `phone_2`
- `website_1`, `website_2`
- `label` (`1` for match, `0` for non-match; omitted in test set)

### Format B: Separate Tables with Labeled Pairs
In this format, entities are provided as tables, and a separate file contains labeled ID pairs:
- `data/raw/entities_left.tsv`: Entity records with `id`, `name`, `address`, `city`, etc.
- `data/raw/entities_right.tsv`: Target entity records to match against.
- `data/raw/train_pairs.tsv`: `id_left`, `id_right`, `label`

---

## Step-by-Step: Adding Your Real Challenge Files

1. **Copy your files** into `data/raw/`:
   ```bash
   cp /path/to/challenge/train.tsv data/raw/
   cp /path/to/challenge/test.tsv data/raw/
   ```

2. **Auto-Inspect & Validate Headers with the Ingestion Layer**:
   Run the ingestion command to auto-detect columns, validate required business identity fields (name, address), and verify encoding:
   ```bash
   python backend/run_pipeline.py ingest --tsv-path data/raw/train.tsv
   ```

3. **Optional manual overrides in `backend/config.py`**:
   If your TSV uses unusual headers not caught by auto-detection, you can configure `ColumnMapping` in `backend/config.py`:
   ```python
   column_mapping = ColumnMapping(
       id_col="record_id",
       name_col="company_name",
       address_col="street_address",
       city_col="city",
       state_col="state",
       postal_code_col="zip",
       phone_col="telephone",
       website_col="url",
       label_col="is_match"
   )
   ```

4. **Run pipeline**:
   - Run training:
     ```bash
     python backend/run_pipeline.py train --train-path data/raw/train.tsv
     ```
   - Run prediction:
     ```bash
     python backend/run_pipeline.py predict --test-path data/raw/test.tsv --output-path outputs/predictions/submission.tsv
     ```
