# EntityMatch AI - Business Entity Resolution ML Pipeline

EntityMatch AI is a modular, production-ready machine learning foundation engineered for Business Entity Resolution challenges (record linkage, entity deduplication, and company matching across heterogeneous databases).

> [!IMPORTANT]
> **No Fake Results Policy**: This repository contains the complete structural architecture, algorithms, and modular pipelines. It intentionally does **not** include synthetic training results, fake F0.5 scores, or mockup predictions. It is ready for your real challenge TSV files.

---

## Repository Structure

```text
EntityMatch_AI/
├── backend/
│   ├── app/                    # FastAPI web service layer for real-time inference & inspection
│   │   ├── main.py             # FastAPI entrypoint
│   │   ├── schemas.py          # Pydantic data schemas
│   │   └── api/
│   │       └── routes.py       # Endpoints: /health, /match-pair
│   ├── pipeline/               # Core Machine Learning Pipeline
│   │   ├── data_loader.py      # TSV loading with encoding fallbacks and chunking
│   │   ├── preprocessing.py    # Entity normalization (legal suffixes, street abbreviations, phones, URLs)
│   │   ├── blocking.py         # Candidate generation to reduce O(N^2) comparison space
│   │   ├── feature_engineering.py # String similarities (Jaro-Winkler, Levenshtein, token ratios, n-grams)
│   │   ├── train.py            # Model training & cross-validation with F0.5 threshold optimization
│   │   ├── predict.py          # Batch inference & submission TSV generator
│   │   ├── evaluation.py       # Precision, Recall, F1, F0.5 metric calculations & threshold sweep
│   │   └── explainability.py   # Global feature importance & pair-level match driver explanations
│   ├── config.py               # Global settings, column mappings, and hyperparameters
│   └── run_pipeline.py         # CLI dispatcher for train / predict / evaluate / explain
├── frontend/                   # UI dashboard placeholder for verification & manual review
├── models/                     # Directory where trained model bundles (.joblib) are stored
├── data/
│   ├── raw/                    # Destination for challenge TSV files (e.g., train.tsv, test.tsv)
│   ├── processed/              # Processed pair cache and feature matrices
│   └── README.md               # Detailed data ingestion guide
├── outputs/
│   ├── predictions/            # Exported prediction files & competition submission TSVs
│   └── reports/                # Evaluation metric reports and feature importance rankings
├── tests/                      # Unit tests verifying pipeline algorithms
├── requirements.txt            # Dependency specification
└── README.md                   # This documentation
```

---

## How to Add Your Real TSV Dataset

When you receive the challenge dataset:

### Step 1: Copy TSV Files to `data/raw/`
Place your training and testing TSV files in the `data/raw/` directory:
```bash
# Example
copy path\to\train.tsv data\raw\train.tsv
copy path\to\test.tsv data\raw\test.tsv
```

### Step 2: Configure Column Mappings in `backend/config.py`
If your TSV uses custom column names, adjust `ColumnMapping` in [backend/config.py](file:///C:/Users/nagav/.gemini/antigravity/scratch/EntityMatch_AI/backend/config.py):

```python
# backend/config.py
@dataclass
class ColumnMapping:
    id_col: str = "id"              # Record identifier column
    label_col: str = "label"        # 1 for match, 0 for non-match

    # Suffixes used for pairwise records (e.g. name_1 vs name_2)
    left_suffix: str = "_1"
    right_suffix: str = "_2"

    # Entity attribute columns
    name_col: str = "name"
    address_col: str = "address"
    city_col: str = "city"
    state_col: str = "state"
    postal_code_col: str = "postal_code"
    phone_col: str = "phone"
    website_col: str = "website"
```

### Step 3: Run the Pipeline via CLI

#### 1. Train the Model and Optimize F0.5 Threshold:
```bash
python backend/run_pipeline.py train --train-path data/raw/train.tsv --label-col label --model-type lightgbm --f-beta 0.5
```
*What happens:*
- Loads TSV with encoding validation and missing-value normalization.
- Preprocesses company names (standardizing legal suffixes like Inc/Corp/LLC) and addresses.
- Computes pairwise similarity features (Jaro-Winkler, Levenshtein, token set ratio, exact flags).
- Runs 5-fold cross-validation.
- Searches for the decision threshold $\tau \in [0.01, 0.99]$ that maximizes $F_{0.5}$:
  $$F_{0.5} = 1.25 \cdot \frac{\text{Precision} \cdot \text{Recall}}{0.25 \cdot \text{Precision} + \text{Recall}}$$
- Saves the trained model bundle and optimal threshold to `models/entity_matcher.joblib`.

#### 2. Generate Challenge Predictions:
```bash
python backend/run_pipeline.py predict --test-path data/raw/test.tsv --output-path outputs/predictions/submission.tsv
```
*What happens:*
- Loads test TSV and extracts pairwise features.
- Applies the optimal $F_{0.5}$ decision threshold.
- Outputs competition-ready `submission.tsv` with columns `id_1`, `id_2`, `match_probability`, and `is_match`.

#### 3. Evaluate on Validation Set:
```bash
python backend/run_pipeline.py evaluate --val-path data/raw/val.tsv --label-col label --report-path outputs/reports/eval_report.json
```

#### 4. Export Feature Explainability:
```bash
python backend/run_pipeline.py explain --output-path outputs/reports/feature_importance.tsv
```

---

## Key Pipeline Components

### 1. Blocking (`backend/pipeline/blocking.py`)
Comparing all records against each other is $O(N^2)$, which becomes intractable at scale (e.g., $10^5$ records = $10^{10}$ comparisons). The blocking engine reduces this candidate space using:
- **Token Prefix Blocking**: Groups entities sharing token prefixes of length $\ge 3$.
- **Sorted Neighborhood Method (SNM)**: Slides a window over sorted keys, resilient to minor spelling variations.
- **Secondary Exact Attribute Blocking**: Uses postal codes or telephone numbers to capture alternative linkages.

### 2. Feature Engineering (`backend/pipeline/feature_engineering.py`)
Extracts multi-faceted similarity indicators across entity attributes:
- **String Similarities**: Jaro-Winkler (specialized for short business names), Levenshtein edit distance, character 3-gram Jaccard.
- **Token Permutations**: Token sort ratio and token set ratio (matches "Acme Solutions LLC" with "Solutions Acme Corp").
- **Structural Match Flags**: Exact phone match, exact domain match (stripping protocol/subdomain), zip code prefix match.
- **Missingness Indicators**: Flags when fields are missing to avoid penalizing sparse records.

### 3. F0.5 Optimization (`backend/pipeline/evaluation.py` & `train.py`)
In business entity matching, false positives (falsely merging two distinct companies) are typically twice as damaging as false negatives. The pipeline optimizes the threshold directly for $F_{0.5}$ (weights precision twice as much as recall), outperforming the standard 0.5 default cutoff.

---

## Running Unit Tests

Unit tests verify all pipeline algorithms (normalization, blocking, feature extraction, and F0.5 metric math) without requiring the challenge dataset:

```bash
python -m unittest tests/test_pipeline_units.py
```
