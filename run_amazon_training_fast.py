"""Fast real-data Amazon training-data assembly.

Run from the project root:
    python run_amazon_training_fast.py

This samples real ground-truth pairs first, then loads only the source
entities needed for those sampled pairs. It avoids preprocessing all
12.5M Amazon entities before training-data construction.
"""

import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from backend.pipeline.linkage import (
    load_amazon_ground_truth,
    build_amazon_training_dataframe,
)

TRAIN_DIR = Path(
    r"C:\Users\nagav\Downloads\6ab10eb3b23ba_student_resource"
    r"\student_resource\dataset\train"
)

SOURCE_FILES = [
    TRAIN_DIR / "train_source1.tsv",
    TRAIN_DIR / "train_source2.tsv",
    TRAIN_DIR / "train_source3.tsv",
]
GROUND_TRUTH = TRAIN_DIR / "train_ground_truth.tsv"

POSITIVE_SAMPLE_SIZE = 50_000
NEGATIVE_RATIO = 1
RANDOM_STATE = 42
CHUNKSIZE = 50_000

for f in SOURCE_FILES + [GROUND_TRUTH]:
    if not f.is_file():
        print(f"ERROR: File not found: {f}")
        sys.exit(1)

# Step 1: Read GT first and sample real positives.
print("Loading ground truth first...")
gt_all = load_amazon_ground_truth(GROUND_TRUTH)
print(f"  Ground-truth rows (after comma-expansion): {len(gt_all):,}")

gt = gt_all.sample(
    n=min(POSITIVE_SAMPLE_SIZE, len(gt_all)),
    random_state=RANDOM_STATE,
).reset_index(drop=True)

needed_ids = set(gt["left_id"].astype(str))
needed_ids.update(gt["right_id"].astype(str))
print(f"  Sampled positive pairs: {len(gt):,}")
print(f"  Unique entity IDs needed: {len(needed_ids):,}")

# Step 2: Read only source rows whose entity_id occurs in the sampled GT.
print("\nLoading only required source entities (chunked)...")
frames = []

for idx, path in enumerate(SOURCE_FILES):
    label = f"source{idx + 1}"
    kept = 0
    print(f"  Scanning {path.name}...")

    for chunk in pd.read_csv(path, sep="\t", dtype=str, chunksize=CHUNKSIZE):
        chunk = chunk.fillna("")
        mask = chunk["entity_id"].astype(str).isin(needed_ids)
        selected = chunk.loc[mask].copy()

        if selected.empty:
            continue

        selected["source"] = label
        selected["entity_id"] = selected["entity_id"].astype(str)
        selected["id"] = label + "_" + selected["entity_id"]
        frames.append(selected)
        kept += len(selected)

    print(f"    kept {kept:,} rows")

if not frames:
    raise RuntimeError("No sampled ground-truth entities were found in source files.")

combined = pd.concat(frames, ignore_index=True)

# Canonical columns expected by build_amazon_training_dataframe.
combined = combined.rename(
    columns={
        "business_name": "name",
        "business_address": "address",
    }
)

# Preprocess only this small sampled entity table.
from backend.pipeline.preprocessing import EntityPreprocessor
proc = EntityPreprocessor()
combined = proc.preprocess_dataframe(
    combined,
    name_cols=["name"] if "name" in combined.columns else [],
    address_cols=["address"] if "address" in combined.columns else [],
    phone_cols=["phone"] if "phone" in combined.columns else [],
    website_cols=["website"] if "website" in combined.columns else [],
    postal_cols=["postal_code"] if "postal_code" in combined.columns else [],
)

print(f"\n  Loaded sampled entities: {len(combined):,}")

# Step 3: Build labeled training pairs.
print("\nBuilding training DataFrame...")
print(f"  positive_sample_size = {len(gt):,}")
print(f"  negative_ratio       = {NEGATIVE_RATIO}")
print(f"  random_state         = {RANDOM_STATE}")
print()

train_df = build_amazon_training_dataframe(
    combined,
    gt,
    positive_sample_size=None,  # GT was already sampled above.
    negative_ratio=NEGATIVE_RATIO,
    random_state=RANDOM_STATE,
)

# Step 4: Report.
n_pos = (train_df["label"] == 1).sum()
n_neg = (train_df["label"] == 0).sum()
n_total = len(train_df)

print()
print("=" * 50)
print("  Training DataFrame Report")
print("=" * 50)
print(f"  Positive rows (label=1) : {n_pos:,}")
print(f"  Negative rows (label=0) : {n_neg:,}")
print(f"  Total rows              : {n_total:,}")
print(f"  Columns ({len(train_df.columns)}):")
for col in train_df.columns:
    print(f"    {col}")
print("=" * 50)
print()
print("Training DataFrame is ready. Next step: feature extraction + model training.")
