"""Fast real-data training: sampled Amazon pairs -> features -> ML model.

Run from the EntityMatch_AI project root:
    python run_fast_model_training.py

This intentionally uses the already-established 50,000 positive / 50,000
negative real-data training sample so we do not repeat the 12.5M-entity
preprocessing bottleneck.
"""

import sys
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.pipeline.linkage import load_amazon_ground_truth, build_amazon_training_dataframe
from backend.pipeline.preprocessing import EntityPreprocessor
from backend.pipeline.feature_engineering import PairwiseFeatureExtractor
from backend.pipeline.train import ModelTrainer

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

MODEL_PATH = PROJECT_ROOT / "models" / "entitymatch_model.joblib"
FEATURES_PATH = PROJECT_ROOT / "models" / "training_features.csv"

for f in SOURCE_FILES + [GROUND_TRUTH]:
    if not f.is_file():
        print(f"ERROR: File not found: {f}")
        sys.exit(1)

# ------------------------------------------------------------------
# 1. Sample real ground-truth pairs first.
# ------------------------------------------------------------------
print("1/4 Loading ground truth...")
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

# ------------------------------------------------------------------
# 2. Load only entities participating in the sampled positives.
# ------------------------------------------------------------------
print("\n2/4 Loading required source entities...")
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

combined = pd.concat(frames, ignore_index=True)
combined = combined.rename(
    columns={"business_name": "name", "business_address": "address"}
)

proc = EntityPreprocessor()
combined = proc.preprocess_dataframe(
    combined,
    name_cols=["name"] if "name" in combined.columns else [],
    address_cols=["address"] if "address" in combined.columns else [],
    phone_cols=["phone"] if "phone" in combined.columns else [],
    website_cols=["website"] if "website" in combined.columns else [],
    postal_cols=["postal_code"] if "postal_code" in combined.columns else [],
)
print(f"  Loaded/preprocessed entities: {len(combined):,}")

# ------------------------------------------------------------------
# 3. Build 100k labeled training rows.
# ------------------------------------------------------------------
print("\n3/4 Building labeled training pairs...")
train_df = build_amazon_training_dataframe(
    combined,
    gt,
    positive_sample_size=None,
    negative_ratio=NEGATIVE_RATIO,
    random_state=RANDOM_STATE,
)

n_pos = int((train_df["label"] == 1).sum())
n_neg = int((train_df["label"] == 0).sum())
print(f"  Positive rows: {n_pos:,}")
print(f"  Negative rows: {n_neg:,}")
print(f"  Total rows:    {len(train_df):,}")

# ------------------------------------------------------------------
# 4. Feature extraction + CV model training.
# ------------------------------------------------------------------
print("\n4/4 Extracting pairwise features...")
extractor = PairwiseFeatureExtractor()
X = extractor.extract_features_from_pair_df(train_df)
y = train_df["label"].astype(int)

print(f"  Feature matrix shape: {X.shape}")
print(f"  Feature columns: {len(X.columns)}")

# Keep a compact copy for reproducibility/debugging.
FEATURES_PATH.parent.mkdir(parents=True, exist_ok=True)
feature_output = X.copy()
feature_output["label"] = y.to_numpy()
feature_output.to_csv(FEATURES_PATH, index=False)
print(f"  Features saved to: {FEATURES_PATH}")

print("\nTraining model with the project's configured CV/F0.5 pipeline...")
trainer = ModelTrainer()
metadata = trainer.train_with_cv(X, y)

print("\n" + "=" * 60)
print("MODEL TRAINING COMPLETE")
print("=" * 60)
print(f"Model type:        {metadata.get('model_type')}")
print(f"Samples:           {metadata.get('sample_count')}")
print(f"Features:          {metadata.get('feature_count')}")
print(f"Optimal threshold: {metadata.get('optimal_threshold')}")
print(f"CV metrics:        {metadata.get('cv_metrics')}")
print("=" * 60)

trainer.save_model(MODEL_PATH)
print(f"Model saved to: {MODEL_PATH.resolve()}")
print("\nNext step: use the saved model with predict.py on the test data.")
