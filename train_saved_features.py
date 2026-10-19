import sys
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.pipeline.train import ModelTrainer

FEATURES_PATH = PROJECT_ROOT / "models" / "training_features.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "entitymatch_model.joblib"

print("Loading saved training features...")

df = pd.read_csv(FEATURES_PATH)

print(f"  Dataset shape: {df.shape}")

if "label" not in df.columns:
    raise ValueError("label column not found in training_features.csv")

X = df.drop(columns=["label"])
y = df["label"].astype(int)

print(f"  Samples: {len(X):,}")
print(f"  Features: {len(X.columns)}")
print(f"  Positive labels: {(y == 1).sum():,}")
print(f"  Negative labels: {(y == 0).sum():,}")

print("\nTraining model with CV + F0.5 threshold optimization...")

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

MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
trainer.save_model(MODEL_PATH)

print(f"\nModel saved to:")
print(MODEL_PATH.resolve())
