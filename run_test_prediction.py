"""Safe Amazon 3-source test prediction runner for EntityMatch AI.

Run from the EntityMatch_AI project root:
    python run_test_prediction.py

The runner:
- loads the saved LightGBM model (no retraining)
- reads the three Amazon test sources
- processes source pairs sequentially
- keeps one left source plus a bounded right-side chunk in memory
- uses the existing preprocessing, blocking, feature extraction, and predictor
- applies the already-selected threshold 0.8712
- writes predictions incrementally to TSV
"""

from pathlib import Path
import pandas as pd
import numpy as np

from backend.pipeline.preprocessing import EntityPreprocessor
from backend.pipeline.blocking import EntityBlocker
from backend.pipeline.predict import EntityPredictor


import time

ROOT = Path(__file__).resolve().parent
DATASET = Path(
    r"C:\Users\nagav\Downloads\6ab10eb3b23ba_student_resource\student_resource\dataset"
)
MODEL_PATH = ROOT / "models" / "entitymatch_model.joblib"
OUT_DIR = ROOT / "outputs" / "predictions"

THRESHOLD = 0.8712
CHUNK_ROWS = 100_000
FEATURE_BATCH = 20_000

SOURCE_FILES = {
    "source1": DATASET / "test" / "test_source1.tsv",
    "source2": DATASET / "test" / "test_source2.tsv",
    "source3": DATASET / "test" / "test_source3.tsv",
}

CANONICAL_FIELDS = [
    "id",
    "name",
    "address",
    "country",
    "city",
    "state",
    "postal_code",
    "phone",
    "website",
]

PAIR_FEATURE_FIELDS = [
    "name",
    "address",
    "city",
    "state",
    "postal_code",
    "phone",
    "website",
]


def canonicalize_test_df(df: pd.DataFrame, source_name: str) -> pd.DataFrame:
    """Map Amazon test columns to the project's canonical entity schema."""
    df = df.rename(
        columns={
            "entity_id": "id",
            "business_name": "name",
            "business_address": "address",
        }
    )

    for col in CANONICAL_FIELDS:
        if col not in df.columns:
            df[col] = ""

    df = df[CANONICAL_FIELDS].copy()

    # Prevent ID collisions between the three independent test sources.
    df["id"] = source_name + "_" + df["id"].astype(str)

    return df


def load_source(path: Path, source_name: str) -> pd.DataFrame:
    print(f"Loading {source_name}: {path}")
    df = pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
    )
    return canonicalize_test_df(df, source_name)


def preprocess(df: pd.DataFrame, preprocessor: EntityPreprocessor) -> pd.DataFrame:
    """Apply the same project preprocessor used by the training/inference code."""
    return preprocessor.preprocess_dataframe(
        df,
        name_cols=["name"],
        address_cols=["address"],
        phone_cols=["phone"],
        website_cols=["website"],
        postal_cols=["postal_code"],
    )


def make_pair_df(
    left_rows: pd.DataFrame,
    right_rows: pd.DataFrame,
) -> pd.DataFrame:
    """Create the pair dataframe expected by PairwiseFeatureExtractor without index copies."""
    out = {}

    for field in PAIR_FEATURE_FIELDS:
        out[f"{field}_1"] = left_rows[field].to_numpy()
        out[f"{field}_2"] = right_rows[field].to_numpy()

        clean = f"{field}_clean"
        if clean in left_rows.columns:
            out[f"{field}_1_clean"] = left_rows[clean].to_numpy()
            out[f"{field}_2_clean"] = right_rows[clean].to_numpy()

    return pd.DataFrame(out)


def process_source_pair(
    left_name: str,
    left_df: pd.DataFrame,
    right_name: str,
    right_path: Path,
    preprocessor: EntityPreprocessor,
    blocker: EntityBlocker,
    predictor: EntityPredictor,
    output_path: Path,
    candidate_path: Path,
):
    """Process one source pair incrementally."""

    print(f"\n=== {left_name} vs {right_name} ===")

    # Avoid repeated preprocessing if left_df has already been preprocessed
    if "name_clean" in left_df.columns:
        left = left_df
    else:
        left = preprocess(left_df, preprocessor)
    left = left.drop_duplicates(subset=["id"], keep="first").set_index("id", drop=False)

    # Clear blocker cache before starting a new source pair
    blocker.clear_cache()

    pair_count = 0
    match_count = 0
    first_write = not output_path.exists()
    first_candidate_write = not candidate_path.exists()

    right_reader = pd.read_csv(
        right_path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        chunksize=CHUNK_ROWS,
    )

    for chunk_no, raw_right in enumerate(right_reader, 1):
        t_chunk_start = time.time()
        right = canonicalize_test_df(raw_right, right_name)
        right = preprocess(right, preprocessor)
        right = right.drop_duplicates(subset=["id"], keep="first").set_index("id", drop=False)

        # Deduplication is bounded to the current right-side chunk.
        # The blocker itself streams strategy results instead of materializing
        # the complete candidate set.
        batch = []
        seen_chunk = set()

        def flush():
            nonlocal pair_count, match_count
            nonlocal first_write, first_candidate_write

            if not batch:
                return

            ids1 = [pair[0] for pair in batch]
            ids2 = [pair[1] for pair in batch]

            left_rows = left.loc[ids1]
            right_rows = right.loc[ids2]

            pairs = make_pair_df(left_rows, right_rows)

            features = predictor.feature_extractor.extract_features_from_pair_df(
                pairs
            )

            probs, preds = predictor.predict_features(
                features,
                threshold=THRESHOLD,
            )

            results = pd.DataFrame(
                {
                    "id_1": ids1,
                    "id_2": ids2,
                    "match_probability": np.round(probs, 6),
                    "is_match": preds,
                }
            )

            results.to_csv(
                output_path,
                sep="\t",
                index=False,
                mode="w" if first_write else "a",
                header=first_write,
            )
            first_write = False

            # All candidates that reach feature scoring are recorded here.
            results[["id_1", "id_2"]].to_csv(
                candidate_path,
                sep="\t",
                index=False,
                mode="w" if first_candidate_write else "a",
                header=first_candidate_write,
            )
            first_candidate_write = False

            pair_count += len(results)
            match_count += int(results["is_match"].sum())

            batch.clear()

        for id_left, id_right in blocker.stream_candidate_pairs(
            left,
            right,
            id_col_left="id",
            id_col_right="id",
            name_col="name",
            address_col="address",
            postal_col="postal_code",
            phone_col="phone",
            website_col="website",
            country_col="country",
        ):
            key = (id_left, id_right)

            if key in seen_chunk:
                continue

            seen_chunk.add(key)
            batch.append(key)

            if len(batch) >= FEATURE_BATCH:
                flush()

        flush()

        chunk_elapsed = time.time() - t_chunk_start
        print(
            f"{right_name} chunk {chunk_no}: "
            f"{len(right):,} rows processed in {chunk_elapsed:.1f}s; "
            f"cumulative candidates={pair_count:,}; "
            f"matches={match_count:,}"
        )

        del right

    print(
        f"Finished {left_name} vs {right_name}: "
        f"candidates={pair_count:,}, matches={match_count:,}"
    )


def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Saved model not found: {MODEL_PATH}")

    for source_name, source_path in SOURCE_FILES.items():
        if not source_path.exists():
            raise FileNotFoundError(
                f"Test source not found for {source_name}: {source_path}"
            )

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    prediction_path = OUT_DIR / "matching_results.tsv"
    candidate_path = OUT_DIR / "candidate_pairs.tsv"

    # Prevent accidental concatenation with an old run.
    for path in (prediction_path, candidate_path):
        if path.exists():
            path.unlink()

    preprocessor = EntityPreprocessor()
    blocker = EntityBlocker()
    predictor = EntityPredictor(model_path=MODEL_PATH)

    print("Model:", MODEL_PATH)
    print("Model feature count:", len(predictor.trainer.feature_names))
    print("Threshold:", THRESHOLD)
    print("Output columns: id_1, id_2, match_probability, is_match")

    # source1 is the smallest source, so reuse it as the left side for
    # source1-source2 and source1-source3. Preprocess once upfront.
    source1 = load_source(SOURCE_FILES["source1"], "source1")
    source1 = preprocess(source1, preprocessor)

    process_source_pair(
        "source1",
        source1,
        "source2",
        SOURCE_FILES["source2"],
        preprocessor,
        blocker,
        predictor,
        prediction_path,
        candidate_path,
    )

    process_source_pair(
        "source1",
        source1,
        "source3",
        SOURCE_FILES["source3"],
        preprocessor,
        blocker,
        predictor,
        prediction_path,
        candidate_path,
    )

    del source1

    # The final source pair is source2-source3. Only source2 is kept in
    # memory while source3 is read in chunks.
    source2 = load_source(SOURCE_FILES["source2"], "source2")
    source2 = preprocess(source2, preprocessor)

    process_source_pair(
        "source2",
        source2,
        "source3",
        SOURCE_FILES["source3"],
        preprocessor,
        blocker,
        predictor,
        prediction_path,
        candidate_path,
    )

    del source2

    print("\nPrediction complete.")
    print("Predictions:", prediction_path.resolve())
    print("Candidates:", candidate_path.resolve())


if __name__ == "__main__":
    main()
