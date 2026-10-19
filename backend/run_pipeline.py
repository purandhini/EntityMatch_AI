"""Command-line interface to execute EntityMatch AI pipeline stages."""

import argparse
import json
import sys
from pathlib import Path

# Ensure project root is available in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from backend.config import DEFAULT_CONFIG, PipelineConfig
from backend.pipeline.data_loader import TSVDataLoader
from backend.pipeline.evaluation import EvaluationMetrics
from backend.pipeline.explainability import MatchExplainer
from backend.pipeline.feature_engineering import PairwiseFeatureExtractor
from backend.pipeline.predict import EntityPredictor
from backend.pipeline.preprocessing import EntityPreprocessor
from backend.pipeline.train import ModelTrainer
from backend.pipeline.ingestion import SchemaValidator, canonicalize_pairwise_dataframe
from backend.pipeline.linkage import prepare_entity_table


def _prepare_pairwise_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Detect arbitrary pairwise headers, canonicalize them, and preprocess them."""
    validator = SchemaValidator()
    schema = validator.inspect_headers(list(df.columns))
    pairs = canonicalize_pairwise_dataframe(df, schema=schema)
    preprocessor = EntityPreprocessor()
    return preprocessor.preprocess_dataframe(
        pairs,
        name_cols=["name_1", "name_2"],
        address_cols=["address_1", "address_2"],
        phone_cols=["phone_1", "phone_2"],
        website_cols=["website_1", "website_2"],
        postal_cols=["postal_code_1", "postal_code_2"],
    )


def run_train(args):
    """Executes the training and F0.5 threshold optimization pipeline."""
    print("=== [EntityMatch AI] Starting Training Pipeline ===")
    train_path = Path(args.train_path)
    if not train_path.is_file():
        print(f"Error: Training file not found at: {train_path.resolve()}", file=sys.stderr)
        print("Please place your training TSV file in data/raw/ and try again.", file=sys.stderr)
        sys.exit(1)

    loader = TSVDataLoader()
    print(f"1. Loading training TSV dataset from {train_path}...")
    df = loader.load_tsv(train_path)
    print(f"   Loaded {len(df)} records with columns: {list(df.columns)}")

    label_col = args.label_col
    if label_col not in df.columns:
        print(f"Error: Label column '{label_col}' not found in training dataset.", file=sys.stderr)
        sys.exit(1)

    y = df[label_col].astype(int)

    print("2. Detecting pairwise schema and preprocessing text fields...")
    df_clean = _prepare_pairwise_dataframe(df)

    print("3. Extracting pairwise similarity features...")
    extractor = PairwiseFeatureExtractor()
    X = extractor.extract_features_from_pair_df(df_clean)
    print(f"   Generated feature matrix shape: {X.shape}")

    print(f"4. Running cross-validation & optimizing threshold for F{args.f_beta}...")
    config = DEFAULT_CONFIG.training
    config.model_type = args.model_type
    config.f_beta = args.f_beta
    trainer = ModelTrainer(config=config)
    meta = trainer.train_with_cv(X, y)

    save_path = Path(args.model_save_path)
    trainer.save_model(save_path)
    print(f"   Model bundle successfully saved to: {save_path.resolve()}")
    print(f"   Optimal decision threshold: {meta['optimal_threshold']}")
    print(f"   CV Results: {meta['cv_metrics']}")
    print("=== Training Complete ===")


def run_predict(args):
    """Execute prediction for either pairwise TSVs or single-table entity TSVs."""
    print("=== [EntityMatch AI] Starting Prediction Pipeline ===")
    test_path = Path(args.test_path)
    model_path = Path(args.model_path)

    if not test_path.is_file():
        print(f"Error: Test file not found at: {test_path.resolve()}", file=sys.stderr)
        sys.exit(1)
    if not model_path.is_file():
        print(f"Error: Trained model bundle not found at: {model_path.resolve()}", file=sys.stderr)
        print("Please train a model first using 'python backend/run_pipeline.py train'.", file=sys.stderr)
        sys.exit(1)

    predictor = EntityPredictor(model_path=model_path)
    loader = TSVDataLoader()
    test_df = loader.load_tsv(test_path)
    schema = SchemaValidator().inspect_headers(list(test_df.columns))
    print(f"1. Detected dataset format: {schema.dataset_format}")

    if schema.dataset_format == "PAIRWISE":
        test_canonical = canonicalize_pairwise_dataframe(test_df, schema=schema)
        id_1_col = "id_1" if "id_1" in test_canonical.columns else test_canonical.columns[0]
        id_2_col = "id_2" if "id_2" in test_canonical.columns else test_canonical.columns[1]
        results_df = predictor.predict_pairs_df(
            test_canonical, id_1_col=id_1_col, id_2_col=id_2_col, threshold=args.threshold
        )
        candidate_df = pd.DataFrame({"id_1": test_canonical[id_1_col], "id_2": test_canonical[id_2_col]})
    else:
        print("2. Running blocking to generate candidate pairs...")
        results_df, candidate_df = predictor.predict_entity_table(
            test_df, threshold=args.threshold
        )
        print(f"   Generated {len(candidate_df)} candidate pairs from {len(test_df)} entities.")

    candidate_path = Path(args.candidate_output_path)
    candidate_path.parent.mkdir(parents=True, exist_ok=True)
    candidate_df.to_csv(candidate_path, sep="\t", index=False)
    print(f"Candidate pairs written to: {candidate_path.resolve()}")

    out_path = Path(args.output_path)
    predictor.save_predictions_tsv(results_df, out_path)
    print(f"=== Successfully wrote predictions to: {out_path.resolve()} ===")


def run_evaluate(args):
    """Evaluates trained model against a labeled validation set."""
    print("=== [EntityMatch AI] Starting Evaluation Pipeline ===")
    val_path = Path(args.val_path)
    model_path = Path(args.model_path)

    if not val_path.is_file() or not model_path.is_file():
        print("Error: Validation file or model file not found.", file=sys.stderr)
        sys.exit(1)

    predictor = EntityPredictor(model_path=model_path)
    loader = TSVDataLoader()
    val_df = loader.load_tsv(val_path)

    label_col = args.label_col
    y_true = val_df[label_col].astype(int).values

    val_canonical = canonicalize_pairwise_dataframe(val_df, schema=SchemaValidator().inspect_headers(list(val_df.columns)))
    val_clean = predictor.preprocessor.preprocess_dataframe(
        val_canonical,
        name_cols=["name_1", "name_2"],
        address_cols=["address_1", "address_2"],
        phone_cols=["phone_1", "phone_2"],
        website_cols=["website_1", "website_2"],
        postal_cols=["postal_code_1", "postal_code_2"],
    )
    features_df = predictor.feature_extractor.extract_features_from_pair_df(val_clean)
    probs, preds = predictor.predict_features(features_df, threshold=args.threshold)

    metrics = EvaluationMetrics.compute_metrics(y_true, preds, beta=args.f_beta)
    print("Evaluation Results:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    if args.report_path:
        rep_path = Path(args.report_path)
        rep_path.parent.mkdir(parents=True, exist_ok=True)
        with open(rep_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        print(f"Report written to: {rep_path.resolve()}")


def run_explain(args):
    """Generates global feature importance ranking from trained model."""
    print("=== [EntityMatch AI] Feature Explainability ===")
    model_path = Path(args.model_path)
    if not model_path.is_file():
        print(f"Error: Model not found at: {model_path.resolve()}", file=sys.stderr)
        sys.exit(1)

    trainer = ModelTrainer.load_model(model_path)
    explainer = MatchExplainer(trainer)

    df_imp = explainer.get_global_feature_importance()
    print("Top Feature Importances:")
    print(df_imp.head(15).to_string(index=False))

    out_path = Path(args.output_path)
    explainer.export_importance_report(out_path)


def run_ingest(args):
    """Executes data ingestion, schema inspection, and validation on a raw TSV file."""
    print("=== [EntityMatch AI] Ingesting and Validating TSV Dataset ===")
    from backend.pipeline.ingestion import (
        DataIngestionPipeline,
        EmptyFileError,
        MalformedFileError,
        SchemaValidationError,
    )

    tsv_path = Path(args.tsv_path)
    pipeline = DataIngestionPipeline()
    try:
        result = pipeline.ingest_file(tsv_path, copy_to_raw=args.copy_to_raw, save_preprocessed=True)
        print("Success! TSV Ingestion Summary:")
        print(f"  Raw file: {result['raw_file_path']}")
        print(f"  SHA256: {result['sha256']}")
        print(f"  Detected encoding: {result['detected_encoding']}")
        print(f"  Dataset format: {result['dataset_format']}")
        print(f"  Total records: {result['total_records']}")
        print(f"  Detected columns: {result['detected_column_mapping']}")
        print(f"  Preprocessed file: {result['preprocessed_file_path']}")
    except (EmptyFileError, MalformedFileError, SchemaValidationError, FileNotFoundError) as err:
        print(f"Ingestion Validation Error: {err}", file=sys.stderr)
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="EntityMatch AI - Modular Machine Learning Pipeline for Entity Resolution"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Train subparser
    train_p = subparsers.add_parser("train", help="Train model and optimize F0.5 decision threshold")
    train_p.add_argument("--train-path", required=True, help="Path to training TSV file")
    train_p.add_argument("--label-col", default="label", help="Target binary column name")
    train_p.add_argument("--model-save-path", default="models/entity_matcher.joblib", help="Output path for model")
    train_p.add_argument("--model-type", default="lightgbm", choices=["lightgbm", "logistic_regression", "random_forest"])
    train_p.add_argument("--f-beta", type=float, default=0.5, help="F-beta score weight (default: 0.5)")
    train_p.set_defaults(func=run_train)

    # Predict subparser
    pred_p = subparsers.add_parser("predict", help="Generate predictions on unlabeled test TSV")
    pred_p.add_argument("--test-path", required=True, help="Path to test TSV file")
    pred_p.add_argument("--model-path", default="models/entity_matcher.joblib", help="Path to trained model")
    pred_p.add_argument("--output-path", default="outputs/predictions/submission.tsv", help="Output TSV path")
    pred_p.add_argument("--threshold", type=float, default=None, help="Optional manual threshold override")
    pred_p.add_argument("--candidate-output-path", default="outputs/predictions/candidate_pairs.tsv", help="Path for blocked candidate pairs")
    pred_p.set_defaults(func=run_predict)

    # Evaluate subparser
    eval_p = subparsers.add_parser("evaluate", help="Evaluate model metrics on labeled data")
    eval_p.add_argument("--val-path", required=True, help="Path to labeled validation TSV")
    eval_p.add_argument("--model-path", default="models/entity_matcher.joblib", help="Path to trained model")
    eval_p.add_argument("--label-col", default="label", help="Target label column name")
    eval_p.add_argument("--report-path", default="outputs/reports/eval_report.json", help="Path to output report")
    eval_p.add_argument("--threshold", type=float, default=None, help="Threshold to evaluate")
    eval_p.add_argument("--f-beta", type=float, default=0.5, help="F-beta metric parameter (default: 0.5)")
    eval_p.set_defaults(func=run_evaluate)

    # Explain subparser
    exp_p = subparsers.add_parser("explain", help="Generate feature importance and explainability report")
    exp_p.add_argument("--model-path", default="models/entity_matcher.joblib", help="Path to trained model")
    exp_p.add_argument("--output-path", default="outputs/reports/feature_importance.tsv", help="Output report path")
    exp_p.set_defaults(func=run_explain)

    # Ingest subparser
    ingest_p = subparsers.add_parser("ingest", help="Inspect headers, validate identity fields, and ingest raw TSV")
    ingest_p.add_argument("--tsv-path", required=True, help="Path to TSV file to ingest")
    ingest_p.add_argument("--copy-to-raw", action="store_true", help="Copy source file into data/raw/ if outside it")
    ingest_p.set_defaults(func=run_ingest)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
