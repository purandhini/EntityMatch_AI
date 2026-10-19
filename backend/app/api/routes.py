"""FastAPI routes for real-time entity matching and explainability."""

from pathlib import Path
from typing import Optional
from fastapi import APIRouter, HTTPException

from ...config import DEFAULT_CONFIG
from ...pipeline.predict import EntityPredictor
from ...pipeline.explainability import MatchExplainer
from ...pipeline.preprocessing import EntityPreprocessor
from ...pipeline.feature_engineering import PairwiseFeatureExtractor
from ..schemas import HealthResponse, PairMatchRequest, PairMatchResponse, InspectPathRequest, IngestionResponse

router = APIRouter()

# Global predictor holder
_predictor: Optional[EntityPredictor] = None


def get_predictor() -> Optional[EntityPredictor]:
    """Lazy loader for trained model bundle."""
    global _predictor
    default_model_path = DEFAULT_CONFIG.models_dir / "entity_matcher.joblib"
    if _predictor is None and default_model_path.is_file():
        try:
            _predictor = EntityPredictor(model_path=default_model_path)
        except Exception:
            _predictor = None
    return _predictor


@router.get("/health", response_model=HealthResponse)
def health_check():
    """Health check verifying API readiness and whether a model is loaded."""
    predictor = get_predictor()
    model_loaded = predictor is not None and predictor.trainer is not None
    return HealthResponse(
        status="healthy",
        model_loaded=model_loaded,
        version="1.0.0",
    )


@router.post("/match-pair", response_model=PairMatchResponse)
def match_pair(request: PairMatchRequest):
    """
    Evaluates similarity between two business entity records.
    Returns match probability, binary decision, and feature drivers.
    """
    predictor = get_predictor()
    if predictor is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not yet trained or loaded. Run training on real TSV data first.",
        )

    # Convert request schemas to dict
    rec_l = request.record_left.dict() if hasattr(request.record_left, "dict") else vars(request.record_left)
    rec_r = request.record_right.dict() if hasattr(request.record_right, "dict") else vars(request.record_right)

    pred_res = predictor.predict_pair(rec_l, rec_r, threshold=request.threshold_override)

    # Compute explanation
    explainer = MatchExplainer(predictor.trainer)
    preprocessor = EntityPreprocessor()
    fe = PairwiseFeatureExtractor()
    clean_l = preprocessor.preprocess_record(rec_l)
    clean_r = preprocessor.preprocess_record(rec_r)
    features = fe.extract_pair_features(clean_l, clean_r)
    explanation = explainer.explain_pair(features)

    return PairMatchResponse(
        match_probability=pred_res["match_probability"],
        is_match=pred_res["is_match"],
        decision_threshold=pred_res["threshold_applied"],
        explanation_summary=explanation["explanation_summary"],
        top_drivers=explanation["top_drivers"],
    )


@router.post("/ingest/inspect-path", response_model=IngestionResponse)
def inspect_and_ingest_path(request: InspectPathRequest):
    """
    Ingests and validates a TSV dataset placed at a local filesystem path (e.g. data/raw/).
    Detects headers, validates business identity fields, preserves raw file, and runs preprocessing.
    """
    from ...pipeline.ingestion import (
        DataIngestionPipeline,
        EmptyFileError,
        MalformedFileError,
        SchemaValidationError,
    )

    pipeline = DataIngestionPipeline()
    try:
        result = pipeline.ingest_file(
            file_path=request.file_path,
            copy_to_raw=True,
            save_preprocessed=request.save_preprocessed,
        )
        return IngestionResponse(**result)
    except FileNotFoundError as fnf:
        raise HTTPException(status_code=404, detail=str(fnf))
    except EmptyFileError as ef:
        raise HTTPException(status_code=400, detail={"error": "EmptyFileError", "message": str(ef)})
    except SchemaValidationError as sve:
        raise HTTPException(
            status_code=422,
            detail={
                "error": "SchemaValidationError",
                "message": str(sve),
                "missing_fields": sve.missing_fields,
                "found_columns": sve.found_columns,
            },
        )
    except MalformedFileError as mfe:
        raise HTTPException(status_code=400, detail={"error": "MalformedFileError", "message": str(mfe)})


try:
    from fastapi import UploadFile, File

    @router.post("/ingest/upload", response_model=IngestionResponse)
    async def upload_tsv_dataset(
        file: UploadFile = File(...),
        save_preprocessed: bool = True,
    ):
        """
        Accepts uploaded TSV files, stores a pristine copy in data/raw/,
        inspects headers, validates identity fields, and passes data to preprocessing.
        """
        from ...pipeline.ingestion import (
            DataIngestionPipeline,
            EmptyFileError,
            MalformedFileError,
            SchemaValidationError,
        )

        pipeline = DataIngestionPipeline()
        content = await file.read()

        try:
            result = pipeline.ingest_bytes(
                file_bytes=content,
                filename=file.filename or "uploaded.tsv",
                save_preprocessed=save_preprocessed,
            )
            return IngestionResponse(**result)
        except EmptyFileError as ef:
            raise HTTPException(status_code=400, detail={"error": "EmptyFileError", "message": str(ef)})
        except SchemaValidationError as sve:
            raise HTTPException(
                status_code=422,
                detail={
                    "error": "SchemaValidationError",
                    "message": str(sve),
                    "missing_fields": sve.missing_fields,
                    "found_columns": sve.found_columns,
                },
            )
        except MalformedFileError as mfe:
            raise HTTPException(status_code=400, detail={"error": "MalformedFileError", "message": str(mfe)})
except ImportError:
    pass
