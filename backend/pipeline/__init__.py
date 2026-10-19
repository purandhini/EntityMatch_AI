"""EntityMatch AI - Core Pipeline Package."""

from .data_loader import TSVDataLoader
from .preprocessing import EntityPreprocessor
from .blocking import EntityBlocker
from .feature_engineering import PairwiseFeatureExtractor
from .train import ModelTrainer
from .predict import EntityPredictor
from .evaluation import EvaluationMetrics
from .explainability import MatchExplainer
from .ingestion import (
    DataIngestionPipeline,
    SchemaValidator,
    DetectedSchema,
    IngestionError,
    SchemaValidationError,
    EmptyFileError,
    MalformedFileError,
)

__all__ = [
    "TSVDataLoader",
    "EntityPreprocessor",
    "EntityBlocker",
    "PairwiseFeatureExtractor",
    "ModelTrainer",
    "EntityPredictor",
    "EvaluationMetrics",
    "MatchExplainer",
    "DataIngestionPipeline",
    "SchemaValidator",
    "DetectedSchema",
    "IngestionError",
    "SchemaValidationError",
    "EmptyFileError",
    "MalformedFileError",
]
