"""Data schemas for EntityMatch AI API."""

from typing import Any, Dict, List, Optional
try:
    from pydantic import BaseModel, Field
except ImportError:
    # Minimal fallback schema mock if pydantic is not yet installed
    class BaseModel:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)
        def dict(self):
            return self.__dict__
    def Field(default=None, **kwargs):
        return default


class EntityRecord(BaseModel):
    """Business entity attributes for comparison."""
    name: str
    address: Optional[str] = ""
    city: Optional[str] = ""
    state: Optional[str] = ""
    postal_code: Optional[str] = ""
    phone: Optional[str] = ""
    website: Optional[str] = ""


class PairMatchRequest(BaseModel):
    """Request payload to compare two business entities."""
    record_left: EntityRecord
    record_right: EntityRecord
    threshold_override: Optional[float] = None


class PairMatchResponse(BaseModel):
    """Response payload containing similarity score, binary decision, and factors."""
    match_probability: float
    is_match: int
    decision_threshold: float
    explanation_summary: Optional[str] = None
    top_drivers: Optional[List[Dict[str, Any]]] = None


class HealthResponse(BaseModel):
    """Status health check response."""
    status: str
    model_loaded: bool
    version: str = "1.0.0"


class InspectPathRequest(BaseModel):
    """Request payload to ingest and validate a file existing at a local path."""
    file_path: str
    save_preprocessed: bool = True


class IngestionResponse(BaseModel):
    """Result of data ingestion, validation, and preprocessing."""
    status: str
    raw_file_path: str
    file_size_bytes: int
    sha256: str
    detected_encoding: str
    dataset_format: str
    detected_column_mapping: Dict[str, str]
    total_records: int
    columns: List[str]
    preprocessed_file_path: Optional[str] = None
    warnings: List[str] = []
