"""Configuration module for EntityMatch AI."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
PREDICTIONS_DIR = OUTPUTS_DIR / "predictions"
REPORTS_DIR = OUTPUTS_DIR / "reports"


@dataclass
class ColumnMapping:
    """Configures column name mappings for TSV datasets."""
    # Identifier and target columns
    id_col: str = "id"
    label_col: str = "label"

    # Pairwise column suffix conventions (used when data is stored as pair records)
    left_suffix: str = "_1"
    right_suffix: str = "_2"

    # Core entity attributes
    name_col: str = "name"
    address_col: str = "address"
    city_col: str = "city"
    state_col: str = "state"
    postal_code_col: str = "postal_code"
    phone_col: str = "phone"
    website_col: str = "website"

    # Custom column overrides if dataset uses arbitrary header names
    custom_attribute_pairs: List[tuple] = field(default_factory=list)


@dataclass
class BlockingConfig:
    """Settings for candidate pair generation to reduce O(N^2) complexity."""
    # Individual strategy toggles — all enabled by default for maximum recall
    use_token_prefix_blocking: bool = True
    use_sorted_neighborhood_blocking: bool = True
    use_address_street_number_blocking: bool = True
    use_exact_attribute_blocking: bool = True

    # Token/prefix blocking settings
    prefix_length: int = 4
    min_token_length: int = 3

    # Sorted Neighborhood Method window size (how many adjacent records to pair)
    sorted_neighborhood_window: int = 5

    # Maximum cluster size before a blocking key is treated as too common (stop-key)
    # Prevents mega-clusters from generic tokens like "the", "group", "services"
    max_cluster_size: int = 500

    # Minimum length for a secondary attribute value to be a useful blocking key
    min_attribute_value_length: int = 3


@dataclass
class FeatureConfig:
    """Feature engineering toggles and similarity metric configurations."""
    use_jaro_winkler: bool = True
    use_levenshtein: bool = True
    use_token_sort_ratio: bool = True
    use_token_set_ratio: bool = True
    use_jaccard_ngram: bool = True
    ngram_size: int = 3

    # Exact matches and specialized features
    use_exact_match_flags: bool = True
    use_length_ratio: bool = True
    use_missingness_indicator: bool = True
    clean_legal_suffixes: bool = True


@dataclass
class TrainingConfig:
    """Training, cross-validation, and optimization settings."""
    model_type: str = "lightgbm"  # Options: 'lightgbm', 'logistic_regression', 'random_forest'
    cv_folds: int = 5
    random_state: int = 42

    # Competition metric: F-beta with beta=0.5 (weights precision twice as much as recall)
    f_beta: float = 0.5
    optimize_threshold: bool = True
    threshold_search_steps: int = 100

    # Model specific hyperparameters
    lightgbm_params: Dict = field(default_factory=lambda: {
        "n_estimators": 250,
        "learning_rate": 0.05,
        "max_depth": 6,
        "num_leaves": 31,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42,
        "verbose": -1,
    })


@dataclass
class PipelineConfig:
    """Master pipeline configuration combining all modules."""
    project_root: Path = PROJECT_ROOT
    raw_data_dir: Path = RAW_DATA_DIR
    processed_data_dir: Path = PROCESSED_DATA_DIR
    models_dir: Path = MODELS_DIR
    outputs_dir: Path = OUTPUTS_DIR
    predictions_dir: Path = PREDICTIONS_DIR
    reports_dir: Path = REPORTS_DIR

    columns: ColumnMapping = field(default_factory=ColumnMapping)
    blocking: BlockingConfig = field(default_factory=BlockingConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)


# Global default configuration instance
DEFAULT_CONFIG = PipelineConfig()
