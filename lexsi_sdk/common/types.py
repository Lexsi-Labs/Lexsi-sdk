from datetime import datetime
from typing import List, Optional, TypedDict, Dict


class ProjectConfig(TypedDict):
    """Configuration keys required to describe a project."""

    project_type: Optional[str] = None
    model_name: Optional[str] = None
    unique_identifier: str
    true_label: str
    tag: str
    pred_label: Optional[str]
    feature_exclude: Optional[List[str]]
    drop_duplicate_uid: Optional[bool]
    handle_errors: Optional[bool]
    feature_encodings: Optional[dict]
    handle_data_imbalance: Optional[bool]
    sample_percentage: Optional[float] = None
    explainability_method: Optional[List[str]] = None

class DataConfig(TypedDict):
    """Training data configuration for tabular workloads."""

    tags: List[str]
    test_tags: Optional[List[str]]
    use_optuna: Optional[bool] = False
    feature_exclude: List[str]
    feature_encodings: Dict[str, str]
    drop_duplicate_uid: bool
    sample_percentage: float
    explainability_sample_percentage: float
    lime_explainability_iterations: int
    explainability_method: List[str]
    handle_data_imbalance: Optional[bool]

class SyntheticDataConfig(TypedDict):
    """Configuration required when generating synthetic data."""

    model_name: str
    tags: List[str]
    feature_exclude: List[str]
    feature_include: List[str]
    feature_actual_used: List[str]
    drop_duplicate_uid: bool


class SyntheticModelHyperParams(TypedDict):
    """Common hyperparameter keys for supported synthetic models."""

    # GPT2 hyper params
    batch_size: Optional[int]
    early_stopping_patience: Optional[int]
    early_stopping_threshold: Optional[float]
    epochs: Optional[int]
    model_type: Optional[str]
    random_state: Optional[int]
    tabular_config: Optional[str]
    train_size: Optional[float]

    # CTGAN hyper params
    epochs: Optional[int]
    test_ratio: Optional[float]

class GCSConfig(TypedDict):
    """Google Cloud Storage connector configuration."""

    project_id: str
    gcp_project_name: str
    type: str
    private_key_id: str
    private_key: str
    client_email: str
    client_id: str
    auth_uri: str
    token_uri: str

class S3Config(TypedDict):
    """Amazon S3 connector configuration."""

    region: Optional[str] = None
    access_key: str
    secret_key: str

class GDriveConfig(TypedDict):
    """Google Drive connector configuration."""

    project_id: str
    type: str
    private_key_id: str
    private_key: str
    client_email: str
    client_id: str
    auth_uri: str
    token_uri: str

class SFTPConfig(TypedDict):
    """SFTP connector configuration."""

    hostname: str
    port: str
    username: str
    password: str

class CustomServerConfig(TypedDict):
    """Scheduling options when requesting dedicated inference compute."""

    start: Optional[datetime] = None
    stop: Optional[datetime] = None
    shutdown_after: Optional[int] = 1
    op_hours: Optional[bool] = None
    auto_start: bool = False

class InferenceCompute(TypedDict):
    """Inference compute selection payload."""

    instance_type: str
    custom_server_config: Optional[CustomServerConfig] = CustomServerConfig()

class InferenceSettings(TypedDict):
    """Inference settings that can be applied to text models."""

    inference_engine: str
