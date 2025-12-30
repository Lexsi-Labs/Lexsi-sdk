from datetime import datetime
from typing import List, Optional, TypedDict, Dict


class ProjectConfig(TypedDict):
    """
    Configuration keys required to describe a project.

    :param project_type: Project type identifier.
    :type project_type: str | None

    :param model_name: Model name associated with the project.
    :type model_name: str | None

    :param unique_identifier: Column name used as the unique identifier.
    :type unique_identifier: str

    :param true_label: Column name containing ground-truth labels.
    :type true_label: str

    :param tag: Column name used to tag/filter records.
    :type tag: str

    :param pred_label: Column name containing predicted labels (if present).
    :type pred_label: str | None

    :param feature_exclude: Features to exclude from training/inference.
    :type feature_exclude: list[str] | None

    :param drop_duplicate_uid: Drop duplicate records based on the unique identifier.
    :type drop_duplicate_uid: bool | None

    :param handle_errors: Whether to handle/ignore errors during processing.
    :type handle_errors: bool | None

    :param feature_encodings: Mapping of feature names to encoding strategies.
    :type feature_encodings: dict | None

    :param handle_data_imbalance: Apply imbalance handling (e.g., SMOTE).
    :type handle_data_imbalance: bool | None

    :param sample_percentage: Fraction of data used for training (0.0–1.0).
    :type sample_percentage: float | None

    :param explainability_method: Explainability methods to apply.
    :type explainability_method: list[str] | None
    """

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
    """
    Configuration controlling data selection, preprocessing, sampling,
    imbalance handling, and explainability.

    :param tags: Tags used to filter training data.
    :type tags: list[str] | None

    :param test_tags: Tags used to construct the test/holdout dataset.
    :type test_tags: list[str] | None

    :param feature_exclude: Features to exclude from training.
    :type feature_exclude: list[str] | None

    :param feature_encodings: Mapping of feature names to encoding strategies.
        Example: ``{"feature_a": "labelencode", "feature_b": "countencode"}``
    :type feature_encodings: dict[str, str] | None

    :param drop_duplicate_uid: Drop duplicate records based on a unique identifier.
    :type drop_duplicate_uid: bool

    :param use_optuna: Enable Optuna for hyperparameter optimization.
    :type use_optuna: bool

    :param sample_percentage: Fraction of data used for training (0.0–1.0).
    :type sample_percentage: float

    :param explainability_sample_percentage: Fraction of data used for explainability computations.
    :type explainability_sample_percentage: float

    :param lime_explainability_iterations: Number of LIME perturbation iterations.
    :type lime_explainability_iterations: int

    :param explainability_method: Explainability method to apply.
        Supported values: ``"shap"``, ``"lime"``.
    :type explainability_method: Literal["shap", "lime"] | None

    :param handle_data_imbalance: Apply SMOTE to address class imbalance.
    :type handle_data_imbalance: bool
    """

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
    """
    Configuration required when generating synthetic data.

    :param model_name: Synthetic model name (e.g., CTGAN/GPT2 tabular).
    :type model_name: str

    :param tags: Tags used to filter source data.
    :type tags: list[str]

    :param feature_exclude: Features to exclude from synthetic training/generation.
    :type feature_exclude: list[str]

    :param feature_include: Features to include for synthetic training/generation.
    :type feature_include: list[str]

    :param feature_actual_used: Final set of features actually used (post-validation).
    :type feature_actual_used: list[str]

    :param drop_duplicate_uid: Drop duplicate records based on a unique identifier.
    :type drop_duplicate_uid: bool
    """

    model_name: str
    tags: List[str]
    feature_exclude: List[str]
    feature_include: List[str]
    feature_actual_used: List[str]
    drop_duplicate_uid: bool


class SyntheticModelHyperParams(TypedDict):
    """
    Common hyperparameter keys for supported synthetic models.

    GPT2-related keys:

    :param batch_size: Training batch size.
    :type batch_size: int | None

    :param early_stopping_patience: Epochs to wait before early stopping.
    :type early_stopping_patience: int | None

    :param early_stopping_threshold: Minimum improvement threshold for early stopping.
    :type early_stopping_threshold: float | None

    :param epochs: Training epochs.
    :type epochs: int | None

    :param model_type: Model type identifier.
    :type model_type: str | None

    :param random_state: Random seed.
    :type random_state: int | None

    :param tabular_config: Tabular configuration identifier/name.
    :type tabular_config: str | None

    :param train_size: Fraction of data used for training (0.0–1.0).
    :type train_size: float | None

    CTGAN-related keys:

    :param test_ratio: Fraction of data used for validation/testing (0.0–1.0).
    :type test_ratio: float | None
    """

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
    """
    Google Cloud Storage connector configuration.

    :param project_id: GCP project identifier.
    :type project_id: str

    :param gcp_project_name: GCP project name.
    :type gcp_project_name: str

    :param type: Credentials type.
    :type type: str

    :param private_key_id: Service account private key ID.
    :type private_key_id: str

    :param private_key: Service account private key PEM string.
    :type private_key: str

    :param client_email: Service account email.
    :type client_email: str

    :param client_id: Service account client ID.
    :type client_id: str

    :param auth_uri: Auth URI.
    :type auth_uri: str

    :param token_uri: Token URI.
    :type token_uri: str
    """

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
    """
    Amazon S3 connector configuration.

    :param region: AWS region (e.g., ``"us-east-1"``).
    :type region: str | None

    :param access_key: AWS access key ID.
    :type access_key: str

    :param secret_key: AWS secret access key.
    :type secret_key: str
    """

    region: Optional[str] = None
    access_key: str
    secret_key: str


class GDriveConfig(TypedDict):
    """
    Google Drive connector configuration.

    :param project_id: GCP project identifier.
    :type project_id: str

    :param type: Credentials type.
    :type type: str

    :param private_key_id: Service account private key ID.
    :type private_key_id: str

    :param private_key: Service account private key PEM string.
    :type private_key: str

    :param client_email: Service account email.
    :type client_email: str

    :param client_id: Service account client ID.
    :type client_id: str

    :param auth_uri: Auth URI.
    :type auth_uri: str

    :param token_uri: Token URI.
    :type token_uri: str
    """

    project_id: str
    type: str
    private_key_id: str
    private_key: str
    client_email: str
    client_id: str
    auth_uri: str
    token_uri: str


class SFTPConfig(TypedDict):
    """
    SFTP connector configuration.

    :param hostname: SFTP host.
    :type hostname: str

    :param port: SFTP port.
    :type port: str

    :param username: SFTP username.
    :type username: str

    :param password: SFTP password.
    :type password: str
    """

    hostname: str
    port: str
    username: str
    password: str


class CustomServerConfig(TypedDict):
    """
    Scheduling options when requesting dedicated inference compute.

    :param start: Start time for the server.
    :type start: datetime | None

    :param stop: Stop time for the server.
    :type stop: datetime | None

    :param shutdown_after: Auto-shutdown timeout (in hours).
    :type shutdown_after: int | None

    :param op_hours: Whether to restrict to business hours.
    :type op_hours: bool | None

    :param auto_start: Automatically start the server when requested.
    :type auto_start: bool
    """

    start: Optional[datetime] = None
    stop: Optional[datetime] = None
    shutdown_after: Optional[int] = 1
    op_hours: Optional[bool] = None
    auto_start: bool = False


class InferenceCompute(TypedDict):
    """
    Inference compute selection payload.

    :param instance_type: Instance type identifier.
    :type instance_type: str

    :param custom_server_config: Optional scheduling configuration.
    :type custom_server_config: CustomServerConfig | None
    """

    instance_type: str
    custom_server_config: Optional[CustomServerConfig] = CustomServerConfig()


class InferenceSettings(TypedDict):
    """
    Inference settings that can be applied to text models.

    :param inference_engine: Inference engine identifier (e.g., provider/runtime name).
    :type inference_engine: str
    """

    inference_engine: str
