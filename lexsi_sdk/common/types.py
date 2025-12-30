from datetime import datetime
from typing import Any, List, Literal, Optional, TypedDict, Dict, Union


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

class ClassicModelParams(TypedDict, total=False):
    """
    Unified configuration parameters for all non-foundational models.

    This TypedDict covers a wide range of models including:
    XGBoost (classification & regression), LightGBM, CatBoost,
    RandomForest, PEFT, TabDPT, Mitra, ContextTab, and others.
    All fields are optional to allow flexible usage per model type.

    -------------------------------
    XGBoost Parameters
    -------------------------------
    :param objective: Learning objective. Examples: "binary:logistic", "reg:squarederror".
    :param booster: Booster type. Options: "gbtree", "gblinear".
    :param eval_metric: Evaluation metric. Examples: "logloss", "auc", "rmse".
    :param grow_policy: Tree growth policy. Options: "depthwise", "lossguide".
    :param max_depth: Maximum depth of the tree.
    :param max_leaves: Maximum number of leaves per tree.
    :param min_child_weight: Minimum sum of instance weight needed in a child.
    :param colsample_bytree: Subsample ratio of columns per tree.
    :param colsample_bylevel: Subsample ratio of columns per level.
    :param colsample_bynode: Subsample ratio of columns per node.
    :param learning_rate: Step size shrinkage used in update to prevents overfitting.
    :param n_estimators: Number of boosting rounds.
    :param subsample: Subsample ratio of training instance.
    :param alpha: L1 regularization term on weights.
    :param lambda_: L2 regularization term on weights.
    :param seed: Random seed for reproducibility.

    -------------------------------
    LightGBM Parameters
    -------------------------------
    :param boosting_type: Type of boosting algorithm. Options: "gbdt", "dart".
    :param num_leaves: Maximum number of leaves in one tree.
    :param min_child_samples: Minimum number of data needed in a child.
    :param min_child_weight: Minimum sum of instance weight in a child.
    :param min_split_gain: Minimum gain to perform a split.
    :param tree_learner: Tree learning algorithm. Options: "serial", "voting", "data", "feature".
    :param class_weight: Class weights. Option: "balanced".

    -------------------------------
    CatBoost Parameters
    -------------------------------
    :param iterations: Number of boosting iterations.
    :param depth: Depth of the tree.
    :param colsample_bylevel_cb: Subsample ratio of columns per level (CatBoost).
    :param min_data_in_leaf: Minimum data in a leaf node.
    :param subsample_cb: Subsample ratio of training data (CatBoost).

    -------------------------------
    RandomForest Parameters
    -------------------------------
    :param max_features: Maximum features considered for split. Options: int, float, "auto", "sqrt", "log2".
    :param max_leaf_nodes: Maximum number of leaf nodes.
    :param min_samples_leaf: Minimum number of samples per leaf.
    :param min_samples_split: Minimum number of samples to split a node.
    :param criterion: Function to measure quality of split. Options: "gini", "entropy", "mse", "squared_error".

    """
    # -------------------------------
    # XGBoost
    # -------------------------------
    objective: Optional[str]  # e.g., 'binary:logistic', 'reg:squarederror'
    booster: Optional[str]  # 'gbtree', 'gblinear'
    eval_metric: Optional[str]  # 'logloss', 'auc', 'rmse'
    grow_policy: Optional[str]  # 'depthwise', 'lossguide'
    max_depth: Optional[int]
    max_leaves: Optional[int]
    min_child_weight: Optional[float]
    colsample_bytree: Optional[float]
    colsample_bylevel: Optional[float]
    colsample_bynode: Optional[float]
    learning_rate: Optional[float]
    n_estimators: Optional[int]
    subsample: Optional[float]
    alpha: Optional[float]
    lambda_: Optional[float]
    seed: Optional[int]

    # -------------------------------
    # LightGBM
    # -------------------------------
    boosting_type: Optional[Literal["gbdt", "dart"]]
    num_leaves: Optional[int]
    min_child_samples: Optional[int]
    min_child_weight: Optional[float]
    min_split_gain: Optional[float]
    tree_learner: Optional[Literal["serial", "voting", "data", "feature"]]
    class_weight: Optional[Literal["balanced"]]

    # -------------------------------
    # CatBoost
    # -------------------------------
    iterations: Optional[int]
    depth: Optional[int]
    colsample_bylevel_cb: Optional[float]
    min_data_in_leaf: Optional[int]
    subsample_cb: Optional[float]

    # -------------------------------
    # RandomForest
    # -------------------------------
    max_features: Optional[Union[int, float, Literal["auto", "sqrt", "log2"]]]
    max_leaf_nodes: Optional[int]
    min_samples_leaf: Optional[int]
    min_samples_split: Optional[int]
    criterion: Optional[str]

class FoundationalModelParams(TypedDict, total=False):
    """
    Core model configuration parameters.

    These parameters control model execution, reproducibility,
    and high-level training behavior.

    :param device: Device on which the model will run.
        Supported values: ``"cpu"``, ``"cuda"``, ``"auto"``.
    :type device: Literal["cpu", "cuda", "auto"]

    :param fit_mode: Mode controlling how the model is trained or fitted.
        Example values: ``"fit_preprocessors"``, ``"fit_model"``.
    :type fit_mode: str

    :param n_estimators: Number of estimators or ensemble members.
    :type n_estimators: int

    :param n_jobs: Number of parallel jobs to run.
        Use ``-1`` to utilize all available cores.
    :type n_jobs: int

    :param random_state: Random seed for reproducibility.
    :type random_state: int

    :param softmax_temperature: Temperature parameter applied to softmax
        for probability calibration.
    :type softmax_temperature: float
    """

    device: Optional[Literal["cpu", "cuda", "auto"]]
    fit_mode: Optional[str]
    n_estimators: Optional[int]
    n_jobs: Optional[int]
    random_state: Optional[int]
    softmax_temperature: Optional[float]

class TuningParams(TypedDict, total=False):
    """
    Hyperparameter tuning and fine-tuning configuration.

    These parameters are primarily used during meta-learning,
    few-shot training, or iterative optimization.

    :param epochs: Number of training epochs.
    :type epochs: int

    :param learning_rate: Learning rate used during optimization.
    :type learning_rate: float

    :param batch_size: Number of samples processed per batch.
    :type batch_size: int

    :param support_size: Number of support samples in few-shot learning.
    :type support_size: int

    :param query_size: Number of query samples in few-shot learning.
    :type query_size: int

    :param n_episodes: Number of episodes in meta-learning.
    :type n_episodes: int

    :param steps_per_epoch: Training steps per epoch.
    :type steps_per_epoch: int
    """

    epochs: Optional[int]
    learning_rate: Optional[float]
    batch_size: Optional[int]
    support_size: Optional[int]
    query_size: Optional[int]
    n_episodes: Optional[int]
    steps_per_epoch: Optional[int]

class PEFTParams(TypedDict, total=False):
    """
    Parameter-Efficient Fine-Tuning (PEFT) configuration.

    These parameters control lightweight adaptation strategies
    such as LoRA.

    :param r: Rank of the low-rank adaptation matrices.
    :type r: int

    :param lora_alpha: Scaling factor applied to LoRA layers.
    :type lora_alpha: int

    :param lora_dropout: Dropout rate applied within LoRA layers.
    :type lora_dropout: float
    """

    r: Optional[int]
    lora_alpha: Optional[int]
    lora_dropout: Optional[float]


class ProcessorParams(TypedDict, total=False):
    """
    Data preprocessing and feature engineering configuration.

    These parameters control how input data is cleaned,
    transformed, and balanced prior to training.

    :param imputation_strategy: Strategy used to handle missing values.
        Supported values: ``"mean"``, ``"median"``, ``"mode"``, ``"knn"``.
    :type imputation_strategy: str

    :param scaling_strategy: Feature scaling method.
        Supported values: ``"standard"``, ``"minmax"``, ``"robust"``.
    :type scaling_strategy: str

    :param resampling_strategy: Strategy used to address class imbalance.
        Supported values: ``"smote"``, ``"random_oversample"``.
    :type resampling_strategy: str
    """

    imputation_strategy: str
    scaling_strategy: str
    resampling_strategy: str

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
