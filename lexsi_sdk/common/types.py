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

class XGBoostParams(TypedDict, total=False):
    """
    XGBoost hyperparameter configuration.

    Keys correspond to common XGBoost training parameters. Provide only the keys
    you want to override.

    :param objective: Learning objective.
        Common values:
        ``"binary:logistic"``, ``"binary:logitraw"``, ``"multi:softprob"``,
        ``"reg:squarederror"``, ``"reg:logistic"``, ``"rank:pairwise"``.
    :type objective: str | None

    :param booster: Booster type.
        Allowed values: ``"gbtree"``, ``"gblinear"``.
    :type booster: Literal["gbtree", "gblinear"] | None

    :param eval_metric: Evaluation metric used during training.
        Common values:
        ``"logloss"``, ``"auc"``, ``"aucpr"``, ``"error"``, ``"rmse"``, ``"mae"``,
        ``"merror"``, ``"mlogloss"``.
    :type eval_metric: str | None

    :param grow_policy: Tree growth policy (tree-based boosters).
        Allowed values: ``"depthwise"``, ``"lossguide"``.
    :type grow_policy: Literal["depthwise", "lossguide"] | None

    :param max_depth: Maximum depth of a tree.
        Typical range: 1–16.
    :type max_depth: int | None

    :param max_leaves: Maximum number of leaves per tree (used with ``lossguide``).
        Typical range: 0–4096 (0 means "no limit" depending on implementation).
    :type max_leaves: int | None

    :param min_child_weight: Minimum sum of instance weight needed in a child.
        Higher values make the model more conservative.
    :type min_child_weight: float | None

    :param colsample_bytree: Subsample ratio of columns when constructing each tree.
        Range: 0.0–1.0.
    :type colsample_bytree: float | None

    :param colsample_bylevel: Subsample ratio of columns for each level.
        Range: 0.0–1.0.
    :type colsample_bylevel: float | None

    :param colsample_bynode: Subsample ratio of columns for each split/node.
        Range: 0.0–1.0.
    :type colsample_bynode: float | None

    :param learning_rate: Step size shrinkage (eta).
        Range: 0.0–1.0 (commonly 0.01–0.3).
    :type learning_rate: float | None

    :param n_estimators: Number of boosting rounds / trees.
        Typical range: 50–5000.
    :type n_estimators: int | None

    :param subsample: Subsample ratio of the training instances.
        Range: 0.0–1.0.
    :type subsample: float | None

    :param alpha: L1 regularization term on weights (reg_alpha).
        Range: >= 0.
    :type alpha: float | None

    :param lambda_: L2 regularization term on weights (reg_lambda).
        Range: >= 0.
    :type lambda_: float | None

    :param seed: Random seed for reproducibility.
    :type seed: int | None
    """
    objective: Optional[str]
    booster: Optional[Literal["gbtree", "gblinear"]]
    eval_metric: Optional[str]
    grow_policy: Optional[Literal["depthwise", "lossguide"]]
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

class LightGBMParams(TypedDict, total=False):
    """
    LightGBM hyperparameter configuration.

    Keys correspond to common LightGBM training parameters. Provide only the keys
    you want to override.

    :param boosting_type: Boosting algorithm type.
        Allowed values: ``"gbdt"``, ``"dart"``.
    :type boosting_type: Literal["gbdt", "dart"] | None

    :param num_leaves: Maximum number of leaves in one tree.
        Typical range: 16–1024.
    :type num_leaves: int | None

    :param max_depth: Maximum tree depth.
        Use -1 for no limit (LightGBM convention) if your training code supports it.
    :type max_depth: int | None

    :param learning_rate: Learning rate (shrinkage).
        Range: 0.0–1.0 (commonly 0.01–0.3).
    :type learning_rate: float | None

    :param n_estimators: Number of boosting iterations / trees.
        Typical range: 50–5000.
    :type n_estimators: int | None

    :param min_child_samples: Minimum number of data points in a leaf.
        Typical range: 5–200.
    :type min_child_samples: int | None

    :param min_child_weight: Minimum sum of hessian in one leaf.
        Range: >= 0.
    :type min_child_weight: float | None

    :param min_split_gain: Minimum gain required to make a split.
        Range: >= 0.
    :type min_split_gain: float | None

    :param subsample: Subsample ratio of the training instances (a.k.a. bagging_fraction).
        Range: 0.0–1.0.
    :type subsample: float | None

    :param colsample_bytree: Subsample ratio of columns when constructing each tree
        (a.k.a. feature_fraction).
        Range: 0.0–1.0.
    :type colsample_bytree: float | None

    :param tree_learner: Tree learning algorithm.
        Allowed values: ``"serial"``, ``"voting"``, ``"data"``, ``"feature"``.
    :type tree_learner: Literal["serial", "voting", "data", "feature"] | None

    :param class_weight: Class weights.
        Allowed values: ``"balanced"``.
    :type class_weight: Literal["balanced"] | None

    :param random_state: Random seed for reproducibility.
    :type random_state: int | None
    """

    boosting_type: Optional[Literal["gbdt", "dart"]]
    num_leaves: Optional[int]
    max_depth: Optional[int]
    learning_rate: Optional[float]
    n_estimators: Optional[int]
    min_child_samples: Optional[int]
    min_child_weight: Optional[float]
    min_split_gain: Optional[float]
    subsample: Optional[float]
    colsample_bytree: Optional[float]
    tree_learner: Optional[Literal["serial", "voting", "data", "feature"]]
    class_weight: Optional[Literal["balanced"]]
    random_state: Optional[int]

class CatBoostParams(TypedDict, total=False):
    """
    CatBoost hyperparameter configuration.

    Provide only the keys you want to override.

    :param iterations: Number of boosting iterations.
        Typical range: 100–50000.
    :type iterations: int | None

    :param learning_rate: Learning rate.
        Range: 0.0–1.0 (commonly 0.01–0.3).
    :type learning_rate: float | None

    :param depth: Depth of the tree.
        Typical range: 1–16.
    :type depth: int | None

    :param subsample_cb: Subsample ratio of training data (CatBoost).
        Range: 0.0–1.0.
    :type subsample_cb: float | None

    :param colsample_bylevel_cb: Subsample ratio of columns per level (CatBoost).
        Range: 0.0–1.0.
    :type colsample_bylevel_cb: float | None

    :param min_data_in_leaf: Minimum number of samples in a leaf node.
        Typical range: 1–200.
    :type min_data_in_leaf: int | None
    """
    iterations: Optional[int]
    learning_rate: Optional[float]
    depth: Optional[int]
    subsample_cb: Optional[float]
    colsample_bylevel_cb: Optional[float]
    min_data_in_leaf: Optional[int]

class RandomForestParams(TypedDict, total=False):
    """
    RandomForest hyperparameter configuration.

    Provide only the keys you want to override.

    :param max_depth: Maximum depth of the tree.
        Use ``None`` for unlimited depth (if supported by your training wrapper).
    :type max_depth: int | None

    :param max_features: Number of features to consider when looking for the best split.
        Allowed values:
        - ``"auto"`` (implementation-dependent; often same as ``"sqrt"`` for classification)
        - ``"sqrt"``
        - ``"log2"``
        - ``int`` (absolute number of features)
        - ``float`` (fraction of features, 0.0–1.0)
    :type max_features: int | float | Literal["auto", "sqrt", "log2"] | None

    :param max_leaf_nodes: Maximum number of leaf nodes.
    :type max_leaf_nodes: int | None

    :param min_samples_leaf: Minimum number of samples required to be at a leaf node.
        Typical range: 1–50.
    :type min_samples_leaf: int | None

    :param min_samples_split: Minimum number of samples required to split an internal node.
        Typical range: 2–200.
    :type min_samples_split: int | None

    :param n_estimators: Number of trees in the forest.
        Typical range: 10–5000.
    :type n_estimators: int | None

    :param criterion: Function to measure the quality of a split.
        Allowed values:
        - Classification: ``"gini"``, ``"entropy"``
        - Regression: ``"squared_error"``, ``"mse"``
    :type criterion: Literal["gini", "entropy", "mse", "squared_error"] | None
    """
    max_depth: Optional[int]
    max_features: Optional[Union[int, float, Literal["auto", "sqrt", "log2"]]]
    max_leaf_nodes: Optional[int]
    min_samples_leaf: Optional[int]
    min_samples_split: Optional[int]
    n_estimators: Optional[int]
    criterion: Optional[Literal["gini", "entropy", "mse", "squared_error"]]

class FoundationalModelParams(TypedDict, total=False):
    """
    Tabular foundational model configuration (TabTune Library).

    This config is used when ``model_type`` is one of the foundational models
    (e.g., ``TabPFN``, ``TabICL``, ``TabDPT``, ``OrionMSP``, ``OrionBix``,
    ``Mitra``, ``ContextTab``). It controls execution device, fitting behavior,
    reproducibility, and probability calibration.

    Notes
    -----
    - This wrapper passes these fields into the underlying TabTune model runner.
      Unsupported fields are ignored or may raise validation errors depending on
      wrapper strictness.
    - Some foundational models may not use all fields (e.g., ``n_estimators``).

    :param device: Execution device for the foundational model.
        Supported by this wrapper:
        - ``"cpu"``: Force CPU execution
        - ``"cuda"``: Force GPU execution
        - ``"auto"``: Select device automatically
    :type device: Literal["cpu", "cuda", "auto"] | None

    :param fit_mode: Controls what is "fit" during the training stage.
        Common wrapper modes:
        - ``"fit_preprocessors"``: fit only preprocessing / encoders
        - ``"fit_model"``: fit the foundational model (and preprocessors if needed)
        - ``"fit_all"``: run full pipeline fitting (preprocessors + model)
        If your wrapper only supports a subset, document/validate accordingly.
    :type fit_mode: str | None

    :param n_estimators: Number of estimators / ensemble members (if supported by the model).
        For models that don’t use ensembles, this may be ignored.
    :type n_estimators: int | None

    :param n_jobs: Number of parallel jobs/threads to use.
        Use ``-1`` to utilize all available cores.
    :type n_jobs: int | None

    :param random_state: Random seed for reproducibility.
    :type random_state: int | None

    :param softmax_temperature: Temperature applied to logits before softmax for
        probability calibration.
        - ``1.0`` keeps probabilities unchanged
        - ``< 1.0`` sharpens probabilities
        - ``> 1.0`` smooths probabilities
    :type softmax_temperature: float | None
    """

    device: Optional[Literal["cpu", "cuda", "auto"]]
    fit_mode: Optional[str]
    n_estimators: Optional[int]
    n_jobs: Optional[int]
    random_state: Optional[int]
    softmax_temperature: Optional[float]

class TuningParams(TypedDict, total=False):
    """
    Hyperparameter tuning / fine-tuning configuration (TabTune wrapper).

    This config controls optimization loops used in:
    - meta-learning or episodic training
    - few-shot adaptation
    - iterative fine-tuning / search

    Notes
    -----
    - These fields may be used only for foundational models (depending on the
      wrapper logic).
    - If both episodic (support/query) and standard training fields are provided,
      the wrapper should define precedence clearly.

    :param epochs: Number of training epochs.
        Typical range: 1–200 (depends on model and dataset size).
    :type epochs: int | None

    :param learning_rate: Learning rate used during optimization.
        Common range: 1e-5–1e-1.
    :type learning_rate: float | None

    :param batch_size: Number of samples per batch.
        Typical range: 8–4096 depending on model and memory.
    :type batch_size: int | None

    :param support_size: Number of support samples per episode (few-shot).
        Example: 16, 32, 64.
    :type support_size: int | None

    :param query_size: Number of query samples per episode (few-shot).
        Example: 16, 32, 64.
    :type query_size: int | None

    :param n_episodes: Number of episodes for meta-learning / episodic training.
        Typical range: 50–5000.
    :type n_episodes: int | None

    :param steps_per_epoch: Number of optimization steps per epoch.
        If not provided, the wrapper may infer it from dataset size and batch size.
    :type steps_per_epoch: int | None
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
    Parameter-Efficient Fine-Tuning (PEFT) configuration (TabTune wrapper).

    This config enables lightweight adaptation (e.g., LoRA) for foundational models.
    It is typically used when ``tuning_strategy="peft"``.

    Notes
    -----
    - Applies only to models/backbones that support PEFT in the wrapper.
    - If the underlying model does not support PEFT, these options may be ignored
      or raise an error depending on wrapper strictness.
    - All parameters are optional; unspecified values fall back to defaults.

    :param r: Rank of the low-rank adaptation matrices (LoRA rank).
        Typical values: ``4``, ``8``, ``16``, ``32``.
        Default: ``8``.
    :type r: int | None

    :param lora_alpha: Scaling factor for LoRA layers.
        Typical values: ``8``, ``16``, ``32``, ``64``.
        Default: ``16``.
    :type lora_alpha: int | None

    :param lora_dropout: Dropout rate applied within LoRA layers.
        Range: ``0.0`` – ``0.5`` (commonly ``0.0`` – ``0.1``).
        Default: ``0.05``.
    :type lora_dropout: float | None
    """

    r: Optional[int]
    lora_alpha: Optional[int]
    lora_dropout: Optional[float]


class ProcessorParams(TypedDict, total=False):
    """
    Data preprocessing and feature engineering configuration.

    These parameters control how input data is cleaned and transformed prior to training.
    The wrapper typically applies them before fitting either classical ML models or
    foundational tabular models.

    :param imputation_strategy: Strategy to handle missing values.
        Supported by this wrapper:
        - ``"mean"``: numerical mean
        - ``"median"``: numerical median
        - ``"mode"``: most frequent value
        - ``"knn"``: kNN-based imputation
    :type imputation_strategy: Literal["mean", "median", "mode", "knn"] | None

    :param scaling_strategy: Feature scaling method.
        Supported by this wrapper:
        - ``"standard"``: standardization (z-score)
        - ``"minmax"``: min-max scaling
        - ``"robust"``: robust scaling (median/IQR)
    :type scaling_strategy: Literal["standard", "minmax", "robust"] | None

    :param resampling_strategy: Strategy to address class imbalance (classification).
        Supported by this wrapper:
        - ``"smote"``: SMOTE oversampling
        - ``"random_oversample"``: random oversampling
        - ``"none"``: do not resample
    :type resampling_strategy: Literal["smote", "random_oversample", "none"] | None
    """

    imputation_strategy: Optional[Literal["mean", "median", "mode", "knn"]]
    scaling_strategy: Optional[Literal["standard", "minmax", "robust"]]
    resampling_strategy: Optional[Literal["smote", "random_oversample", "none"]]

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
