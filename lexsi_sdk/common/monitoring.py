from typing import List, Optional, TypedDict


class ImageDashboardPayload(TypedDict):
    """Payload schema for image monitoring dashboards.

    :param base_line_tag: Baseline dataset tags to compare against.
    :type base_line_tag: List[str]
    :param current_tag: Current dataset tags to compare.
    :type current_tag: List[str]
    """

    base_line_tag: List[str]
    current_tag: List[str]


class DataDriftPayload(TypedDict):
    """Payload schema for data drift dashboards.

    :param project_name: Project identifier,
    :type project_name: Optional[str]
    :param base_line_tag: List of tags to be used for Baseline dataset.
    :type base_line_tag: List[str]
    :param current_tag: List of tags to be used for Current dataset.
    :type current_tag: List[str]
    :param date_feature: Optional feature used for Date Feature.
    :type date_feature: Optional[str]
    :param baseline_date: Optional baseline date range filter { "start_date": "", "end_date": ""}.
    :type baseline_date: Optional[dict]
    :param current_date: Optional current date range filter { "start_date": "", "end_date": ""}.
    :type current_date: Optional[dict]
    :param features_to_use: List of feature names to be used for drift analysis.
    :type features_to_use: List[str]
    :param stat_test_name: Statistical test name for drift detection.
    key values for payload:
            stat_test_name
                ``chisquare`` (Chi-Square test):
                    default for categorical features if the number of labels for feature > 2
                    only for categorical features
                    returns p_value
                    default threshold 0.05
                    drift detected when p_value < threshold
                ``jensenshannon`` (Jensen-Shannon distance):
                    for numerical and categorical features
                    returns distance
                    default threshold 0.05
                    drift detected when distance >= threshold
                ``ks`` (Kolmogorov–Smirnov (K-S) test):
                    default for numerical features
                    only for numerical features
                    returns p_value
                    default threshold 0.05
                    drift detected when p_value < threshold
                ``kl_div`` (Kullback-Leibler divergence):
                    for numerical and categorical features
                    returns divergence
                    default threshold 0.05
                    drift detected when divergence >= threshold,
                ``psi`` (Population Stability Index):
                    for numerical and categorical features
                    returns psi_value
                    default_threshold=0.1
                    drift detected when psi_value >= threshold
                ``wasserstein`` (Wasserstein distance (normed)):
                    only for numerical features
                    returns distance
                    default threshold 0.05
                    drift detected when distance >= threshold
                ``z`` (Ztest):
                    default for categorical features if the number of labels for feature <= 2
                    only for categorical features
                    returns p_value
                    default threshold 0.05
                    drift detected when p_value < threshold
    :type stat_test_name: str
    :param stat_test_threshold: Threshold for the statistical test.
    :type stat_test_threshold: str
    """

    project_name: Optional[str]
    base_line_tag: List[str]
    current_tag: List[str]

    date_feature: Optional[str]
    baseline_date: Optional[dict]
    current_date: Optional[dict]

    features_to_use: List[str]

    stat_test_name: str
    stat_test_threshold: str


class TargetDriftPayload(TypedDict):
    """Payload schema for target drift dashboards.

    :param project_name: Project identifier,
    :type project_name: str
    :param base_line_tag: List of tags to be used for Baseline dataset.
    :type base_line_tag: List[str]
    :param current_tag: List of tags to be used for Current dataset.
    :type current_tag: List[str]
    :param date_feature: Optional feature used for Date Feature.
    :type date_feature: Optional[str]
    :param baseline_date: Optional baseline date range filter { "start_date": "", "end_date": ""}.
    :type baseline_date: Optional[dict]
    :param current_date: Optional current date range filter { "start_date": "", "end_date": ""}.
    :type current_date: Optional[dict]
    :param model_type: Model type for drift analysis ``classification``, ``regression``.
    :type model_type: str
    :param baseline_true_label: Baseline true label column name.
    :type baseline_true_label: str
    :param current_true_label: Current true label column name.
    :type current_true_label: str
    :param stat_test_name: Statistical test name for drift detection.
    key values for payload:
            stat_test_name
                ``chisquare`` (Chi-Square test):
                    default for categorical features if the number of labels for feature > 2
                    only for categorical features
                    returns p_value
                    default threshold 0.05
                    drift detected when p_value < threshold
                ``jensenshannon`` (Jensen-Shannon distance):
                    for numerical and categorical features
                    returns distance
                    default threshold 0.05
                    drift detected when distance >= threshold
                ``ks`` (Kolmogorov–Smirnov (K-S) test):
                    default for numerical features
                    only for numerical features
                    returns p_value
                    default threshold 0.05
                    drift detected when p_value < threshold
                ``kl_div`` (Kullback-Leibler divergence):
                    for numerical and categorical features
                    returns divergence
                    default threshold 0.05
                    drift detected when divergence >= threshold,
                ``psi`` (Population Stability Index):
                    for numerical and categorical features
                    returns psi_value
                    default_threshold=0.1
                    drift detected when psi_value >= threshold
                ``wasserstein`` (Wasserstein distance (normed)):
                    only for numerical features
                    returns distance
                    default threshold 0.05
                    drift detected when distance >= threshold
                ``z`` (Ztest):
                    default for categorical features if the number of labels for feature <= 2
                    only for categorical features
                    returns p_value
                    default threshold 0.05
                    drift detected when p_value < threshold
        :type stat_test_name: str
    :param stat_test_threshold: Threshold for the statistical test.
    :type stat_test_threshold: float
    """

    project_name: str
    base_line_tag: List[str]
    current_tag: List[str]

    date_feature: Optional[str]
    baseline_date: Optional[dict]
    current_date: Optional[dict]

    model_type: str

    baseline_true_label: str
    current_true_label: str

    stat_test_name: str
    stat_test_threshold: float


class BiasMonitoringPayload(TypedDict):
    """Payload schema for bias monitoring dashboards.

    :param project_name: Project identifier,
    :type project_name: str
    :param base_line_tag: List of tags to be used for Baseline dataset.
    :type base_line_tag: List[str]
    :param date_feature: Optional feature used for Date Feature.
    :type date_feature: Optional[str]
    :param baseline_date: Optional baseline date range filter { "start_date": "", "end_date": ""}.
    :type baseline_date: Optional[dict]
    :param model_type: Model type for drift analysis ``classification``, ``regression``.
    :type model_type: str
    :param baseline_true_label: Baseline true label column name.
    :type baseline_true_label: str
    :param baseline_pred_label: Baseline predicted label column name.
    :type baseline_pred_label: str
    :param features_to_use: List of feature names to be used for bias analysis.
    :type features_to_use: List[str]
    :param stat_test_threshold: Threshold for the statistical test.
    :type stat_test_threshold: Optional[str]
    """

    project_name: str
    base_line_tag: List[str]

    date_feature: Optional[str]
    baseline_date: Optional[dict]

    baseline_true_label: str
    baseline_pred_label: str

    features_to_use: List[str]
    model_type: str


class ModelPerformancePayload(TypedDict):
    """Payload schema for model performance dashboards.

    :param project_name: Project identifier.
    :type project_name: str
    :param base_line_tag: list of Baseline dataset tags to compare against.
    :type base_line_tag: List[str]
    :param current_tag: list of Current dataset tags to compare.
    :type current_tag: List[str]
    :param date_feature: Optional feature used for temporal filtering.
    :type date_feature: Optional[str]
    :param baseline_date: Optional baseline date range filter { "start_date": "", "end_date": ""}.
    :type baseline_date: Optional[dict]
    :param current_date: Optional current date range filter { "start_date": "", "end_date": ""}.
    :type current_date: Optional[dict]
    :param baseline_true_label: Baseline true label column name.
    :type baseline_true_label: str
    :param baseline_pred_label: Baseline predicted label column name.
    :type baseline_pred_label: str
    :param current_true_label: Current true label column name.
    :type current_true_label: str
    :param current_pred_label: Current predicted label column name.
    :type current_pred_label: str
    :param model_type: Model type for performance evaluation.
    :type model_type: str
    """

    project_name: str
    base_line_tag: List[str]
    current_tag: List[str]

    date_feature: Optional[str]
    baseline_date: Optional[dict]
    current_date: Optional[dict]

    baseline_true_label: str
    baseline_pred_label: str
    current_true_label: str
    current_pred_label: str

    model_type: str
