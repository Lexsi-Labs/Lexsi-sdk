from typing import List, Optional, TypedDict


class ImageDashboardPayload(TypedDict):
    """Payload schema for image monitoring dashboards."""

    base_line_tag: List[str]
    current_tag: List[str]


class DataDriftPayload(TypedDict):
    """Payload schema for data drift dashboards."""

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
    """Payload schema for target drift dashboards."""

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
    """Payload schema for bias monitoring dashboards."""

    project_name: str
    base_line_tag: List[str]

    date_feature: Optional[str]
    baseline_date: Optional[dict]
    current_date: Optional[dict]

    baseline_true_label: str
    baseline_pred_label: str

    features_to_use: List[str]
    model_type: str


class ModelPerformancePayload(TypedDict):
    """Payload schema for model performance dashboards."""

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
