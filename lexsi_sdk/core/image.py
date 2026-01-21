from datetime import datetime
import io
import json
from typing import Union
from mistralai import Optional
import pandas as pd
from lexsi_sdk.common.constants import MODEL_PERF_DASHBOARD_REQUIRED_FIELDS, MODEL_TYPES
from lexsi_sdk.common.monitoring import ImageDashboardPayload, ModelPerformancePayload
from lexsi_sdk.common.types import CatBoostParams, FoundationalModelParams, LightGBMParams, PEFTParams, ProcessorParams, ProjectConfig, RandomForestParams, TuningParams, XGBoostParams
from lexsi_sdk.common.utils import poll_events
from lexsi_sdk.common.validation import Validate
from lexsi_sdk.common.xai_uris import ALL_DATA_FILE_URI, AVAILABLE_BATCH_SERVERS_URI, CASE_INFO_URI, CREATE_TRIGGER_URI, DASHBOARD_LOGS_URI, DELETE_CASE_URI, DELETE_TRIGGER_URI, DOWNLOAD_TAG_DATA_URI, DUPLICATE_MONITORS_URI, EXECUTED_TRIGGER_URI, GENERATE_DASHBOARD_URI, GET_CASES_URI, GET_DASHBOARD_SCORE_URI, GET_DASHBOARD_URI, GET_EXECUTED_TRIGGER_INFO, GET_MODEL_TYPES_URI, GET_MODELS_URI, GET_MONITORS_ALERTS, GET_PROJECT_CONFIG, GET_TRIGGERS_URI, LIST_DATA_CONNECTORS, MODEL_INFERENCES_URI, MODEL_PARAMETERS_URI, MODEL_PERFORMANCE_DASHBOARD_URI, RUN_MODEL_ON_DATA_URI, SEARCH_CASE_URI, UPLOAD_DATA_FILE_INFO_URI, UPLOAD_DATA_FILE_URI, UPLOAD_DATA_URI, UPLOAD_DATA_WITH_CHECK_URI, UPLOAD_FILE_DATA_CONNECTORS, UPLOAD_MODEL_URI
from lexsi_sdk.core.alert import Alert
from lexsi_sdk.core.case import CaseImage, CaseText
from lexsi_sdk.core.dashboard import DASHBOARD_TYPES, Dashboard
from lexsi_sdk.core.project import Project
from lexsi_sdk.core.utils import build_list_data_connector_url


class ImageProject(Project):
    """Image Project class extending the base Project class with image-specific methods."""

    def config(self) -> str:
        """Retrieve the full configuration of the project, including feature selections, encodings, and tags. Returns a dictionary.

        :return: response
        """
        res = self.api_client.get(
            f"{GET_PROJECT_CONFIG}?project_name={self.project_name}"
        )
        if res.get("details") != "Not Found":
            res["details"].pop("updated_by", None)
            res["details"]["metadata"].pop("path", None)
            res["details"]["metadata"].pop("avaialble_tags", None)

        return res.get("details")
    

    def upload_data(
        self,
        data: str | pd.DataFrame,
        tag: str,
        model: Optional[str] = None,
        model_name: Optional[str] = None,
        model_architecture: Optional[str] = None,
        model_type: Optional[str] = None,
        config: Optional[ProjectConfig] = None,
        model_config: Optional[Union[XGBoostParams, LightGBMParams, CatBoostParams, RandomForestParams, FoundationalModelParams]] = None,
        tunning_config: Optional[TuningParams] = None,
        peft_config: Optional[PEFTParams] = None,
        processor_config: Optional[ProcessorParams] = None,
        finetune_mode: Optional[str] = None,
        tunning_strategy: Optional[str] = None,
        compute_type: Optional[str] = "shared"
    ) -> str:
        """
        Upload dataset(s) to the project and triggers model training.
        If a model is specified, it trains the requested model; otherwise,
        an ``XGBoost`` model is trained by default.

        It executes the full end-to-end training pipeline:

        - dataset upload (tag-based, feature exclusion, sampling)
        - selects and prepares data (filtering, sampling, feature handling, imbalance handling)
        - applies preprocessing / feature engineering (optional)
        - trains either a **classic ML model** or a **tabular foundation model**
        - optionally performs hyperparameter tuning (classic or foundational depending on strategy)
        - optionally performs fine-tuning / PEFT for foundation models
        - produces a trained model artifact and returns its identifier/reference

        :param data: Dataset to upload. Can be a file path or an in-memory pandas DataFrame.
        :type data: str | pandas.DataFrame

        :param tag: Tag associated with the uploaded dataset, used for filtering
            and train/test selection.
        :type tag: str

        :param model: Optional model identifier or alias.
        :type model: str | None

        :param model_name: Optional human-readable name for the trained model.
        :type model_name: str | None

        :param model_architecture: Optional architecture identifier
            (used mainly for foundation models).
        :type model_architecture: str | None

        :param model_type: Type of model to train.

            **Classic ML models**
            - ``XGBoost``
            - ``LightGBM``
            - ``CatBoost``
            - ``RandomForest``
            - ``SGD``
            - ``LogisticRegression``
            - ``LinearRegression``
            - ``GaussianNaiveBayes``

            **Tabular foundation models**
            - ``TabPFN``
            - ``TabICL``
            - ``TabDPT``
            - ``OrionMSP``
            - ``OrionBix``
            - ``Mitra``
            - ``ContextTab``
        :type model_type: str | None

        :param config: Dataset and training configuration controlling feature
            selection, encodings, sampling, and data behavior.
        :type config: ProjectConfig | None

        :param processor_config: Optional preprocessing and feature engineering
            configuration (e.g., imputation, scaling, resampling).
        :type processor_config: ProcessorParams | None

        :param model_config: Hyperparameters for the selected ``model_type``.
            Must match the chosen model family.
        :type model_config: XGBoostParams | LightGBMParams | CatBoostParams |
            RandomForestParams | FoundationalModelParams | None

        :param tunning_config: Optional tuning or adaptation configuration.
        :type tunning_config: TuningParams | None

        :param tunning_strategy: Training or fine-tuning strategy.

            - ``"inference"``: Zero-shot inference only
            - ``"base-ft"`` / ``"finetune"``: Full fine-tuning
            - ``"peft"``: Parameter-efficient fine-tuning (requires ``peft_config``)
        :type tunning_strategy: str | None

        :param finetune_mode: Fine-tuning mode for foundation models.

            - ``"meta-learning"``: Episodic meta-learning
            - ``"sft"``: Standard supervised fine-tuning
        :type finetune_mode: str | None

        :param peft_config: PEFT (e.g., LoRA) configuration, used when
            ``tunning_strategy="peft"``.
        :type peft_config: PEFTParams | None

        :param compute_type: Compute instance used for training.
            Examples: ``"shared"``, ``"small"``, ``"medium"``, ``"large"``,
            ``"T4.small"``, ``"A10G.xmedium"``.
        :type compute_type: str | None

        :return: Identifier or reference to the trained model artifact.
        :rtype: str
        """

        def build_upload_data(data):
            """Build a multipart-upload payload from a file path or DataFrame.
            Converts DataFrames to an in-memory CSV buffer and returns a `(filename, bytes)` tuple.

            :param data: Local file path or a pandas DataFrame.
            :return: A file handle (path input) or `(filename, bytes)` tuple (DataFrame input).
            """
            if isinstance(data, str):
                file = open(data, "rb")
                return file
            elif isinstance(data, pd.DataFrame):
                csv_buffer = io.BytesIO()
                data.to_csv(csv_buffer, index=False, encoding="utf-8")
                csv_buffer.seek(0)
                file_name = f"{tag}_sdk_{datetime.now().replace(microsecond=0)}.csv"
                file = (file_name, csv_buffer.getvalue())
                return file
            else:
                raise Exception("Invalid Data Type")

        def upload_file_and_return_path(data, data_type, tag=None) -> str:
            """Upload a data/model artifact to Lexsi file storage.
            Returns the server-side `filepath` that other project APIs reference.

            :param data: File path or DataFrame to upload.
            :param data_type: Upload type such as `data`, `model`, etc.
            :param tag: Optional tag to associate with the upload.
            :return: Server-side filepath for the uploaded artifact."""
            files = {"in_file": build_upload_data(data)}
            res = self.api_client.file(
                f"{UPLOAD_DATA_FILE_URI}?project_name={self.project_name}&data_type={data_type}&tag={tag}",
                files,
            )

            if not res["success"]:
                raise Exception(res.get("details"))
            uploaded_path = res.get("metadata").get("filepath")

            return uploaded_path

        project_config = self.config()

        if project_config == "Not Found":
            if self.metadata.get("modality") == "image":
                if (
                    not model
                    or not model_architecture
                    or not model_type
                    or not model_name
                ):
                    raise Exception("Model details is required for Image project type")

                uploaded_path = upload_file_and_return_path(data, "data", tag)

                model_uploaded_path = upload_file_and_return_path(model, "model")

                payload = {
                    "project_name": self.project_name,
                    "project_type": self.metadata.get("project_type"),
                    "metadata": {
                        "path": uploaded_path,
                        "model_name": model_name,
                        "model_path": model_uploaded_path,
                        "model_architecture": model_architecture,
                        "model_type": model_type,
                        "tag": tag,
                        "tags": [tag],
                    },
                }

            if self.metadata.get("modality") == "tabular":
                if not config:
                    config = {
                        "project_type": "",
                        "unique_identifier": "",
                        "true_label": "",
                        "pred_label": "",
                        "feature_exclude": [],
                        "drop_duplicate_uid": False,
                        "handle_errors": False,
                        "handle_data_imbalance": False,
                    }
                    raise Exception(
                        f"Project Config is required, since no config is set for project \n {json.dumps(config,indent=1)}"
                    )

                Validate.check_for_missing_keys(
                    config, ["project_type", "unique_identifier", "true_label"]
                )

                Validate.value_against_list(
                    "project_type", config, ["classification", "regression"]
                )

                uploaded_path = upload_file_and_return_path(data, "data", tag)

                file_info = self.api_client.post(
                    UPLOAD_DATA_FILE_INFO_URI, {"path": uploaded_path}
                )

                column_names = file_info.get("details").get("column_names")

                Validate.value_against_list(
                    "unique_identifier",
                    config["unique_identifier"],
                    column_names,
                    lambda: self.delete_file(uploaded_path),
                )

                if config.get("feature_exclude"):
                    Validate.value_against_list(
                        "feature_exclude",
                        config["feature_exclude"],
                        column_names,
                        lambda: self.delete_file(uploaded_path),
                    )

                feature_exclude = [
                    config["unique_identifier"],
                    config["true_label"],
                    *config.get("feature_exclude", []),
                ]

                feature_include = [
                    feature
                    for feature in column_names
                    if feature not in feature_exclude
                ]

                feature_encodings = config.get("feature_encodings", {})
                if feature_encodings:
                    Validate.value_against_list(
                        "feature_encodings_feature",
                        list(feature_encodings.keys()),
                        column_names,
                    )
                    Validate.value_against_list(
                        "feature_encodings_feature",
                        list(feature_encodings.values()),
                        ["labelencode", "countencode", "onehotencode"],
                    )

                payload = {
                    "project_name": self.project_name,
                    "project_type": config["project_type"],
                    "unique_identifier": config["unique_identifier"],
                    "true_label": config["true_label"],
                    "pred_label": config.get("pred_label"),
                    "metadata": {
                        "path": uploaded_path,
                        "tag": tag,
                        "tags": [tag],
                        "drop_duplicate_uid": config.get("drop_duplicate_uid"),
                        "handle_errors": config.get("handle_errors", False),
                        "feature_exclude": feature_exclude,
                        "feature_include": feature_include,
                        "feature_encodings": feature_encodings,
                        "feature_actual_used": [],
                        "handle_data_imbalance": config.get(
                            "handle_data_imbalance", False
                        ),
                    },
                    # "gpu": gpu,
                    "instance_type": compute_type,
                    "sample_percentage": config.get("sample_percentage", None),
                }
                if config.get("model_name"):
                    payload["metadata"]["model_name"] = config.get("model_name")

            if config.get("xai_method"):
                payload["metadata"]["explainability_method"] = config.get(
                    "xai_method"
                )
            if model_config:
                payload["metadata"]["model_parameters"] = model_config
            if tunning_config:
                payload["metadata"]["tunning_parameters"] = tunning_config
            if peft_config:
                payload["metadata"]["peft_parameters"] = peft_config
            if processor_config:
                payload["metadata"]["processor_parameters"] = processor_config
            if finetune_mode:
                payload["metadata"]["finetune_mode"] = finetune_mode
            if tunning_strategy:
                payload["metadata"]["tunning_strategy"] = tunning_strategy
            res = self.api_client.post(UPLOAD_DATA_WITH_CHECK_URI, payload)

            if not res["success"]:
                self.delete_file(uploaded_path)
                raise Exception(res.get("details"))
            try:
                poll_events(self.api_client, self.project_name, res["event_id"])
            except Exception as e:
                self.delete_file(uploaded_path)
                raise e
            return res.get("details")

        if project_config != "Not Found" and config:
            raise Exception("Config already exists, please remove config")

        uploaded_path = upload_file_and_return_path(data, "data", tag)

        payload = {
            "path": uploaded_path,
            "tag": tag,
            "type": "data",
            "project_name": self.project_name,
        }
        res = self.api_client.post(UPLOAD_DATA_URI, payload)

        if not res["success"]:
            self.delete_file(uploaded_path)
            raise Exception(res.get("details"))

        return res.get("details")

    def upload_model_types(self) -> dict:
        """Model types which can be uploaded using upload_model()

        :return: response
        """
        model_types = self.api_client.get(GET_MODEL_TYPES_URI)

        return model_types

    def upload_model(
        self,
        model_path: str,
        model_architecture: str,
        model_type: str,
        model_name: str,
        model_train: list,
        model_test: Optional[list],
        pod: Optional[str] = None,
        xai_method: Optional[list] = ["shap"],
        feature_list: Optional[list] = None,
    ):
        """Uploads a custom trained model to Lexsi.ai for inference and evaluation.

        :param model_path: path of the model
        :param model_architecture: architecture of model ["machine_learning", "deep_learning"]
        :param model_type: type of the model based on the architecture ["Xgboost","Lgboost","CatBoost","Random_forest","Linear_Regression","Logistic_Regression","Gaussian_NaiveBayes","SGD"]
                use upload_model_types() method to get all allowed model_types
        :param model_name: name of the model
        :param model_train: data tags for model
        :param model_test: test tags for model (optional)
        :param pod: pod to be used for uploading model (optional)
        :param explainability_method: explainability method to be used while uploading model ["shap", "lime"] (optional)
        :param feature_list: list of features in sequence which are to be passed in the model (optional)
        """

        def upload_file_and_return_path() -> str:
            """Upload a local model artifact to Lexsi file storage.
            Returns the stored `filepath` referenced by the model upload request."""
            files = {"in_file": open(model_path, "rb")}
            model_data_tags_str = ",".join(model_train)
            res = self.api_client.file(
                f"{UPLOAD_DATA_FILE_URI}?project_name={self.project_name}&data_type=model&tag={model_data_tags_str}",
                files,
            )

            if not res["success"]:
                raise Exception(res.get("details"))
            uploaded_path = res.get("metadata").get("filepath")

            return uploaded_path

        model_types = self.api_client.get(GET_MODEL_TYPES_URI)
        valid_model_architecture = model_types.get("model_architecture").keys()
        Validate.value_against_list(
            "model_achitecture", model_architecture, valid_model_architecture
        )

        valid_model_types = model_types.get("model_architecture")[model_architecture]
        Validate.value_against_list("model_type", model_type, valid_model_types)

        tags = self.tags()
        Validate.value_against_list("model_train", model_train, tags)

        if model_test:
            Validate.value_against_list("model_test", model_test, tags)

        uploaded_path = upload_file_and_return_path()

        if pod:
            custom_batch_servers = self.api_client.get(AVAILABLE_BATCH_SERVERS_URI)
            Validate.value_against_list(
                "pod",
                pod,
                [
                    server["instance_name"]
                    for server in custom_batch_servers.get("details", [])
                ],
            )

        if xai_method:
            Validate.value_against_list(
                "explainability_method", xai_method, ["shap", "lime"]
            )

        payload = {
            "project_name": self.project_name,
            "model_name": model_name,
            "model_architecture": model_architecture,
            "model_type": model_type,
            "model_path": uploaded_path,
            "model_data_tags": model_train,
            "model_test_tags": model_test,
            "explainability_method": xai_method,
            "feature_list": feature_list,
        }

        if pod:
            payload["instance_type"] = pod

        res = self.api_client.post(UPLOAD_MODEL_URI, payload)

        if not res.get("success"):
            raise Exception(res.get("details"))

        poll_events(
            self.api_client,
            self.project_name,
            res["event_id"],
            lambda: self.delete_file(uploaded_path),
        )

    def upload_model_dataconnectors(
        self,
        data_connector_name: str,
        model_architecture: str,
        model_type: str,
        model_name: str,
        model_train: list,
        model_test: Optional[list],
        pod: Optional[str] = None,
        xai_method: Optional[list] = ["shap"],
        bucket_name: Optional[str] = None,
        file_path: Optional[str] = None,
    ):
        """Uploads a custom trained model to Lexsi.ai for inference and evaluation.

        :param data_connector_name: name of the data connector
        :param model_architecture: architecture of model ["machine_learning", "deep_learning"]
        :param model_type: type of the model based on the architecture ["Xgboost","Lgboost","CatBoost","Random_forest","Linear_Regression","Logistic_Regression","Gaussian_NaiveBayes","SGD"]
                use upload_model_types() method to get all allowed model_types
        :param model_name: name of the model
        :param model_train: data tags for model
        :param model_test: test tags for model (optional)
        :param pod: pod to be used for uploading model (optional)
        :param xai_method: explainability method to be used while uploading model ["shap", "lime"] (optional)
        :param bucket_name: if data connector has buckets # Example: s3/gcs buckets
        :param file_path: filepath from the bucket for the data to read
        """

        def get_connector() -> str | pd.DataFrame:
            """Look up the configured data connector by name.
            Returns a one-row DataFrame (or an error string) with connector metadata."""
            url = build_list_data_connector_url(
                LIST_DATA_CONNECTORS, self.project_name, self.organization_id
            )
            res = self.api_client.post(url)

            if res["success"]:
                df = pd.DataFrame(res["details"])
                filtered_df = df.loc[df["link_service_name"] == data_connector_name]
                if filtered_df.empty:
                    return "No data connector found"
                return filtered_df

            return res["details"]

        connectors = get_connector()
        if isinstance(connectors, pd.DataFrame):
            value = connectors.loc[
                connectors["link_service_name"] == data_connector_name,
                "link_service_type",
            ].values[0]
            ds_type = value

            if ds_type == "s3" or ds_type == "gcs":
                if not bucket_name:
                    return "Missing argument bucket_name"
                if not file_path:
                    return "Missing argument file_path"
        else:
            return connectors

        def upload_file_and_return_path() -> str:
            """Trigger a connector-to-Lexsi upload for model artifacts.
            Returns the stored `filepath` referenced by the model upload request."""
            if not self.project_name:
                return "Missing Project Name"
            model_data_tags_str = ",".join(model_train)
            if self.organization_id:
                res = self.api_client.post(
                    f"{UPLOAD_FILE_DATA_CONNECTORS}?project_name={self.project_name}&organization_id={self.organization_id}&link_service_name={data_connector_name}&data_type=model&bucket_name={bucket_name}&file_path={file_path}&tag={model_data_tags_str}"
                )
            else:
                res = self.api_client.post(
                    f"{UPLOAD_FILE_DATA_CONNECTORS}?project_name={self.project_name}&link_service_name={data_connector_name}&data_type=model&bucket_name={bucket_name}&file_path={file_path}&tag={model_data_tags_str}"
                )
            print(res)
            if not res["success"]:
                raise Exception(res.get("details"))
            uploaded_path = res.get("metadata").get("filepath")

            return uploaded_path

        model_types = self.api_client.get(GET_MODEL_TYPES_URI)
        valid_model_architecture = model_types.get("model_architecture").keys()
        Validate.value_against_list(
            "model_achitecture", model_architecture, valid_model_architecture
        )

        valid_model_types = model_types.get("model_architecture")[model_architecture]
        Validate.value_against_list("model_type", model_type, valid_model_types)

        tags = self.tags()
        Validate.value_against_list("model_train", model_train, tags)

        if model_test:
            Validate.value_against_list("model_test", model_test, tags)

        uploaded_path = upload_file_and_return_path()

        if pod:
            custom_batch_servers = self.api_client.get(AVAILABLE_BATCH_SERVERS_URI)
            Validate.value_against_list(
                "pod",
                pod,
                [
                    server["instance_name"]
                    for server in custom_batch_servers.get("details", [])
                ],
            )

        if xai_method:
            Validate.value_against_list(
                "explainability_method", xai_method, ["shap", "lime"]
            )

        payload = {
            "project_name": self.project_name,
            "model_name": model_name,
            "model_architecture": model_architecture,
            "model_type": model_type,
            "model_path": uploaded_path,
            "model_data_tags": model_train,
            "model_test_tags": model_test,
            "explainability_method": xai_method,
        }

        if pod:
            payload["instance_type"] = pod

        res = self.api_client.post(UPLOAD_MODEL_URI, payload)

        if not res.get("success"):
            raise Exception(res.get("details"))

        poll_events(
            self.api_client,
            self.project_name,
            res["event_id"],
            lambda: self.delete_file(uploaded_path),
        )

    
    def model_inference(
        self,
        tag: Optional[str] = None,
        file_name: Optional[str] = None,
        model_name: Optional[str] = None,
        pod: Optional[str] = None
    ) -> pd.DataFrame:
        """Run model inference on tag or file_name data. Either tag or file_name is required for running inference

        :param tag: data tag for running inference
        :param file_name: data file name for running inference
        :param model_name: name of the model, defaults to active model for the project
        :param pod: pod for running inference
        :return: model inference dataframe
        :rtype: pd.DataFrame
        """

        if not tag and not file_name:
            raise Exception("Either tag or file_name is required.")
        if tag and file_name:
            raise Exception("Provide either tag or file_name, not both.")
        available_tags = self.tags()
        if tag and tag not in available_tags:
            raise Exception(
                f"{tag} tag is not valid, select valid tag from :\n{available_tags}"
            )

        files = self.api_client.get(
            f"{ALL_DATA_FILE_URI}?project_name={self.project_name}"
        )
        file_names = []
        for file in files.get("details"):
            file_names.append(file.get("filepath").split("/")[-1])

        if file_name and file_name not in file_names:
            raise Exception(
                f"{file_name} file name is not valid, select valid tag from :\n{file_names.join(',')}"
            )
        filepath = None
        for file in files["details"]:
            file_path = file["filepath"]
            curr_file_name = file_path.split("/")[-1]
            if file_name == curr_file_name:
                filepath = file_path
                break

        models = self.models()

        available_models = models["model_name"].to_list()

        if model_name:
            Validate.value_against_list("model_name", model_name, available_models)

        model = (
            model_name
            or models.loc[models["status"] == "active"]["model_name"].values[0]
        )

        if pod and self.metadata.get("modality") == "tabular":
            custom_batch_servers = self.api_client.get(AVAILABLE_BATCH_SERVERS_URI)
            available_custom_batch_servers = custom_batch_servers.get("details", []) + custom_batch_servers.get("available_gpu_custom_servers", [])
            Validate.value_against_list(
                "pod",
                pod,
                [
                    server["instance_name"]
                    for server in available_custom_batch_servers
                ],
            )
        else:
            Validate.value_against_list(
                "pod",
                pod,
                [
                    server["instance_name"]
                    for server in custom_batch_servers.get("details", [])
                ],
            )

        run_model_payload = {
            "project_name": self.project_name,
            "model_name": model,
            "tags": tag,
            "instance_type": pod
        }
        if filepath:
            run_model_payload["filepath"] = filepath

        run_model_res = self.api_client.post(RUN_MODEL_ON_DATA_URI, run_model_payload)

        if not run_model_res["success"]:
            raise Exception(run_model_res["details"])

        poll_events(
            api_client=self.api_client,
            project_name=self.project_name,
            event_id=run_model_res["event_id"],
        )

        auth_token = self.api_client.get_auth_token()

        if tag:
            uri = f"{DOWNLOAD_TAG_DATA_URI}?project_name={self.project_name}&tag={tag}_{model}_Inference&token={auth_token}"
        else:
            file_name = file_name.replace(".", "_")
            uri = f"{DOWNLOAD_TAG_DATA_URI}?project_name={self.project_name}&tag={file_name}_{model}_Inference&token={auth_token}"
        tag_data = self.api_client.base_request("GET", uri)

        tag_data_df = pd.read_csv(io.StringIO(tag_data.text))

        return tag_data_df

    def model_inferences(self) -> pd.DataFrame:
        """returns model inferences for the project

        :return: model inferences dataframe
        """

        res = self.api_client.get(
            f"{MODEL_INFERENCES_URI}?project_name={self.project_name}"
        )

        if not res["success"]:
            raise Exception(res.get("details"))

        model_inference_df = pd.DataFrame(res["details"]["inference_details"])

        return model_inference_df


    def cases(
        self,
        unique_identifier: Optional[str] = None,
        tag: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        page: Optional[int] = 1,
    ) -> pd.DataFrame:
        """Cases for the Project

        :param unique_identifier: unique identifier of the case for filtering, defaults to None
        :param tag: data tag for filtering, defaults to None
        :param start_date: start date for filtering, defaults to None
        :param end_date: end data for filtering, defaults to None
        :return: casse details dataframe
        """

        def get_cases():
            """Fetch paginated cases without any search filters.
            Used when no identifier/tag/date filters are provided."""
            payload = {
                "project_name": self.project_name,
                "page_num": page,
            }
            res = self.api_client.post(GET_CASES_URI, payload)
            return res

        def search_cases():
            """Search cases using identifier/tag/date filters.
            Posts the filter payload to the search endpoint and returns the raw API response.
            """
            payload = {
                "project_name": self.project_name,
                "unique_identifier": unique_identifier,
                "start_date": start_date,
                "end_date": end_date,
                "tag": tag,
                "page_num": page,
            }
            res = self.api_client.post(SEARCH_CASE_URI, payload)
            return res

        cases = (
            search_cases()
            if unique_identifier or tag or start_date or end_date
            else get_cases()
        )

        if not cases["success"]:
            raise Exception("No cases found")

        cases_df = pd.DataFrame(cases.get("details"))

        return cases_df

    def case_predict(
        self,
        unique_identifier: str,
        case_id: Optional[str] = None,
        tag: Optional[str] = None,
        model_name: Optional[str] = None,
        serverless_type: Optional[str] = None,
        xai: Optional[list] = [],
        risk_policies: Optional[bool] = False,
    ):
        """Case Prediction for given unique identifier

        :param unique_identifier: unique identifier of case
        :param case_id: case id, defaults to None
        :param tag: case tag, defaults to None
        :param model_name: trained model name, defaults to None
        :param serverless_type: instance to be used for case
                Eg:- nova-0.5, nova-1, nova-1.5
        :param xai: xai methods for explainability you want to run
                Eg:- ['shap', 'lime', 'dtree', 'ig', 'gradcam', 'dlb']
        :param risk_policies: Whether to run policies during prediction. Set to True to run policies. Defaults to False.
        :return: Case object with details
        """
        payload = {
            "project_name": self.project_name,
            "case_id": case_id,
            "unique_identifier": unique_identifier,
            "tag": tag,
            "model_name": model_name,
            "instance_type": serverless_type,
            "risk_policies": risk_policies,
            "xai": xai,
        }
        res = self.api_client.post(CASE_INFO_URI, payload)
        if not res["success"]:
            raise Exception(res["details"])

        res["details"]["project_name"] = self.project_name
        res["details"]["api_client"] = self.api_client
        case = CaseImage(**res["details"])
        return case

    def delete_cases(
        self,
        unique_identifier: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        tag: Optional[str] = None,
    ) -> str:
        """Delete Case with given filters. Atleast one filter is required to delete cases.

        :param unique_identifier: unique identifier of case, defaults to None
        :param start_date: start date of case, defaults to None
        :param end_date: end date of case, defaults to None
        :param tag: tag of case, defaults to None
        :return: response
        """
        if tag:
            all_tags = self.all_tags()
            Validate.value_against_list("tag", tag, all_tags)

        paylod = {
            "project_name": self.project_name,
            "unique_identifier": [unique_identifier],
            "start_date": start_date,
            "end_date": end_date,
            "tag": tag,
        }

        res = self.api_client.post(DELETE_CASE_URI, paylod)

        if not res["success"]:
            raise Exception(res["details"])

        return res["details"]


    def get_default_dashboard(self, type: str) -> Dashboard:
        """Returns the default dashboard for the specified type.

        :param type: type of the dashboard
        :return: Dashboard
        """

        res = self.api_client.get(
            f"{GET_DASHBOARD_URI}?type={type}&project_name={self.project_name}"
        )

        if res["success"]:
            auth_token = self.api_client.get_auth_token()
            query_params = f"?project_name={self.project_name}&type={type}&access_token={auth_token}"
            return Dashboard(
                config=res.get("config"),
                raw_data=res.get("details"),
                query_params=query_params,
            )

        raise Exception(
            "Cannot retrieve default dashboard, please create new dashboard"
        )

    def get_all_dashboards(self, type: str, page: Optional[int] = 1) -> pd.DataFrame:
        """Fetch all dashboards in the project

        :param type: type of the dashboard
        :param page: page number defaults to 1
        :return: Result DataFrame
        """

        Validate.value_against_list(
            "type",
            type,
            DASHBOARD_TYPES,
        )

        res = self.api_client.get(
            f"{DASHBOARD_LOGS_URI}?project_name={self.project_name}&type={type}&page={page}",
        )
        if not res["success"]:
            raise Exception(res.get("details", "Failed to get all dashboard"))
        res = res.get("details").get("dashboards")

        logs = pd.DataFrame(res)
        logs.drop(
            columns=[
                "max_features",
                "limit_features",
                "baseline_date",
                "current_date",
                "task_id",
                "date_feature",
                "stat_test_threshold",
                "project_name",
                "file_id",
                "updated_at",
                "features_to_use",
            ],
            inplace=True,
            errors="ignore",
        )
        return logs

    def get_dashboard_metadata(self, type: str, dashboard_id: str) -> Dashboard:
        """Get dashboard generated dashboard with id

        :param type: type of the dashboard
        :param dashboard_id: id of dashboard
        :return: A Dashboard object that provides the following capabilities:
            - plot(): Re-render the dashboard with custom width/height
            - get_config(): Retrieve the dashboard configuration (excluding metadata)
            - get_raw_data(): Access processed metric data underlying the dashboard
            - print_config(): Pretty-print the dashboard configuration
        :rtype: Dashboard
        """
        Validate.value_against_list(
            "type",
            type,
            DASHBOARD_TYPES,
        )

        res = self.api_client.get(
            f"{GET_DASHBOARD_URI}?type={type}&project_name={self.project_name}&dashboard_id={dashboard_id}"
        )

        if not res["success"]:
            raise Exception(res.get("details", "Failed to get dashboard"))

        auth_token = self.api_client.get_auth_token()
        query_params = f"?project_name={self.project_name}&type={type}&dashboard_id={dashboard_id}&access_token={auth_token}"

        return res

    def get_dashboard(self, type: str, dashboard_id: str) -> Dashboard:
        """Get dashboard generated dashboard with id

        :param type: type of the dashboard
        :param dashboard_id: id of dashboard
        :return: A Dashboard object that provides the following capabilities:
            - plot(): Re-render the dashboard with custom width/height
            - get_config(): Retrieve the dashboard configuration (excluding metadata)
            - get_raw_data(): Access processed metric data underlying the dashboard
            - print_config(): Pretty-print the dashboard configuration
        :rtype: Dashboard
        """
        Validate.value_against_list(
            "type",
            type,
            DASHBOARD_TYPES,
        )

        res = self.api_client.get(
            f"{GET_DASHBOARD_URI}?type={type}&project_name={self.project_name}&dashboard_id={dashboard_id}"
        )

        if not res["success"]:
            raise Exception(res.get("details", "Failed to get dashboard"))

        auth_token = self.api_client.get_auth_token()
        query_params = f"?project_name={self.project_name}&type={type}&dashboard_id={dashboard_id}&access_token={auth_token}"

        return Dashboard(
            config=res.get("config"),
            raw_data=res.get("details"),
            query_params=query_params,
        )

    def monitoring_triggers(self) -> pd.DataFrame:
        """List of monitoring triggers for the project.

        :return: DataFrame
        :rtype: pd.DataFrame
        """
        url = f"{GET_TRIGGERS_URI}?project_name={self.project_name}"
        res = self.api_client.get(url)

        if not res["success"]:
            return Exception(res.get("details", "Failed to get triggers"))

        monitoring_triggers = res.get("details", [])

        if not monitoring_triggers:
            return "No monitoring triggers found."

        monitoring_triggers = pd.DataFrame(monitoring_triggers)
        monitoring_triggers = monitoring_triggers[
            monitoring_triggers["deleted"] == False
        ]
        monitoring_triggers = monitoring_triggers.drop("project_name", axis=1)

        return monitoring_triggers

    def duplicate_monitoring_triggers(self, trigger_name: str, new_trigger_name: str) -> str:
        """Duplicate an existing monitoring trigger under a new name.
        Calls the backend duplication endpoint and returns the server response message.

        :param trigger_name: Existing trigger name to duplicate.
        :param new_trigger_name: New name for the duplicated trigger.
        :return: Backend response message.
        :rtype: str"""
        if trigger_name == new_trigger_name:
            return "Duplicate trigger name can't be same"
        url = f"{DUPLICATE_MONITORS_URI}?project_name={self.project_name}&trigger_name={trigger_name}&new_trigger_name={new_trigger_name}"
        res = self.api_client.post(url)

        if not res["success"]:
            return res.get("details", "Failed to clone triggers")

        return res["details"]

    def create_monitor(self, payload: dict) -> str:
        """Create monitoring trigger for project

        :param payload: Data Drift Trigger Payload
                {
                    "trigger_type": ""  #["Data Drift", "Target Drift", "Model Performance"]
                    "trigger_name": "",
                    "mail_list": [],
                    "frequency": "",   #['daily','weekly','monthly','quarterly','yearly']
                    "stat_test_name": "",
                    "stat_test_threshold": 0,
                    "datadrift_features_per": 7,
                    "dataset_drift_percentage": 50,
                    "features_to_use": [],
                    "date_feature": "",
                    "baseline_date": { "start_date": "", "end_date": ""},
                    "current_date": { "start_date": "", "end_date": ""},
                    "base_line_tag": [""],
                    "current_tag": [""],
                    "priority": 2, # between 1-5 
                    "pod": ""  #Pod type to used for running trigger
                } OR Target Drift Trigger Payload
                {
                    "trigger_type": ""  #["Data Drift", "Target Drift", "Model Performance"]
                    "trigger_name": "",
                    "mail_list": [],
                    "frequency": "",   #['daily','weekly','monthly','quarterly','yearly']
                    "model_type": "",
                    "stat_test_name": ""
                    "stat_test_threshold": 0,
                    "date_feature": "",
                    "baseline_date": { "start_date": "", "end_date": ""},
                    "current_date": { "start_date": "", "end_date": ""},
                    "base_line_tag": [""],
                    "current_tag": [""],
                    "baseline_true_label": "",
                    "current_true_label": "",
                    "priority": 2, # between 1-5 
                    "pod": ""  #Pod type to used for running trigger
                } OR Model Performance Trigger Payload
                {
                    "trigger_type": ""  #["Data Drift", "Target Drift", "Model Performance"]
                    "trigger_name": "",
                    "mail_list": [],
                    "frequency": "",   #['daily','weekly','monthly','quarterly','yearly']
                    "model_type": "",
                    "model_performance_metric": "",
                    "model_performance_threshold": "",
                    "date_feature": "",
                    "baseline_date": { "start_date": "", "end_date": ""},
                    "current_date": { "start_date": "", "end_date": ""},
                    "base_line_tag": [""],
                    "baseline_true_label": "",
                    "baseline_pred_label": "",
                    "priority": 2, # between 1-5 
                    "pod": ""  #Pod type to used for running trigger
                }
        :return: response
        """
        payload["project_name"] = self.project_name

        required_payload_keys = [
            "trigger_type",
            "priority",
            "mail_list",
            "frequency",
            "trigger_name",
        ]

        Validate.check_for_missing_keys(payload, required_payload_keys)
        if payload.get("pod", None):
            payload["instance_type"] = payload["pod"]
        payload = {
            "project_name": self.project_name,
            "modify_req": {
                "create_trigger": payload,
            },
        }
        res = self.api_client.post(CREATE_TRIGGER_URI, payload)

        if not res["success"]:
            return Exception(res.get("details", "Failed to create trigger"))

        return "Trigger created successfully."

    def delete_monitoring_trigger(self, name: str) -> str:
        """delete a monitoring trigger from project

        :param name: trigger name
        :return: str
        :rtype: str
        """
        payload = {
            "project_name": self.project_name,
            "modify_req": {
                "delete_trigger": name,
            },
        }

        res = self.api_client.post(DELETE_TRIGGER_URI, payload)

        if not res["success"]:
            return Exception(res.get("details", "Failed to delete trigger"))

        return "Monitoring trigger deleted successfully."

    def alerts(self, page_num: int = 1) -> pd.DataFrame:
        """Retrieves monitoring alerts for the project. Each page contains 20 alerts.

        :param page_num: page num, defaults to 1
        :return: alerts DataFrame
        :rtype: pd.DataFrame
        """
        payload = {"page_num": page_num, "project_name": self.project_name}

        res = self.api_client.post(EXECUTED_TRIGGER_URI, payload)

        if not res["success"]:
            return Exception(res.get("details", "Failed to get alerts"))

        monitoring_alerts = res.get("details", [])

        if not monitoring_alerts:
            return "No monitoring alerts found."

        return pd.DataFrame(monitoring_alerts)

    def get_alert_details(self, id: str) -> Alert:
        """Alert details of the provided id

        :param id: alert or trigger id
        :return: Alert
        :rtype: Alert
        """
        payload = {
            "project_name": self.project_name,
            "id": id,
        }
        res = self.api_client.post(GET_EXECUTED_TRIGGER_INFO, payload)

        if not res["success"]:
            return Exception(res.get("details", "Failed to get trigger details"))

        trigger_info = res["details"][0]

        if not trigger_info["successful"]:
            return Alert(info={}, detailed_report=[], not_used_features=[])

        trigger_info = trigger_info["details"]

        detailed_report = trigger_info.get("detailed_report")
        not_used_features = trigger_info.get("Not_Used_Features")

        trigger_info.pop("detailed_report", None)
        trigger_info.pop("Not_Used_Features", None)

        return Alert(
            info=trigger_info,
            detailed_report=detailed_report,
            not_used_features=not_used_features,
        )

    def get_monitors_alerts(self, monitor_id: str, time: int):
        """Retrieves alerts for a specific monitor within a given time window.

        :param monitor_id: Monitor identifier.
        :param time: Time range (in hours) from the current time used to fetch alerts.
        :return: Alerts as a pandas DataFrame.
        :rtype: pd.DataFrame"""
        url = f"{GET_MONITORS_ALERTS}?project_name={self.project_name}&monitor_id={monitor_id}&time={time}"
        res = self.api_client.get(url)
        data = pd.DataFrame(res.get("details"))
        return data

    def get_model_performance(self, model_name: str = None) -> Dashboard:
        """Get model performance dashboard data for this project.

        :param model_name: Optional model name to filter dashboard data.
        :return: A Dashboard object that provides the following capabilities:
            - plot(): Re-render the dashboard with custom width/height
            - get_config(): Retrieve the dashboard configuration (excluding metadata)
            - get_raw_data(): Access processed metric data underlying the dashboard
            - print_config(): Pretty-print the dashboard configuration
        :rtype: Dashboard"""
        auth_token = self.api_client.get_auth_token()
        dashboard_query_params = f"?type=model_performance&project_name={self.project_name}&access_token={auth_token}"
        raw_data_query_params = f"?project_name={self.project_name}"

        if model_name:
            dashboard_query_params = f"{dashboard_query_params}&model_name={model_name}"
            raw_data_query_params = f"{raw_data_query_params}&model_name={model_name}"

        raw_data = self.api_client.get(
            f"{MODEL_PERFORMANCE_DASHBOARD_URI}{raw_data_query_params}"
        )

        return Dashboard(
            config={},
            query_params=dashboard_query_params,
            raw_data=raw_data.get("details"),
        )

    def model_parameters(self) -> dict:
        """Model Parameters

        :return: response
        """

        model_params = self.api_client.get(MODEL_PARAMETERS_URI)

        return model_params
    
    def get_model_performance_dashboard(
        self,
        payload: ModelPerformancePayload = {},
        pod: Optional[str] = None,
        run_in_background: bool = False,
    ) -> Dashboard:
        """Generate Model Performance Dashboard for the given parameters.

        :param run_in_background: runs in background without waiting for dashboard generation to complete
        :param pod: pod to be used for generating model performance diagnosis (optional)
        :param payload: model performance payload
                {
                    "base_line_tag": [""],
                    "current_tag": [""],
                    "date_feature": "",
                    "baseline_date": { "start_date": "", "end_date": ""},
                    "current_date": { "start_date": "", "end_date": ""},
                    "model_type": "",
                    "baseline_true_label": "",
                    "baseline_pred_label": "",
                    "current_true_label": "",
                    "current_pred_label": ""
                }
                defaults to None
        :return: A Dashboard object that provides the following capabilities:
            - plot(): Re-render the dashboard with custom width/height
            - get_config(): Retrieve the dashboard configuration (excluding metadata)
            - get_raw_data(): Access processed metric data underlying the dashboard
            - print_config(): Pretty-print the dashboard configuration
        :rtype: Dashboard
        """
        if not payload:
            return self.get_default_dashboard("performance")

        payload["project_name"] = self.project_name

        tags_info = self.available_tags()
        all_tags = tags_info["alltags"]

        if self.metadata.get("modality") == "image":
            Validate.check_for_missing_keys(payload, ["base_line_tag", "current_tag"])

        Validate.value_against_list("base_line_tag", payload["base_line_tag"], all_tags)
        Validate.value_against_list("current_tag", payload["current_tag"], all_tags)

        if self.metadata.get("modality") == "tabular":
            Validate.check_for_missing_keys(
                payload, MODEL_PERF_DASHBOARD_REQUIRED_FIELDS
            )
            Validate.validate_date_feature_val(
                payload, tags_info["alldatetimefeatures"]
            )

            Validate.value_against_list(
                "model_type", payload["model_type"], MODEL_TYPES
            )

            Validate.value_against_list(
                "baseline_true_label",
                [payload["baseline_true_label"]],
                tags_info["alluniquefeatures"],
            )

            Validate.value_against_list(
                "baseline_pred_label",
                [payload["baseline_pred_label"]],
                tags_info["alluniquefeatures"],
            )

            Validate.value_against_list(
                "current_true_label",
                [payload["current_true_label"]],
                tags_info["alluniquefeatures"],
            )

            Validate.value_against_list(
                "current_pred_label",
                [payload["current_pred_label"]],
                tags_info["alluniquefeatures"],
            )

        custom_batch_servers = self.api_client.get(AVAILABLE_BATCH_SERVERS_URI)
        Validate.value_against_list(
            "pod",
            pod,
            [
                server["instance_name"]
                for server in custom_batch_servers.get("details", [])
            ],
        )

        if payload.get("pod", None):
            payload["instance_type"] = payload["pod"]
        if pod:
            payload["instance_type"] = pod

        res = self.api_client.post(
            f"{GENERATE_DASHBOARD_URI}?type=performance", payload
        )

        if not res["success"]:
            error_details = res.get("details", "Failed to get dashboard url")
            raise Exception(error_details)

        if not run_in_background:
            poll_events(self.api_client, self.project_name, res["task_id"])
            return self.get_default_dashboard("performance")

        return "Model performance dashboard generation initiated"

    def get_image_property_drift_dashboard(
        self,
        payload: ImageDashboardPayload = {},
        pod: Optional[str] = None,
        run_in_background: bool = False,
    ) -> Dashboard:
        """Generate an Image Property Drift dashboard for this project with given baseline and current tags.

        :param payload: 
                {
                    "base_line_tag": List[str]
                    "current_tag": List[str]
                }
        :param pod: Optional compute instance for generation jobs.
        :param run_in_background: If True, trigger generation and return immediately.
        :return: A Dashboard object that provides the following capabilities:
            - plot(): Re-render the dashboard with custom width/height
            - get_config(): Retrieve the dashboard configuration (excluding metadata)
            - get_raw_data(): Access processed metric data underlying the dashboard
            - print_config(): Pretty-print the dashboard configuration
        :rtype: Dashboard
        """
        if not payload:
            return self.get_default_dashboard("image_property_drift")

        payload["project_name"] = self.project_name

        # validate tags and labels
        tags_info = self.available_tags()
        all_tags = tags_info["alltags"]

        Validate.value_against_list("base_line_tag", payload["base_line_tag"], all_tags)
        Validate.value_against_list("current_tag", payload["current_tag"], all_tags)

        custom_batch_servers = self.api_client.get(AVAILABLE_BATCH_SERVERS_URI)
        Validate.value_against_list(
            "pod",
            pod,
            [
                server["instance_name"]
                for server in custom_batch_servers.get("details", [])
            ],
        )

        if payload.get("pod", None):
            payload["instance_type"] = payload["pod"]
        if pod:
            payload["instance_type"] = pod

        res = self.api_client.post(
            f"{GENERATE_DASHBOARD_URI}?type=image_property_drift", payload
        )

        if not res["success"]:
            error_details = res.get("details", "Failed to generate dashboard")
            raise Exception(error_details)

        if not run_in_background:
            poll_events(self.api_client, self.project_name, res["task_id"])
            return self.get_default_dashboard("image_property_drift")

        return "Image Property Drift dashboard generation initiated"

    def get_label_drift_dashboard(
        self,
        payload: ImageDashboardPayload = {},
        pod: Optional[str] = None,
        run_in_background: bool = False,
    ) -> Dashboard:
        """Generate an Image Label Drift dashboard for this project with given baseline and current tags.

        :param payload: Dashboard configuration payload (tags/labels and parameters).
                {
                    "base_line_tag": List[str]
                    "current_tag": List[str]
                }
        :param instance_type: Optional compute instance for generation jobs.
        :param run_in_background: If True, trigger generation and return immediately.
        :return: A Dashboard object that provides the following capabilities:
            - plot(): Re-render the dashboard with custom width/height
            - get_config(): Retrieve the dashboard configuration (excluding metadata)
            - get_raw_data(): Access processed metric data underlying the dashboard
            - print_config(): Pretty-print the dashboard configuration
        :rtype: Dashboard
        """
        if not payload:
            return self.get_default_dashboard("label_drift")

        payload["project_name"] = self.project_name

        # validate tags and labels
        tags_info = self.available_tags()
        all_tags = tags_info["alltags"]

        Validate.value_against_list("base_line_tag", payload["base_line_tag"], all_tags)
        Validate.value_against_list("current_tag", payload["current_tag"], all_tags)

        custom_batch_servers = self.api_client.get(AVAILABLE_BATCH_SERVERS_URI)
        Validate.value_against_list(
            "pod",
            pod,
            [
                server["instance_name"]
                for server in custom_batch_servers.get("details", [])
            ],
        )

        if payload.get("pod", None):
            payload["instance_type"] = payload["pod"]
        if pod:
            payload["instance_type"] = pod

        res = self.api_client.post(
            f"{GENERATE_DASHBOARD_URI}?type=label_drift", payload
        )

        if not res["success"]:
            error_details = res.get("details", "Failed to generate dashboard")
            raise Exception(error_details)

        if not run_in_background:
            poll_events(self.api_client, self.project_name, res["task_id"])
            return self.get_default_dashboard("label_drift")

        return "Label Drift dashboard generation initiated"

    def get_property_label_correlation_dashboard(
        self,
        payload: ImageDashboardPayload = {},
        pod: Optional[str] = None,
        run_in_background: bool = False,
    ) -> Dashboard:
        """Generate an Image Property Label Correlation dashboard for this project with given baseline and current tags.

        :param payload: Dashboard configuration payload (tags/labels and parameters).
                {
                    "base_line_tag": List[str]
                    "current_tag": List[str]
                }
        :param pod: Optional compute instance for generation jobs.
        :param run_in_background: If True, trigger generation and return immediately.
        :return: A Dashboard object that provides the following capabilities:
            - plot(): Re-render the dashboard with custom width/height
            - get_config(): Retrieve the dashboard configuration (excluding metadata)
            - get_raw_data(): Access processed metric data underlying the dashboard
            - print_config(): Pretty-print the dashboard configuration
        :rtype: Dashboard
        """
        if not payload:
            return self.get_default_dashboard("property_label_correlation")

        payload["project_name"] = self.project_name

        # validate tags and labels
        tags_info = self.available_tags()
        all_tags = tags_info["alltags"]

        Validate.value_against_list("base_line_tag", payload["base_line_tag"], all_tags)
        Validate.value_against_list("current_tag", payload["current_tag"], all_tags)

        custom_batch_servers = self.api_client.get(AVAILABLE_BATCH_SERVERS_URI)
        Validate.value_against_list(
            "pod",
            pod,
            [
                server["instance_name"]
                for server in custom_batch_servers.get("details", [])
            ],
        )

        if payload.get("pod", None):
            payload["instance_type"] = payload["pod"]
        if pod:
            payload["instance_type"] = pod

        res = self.api_client.post(
            f"{GENERATE_DASHBOARD_URI}?type=property_label_correlation", payload
        )

        if not res["success"]:
            error_details = res.get("details", "Failed to generate dashboard")
            raise Exception(error_details)

        if not run_in_background:
            poll_events(self.api_client, self.project_name, res["task_id"])
            return self.get_default_dashboard("property_label_correlation")

        return "Property label correlation dashboard generation initiated"

    def get_image_dataset_drift_dashboard(
        self,
        payload: ImageDashboardPayload = {},
        pod: Optional[str] = None,
        run_in_background: bool = False,
    ) -> Dashboard:
        """Generate an Image Dataset Drift dashboard for this project with given baseline and current tags.

        :param payload: Dashboard configuration payload (tags/labels and parameters).
                {
                    "base_line_tag": List[str]
                    "current_tag": List[str]
                }
        :param pod: Optional compute instance for generation jobs.
        :param run_in_background: If True, trigger generation and return immediately.
        :return: A Dashboard object that provides the following capabilities:
            - plot(): Re-render the dashboard with custom width/height
            - get_config(): Retrieve the dashboard configuration (excluding metadata)
            - get_raw_data(): Access processed metric data underlying the dashboard
            - print_config(): Pretty-print the dashboard configuration
        :rtype: Dashboard
        """
        if not payload:
            return self.get_default_dashboard("image_dataset_drift")

        payload["project_name"] = self.project_name

        # validate tags and labels
        tags_info = self.available_tags()
        all_tags = tags_info["alltags"]

        Validate.value_against_list("base_line_tag", payload["base_line_tag"], all_tags)
        Validate.value_against_list("current_tag", payload["current_tag"], all_tags)

        custom_batch_servers = self.api_client.get(AVAILABLE_BATCH_SERVERS_URI)
        Validate.value_against_list(
            "pod",
            pod,
            [
                server["instance_name"]
                for server in custom_batch_servers.get("details", [])
            ],
        )

        if payload.get("pod", None):
            payload["instance_type"] = payload["pod"]
        if pod:
            payload["instance_type"] = pod

        res = self.api_client.post(
            f"{GENERATE_DASHBOARD_URI}?type=image_dataset_drift", payload
        )

        if not res["success"]:
            error_details = res.get("details", "Failed to generate dashboard")
            raise Exception(error_details)

        if not run_in_background:
            poll_events(self.api_client, self.project_name, res["task_id"])
            return self.get_default_dashboard("image_dataset_drift")

        return "Image Dataset Drift dashboard generation initiated"

