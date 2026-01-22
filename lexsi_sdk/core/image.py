from __future__ import annotations
from datetime import datetime
import io
import json
from typing import Dict, List, Union, Optional
import httpx
from pydantic import BaseModel, ConfigDict
import pandas as pd
import plotly.graph_objects as go
from lexsi_sdk.common.constants import MODEL_PERF_DASHBOARD_REQUIRED_FIELDS, MODEL_TYPES
from lexsi_sdk.common.monitoring import ImageDashboardPayload, ModelPerformancePayload
from lexsi_sdk.common.types import CatBoostParams, FoundationalModelParams, LightGBMParams, PEFTParams, ProcessorParams, ProjectConfig, RandomForestParams, TuningParams, XGBoostParams
from lexsi_sdk.common.utils import poll_events
from lexsi_sdk.common.validation import Validate
from lexsi_sdk.common.xai_uris import ALL_DATA_FILE_URI, AVAILABLE_BATCH_SERVERS_URI, CASE_INFO_URI, CREATE_TRIGGER_URI, DASHBOARD_LOGS_URI, DELETE_CASE_URI, DELETE_TRIGGER_URI, DOWNLOAD_TAG_DATA_URI, DUPLICATE_MONITORS_URI, EXECUTED_TRIGGER_URI, GENERATE_DASHBOARD_URI, GET_CASES_URI, GET_DASHBOARD_SCORE_URI, GET_DASHBOARD_URI, GET_EXECUTED_TRIGGER_INFO, GET_MODEL_TYPES_URI, GET_MODELS_URI, GET_MONITORS_ALERTS, GET_PROJECT_CONFIG, GET_TRIGGERS_URI, GET_TRIGGERS_DAYS_URI, LIST_DATA_CONNECTORS, MODEL_INFERENCES_URI, MODEL_PARAMETERS_URI, MODEL_PERFORMANCE_DASHBOARD_URI, RUN_MODEL_ON_DATA_URI, SEARCH_CASE_URI, UPLOAD_DATA_FILE_INFO_URI, UPLOAD_DATA_FILE_URI, UPLOAD_DATA_PROJECT_URI, UPLOAD_DATA_URI, UPLOAD_DATA_WITH_CHECK_URI, UPLOAD_FILE_DATA_CONNECTORS, UPLOAD_MODEL_URI
from lexsi_sdk.core.alert import Alert
from lexsi_sdk.core.dashboard import DASHBOARD_TYPES, Dashboard
from lexsi_sdk.core.project import Project
from lexsi_sdk.core.utils import build_list_data_connector_url
from lexsi_sdk.client.client import APIClient

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
    ) -> str:
        """
        Upload dataset(s) to the project and triggers model training.

        It executes the full end-to-end training pipeline:

        - dataset upload (tag-based, feature exclusion, sampling)
        - selects and prepares data (filtering, sampling, feature handling, imbalance handling)
        - applies preprocessing / feature engineering (optional)
        - trains uploaded model
        - produces a trained model artifact and returns its identifier/reference

        :param data: Dataset to upload. Can be a file path or an in-memory pandas DataFrame.
        :type data: str | pandas.DataFrame

        :param tag: Tag associated with the uploaded dataset, used for filtering
            and train/test selection.
        :type tag: str

        :param model: model path to upload.
        :type model: str | None

        :param model_name: human-readable name for the trained model.
        :type model_name: str | None

        :param model_architecture: architecture identifier currently supported only deep_learning model architecture.
        :type model_architecture: str | None

        :param model_type: Type of model to train.

            **Model Type**
            -``tensorflow``
        :type model_type: str | None

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

    def upload_data_dataconnectors(
        self,
        data_connector_name: str,
        tag: str,
        model_path: Optional[str] = None,
        model_name: Optional[str] = None,
        model_architecture: Optional[str] = None,
        model_type: Optional[str] = None,
        bucket_name: Optional[str] = None,
        file_path: Optional[str] = None,
    ) -> str:
        """Uploads data for the current project with data connectors
        :param data_connector_name: name of the data connector
        :param tag: Tag associated with the uploaded dataset, used for filtering and train/test selection.
        :param model_path: model path to upload
        :param model_name: human-readable name for the trained model.
        :param bucket_name: if data connector has buckets # Example: s3/gcs buckets
        :param file_path: filepath from the bucket for the data to read
        :return: response
        """
        print("Preparing Data Upload")

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

        def upload_file_and_return_path(file_path, data_type, tag=None) -> str:
            """Trigger a connector-to-Lexsi upload for a file path.
            Returns the stored `filepath` in Lexsi storage to be referenced by other APIs.

            :param file_path: Source path in the connector (bucket/object path, sftp path, etc.).
            :param data_type: Upload type such as `data`, `model`, etc.
            :param tag: Optional tag to associate with the upload.
            :return: Server-side filepath for the uploaded artifact."""
            if not self.project_name:
                return "Missing Project Name"
            query_params = f"project_name={self.project_name}&link_service_name={data_connector_name}&data_type={data_type}&tag={tag}&bucket_name={bucket_name}&file_path={file_path}"
            if self.organization_id:
                query_params += f"&organization_id={self.organization_id}"
            res = self.api_client.post(f"{UPLOAD_FILE_DATA_CONNECTORS}?{query_params}")
            if not res["success"]:
                raise Exception(res.get("details"))
            uploaded_path = res.get("metadata").get("filepath")

            return uploaded_path

        project_config = self.config()

        if project_config == "Not Found":
            if (
                not model_path
                or not model_architecture
                or not model_type
                or not model_name
            ):
                raise Exception("Model details is required for Image project type")

            uploaded_path = upload_file_and_return_path(file_path, "data", tag)

            model_uploaded_path = upload_file_and_return_path(model_path, "model")

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
            res = self.api_client.post(UPLOAD_DATA_WITH_CHECK_URI, payload)

            if not res["success"]:
                self.delete_file(uploaded_path)
                raise Exception(res.get("details"))

            poll_events(self.api_client, self.project_name, res["event_id"])

            return res.get("details")

        uploaded_path = upload_file_and_return_path(file_path, "data", tag)

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
    ):
        """Uploads a custom trained model to Lexsi.ai for inference and evaluation.

        :param model_path: path of the model
        :param model_architecture: architecture of model ["deep_learning"]
        :param model_type: type of the model based on the architecture ["tensorflow"]
                use upload_model_types() method to get all allowed model_types
        :param model_name: name of the model
        :param model_train: data tags for model
        :param model_test: test tags for model (optional)
        :param pod: pod to be used for uploading model (optional)
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

        payload = {
            "project_name": self.project_name,
            "model_name": model_name,
            "model_architecture": model_architecture,
            "model_type": model_type,
            "model_path": uploaded_path,
            "model_data_tags": model_train,
            "model_test_tags": model_test
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
        file_path: str,
        model_test: Optional[list],
        pod: Optional[str] = None,
        bucket_name: Optional[str] = None,
    ):
        """Uploads a custom trained model to Lexsi.ai for inference and evaluation.

        :param data_connector_name: name of the data connector
        :param model_architecture: architecture of model ["deep_learning"]
        :param model_type: type of the model based on the architecture ["tensorflow"]
                use upload_model_types() method to get all allowed model_types
        :param model_name: name of the model
        :param model_train: data tags for model
        :param model_test: test tags for model (optional)
        :param pod: pod to be used for uploading model (optional)
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

        payload = {
            "project_name": self.project_name,
            "model_name": model_name,
            "model_architecture": model_architecture,
            "model_type": model_type,
            "model_path": uploaded_path,
            "model_data_tags": model_train,
            "model_test_tags": model_test,
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
        
        custom_batch_servers = self.api_client.get(AVAILABLE_BATCH_SERVERS_URI)
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
        xai: Optional[list] = []
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
        :return: Case object with details
        """
        payload = {
            "project_name": self.project_name,
            "case_id": case_id,
            "unique_identifier": unique_identifier,
            "tag": tag,
            "model_name": model_name,
            "instance_type": serverless_type,
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

    def monitors(self) -> pd.DataFrame:
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

    def duplicate_monitor(self, monitor_name: str, new_monitor_name: str) -> str:
        """Duplicate an existing monitoring trigger under a new name.
        Calls the backend duplication endpoint and returns the server response message.

        :param monitor_name: Existing monitor name to duplicate.
        :param new_monitor_name: New name for the duplicated monitor.
        :return: Backend response message.
        :rtype: str"""
        if monitor_name == new_monitor_name:
            return "Duplicate trigger name can't be same"
        url = f"{DUPLICATE_MONITORS_URI}?project_name={self.project_name}&trigger_name={monitor_name}&new_trigger_name={new_monitor_name}"
        res = self.api_client.post(url)

        if not res["success"]:
            return res.get("details", "Failed to clone triggers")

        return res["details"]

    def create_monitor(self, payload: dict) -> str:
        """Create monitoring trigger for project

        :param payload: **Data Drift Trigger Payload**
            .. code-block:: json
                {
                    "trigger_type": "Data Drift",
                    "trigger_name": "",
                    "mail_list": [],
                    "frequency": "daily",
                    "stat_test_name": "",
                    "stat_test_threshold": 0,
                    "datadrift_features_per": 7,
                    "dataset_drift_percentage": 50,
                    "features_to_use": [],
                    "date_feature": "",
                    "baseline_date": {"start_date": "", "end_date": ""},
                    "current_date": {"start_date": "", "end_date": ""},
                    "base_line_tag": [""],
                    "current_tag": [""],
                    "priority": 2,
                    "pod": ""
                }

            **Target Drift Trigger Payload**
            .. code-block:: json

                {
                    "trigger_type": "Target Drift",
                    "trigger_name": "",
                    "mail_list": [],
                    "frequency": "daily",
                    "model_type": "",
                    "stat_test_name": "",
                    "stat_test_threshold": 0,
                    "date_feature": "",
                    "baseline_date": {"start_date": "", "end_date": ""},
                    "current_date": {"start_date": "", "end_date": ""},
                    "base_line_tag": [""],
                    "current_tag": [""],
                    "baseline_true_label": "",
                    "current_true_label": "",
                    "priority": 2,
                    "pod": ""
                }

            **Model Performance Trigger Payload**
            .. code-block:: json

                {
                    "trigger_type": "Model Performance",
                    "trigger_name": "",
                    "mail_list": [],
                    "frequency": "daily",
                    "model_type": "",
                    "model_performance_metric": "",
                    "model_performance_threshold": "",
                    "date_feature": "",
                    "baseline_date": {"start_date": "", "end_date": ""},
                    "current_date": {"start_date": "", "end_date": ""},
                    "base_line_tag": [""],
                    "baseline_true_label": "",
                    "baseline_pred_label": "",
                    "priority": 2,
                    "pod": ""
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

    def delete_monitor(self, name: str) -> str:
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
        pod: Optional[str] = "small",
        run_in_background: bool = False,
    ) -> Dashboard:
        """Generate an Image Label Drift dashboard for this project with given baseline and current tags.

        :param payload: Dashboard configuration payload (tags/labels and parameters).
                {
                    "base_line_tag": List[str]
                    "current_tag": List[str]
                }
        :param pod: Optional pod for generation jobs.
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
    
    def register_case(
        self,
        token: str,
        client_id: str,
        unique_identifier: Optional[str] = None,
        project_name: str = None,
        tag: Optional[str] = None,
        image_class: Optional[str] = None,
        serverless_instance_type: Optional[str] = None,
        xai: Optional[str] = None,
        file_path: Optional[str] = None,
    ) -> dict:
        """
        Register a new case entry using an image file path for the project and return the computed result.

        :param token: Lexsi API authentication token
        :param client_id: Lexsi username or client ID
        :param unique_identifier: Filename or unique identifier for the image case
        :param project_name: Target project name for this image case
        :param tag: Dataset tag to associate with this upload or prediction
        :param image_class: Ground-truth class label, if available
        :param serverless_type: Serverless instance type (e.g., NOVA, GOVA, or local)
        :param xai: Explainability technique to run (e.g., SHAP, LIME, IG, Grad-CAM)
        :param file_path: Path to the image file to register (file name must be unique and not previously uploaded)

        :return: Response containing the prediction results for the registered case
        """

        form_data = {
            "client_id": client_id,
            "project_name": project_name,
            "unique_identifier": unique_identifier,
            "tag": tag,
            "image_class": image_class,
            "serverless_instance_type": serverless_instance_type,
            "xai": xai,
        }
        headers = {"x-api-token": token}
        form_data = {k: v for k, v in form_data.items() if v is not None}
        files = {}
        if file_path:
            files["in_file"] = open(file_path, "rb")

        with httpx.Client(http2=True, timeout=None) as client:
            response = client.post(
                self.env.get_base_url() + "/" + UPLOAD_DATA_PROJECT_URI,
                data=form_data,
                files=files or None,
                headers=headers,
            )
            response.raise_for_status()
            response = response.json()

        if files:
            files["in_file"].close()
        return response


class CaseImage(BaseModel):
    """Represents an explainability case for a prediction. Provides visualization helpers such as SHAP, LIME, Integrated Gradients, GradCAM"""

    status: str
    true_value: str | int
    pred_value: str | int
    pred_category: str | int
    model_name: str
    final_decision: Optional[str] = ""
    unique_identifier: Optional[str] = ""
    tag: Optional[str] = ""
    created_at: Optional[str] = ""
    data: Optional[Dict] = {}
    similar_cases_data: Optional[List] = []
    audit_trail: Optional[dict] = {}
    project_name: Optional[str] = ""
    image_data: Optional[Dict] = {}
    data_id: Optional[str] = ""
    summary: Optional[str] = ""

    api_client: APIClient

    def __init__(self, **kwargs):
        """Capture API client used to fetch additional explainability data.
        Stores configuration and prepares the object for use."""
        super().__init__(**kwargs)
        self.api_client = kwargs.get("api_client")

    def inference_output(self) -> pd.DataFrame:
        """Return a DataFrame summarizing the final decision for the case, including the true value, predicted value, predicted category, and final decision.

        :return: decision dataframe
        """
        data = {
            "True Value": self.true_value,
            "Prediction Value": self.pred_value,
            "Prediction Category": self.pred_category,
            "Final Prediction": self.final_decision,
        }
        decision_df = pd.DataFrame([data])

        return decision_df

    def xai_gradcam(self):
        """Visualize Grad-CAM results for image data, showing heatmaps and superimposed regions that contributed to the prediction."""
        if not self.image_data.get("gradcam", None):
            return "No Gradcam method found for this case"
        fig = go.Figure()

        fig.add_layout_image(
            dict(
                source=self.image_data.get("gradcam", {}).get("heatmap"),
                xref="x",
                yref="y",
                x=0,
                y=1,
                sizex=1,
                sizey=1,
                xanchor="left",
                yanchor="top",
                layer="below",
            )
        )

        fig.add_layout_image(
            dict(
                source=self.image_data.get("gradcam", {}).get("superimposed"),
                xref="x",
                yref="y",
                x=1.2,
                y=1,
                sizex=1,
                sizey=1,
                xanchor="left",
                yanchor="top",
                layer="below",
            )
        )

        fig.add_annotation(
            x=0.5,
            y=0.1,
            text="Attributions",
            showarrow=False,
            font=dict(size=16),
            xref="x",
            yref="y",
        )
        fig.add_annotation(
            x=1.7,
            y=0.1,
            text="Superimposed",
            showarrow=False,
            font=dict(size=16),
            xref="x",
            yref="y",
        )
        fig.update_layout(
            xaxis=dict(visible=False, range=[0, 2.5]),
            yaxis=dict(visible=False, range=[0, 1]),
            margin=dict(l=30, r=30, t=30, b=30),
        )

        fig.show(config={"displaylogo": False})

    def xai_shap(self):
        """Render a SHAP attribution plot for image cases, displaying attributions as an overlay on the original image."""
        if not self.image_data.get("shap", None):
            return "No Shap method found for this case"
        fig = go.Figure()

        fig.add_layout_image(
            dict(
                source=self.image_data.get("shap", {}).get("plot"),
                xref="x",
                yref="y",
                x=0,
                y=1,
                sizex=1,
                sizey=1,
                xanchor="left",
                yanchor="top",
                layer="below",
            )
        )

        fig.update_layout(
            xaxis=dict(visible=False, range=[0, 2.5]),
            yaxis=dict(visible=False, range=[0, 1]),
            margin=dict(l=30, r=30, t=30, b=30),
        )

        fig.show(config={"displaylogo": False})

    def xai_lime(self):
        """Render a LIME attribution plot for image cases, displaying attributions as an overlay on the original image."""
        if not self.image_data.get("lime", None):
            return "No Lime method found for this case"
        fig = go.Figure()

        fig.add_layout_image(
            dict(
                source=self.image_data.get("lime", {}).get("plot"),
                xref="x",
                yref="y",
                x=0,
                y=1,
                sizex=1,
                sizey=1,
                xanchor="left",
                yanchor="top",
                layer="below",
            )
        )

        fig.update_layout(
            xaxis=dict(visible=False, range=[0, 2.5]),
            yaxis=dict(visible=False, range=[0, 1]),
            margin=dict(l=30, r=30, t=30, b=30),
        )

        fig.show(config={"displaylogo": False})

    def xai_ig(self):
        """Render an integrated gradients attribution plot for image cases, showing positive and negative attributions side-by-side."""
        if not self.image_data.get("integrated_gradients", None):
            return "No Integrated Gradients method found for this case"
        fig = go.Figure()

        fig.add_layout_image(
            dict(
                source=self.image_data.get("integrated_gradients", {}).get(
                    "attributions"
                ),
                xref="x",
                yref="y",
                x=0,
                y=1,
                sizex=1,
                sizey=1,
                xanchor="left",
                yanchor="top",
                layer="below",
            )
        )

        fig.add_layout_image(
            dict(
                source=self.image_data.get("integrated_gradients", {}).get(
                    "positive_attributions"
                ),
                xref="x",
                yref="y",
                x=1.2,
                y=1,
                sizex=1,
                sizey=1,
                xanchor="left",
                yanchor="top",
                layer="below",
            )
        )

        fig.add_layout_image(
            dict(
                source=self.image_data.get("integrated_gradients", {}).get(
                    "negative_attributions"
                ),
                xref="x",
                yref="y",
                x=2.4,
                y=1,
                sizex=1,
                sizey=1,
                xanchor="left",
                yanchor="top",
                layer="below",
            )
        )

        fig.add_annotation(
            x=0.5,
            y=0.1,
            text="Attributions",
            showarrow=False,
            font=dict(size=16),
            xref="x",
            yref="y",
        )
        fig.add_annotation(
            x=1.7,
            y=0.1,
            text="Positive Attributions",
            showarrow=False,
            font=dict(size=16),
            xref="x",
            yref="y",
        )
        fig.add_annotation(
            x=2.9,
            y=0.1,
            text="Negative Attributions",
            showarrow=False,
            font=dict(size=16),
            xref="x",
            yref="y",
        )
        fig.update_layout(
            xaxis=dict(visible=False, range=[0, 2.5]),
            yaxis=dict(visible=False, range=[0, 1]),
            margin=dict(l=30, r=30, t=30, b=30),
        )

        fig.show(config={"displaylogo": False})

    def alerts_trail(self, page_num: Optional[int] = 1, days: Optional[int] = 7):
        """Fetch alerts for this case over the given window.
        Encapsulates a small unit of SDK logic and returns the computed result."""
        if days == 7:
            return pd.DataFrame(self.audit_trail.get("alerts", {}))
        resp = self.api_client.post(
            f"{GET_TRIGGERS_DAYS_URI}?project_name={self.project_name}&page_num={page_num}&days={days}"
        )
        if resp.get("details"):
            return pd.DataFrame(resp.get("details"))
        else:
            return "No alerts found."

    def audit(self):
        """Return stored audit trail.
        Encapsulates a small unit of SDK logic and returns the computed result."""
        return self.audit_trail

