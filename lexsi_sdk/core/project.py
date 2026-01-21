from __future__ import annotations
import re
import json
from pydantic import BaseModel
from typing import Dict, List, Optional, Union
from lexsi_sdk.client.client import APIClient
from lexsi_sdk.common.types import (
    ProjectConfig,
    GCSConfig,
    S3Config,
    GDriveConfig,
    SFTPConfig,
)
from lexsi_sdk.common.utils import parse_datetime, parse_float, poll_events
from lexsi_sdk.common.validation import Validate
import pandas as pd
from lexsi_sdk.common.xai_uris import (
    ALL_DATA_FILE_URI,
    AVAILABLE_CUSTOM_SERVERS_URI,
    AVAILABLE_TAGS_URI,
    CASE_LOGS_URI,
    CLEAR_NOTIFICATIONS_URI,
    DELETE_DATA_FILE_URI,
    DOWNLOAD_TAG_DATA_URI,
    FETCH_EVENTS,
    GET_MODELS_URI,
    GET_NOTIFICATIONS_URI,
    GET_VIEWED_CASE_URI,
    MODEL_SUMMARY_URI,
    PROJECT_OVERVIEW_TEXT_URI,
    REMOVE_MODEL_URI,
    START_CUSTOM_SERVER_URI,
    STOP_CUSTOM_SERVER_URI,
    TAG_DATA_URI,
    UPDATE_ACTIVE_MODEL_URI,
    UPDATE_PROJECT_URI,
    UPLOAD_DATA_FILE_INFO_URI,
    UPLOAD_DATA_URI,
    UPLOAD_DATA_WITH_CHECK_URI,
    CREATE_DATA_CONNECTORS,
    LIST_DATA_CONNECTORS,
    DELETE_DATA_CONNECTORS,
    TEST_DATA_CONNECTORS,
    LIST_BUCKETS,
    LIST_FILEPATHS,
    UPLOAD_FILE_DATA_CONNECTORS,
    DROPBOX_OAUTH,
    VALIDATE_POLICY_URI,
)
import io
from lexsi_sdk.core.alert import Alert
from lexsi_sdk.core.case import CaseImage, CaseTabular, CaseText
from lexsi_sdk.core.dashboard import DASHBOARD_TYPES, Dashboard
from datetime import datetime
from lexsi_sdk.core.model_summary import ModelSummary
from lexsi_sdk.core.utils import build_url, build_list_data_connector_url


class Project(BaseModel):
    """Represents a project within a workspace. Provides APIs for model monitoring, explainability cases, alerts, dashboards, and data uploads."""

    organization_id: Optional[str] = None
    created_by: str
    project_name: str
    user_project_name: str
    user_workspace_name: str
    workspace_name: str
    metadata: dict

    api_client: APIClient

    def __new__(cls, *args, **kwargs):
        modality = kwargs.get("metadata", {}).get("modality")

        if cls is Project and modality == "tabular":
            from lexsi_sdk.core.tabular import TabularProject
            return super().__new__(TabularProject)
        elif cls is Project and modality == "image":
            from lexsi_sdk.core.image import ImageProject
            return super().__new__(ImageProject)

        return super().__new__(cls)

    def __init__(self, **kwargs):
        """Initialize a `Project` instance and attach the API client.
        Populates model fields from `kwargs` and stores `api_client` for later requests.

        :param kwargs: Project fields used to construct the instance (including `api_client`).
        """
        super().__init__(**kwargs)
        self.api_client = kwargs.get("api_client")

    def rename_project(self, new_project_name: str) -> str:
        """Rename the project by providing a new name. Sends an update request and returns a confirmation message.

        :param new_project_name: new name for the project
        :return: response
        """
        payload = {
            "project_name": self.project_name,
            "modify_req": {
                "update_project": {
                    "project_name": new_project_name,
                }
            },
        }
        res = self.api_client.post(UPDATE_PROJECT_URI, payload)
        self.user_project_name = new_project_name
        return res.get("details")

    def delete_project(self) -> str:
        """Delete the project. Sends a delete request to the API and returns the response message.

        :return: response
        """
        payload = {
            "project_name": self.project_name,
            "modify_req": {
                "delete_project": self.user_project_name,
            },
        }
        res = self.api_client.post(UPDATE_PROJECT_URI, payload)
        return res.get("details")

    def add_user_to_project(self, email: str, role: str) -> str:
        """Add a user to the project with a specified role such as admin, manager, or user.

        :param email: user email
        :param role: user role ["admin", "manager", "user"]
        :return: response
        """
        payload = {
            "project_name": self.project_name,
            "modify_req": {
                "add_user_project": {
                    "email": email,
                    "role": role,
                },
            },
        }
        res = self.api_client.post(UPDATE_PROJECT_URI, payload)
        return res.get("details")

    def remove_user_from_project(self, email: str) -> str:
        """Remove a user from the project using their email address. Returns a confirmation message.

        :param email: user email
        :return: response
        """
        payload = {
            "project_name": self.project_name,
            "modify_req": {"remove_user_project": email},
        }
        res = self.api_client.post(UPDATE_PROJECT_URI, payload)
        return res.get("details")

    def update_user_access_for_project(self, email: str, role: str) -> str:
        """Update the role of a user within the project. Accepts an email and new role and returns a response.

        :param email: user email
        :param role: user role
        :return: response
        """
        payload = {
            "project_name": self.project_name,
            "modify_req": {
                "update_user_project": {
                    "email": email,
                    "role": role,
                },
            },
        }
        res = self.api_client.post(UPDATE_PROJECT_URI, payload)
        return res.get("details")

    def start_server(self) -> str:
        """Start a dedicated server for the project, enabling training or inference activities.

        :return: response
        """
        res = self.api_client.post(
            f"{START_CUSTOM_SERVER_URI}?project_name={self.project_name}"
        )

        if not res["success"]:
            raise Exception(res.get("message"))

        return res["message"]

    def stop_server(self) -> str:
        """Stop the dedicated project server to release compute resources.

        :return: response
        """
        res = self.api_client.post(
            f"{STOP_CUSTOM_SERVER_URI}?project_name={self.project_name}"
        )

        if not res["success"]:
            raise Exception(res.get("message"))

        return res["message"]

    def update_server(self, server_type: str) -> str:
        """Update the dedicated server for the project by specifying a new instance type.
        :param server_type: dedicated instance to run workloads
            for all available instances check xai.available_custom_servers()

        :return: response
        """
        custom_servers = self.api_client.get(AVAILABLE_CUSTOM_SERVERS_URI)
        Validate.value_against_list(
            "server_type",
            server_type,
            [server["name"] for server in custom_servers],
        )

        payload = {
            "project_name": self.project_name,
            "modify_req": {
                "update_project": {
                    "project_name": self.user_project_name,
                    "instance_type": server_type,
                },
                "update_operational_hours": {},
            },
        }

        res = self.api_client.post(UPDATE_PROJECT_URI, payload)

        if not res["success"]:
            raise Exception(res.get("details"))

        return "Server Updated"

    def available_tags(self) -> str:
        """Return a list of tags that are available for data categorization within the project.

        :return: response
        """
        res = self.api_client.get(
            f"{AVAILABLE_TAGS_URI}?project_name={self.project_name}"
        )

        if not res["success"]:
            error_details = res.get("details", "Failed to get available tags.")
            raise Exception(error_details)
        res["details"]["lexsi_tags"] = res["details"].pop("arya_tags", [])
        return res["details"]

    def files(self) -> pd.DataFrame:
        """List files uploaded to the project. Returns a DataFrame with file names and statuses. Only active files are included.

        :return: user uploaded files dataframe
        """
        files = self.api_client.get(
            f"{ALL_DATA_FILE_URI}?project_name={self.project_name}"
        )

        if not files.get("details"):
            raise Exception("Please upload files first")

        files_df = (
            pd.DataFrame(files["details"])
            .drop(["metadata", "project_name", "version"], axis=1)
            .rename(columns={"filepath": "file_name"})
        )

        files_df = files_df.loc[files_df["status"] == "active"]
        files_df["file_name"] = files_df["file_name"].apply(
            lambda file_path: file_path.split("/")[-1]
        )

        return files_df

    def delete_file(self, file_name: str) -> str:
        """Delete a file with the given name from the project. Accepts the file name.

        :param file_name: uploaded file name
        :return: response
        """
        files = self.api_client.get(
            f"{ALL_DATA_FILE_URI}?project_name={self.project_name}"
        )

        if not files.get("details"):
            raise Exception("Please upload files first")

        file_data = next(
            filter(
                lambda file: file["filepath"] == file_name
                or file["filepath"].split("/")[-1] == file_name,
                files["details"],
            ),
            None,
        )

        if not file_data:
            raise Exception("File Not Found, please pass valid file name")

        payload = {
            "project_name": self.project_name,
            "workspace_name": self.workspace_name,
            "path": file_data["filepath"],
        }

        res = self.api_client.post(DELETE_DATA_FILE_URI, payload)
        return res.get("details")

    def active_model(self) -> pd.DataFrame:
        """Current Active Model for project

        :return: current active model dataframe
        """
        staged_models_df = self.models()
        active_model = staged_models_df[staged_models_df["status"] == "active"]
        return active_model

    def activate_model(self, model_name: str) -> str:
        """Sets the provided model to active for the project

        :param model_name: name of the model
        :return: response
        """
        payload = {
            "project_name": self.project_name,
            "model_name": model_name,
        }
        res = self.api_client.post(UPDATE_ACTIVE_MODEL_URI, payload)

        if not res["success"]:
            raise Exception(res["details"])

        return res.get("details")
    
    def models(self) -> pd.DataFrame:
        """List of models trained for the project

        :return: Dataframe with details of all models
        """
        res = self.api_client.get(f"{GET_MODELS_URI}?project_name={self.project_name}")

        if not res["success"]:
            raise Exception(res["details"])

        staged_models = res["details"]["staged"]

        staged_models_df = pd.DataFrame(staged_models)
        staged_models_df = staged_models_df.drop(columns=['model_provider'])
        staged_models_df = staged_models_df[
            ~staged_models_df["status"].isin(["inactive", "failed"])
        ]

        return staged_models_df

    def remove_model(self, model_name: str) -> str:
        """Removes the trained model from the project

        :param model_name: name of the model
        :return: response
        """
        payload = {
            "project_name": self.project_name,
            "model_name": model_name,
        }
        res = self.api_client.post(REMOVE_MODEL_URI, payload)

        if not res["success"]:
            raise Exception(res["details"])

        return res.get("details")

    def model_summary(self, model_name: Optional[str] = None) -> ModelSummary:
        """Model Summary for the project

        :param model_name: name of the model, defaults to active model for project
        :return: model summary
        """
        if self.metadata.get("modality") == "text":
            res = self.api_client.post(
                f"{PROJECT_OVERVIEW_TEXT_URI}?project_name={self.project_name}"
            )
            return res.get("details")
        else:
            res = self.api_client.get(
                f"{MODEL_SUMMARY_URI}?project_name={self.project_name}"
                + (f"&model_name={model_name}" if model_name else "")
            )

        if not res["success"]:
            raise Exception(res["details"])

        return ModelSummary(api_client=self.api_client, **res.get("details"))

    def tags(self) -> List[str]:
        """Available User Tags for Project

        :return: list of tags
        """
        available_tags = self.available_tags()

        tags = available_tags.get("user_tags")

        return tags

    def all_tags(self) -> List[str]:
        """Available All Tags for Project

        :return: list of tags
        """
        available_tags = self.available_tags()

        tags = available_tags.get("alltags")

        return tags

    def tag_data(self, tag: str, page: Optional[int] = 1) -> pd.DataFrame:
        """Tag Data

        :param tag: Tag name to filter data by.
        :param page: Page number for paginated results.
        :return: tag data dataframe
        """
        tags = self.all_tags()

        Validate.value_against_list("tag", tag, tags)

        payload = {"page": page, "project_name": self.project_name, "tag": tag}
        res = self.api_client.post(TAG_DATA_URI, payload)

        if not res["success"]:
            raise Exception(res.get("details", "Tag data Not Found"))

        tag_data_df = pd.DataFrame(res["details"]["data"])

        return tag_data_df

    def get_tag_data(
        self,
        tag: str,
    ) -> pd.DataFrame:
        """Run model inference on data

        :param tag: data tag for downloading
        :return: dataframe
        """

        tags = self.available_tags()
        available_tags = tags["alltags"]
        if tag not in available_tags:
            raise Exception(
                f"{tag} tag is not valid, select valid tag from :\n{available_tags}"
            )

        auth_token = self.api_client.get_auth_token()

        uri = f"{DOWNLOAD_TAG_DATA_URI}?project_name={self.project_name}&tag={tag}&token={auth_token}"

        tag_data = self.api_client.base_request("GET", uri)

        tag_data_df = pd.read_csv(io.StringIO(tag_data.text))

        return tag_data_df

    def create_data_connectors(
        self,
        data_connector_name: str,
        data_connector_type: str,
        gcs_config: Optional[GCSConfig] = None,
        s3_config: Optional[S3Config] = None,
        gdrive_config: Optional[GDriveConfig] = None,
        sftp_config: Optional[SFTPConfig] = None,
        hf_token: Optional[str] = None,
    ) -> str:
        """Create a data connector for a project, allowing external data (e.g., S3, GCS, Google Drive, SFTP, Dropbox, HuggingFace) to be linked. Requires the connector name and type, plus the corresponding credential dictionary depending on the connector type. 
        For Dropbox, an authentication link will be generated during execution, and user authorization code is required to complete setup.

        :param data_connector_name: name for data connector
        :param data_connector_type: type of data connector (s3 | gcs | gdrive | dropbox | sftp | HuggingFace)
        :param gcs_config: credentials from service account json
        :param s3_config: credentials of s3 storage
        :param gdrive_config: credentials from service account json
        :param sftp_config: hostname, port, username and password for sftp connection
        :return: response
        """
        if not self.organization_id and not self.project_name:
            return "No Project Name or Organization id found"
        if data_connector_type.lower() == "s3":
            if not s3_config:
                return "No configuration for S3 found"

            Validate.value_against_list(
                "s3 config",
                list(s3_config.keys()),
                ["region", "access_key", "secret_key"],
            )

            payload = {
                "link_service": {
                    "service_name": data_connector_name,
                    "region": s3_config.get("region", "ap-south-1"),
                    "access_key": s3_config.get("access_key"),
                    "secret_key": s3_config.get("secret_key"),
                },
                "link_service_type": data_connector_type,
            }

        if data_connector_type.lower() == "gcs":
            if not gcs_config:
                return "No configuration for GCS found"

            Validate.value_against_list(
                "gcs config",
                list(gcs_config.keys()),
                [
                    "project_id",
                    "gcp_project_name",
                    "type",
                    "private_key_id",
                    "private_key",
                    "client_email",
                    "client_id",
                    "auth_uri",
                    "token_uri",
                ],
            )

            payload = {
                "link_service": {
                    "service_name": data_connector_name,
                    "project_id": gcs_config.get("project_id"),
                    "gcp_project_name": gcs_config.get("gcp_project_name"),
                    "service_account_json": {
                        "type": gcs_config.get("type"),
                        "project_id": gcs_config.get("project_id"),
                        "private_key_id": gcs_config.get("private_key_id"),
                        "private_key": gcs_config.get("private_key"),
                        "client_email": gcs_config.get("client_email"),
                        "client_id": gcs_config.get("client_id"),
                        "auth_uri": gcs_config.get("auth_uri"),
                        "token_uri": gcs_config.get("token_uri"),
                    },
                },
                "link_service_type": data_connector_type,
            }

        if data_connector_type == "gdrive":
            if not gdrive_config:
                return "No configuration for Google Drive found"

            Validate.value_against_list(
                "gdrive config",
                list(gdrive_config.keys()),
                [
                    "project_id",
                    "type",
                    "private_key_id",
                    "private_key",
                    "client_email",
                    "client_id",
                    "auth_uri",
                    "token_uri",
                ],
            )

            payload = {
                "link_service": {
                    "service_name": data_connector_name,
                    "service_account_json": {
                        "type": gdrive_config.get("type"),
                        "project_id": gdrive_config.get("project_id"),
                        "private_key_id": gdrive_config.get("private_key_id"),
                        "private_key": gdrive_config.get("private_key"),
                        "client_email": gdrive_config.get("client_email"),
                        "client_id": gdrive_config.get("client_id"),
                        "auth_uri": gdrive_config.get("auth_uri"),
                        "token_uri": gdrive_config.get("token_uri"),
                    },
                },
                "link_service_type": data_connector_type,
            }

        if data_connector_type == "sftp":
            if not sftp_config:
                return "No configuration for Google Drive found"

            Validate.value_against_list(
                "sftp config",
                list(sftp_config.keys()),
                ["hostname", "port", "username", "password"],
            )

            payload = {
                "link_service": {
                    "service_name": data_connector_name,
                    "sftp_json": {
                        "hostname": sftp_config.get("hostname"),
                        "port": sftp_config.get("port"),
                        "username": sftp_config.get("username"),
                        "password": sftp_config.get("password"),
                    },
                },
                "link_service_type": data_connector_type,
            }

        if data_connector_type == "dropbox":
            url_data = self.api_client.get(
                f"{DROPBOX_OAUTH}?project_name={self.project_name}"
            )
            print(f"Url: {url_data['details']['url']}")
            code = input(f"{url_data['details']['message']}: ")

            if not code:
                return "No authentication code provided."

            payload = {
                "link_service": {
                    "service_name": data_connector_name,
                    "dropbox_json": {"code": code},
                },
                "link_service_type": data_connector_type,
            }

        if data_connector_type == "HuggingFace":
            if not hf_token:
                return "No hf_token provided"

            payload = {
                "link_service": {
                    "service_name": data_connector_name,
                    "hf_token": hf_token,
                },
                "link_service_type": data_connector_type,
            }

        url = build_url(
            CREATE_DATA_CONNECTORS,
            data_connector_name,
            self.project_name,
            self.organization_id,
        )
        res = self.api_client.post(url, payload)
        return res["details"]

    def test_data_connectors(self, data_connector_name: str) -> str:
        """Test the connection of an existing data connector to ensure credentials and connectivity are valid. Takes the connector name as input and returns the status of the connection test.

        :param data_connector_name: name of the data connector to be tested.
        """
        if not data_connector_name:
            return "Missing argument data_connector_name"
        if not self.organization_id and not self.project_name:
            return "No Project Name or Organization id found"
        url = build_url(
            TEST_DATA_CONNECTORS,
            data_connector_name,
            self.project_name,
            self.organization_id,
        )
        res = self.api_client.post(url)
        return res["details"]

    def delete_data_connectors(self, data_connector_name: str) -> str:
        """Delete a data connector from the organization using its name. This removes the external data link and returns a confirmation message.

        :param data_connector_name: name of the data connector to be deleted.
        :return: str
        """
        if not data_connector_name:
            return "Missing argument data_connector_name"
        if not self.organization_id and not self.project_name:
            return "No Project Name or Organization id found"

        url = build_url(
            DELETE_DATA_CONNECTORS,
            data_connector_name,
            self.project_name,
            self.organization_id,
        )
        res = self.api_client.post(url)
        return res["details"]

    def list_data_connectors(self) -> str | pd.DataFrame:
        """List all data connectors configured in the project and organization. If successful, returns a DataFrame with details about each connector; otherwise returns an error message."""
        url = build_list_data_connector_url(
            LIST_DATA_CONNECTORS, self.project_name, self.organization_id
        )
        res = self.api_client.post(url)

        if res["success"]:
            df = pd.DataFrame(res["details"])
            df = df.drop(
                [
                    "_id",
                    "region",
                    "gcp_project_name",
                    "gcp_project_id",
                    "gdrive_file_name",
                ],
                axis=1,
                errors="ignore",
            )
            return df

        return res["details"]

    def list_data_connectors_buckets(self, data_connector_name: str) -> str | List:
        """Retrieve the list of buckets (for S3 or GCS connectors) or similar container names for the specified data connector.

        :param data_connector_name: name of the data connector
        :return: str | List
        """
        if not data_connector_name:
            return "Missing argument data_connector_name"
        if not self.organization_id and not self.project_name:
            return "No Project Name or Organization id found"

        url = build_url(
            LIST_BUCKETS, data_connector_name, self.project_name, self.organization_id
        )
        res = self.api_client.get(url)

        if res.get("message", None):
            print(res["message"])
        return res["details"]

    def list_data_connectors_filepath(
        self,
        data_connector_name: str,
        bucket_name: Optional[str] = None,
        root_folder: Optional[str] = None,
    ) -> str | Dict:
        """List file paths within the specified data connector. For S3/GCS connectors you may need to provide a bucket_name; for SFTP connectors you may need to provide a root_folder.

        :param data_connector_name: name of the data connector
        :param bucket_name: Required for S3 & GCS
        :param root_folder: Root folder of SFTP
        """
        if not data_connector_name:
            return "Missing argument data_connector_name"
        if not self.organization_id and not self.project_name:
            return "No Project Name or Organization id found"

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

            if ds_type == "sftp":
                if not root_folder:
                    return "Missing argument root_folder"

        if self.project_name:
            url = f"{LIST_FILEPATHS}?project_name={self.project_name}&link_service_name={data_connector_name}&bucket_name={bucket_name}&root_folder={root_folder}"
        elif self.organization_id:
            url = f"{LIST_FILEPATHS}?organization_id={self.organization_id}&link_service_name={data_connector_name}&bucket_name={bucket_name}&root_folder={root_folder}"
        res = self.api_client.get(url)

        if res.get("message", None):
            print(res["message"])
        return res["details"]

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
        config: Optional[ProjectConfig] = None,
    ) -> str:
        """Uploads data for the current project with data connectors
        :param data_connector_name: name of the data connector
        :param tag: tag for data
        :param bucket_name: if data connector has buckets # Example: s3/gcs buckets
        :param file_path: filepath from the bucket for the data to read
        :param config: project config
                {
                    "project_type": "",
                    "unique_identifier": "",
                    "true_label": "",
                    "pred_label": "",
                    "feature_exclude": [],
                    "drop_duplicate_uid: "",
                    "handle_errors": False,
                    "feature_encodings": Dict[str, str]   # {"feature_name":"labelencode | countencode | onehotencode"}
                },
                defaults to None
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
            if self.metadata.get("modality") == "image":
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

            if self.metadata.get("modality") == "tabular":
                if not config.get("project_type"):
                    config["project_type"] = self.metadata.get("project_type")
                if not config:
                    config = {
                        "project_type": "",
                        "unique_identifier": "",
                        "true_label": "",
                        "pred_label": "",
                        "feature_exclude": [],
                        "drop_duplicate_uid": False,
                        "handle_errors": False,
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

                uploaded_path = upload_file_and_return_path(file_path, "data", tag)

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
                    },
                }

            res = self.api_client.post(UPLOAD_DATA_WITH_CHECK_URI, payload)

            if not res["success"]:
                self.delete_file(uploaded_path)
                raise Exception(res.get("details"))

            poll_events(self.api_client, self.project_name, res["event_id"])

            return res.get("details")

        if project_config != "Not Found" and config:
            raise Exception("Config already exists, please remove config")

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

    def case_logs(self, page: Optional[int] = 1) -> pd.DataFrame:
        """Get already viewed case logs

        :param page: page number, defaults to 1
        :return: Case object with details
        """

        res = self.api_client.get(
            f"{CASE_LOGS_URI}?project_name={self.project_name}&page={page}"
        )

        if not res["success"]:
            raise Exception(res.get("details", "Failed to get case logs"))

        case_logs_df = pd.DataFrame(
            res["details"]["logs"],
            columns=[
                "case_log_id",
                "case_id",
                "unique_identifier",
                "tag",
                "model_name",
                "time_taken",
                "created_at",
            ],
        )
        case_logs_df["case_log_id"] = case_logs_df["case_id"].astype(str)
        case_logs_df.drop(columns=["case_id"], inplace=True)

        return case_logs_df

    def case_record(self, case_id: str):
        """Get already viewed case

        :param case_id: case id
        :return: Case object with details
        """

        res = self.api_client.get(
            f"{GET_VIEWED_CASE_URI}?project_name={self.project_name}&case_id={case_id}"
        )

        if not res["success"]:
            raise Exception(res.get("details", "Failed to get viewed case"))
        data = {**res["details"], **res["details"].get("result", {})}
        data["api_client"] = self.api_client
        if self.metadata.get("modality") == "tabular":
            case = CaseTabular(**data)
        elif self.metadata.get("modality") == "image":
            case = CaseImage(**data)
        elif self.metadata.get("modality") == "text":
            case = CaseText(**data)
        return case

    def get_notifications(self) -> pd.DataFrame:
        """get user project notifications

        :return: DataFrame
        """
        url = f"{GET_NOTIFICATIONS_URI}?project_name={self.project_name}"

        res = self.api_client.get(url)

        if not res["success"]:
            raise Exception("Error while getting project notifications.")

        notifications = [
            notification
            for notification in res["details"]
            if notification.get("project_name", None)
        ]

        if not notifications:
            return "No notifications found."

        return pd.DataFrame(notifications).reindex(columns=["message", "time"])

    def clear_notifications(self) -> str:
        """clear user project notifications

        :raises Exception: _description_
        :return: response
        """
        url = f"{CLEAR_NOTIFICATIONS_URI}?project_name={self.project_name}"

        res = self.api_client.post(url)

        if not res["success"]:
            raise Exception("Error while clearing project notifications.")

        return res["details"]

    def events(
        self,
        event_id: Optional[str] = None,
        event_names: Optional[List[str]] = None,
        status: Optional[List[str]] = None,
    ) -> List[Dict]:
        """Fetches event details for the project.

        :param event_id: Optional event id to filter by.
        :param event_names: Optional list of event names to filter by.
        :param status: Optional list of statuses to filter by.
        :return: event details
        """
        payload = {
            "project_name": self.project_name,
            "event_id": event_id,
            "event_names": event_names,
            "status": status,
        }

        res = self.api_client.post(FETCH_EVENTS, payload)

        if not res["success"]:
            raise Exception(res["details"])

        return res.get("details")

    def __print__(self) -> str:
        """Return a short string identifying this project.
        Used by `__str__` and `__repr__` to display key fields."""
        return f"Project(user_project_name='{self.user_project_name}', created_by='{self.created_by}')"

    def __str__(self) -> str:
        """Return printable representation.
        Summarizes the instance in a concise form."""
        return self.__print__()

    def __repr__(self) -> str:
        """Return developer-friendly representation.
        Includes key fields useful for logging and troubleshooting."""
        return self.__print__()


def generate_expression(expression):
    """Render a provided tokenized expression into a readable string.
    Used to display stored observation/policy expressions from their metadata format.

    :param expression: Token list produced by `build_expression`.
    :return: Rendered expression string, or None if input is empty."""
    if not expression:
        return None
    generated_expression = ""
    for item in expression:
        if isinstance(item, str):
            generated_expression += " " + item
        else:
            generated_expression += (
                f" {item['column']} {item['expression']} {item['value']}"
            )
    return generated_expression


def build_expression(expression_string):
    """Parse a human expression string into configuration and metadata tokens.
    Maps operators to backend enums and preserves parentheses/logical operator ordering.

    :param expression_string: Expression string like `A == 1 and B !== 2`.
    :return: `(configuration, metadata_expression)` token lists."""
    condition_operators = {
        "!==": "_NOTEQ",
        "==": "_ISEQ",
        ">": "_GRT",
        "<": "_LST",
    }
    logical_operators = {"and": "_AND", "or": "_OR"}

    metadata_expression = []
    configuration = []
    string_to_be_parsed = expression_string

    matches = re.findall(r"(\w+)\s*([!=<>]+)\s*(\w+)", expression_string)

    total_opening_parentheses = re.findall(r"\(", expression_string)
    total_closing_parentheses = re.findall(r"\)", expression_string)

    if len(total_opening_parentheses) != len(total_closing_parentheses):
        raise Exception("Invalid expression, check parentheses")

    for i, match in enumerate(matches):
        column, expression, value = match
        if expression not in condition_operators.keys():
            raise Exception(f"Not a valid condition operator in {match}")

        opening_parentheses = re.findall(r"\(", string_to_be_parsed.split(column, 1)[0])
        if opening_parentheses:
            metadata_expression.extend(opening_parentheses)
            configuration.extend(opening_parentheses)

        metadata_expression.append(
            {
                "column": column,
                "value": value,
                "expression": expression,
            }
        )
        configuration.append(
            {
                "column": column,
                "value": value,
                "expression": condition_operators[expression],
            }
        )

        string_to_be_parsed = string_to_be_parsed.split(value, 1)[1]
        between_conditions_split = string_to_be_parsed.split(
            matches[i + 1][0] if i < len(matches) - 1 else None, 1
        )
        closing_parentheses = re.findall(
            r"\)",
            between_conditions_split[0] if len(between_conditions_split) > 0 else "",
        )
        if closing_parentheses:
            metadata_expression.extend(closing_parentheses)
            configuration.extend(closing_parentheses)

        if i < len(matches) - 1:
            between_conditions = between_conditions_split[0].strip()
            between_conditions = between_conditions.replace(")", "").replace("(", "")
            logical_operator = re.search(r"and|or", between_conditions)
            if not logical_operator:
                raise Exception(f"{between_conditions} is not valid logical operator")
            log_operator = logical_operator.group()
            log_operator_split = list(
                filter(
                    lambda op: op != "" and op != " ",
                    between_conditions.split(log_operator, 1),
                )
            )
            if len(log_operator_split) > 0:
                raise Exception(f"{between_conditions} is not valid logical operator")
            metadata_expression.append(log_operator)
            configuration.append(logical_operators[log_operator])

    return configuration, metadata_expression


def validate_configuration(
    configuration, params, project_name="", api_client=APIClient(), observations=False
):
    """Validate an expression provided configuration against allowed features/operators.
    Raises exceptions for invalid columns/operators/values and can validate observation comparisons.

    :param configuration: Configuration token list (from `build_expression`).
    :param params: Allowed features/operators payload fetched from the backend.
    :param project_name: Project name used for backend validation calls.
    :param api_client: API client used for optional backend validation.
    :param observations: If True, validate observation column-vs-column comparisons.
    :raises Exception: If the configuration is invalid."""
    for expression in configuration:
        if isinstance(expression, str):
            if expression not in ["(", ")", *params.get("logical_operators")]:
                raise Exception(f"{expression} not a valid logical operator")

        if isinstance(expression, dict):
            # validate column name
            Validate.value_against_list(
                "feature",
                expression.get("column"),
                list(params.get("features", {}).keys()),
            )

            # validate operator
            Validate.value_against_list(
                "condition_operator",
                expression.get("expression"),
                params.get("condition_operators"),
            )

            # validate value(s)
            expression_value = expression.get("value")
            valid_feature_values = params.get("features").get(expression.get("column"))
            if observations and isinstance(valid_feature_values, list):
                condition_operators = {
                    "_NOTEQ": "!==",
                    "_ISEQ": "==",
                    "_GRT": ">",
                    "_LST": "<",
                }
                res = api_client.get(
                    f"{VALIDATE_POLICY_URI}?project_name={project_name}&column1_name={expression.get('column')}&column2_name={expression.get('value')}&operation={condition_operators[expression.get('expression')]}"
                )
                if not res.get("success"):
                    raise Exception(res.get("message"))
            if isinstance(valid_feature_values, str):
                #     if valid_feature_values == "input" and not parse_float(
                #         expression_value
                #     ):
                #         raise Exception(
                #             f"Invalid value comparison with {expression_value} for {expression.get('column')}"
                #         )
                if valid_feature_values == "datetime" and not parse_datetime(
                    expression_value
                ):
                    raise Exception(
                        f"Invalid value comparison with {expression_value} for {expression.get('column')}"
                    )

                else:
                    condition_operators = {
                        "_NOTEQ": "!==",
                        "_ISEQ": "==",
                        "_GRT": ">",
                        "_LST": "<",
                    }
                    res = api_client.get(
                        f"{VALIDATE_POLICY_URI}?project_name={project_name}&column1_name={expression.get('column')}&column2_name={expression.get('value')}&operation={condition_operators[expression.get('expression')]}"
                    )
                    if not res.get("success"):
                        raise Exception(res.get("message"))
