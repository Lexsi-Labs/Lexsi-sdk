import json
import os
from typing import List, Optional
import httpx
import pandas as pd
from pydantic import BaseModel
import requests
from lexsi_sdk.client.client import APIClient
from lexsi_sdk.common.environment import Environment
from lexsi_sdk.core.organization import Organization
from lexsi_sdk.common.xai_uris import (
    AVAILABLE_BATCH_SERVERS_URI,
    AVAILABLE_CUSTOM_SERVERS_URI,
    AVAILABLE_SYNTHETIC_CUSTOM_SERVERS_URI,
    CLEAR_NOTIFICATIONS_URI,
    CREATE_ORGANIZATION_URI,
    GET_CASE_PROFILE_URI,
    GET_NOTIFICATIONS_URI,
    LOGIN_URI,
    UPLOAD_DATA_PROJECT_URI,
    USER_ORGANIZATION_URI,
)
import getpass


class LEXSI(BaseModel):
    """Base entry-point class for interacting with the Lexsi.ai platform. Handles authentication, organization discovery and selection, notification retrieval, and provides access to higher-level SDK abstractions."""

    env: Environment = Environment()
    api_client: APIClient = APIClient()

    def __init__(self, **kwargs):
        """Initialize the API client using environment-derived settings.
        Stores configuration and prepares the object for use."""
        super().__init__(**kwargs)

        debug = self.env.get_debug()
        base_url = self.env.get_base_url()

        self.api_client = APIClient(debug=debug, base_url=base_url)

    def login(self, sdk_access_token: Optional[str] = None):
        """Authenticate with Lexsi.ai using an access token. It prompts for or reads the access token from the environment variable XAI_ACCESS_TOKEN and sets it on the API client, enabling subsequent calls to the platform.

        :param sdk_access_token: SDK Access Token, defaults to XAI_ACCESS_TOKEN environment variable
        """
        if not sdk_access_token:
            sdk_access_token = os.environ.get("XAI_ACCESS_TOKEN", None) or getpass.getpass(
                "Enter your Lexsi.ai SDK Access Token: "
            )

        if not sdk_access_token:
            raise ValueError("Either set XAI_ACCESS_TOKEN or pass the Access token")

        res = self.api_client.post(LOGIN_URI, payload={"access_token": sdk_access_token})
        self.api_client.update_headers(res["access_token"])
        self.api_client.set_access_token(sdk_access_token)

        print("Authenticated successfully.")

    def organizations(self) -> pd.DataFrame:
        """Retrieve all organizations associated with the authenticated user. Returns a DataFrame listing organization names and metadata such as ownership, admin status, number of users, creator, and creation date.

        :return: Organization details dataframe
        """

        res = self.api_client.get(USER_ORGANIZATION_URI)

        if not res["success"]:
            raise Exception(res.get("details", "Failed to get organizations"))

        res["details"].insert(
            0,
            {
                "name": "personal",
                "organization_owner": True,
                "organization_admin": True,
                "current_users": 1,
                "created_by": res.get("current_user", {}).get("username", ""),
                "created_at": res.get("current_user", {}).get("created_at", ""),
            },
        )

        organization_df = pd.DataFrame(
            res["details"],
            columns=[
                "name",
                "organization_owner",
                "organization_admin",
                "current_users",
                "created_by",
                "created_at",
            ],
        )

        return organization_df

    def organization(self, organization_name: str) -> Organization:
        """Select a specific organization by its name. If the name is "personal", returns the personal organization. Otherwise, it searches the user’s organizations and returns an Organization object for further management.

        :param organization_name: Name of the organization to be used
        :return: Organization object
        """
        if organization_name == "personal":
            return Organization(
                api_client=self.api_client,
                **{
                    "name": "Personal",
                    "organization_owner": True,
                    "organization_admin": True,
                    "current_users": 1,
                    "created_by": "you",
                },
            )

        organizations = self.api_client.get(USER_ORGANIZATION_URI)

        if not organizations["success"]:
            raise Exception(organizations.get("details", "Failed to get organizations"))

        user_organization = [
            Organization(api_client=self.api_client, **organization)
            for organization in organizations["details"]
        ]

        organization = next(
            filter(
                lambda organization: organization.name == organization_name,
                user_organization,
            ),
            None,
        )

        if organization is None:
            raise Exception("Organization Not Found")

        return organization

    def create_organization(self, organization_name: str) -> Organization:
        """Create a new organization with the given name. It sends a POST request to the API and returns an Organization object representing the created organization.

        :param organization_name: Name of the new organization
        :return: Organization object
        """
        payload = {"organization_name": organization_name}
        res = self.api_client.post(CREATE_ORGANIZATION_URI, payload)

        if not res["success"]:
            raise Exception(res.get("details", "Failed to create organization"))

        return Organization(api_client=self.api_client, **res["organization_details"])

    def get_notifications(self) -> pd.DataFrame:
        """Fetch notifications for the user from Lexsi.ai. Notifications include project names, messages and timestamps and are returned as a DataFrame.

        :return: notification details dataFrame
        """
        res = self.api_client.get(GET_NOTIFICATIONS_URI)

        if not res["success"]:
            raise Exception("Error while getting user notifications.")

        notifications = res["details"]

        if not notifications:
            return "No notifications found."

        return pd.DataFrame(notifications).reindex(
            columns=["project_name", "message", "time"]
        )

    def clear_notifications(self) -> str:
        """Clear all notifications for the user by sending a POST request. Returns a confirmation string indicating success.

        :return: response
        """
        res = self.api_client.post(CLEAR_NOTIFICATIONS_URI)

        if not res["success"]:
            raise Exception("Error while clearing user notifications.")

        return res["details"]

    def available_batch_servers(self) -> dict:
        """Retrieve a dictionary of available batch servers (compute instances) that can be used for running custom batch tasks. Useful for selecting compute resources.

        :return: response
        """
        res = self.api_client.get(AVAILABLE_BATCH_SERVERS_URI)
        return res["details"]

    def available_custom_servers(self) -> dict:
        """Retrieve a dictionary or list of available custom servers that can be used for deploying models or running compute-heavy workloads.

        :return: response
        """
        res = self.api_client.get(AVAILABLE_CUSTOM_SERVERS_URI)
        return res

    def available_synthetic_custom_servers(self) -> dict:
        """Retrieve details of custom servers available for generating synthetic data. This helps select the appropriate compute instance for synthetic data generation.

        :return: response
        """
        res = self.api_client.get(AVAILABLE_SYNTHETIC_CUSTOM_SERVERS_URI)
        return res["details"]

    def register_case(
        self,
        token: str,
        client_id: str,
        unique_identifier: Optional[str] = None,
        project_name: str = None,
        tag: Optional[str] = None,
        data: Optional[str] = None,
        processed_data: Optional[bool] = False,
        merge: Optional[bool] = False,
        image_class: Optional[str] = None,
        prompt: Optional[str] = None,
        serverless_instance_type: Optional[str] = None,
        explainability_method: Optional[str] = None,
        explain_model: Optional[bool] = False,
        session_id: Optional[str] = None,
        xai: Optional[str] = None,
        file_path: Optional[str] = None,
    ):
        """Register a new case entry with raw or processed payloads.
        Encapsulates a small unit of SDK logic and returns the computed result."""
        form_data = {
            "client_id": client_id,
            "project_name": project_name,
            "unique_identifier": unique_identifier,
            "tag": tag,
            "data": json.dumps(data) if isinstance(data, list) else data,
            "processed_data": str(processed_data).lower(),
            "merge": str(merge).lower(),
            "image_class": image_class,
            "prompt": prompt,
            "serverless_instance_type": serverless_instance_type,
            "explainability_method": explainability_method,
            "explain_model": str(explain_model).lower(),
            "session_id": str(session_id).lower(),
            "xai": xai,
        }
        headers = {"x-api-token": token}
        form_data = {k: v for k, v in form_data.items() if v is not None}
        files = {}
        if file_path:
            files["in_file"] = open(file_path, "rb")
        # response = requests.post(
        #     self.env.get_base_url() + "/" + UPLOAD_DATA_PROJECT_URI,
        #     data=form_data,
        #     files=files if files else None,
        #     headers=headers
        # ).json()

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

    def case_profile(
        self,
        token: str,
        client_id: str,
        unique_identifier: Optional[str] = None,
        project_name: str = None,
        tag: str = None,
        xai: Optional[List[str]] = None,
        refresh: Optional[bool] = None,
    ):
        """Fetch case profile details for a given identifier and tag.
        Encapsulates a small unit of SDK logic and returns the computed result."""
        headers = {"x-api-token": token}
        payload = {
            "client_id": client_id,
            "project_name": project_name,
            "unique_identifier": unique_identifier,
            "tag": tag,
            "xai": xai,
            "refresh": refresh,
        }
        # res = requests.post(
        #     self.env.get_base_url() + "/" + GET_CASE_PROFILE_URI,
        #     headers=headers,
        #     json=payload
        # ).json()

        with httpx.Client(http2=True, timeout=None) as client:
            res = client.post(
                self.env.get_base_url() + "/" + GET_CASE_PROFILE_URI,
                headers=headers,
                json=payload,
            )
            res.raise_for_status()
            res = res.json()

        return res["details"]
