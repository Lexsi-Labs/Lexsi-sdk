import json
import os
from typing import List, Literal, Optional
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
    AVAILABLE_SERVERLESS_URI,
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
from lexsi_sdk.core.utils import split_cpu_gpu_servers


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

    def available_pod_servers(self, type: Optional[Literal["GPU", "CPU"]]= None) -> dict:
        """Retrieve a dictionary of available batch servers (compute instances) that can be used for running custom batch tasks. Useful for selecting compute resources.
        :param type: Type of server to filter by GPU/CPU
        :return: response
        """
        if type and type not in ["GPU", "CPU"]:
            raise ValueError("Invalid type. Must be 'GPU' or 'CPU'.")
        res = self.api_client.get(AVAILABLE_BATCH_SERVERS_URI)
        if type=="GPU":
            return res["available_gpu_custom_servers"]
        elif type=="CPU":
            return res["details"]
        else:
            return {"CPU pods": res["details"], "GPU pods": res["available_gpu_custom_servers"]}

    def available_node_servers(self, type: Optional[Literal["GPU", "CPU"]]= None) -> dict:
        """Retrieve a dictionary or list of available custom servers that can be used for deploying models or running compute-heavy workloads.
        :param type: Type of server to filter by GPU/CPU
        :return: response
        """
        if type and type not in ["GPU", "CPU"]:
            raise ValueError("Invalid type. Must be 'GPU' or 'CPU'.")
        res = self.api_client.get(AVAILABLE_CUSTOM_SERVERS_URI)
        cpu_gpu_dict = split_cpu_gpu_servers(res)
        if type=="GPU":
            return cpu_gpu_dict["gpu_servers"]
        elif type=="CPU":
            return cpu_gpu_dict["cpu_servers"]
        else:
            return {"CPU nodes": cpu_gpu_dict["cpu_servers"], "GPU nodes": cpu_gpu_dict["gpu_servers"]}

    def available_serverless_types(self, type: Optional[Literal["GPU", "CPU"]]= None) -> List[str]:
        """Retrieve a list of available serverless types that can be used for deploying models or running workloads.
        :return: response
        """
        if type and type not in ["GPU", "CPU"]:
            raise ValueError("Invalid type. Must be 'GPU' or 'CPU'.")
        res = self.api_client.get(AVAILABLE_SERVERLESS_URI)
        cpu_gpu_dict = split_cpu_gpu_servers(res)
        if type=="GPU":
            return cpu_gpu_dict["gpu_servers"]
        elif type=="CPU":
            return cpu_gpu_dict["cpu_servers"]
        else:
            return {"CPU serverless": cpu_gpu_dict["cpu_servers"], "GPU serverless": cpu_gpu_dict["gpu_servers"]}

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
