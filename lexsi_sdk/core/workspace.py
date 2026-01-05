import pandas as pd
from pydantic import BaseModel
from typing import Optional
from lexsi_sdk.client.client import APIClient
from lexsi_sdk.common.enums import UserRole
from lexsi_sdk.common.validation import Validate
from lexsi_sdk.common.xai_uris import (
    AVAILABLE_CUSTOM_SERVERS_URI,
    CREATE_PROJECT_URI,
    GET_WORKSPACES_DETAILS_URI,
    START_CUSTOM_SERVER_URI,
    STOP_CUSTOM_SERVER_URI,
    UPDATE_WORKSPACE_URI,
    GET_NOTIFICATIONS_URI,
    CLEAR_NOTIFICATIONS_URI,
)
from lexsi_sdk.core.project import Project
from lexsi_sdk.core.text import TextProject
from lexsi_sdk.core.agent import AgentProject


class Workspace(BaseModel):
    """Logical container inside an organization that groups projects, users, and compute resources. Supports workspace-level user access and project lifecycle management."""

    organization_id: Optional[str] = None
    created_by: str
    user_workspace_name: str
    workspace_name: str
    created_at: str

    api_client: APIClient

    def __init__(self, **kwargs):
        """Attach API client for workspace operations.
        Stores configuration and prepares the object for use."""
        super().__init__(**kwargs)
        self.api_client = kwargs.get("api_client")

    def rename_workspace(self, new_workspace_name: str) -> str:
        """Rename the current workspace to a new name by sending an update request to the API. Updates internal properties and returns the response message.

        :param new_workspace_name: name for the workspace to be renamed to
        :return: response
        """
        payload = {
            "workspace_name": self.workspace_name,
            "modify_req": {
                "update_workspace": {
                    "workspace_name": new_workspace_name,
                }
            },
        }
        res = self.api_client.post(UPDATE_WORKSPACE_URI, payload)
        self.user_workspace_name = new_workspace_name
        return res.get("details")

    def delete_workspace(self) -> str:
        """Delete the current workspace by sending a delete request. Returns a confirmation message upon success.
        :return: response
        """
        payload = {
            "workspace_name": self.workspace_name,
            "modify_req": {"delete_workspace": self.user_workspace_name},
        }
        res = self.api_client.post(UPDATE_WORKSPACE_URI, payload)
        return res.get("details")

    def add_user_to_workspace(self, email: str, role: str) -> str:
        """Add a user to the workspace with a specified role. Valid roles include admin, manager, or user.

        :param email: user email
        :param role: user role ["admin", "manager", "user"]
        :return: response
        """
        payload = {
            "workspace_name": self.workspace_name,
            "modify_req": {
                "add_user_workspace": {
                    "email": email,
                    "role": role,
                },
            },
        }
        res = self.api_client.post(UPDATE_WORKSPACE_URI, payload)
        return res.get("details")

    def remove_user_from_workspace(self, email: str) -> str:
        """Remove a user from the workspace using their email address. Returns a response message.

        :param email: user email
        :return: response
        """
        payload = {
            "workspace_name": self.workspace_name,
            "modify_req": {
                "remove_user_workspace": email,
            },
        }
        res = self.api_client.post(UPDATE_WORKSPACE_URI, payload)
        return res.get("details")

    def update_user_access_for_workspace(self, email: str, role: UserRole) -> str:
        """Update the role of a user in the workspace. Accepts the user’s email and the new role (admin or user).

        :param email: user email
        :param role: new user role ["admin", "user"]
        :return: _description_
        """
        payload = {
            "workspace_name": self.workspace_name,
            "modify_req": {
                "update_user_workspace": {
                    "email": email,
                    "role": role,
                }
            },
        }
        res = self.api_client.post(UPDATE_WORKSPACE_URI, payload)
        return res.get("details")

    def projects(self) -> pd.DataFrame:
        """Retrieve a DataFrame listing all projects in the workspace, with details like project name, access type, creator, and instance type.

        :return: Projects details dataframe
        """
        workspace = self.api_client.get(
            f"{GET_WORKSPACES_DETAILS_URI}?workspace_name={self.workspace_name}"
        )
        projects_df = pd.DataFrame(
            workspace.get("data", {}).get("projects", []),
            columns=[
                "user_project_name",
                "access_type",
                "created_by",
                "created_at",
                "updated_at",
                "instance_type",
                "instance_status",
            ],
        )
        return projects_df

    def project(self, project_name: str) -> Project:
        """Select a specific project by name from the workspace. Returns a Project object (or a subclass like TextProject or AgentProject) for the chosen project.

        :param project_name: Name of the project
        :return: Project
        """
        workspace = self.api_client.get(
            f"{GET_WORKSPACES_DETAILS_URI}?workspace_name={self.workspace_name}"
        )

        project = next(
            filter(
                lambda project: project.get("user_project_name") == project_name,
                workspace.get("data", {}).get("projects", []),
            ),
            None,
        )

        if project is None:
            raise Exception("Project Not Found")

        if project.get("metadata", {}).get("modality") == "text":
            return TextProject(api_client=self.api_client, **project)
        elif project.get("metadata", {}).get("modality") == "agent":
            return AgentProject(api_client=self.api_client, **project)

        return Project(api_client=self.api_client, **project)

    def create_project(
        self,
        project_name: str,
        modality: str,
        project_type: str,
        project_sub_type: Optional[str] = None,
        server_type: Optional[str] = None,
    ) -> Project:
        """Create a new project within the workspace. Requires project_name, modality (e.g., tabular, text, image), project_type (e.g., classification), and optional project_sub_type and server_type. Returns the created Project object.

        :param project_name: name for the project
        :param modality: modality for the project
            Eg:- tabular, image, text
        :project_type: type for the project
            Eg:- classification, regression
        :return: response
        """
        payload = {
            "project_name": project_name,
            "modality": modality,
            "project_type": project_type,
            "project_sub_type": project_sub_type,
            "workspace_name": self.workspace_name,
        }

        if self.organization_id:
            payload["organization_id"] = self.organization_id

        if server_type:
            custom_servers = self.api_client.get(AVAILABLE_CUSTOM_SERVERS_URI)
            Validate.value_against_list(
                "server_type",
                server_type,
                [server["name"] for server in custom_servers],
            )

            payload["instance_type"] = server_type
            payload["server_config"] = {}

        res = self.api_client.post(CREATE_PROJECT_URI, payload)

        if not res["success"]:
            raise Exception(res["details"])

        if modality == "text":
            project = TextProject(api_client=self.api_client, **res["details"])
        elif modality == "agent":
            project = AgentProject(api_client=self.api_client, **res["details"])
        else:
            project = Project(api_client=self.api_client, **res["details"])

        return project

    def get_notifications(self) -> pd.DataFrame:
        """Get notifications specific to the workspace. Returns a DataFrame listing notifications including the project name, message, and timestamp.

        :return: DataFrame
        """
        url = f"{GET_NOTIFICATIONS_URI}?workspace_name={self.workspace_name}"

        res = self.api_client.get(url)

        if not res["success"]:
            raise Exception("Error while getting workspace notifications.")

        notifications = res["details"]

        if not notifications:
            return "No notifications found."

        return pd.DataFrame(notifications).reindex(
            columns=["project_name", "message", "time"]
        )

    def clear_notifications(self) -> str:
        """Clear all notifications for the workspace. Sends a POST request and returns a confirmation message.

        :raises Exception: _description_
        :return: str
        """
        url = f"{CLEAR_NOTIFICATIONS_URI}?workspace_name={self.workspace_name}"

        res = self.api_client.post(url)

        if not res["success"]:
            raise Exception("Error while clearing workspace notifications.")

        return res["details"]

    def start_server(self) -> str:
        """Start a dedicated compute server for the workspace, enabling compute-intensive tasks.

        :return: response
        """

        res = self.api_client.post(
            f"{START_CUSTOM_SERVER_URI}?workspace_name={self.workspace_name}"
        )

        if not res["success"]:
            raise Exception(res.get("message"))

        return res["message"]

    def stop_server(self) -> str:
        """Stop the dedicated compute server associated with the workspace.

        :return: response
        """
        res = self.api_client.post(
            f"{STOP_CUSTOM_SERVER_URI}?workspace_name={self.workspace_name}"
        )

        if not res["success"]:
            raise Exception(res.get("message"))

        return res["message"]

    def update_server(self, server_type: str) -> str:
        """Change the compute instance type for the workspace by specifying a new server_type. Valid values depend on available custom servers.
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
            "workspace_name": self.workspace_name,
            "modify_req": {
                "update_workspace": {
                    "workspace_name": self.user_workspace_name,
                    "instance_type": server_type,
                },
                "update_operational_hours": {},
            },
        }

        res = self.api_client.post(UPDATE_WORKSPACE_URI, payload)

        if not res["success"]:
            raise Exception(res.get("details"))

        return "Server Updated"

    def __print__(self) -> str:
        """User-friendly string representation.
        Encapsulates a small unit of SDK logic and returns the computed result."""
        return f"Workspace(user_workspace_name='{self.user_workspace_name}', created_by='{self.created_by}', created_at='{self.created_at}')"

    def __str__(self) -> str:
        """Return printable representation.
        Summarizes the instance in a concise form."""
        return self.__print__()

    def __repr__(self) -> str:
        """Return developer-friendly representation.
        Includes key fields useful for logging and troubleshooting."""
        return self.__print__()
