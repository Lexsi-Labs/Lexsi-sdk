import json
from typing import Optional
from lexsi_sdk.common.types import InferenceCompute
from lexsi_sdk.common.xai_uris import (
    AGENT_DEPLOYMENTS_URI,
    CREATE_AGENT_DEPLOYMENT_URI,
    DELETE_AGENT_DEPLOYMENT_URI,
    MESSAGES_URI,
    SESSIONS_URI,
    START_AGENT_DEPLOYMENTS_URI,
    STOP_AGENT_DEPLOYMENTS_URI,
    TRACES_URI,
    UPDATE_AGENT_DEPLOYMENT_URI,
)
from lexsi_sdk.core.project import Project
import pandas as pd


class AgentProject(Project):
    """Project abstraction for agent-based workflows. Enables tracing, guardrail enforcement, tool invocation tracking, and agent execution analysis."""

    def sessions(self) -> pd.DataFrame:
        """Return a DataFrame listing all conversation sessions for this agent project.

        :return: response
        """
        res = self.api_client.get(f"{SESSIONS_URI}?project_name={self.project_name}")
        if not res["success"]:
            raise Exception(res.get("details"))

        return pd.DataFrame(res.get("details"))

    def messages(self, session_id: str) -> pd.DataFrame:
        """Return a DataFrame listing all messages for a given session. Requires the session_id.

        :param session_id: id of the session
        :return: response
        """
        res = self.api_client.get(
            f"{MESSAGES_URI}?project_name={self.project_name}&session_id={session_id}"
        )
        if not res["success"]:
            raise Exception(res.get("details"))

        return pd.DataFrame(res.get("details"))

    def traces(self, trace_id: str) -> pd.DataFrame:
        """Retrieve execution traces for a given trace ID for agent conversations. Returns a DataFrame of trace details.

        :param trace_id: id of the trace
        :return: response
        """
        res = self.api_client.get(
            f"{TRACES_URI}?project_name={self.project_name}&trace_id={trace_id}"
        )
        if not res["success"]:
            raise Exception(res.get("details"))

        return pd.DataFrame(res.get("details"))
    
    def agent_deployments(self) -> pd.DataFrame:
        """List all agent deployments for this project. Returns a DataFrame of deployment details.

        :return: response
        """
        res = self.api_client.get(f"{AGENT_DEPLOYMENTS_URI}?project_name={self.project_name}")
        if not res["success"]:
            raise Exception(res.get("details"))

        return pd.DataFrame(res.get("details"))

    
    def create_agent_deployment(
        self, 
        agent_name: str, 
        compute: InferenceCompute,
        docker_compose_file: str,
    ) -> dict:
        """Create a new agent deployment with the given name and configuration.

        :param agent_name: name of the agent deployment
        :param compute: compute configuration for the agent deployment
        :param docker_compose_file: path to the docker compose file defining the agent deployment
        :return: response
        """
        data = {
            "project_name": self.project_name,
            "agent_name": agent_name,
            "compute": compute,
        }
        payload = {
            "data": (None, json.dumps(data)),
            "docker_compose_file": ('docker-compose.yaml', open(docker_compose_file, 'rb')),
        }
        res = self.api_client.file(CREATE_AGENT_DEPLOYMENT_URI, files=payload)
        if not res["success"]:
            raise Exception(res.get("details"))

        return res.get("details")

    def update_agent_deployment(
        self, 
        agent_id: str,
        agent_name: Optional[str] = None, 
        compute: Optional[InferenceCompute] = None,
        docker_compose_file: Optional[str] = None,
    ) -> dict:
        """Update an existing agent deployment with the given name and configuration.

        :param agent_name: name of the agent deployment
        :param compute: compute configuration for the agent deployment
        :param docker_compose_file: path to the docker compose file defining the agent deployment
        :return: response
        """
        data = {
            "agent_id": agent_id,
            "project_name": self.project_name,
            "agent_name": agent_name,
            "compute": compute,
        }
        payload = {
            "data": (None, json.dumps(data)),
        }
        if docker_compose_file:
            payload["docker_compose_file"] = ('docker-compose.yaml', open(docker_compose_file, 'rb'))
            
        res = self.api_client.file(UPDATE_AGENT_DEPLOYMENT_URI, files=payload)
        if not res["success"]:
            raise Exception(res.get("details"))

        return res.get("details")

    def start_agent_deployment(self, agent_id: str) -> dict:
        """Start an existing agent deployment by ID.

        :param agent_id: id of the agent deployment to start
        :return: response
        """

        payload = {
            "project_name": self.project_name,
            "agent_id": agent_id,
        }
        res = self.api_client.post(f"{START_AGENT_DEPLOYMENTS_URI}", payload=payload)
        if not res["success"]:
            raise Exception(res.get("details"))

        return res.get("details")
    
    def stop_agent_deployment(self, agent_id: str) -> dict:
        """Stop a running agent deployment by ID.

        :param agent_id: id of the agent deployment to stop
        :return: response
        """

        payload = {
            "project_name": self.project_name,
            "agent_id": agent_id,
        }
        res = self.api_client.post(f"{STOP_AGENT_DEPLOYMENTS_URI}", payload=payload)
        if not res["success"]:
            raise Exception(res.get("details"))

        return res.get("details")

    
    def delete_agent_deployment(self, agent_id: str) -> dict:
        """Delete an existing agent deployment by ID.

        :param agent_id: id of the agent deployment to delete
        :return: response
        """

        payload = {
            "project_name": self.project_name,
            "agent_id": agent_id,
        }
        res = self.api_client.post(f"{DELETE_AGENT_DEPLOYMENT_URI}", payload=payload)
        if not res["success"]:
            raise Exception(res.get("details"))

        return res.get("details")
