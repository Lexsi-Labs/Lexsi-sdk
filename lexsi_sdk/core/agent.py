from lexsi_sdk.common.xai_uris import (
    AVAILABLE_GUARDRAILS_URI,
    CONFIGURE_GUARDRAILS_URI,
    DELETE_GUARDRAILS_URI,
    GET_AVAILABLE_TEXT_MODELS_URI,
    GET_GUARDRAILS_URI,
    INITIALIZE_TEXT_MODEL_URI,
    MESSAGES_URI,
    SESSIONS_URI,
    TRACES_URI,
    UPDATE_GUARDRAILS_STATUS_URI,
)
from lexsi_sdk.core.project import Project
import pandas as pd

from lexsi_sdk.core.wrapper import monitor


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
