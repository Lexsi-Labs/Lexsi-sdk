from opentelemetry import trace as trace_api
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.sdk.trace import SpanProcessor
from opentelemetry.sdk.trace.export import (
    ConsoleSpanExporter,
    SimpleSpanProcessor,
    BatchSpanProcessor,
)
from opentelemetry.sdk.resources import Resource
from openinference.instrumentation.langchain import LangChainInstrumentor

# from openinference.instrumentation.autogen_agent import AutogenInstrumentor
# from opentelemetry.instrumentation.openai import OpenAIInstrumentor
from openinference.instrumentation.autogen_agentchat import AutogenAgentChatInstrumentor
from openinference.instrumentation.crewai import CrewAIInstrumentor
from openinference.instrumentation.openai_agents import OpenAIAgentsInstrumentor
from openinference.instrumentation.dspy import DSPyInstrumentor
from openinference.instrumentation.llama_index import LlamaIndexInstrumentor
from openinference.instrumentation.smolagents import SmolagentsInstrumentor

from lexsi_sdk.core.guardrails._common import installed_version, version_tuple

# from pydantic import BaseModel
import logging
import os
import uuid
from typing import Callable, Optional

logger = logging.getLogger("lexsi_sdk.tracer")

# autogen-core >= 0.6.2 creates its own spans (create_agent, invoke_agent, execute_tool, and
# runtime message spans) through the global tracer provider. The OpenInference AutoGen
# instrumentor already reports the agent steps, so these duplicates are not exported.
_AUTOGEN_OWN_SCOPE = "autogen-core"
_AUTOGEN_RUNTIME_SCOPE_PREFIX = "autogen "  # TraceHelper: f"autogen {runtime name}"


def _session_id_attribute(session_id: Optional[str]) -> Optional[str]:
    """The session id to send, or None for the project's default session.

    The platform only accepts UUID session ids and rejects the spans of any other id, so an
    invalid id is refused here, before any span is exported.
    """
    if session_id is None or session_id == "" or session_id == "None":
        return None
    try:
        return str(uuid.UUID(str(session_id)))
    except (ValueError, TypeError, AttributeError):
        raise ValueError(
            f"session_id must be a UUID string such as str(uuid.uuid4()), got {session_id!r}. "
            "Leave it out to use the project's default session."
        ) from None


def _is_autogen_own_span(span) -> bool:
    scope = getattr(span, "instrumentation_scope", None)
    name = getattr(scope, "name", None) or ""
    return name == _AUTOGEN_OWN_SCOPE or name.startswith(_AUTOGEN_RUNTIME_SCOPE_PREFIX)


class _ScopeFilterSpanProcessor(SpanProcessor):
    """Passes spans to ``processor`` except those ``drop`` selects."""

    def __init__(self, processor: SpanProcessor, drop: Callable[[object], bool]) -> None:
        self._processor = processor
        self._drop = drop

    def on_start(self, span, parent_context=None) -> None:
        if not self._drop(span):
            self._processor.on_start(span, parent_context=parent_context)

    def on_end(self, span) -> None:
        if not self._drop(span):
            self._processor.on_end(span)

    def shutdown(self) -> None:
        self._processor.shutdown()

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return self._processor.force_flush(timeout_millis)


def _warn_if_openai_agents_untraceable() -> None:
    """OpenInference's OpenAI Agents instrumentor 2.x only instruments openai-agents >= 0.11.0
    and otherwise traces nothing; say so instead of exporting no spans silently."""
    instrumentor = installed_version("openinference-instrumentation-openai-agents")
    agents = installed_version("openai-agents")
    if instrumentor and agents and version_tuple(instrumentor) >= (2, 0, 0) and version_tuple(agents) < (0, 11, 0):
        logger.warning(
            "openinference-instrumentation-openai-agents %s needs openai-agents >= 0.11.0, but %s "
            "is installed, so no OpenAI Agents spans will be traced. Install "
            "'openinference-instrumentation-openai-agents<2' for this openai-agents version.",
            instrumentor,
            agents,
        )


class Tracer:
    """Helpers to instrument various agent frameworks with OpenTelemetry."""

    def __init__(self):
        """Initialize exporter endpoint from environment.
        Stores configuration and prepares the object for use."""
        self.base_url = os.getenv("XAI_API_URL", "https://api.lexsi.ai")
        self.endpoint = f"{self.base_url}"

    def _setup_provider(
        self,
        service_name: str,
        project: object,
        session_id: Optional[str] = None,
        drop: Optional[Callable[[object], bool]] = None,
    ) -> trace_sdk.TracerProvider:
        """Create the tracer provider every framework setup uses: a resource with the service,
        project and session, set as the global provider, exporting each span over OTLP.

        :param drop: optional filter; spans it selects are not exported.
        """
        session = _session_id_attribute(session_id)
        attributes = {
            "service.name": service_name,
            "project_name": getattr(project, "project_name"),
        }
        if session:
            attributes["session_id"] = session

        tracer_provider = trace_sdk.TracerProvider(resource=Resource(attributes=attributes))
        trace_api.set_tracer_provider(tracer_provider)

        processor = SimpleSpanProcessor(OTLPSpanExporter(self.endpoint))
        tracer_provider.add_span_processor(
            _ScopeFilterSpanProcessor(processor, drop) if drop else processor
        )
        return tracer_provider

    def setup_langchain_tracing(self, project: object, session_id: str = None) -> None:
        """
        Sets up OpenTelemetry tracing for a given project with OTLP and console exporters.

        :param project: Object containing project details; must expose project_name.
        :param session_id: Optional session id (a UUID string) to group the spans under.
        """
        self._setup_provider("Langgraph", project, session_id)
        LangChainInstrumentor().instrument()

    def setup_autogen_tracing(self, project: object, session_id: str = None) -> None:
        """
        Sets up OpenTelemetry tracing for a given project with OTLP and console exporters.

        AutoGen's own spans (create_agent, invoke_agent, execute_tool and runtime messages)
        are not exported; the OpenInference AutoGen instrumentor reports the agent steps.

        :param project: Object containing project details; must expose project_name.
        :param session_id: Optional session id (a UUID string) to group the spans under.
        """
        self._setup_provider("Autogen", project, session_id, drop=_is_autogen_own_span)
        AutogenAgentChatInstrumentor().instrument()

    def setup_crewai_tracing(self, project: object, session_id: str = None) -> None:
        """
        Sets up OpenTelemetry tracing for a given project with OTLP and console exporters.

        :param project: Object containing project details; must expose project_name.
        :param session_id: Optional session id (a UUID string) to group the spans under.
        """
        self._setup_provider("Crewai", project, session_id)
        CrewAIInstrumentor().instrument()

    def setup_agents_tracing(self, project: object, session_id: str = None) -> None:
        """
        Sets up OpenTelemetry tracing for a given project with OTLP and console exporters.

        The OpenInference processor replaces the Agents SDK's default trace processor
        (exclusive_processor=True), so Agents SDK traces go to Lexsi only.

        :param project: Object containing project details; must expose project_name.
        :param session_id: Optional session id (a UUID string) to group the spans under.
        """
        self._setup_provider("OpenAI-Agents", project, session_id)
        _warn_if_openai_agents_untraceable()
        OpenAIAgentsInstrumentor().instrument(exclusive_processor=True)

    def setup_dspy_tracing(self, project: object, session_id: str = None) -> None:
        """
        Sets up OpenTelemetry tracing for a given project with OTLP and console exporters.

        :param project: Object containing project details; must expose project_name.
        :param session_id: Optional session id (a UUID string) to group the spans under.
        """
        self._setup_provider("DSPy", project, session_id)
        DSPyInstrumentor().instrument()

    def setup_llamaindex_tracing(self, project: object, session_id: str = None) -> None:
        """Enable tracing for LlamaIndex runs.

        :param session_id: Optional session id (a UUID string) to group the spans under.
        """
        self._setup_provider("Llamaindex", project, session_id)
        LlamaIndexInstrumentor().instrument()

    def setup_smolagents_tracing(self, project: object, session_id: str = None) -> None:
        """Enable tracing for Smolagents runs.

        :param session_id: Optional session id (a UUID string) to group the spans under.
        """
        self._setup_provider("Smolagents", project, session_id)
        SmolagentsInstrumentor().instrument()
