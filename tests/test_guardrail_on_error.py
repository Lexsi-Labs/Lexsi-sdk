"""AutoGen and LangGraph guardrails when the checks cannot run (on_error)."""
import logging
from types import SimpleNamespace

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import StatusCode


class FakeClient:
    def __init__(self, fetch_ok=True, run_ok=True):
        self.fetch_ok, self.run_ok = fetch_ok, run_ok

    def get(self, url):
        return {"success": True, "details": {"guardrail_flows": [{"name": "pii"}]}} if self.fetch_ok else {"success": False}

    def post(self, url, payload=None):
        if not self.run_ok:
            return {"success": False, "details": "guardrail service returned 502"}
        return {"success": True, "data": {"flow_summaries": [{"flow_name": "pii", "passed": True}]}}


def project(client):
    return SimpleNamespace(api_client=client, project_name="demo", organization_id=None)


def tracer():
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    return provider.get_tracer("test"), exporter


# ── LangGraph ────────────────────────────────────────────────────────────────

@pytest.fixture
def langgraph(monkeypatch):
    module = pytest.importorskip("lexsi_sdk.core.guardrails.guardrails_langgraph")
    t, exporter = tracer()
    span = t.start_span("node")
    # Without LangChain tracing, get_current_span() is None on main and no checks run at all
    # (open PR #91); give the guardrail a current span so the checks are attempted.
    monkeypatch.setattr(module, "get_current_span", lambda: span)

    def make(client, on_error="allow"):
        guard = module.LangGraphGuardrail(project(client), on_error=on_error)
        guard.tracer = t
        return guard

    return make, exporter


@pytest.mark.parametrize("client", [FakeClient(fetch_ok=False), FakeClient(run_ok=False)])
def test_langgraph_allow_continues_and_reports(langgraph, caplog, client):
    make, exporter = langgraph
    with caplog.at_level(logging.WARNING, logger="lexsi_sdk.guardrails"):
        out = make(client)._process_content("hello", "node", "input", "block", "grp")
    assert out == "hello" and "Continuing without the checks" in caplog.text
    span = next(s for s in exporter.get_finished_spans() if s.name == "guardrails:input")
    assert span.status.status_code == StatusCode.ERROR


@pytest.mark.parametrize("client", [FakeClient(fetch_ok=False), FakeClient(run_ok=False)])
def test_langgraph_block_stops_the_graph(langgraph, client):
    make, _ = langgraph
    with pytest.raises(ValueError, match="on_error='block'"):
        make(client, on_error="block")._process_content("hello", "node", "input", "warn", "grp")


def test_langgraph_checks_that_run_are_unchanged(langgraph):
    make, _ = langgraph
    assert make(FakeClient())._process_content("hello", "node", "input", "block", "grp") == "hello"


# ── AutoGen ──────────────────────────────────────────────────────────────────

@pytest.fixture
def autogen():
    module = pytest.importorskip("lexsi_sdk.core.guardrails.guardrail_autogen")
    t, exporter = tracer()

    def make(client, on_error="allow"):
        supervisor = module.GuardrailSupervisor("grp", project=project(client), on_error=on_error)
        supervisor.tracer = t
        return supervisor

    return make, exporter


@pytest.mark.parametrize("client", [FakeClient(fetch_ok=False), FakeClient(run_ok=False)])
def test_autogen_allow_continues_and_reports(autogen, caplog, client):
    make, exporter = autogen
    with caplog.at_level(logging.WARNING, logger="lexsi_sdk.guardrails"):
        failed = make(client)._run_guardrails("hello", "agent", "input", None)
    assert failed == [] and "Continuing without the checks" in caplog.text
    assert exporter.get_finished_spans()[-1].status.status_code == StatusCode.ERROR


@pytest.mark.parametrize("client", [FakeClient(fetch_ok=False), FakeClient(run_ok=False)])
def test_autogen_block_stops_the_agent(autogen, client):
    make, _ = autogen
    with pytest.raises(ValueError, match="on_error='block'"):
        make(client, on_error="block")._run_guardrails("hello", "agent", "input", None)
