"""Tracer and agent-guardrail fixes from the agent-tracing review.

session_id validation, AutoGen span filter, OpenAI Agents instrumentor versions, guardrail
names, checks that cannot run, the Runner keeping input/output types, and the rewrite
fallback. The Lexsi API client and the model are faked; nothing calls a service.
"""
import asyncio
import dataclasses
import logging
import uuid
from types import SimpleNamespace

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import StatusCode

from lexsi_sdk.core.guardrails import _common


# ── tracer ───────────────────────────────────────────────────────────────────

tracer_mod = pytest.importorskip("lexsi_sdk.core.tracer")


@pytest.fixture
def setup_provider(monkeypatch):
    monkeypatch.setattr(tracer_mod.trace_api, "set_tracer_provider", lambda provider: None)
    monkeypatch.setattr(tracer_mod, "OTLPSpanExporter", lambda endpoint: InMemorySpanExporter())
    project = SimpleNamespace(project_name="demo")
    return lambda session_id=None, drop=None: tracer_mod.Tracer()._setup_provider(
        "Test", project, session_id, drop=drop
    )


def test_uuid_session_id_is_sent(setup_provider):
    sid = str(uuid.uuid4())
    attrs = setup_provider(sid).resource.attributes
    assert attrs["session_id"] == sid and attrs["project_name"] == "demo"


@pytest.mark.parametrize("missing", [None, "", "None"])
def test_missing_session_id_is_left_out(setup_provider, missing):
    assert "session_id" not in setup_provider(missing).resource.attributes


@pytest.mark.parametrize("bad", ["my-session-1", "1234", "not a uuid"])
def test_non_uuid_session_id_is_refused(setup_provider, bad):
    with pytest.raises(ValueError, match="session_id must be a UUID"):
        setup_provider(bad)


def test_autogen_own_spans_are_not_exported():
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(
        tracer_mod._ScopeFilterSpanProcessor(SimpleSpanProcessor(exporter), tracer_mod._is_autogen_own_span)
    )
    for scope in ("autogen-core", "autogen SingleThreadedAgentRuntime", "autogen-app",
                  "openinference.instrumentation.autogen_agentchat"):
        with provider.get_tracer(scope).start_as_current_span(f"span from {scope}"):
            pass
    assert sorted(s.name for s in exporter.get_finished_spans()) == [
        "span from autogen-app",  # Lexsi's AutoGen guardrail spans
        "span from openinference.instrumentation.autogen_agentchat",
    ]


@pytest.mark.parametrize("instrumentor,agents,warns", [
    ("2.8.7", "0.3.2", True), ("1.3.0", "0.3.2", False), ("2.8.7", "0.26.0", False),
])
def test_untraceable_openai_agents_versions_warn(monkeypatch, caplog, instrumentor, agents, warns):
    versions = {"openinference-instrumentation-openai-agents": instrumentor, "openai-agents": agents}
    monkeypatch.setattr(tracer_mod, "installed_version", versions.get)
    with caplog.at_level(logging.WARNING, logger="lexsi_sdk.tracer"):
        tracer_mod._warn_if_openai_agents_untraceable()
    assert ("no OpenAI Agents spans will be traced" in caplog.text) is warns


# ── shared content helpers ───────────────────────────────────────────────────

CONVERSATION = [
    {"role": "user", "content": "hello"},
    {"role": "assistant", "content": "hi"},
    {"role": "user", "content": "my card is 4111"},
]


def test_latest_user_message_is_checked():
    assert _common.latest_user_text(CONVERSATION) == "my card is 4111"
    assert _common.latest_user_text("plain") == "plain"
    parts = [{"role": "user", "content": [{"type": "input_text", "text": "a"}, {"type": "input_text", "text": "b"}]}]
    assert _common.latest_user_text(parts) == "a b"


def test_rewrite_replaces_only_latest_user_message():
    out = _common.replace_latest_user_text(CONVERSATION, "my card is ****")
    assert out[:2] == CONVERSATION[:2] and out[2] == {"role": "user", "content": "my card is ****"}
    assert CONVERSATION[2]["content"] == "my card is 4111"  # input not mutated
    assert _common.replace_latest_user_text("x", "y") == "y"


def test_typed_output_round_trips():
    pydantic = pytest.importorskip("pydantic")

    class Out(pydantic.BaseModel):
        answer: str

    text = _common.output_text(Out(answer="card 4111"))
    restored = _common.restore_output(Out(answer="card 4111"), text.replace("4111", "****"))
    assert isinstance(restored, Out) and restored.answer == "card ****"
    assert _common.restore_output({"a": 1}, '{"a": 2}') == {"a": 2}
    assert _common.restore_output("old", "new") == "new"


def test_results_without_rewrite_change_nothing():
    results = [SimpleNamespace(output=SimpleNamespace(sanitized_content=None)),
               SimpleNamespace(output=SimpleNamespace())]
    assert _common.apply_sanitized_input(CONVERSATION, results) is CONVERSATION
    assert _common.apply_sanitized_output({"a": 1}, results) == {"a": 1}


@pytest.mark.parametrize("found,ok", [("0.2.8", True), ("0.3.2", True), ("0.3.3", False),
                                      ("0.26.0", False), ("0.2.7", False), (None, False)])
def test_runner_version_check(monkeypatch, found, ok):
    monkeypatch.setattr(_common, "installed_version", lambda name: found)
    if ok:
        _common.require_openai_agents_for_runner()
    else:
        with pytest.raises(ImportError, match="openai-agents>=0.2.8,<0.3.3"):
            _common.require_openai_agents_for_runner()


# ── OpenAI Agents guardrail ──────────────────────────────────────────────────

agents = pytest.importorskip("agents")
guardrails_openai = pytest.importorskip("lexsi_sdk.core.guardrails.guardrails_openai")


class FakeClient:
    """A guardrail group with one flow that fails on text containing 4111."""

    def __init__(self, fetch_ok=True, run_ok=True, flows=True):
        self.fetch_ok, self.run_ok, self.flows = fetch_ok, run_ok, flows

    def get(self, url):
        if not self.fetch_ok:
            return {"success": False}
        return {"success": True, "details": {"guardrail_flows": [{"name": "pii"}] if self.flows else []}}

    def post(self, url, payload=None):
        if not self.run_ok:
            return {"success": False, "details": "guardrail service returned 502"}
        passed = "4111" not in str(payload["input_data"])
        return {"success": True, "data": {"flow_summaries": [{"flow_name": "pii", "passed": passed}]}}


def make_guard(client=None, model=None, on_error="allow"):
    project = SimpleNamespace(api_client=client or FakeClient(), project_name="demo", organization_id=None)
    guard = guardrails_openai.OpenAIAgentsGuardrail(project, model=model, max_retries=1, on_error=on_error)
    guard.retry_delay = 0
    return guard


@pytest.fixture
def spans(monkeypatch):
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(guardrails_openai.trace, "get_tracer", lambda name: provider.get_tracer(name))
    return exporter


def apply(guard, content="hello", action="block"):
    guard.tracer = guardrails_openai.trace.get_tracer("t")
    return asyncio.run(guard._apply_guardrail_group(content, "grp", "input", action, "Agent"))


def test_guardrail_reports_the_given_name():
    guard = make_guard()
    assert guard.create_input_guardrail("grp", name="pii_input").get_name() == "pii_input"
    assert guard.create_output_guardrail("grp", name="pii_output").get_name() == "pii_output"


def test_on_error_is_validated():
    with pytest.raises(ValueError, match="on_error"):
        make_guard(on_error="ignore")


@pytest.mark.parametrize("client,reason", [
    (FakeClient(fetch_ok=False), "Failed to fetch guardrail group"),
    (FakeClient(flows=False), "returned no flows"),
    (FakeClient(run_ok=False), "guardrail service returned 502"),
])
def test_checks_that_cannot_run_are_reported(spans, caplog, client, reason):
    with caplog.at_level(logging.WARNING, logger="lexsi_sdk.guardrails"):
        content, tripped, info = apply(make_guard(client))
    assert (content, tripped) == ("hello", False)  # on_error="allow": continues unguarded
    assert reason in info["execution_error"] and "Continuing without the checks" in caplog.text
    span = next(s for s in spans.get_finished_spans() if s.name == "guardrails:input")
    assert span.status.status_code == StatusCode.ERROR and reason in span.attributes["guardrail.error"]


@pytest.mark.parametrize("client", [FakeClient(fetch_ok=False), FakeClient(run_ok=False)])
def test_on_error_block_trips_the_guardrail(spans, client):
    _, tripped, _ = apply(make_guard(client, on_error="block"))
    assert tripped is True


def test_failed_rewrite_keeps_the_content_not_the_prompt(spans):
    class BrokenModel:
        async def get_response(self, **kwargs):
            raise RuntimeError("model down")

    content, tripped, _ = apply(make_guard(model=BrokenModel()), "card 4111", action="retry")
    assert content == "card 4111" and "Sanitize" not in content
    assert tripped is True  # still failing after the retries


def test_sanitized_content_only_on_a_real_rewrite():
    rewritten = guardrails_openai.OpenAIAgentsGuardrail._rewritten
    assert rewritten("warn", "hello", "hello") is None
    assert rewritten("block", "hello", "hello") is None
    assert rewritten("retry", "hello", "hello") is None
    assert rewritten("retry", "card ****", "card 4111") == "card ****"


# ── Lexsi Runner keeps input and output types ────────────────────────────────

openai_runner = pytest.importorskip("lexsi_sdk.core.guardrails.openai_runner",
                                    reason="needs openai-agents>=0.2.8,<0.3.3")


class FakeModel(agents.Model):
    def __init__(self, text):
        self.text, self.inputs = text, []

    async def get_response(self, *args, **kwargs):
        from openai.types.responses import ResponseOutputMessage, ResponseOutputText

        self.inputs.append(kwargs.get("input"))
        message = ResponseOutputMessage(id="m1", type="message", role="assistant", status="completed",
                                        content=[ResponseOutputText(type="output_text", text=self.text,
                                                                    annotations=[])])
        fields = {f.name for f in dataclasses.fields(agents.ModelResponse)}
        extra = {k: None for k in ("response_id", "referenceable_id", "request_id") if k in fields}
        return agents.ModelResponse(output=[message], usage=agents.Usage(), **extra)

    def stream_response(self, *args, **kwargs):
        raise NotImplementedError


@pytest.fixture(autouse=False)
def no_agents_tracing():
    agents.set_tracing_disabled(True)
    yield
    agents.set_tracing_disabled(False)


def test_runner_keeps_list_input_when_nothing_is_rewritten(no_agents_tracing):
    guard = make_guard()
    model = FakeModel("hi")
    agent = agents.Agent(name="A", model=model, input_guardrails=[guard.create_input_guardrail("grp", action="warn")])
    result = asyncio.run(openai_runner.Runner.run(agent, list(CONVERSATION[:2]) + [{"role": "user", "content": "hello"}]))
    assert isinstance(result.input, list) and len(result.input) == 3
    assert model.inputs[-1][0] == {"role": "user", "content": "hello"}


def test_runner_rewrite_changes_only_latest_user_message(no_agents_tracing):
    class Rewriter:
        async def get_response(self, **kwargs):
            return SimpleNamespace(output=[SimpleNamespace(content=[SimpleNamespace(text="my card is ****")])])

    guard = make_guard(model=Rewriter())
    agent = agents.Agent(name="A", model=FakeModel("ok"),
                         input_guardrails=[guard.create_input_guardrail("grp", action="retry")])
    result = asyncio.run(openai_runner.Runner.run(agent, list(CONVERSATION)))
    assert isinstance(result.input, list) and len(result.input) == 3
    assert result.input[:2] == CONVERSATION[:2]
    assert result.input[2]["content"] == "my card is ****"


def test_runner_keeps_typed_final_output(no_agents_tracing):
    pydantic = pytest.importorskip("pydantic")

    class Out(pydantic.BaseModel):
        answer: str

    guard = make_guard()
    agent = agents.Agent(name="B", model=FakeModel('{"answer": "42"}'), output_type=Out,
                         output_guardrails=[guard.create_output_guardrail("grp", action="warn")])
    result = asyncio.run(openai_runner.Runner.run(agent, "q"))
    assert isinstance(result.final_output, Out) and result.final_output.answer == "42"
