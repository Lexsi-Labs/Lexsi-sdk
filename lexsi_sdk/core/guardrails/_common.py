"""Behaviour shared by the guardrail integrations (OpenAI Agents, AutoGen, LangGraph)."""

import json
import logging
import re
from typing import Any, Optional

from opentelemetry.trace import Span, Status, StatusCode

logger = logging.getLogger("lexsi_sdk.guardrails")

#: openai-agents releases the Lexsi Runner works with. It reuses internal modules of the
#: Agents SDK's run loop, which later releases moved or changed (0.3.3 changed RunResult,
#: 0.8.0 removed agents.run.get_model_tracing_impl and RunImpl).
OPENAI_AGENTS_RUNNER_SUPPORTED = ">=0.2.8,<0.3.3"
_RUNNER_MIN, _RUNNER_BELOW = (0, 2, 8), (0, 3, 3)


def installed_version(distribution: str) -> Optional[str]:
    try:
        from importlib.metadata import PackageNotFoundError, version
    except ImportError:  # pragma: no cover - Python < 3.8
        return None
    try:
        return version(distribution)
    except PackageNotFoundError:
        return None


def version_tuple(text: str) -> tuple:
    """(major, minor, patch) from a version string; enough to compare release ranges."""
    numbers = [int(n) for n in re.findall(r"\d+", text.split("+")[0])[:3]]
    return tuple(numbers + [0] * (3 - len(numbers)))


def require_openai_agents_for_runner() -> None:
    """Raise a clear ImportError when the installed openai-agents is outside the range the
    Lexsi Runner supports, instead of failing later on a missing internal name."""
    found = installed_version("openai-agents")
    advice = (
        f"The Lexsi Runner supports openai-agents{OPENAI_AGENTS_RUNNER_SUPPORTED}: "
        f"pip install 'openai-agents{OPENAI_AGENTS_RUNNER_SUPPORTED}'. With another version, "
        "use agents.Runner with OpenAIAgentsGuardrail instead (blocking works; 'retry' rewrites "
        "are applied only by the Lexsi Runner)."
    )
    if found is None:
        raise ImportError(f"openai-agents is not installed. {advice}")
    if not (_RUNNER_MIN <= version_tuple(found) < _RUNNER_BELOW):
        raise ImportError(f"openai-agents {found} is installed. {advice}")

#: What to do when the checks cannot run (the guardrail group cannot be fetched, has no
#: flows, or the guardrail service fails): "allow" lets the content through, "block" stops it.
ON_ERROR_ACTIONS = ("allow", "block")


def validate_on_error(on_error: str) -> str:
    """Return ``on_error`` if it is a supported value, else raise ``ValueError``."""
    if on_error not in ON_ERROR_ACTIONS:
        raise ValueError(f"on_error must be one of {ON_ERROR_ACTIONS}, got {on_error!r}")
    return on_error


def report_check_failure(span: Optional[Span], reason: str, on_error: str, where: str) -> bool:
    """Record that the guardrail checks could not run.

    The failure is logged as a warning and marked on the guardrail span (status ERROR and a
    ``guardrail.error`` attribute), so an unguarded step is visible to the agent's developer
    and in the trace instead of looking like a passed check.

    :return: True when ``on_error`` is "block", i.e. the caller must stop the content.
    """
    blocked = on_error == "block"
    logger.warning(
        "Guardrail checks could not run for %s: %s. %s",
        where,
        reason,
        "Blocking the content (on_error='block')." if blocked
        else "Continuing without the checks (on_error='allow').",
    )
    if span is not None:
        span.set_attribute("guardrail.error", str(reason))
        span.set_attribute("guardrail.on_error", on_error)
        span.set_status(Status(StatusCode.ERROR, str(reason)))
    return blocked


def _role(item: Any) -> Optional[str]:
    return item.get("role") if isinstance(item, dict) else getattr(item, "role", None)


def _text_of(item: Any) -> str:
    content = item.get("content") if isinstance(item, dict) else getattr(item, "content", item)
    if isinstance(content, str):
        return content
    if isinstance(content, list):  # content parts: keep the text ones
        parts = []
        for part in content:
            text = part.get("text") if isinstance(part, dict) else getattr(part, "text", None)
            if isinstance(text, str):
                parts.append(text)
        return " ".join(parts)
    return str(content)


def latest_user_text(agent_input: Any) -> str:
    """The text a guardrail checks for an OpenAI Agents input.

    For a list of input items (a conversation or a session), that is the latest user message;
    earlier turns were checked when they were sent. Without a user message, the items' text
    is joined as before.
    """
    if not isinstance(agent_input, list):
        return str(agent_input)
    for item in reversed(agent_input):
        if _role(item) == "user":
            return _text_of(item)
    return " ".join(_text_of(item) for item in agent_input).strip()


def replace_latest_user_text(agent_input: Any, text: str) -> Any:
    """``agent_input`` with the checked text put back, keeping its type.

    A string input becomes ``text``; for a list, only the latest user message's content is
    replaced, so the conversation keeps its items. The input itself is not modified.
    """
    if not isinstance(agent_input, list):
        return text
    items = list(agent_input)
    for index in range(len(items) - 1, -1, -1):
        item = items[index]
        if _role(item) != "user":
            continue
        if isinstance(item, dict):
            items[index] = {**item, "content": text}
        else:
            try:
                updated = item.model_copy(update={"content": text})  # pydantic v2 items
            except AttributeError:
                updated = {"role": "user", "content": text}
            items[index] = updated
        return items
    logger.warning("Input has no user message to put the checked text back into; keeping the input.")
    return agent_input


def output_text(output: Any) -> str:
    """The text a guardrail checks for an agent's final output."""
    if isinstance(output, str):
        return output
    if hasattr(output, "model_dump_json"):  # typed (pydantic) output_type
        return output.model_dump_json()
    if isinstance(output, (dict, list)):
        return json.dumps(output, ensure_ascii=False, default=str)
    if hasattr(output, "response"):
        return str(output.response)
    if hasattr(output, "content"):
        return str(output.content)
    return str(output)


def restore_output(original: Any, text: str) -> Any:
    """The final output with the checked text put back, keeping the output's type when the
    text can be parsed back into it (typed outputs are checked as JSON)."""
    if isinstance(original, str):
        return text
    try:
        if hasattr(original, "model_validate_json"):
            return type(original).model_validate_json(text)
        if isinstance(original, (dict, list)):
            parsed = json.loads(text)
            if isinstance(parsed, type(original)):
                return parsed
    except Exception:
        pass
    logger.warning(
        "Checked output could not be converted back to %s; returning it as text.",
        type(original).__name__,
    )
    return text


def apply_sanitized_input(agent_input: Any, guardrail_results: Any) -> Any:
    """Apply the rewritten text from input guardrail results (only set on a 'retry' rewrite)."""
    for result in guardrail_results or []:
        text = getattr(getattr(result, "output", None), "sanitized_content", None)
        if isinstance(text, str) and text:
            agent_input = replace_latest_user_text(agent_input, text)
    return agent_input


def apply_sanitized_output(output: Any, guardrail_results: Any) -> Any:
    """Apply the rewritten text from output guardrail results (only set on a 'retry' rewrite)."""
    for result in guardrail_results or []:
        text = getattr(getattr(result, "output", None), "sanitized_content", None)
        if isinstance(text, str) and text:
            output = restore_output(output, text)
    return output
