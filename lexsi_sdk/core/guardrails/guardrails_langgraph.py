from __future__ import annotations

import json
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional, TypedDict, Union
from lexsi_sdk.common.xai_uris import GUARDRAILS_GET, GUARDRAILS_RUN
from lexsi_sdk.core.project import Project
from openinference.instrumentation.langchain import get_current_span
from opentelemetry import trace
from langchain_core.messages import AIMessage
from .guard_template import Guard


class LangGraphGuardrail:
    """Guardrail integration for LangGraph workflows. Enables node-level input and output validation."""

    def __init__(
        self,
        project: Optional[Project],
        default_apply_on: str = "input",
        llm: Optional[Any] = None,
        max_retries: int = 1,
    ) -> None:
        if project is not None:
            self.client = project.api_client
            self.project_name = project.project_name
            self.organization_id = getattr(project, "organization_id", None)

        self.default_apply_on = default_apply_on
        self.logs: List[Dict[str, Any]] = []
        self.max_retries = max_retries
        self.tracer = trace.get_tracer(__name__)
        self.llm = llm

    def guardrail(
        self,
        guardrail_group_id: str,
        action: str = "block",
        apply_to: str = "both",
        input_key: Optional[str] = None,
        output_key: Optional[str] = None,
    ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        """Decorator that wraps a LangGraph node and applies the guardrail group identified by
        ``guardrail_group_id``.

        - action: 'block' | 'retry' | 'warn'. If any flow fails:
          - block: raise ValueError
          - retry: LLM-sanitize content, re-run up to max_retries; block if still failing
          - warn: keep content, log only
        - apply_to: 'input' | 'output' | 'both'
        - input_key: optional key in state for input content (defaults to 'messages' or 'input')
        - output_key: optional key in result for output content (defaults to 'messages' or str)
        """

        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                state = args[0] if args else kwargs.get("state")
                node_name = getattr(func, "__name__", "unknown_node")

                if state and apply_to in ("input", "both"):
                    input_content = self._get_content(state, input_key, is_input=True)
                    if input_content is not None:
                        try:
                            processed = self._process_content(
                                content=input_content,
                                node_name=node_name,
                                content_type="input",
                                action=action,
                                guardrail_group_id=guardrail_group_id,
                            )
                            self._set_content(state, processed, input_key, is_input=True)
                        except ValueError:
                            raise  # block on input — stop entire graph

                result = func(*args, **kwargs)

                if apply_to in ("output", "both"):
                    output_content = self._get_content(result, output_key, is_input=False)
                    if output_content is not None:
                        try:
                            processed = self._process_content(
                                content=output_content,
                                node_name=node_name,
                                content_type="output",
                                action=action,
                                guardrail_group_id=guardrail_group_id,
                            )
                            if output_key is not None:
                                if isinstance(result, dict):
                                    result[output_key] = processed
                            else:
                                if isinstance(result, dict):
                                    if "messages" in result:
                                        result["messages"] = processed
                                elif isinstance(result, str):
                                    result = processed
                        except ValueError:
                            raise  # block on output — stop entire graph

                return result

            return wrapper

        return decorator

    def _get_content(self, data: Any, key: Optional[str], is_input: bool) -> Any:
        if key is not None:
            if isinstance(data, dict) and key in data:
                return data[key]
            return None
        else:
            if isinstance(data, dict):
                if "messages" in data:
                    return data["messages"]
                elif is_input and "input" in data:
                    return data["input"]
            if not is_input and isinstance(data, str):
                return data
            return None

    def _set_content(
        self, data: Any, processed: Any, key: Optional[str], is_input: bool
    ) -> None:
        if key is not None:
            if isinstance(data, dict):
                data[key] = processed
        else:
            if isinstance(data, dict):
                if "messages" in data:
                    data["messages"] = processed
                elif is_input and "input" in data:
                    data["input"] = processed

    def _process_content(
        self,
        content: Any,
        node_name: str,
        content_type: str,
        action: str,
        guardrail_group_id: str,
    ) -> Any:
        if not guardrail_group_id:
            return content

        is_list = isinstance(content, list)
        if is_list and content:
            content_to_process = content[-1].content
        elif isinstance(content, str):
            content_to_process = content
        else:
            return content

        current_content = content_to_process

        try:
            parent_span = get_current_span()
            if parent_span is not None:
                ctx = trace.set_span_in_context(parent_span)
                with self.tracer.start_as_current_span(
                    f"guardrails:{content_type}", context=ctx
                ) as parent_gr_span:
                    parent_gr_span.set_attribute("node", str(node_name))
                    parent_gr_span.set_attribute("component", str(node_name))
                    parent_gr_span.set_attribute("content_type", str(content_type))

                    retry_count = 0
                    while retry_count <= self.max_retries:
                        group_result = self._apply_guardrail_group(current_content, guardrail_group_id)

                        if not group_result.get("success", False):
                            break

                        overall_start_iso = group_result.get("start_time", "")
                        data = group_result.get("data", {})
                        flow_summaries = data.get("flow_summaries", [])
                        total_tokens_all = sum(f.get("total_tokens", 0) for f in flow_summaries)

                        parent_gr_span.set_attribute("start_time", overall_start_iso)
                        parent_gr_span.set_attribute("end_time", group_result.get("end_time", ""))
                        parent_gr_span.set_attribute("duration", float(group_result.get("duration", 0.0)))
                        parent_gr_span.set_attribute("input.value", self._safe_str(current_content))
                        parent_gr_span.set_attribute("llm.token_count.total", total_tokens_all)
                        if action == "retry":
                            parent_gr_span.set_attribute("retry_count", retry_count)

                        overall_start_dt = (
                            datetime.fromisoformat(overall_start_iso) if overall_start_iso else datetime.now()
                        )

                        failed_flows: List[str] = []
                        for flow_summary in flow_summaries:
                            flow_name = flow_summary.get("flow_name", "unknown")
                            passed = flow_summary.get("passed", False)
                            detected_issue = not passed

                            flow_duration = flow_summary.get("duration", 0.0)
                            flow_start_iso = overall_start_iso
                            flow_end_iso = (overall_start_dt + timedelta(seconds=flow_duration)).isoformat()

                            individual = flow_summary.get("individual_results", [])
                            prompt_tokens = sum(r.get("prompt_tokens", 0) for r in individual)
                            completion_tokens = sum(r.get("completion_tokens", 0) for r in individual)
                            total_tokens = flow_summary.get("total_tokens", prompt_tokens + completion_tokens)

                            with self.tracer.start_as_current_span(
                                f"guardrail: {flow_name}",
                                context=trace.set_span_in_context(parent_gr_span),
                            ) as flow_span:
                                flow_span.set_attribute("component", str(node_name))
                                flow_span.set_attribute("guard", flow_name)
                                flow_span.set_attribute("content_type", str(content_type))
                                flow_span.set_attribute("detected", detected_issue)
                                flow_span.set_attribute("action", action)
                                flow_span.set_attribute("input.value", self._safe_str(current_content))
                                flow_span.set_attribute("output.value", json.dumps({
                                    "passed": flow_summary.get("passed"),
                                    "status": flow_summary.get("status"),
                                    "individual_results": individual,
                                }))
                                flow_span.set_attribute("start_time", flow_start_iso)
                                flow_span.set_attribute("end_time", flow_end_iso)
                                flow_span.set_attribute("duration", flow_duration)
                                flow_span.set_attribute("llm.token_count.prompt", prompt_tokens)
                                flow_span.set_attribute("llm.token_count.completion", completion_tokens)
                                flow_span.set_attribute("llm.token_count.total", total_tokens)
                                if action == "retry":
                                    flow_span.set_attribute("retry_count", retry_count)

                            if detected_issue:
                                if action == "block":
                                    raise ValueError(
                                        f"Guardrail flow '{flow_name}' detected an issue in {content_type}. Operation blocked."
                                    )
                                elif action == "retry":
                                    failed_flows.append(flow_name)

                        if failed_flows and action == "retry":
                            if self.llm is not None and retry_count < self.max_retries:
                                current_content = self._sanitize_with_llm(
                                    current_content, failed_flows, content_type
                                )
                                retry_count += 1
                                continue
                            else:
                                raise ValueError(
                                    f"Content failed guardrails {failed_flows} after {retry_count} retries "
                                    f"in {content_type}. Operation blocked."
                                )
                        else:
                            break  # all flows passed (or action is warn)

            if is_list:
                content[-1].content = current_content
                return content
            else:
                return current_content

        except ValueError:
            raise  # block signal — wrapper handles it
        except Exception:
            return content  # swallow unexpected errors only

    # --------- Block response ---------

    def _blocked_response(self, state: Any, key: Optional[str], message: str) -> Any:
        """Return a state dict with a 'Blocked by guardrail' AIMessage appended to the
        correct messages key, so the graph receives a clean output instead of an exception."""
        blocked_msg = AIMessage(content=f"Blocked by guardrail: {message}")

        actual_key = key
        if actual_key is None and isinstance(state, dict):
            for candidate in ("messages", "messag", "input"):
                if candidate in state:
                    actual_key = candidate
                    break

        if actual_key and isinstance(state, dict):
            existing = state.get(actual_key, [])
            if isinstance(existing, list):
                return {actual_key: existing + [blocked_msg]}
            return {actual_key: blocked_msg.content}

        return {"messages": [blocked_msg]}

    # --------- Group execution ---------

    def _fetch_guardrail_group(self, guardrail_group_id: str) -> List[Dict[str, Any]]:
        url = f"{GUARDRAILS_GET}/{guardrail_group_id}"
        if getattr(self, "organization_id", None):
            url += f"?organization_id={self.organization_id}"
        response = self.client.get(url)
        if not response.get("success", False):
            raise ValueError(f"Failed to fetch guardrail group '{guardrail_group_id}'")
        details = response.get("details", {})
        if "guardrail" in details and isinstance(details["guardrail"], dict):
            return details["guardrail"].get("guardrail_flows", [])
        return details.get("guardrail_flows", [])

    def _apply_guardrail_group(self, content: str, guardrail_group_id: str) -> Dict[str, Any]:
        guardrail_flows = self._fetch_guardrail_group(guardrail_group_id)
        payload = {"input_data": content, "guardrails": guardrail_flows}
        start_time = datetime.now()
        response = self.client.post(GUARDRAILS_RUN, payload=payload)
        end_time = datetime.now()
        if isinstance(response, dict):
            response["start_time"] = start_time.isoformat()
            response["end_time"] = end_time.isoformat()
            response["duration"] = (end_time - start_time).total_seconds()
        return response

    # --------- Sanitize helpers ---------

    def _sanitize_with_llm(
        self, content: str, failed_flows: List[str], content_type: str
    ) -> str:
        """Use self.llm to sanitize content based on which guardrail flows failed."""
        if self.llm is None:
            return content
        prompt = self._build_sanitize_prompt(failed_flows, content, content_type)
        try:
            response = self.llm.invoke(prompt)
            if hasattr(response, "content"):
                return str(response.content)
            return str(response)
        except Exception:
            return content

    def _build_sanitize_prompt(
        self, guard_names: Union[str, List[str]], content: Any, content_type: str
    ) -> str:
        instructions = {
            "Detect PII": "Remove or mask any personally identifiable information (PII).",
            "NSFW Text": "Remove or mask any not-safe-for-work (NSFW) content.",
            "Ban List": "Remove or mask any banned words.",
            "Bias Check": "Remove or mask any biased language.",
            "Competitor Check": "Remove or mask any competitor names.",
            "Toxic Language": "Remove or mask any toxic language.",
        }
        if isinstance(guard_names, list):
            parts = [instructions[n] for n in guard_names if n in instructions]
            combined = " Also, ".join(parts) if parts else "Sanitize the text according to the guardrail requirements."
        else:
            combined = instructions.get(
                guard_names,
                "Sanitize the text according to the guardrail requirements.",
            )
        return (
            f"Rewrite the following {content_type} text so it passes content safety checks. "
            f"{combined} Do not change anything else.\n\nContent:\n{content}"
        )

    @staticmethod
    def _safe_str(value: Any) -> str:
        try:
            if isinstance(value, (str, int, float, bool)) or value is None:
                return str(value)
            if hasattr(value, "content"):
                return str(getattr(value, "content", ""))
            if isinstance(value, (list, tuple)):
                parts = []
                for item in value:
                    parts.append(
                        Guard._safe_str(item)
                        if hasattr(Guard, "_safe_str")
                        else str(item)
                    )
                return ", ".join(parts)
            if isinstance(value, dict):
                safe_dict: Dict[str, Any] = {}
                for k, v in value.items():
                    key = str(k)
                    if isinstance(v, (str, int, float, bool)) or v is None:
                        safe_dict[key] = v
                    elif hasattr(v, "content"):
                        safe_dict[key] = str(getattr(v, "content", ""))
                    else:
                        safe_dict[key] = str(v)
                return json.dumps(safe_dict, ensure_ascii=False)
            return str(value)
        except Exception:
            return "<unserializable>"


def create_guardrail(project: Project) -> LangGraphGuardrail:
    """Quick factory function to create a guardrail instance with a project."""
    return LangGraphGuardrail(project=project)
