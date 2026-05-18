from __future__ import annotations

import json
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional, TypedDict, Union
from lexsi_sdk.common.xai_uris import GUARDRAILS_GET, GUARDRAILS_RUN
from lexsi_sdk.core.project import Project
from opentelemetry import trace
import asyncio

from agents import (
    Agent,
    RunContextWrapper,
    TResponseInputItem,
    input_guardrail,
    output_guardrail,
    ModelSettings,
    ModelTracing,
)
from dataclasses import dataclass


@dataclass
class GuardrailFunctionOutput:
    """Output of a single guardrail execution."""

    output_info: Any
    tripwire_triggered: bool
    sanitized_content: str


class OpenAIAgentsGuardrail:
    """Decorator-style guardrail utility for OpenAI Agents."""

    def __init__(
        self,
        project: Optional[Project],
        model: Optional[Any] = None,
        max_retries: int = 2,
    ) -> None:
        if project is not None:
            self.client = project.api_client
            self.project_name = project.project_name
            self.organization_id = getattr(project, "organization_id", None)

        self.logs: List[Dict[str, Any]] = []
        self.max_retries = max_retries
        self.retry_delay = 1.0
        self.tracer = trace.get_tracer(__name__)
        self.model = model

    def create_input_guardrail(
        self,
        guardrail_group_id: str,
        action: str = "block",
        name: str = "input_guardrail",
    ) -> Callable:
        """Create an input guardrail function backed by the guardrail group
        identified by ``guardrail_group_id``.

        :param guardrail_group_id: ID of the organization guardrail group.
        :param action: 'block' | 'retry' | 'warn'.
        :param name: Name for the guardrail function.
        :return: Callable suitable for OpenAI Agents guardrail hook.
        """

        @input_guardrail
        async def guardrail_function(
            ctx: RunContextWrapper[None],
            agent: Agent,
            input: str | list[TResponseInputItem],
        ) -> GuardrailFunctionOutput:
            if isinstance(input, list):
                input_text = " ".join(
                    str(item.content) if hasattr(item, "content") else str(item)
                    for item in input
                ).strip()
            else:
                input_text = str(input)

            current_content, tripwire_triggered, output_info = (
                await self._apply_guardrail_group(
                    content=input_text,
                    guardrail_group_id=guardrail_group_id,
                    guardrail_type="input",
                    action=action,
                    agent_name=agent.name,
                )
            )

            return GuardrailFunctionOutput(
                output_info=output_info,
                tripwire_triggered=tripwire_triggered,
                sanitized_content=current_content if action == "retry" else input_text,
            )

        guardrail_function.__name__ = name
        return guardrail_function

    def create_output_guardrail(
        self,
        guardrail_group_id: str,
        action: str = "block",
        name: str = "output_guardrail",
    ) -> Callable:
        """Create an output guardrail function backed by the guardrail group
        identified by ``guardrail_group_id``.

        :param guardrail_group_id: ID of the organization guardrail group.
        :param action: 'block' | 'retry' | 'warn'.
        :param name: Name for the guardrail function.
        :return: Callable suitable for OpenAI Agents guardrail hook.
        """

        @output_guardrail
        async def guardrail_function(
            ctx: RunContextWrapper, agent: Agent, output: Any
        ) -> GuardrailFunctionOutput:
            if hasattr(output, "response"):
                output_text = str(output.response)
            elif hasattr(output, "content"):
                output_text = str(output.content)
            else:
                output_text = str(output)

            current_content, tripwire_triggered, output_info = (
                await self._apply_guardrail_group(
                    content=output_text,
                    guardrail_group_id=guardrail_group_id,
                    guardrail_type="output",
                    action=action,
                    agent_name=agent.name,
                )
            )

            return GuardrailFunctionOutput(
                output_info=output_info,
                tripwire_triggered=tripwire_triggered,
                sanitized_content=current_content if action == "retry" else output_text,
            )

        guardrail_function.__name__ = name
        return guardrail_function

    async def _apply_guardrail_group(
        self,
        content: Any,
        guardrail_group_id: str,
        guardrail_type: str,
        action: str,
        agent_name: str,
    ) -> tuple[Any, bool, Dict[str, Any]]:
        """Fetch the guardrail group, run all its flows in parallel, and return
        (processed_content, tripwire_triggered, output_info).
        """
        current_content = content
        tripwire_triggered = False
        output_info: Dict[str, Any] = {}
        retry_count = 0

        try:
            parent_span = trace.get_current_span()
            if parent_span is not None:
                ctx = trace.set_span_in_context(parent_span)
                with self.tracer.start_as_current_span(
                    f"guardrails:{guardrail_type}", context=ctx
                ) as parent_gr_span:
                    parent_gr_span.set_attribute("component", str(agent_name))
                    parent_gr_span.set_attribute("content_type", guardrail_type)

                    while retry_count <= self.max_retries:
                        guardrail_flows = self._fetch_guardrail_group(guardrail_group_id)
                        payload = {"input_data": current_content, "guardrails": guardrail_flows}

                        start_time = datetime.now()
                        response = self.client.post(GUARDRAILS_RUN, payload=payload)
                        end_time = datetime.now()

                        response["start_time"] = start_time.isoformat()
                        response["end_time"] = end_time.isoformat()
                        response["duration"] = (end_time - start_time).total_seconds()

                        if not response.get("success", False):
                            output_info["execution_error"] = response.get("details", {})
                            return current_content, tripwire_triggered, output_info

                        data = response.get("data", {})
                        flow_summaries = data.get("flow_summaries", [])
                        total_tokens_all = sum(f.get("total_tokens", 0) for f in flow_summaries)

                        parent_gr_span.set_attribute("start_time", start_time.isoformat())
                        parent_gr_span.set_attribute("end_time", end_time.isoformat())
                        parent_gr_span.set_attribute("duration", response["duration"])
                        parent_gr_span.set_attribute("input.value", self._safe_str(current_content))
                        parent_gr_span.set_attribute("llm.token_count.total", total_tokens_all)

                        detected_issue = False
                        for flow_summary in flow_summaries:
                            flow_name = flow_summary.get("flow_name", "unknown")
                            passed = flow_summary.get("passed", False)
                            is_issue = not passed

                            flow_duration = flow_summary.get("duration", 0.0)
                            flow_start_iso = start_time.isoformat()
                            flow_end_iso = (start_time + timedelta(seconds=flow_duration)).isoformat()

                            individual = flow_summary.get("individual_results", [])
                            prompt_tokens = sum(r.get("prompt_tokens", 0) for r in individual)
                            completion_tokens = sum(r.get("completion_tokens", 0) for r in individual)
                            total_tokens = flow_summary.get("total_tokens", prompt_tokens + completion_tokens)

                            with self.tracer.start_as_current_span(
                                f"guardrail: {flow_name}",
                                context=trace.set_span_in_context(parent_gr_span),
                            ) as flow_span:
                                flow_span.set_attribute("component", str(agent_name))
                                flow_span.set_attribute("guard", flow_name)
                                flow_span.set_attribute("content_type", guardrail_type)
                                flow_span.set_attribute("detected", is_issue)
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

                            self._log_flow_result(
                                flow_name=flow_name,
                                flow_summary=flow_summary,
                                action=action,
                                agent_name=agent_name,
                                guardrail_type=guardrail_type,
                            )

                            if is_issue:
                                detected_issue = True
                                output_info[f"flow_{flow_name}"] = flow_summary
                                if action == "block":
                                    tripwire_triggered = True
                                    return current_content, tripwire_triggered, output_info

                        if (
                            detected_issue
                            and action == "retry"
                            and self.model is not None
                            and retry_count < self.max_retries
                        ):
                            prompt = self._build_sanitize_prompt("combined", current_content, guardrail_type)
                            try:
                                current_content = await self._invoke_llm(prompt)
                            except Exception:
                                pass
                            retry_count += 1
                            await asyncio.sleep(self.retry_delay)
                            continue
                        else:
                            # Only trigger tripwire for retry when retries are exhausted
                            if detected_issue and action == "retry" and retry_count >= self.max_retries:
                                tripwire_triggered = True
                            output_info["retry_count"] = retry_count
                            output_info["final_content"] = current_content
                            return current_content, tripwire_triggered, output_info

            output_info["retry_count"] = retry_count
            output_info["final_content"] = current_content
            return current_content, tripwire_triggered, output_info

        except Exception as e:
            output_info["error"] = f"Guardrail group execution failed: {str(e)}"
            return current_content, tripwire_triggered, output_info

    # --------- Group fetch ---------

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

    # --------- Logging / helpers ---------

    def _log_flow_result(
        self,
        flow_name: str,
        flow_summary: Dict[str, Any],
        action: str,
        agent_name: str,
        guardrail_type: str,
    ) -> None:
        self.logs.append({
            "flow_name": flow_name,
            "guardrail_type": guardrail_type,
            "agent_name": agent_name,
            "action": action,
            "detected_issue": not flow_summary.get("passed", False),
            "status": flow_summary.get("status", ""),
            "duration": float(flow_summary.get("duration", 0.0)),
        })

    def _build_sanitize_prompt(
        self, guard_name: str, content: Any, guardrail_type: str
    ) -> str:
        instructions = {
            "Detect PII": "Sanitize the following text by removing or masking any personally identifiable information (PII). Do not change anything else.",
            "Toxic Language": "Sanitize the following text by removing or masking any toxic language. Do not change anything else.",
        }
        instruction = instructions.get(
            guard_name,
            "Sanitize the following text according to the guardrail requirements. Do not change anything else.",
        )
        return f"{instruction}\n\nContent:\n{content}"

    async def _invoke_llm(self, prompt: str) -> str:
        if self.model is None:
            return prompt
        try:
            data = await self.model.get_response(
                system_instructions="Based on the input you have to provide the best and accurate results",
                input=prompt,
                model_settings=ModelSettings(temperature=0.1),
                tools=[],
                output_schema=None,
                handoffs=[],
                tracing=ModelTracing.DISABLED,
                previous_response_id=None,
            )
            return str(data.output[0].content[0].text)
        except Exception:
            return prompt

    @staticmethod
    def _safe_str(value: Any) -> str:
        try:
            if isinstance(value, (str, int, float, bool)) or value is None:
                return str(value)
            if hasattr(value, "content"):
                return str(getattr(value, "content", ""))
            if isinstance(value, (list, tuple)):
                return ", ".join(OpenAIAgentsGuardrail._safe_str(item) for item in value)
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


def create_guardrail(
    project: Project, model: Optional[Any] = None
) -> OpenAIAgentsGuardrail:
    """Quick factory function to create a guardrail instance with a project."""
    return OpenAIAgentsGuardrail(project=project, model=model)
