import json
from typing import Dict, Any, Optional, List, Union
from datetime import datetime, timedelta
from autogen import ConversableAgent, UserProxyAgent
from autogen_agentchat.agents import AssistantAgent
from lexsi_sdk.core.project import Project
from lexsi_sdk.common.xai_uris import GUARDRAILS_GET, GUARDRAILS_RUN
from opentelemetry import trace, context


class GuardrailSupervisor:
    """Coordinator class that applies a guardrail group to AutoGen agents in parallel."""

    def __init__(
        self,
        guardrail_group_id: Optional[str] = None,
        apply_to: str = "both",
        action: str = "block",
        project: Optional[Project] = None,
        llm: Optional[Any] = None,
        max_retries: int = 1,
    ):
        if apply_to not in ["input", "output", "both"]:
            raise ValueError("apply_to must be one of 'input', 'output', 'both'")
        self.apply_to = apply_to
        if action not in ["block", "retry", "warn"]:
            raise ValueError("action must be one of 'block', 'retry', 'warn'")
        self.action = action
        self.guardrail_group_id = guardrail_group_id
        if project is not None:
            self.api_client = project.api_client
            self.project_name = project.project_name
            self.organization_id = getattr(project, "organization_id", None)
        self.llm = llm
        self.max_retries = max_retries
        self.tracer = trace.get_tracer("autogen-app")

    # --------- Agent instrumentation ---------

    def instrument_agents(
        self, agents: List[Union[ConversableAgent, AssistantAgent]]
    ) -> List[Union[ConversableAgent, AssistantAgent]]:
        """Instrument a list of agents to apply the guardrail group."""
        for agent in agents:
            if isinstance(agent, AssistantAgent):
                self.instrument_agent(agent)
        return agents

    def instrument_agent(self, agent) -> None:
        """Wrap an AssistantAgent's run method with guardrail checks."""
        original_run = agent.run

        async def wrapped_run(*args, **kwargs):
            current_span = trace.get_current_span()
            current_context = context.get_current()

            if not current_span.is_recording():
                with self.tracer.start_as_current_span(f"{agent.name}_run"):
                    current_context = context.get_current()
                    return await self._execute_guarded_run(
                        agent, original_run, args, kwargs, current_context
                    )
            else:
                return await self._execute_guarded_run(
                    agent, original_run, args, kwargs, current_context
                )

        agent.run = wrapped_run

    async def _execute_guarded_run(self, agent, original_run, args, kwargs, current_context):
        task = kwargs.get("task") or (args[0] if args else None)

        if self.apply_to in ["input", "both"] and task:
            request_content = self._extract_task_content(task)
            if request_content:
                current_span = trace.get_current_span()
                if current_span.is_recording():
                    current_span.set_attribute(
                        "guardrail.input_content", self._safe_str(request_content)
                    )
                sanitized = self._run_with_action(request_content, agent.name, "input", current_context)
                if sanitized != request_content:
                    # propagate sanitized content back into call args
                    if "task" in kwargs:
                        kwargs["task"] = sanitized
                    elif args:
                        args = (sanitized,) + args[1:]

        try:
            reply = await original_run(*args, **kwargs)
        except ValueError:
            raise
        except Exception as e:
            raise ValueError(f"Error generating response: {str(e)}")

        if self.apply_to in ["output", "both"] and reply:
            response_content = self._extract_response_content(reply)
            if response_content:
                sanitized = self._run_with_action(response_content, agent.name, "output", current_context)
                if sanitized != response_content:
                    reply = sanitized

        return self._format_reply(reply, agent)

    # --------- Action dispatcher ---------

    def _run_with_action(
        self, content: str, agent_id: str, content_type: str, ctx
    ) -> str:
        """Run guardrails and apply the configured action. Returns (possibly sanitized) content."""
        if self.action in ("block", "warn"):
            self._run_guardrails(content, agent_id, content_type, ctx, retry_count=0)
            return content

        # retry
        current_content = content
        retry_count = 0
        while retry_count <= self.max_retries:
            failed_flows = self._run_guardrails(
                current_content, agent_id, content_type, ctx, retry_count=retry_count
            )
            if failed_flows:
                if self.llm is not None and retry_count < self.max_retries:
                    current_content = self._sanitize_with_llm(current_content, failed_flows, content_type)
                    retry_count += 1
                else:
                    raise ValueError(
                        f"Content failed guardrails {failed_flows} after {retry_count} retries "
                        f"in {content_type} for agent '{agent_id}'. Operation blocked."
                    )
            else:
                break
        return current_content

    # --------- Core guardrail execution ---------

    def _run_guardrails(
        self, content: str, agent_id: str, content_type: str, ctx, retry_count: int = 0
    ) -> List[str]:
        """Run the guardrail group and return the list of failed flow names.
        Raises ValueError immediately for action='block'."""
        if not self.guardrail_group_id:
            return []

        failed_flows: List[str] = []

        with self.tracer.start_as_current_span(
            f"guardrails:{content_type}", context=ctx
        ) as parent_gr_span:
            parent_gr_span.set_attribute("component", agent_id)
            parent_gr_span.set_attribute("content_type", content_type)
            if self.action == "retry":
                parent_gr_span.set_attribute("retry_count", retry_count)

            try:
                guardrail_flows = self._fetch_guardrail_group(self.guardrail_group_id)
                payload = {"input_data": content, "guardrails": guardrail_flows}

                start_time = datetime.now()
                response = self.api_client.post(GUARDRAILS_RUN, payload=payload)
                end_time = datetime.now()

                if not response.get("success", False):
                    parent_gr_span.set_attribute("error.message", str(response.get("details", "")))
                    return []

                data = response.get("data", {})
                flow_summaries = data.get("flow_summaries", [])
                total_tokens_all = sum(f.get("total_tokens", 0) for f in flow_summaries)

                parent_gr_span.set_attribute("start_time", start_time.isoformat())
                parent_gr_span.set_attribute("end_time", end_time.isoformat())
                parent_gr_span.set_attribute("duration", (end_time - start_time).total_seconds())
                parent_gr_span.set_attribute("input.value", self._safe_str(content))
                parent_gr_span.set_attribute("llm.token_count.total", total_tokens_all)

                for flow_summary in flow_summaries:
                    flow_name = flow_summary.get("flow_name", "unknown")
                    passed = flow_summary.get("passed", False)
                    detected_issue = not passed

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
                        flow_span.set_attribute("component", agent_id)
                        flow_span.set_attribute("guard", flow_name)
                        flow_span.set_attribute("content_type", content_type)
                        flow_span.set_attribute("detected", detected_issue)
                        flow_span.set_attribute("action", self.action)
                        flow_span.set_attribute("input.value", self._safe_str(content))
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
                        if self.action == "retry":
                            flow_span.set_attribute("retry_count", retry_count)

                        if detected_issue:
                            flow_span.add_event(
                                f"{content_type}_guardrail_violation",
                                {"agent": agent_id, "flow": flow_name},
                            )
                        else:
                            flow_span.add_event(
                                f"{content_type}_guardrail_passed",
                                {"agent": agent_id, "flow": flow_name},
                            )

                    if detected_issue:
                        if self.action == "block":
                            raise ValueError(
                                f"Guardrail flow '{flow_name}' detected an issue for agent "
                                f"'{agent_id}' in {content_type}. Operation blocked."
                            )
                        elif self.action == "retry":
                            failed_flows.append(flow_name)

            except ValueError:
                raise  # block signal — propagate up
            except Exception as e:
                parent_gr_span.record_exception(e)
                parent_gr_span.set_attribute("error.message", str(e))

        return failed_flows

    # --------- Group fetch ---------

    def _fetch_guardrail_group(self, guardrail_group_id: str) -> List[Dict[str, Any]]:
        url = f"{GUARDRAILS_GET}/{guardrail_group_id}"
        if getattr(self, "organization_id", None):
            url += f"?organization_id={self.organization_id}"
        response = self.api_client.get(url)
        if not response.get("success", False):
            raise ValueError(f"Failed to fetch guardrail group '{guardrail_group_id}'")
        details = response.get("details", {})
        if "guardrail" in details and isinstance(details["guardrail"], dict):
            return details["guardrail"].get("guardrail_flows", [])
        return details.get("guardrail_flows", [])

    # --------- Sanitize helpers ---------

    def _sanitize_with_llm(self, content: str, failed_flows: List[str], content_type: str) -> str:
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

    # --------- Content extraction / formatting ---------

    def _extract_task_content(self, task) -> str:
        if isinstance(task, str):
            return task
        elif isinstance(task, list):
            content_parts = []
            for item in task:
                if isinstance(item, dict):
                    content_parts.append(item.get("content", str(item)))
                elif hasattr(item, "content"):
                    content_parts.append(str(item.content))
                else:
                    content_parts.append(str(item))
            return " ".join(filter(None, content_parts))
        elif isinstance(task, dict):
            return task.get("content", str(task))
        elif hasattr(task, "content"):
            return str(task.content)
        else:
            return str(task)

    def _extract_response_content(self, reply) -> str:
        if isinstance(reply, str):
            return reply
        elif isinstance(reply, dict):
            return reply.get("content", "")
        elif hasattr(reply, "content"):
            return str(reply.content)
        elif hasattr(reply, "text"):
            return str(reply.text)
        else:
            return str(reply)

    def _format_reply(self, reply, agent) -> Dict[str, Any]:
        agent_name = getattr(agent, "name", "assistant")
        if isinstance(reply, str):
            return {"role": "assistant", "content": reply, "name": agent_name}
        elif isinstance(reply, dict):
            formatted = reply.copy()
            formatted.setdefault("role", "assistant")
            formatted.setdefault("name", agent_name)
            if "content" not in formatted:
                content = self._extract_response_content(reply)
                if content:
                    formatted["content"] = content
            return formatted
        else:
            content = self._extract_response_content(reply)
            return {"role": "assistant", "content": content, "name": agent_name}

    def _safe_str(self, value, max_length: int = 1000) -> str:
        if value is None:
            return ""
        s = str(value)
        if len(s) > max_length:
            return s[:max_length] + "... [truncated]"
        return s
