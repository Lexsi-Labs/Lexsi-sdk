import json
import time
import functools
from typing import Callable, Optional
import inspect
import uuid
import botocore.client
from openai import OpenAI
from anthropic import Anthropic
from google import genai
from mistralai import Mistral
from pydantic import BaseModel

import requests
import httpx
from lexsi_sdk.client.client import APIClient
from lexsi_sdk.common.environment import Environment
from lexsi_sdk.common.xai_uris import CASE_INFO_TEXT_URI, GENERATE_TEXT_CASE_STREAM_URI, GENERATE_TEXT_CASE_URI

from together import Together
from groq import Groq
import replicate
from huggingface_hub import InferenceClient
import boto3
from xai_sdk import Client
import botocore


class Wrapper:
    """Wraps SDK clients to add Lexsi tracing, logging, and guardrails."""

    def __init__(self, project_name, api_client):
        """Store project context for downstream wrapper calls.

        :param project_name: Name of the Lexsi project.
        :param api_client: Initialized API client used for telemetry calls.
        """
        self.project_name = project_name
        self.api_client = api_client

    def add_message(self, session_id, trace_id, input_data, output_data, metadata, duration):
        """Persist a message with timing metadata to Lexsi tracing.

        :param session_id: Session identifier returned by tracing APIs.
        :param trace_id: Trace identifier for the current LLM call chain.
        :param input_data: Raw input payload sent to the model.
        :param output_data: Model output payload.
        :param metadata: Any additional metadata to persist.
        :param duration: End-to-end latency for the operation.
        :return: API response from the tracing endpoint.
        """
        payload = {
            "project_name": self.project_name,
            "session_id": session_id,
            "trace_id": trace_id,
            "input_data": input_data,
            "output_data": output_data,
            "metadata": metadata,
            "duration": duration,
        }
        try:
            res = self.api_client.post("sessions/add_session_message", payload=payload)
            return res
        except Exception as e:
            raise e
    
    async def async_add_trace_details(self, session_id, trace_id, component, input_data, metadata, output_data=None, function_to_run=None):
        """Create an async trace entry, optionally executing and recording a coroutine.

        :param session_id: Session identifier returned by tracing APIs.
        :param trace_id: Trace identifier for the current call.
        :param component: Logical component name (Input, LLM, Guardrails, etc.).
        :param input_data: Payload being traced.
        :param metadata: Extra metadata to attach to the trace.
        :param output_data: Optional output to store; computed if omitted.
        :param function_to_run: Awaitable to execute and trace around.
        :return: API response or the result of the wrapped coroutine.
        """
        start_time = time.perf_counter()
        result = None
        if function_to_run:
            try:
                result = await function_to_run()
            except Exception as e:
                raise
        duration = time.perf_counter() - start_time
        if not output_data and result is not None:
            output_data = result
            if isinstance(result, BaseModel):
                output_data = result.model_dump()
        payload = {
            "project_name": self.project_name,
            "trace_id": trace_id,
            "session_id": session_id,
            "component": component,
            "input_data": input_data,
            "output_data": output_data,
            "metadata": metadata,
            "duration": duration,
        }
        res = self.api_client.post("traces/add_trace", payload=payload)
        if function_to_run:
            if component in ["Input Guardrails", "Output Guardrails"]:
                if not result.get("success", True):
                    return result.get("details")
            return result
        return res

    def add_trace_details(self, session_id, trace_id, component, input_data, metadata, is_grok = False, output_data=None, function_to_run=None):
        """Create a trace entry for synchronous flows, executing optional callable.

        :param session_id: Session identifier returned by tracing APIs.
        :param trace_id: Trace identifier for the current call.
        :param component: Logical component name (Input, LLM, Guardrails, etc.).
        :param input_data: Payload being traced.
        :param metadata: Extra metadata to attach to the trace.
        :param is_grok: Whether the result is from Grok and needs special handling.
        :param output_data: Optional output to store; computed if omitted.
        :param function_to_run: Callable to execute and trace around.
        :return: API response or the result of the wrapped callable.
        """
        start_time = time.perf_counter()
        result = None
        if function_to_run:
            try:
                if is_grok:
                    result = function_to_run()
                    result = {
                        "id": result.id,
                        "content" : result.content,
                        "reasoning_content":result.reasoning_content,
                        "system_fingerprint": result.system_fingerprint,
                        "usage" : {
                            "completion_tokens" : result.usage.completion_tokens,
                            "prompt_tokens:" : result.usage.prompt_tokens,
                            "total_tokens" : result.usage.total_tokens,
                            "prompt_text_tokens" : result.usage.prompt_text_tokens,
                            "reasoning_tokens" : result.usage.reasoning_tokens,
                            "cached_prompt_text_tokens" : result.usage.cached_prompt_text_tokens
                        }
                    }
                else:
                    result = function_to_run()
            except Exception as e:
                raise
        duration = time.perf_counter() - start_time
        if not output_data and result is not None:
            output_data = result
            if isinstance(result, BaseModel):
                output_data = result.model_dump()
        
        payload = {
            "project_name": self.project_name,
            "trace_id": trace_id,
            "session_id": session_id,
            "component": component,
            "input_data": input_data,
            "output_data": output_data,
            "metadata": metadata,
            "duration": duration,
        }
        res = self.api_client.post("traces/add_trace", payload=payload)
        if function_to_run:
            if component in ["Input Guardrails", "Output Guardrails"]:
                if not result.get("success", True):
                    return result.get("details")
            return result
        return res

    def run_guardrails(self, input_data, trace_id, session_id, model_name, apply_on):
        """Invoke server-side guardrails for the provided input/output payload.

        :param input_data: Payload to validate with guardrails.
        :param trace_id: Trace identifier for this run.
        :param session_id: Session identifier for this run.
        :param model_name: Model identifier to supply to guardrails.
        :param apply_on: Whether guardrails apply to input or output.
        :return: API response from guardrails execution.
        """
        payload = {
            "trace_id": trace_id,
            "session_id": session_id,
            "input_data": input_data,
            "model_name": model_name,
            "project_name": self.project_name,
            "apply_on": apply_on
        }
        try:
            res = self.api_client.post("v2/ai-models/run_guardrails", payload=payload)
            return res
        except Exception as e:
            raise

    def _get_wrapper(self, original_method: Callable, method_name: str, provider: str, session_id: Optional[str] = None, chat=None , **extra_kwargs) -> Callable:
        """Return a callable that wraps LLM SDK methods with Lexsi telemetry.

        :param original_method: SDK method being wrapped.
        :param method_name: Identifier describing the wrapped method.
        :param provider: Name of the model provider (OpenAI, Anthropic, etc.).
        :param session_id: Optional session id to reuse.
        :param chat: Optional chat object for Grok/XAI.
        :param extra_kwargs: Extra kwargs to inject into wrapped calls.
        :return: Wrapped callable preserving the original signature.
        """
        if inspect.iscoroutinefunction(original_method):
            @functools.wraps(original_method)
            async def async_wrapper(*args, **kwargs):
                """Async wrapper around SDK method to add tracing and guardrails."""
                total_start_time = time.perf_counter()
                trace_id = str(uuid.uuid4())
                # model_name = kwargs.get("model")
                model_name = provider
                input_data = kwargs.get("messages")

                trace_res = self.add_trace_details(
                    trace_id=trace_id,
                    session_id=session_id,
                    component="Input",
                    input_data=input_data,
                    output_data=input_data,
                    metadata={},
                )
                id_session = trace_res.get("details", {}).get("session_id")
                
               
                self.add_trace_details(
                    trace_id=trace_id,
                    session_id=id_session,
                    component="Input Guardrails",
                    input_data=input_data,
                    metadata={},
                    function_to_run=lambda: self.run_guardrails(
                        session_id=id_session,
                        trace_id=trace_id,
                        input_data=input_data,
                        model_name=model_name,
                        apply_on="input"
                    )
                )

                result = await self.async_add_trace_details(
                    trace_id=trace_id,
                    session_id=id_session,
                    component="LLM",
                    input_data=input_data,
                    metadata=kwargs,
                    function_to_run= lambda : original_method(*args, **kwargs)
                )

                output_data = result.choices[0].message.content


                self.add_trace_details(
                    trace_id=trace_id,
                    session_id=id_session,
                    component="Output Guardrails",
                    input_data=output_data,
                    metadata={},
                    function_to_run=lambda: self.run_guardrails(
                        session_id=id_session,
                        trace_id=trace_id,
                        model_name=model_name,
                        input_data=output_data,
                        apply_on="output"
                    )
                )

                self.add_trace_details(
                    trace_id=trace_id,
                    session_id=id_session,
                    component="Output",
                    input_data=input_data,
                    output_data=output_data,
                    metadata={},
                )

                self.add_message(
                    trace_id=trace_id,
                    session_id=id_session,
                    input_data=input_data,
                    output_data=output_data,
                    duration=time.perf_counter() - total_start_time,
                    metadata={}
                )

                return result
            return async_wrapper
        else:
            @functools.wraps(original_method)
            def wrapper(*args, **kwargs):
                """Sync wrapper around SDK method to add tracing and guardrails."""
                total_start_time = time.perf_counter()
                trace_id = str(uuid.uuid4())
                model_name = None
                input_data = None

                if extra_kwargs:
                    kwargs.update(extra_kwargs)
                # Handle input data based on method
                if method_name == "client.chat.completions.create":  # OpenAI (Completions)
                    input_data = kwargs.get("messages")
                    # model_name = kwargs.get("model")
                    model_name = provider
                elif method_name == "client.responses.create":  # OpenAI (Response)
                    input_data = kwargs.get("input")
                    # model_name = kwargs.get("model")
                    model_name = provider
                elif method_name == "client.messages.create":  # Anthropic Messages API
                    input_data = kwargs.get("messages")
                    # model_name = kwargs.get("model")
                    model_name = provider
                elif method_name == "client.models.generate_content":  # Gemini
                    input_data = kwargs.get("contents")
                    # model_name = kwargs.get("model")
                    model_name = provider
                elif method_name == "client.chat.complete":  # Mistral
                    input_data = kwargs.get("messages")
                    # model_name = kwargs.get("model")
                    model_name = provider
                elif method_name == "client.chat_completion":
                    input_data = kwargs.get("messages")
                    # model_name = kwargs.get("model")
                    model_name = provider
                elif method_name == "client.generate_text_case":  # LexsiModels
                    input_data = kwargs.get("prompt")
                    model_name = kwargs.get("model_name")
                elif method_name == "client.run":
                    input_data = kwargs.get("input")
                    # model_name = args.index(0)
                    model_name = provider
                elif method_name == "client.converse":  #  Bedrock
                    input_data = kwargs.get("messages")
                    # model_name = kwargs.get("modelId")
                    model_name = provider
                elif method_name == "chat.sample":  # XAI Grok
                        input_data = chat.messages[1].content[0].text
                        # model_name = None
                        model_name = provider
                else:
                    input_data = kwargs
                    model_name = None


                trace_res = self.add_trace_details(
                    trace_id=trace_id,
                    session_id=session_id,
                    component="Input",
                    input_data=input_data,
                    output_data=input_data,
                    metadata={},
                )
                id_session = trace_res.get("details", {}).get("session_id")

                self.add_trace_details(
                    trace_id=trace_id,
                    session_id=id_session,
                    component="Input Guardrails",
                    input_data=input_data,
                    metadata={},
                    function_to_run=lambda: self.run_guardrails(
                        session_id=id_session,
                        trace_id=trace_id,
                        input_data=input_data,
                        model_name=model_name,
                        apply_on="input"
                    )
                )

                if method_name == "client.generate_text_case":
                    kwargs["session_id"] = id_session
                    kwargs["trace_id"] = trace_id

                sanitized_kwargs = {}
                if method_name == "chat.sample":
                    # For XAI, use the stored kwargs from chat creation
                    chat_obj = args[0] if args else None
                    if chat_obj and hasattr(chat_obj, '_original_kwargs'):
                        for key, value in chat_obj._original_kwargs.items():
                            try:
                                import json
                                json.dumps(value)
                                sanitized_kwargs[key] = value
                            except (TypeError, ValueError):
                                sanitized_kwargs[key] = str(value)


                if method_name == "chat.sample":
                    result = self.add_trace_details(
                        trace_id=trace_id,
                        session_id=id_session,
                        component="LLM",
                        is_grok=True,
                        input_data=input_data,
                        metadata=sanitized_kwargs,
                        function_to_run=lambda: original_method(*args) if method_name == "chat.sample" else lambda: original_method(*args, **kwargs)
                    )
                else:
                    result = self.add_trace_details(
                        trace_id=trace_id,
                        session_id=id_session,
                        component="LLM",
                        input_data=input_data,
                        metadata=kwargs,
                        function_to_run=lambda: original_method(*args, **kwargs)
                    )

                # Handle output data based on method
                if method_name == "chat.sample":  # XAI Grok
                    output_data = result
                elif method_name == "client.converse": # Bedrock
                    output_data = result["output"]["message"]["content"][-1]["text"]
                elif method_name == "client.responses.create":
                    output_data = result.output_text
                elif method_name == "client.messages.create":  # Anthropic Messages API
                    output_data = result.content[0].text
                elif method_name == "client.models.generate_content":  # Gemini
                    output_data = result.text
                elif method_name == "client.chat.complete":  # Mistral 
                    output_data = result.choices[0].message.content
                elif method_name == "client.chat.complete_async": # Mistral Async
                    output_data = result.choices[0].message.content
                elif method_name == "client.generate_text_case":  # LexsiModels
                    output_data = result.get("details", {}).get("result", {}).get("output")
                elif method_name == "client.run":   # Replicate
                    output_data == result
                elif method_name == "client.chat.completions.create" or "client.chat_completion":  # OpenAI
                    output_data = result.choices[0].message.content
                else:
                    output_data = result

                self.add_trace_details(
                    trace_id=trace_id,
                    session_id=id_session,
                    component="Output Guardrails",
                    input_data=output_data,
                    metadata={},
                    function_to_run=lambda: self.run_guardrails(
                        session_id=id_session,
                        trace_id=trace_id,
                        model_name=model_name,
                        input_data=output_data,
                        apply_on="output"
                    )
                )

                self.add_trace_details(
                    trace_id=trace_id,
                    session_id=id_session,
                    component="Output",
                    input_data=input_data,
                    output_data=output_data,
                    metadata={},
                )
                metadata = {}               
                if method_name == "client.generate_text_case":
                    
                    input_tokens = 0
                    output_tokens = 0
                    if result.get("details", {}).get("result", {}).get("audit_trail", {}).get("tokens", {}).get("input_tokens", None):
                        input_tokens = result.get("details", {}).get("result", {}).get("audit_trail", {}).get("tokens", {}).get("input_tokens")
                    elif result.get("details", {}).get("result", {}).get("audit_trail", {}).get("tokens", {}).get("input_decoded_length", None):
                        input_tokens = result.get("details", {}).get("result", {}).get("audit_trail", {}).get("tokens", {}).get("input_decoded_length")
                    
                    if result.get("details", {}).get("result", {}).get("audit_trail", {}).get("tokens", {}).get("output_tokens", None):
                        output_tokens = result.get("details", {}).get("result", {}).get("audit_trail", {}).get("tokens", {}).get("output_tokens")
                    elif result.get("details", {}).get("result", {}).get("audit_trail", {}).get("tokens", {}).get("output_decoded_length", None):
                        output_tokens = result.get("details", {}).get("result", {}).get("audit_trail", {}).get("tokens", {}).get("output_decoded_length")
                    total_tokens = input_tokens + output_tokens
                    metadata = {
                        "case_id":result.get("details",{}).get("case_id"),
                        "input_tokens": input_tokens,
                        "output_tokens": output_tokens,
                        "total_tokens": total_tokens
                    }
                self.add_message(
                    trace_id=trace_id,
                    session_id=id_session,
                    input_data=input_data,
                    output_data=output_data,
                    duration=time.perf_counter() - total_start_time,
                    metadata=metadata
                )

                return result
            return wrapper


class LexsiModels:
    """Convenience wrapper for Lexsi hosted text models."""

    def __init__(self, project, api_client: APIClient):
        """Bind project and API client for model operations.

        :param project: Project instance owning the model.
        :param api_client: API client with auth configured.
        """
        self.project = project
        self.api_client = api_client

    def generate_text_case(
        self,
        model_name: str,
        prompt: str,
        instance_type: Optional[str] = "xsmall",
        serverless_instance_type: str = "gova-2",
        explainability_method: list = ["DLB"],
        explain_model: bool = False,
        trace_id: str = None,
        session_id: str = None,
        min_tokens: int = 100,
        max_tokens: int = 500,
        stream: bool = False,
    ):
        """Generate an explainable text case using a hosted Lexsi model.

        :param model_name: Name of the deployed text model.
        :param prompt: Input prompt for generation.
        :param instance_type: Dedicated instance type (if applicable).
        :param serverless_instance_type: Serverless instance flavor.
        :param explainability_method: Methods to compute explanations with.
        :param explain_model: Whether to explain the model behavior.
        :param trace_id: Optional existing trace id.
        :param session_id: Optional existing session id.
        :param min_tokens: Minimum tokens to generate.
        :param max_tokens: Maximum tokens to generate.
        :param stream: Whether to stream responses.
        :return: API response with generation details.
        """
        payload = {
            "session_id": session_id,
            "trace_id": trace_id,
            "project_name": self.project.project_name,
            "model_name": model_name,
            "input_text": prompt,
            "instance_type": instance_type,
            "serverless_instance_type": serverless_instance_type,
            "explainability_method": explainability_method,
            "explain_model": explain_model,
            "max_tokens": max_tokens,
            "min_tokens": min_tokens,
            "stream": stream,
        }
        
    #     if stream:
    #         env = Environment()
    #         url = env.get_base_url() + "/" + GENERATE_TEXT_CASE_STREAM_URI
    #         with requests.post(
    #             url,
    #             headers={**self.api_client.headers, "Accept": "text/event-stream"},
    #             json=payload,
    #             stream=True,
    #         ) as response:
    #             response.raise_for_status()

    #             buffer = ""
    #             for line in response.iter_lines(decode_unicode=True):
    #                 if not line or line.strip() == "[DONE]":
    #                     continue

    #                 if line.startswith("data:"):
    #                     line = line[len("data:"):].strip()
    #                 try:
    #                     event = json.loads(line)
    #                     text_piece = event.get("text", "")
    #                 except Exception as e:
    #                     text_piece = line
    #                 buffer += text_piece
    #                 print(text_piece, end="", flush=True)
    #         response = {"details": {"result": {"output": buffer}}}
    #         payload = {
    #             "session_id": session_id,
    #             "trace_id": trace_id,
    #             "project_name": self.project.project_name
    #         }
    #         res = self.api_client.post(CASE_INFO_TEXT_URI, payload)
    #         return res
    #     else:
    #         #return "Text case generation is not supported for this modality type"
    #         res = self.api_client.post(GENERATE_TEXT_CASE_URI, payload)
    #         if not res.get("success"):
    #             raise Exception(res.get("details"))
    #         return res

        if stream:
            env = Environment()
            url = env.get_base_url() + "/" + GENERATE_TEXT_CASE_STREAM_URI

            headers = {
                **self.api_client.headers,
                "Accept": "text/event-stream",
            }

            # Use a client so we can enable HTTP/2 and connection reuse if needed
            with httpx.Client(http2=True, timeout=None) as client:
                with client.stream(
                    "POST",
                    url,
                    headers=headers,
                    json=payload,
                ) as response:
                    response.raise_for_status()

                    buffer = ""

                    for line in response.iter_lines():
                        if not line:
                            continue

                        # httpx may return str or bytes depending on encoding
                        if isinstance(line, bytes):
                            line = line.decode("utf-8", errors="ignore")

                        line = line.strip()
                        if not line or line == "[DONE]":
                            continue

                        if line.startswith("data:"):
                            line = line[len("data:"):].strip()

                        try:
                            event = json.loads(line)
                            text_piece = event.get("text", "")
                        except Exception:
                            # Fallback: treat raw line as text content
                            text_piece = line

                        buffer += text_piece
                        print(text_piece, end="", flush=True)

            # After stream finishes, send the case info payload
            session_id = payload.get("session_id")
            trace_id = payload.get("trace_id")
            payload_case = {
                "session_id": session_id,
                "trace_id": trace_id,
                "project_name": self.project.project_name,
            }
            res = self.api_client.post(CASE_INFO_TEXT_URI, payload_case)
            return res

        else:
            res = self.api_client.post(GENERATE_TEXT_CASE_URI, payload)
            if not res.get("success"):
                raise Exception(res.get("details"))
            return res

def monitor(project, client, session_id=None):
    """Attach tracing wrappers to supported SDK clients.

    :param project: Project instance providing API client and name.
    :param client: SDK client instance to instrument.
    :param session_id: Optional session id to reuse.
    :return: The same client instance with wrapped methods.
    """
    wrapper = Wrapper(project_name=project.project_name, api_client=project.api_client)
    if isinstance(client, OpenAI):
        models = project.models()["model_name"].to_list()
        if "OpenAI" not in models:
            raise Exception("OpenAI Model Not Initialized")
        client.chat.completions.create = wrapper._get_wrapper(
            original_method=client.chat.completions.create,
            method_name="client.chat.completions.create",
            session_id=session_id,
            provider="OpenAI"
        )
        client.responses.create = wrapper._get_wrapper(
            original_method=client.responses.create,
            method_name="client.responses.create",
            session_id=session_id,
            provider="OpenAI"
        )
    elif isinstance(client, Anthropic):
        client.messages.create = wrapper._get_wrapper(
            original_method=client.messages.create,
            method_name="client.messages.create",
            session_id=session_id,
            provider="Anthropic"
        )
    elif isinstance(client, genai.Client):        
        client.models.generate_content = wrapper._get_wrapper(
            original_method=client.models.generate_content,
            method_name="client.models.generate_content",
            session_id=session_id,
            provider="Gemini"
        )
    elif isinstance(client , Groq):
        client.chat.completions.create = wrapper._get_wrapper(
            original_method=client.chat.completions.create,
            method_name="client.chat.completions.create",
            session_id=session_id,
            provider="Groq"
        )
    elif isinstance(client , Together):
        client.chat.completions.create = wrapper._get_wrapper(
            original_method=client.chat.completions.create,
            method_name="client.chat.completions.create",
            session_id=session_id,
            provider="Together"
        )
    elif isinstance(client , InferenceClient):
        client.chat_completion = wrapper._get_wrapper(
            original_method=client.chat_completion,
            method_name="client.chat_completion",
            session_id=session_id,
            provider="HuggingFace"
        )    
    elif isinstance(client, replicate.Client) or client is replicate:        
        client.run = wrapper._get_wrapper(
            original_method=client.run,
            method_name="run",
            session_id=session_id,
            provider="Replicate"
        )
    elif isinstance(client, Mistral):        
        client.chat.complete = wrapper._get_wrapper(
            original_method=client.chat.complete,
            method_name="client.chat.complete",
            session_id=session_id,
            provider="Mistral"
        )
        client.chat.complete_async = wrapper._get_wrapper(
            original_method=client.chat.complete_async,
            method_name="client.chat.complete_async",
            session_id=session_id,
            provider="Mistral"
        )
    elif isinstance(client, botocore.client.BaseClient):
        client.converse = wrapper._get_wrapper(
            original_method=client.converse,
            method_name="client.converse",
            session_id=session_id,
            provider="AWS Bedrock"
        )
    elif isinstance(client, Client):  # XAI Client
    # Wrap the chat.create method to return a wrapped chat object
        original_chat_create = client.chat.create 
        def wrapped_chat_create(*args, **kwargs):
            """Wrap chat creation to instrument returned chat object.

            :param args: Positional args forwarded to chat.create.
            :param kwargs: Keyword args forwarded to chat.create.
            :return: Wrapped chat object with instrumented sample method.
            """
            chat = original_chat_create(*args, **kwargs)
            chat.sample = wrapper._get_wrapper(
                chat=chat,
                original_method=chat.sample,
                method_name="chat.sample",
                session_id=session_id,
                xai_kwargs=kwargs,
                provider="Grok"
            )
            return chat
        
        client.chat.create = wrapped_chat_create

    elif isinstance(client, LexsiModels):
        client.generate_text_case = wrapper._get_wrapper(
            original_method=client.generate_text_case,
            method_name="client.generate_text_case",
            session_id=session_id,
            provider='Lexsi'
        )
    else:
        raise Exception("Not a valid SDK to monitor")
    return client
