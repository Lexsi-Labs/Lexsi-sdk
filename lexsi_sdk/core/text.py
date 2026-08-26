from __future__ import annotations
from datetime import datetime, timedelta
import io
from io import BytesIO
from typing import Optional, List, Dict, Any, Union, Literal

import httpx
from lexsi_sdk.common.types import DedicatedGPUNodeValues, InferenceCompute, InferenceSettings
from pydantic import BaseModel
import plotly.graph_objects as go
import base64
from PIL import Image
from lexsi_sdk.common.types import InferenceCompute, InferenceSettings
from lexsi_sdk.common.utils import fetch_event_metrics, normalize_time, poll_events
from lexsi_sdk.common.xai_uris import (
    AVAILABLE_GUARDRAILS_URI,
    CONFIGURE_GUARDRAILS_URI,
    DELETE_GUARDRAILS_URI,
    FETCH_EVENTS,
    FINETUNE_MODEL_URI,
    GET_AVAILABLE_TEXT_MODELS_URI,
    GET_GUARDRAILS_URI,
    INITIALIZE_TEXT_MODEL_URI,
    LIST_DATA_CONNECTORS,
    MESSAGES_URI,
    MODEL_LOGS_URI,
    PRUNING_MODEL_URI,
    QUANTIZE_MODELS_URI,
    SESSIONS_URI,
    STOP_EVENT_URI,
    TEXT_MODEL_INFERENCE_SETTINGS_URI,
    TRACES_URI,
    UPDATE_ACTIVE_INFERENCE_MODEL_URI,
    UPDATE_GUARDRAILS_STATUS_URI,
    UPLOAD_DATA_FILE_URI,
    UPLOAD_DATA_URI,
    UPLOAD_FILE_DATA_CONNECTORS,
    RUN_CHAT_COMPLETION,
    RUN_IMAGE_GENERATION,
    RUN_CREATE_EMBEDDING,
    RUN_COMPLETION,
    GENERATE_TEXT_CASE_URI,
    GUARDRAILS_APPLY_TO_MODEL,
    GUARDRAILS_RUN,
    GUARDRAILS_REMOVE_FROM_MODEL,
    CATALOG_URI,
    CATALOG_SCORERS_URI,
    CATALOG_ADAPTERS_URI,
    CATALOG_ANNOTATORS_URI,
    CATALOG_GUARD_PROFILES_URI,
    EVALS_RUN_URI,
    EVALS_VALIDATE_URI,
    RUNS_URI,
    RUNS_STATUS_URI,
    RUNS_PREDICTIONS_URI,
    COMPARE_URI,
    BENCHMARKS_URI,
    BENCHMARKS_RUN_URI,
)
from lexsi_sdk.core.project import Project
import pandas as pd

from lexsi_sdk.core.utils import build_list_data_connector_url
import json
from typing import Iterator
from uuid import UUID
from IPython.display import SVG, display

class TextProject(Project):
    """Specialized project abstraction for text and LLM-based workloads. Supports sessions, messages, traces, guardrails, and token-level explainability."""

    def sessions(self) -> pd.DataFrame:
        """Return a DataFrame listing all conversation sessions for this text project.
        Each row corresponds to a single session metadata record.

        :return: a DataFrame containing the conversation session metadata
        """
        res = self.api_client.get(f"{SESSIONS_URI}?project_name={self.project_name}")
        if not res["success"]:
            raise Exception(res.get("details"))

        return pd.DataFrame(res.get("details"))

    def messages(self, session_id: str) -> pd.DataFrame:
        """Return a DataFrame listing all messages in a given session. Requires the session_id.
        Each row corresponds to a single message record.

        :param session_id: UUID of the session
            (e.g., 10f2510c-17dd-4b99-8926-ef4625513a2f).
        :return: a DataFrame containing all messages for the specified session
        """
        res = self.api_client.get(
            f"{MESSAGES_URI}?project_name={self.project_name}&session_id={session_id}"
        )
        if not res["success"]:
            raise Exception(res.get("details"))

        return pd.DataFrame(res.get("details"))

    def traces(self, trace_id: str) -> pd.DataFrame:
        """Retrieve the execution traces for a given trace ID and return them as a DataFrame.
        Each row corresponds to a single trace record.

        :param trace_id: UUID of the trace
            (e.g., 10f2510c-17dd-4b99-8926-ef4625513a2f).
        :return: a DataFrame containing the execution traces for the specified trace ID
        """
        res = self.api_client.get(
            f"{TRACES_URI}?project_name={self.project_name}&trace_id={trace_id}"
        )
        if not res["success"]:
            raise Exception(res.get("details"))

        return pd.DataFrame(res.get("details"))

    def guardrails(self) -> pd.DataFrame:
        """List all guardrails currently configured for the project.
        Returns a DataFrame describing each guardrail and its configuration.

        :return: a DataFrame containing the configured guardrails and their details
        """
        res = self.api_client.get(
            f"{GET_GUARDRAILS_URI}?project_name={self.project_name}"
        )
        if not res["success"]:
            raise Exception(res.get("details"))

        return pd.DataFrame(res.get("details"))

    def update_guardrail_status(self, group_id: str, status: Literal["active", "inactive"]) -> str:
        """Update the status (active or inactive) of a specified guardrail.
        Requires the guardrail_id and a boolean status value.

        :param group_id: ID of the guardrail group
        :param status: Status of the guardrail ("active" or "inactive")
        :return: a response indicating the result of the update operation
        """

        payload = {
            "group_id": group_id,
            "status": status,
        }
        if self.organization_id:
            payload["organization_id"] = self.organization_id
        res = self.api_client.post(UPDATE_GUARDRAILS_STATUS_URI, payload=payload)
        if not res["success"]:
            raise Exception(res.get("details"))

        return res.get("details")

    def delete_guardrail(self, guardrail_id: str) -> str:
        """Delete a guardrail from the project using its ID.
        Returns the API response message.

        :param guardrail_id: ID of the guardrail
        :return: a response indicating the result of the delete operation
        """
        payload = {
            "project_name": self.project_name,
            "guardrail_id": guardrail_id,
        }
        res = self.api_client.post(DELETE_GUARDRAILS_URI, payload=payload)
        if not res["success"]:
            raise Exception(res.get("details"))

        return res.get("details")

    def available_guardrails(self) -> pd.DataFrame:
        """Return a DataFrame of all guardrails available to configure in this project.
        Each row describes a single guardrail type.

        :return: a DataFrame containing all available guardrail types
        """
        res = self.api_client.get(AVAILABLE_GUARDRAILS_URI)
        if not res["success"]:
            raise Exception(res.get("details"))

        return pd.DataFrame(res.get("details"))

    def configure_guardrail(
        self,
        guardrail_name: str,
        guardrail_config: dict,
        model_name: str,
        apply_on: str,
    ) -> dict:
        """Configure a new guardrail in the project.
        Requires the guardrail name, a configuration dictionary, the model name, and where to apply it (input or output).

        :param guardrail_name: Name of the guardrail.
        :param guardrail_config: Configuration dictionary for the guardrail.
        :param model_name: Name of the model to which the guardrail applies.
        :param apply_on: Specifies when to apply the guardrail ("input" or "output").
        :return: The response details from the API.
        """
        payload = {
            "name": guardrail_name,
            "config": guardrail_config,
            "model_name": model_name,
            "apply_on": apply_on,
            "project_name": self.project_name,
        }
        res = self.api_client.post(CONFIGURE_GUARDRAILS_URI, payload)
        if not res["success"]:
            raise Exception(res.get("details"))

        return res.get("details")

    def initialize_text_model(
        self, 
        model_provider: str, 
        model_name: str, 
        model_task_type:str, 
        model_architecture: str,  
        inference_compute: Optional[InferenceCompute] = None,
        inference_settings: Optional[InferenceSettings] = None,
        assets: Optional[dict] = None,
        requirements_file: Optional[str] = None,
        app_file: Optional[str] = None
    ) -> str:
        """Initialize a text model for the project, specifying the model provider, model name, task type, model type (classification/regression), inference compute settings, inference settings, and optional assets. Polls for completion and returns when done.

        :param model_provider: model provider name for initialization
            **Model Providers**
            - ``Hugging Face``
            - ``OpenAI``
            - ``Anthropic``
            - ``Groq``
            - ``Grok``
            - ``Gemini``
            - ``Together``
            - ``Replicate``
            - ``Mistral``
            - ``AWS Bedrock``
            - ``Open Router``

        :param model_name: name of the model to be initialized
            (e.g., meta-llama/Llama-3.2-1B-Instruct).

        :param model_task_type: task type of model
            **Model Task Types**
            - ``question-answering``
            - ``summarization``
            - ``text-classification``
            - ``text-generation``
            - ``text2text-generation``
            - ``token-classification``

        :param model_architecture: architecture of the model to be initialized
            **Model Architecture**
            - ``bert``
            - ``llm``

        :param inference_compute: inference compute configuration used to run the model during inference
            (e.g., CPU/GPU type, memory, replicas, and other hardware or scaling settings).
            Required for the Hugging Face provider models, not required for other providers
        :type inference_compute: InferenceCompute | None

        :param inference_settings: inference runtime settings.
            Required for the Hugging Face provider models, not required for other providers
        :type inference_settings: InferenceSettings | None

        :param assets: assets required for the model, including provider credentials, access tokens,
            or other secrets needed at runtime
            (e.g., {"HF_TOKEN":"hf_njbjkfdsnjfkdnskbfk"}).

        :param requirements_file: file path for the requirements file
            a YAML file defining the runtime environment, including base Docker image,
            system-level dependencies, and Python packages required for model deployment.
            Not required for the transformers serverless inference engine.

            Example::
                image: nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04
                system_packages:
                    - build-essential
                python_packages:
                    - fastapi>=0.115.5
                    - uvicorn>=0.30.6
                    - transformers==4.52.3
                    - pydantic>=2.9.2
                    - torch==2.7.0
                    - accelerate==1.8.1

        :param app_file: file path for the app file
            a Python application file that implements the model inference logic,
            including how inputs are processed and how predictions are generated and returned

        :return: response
        """
        data = {
            "model_provider": model_provider,
            "model_name": model_name,
            "model_task_type": model_task_type,
            "project_name": self.project_name,
            "model_type": model_architecture,
            "inference_compute": inference_compute,
            "inference_settings": inference_settings,
            "assets": assets
        }
        if inference_compute:
            if inference_compute.get("custom_server_config", {}):
                server_config = inference_compute.get("custom_server_config", {})
                server_config["start"] = normalize_time(server_config.get("start"))
                server_config["stop"] = normalize_time(server_config.get("stop"))
                if server_config["start"] and not server_config["stop"]:
                    raise ValueError("If start is provided, stop cannot be None.")

                if server_config["stop"] and not server_config["start"]:
                    raise ValueError("If stop is provided, start cannot be None.")

                if server_config["start"] and server_config["stop"]:
                    start_dt = datetime.fromisoformat(server_config["start"])
                    stop_dt = datetime.fromisoformat(server_config["stop"])

                    if stop_dt - start_dt < timedelta(minutes=15):
                        raise ValueError("Stop time must be at least 15 minutes greater than start time.")

                    if not server_config.get("op_hours") and server_config.get("auto_start"):
                        server_config["op_hours"] = True
            data["inference_compute"] = {**inference_compute, "instance_type": inference_compute.get("compute_type")}

        payload ={
            "data": (None,json.dumps(data)),
        }
        if requirements_file:
            payload["requirements_file"] = ("requirements.yaml", open(requirements_file, "rb"))
        if app_file:
            payload["app_file"] = ("app.py", open(app_file, "rb"))
            
        res = self.api_client.file(f"{INITIALIZE_TEXT_MODEL_URI}", payload)
        if not res.get("success"):
            raise Exception(res.get("details", "Model Initialization Failed"))
        poll_events(self.api_client, self.project_name, res["event_id"])

    def model_inference_settings(
        self,
        model_name: str,
        inference_compute: InferenceCompute,
        inference_settings: Optional[InferenceSettings] = None,
        requirements_file: Optional[str] = None,
        app_file: Optional[str] = None
    ) -> str:
        """Configure inference compute and runtime settings for a model.
         Only for Hugging Face provider models

        :param model_name: Name of the model for inference settings update
            (e.g., meta-llama/Llama-3.2-1B-Instruct).

        :param inference_compute: Inference compute configuration for the model.
        :type inference_compute: InferenceCompute | None

        :param inference_settings: Inference runtime settings for the model.
        :type inference_settings: InferenceSettings | None

        :param requirements_file: file path for the requirements file
            a YAML file defining the runtime environment, including base Docker image,
            system-level dependencies, and Python packages required for model deployment.
            Not required for the transformers serverless inference engine.
            Requirements file is needed when updating inference settings for an existing model.

            Example::
                image: nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04
                system_packages:
                    - build-essential
                python_packages:
                    - fastapi>=0.115.5
                    - uvicorn>=0.30.6
                    - transformers==4.52.3
                    - pydantic>=2.9.2
                    - torch==2.7.0
                    - accelerate==1.8.1

        :param app_file: file path for the app file
            a Python application file that implements the model inference logic,
            including how inputs are processed and how predictions are generated and returned

        :return: a response indicating the result of the inference settings configuration
        """
        server_config = inference_compute.get("custom_server_config", {})
        server_config["start"] = normalize_time(server_config.get("start"))
        server_config["stop"] = normalize_time(server_config.get("stop"))
        if server_config["start"] and not server_config["stop"]:
            raise ValueError("If start is provided, stop cannot be None.")

        if server_config["stop"] and not server_config["start"]:
            raise ValueError("If stop is provided, start cannot be None.")

        if server_config["start"] and server_config["stop"]:
            start_dt = datetime.fromisoformat(server_config["start"])
            stop_dt = datetime.fromisoformat(server_config["stop"])

            if stop_dt - start_dt < timedelta(minutes=15):
                raise ValueError("Stop time must be at least 15 minutes greater than start time.")

            if not server_config.get("op_hours") and server_config.get("auto_start"):
                server_config["op_hours"] = True

        data = {
            "model_name": model_name,
            "project_name": self.project_name,
            "inference_compute": {**inference_compute, "instance_type": inference_compute.get("compute_type")},
            "inference_settings": inference_settings,
        }
        payload ={
            "data": (None,json.dumps(data)),
        }
        if requirements_file:
            payload["requirements_file"] = ("requirements.yaml", open(requirements_file, "rb"))
        if app_file:
            payload["app_file"] = ("app.py", open(app_file, "rb"))
            
        res = self.api_client.file(f"{TEXT_MODEL_INFERENCE_SETTINGS_URI}", payload)
        if not res.get("success"):
            raise Exception(res.get("details", "Failed to update inference settings"))
        
        return res.get("details", "Inference Settings Updated")


    def upload_data(
        self,
        data: str | pd.DataFrame,
        tag: str,
    ) -> str:
        """Upload text data to the project by specifying either a file path or a pandas DataFrame and a tag.
        Handles conversion to CSV for DataFrame uploads and returns the API response.

        :param data: File path or pandas DataFrame containing the rows to upload
        :param tag: Tag to associate with the uploaded data
        :return: a response containing the server’s upload result
        """

        def build_upload_data(data):
            """Prepare file payload from path or DataFrame.

            :param data: File path or DataFrame to convert.
            :return: Tuple or file handle suitable for multipart upload.
            """
            if isinstance(data, str):
                file = open(data, "rb")
                return file
            elif isinstance(data, pd.DataFrame):
                csv_buffer = io.BytesIO()
                data.to_csv(csv_buffer, index=False, encoding="utf-8")
                csv_buffer.seek(0)
                file_name = f"{tag}_sdk_{datetime.now().replace(microsecond=0)}.csv"
                file = (file_name, csv_buffer.getvalue())
                return file
            else:
                raise Exception("Invalid Data Type")

        def upload_file_and_return_path(data, data_type, tag=None) -> str:
            """Upload a file and return the stored path.

            :param data: Data payload (path or DataFrame).
            :param data_type: Type of data being uploaded.
            :param tag: Optional tag.
            :return: File path stored on the server.
            """
            files = {"in_file": build_upload_data(data)}
            res = self.api_client.file(
                f"{UPLOAD_DATA_FILE_URI}?project_name={self.project_name}&data_type={data_type}&tag={tag}",
                files,
            )

            if not res["success"]:
                raise Exception(res.get("details"))
            uploaded_path = res.get("metadata").get("filepath")

            return uploaded_path

        uploaded_path = upload_file_and_return_path(data, "data", tag)

        payload = {
            "path": uploaded_path,
            "tag": tag,
            "type": "data",
            "project_name": self.project_name,
        }
        res = self.api_client.post(UPLOAD_DATA_URI, payload)

        if not res["success"]:
            self.delete_file(uploaded_path)
            raise Exception(res.get("details"))

        return res.get("details")

    def upload_data_dataconnectors(
        self,
        data_connector_name: str,
        tag: str,
        bucket_name: Optional[str] = None,
        file_path: Optional[str] = None,
        dataset_name: Optional[str] = None,
    ) ->  str:
        """Upload text data stored in a configured data connector (such as S3 or GCS).
        Requires the connector name, a tag, and optionally the bucket name and file path.
        Returns the API response.

        :param data_connector_name: Name of the configured data connector
        :param tag: Tag to associate with the uploaded data
        :param bucket_name: Name of the bucket or storage location, if required by the connector
        :param file_path: File path within the connector storage
        :param dataset_name: Optional dataset name to persist the uploaded data
        :return: a response containing the server’s upload result
        """

        def get_connector() -> str | pd.DataFrame:
            """Fetch connector metadata for the requested link service.

            :return: DataFrame of connector info or error string.
            """
            url = build_list_data_connector_url(
                LIST_DATA_CONNECTORS, self.project_name, self.organization_id
            )
            res = self.api_client.post(url)

            if res["success"]:
                df = pd.DataFrame(res["details"])
                filtered_df = df.loc[df["link_service_name"] == data_connector_name]
                if filtered_df.empty:
                    return "No data connector found"
                return filtered_df

            return res["details"]

        connectors = get_connector()
        if isinstance(connectors, pd.DataFrame):
            value = connectors.loc[
                connectors["link_service_name"] == data_connector_name,
                "link_service_type",
            ].values[0]
            ds_type = value

            if ds_type == "s3" or ds_type == "gcs":
                if not bucket_name:
                    return "Missing argument bucket_name"
                if not file_path:
                    return "Missing argument file_path"
        else:
            return connectors

        def upload_file_and_return_path(file_path, data_type, tag=None) -> str:
            """Upload a file from connector storage and return stored path.

            :param file_path: Path within the connector store.
            :param data_type: Type of data being uploaded.
            :param tag: Optional tag for the upload.
            :return: Stored file path returned by the API.
            """
            if not self.project_name:
                return "Missing Project Name"
            query_params = f"project_name={self.project_name}&link_service_name={data_connector_name}&data_type={data_type}&tag={tag}&bucket_name={bucket_name}&file_path={file_path}&dataset_name={dataset_name}"
            if self.organization_id:
                query_params += f"&organization_id={self.organization_id}"
            res = self.api_client.post(f"{UPLOAD_FILE_DATA_CONNECTORS}?{query_params}")
            if not res["success"]:
                raise Exception(res.get("details"))
            uploaded_path = res.get("metadata").get("filepath")

            return uploaded_path

        uploaded_path = upload_file_and_return_path(file_path, "data", tag)

        payload = {
            "path": uploaded_path,
            "tag": tag,
            "type": "data",
            "project_name": self.project_name,
        }
        res = self.api_client.post(UPLOAD_DATA_URI, payload)

        if not res["success"]:
            self.delete_file(uploaded_path)
            raise Exception(res.get("details"))

        return res.get("details")

    def quantize_model(
        self,
        model_name: str,
        quant_name: str,
        quantization_type: str,
        qbit: int,
        node: str,
        tag: Optional[str] = None,
        input_column: Optional[str] = None,
        no_of_samples: Optional[int] = None,
        assets: Optional[dict] = None,
        max_seq_len: Optional[int] = 512,
        target_layers: Optional[List[str]] = ["Linear"],
        ignore_layers: Optional[List[str]] = ["lm_head"],
        scheme: Optional[str] = None,
        torch_compile: Optional[bool] = False,
        batch_size: Optional[int] = 2,
        apply_kvcache_quant: Optional[bool] = False,
        smoothing_strength: Optional[float] = 0.8
    ):
        """Quantize a trained model to reduce its size and improve inference efficiency.
        Requires the model name, quantization method, quantization type,number of bits, and compute instance type. 
        Optional parameters allow specifying a tag,input column, and number of samples used during quantization.

        :param model_name: Name of the base model to be quantized
        :param quant_name: Name of the quantization method to use
            **Quantization Methods**
            - ``quanto``
            - ``bnb``
            - ``hqq``
            - ``torchao``
            - ``gptq``
            - ``awq``
            - ``llmcomp-awq``
            - ``llmcomp-gptq``
            - ``llmcomp-simple``
            - ``llmcomp-smoothquant``
        :param quantization_type: Type of quantization to apply
            **Quantization Types**
            - ``static``
            - ``dynamic``
        :param qbit: Number of bits to use for quantization
            **Quantization Bits**
            - ``4``
            - ``8``
        :param node: Node name used for performing quantization
        :param tag: Optional tag name to associate with the quantized model
        :param input_column: Optional input column used from the dataset for quantization
        :param no_of_samples: Optional number of samples to use for quantization
        :return: a response indicating the result of the quantization operation
        """
        payload = {
            "project_name": self.project_name,
            "model_name": model_name,
            "quant_name": quant_name,
            "quantization_type": quantization_type,
            "qbit": qbit,
            "instance_type": node,
            "tag": tag,
            "assets": assets,
            "max_seq_len": max_seq_len,
            "target_layers": target_layers,
            "ignore_layers": ignore_layers,
            "input_column": input_column,
            "no_of_samples": no_of_samples,
            "scheme": scheme,
            "torch_compile": torch_compile,
            "batch_size": batch_size,
            "apply_kvcache_quant": apply_kvcache_quant,
            "smoothing_strength": smoothing_strength
        }

        res = self.api_client.post(QUANTIZE_MODELS_URI, payload)
        if not res["success"]:
            raise Exception(res.get("details"))

        poll_events(self.api_client, self.project_name, res.get("event_id"))

    def chat_completion(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        provider: str,
        api_key: Optional[str] = None,
        session_id: Optional[UUID] = None,
        max_tokens: Optional[int] = None,
        max_completion_tokens: Optional[int] = None,
        stream: Optional[bool] = False,
    ) -> Union[dict, Iterator[str]]:
        """Generate a chat completion using an OpenAI-compliant interface.

        :param model: Name of the model to use for generating the chat completion
        :param messages: List of chat messages, where each message contains a role and content
        :param provider: Model provider (e.g., "OpenAI", "Anthropic")
        :param api_key: API key for the selected provider, if required
        :param session_id: Session ID associated with this chat completion, if provided
        :param max_tokens: Maximum number of tokens to generate
        :param max_completion_tokens: Maximum number of tokens to generate for the completion
        :param stream: Whether to stream the response
        :return: a chat completion response dictionary or a streaming iterator of response chunks
        """
        payload = {
            "model": model,
            "messages": messages,
            "stream": stream,
            "project_name": self.project_name,
            "provider": provider,
            "api_key": api_key,
            "session_id": session_id,
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if max_completion_tokens is not None:
            payload["max_completion_tokens"] = max_completion_tokens

        if not stream:
            return self.api_client.post(RUN_CHAT_COMPLETION, payload=payload)

        return self.api_client.stream(
            uri=RUN_CHAT_COMPLETION, method="POST", payload=payload
        )

    def create_embeddings(
        self,
        input: Union[str, List[str]],
        model: str,
        provider: str,
        api_key : Optional[str] = None,
        session_id : Optional[UUID] = None,
    ) -> dict:  
        """Create embeddings using an OpenAI-compliant embeddings interface.

        :param input: Input text or list of text strings to generate embeddings for
        :param model: Name of the model to use for generating embeddings
        :param provider: Model provider (e.g., "OpenAI", "Anthropic")
        :param api_key: API key for the selected provider, if required
        :param session_id: Session ID associated with this embeddings request, if provided
        :return: a dictionary containing the embeddings response
        """
        payload = {
            "model": model,
            "input": input,
            "project_name": self.project_name,
            "provider": provider,
            "api_key": api_key,
            "session_id": session_id,
        }

        res = self.api_client.post(RUN_CREATE_EMBEDDING, payload=payload)
        return res

    def completion(
        self,
        model: str,
        prompt: str,
        provider: str,
        api_key: Optional[str] = None,
        session_id: Optional[UUID] = None,
        max_tokens: Optional[int] = None,
        max_completion_tokens: Optional[int] = None,
        stream: Optional[bool] = False,
    ) -> dict:
        """Generate a text completion using an OpenAI-compliant interface.

        :param model: Name of the model to use for generating the completion
        :param prompt: Input prompt to be provided to the model
        :param provider: Model provider (e.g., "OpenAI", "Anthropic")
        :param api_key: API key for the selected provider, if required
        :param session_id: Session ID associated with this completion request, if provided
        :param max_tokens: Maximum number of tokens to generate
        :param stream: Whether to stream the response
        :return: a completion response dictionary or a streaming iterator of response chunks
        """

        payload = {
            "model": model,
            "prompt": prompt,
            "stream": stream,
            "project_name": self.project_name,
            "provider": provider,
            "api_key": api_key,
            "session_id": session_id,
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if max_completion_tokens is not None:
            payload["max_completion_tokens"] = max_completion_tokens
        if not stream:
            return self.api_client.post(RUN_COMPLETION, payload=payload)

        return self.api_client.stream(
            uri=RUN_COMPLETION, method="POST", payload=payload
        )

    def image_generation(
        self,
        model: str,
        prompt: str,
        provider: str,
        api_key: Optional[str] = None,
        session_id : Optional[UUID] = None,
    ) -> dict:
        """Generate images using an OpenAI-compliant image generation interface.

        :param model: Name of the model to use for image generation
        :param prompt: Text prompt describing the image to generate
        :param provider: Model provider (e.g., "OpenAI", "Anthropic")
        :param api_key: API key for the selected provider, if required
        :param session_id: Session ID associated with this image generation request, if provided
        :return: a dictionary containing the image generation response
        """

        payload = {
            "model": model,
            "prompt": prompt,
            "project_name": self.project_name,
            "provider": provider,
            "api_key": api_key,
            "session_id": session_id,
        }

        res = self.api_client.post(RUN_IMAGE_GENERATION, payload=payload)

        return res
    
    def text_generation(
        self,
        model: str,
        prompt: str,
        session_id: Optional[UUID] = None,
        min_tokens: int = 100,
        max_tokens: int = 1024,
        explain_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = 1.0,
        top_k: Optional[int] = 0,
    ) -> dict :
        """Generate an explainable text case using a hosted Lexsi model.

        :param model: Name of the deployed text model.
        :param prompt: Input prompt for generation.
        :param XAI_pods: Dedicated instance type (if applicable).
        :param XAI: Whether to explain the model behavior.
        :param session_id: Optional existing session id.
        :param min_tokens: Minimum tokens to generate.
        :param max_tokens: Maximum tokens to generate.
        :param explain_tokens: Number of tokens to explain.
        :param temperature: Temperature for text generation.
        :param top_p: Top-p sampling parameter.
        :param top_k: Top-k sampling parameter.
        :return: API response with generation details.
        """
        payload = {
            "session_id": session_id,
            "project_name": self.project_name,
            "model": model,
            "prompt": prompt,
            "provider" : "Lexsi",
            "max_tokens": max_tokens,
            "min_tokens": min_tokens,
            "explain_tokens": explain_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
        }
    
        res = self.api_client.post(GENERATE_TEXT_CASE_URI, payload)
        if not res.get("success"):
            raise Exception(res.get("details"))
        return res
    
    def update_inference_model_status(self, model_name: str, activate: bool) -> str:
        """Sets the provided model to active for inferencing

        :param model_name: name of the model
        :param activate: Boolean flag to control the model inference state.
                     - True: Activates the model inference.
                     - False: Deactivates the model inference.
        :return: response
        """
        payload = {
            "project_name": self.project_name,
            "model_name": model_name,
            "activate": activate,
        }

        res = self.api_client.post(UPDATE_ACTIVE_INFERENCE_MODEL_URI, payload)

        if not res["success"]:
            raise Exception(res["details"])

        return res.get("details")

    def run_guardrails(self, guardrails: List[Dict[str, Any]], input_data: str) -> dict:
        """Run the provided guardrail flows against the given input.

        :param guardrails: Guardrail flow definitions to execute.
        :param input_data: Input text to validate against the guardrails.
        :return: The API response data.
        """
        payload = {"guardrails": guardrails, "input_data": input_data}
        res = self.api_client.post(GUARDRAILS_RUN, payload=payload)
        if not res["success"]:
            raise Exception(res["details"])
        return dict(res["data"])
    
    def apply_guardrail_to_models(
        self,
        group_id: str,
        model_name: str,
        apply_on: str = "input",
        retry: bool = False,
        retry_attempts: int = 1,
    ) -> dict:
        """Apply an existing organization guardrail to a model in a project.

        :param group_id: Identifier of the guardrail group to apply.
        :param model_name: Name of the model to apply the guardrail to.
        :param apply_on: Whether to apply the guardrail on "input" or "output".
        :param retry: Whether to retry on failure.
        :param retry_attempts: Number of retry attempts.
        :return: The API response details.
        """
        payload = {
            "project_name": self.project_name,
            "group_id": group_id,
            "model_name": model_name,
            "apply_on": apply_on,
            "retry": retry,
            "retry_attempts": retry_attempts,
        }
        if self.organization_id:
            payload["organization_id"] = self.organization_id
        res = self.api_client.post(GUARDRAILS_APPLY_TO_MODEL, payload=payload)
        if not res["success"]:
            raise Exception(res["details"])
        return dict(res["details"])
    

    def finetune_model(
        self,
        model_name: str,
        node: DedicatedGPUNodeValues,
        assets: Optional[dict] = None,
        config: Optional[dict] = None,
        plot: bool = True,
    ) -> str:
        """Fine-tune a model using the provided training data and settings.

        :param model_name: Name of the new fine-tuned model.
        :param node: Dedicated GPU node configuration for fine-tuning.
        :param assets: Assets required for fine-tuning, such as hugging face secrets.
        :param config: Configuration settings for fine-tuning, including hyperparameters, training settings,
        :param plot: When True (default), live model/system metrics are plotted in
            notebook environments; when False, raw metric summaries are printed
            instead even where plotting is possible.
        :return: response with fine-tuning details.
        """
        
        payload  = {
            "project_name": self.project_name,
            "model_name": model_name,
            "assets": assets,
            "config": config,
            "compute": {
                "node": node
            }
        }
        res = self.api_client.post(FINETUNE_MODEL_URI, payload)

        if not res["success"]:
            raise Exception(res.get("details", "Model Fine-tuning Failed"))

        poll_events(self.api_client, self.project_name, res["event_id"], plot=plot)

    def stop_model_training(self, model_name: str) -> str:
        """Stop a running model-training job (fine-tuning, quantization, etc.).

        Locates the in-progress training/processing event for ``model_name`` (the
        generated model name, e.g. ``meta-llama-Llama-3.2-1B-rl-dpo_v3``, as
        listed by the model list / :meth:`model_metrics`), terminates the GPU
        server running it, and marks the job as terminated. Requires
        admin/manager access to the project.

        :param model_name: Name of the model whose training job to stop.
        :return: Confirmation message from the API.
        """
        res = self.api_client.post(
            FETCH_EVENTS,
            {
                "project_name": self.project_name,
                "task_name": ["fine_tune_model", "quantize_model", "prune_model"],
            },
        )
        if not res["success"]:
            raise Exception(res.get("details", "Failed to fetch training events"))

        event_id = next(
            (
                event.get("_id")
                for event in (res.get("details") or [])
                if (event.get("config") or {}).get("model_name") == model_name
                or (event.get("params") or {}).get("model_name") == model_name
            ),
            None,
        )
        if not event_id:
            raise Exception(f"No training job found for model '{model_name}'")

        res = self.api_client.post(STOP_EVENT_URI, payload={"event_id": event_id})
        if not res["success"]:
            raise Exception(res.get("details", "Failed to stop the training job"))
        return res.get("details")

    def remove_guardrail_from_model(self, model_name: str, apply_on: str = "input") -> str:
        """Remove a guardrail from a specific model.

        :param model_name: Name of the model to remove the guardrail from.
        :param apply_on: Whether the guardrail is applied on 'input' or 'output'.
        :return: Confirmation message from the API.
        """
        payload = {
            "project_name": self.project_name,
            "model_name": model_name,
            "apply_on": apply_on,
        }
        res = self.api_client.post(GUARDRAILS_REMOVE_FROM_MODEL, payload=payload)
        if not res["success"]:
            raise Exception(res.get("details", "Failed to remove guardrail from model"))
        return dict(res["details"])
    
    def model_logs(self, model_name: str, return_logs: Optional[bool] = False) -> str | None:
        """Fetch and return logs for a specific finetuned and quantized model.

        :param model_name: Name of the model to retrieve logs for.
        :param return_logs: Whether to return the logs as a string. If False, logs will be printed line by line. If True, logs will be returned as a single string.
        :return: Logs data as a string or None.
        """
        res = self.api_client.get(f"{MODEL_LOGS_URI}?project_name={self.project_name}&model_name={model_name}")
        if not res["success"]:
            raise Exception(res.get("details", "Failed to fetch model logs"))
        logs = res.get("details", "No logs found for the model").get("logs", "")
        if return_logs:
            return logs
        for line in logs.split("\n"):
            print(line)

    def prune_model(
        self,
        model_name: str,
        node: DedicatedGPUNodeValues,
        assets: Optional[dict] = None,
        config: Optional[dict] = None,
    ):
        payload  = {
            "project_name": self.project_name,
            "model_name": model_name,
            "assets": assets,
            "instance_type": node
        }
        if config:
            payload.update({k: v for k, v in config.items() if v is not None})
        res = self.api_client.post(PRUNING_MODEL_URI, payload)
        
        if not res["success"]:
            raise Exception(res.get("details", "Model Fine-tuning Failed"))
        
        poll_events(self.api_client, self.project_name, res["event_id"])
        
        
    def model_metrics(
        self,
        model_name: str,
        return_metrics: Optional[bool] = False,
        plot: bool = True,
    ) -> dict | None:
        """Fetch and plot the system and model metrics for a finetuned model.

        Mirrors :meth:`model_logs` but for metrics: it locates the finetuning
        event for ``model_name`` and replays its metrics from the ``events/poll``
        stream (the same source used for live metrics during training), then
        renders the system and model metric charts — one subplot per metric — in
        a notebook, or prints the final values otherwise.

        :param model_name: Name of the finetuned model to retrieve metrics for.
        :param return_metrics: When True, return the metrics dict for
            programmatic use. Defaults to False so notebooks show only the charts
            instead of also echoing the raw series.
        :param plot: When True (default), metrics are plotted in notebook
            environments; when False, raw metric summaries are printed instead
            even where plotting is possible.
        :return: dict with ``model_metrics`` and ``system_metrics`` series when
            ``return_metrics`` is True, otherwise None.
        """
        res = self.api_client.post(
            FETCH_EVENTS,
            {"project_name": self.project_name, "task_name": ["fine_tune_model"]},
        )
        if not res["success"]:
            raise Exception(res.get("details", "Failed to fetch finetuning events"))

        # Events come back newest-first; match the finetuning event for this model.
        event_id = next(
            (
                event.get("_id")
                for event in (res.get("details") or [])
                if (event.get("config") or {}).get("model_name") == model_name
                or (event.get("params") or {}).get("model_name") == model_name
            ),
            None,
        )
        if not event_id:
            raise Exception(f"No finetuning event found for model '{model_name}'")

        metrics = fetch_event_metrics(self.api_client, self.project_name, event_id, plot=plot)
        if return_metrics:
            return metrics

    def get_full_catalog(self) -> pd.DataFrame:
        """Return the full catalog of scorers, adapters, annotators, and guard profiles.

        :return: a DataFrame containing all catalog items with a 'type' column
        """
        res = self.api_client.get(CATALOG_URI)
        if not res["success"]:
            raise Exception(res.get("details"))
        details = res.get("details", {})
        dfs = [
            pd.DataFrame(items).assign(type=cat_type)
            for cat_type, items in details.items()
            if isinstance(items, list) and items
        ]
        return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

    def get_scorers(self) -> pd.DataFrame:
        """Return a DataFrame listing all registered scorers.

        :return: a DataFrame containing scorer details
        """
        res = self.api_client.get(CATALOG_SCORERS_URI)
        if not res["success"]:
            raise Exception(res.get("details"))
        return pd.DataFrame(res.get("details"))

    def get_adapters(self) -> pd.DataFrame:
        """Return a DataFrame listing all registered adapters.

        :return: a DataFrame containing adapter details
        """
        res = self.api_client.get(CATALOG_ADAPTERS_URI)
        if not res["success"]:
            raise Exception(res.get("details"))
        return pd.DataFrame(res.get("details"))

    def get_annotators(self) -> pd.DataFrame:
        """Return a DataFrame listing all registered annotators.

        :return: a DataFrame containing annotator details
        """
        res = self.api_client.get(CATALOG_ANNOTATORS_URI)
        if not res["success"]:
            raise Exception(res.get("details"))
        return pd.DataFrame(res.get("details"))

    def get_guard_profiles(self) -> pd.DataFrame:
        """Return a DataFrame listing all available guard profiles.

        :return: a DataFrame containing guard profile details
        """
        res = self.api_client.get(CATALOG_GUARD_PROFILES_URI)
        if not res["success"]:
            raise Exception(res.get("details"))
        return pd.DataFrame(res.get("details"))

    def run_eval(
        self,
        config: dict,
        pod: str,
        samples: Optional[List[Dict[str, Any]]] = None,
        tag: Optional[str] = None,
        input_column: Optional[str] = None,
        target_column: Optional[str] = None,
        actual_output_column: Optional[str] = None,
        choices_column: Optional[str] = None,
        retrieval_context_column: Optional[str] = None,
        task_column: Optional[str] = None
    ) -> dict:
        """Start an evaluation run. Returns an event_id for polling.

        :param config: Evaluation configuration dictionary
        :param pod: The pod type for evaluation
        :param samples: Optional list of sample dictionaries for evaluation
        :return: API response with event_id for polling
        """
        payload = {**config, "compute": {"pod": pod}, "project_name": self.project_name}
        if samples:
            payload["samples"] = samples
        if tag:
            payload["tag"] = tag
        if input_column:
            payload["input_column"] = input_column
        if target_column:
            payload["target_column"] = target_column
        if actual_output_column:
            payload["actual_output_column"] = actual_output_column
        if choices_column:
            payload["choices_column"] = choices_column
        if retrieval_context_column:
            payload["retrieval_context_column"] = retrieval_context_column
        if task_column:
            payload["task_column"] = task_column

        res = self.api_client.post(EVALS_RUN_URI, payload=payload)
        if not res["success"]:
            raise Exception(res.get("details"))
        if res.get("details", {}).get("existing_run", False):
            return res
        poll_events(
            api_client=self.api_client,
            project_name=self.project_name,
            event_id=res.get("details", {}).get("event_id"),
        )
        return {"success": True, "details": "Evaluation Completed Successfully"}

    def run_benchmark(
        self,
        config: dict,
        node: DedicatedGPUNodeValues,
        model_name: str,
    ) -> dict:
        """Start a benchmark run using lm-eval. Returns an event_id for polling.

        :param task: Benchmark task name (e.g., "ARC-Challenge", "GSM8K", "MMLU")
        :param model_spec: Model specification dictionary for the benchmark
        :return: API response with event_id for polling
        """
        payload = {
            **config,
            "model_name": model_name,
            "project_name": self.project_name,
            "compute": {
                "node": node
            }
        }
        res = self.api_client.post(BENCHMARKS_RUN_URI, payload=payload)
        if not res["success"]:
            raise Exception(res.get("details"))
        poll_events(
            api_client=self.api_client,
            project_name=self.project_name,
            event_id=res.get("details", {}).get("event_id"),
        )
        return {"success": True, "details": "Benchmarking Completed Successfully"}
       

    def validate_eval_config(
        self,
        config: dict,
        samples: Optional[list] = [],
        tag: Optional[str] = None,
        input_column: Optional[str] = None,
        target_column: Optional[str] = None,
        actual_output_column: Optional[str] = None,
        choices_column: Optional[str] = None,
        retrieval_context_column: Optional[str] = None,
        task_column: Optional[str] = None
    ) -> dict:
        """Validate an evaluation configuration without running it.

        :param config: Evaluation configuration dictionary to validate
        :param samples: List of sample dictionaries for validation
        :param tag: Optional tag to associate with the evaluation
        :param input_column: Optional input column name from the dataset
        :param target_column: Optional target column name from the dataset
        :param actual_output_column: Optional actual output column name from the dataset
        :param choices_column: Optional choices column name from the dataset
        :param retrieval_context_column: Optional retrieval context column name from the dataset
        :param task_column: Optional task column name from the dataset
        :return: validation results including warnings
        """
        payload = {**config, "project_name": self.project_name, "samples": samples}
        if tag:
            payload["tag"] = tag
        if input_column:
            payload["input_column"] = input_column
        if target_column:
            payload["target_column"] = target_column
        if actual_output_column:
            payload["actual_output_column"] = actual_output_column
        if choices_column:
            payload["choices_column"] = choices_column
        if retrieval_context_column:
            payload["retrieval_context_column"] = retrieval_context_column
        if task_column:
            payload["task_column"] = task_column
        res = self.api_client.post(EVALS_VALIDATE_URI, payload=payload)
        if not res["success"]:
            raise Exception(res.get("details"))
        return res.get("details")

    def list_runs(self) -> pd.DataFrame:
        """Return a DataFrame listing all evaluation runs for this project.

        :return: a DataFrame containing run summaries
        """
        res = self.api_client.get(f"{RUNS_URI}?project_name={self.project_name}")
        if not res["success"]:
            raise Exception(res.get("details"))
        return pd.DataFrame(res.get("details"))

    def get_run_status(self, event_id: str) -> dict:
        """Poll the status of a long-running evaluation job.

        :param event_id: The event ID returned by run_eval
        :return: status information including progress and logs
        """
        res = self.api_client.get(f"{RUNS_STATUS_URI}/{event_id}")
        if not res["success"]:
            raise Exception(res.get("details"))
        return res.get("details")

    def get_run(self, run_id: str) -> dict:
        """Return the full RunResult for a specific evaluation run.

        :param run_id: The unique identifier of the run
        :return: complete run result including stats, predictions, and metrics
        """
        res = self.api_client.get(f"{RUNS_URI}/{run_id}?project_name={self.project_name}")
        if not res["success"]:
            raise Exception(res.get("details"))
        return res.get("details")

    def get_run_predictions(
        self,
        run_id: str,
        correct: Optional[bool] = None,
        offset: int = 0,
        limit: int = 100,
    ) -> dict:
        """Return paginated predictions/answers for a specific run.

        :param run_id: The unique identifier of the run
        :param correct: Optional filter for correct/incorrect predictions
        :param offset: Pagination offset
        :param limit: Maximum number of predictions to return
        :return: paginated predictions data
        """
        params = f"project_name={self.project_name}&offset={offset}&limit={limit}"
        if correct is not None:
            params += f"&correct={str(correct).lower()}"
        res = self.api_client.get(f"{RUNS_URI}/{run_id}/predictions?{params}")
        if not res["success"]:
            raise Exception(res.get("details"))
        return res.get("details")

    def run_comparison(
        self,
        baseline_run_id: str,
        candidate_run_id: str,
    ) -> dict:
        """Compare two evaluation runs and return deltas, grades, and significance.
        :param baseline_run_id: ID of the baseline run
        :param candidate_run_id: ID of the candidate run
        :return: comparison results including deltas, grade, retention, and significance
        """
        payload = {"project_name": self.project_name}
        if baseline_run_id and candidate_run_id:
            payload["baseline_run_id"] = baseline_run_id
            payload["candidate_run_id"] = candidate_run_id
        else:
            raise ValueError("Either run_ids or both baseline_run_id and candidate_run_id must be provided")
        res = self.api_client.post(COMPARE_URI, payload=payload)
        if not res["success"]:
            raise Exception(res.get("details"))
        return res.get("details")

    def list_comparisons(self) -> pd.DataFrame:
        """Return a DataFrame listing all saved comparisons for this project.

        :return: a DataFrame containing comparison summaries
        """
        res = self.api_client.get(f"{COMPARE_URI}?project_name={self.project_name}")
        if not res["success"]:
            raise Exception(res.get("details"))
        return pd.DataFrame(res.get("details"))

    def available_benchmarks(self) -> pd.DataFrame:
        """Return a DataFrame listing all available benchmark tasks.

        :return: a DataFrame containing benchmark task details
        """
        res = self.api_client.get(BENCHMARKS_URI)
        if not res["success"]:
            raise Exception(res.get("details"))
        return pd.DataFrame(res.get("details"))


class CaseText(BaseModel):
    """Explainability view for text-based cases. Supports token-level importance, attention visualization, and LLM output analysis."""

    model_name: str
    status: str
    prompt: str
    output: str
    explainability: Optional[Dict] = {}
    audit_trail: Optional[Dict] = {}

    def prompt(self):
        """Get prompt
        Encapsulates a small unit of SDK logic and returns the computed result."""
        return self.prompt

    def output(self):
        """Get output
        Encapsulates a small unit of SDK logic and returns the computed result."""
        return self.output

    def xai_raw_data(self) -> pd.DataFrame:
        """Return the raw data used for the case as a DataFrame, with feature names and values.

        :return: raw data dataframe
        """
        raw_data_df = (
            pd.DataFrame([self.explainability.get("feature_importance", {})])
            .transpose()
            .reset_index()
            .rename(columns={"index": "Feature", 0: "Value"})
            .sort_values(by="Value", ascending=False)
        )
        return raw_data_df

    def xai_feature_importance(self):
        """Plots Feature Importance chart
        Encapsulates a small unit of SDK logic and returns the computed result."""
        fig = go.Figure()
        feature_importance = self.explainability.get("feature_importance", {})

        if not feature_importance:
            return "No Feature Importance for the case"
        raw_data_df = (
            pd.DataFrame([feature_importance])
            .transpose()
            .reset_index()
            .rename(columns={"index": "Feature", 0: "Value"})
            .sort_values(by="Value", ascending=False)
        )
        fig.add_trace(
            go.Bar(x=raw_data_df["Value"], y=raw_data_df["Feature"], orientation="h")
        )
        fig.update_layout(
            barmode="relative",
            height=max(400, len(raw_data_df) * 20),
            width=800,
            yaxis=dict(
                autorange="reversed",
                tickmode="array",
                tickvals=list(raw_data_df["Feature"]),
                ticktext=list(raw_data_df["Feature"]),
                tickfont=dict(size=10),
            ),
            bargap=0.01,
            margin=dict(l=150, r=20, t=30, b=30),
            legend_orientation="h",
            legend_x=0.1,
            legend_y=0.5,
        )

        fig.show(config={"displaylogo": False})

    def network_graph(self):
        """Decode and return a base64-encoded network graph image.
        Encapsulates a small unit of SDK logic and returns the computed result."""
        network_graph_data = self.explainability.get("network_graph", {})
        if not network_graph_data:
            return "No Network graph found for this case"
        base64_str = network_graph_data
        try:
            img_bytes = base64.b64decode(base64_str)
            svg_str = img_bytes.decode("utf-8")
            display(SVG(svg_str))
        except Exception as e:
            print(f"Error decoding base64 image: {e}")
            return None

    def token_attribution_graph(self):
        """Decode and return a base64-encoded token attribution graph.
        Encapsulates a small unit of SDK logic and returns the computed result."""
        relevance_data = self.explainability.get("relevance", {})
        if not relevance_data:
            return "No Token Attribution graph found for this case"
        base64_str = relevance_data
        try:
            img_bytes = base64.b64decode(base64_str)
            image = Image.open(BytesIO(img_bytes))
            return image
        except Exception as e:
            print(f"Error decoding base64 image: {e}")
            return None

    def audit(self):
        """Return audit details for the text case.
        Encapsulates a small unit of SDK logic and returns the computed result."""
        return self.audit_trail
