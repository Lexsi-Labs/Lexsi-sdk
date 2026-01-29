from __future__ import annotations
from datetime import datetime
import io
from io import BytesIO
from typing import Optional, List, Dict, Any, Union

import httpx
from lexsi_sdk.common.types import InferenceCompute, InferenceSettings
from pydantic import BaseModel
from pydantic import BaseModel
import plotly.graph_objects as go
import base64
from PIL import Image
from lexsi_sdk.common.types import InferenceCompute, InferenceSettings
from lexsi_sdk.common.utils import poll_events
from lexsi_sdk.common.xai_uris import (
    AVAILABLE_GUARDRAILS_URI,
    CONFIGURE_GUARDRAILS_URI,
    DELETE_GUARDRAILS_URI,
    GET_AVAILABLE_TEXT_MODELS_URI,
    GET_GUARDRAILS_URI,
    INITIALIZE_TEXT_MODEL_URI,
    LIST_DATA_CONNECTORS,
    MESSAGES_URI,
    QUANTIZE_MODELS_URI,
    SESSIONS_URI,
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
    GENERATE_TEXT_CASE_URI
)
from lexsi_sdk.core.project import Project
import pandas as pd

from lexsi_sdk.core.utils import build_list_data_connector_url
import json
from typing import Iterator
from uuid import UUID


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

    def update_guardrail_status(self, guardrail_id: str, status: bool) -> str:
        """Update the status (active or inactive) of a specified guardrail.
        Requires the guardrail_id and a boolean status value.

        :param guardrail_id: ID of the guardrail
        :param status: Boolean value indicating whether the guardrail should be active (True) or inactive (False)
        :return: a response indicating the result of the update operation
        """

        payload = {
            "project_name": self.project_name,
            "guardrail_id": guardrail_id,
            "status": status,
        }
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
    ) -> str:
        """Configure a new guardrail in the project.
        Requires the guardrail name, a configuration dictionary, the model name, and where to apply it (input or output).
        Returns a confirmation message.

        :param guardrail_name: Name of the guardrail
        :param guardrail_config: Configuration dictionary for the guardrail
        :param model_name: Name of the model to which the guardrail applies
        :param apply_on: Specifies when to apply the guardrail ("input" or "output")
        :return: a response indicating the result of the configuration operation
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
        instance_type: str,
        tag: Optional[str] = None,
        input_column: Optional[str] = None,
        no_of_samples: Optional[str] = None,
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
        :param instance_type: Instance type used for performing quantization
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
            "instance_type": instance_type,
            "tag": tag,
            "input_column": input_column,
            "no_of_samples": no_of_samples,
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
        stream: Optional[bool] = False,
    ) -> Union[dict, Iterator[str]]:
        """Generate a chat completion using an OpenAI-compliant interface.

        :param model: Name of the model to use for generating the chat completion
        :param messages: List of chat messages, where each message contains a role and content
        :param provider: Model provider (e.g., "OpenAI", "Anthropic")
        :param api_key: API key for the selected provider, if required
        :param session_id: Session ID associated with this chat completion, if provided
        :param max_tokens: Maximum number of tokens to generate
        :param stream: Whether to stream the response
        :return: a chat completion response dictionary or a streaming iterator of response chunks
        """
        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "stream": stream,
            "project_name": self.project_name,
            "provider": provider,
            "api_key": api_key,
            "session_id": session_id,
        }

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
            "max_tokens": max_tokens,
            "stream": stream,
            "project_name": self.project_name,
            "provider": provider,
            "api_key": api_key,
            "session_id": session_id,
        }
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
        XAI_pods: Optional[str] = "xlarge",
        # explainability_method: List[str] = ["DLB"],
        XAI: bool = False,
        session_id: Optional[UUID] = None,
        min_tokens: int = 100,
        max_tokens: int = 1024,
    ) -> dict :
        """Generate an explainable text case using a hosted Lexsi model.

        :param model: Name of the deployed text model.
        :param prompt: Input prompt for generation.
        :param XAI_pods: Dedicated instance type (if applicable).
        :param XAI: Whether to explain the model behavior.
        :param session_id: Optional existing session id.
        :param min_tokens: Minimum tokens to generate.
        :param max_tokens: Maximum tokens to generate.
        :return: API response with generation details.
        """
        payload = {
            "session_id": session_id,
            "project_name": self.project_name,
            "model": model,
            "prompt": prompt,
            "XAI_pods": XAI_pods,
            "provider" : "Lexsi",
            "XAI": XAI,
            "max_tokens": max_tokens,
            "min_tokens": min_tokens,
        }
    
        res = self.api_client.post(GENERATE_TEXT_CASE_URI, payload)
        if not res.get("success"):
            raise Exception(res.get("details"))
        return res
    
    def update_inference_model_status(self, model_name: str, activate: bool) -> str:
        """Sets the provided model to active for inferencing

        :param model_name: name of the model
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
            image = Image.open(BytesIO(img_bytes))
            return image
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

