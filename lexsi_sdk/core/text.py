from datetime import datetime
import io
from typing import Optional, List, Dict, Any, Union

import httpx
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
    UPDATE_GUARDRAILS_STATUS_URI,
    UPLOAD_DATA_FILE_URI,
    UPLOAD_DATA_URI,
    UPLOAD_FILE_DATA_CONNECTORS,
    RUN_CHAT_COMPLETION,
    RUN_IMAGE_GENERATION,
    RUN_CREATE_EMBEDDING,
    RUN_COMPLETION
)
from lexsi_sdk.core.project import Project
import pandas as pd

from lexsi_sdk.core.utils import build_list_data_connector_url
from lexsi_sdk.core.wrapper import LexsiModels, monitor
import json
import aiohttp
from typing import AsyncIterator, Iterator
import requests
from uuid import UUID

class TextProject(Project):
    """Project for text modality

    :return: TextProject
    """

    def llm_monitor(self, client, session_id=None):
        """llm monitoring for custom clients

        :param client: client to monitor like OpenAI
        :param session_id: id of the session
        :return: response
        """
        return monitor(project=self, client=client, session_id=session_id)

    def sessions(self) -> pd.DataFrame:
        """All sessions

        :return: response
        """
        res = self.api_client.get(f"{SESSIONS_URI}?project_name={self.project_name}")
        if not res["success"]:
            raise Exception(res.get("details"))

        return pd.DataFrame(res.get("details"))

    def messages(self, session_id: str) -> pd.DataFrame:
        """All messages for a session

        :param session_id: id of the session
        :return: response
        """
        res = self.api_client.get(
            f"{MESSAGES_URI}?project_name={self.project_name}&session_id={session_id}"
        )
        if not res["success"]:
            raise Exception(res.get("details"))

        return pd.DataFrame(res.get("details"))

    def traces(self, trace_id: str) -> pd.DataFrame:
        """Traces generated for trace_id

        :param trace_id: id of the trace
        :return: response
        """
        res = self.api_client.get(
            f"{TRACES_URI}?project_name={self.project_name}&trace_id={trace_id}"
        )
        if not res["success"]:
            raise Exception(res.get("details"))

        return pd.DataFrame(res.get("details"))

    def guardrails(self) -> pd.DataFrame:
        """Guardrails configured in project

        :return: response
        """
        res = self.api_client.get(
            f"{GET_GUARDRAILS_URI}?project_name={self.project_name}"
        )
        if not res["success"]:
            raise Exception(res.get("details"))

        return pd.DataFrame(res.get("details"))

    def update_guardrail_status(self, guardrail_id: str, status: bool) -> str:
        """Update Guardrail Status

        :param guardrail_id: id of the guardrail
        :param status: status to active/inactive
        :return: response
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
        """Deletes Guardrail

        :param guardrail_id: id of the guardrail
        :return: response
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
        """Available guardrails to configure

        :return: response
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
        """Configure guardrail for project

        :param guardrail_name: name of the guardrail
        :param guardrail_config: config for the guardrail
        :param model_name: name of the model
        :param apply_on: when to apply guardrails input/output
        :return: response
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
        model_type: str,  
        inference_compute: InferenceCompute,
        inference_settings: InferenceSettings,
        assets: Optional[dict] = None,
    ) -> str:
        """Initialize text model

        :param model_provider: model of provider
        :param model_name: name of the model to be initialized
        :param model_task_type: task type of model
        :return: response
        """
        payload = {
            "model_provider": model_provider,
            "model_name": model_name,
            "model_task_type": model_task_type,
            "project_name": self.project_name,
            "model_type": model_type,
            "inference_compute": inference_compute,
            "inference_settings": inference_settings
        }
        if assets:
            payload["assets"] = assets
        res = self.api_client.post(f"{INITIALIZE_TEXT_MODEL_URI}", payload)
        if not res["success"]:
            raise Exception(res.get("details", "Model Initialization Failed"))
        poll_events(self.api_client, self.project_name, res["event_id"])

    def model_inference_settings(
        self,  
        model_name: str,
        inference_compute: InferenceCompute,
        inference_settings: InferenceSettings,
    ) -> str:
        """Model Inference Settings

        :param model_provider: model of provider
        :param model_name: name of the model to be initialized
        :param model_task_type: task type of model
        :return: response
        """
        payload = {
            "model_name": model_name,
            "project_name": self.project_name,
            "inference_compute": inference_compute,
            "inference_settings": inference_settings
        }

        res = self.api_client.post(f"{TEXT_MODEL_INFERENCE_SETTINGS_URI}", payload)
        if not res["success"]:
            raise Exception(res.get("details", "Failed to update inference settings"))

    def generate_text_case(
        self,
        model_name: str,
        prompt: str,
        serverless_instance_type: str,
        instance_type: Optional[str] = None,
        explainability_method: Optional[list] = ["DLB"],
        explain_model: Optional[bool] = False,
        session_id: Optional[str] = None,
        max_tokens: Optional[int] = None,
        min_tokens: Optional[int] = None,
        stream: Optional[bool] = False,
    ) -> dict:
        """Generate Text Case

        :param model_name: name of the model
        :param model_type: type of the model
        :param input_text: input text for the case
        :param tag: tag for the case
        :param task_type: task type for the case, defaults to None
        :param instance_type: instance type for the case, defaults to None
        :param explainability_method: explainability method for the case, defaults to None
        :param explain_model: explain model for the case, defaults to False
        :return: response
        """
        if explain_model and not instance_type:
            raise Exception("instance_type required for explainability.")
        llm = monitor(
            project=self, client=LexsiModels(project=self, api_client=self.api_client), session_id=session_id
        )
        res = llm.generate_text_case(
            model_name=model_name,
            prompt=prompt,
            instance_type=instance_type,
            serverless_instance_type=serverless_instance_type,
            explainability_method=explainability_method,
            explain_model=explain_model,
            max_tokens=max_tokens,
            min_tokens=min_tokens,
            stream=stream
        )
        return res

    def available_text_models(self) -> pd.DataFrame:
        """Get available text models

        :return: list of available text models
        """
        res = self.api_client.get(f"{GET_AVAILABLE_TEXT_MODELS_URI}")
        if not res["success"]:
            raise Exception(res.get("details"," Failed to fetch available text models"))
        return pd.DataFrame(res.get("details"))
    
    def upload_data(
        self, 
        data: str | pd.DataFrame,
        tag: str,
    ) -> str:
        """Upload text data for the current project.

        :param data: File path or DataFrame containing rows to upload.
        :param tag: Tag to associate with the uploaded data.
        :return: Server response details.
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
        dataset_name: Optional[str] = None
    ):
        """Upload text data stored in a configured data connector.

        :param data_connector_name: Name of the configured connector.
        :param tag: Tag to associate with uploaded data.
        :param bucket_name: Bucket/location name when required by connector.
        :param file_path: File path within the connector store.
        :param dataset_name: Optional dataset name to persist.
        :return: Server response details.
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
        model_name:str,
        quant_name:str,
        quantization_type:str,
        qbit: int,
        instance_type: str,
        tag: Optional[str] = None,
        input_column: Optional[str] = None,
        no_of_samples: Optional[str] = None,
    ):
        """Quantize Model

        :param model_name: name of the model
        :param quant_name: quant name of the model
        :param quantization_type: type of quantization 
        :param qbit: quantization bit
        :param instance_type: instance type for the quantization 
        :param tag: tag name to pass
        :param input_column: input column for the data
        :param no_of_samples: no of samples for quantization to perform 
        :return: response
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
            "no_of_samples": no_of_samples
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
        session_id : Optional[UUID] = None,
        max_tokens: Optional[int] = None,
        stream: Optional[bool] = False,
    ) -> Union[dict, Iterator[str]]:
        """Chat completion endpoint wrapper

        :param model: name of the model
        :param messages: list of chat messages
        :param provider: model provider (e.g., "openai", "anthropic")
        :param api_key: API key for the provider
        :param max_tokens: maximum tokens to generate
        :param stream: whether to stream the response
        :return: chat completion response or stream iterator
        """
        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "stream": stream,
            "project_name": self.project_name,
            "provider": provider,
            "api_key": api_key,
            "session_id" : session_id
        }

        if not stream:
            return self.api_client.post(RUN_CHAT_COMPLETION, payload=payload)
        
        return self.api_client.stream(uri=RUN_CHAT_COMPLETION, method="POST", payload=payload)

    def create_embeddings(
        self,
        input : Union[str, List[str]],
        model: str,
        api_key : str,
        provider: str,
        session_id : Optional[UUID] = None,
    ) -> dict:  
        payload = {
            "model": model,
            "input": input,
            "project_name": self.project_name,
            "provider": provider,
            "api_key": api_key,
            "session_id" : session_id
        }

        res = self.api_client.post(RUN_CREATE_EMBEDDING, payload=payload)
        return res
    
    def completion(
        self,
        model: str,
        prompt: str,
        provider: str,
        api_key: Optional[str] = None,
        session_id : Optional[UUID] = None,
        max_tokens: Optional[int] = None,
        stream: Optional[bool] = False,            
    ) -> dict:
        
        payload = {
            "model": model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "stream": stream,
            "project_name": self.project_name,
            "provider": provider,
            "api_key": api_key,
            "session_id" : session_id
        }
        if not stream:
            return self.api_client.post(RUN_COMPLETION, payload=payload)
        
        return self.api_client.stream(uri=RUN_COMPLETION, method="POST", payload=payload)

    def image_generation(
        self,
        model: str,
        prompt: str,
        provider: str,
        api_key: str,
        session_id : Optional[UUID] = None,
    ) -> dict:
        """Image generation endpoint wrapper

        :param model: name of the model
        :param prompt: image generation prompt
        :param provider: model provider (e.g., "openai", "stability")
        :param api_key: API key for the provider
        :return: image generation response
        """
        payload = {
            "model": model,
            "prompt": prompt,
            "project_name": self.project_name,
            "provider": provider,
            "api_key": api_key,
            "session_id" : session_id
        }

        res = self.api_client.post(RUN_IMAGE_GENERATION, payload=payload)
            
        return res
