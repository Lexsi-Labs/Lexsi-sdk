import time
from typing import List, Optional
import uuid
from fastapi import FastAPI

from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime
from pydantic import BaseModel
from typing_extensions import Literal
from transformers import (
    pipeline
)
import torch
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

pipeline = pipeline(
    task='text-generation', 
    model='meta-llama/Llama-3.2-1B-Instruct', 
    token=os.getenv("HUGGING_FACE_HUB_TOKEN"), 
    trust_remote_code=True, 
    torch_dtype="auto",
)
model = pipeline.model
tokenizer = pipeline.tokenizer

@app.get("/health")
def healthcheck():
    return {
        "success": True, 
        "time": datetime.utcnow(), 
        "details": "API working fine"
    }


class ChatCompletionResponseMessage(BaseModel):
    role: Literal["assistant"]
    content: str

class ChatCompletionChoices(BaseModel):
    index: int
    message: ChatCompletionResponseMessage
    finish_reason: Literal["stop"]

class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

class ChatCompletionResponse(BaseModel):
    id: str
    object: Literal["chat.completion"]
    created: int
    model: str
    choices: List[ChatCompletionChoices]
    usage: Optional[Usage] = None

class ChatCompletionRequestMessage(BaseModel):
    role: Literal["system", "user"]
    content: str

class ChatCompletionRequest(BaseModel):
    model: str
    stream: Optional[bool] = False
    messages: List[ChatCompletionRequestMessage]


@app.post("/v1/chat/completions")
def chat_completions(payload: ChatCompletionRequest) -> ChatCompletionResponse:
    prompt = tokenizer.apply_chat_template(
        payload.model_dump().get("messages"),
        tokenize=False,
        add_generation_prompt=True
    )

    result = pipeline(
        prompt, 
        max_new_tokens=500,
        return_full_text=False,
        do_sample=True,
        eos_token_id=tokenizer.eos_token_id,
    )
    generated = result[0]["generated_text"]

    prompt_tokens = len(tokenizer(prompt)["input_ids"])
    completion_tokens = len(tokenizer(generated)["input_ids"])

    return ChatCompletionResponse(
        id=f"chatcmpl-{uuid.uuid4()}",
        object="chat.completion",
        created=int(time.time()),
        model=payload.model,
        choices=[
            ChatCompletionChoices(
                index=0,
                message=ChatCompletionResponseMessage(role="assistant", content=generated),
                finish_reason="stop"
            )
        ],
        usage=Usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        )
    )

class CompletionChoices(BaseModel):
    index: int
    text: str
    finish_reason: Literal["stop"]

class CompletionResponse(BaseModel):
    id: str
    object: Literal["text_completion"]
    created: int
    model: str
    choices: List[CompletionChoices]
    usage: Optional[Usage] = None

class CompletionRequest(BaseModel):
    model: str
    stream: Optional[bool] = False
    prompt: str
#comment here
@app.post("/v1/completions")
def completions(payload: CompletionRequest) -> CompletionResponse:
    result = pipeline(
        payload.prompt,
        max_new_tokens=500,
        return_full_text=False,
        do_sample=True,
        eos_token_id=tokenizer.eos_token_id,
    )
    generated = result[0]["generated_text"]

    prompt_tokens = len(tokenizer(payload.prompt)["input_ids"])
    completion_tokens = len(tokenizer(generated)["input_ids"])

    return CompletionResponse(
        id=f"cmpl-{uuid.uuid4()}",
        object="text_completion",
        created=int(time.time()),
        model=payload.model,
        choices=[
            CompletionChoices(index=0, text=generated, finish_reason="stop")
        ],
        usage=Usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        )
    )