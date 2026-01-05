# Lexsi REST API (curl snippets)

Use these ready-to-run curl examples with your Lexsi API key. Replace `$LEXSI_API_KEY` and payload values as needed.

## Basic

### Health check

```bash
curl -X GET "https://apiv1.lexsi.ai/healthcheck"
```

## Tabular Modality API's

### Generate Prediction and Explainability

```bash
curl --http2 -X POST "https://apiv1.lexsi.ai/v2/project/case-register" \
   -H "x-api-token: <$X_API_TOKEN>" \
   -H "Content-Type: application/x-www-form-urlencoded" \
   --data-urlencode "client_id=<$USERNAME>" \
   --data-urlencode "project_name=<$PROJECTNAME>" \
   --data-urlencode "unique_identifier=<$UNIQUE_IDENTIFIER>" \
   --data-urlencode "tag=<$Tag>" \
   --data-urlencode "serverless_instance_type=<$SERVERLESS_COMPUTE_TYPE>" \
   --data-urlencode "xai=<$EXPLAINABILITY_METHOD>" \
   --data-urlencode "data=[{
    "$UNIQUE_ID_KEY": "$UNIQUE_IDENTIFIER",
    "$SAMPLE_KEY": "$SAMPLE_VALUE",
    "$SAMPLE_KEY": "$SAMPLE_VALUE",
        .
        .
    "$SAMPLE_KEY": "$SAMPLE_VALUE"
  }]"
```

Placeholders:
- `$X_API_TOKEN`: Your Lexsi API token from the SDK/portal.
- `$USERNAME`: Your Lexsi username/client ID.
- `$PROJECTNAME`: Target project name.
- `$UNIQUE_IDENTIFIER`: Unique row ID for the registered case.
- `$Tag`: Dataset tag you want to attach to this upload / Prediction.
- `$SERVERLESS_COMPUTE_TYPE`: Serverless instance type for processing (e.g., `NOVA`, `GOVA`, or `local`).
- `$EXPLAINABILITY_METHOD`: Explainability technique to run (e.g., `shap`, `lime`).
- `$data`: List of Json containing data in Key:Value pair.

## IMAGE modality API

curl --http2 -X POST "https://apiv1.lexsi.ai/v2/project/case-register" \
   -H "x-api-token: <$X_API_TOKEN>" \
   -F "client_id=<$USERNAME>" \
   -F "project_name=<$PROJECT_NAME>" \
   -F "unique_identifier=<$UNIQUE_IDENTIFIER>" \
   -F "tag=<$TAG>" \
   -F "xai=<$EXPLAINABILITY_METHOD>" \
   -F "serverless_instance_type=<$SERVERLESS_COMPUTE_TYPE>" \
   -F "in_file=<$IMAGE_PATH>"
   -F "image_class=<$IMAGE_CLASS>"

```

Placeholders:
- `$X_API_TOKEN`: Your Lexsi API token.
- `$USERNAME`: Your Lexsi username/client ID.
- `$PROJECT_NAME`: Target project name for this image case.
- `$UNIQUE_IDENTIFIER`: filename of image for the image case.
- `$TAG`: Tag to associate with the upload.
- `$EXPLAINABILITY_METHOD`: Explainability method to run (e.g., `gradcam`).
- `$SERVERLESS_COMPUTE_TYPE`: Serverless instance type for processing.
- `$IMAGE_PATH`: Local path to the image file to upload.
- `$IMAGE_CLASS`: Optional class/label for the image.
```bash


## Text Modality API's

### Text Generation API

```bash
curl --http2 -X POST 'https://apiv1.lexsi.ai/v2/project/case-register' \
  -H "x-api-token: <$API_TOKEN>" \
  -F "provider= <$MODEL_PROVIDER>" \
  -F "client_id=<$USERNAME>" \
  -F "project_name=<$PROJECT_NAME>" \
  -F "prompt=<$INPUT_PROMPT>" \
  -F "serverless_instance_type=<$SERVERLESS_COMPUTE_TYPE>" \
  -F "model_name=<$MODEL_NAME>" \
  -F "min_tokens=<$MIN_TOKENS>" \
  -F "max_tokens=<$MAX_TOKENS>" \
  -F "session_id=<$SESSION_ID>" \
  -F "instance_type=<$POD_INSTANCE_TYPE>" \
  -F "explain_model=<$EXPLAINABILTY_FLAG>"
```

Placeholders:
- `$API_TOKEN`: Your Lexsi API token.
- `$MODEL_PROVIDER`: Provider identifier (e.g., `Lexsi`,`OpenAI`,`Grok`,etc).
- `$USERNAME`: Your Lexsi username/client ID.
- `$PROJECT_NAME`: Target text project name.
- `$INPUT_PROMPT`: User prompt to send to the model.
- `$SERVERLESS_COMPUTE_TYPE`: Serverless instance type for processing.
- `$MODEL_NAME`: Model name within the provider.
- `$MIN_TOKENS`/`$MAX_TOKENS`: Min/Max tokens to generate (integers).
- `$SESSION_ID`: Optional session ID for threaded conversations.
- `$POD_INSTANCE_TYPE`: Optional dedicated instance type for processing.
- `$EXPLAINABILTY_FLAG`: Boolean flag indicating whether to compute explainability.

### Chat completions

```bash
curl --request POST 'https://apiv1.lexsi.ai/gateway/v1/chat/completions' \
  --header "x-api-token: <$API_TOKEN>" \
  --header 'Content-Type: application/json' \
  --data "{
    "provider": "<$MODEL_PROVIDER_NAME>",
    "api_key": "<$API_KEY>",
    "client_id": "<$USERNAME>",
    "max_tokens": <$MAX_NEW_TOKENS>,
    "project_name": "<$PROJECT_NAME>",
    "model": "<$MODEL_NAME>",
    "messages": [
      {
        "role": "<$ROLE>",
        "content": "<$PROMPT>"
      }
    ],
    "stream": <$STREAM_BOOL>
  }"

```

Placeholders:
- `$API_TOKEN`: Your Lexsi API token.
- `$MODEL_PROVIDER_NAME`: Provider identifier (e.g., `Lexsi`,`OpenAI`,`Grok`).
- `$API_KEY`: This is only required for third party models
- `$USERNAME`: Your Lexsi username/client ID.
- `$MAX_NEW_TOKENS`: Maximum tokens to generate (integer).
- `$PROJECT_NAME`: Target project name.
- `$MODEL_NAME`: Model name within the provider.
- `$ROLE`: Message role (e.g., `user`, `system`).
- `$PROMPT`: The input text prompt.
- `$STREAM_BOOL`: `true` or `false` to control streaming responses.


### Completions

```bash
curl --request POST 'https://apiv1.lexsi.ai/gateway/v1/completions' \
  --header "x-api-token: <$API_TOKEN>" \
  --header 'Content-Type: application/json' \
  --data "{
    "provider": "<$MODEL_PROVIDER_NAME>",
    "api_key": "<$RANDOM_TEXT>",
    "client_id": "<$USERNAME>",
    "max_tokens": <$MAX_NEW_TOKENS>,
    "project_name": "<$PROJECT_NAME>",
    "model": "<$MODEL_NAME>",
    "prompt": "<$PROMPT>",
    "stream": <$STREAM_BOOL>
  }"

```

Placeholders:
- `$API_TOKEN`: Your Lexsi API token.
- `$MODEL_PROVIDER_NAME`: Provider identifier (e.g., `openai`).
- `$RANDOM_TEXT`: Provider-specific API key or token if required.
- `$USERNAME`: Your Lexsi username/client ID.
- `$MAX_NEW_TOKENS`: Maximum tokens to generate (integer).
- `$PROJECT_NAME`: Target project name.
- `$MODEL_NAME`: Model name within the provider.
- `$PROMPT`: The input text prompt.
- `$STREAM_BOOL`: `true` or `false` to control streaming responses.


### Embeddings

```bash
curl -X POST "https://apiv1.lexsi.ai/gateway/v1/embeddings" \
  -H "Authorization: Bearer $LEXSI_API_KEY" \
  -H "Content-Type: application/json" \
  -d "{
    "model": "<$MODEL_NAME>",
    "input": ["<$INPUT_PROMPT>"]
  }"
```

Placeholders:
- `$LEXSI_API_KEY`: Your Lexsi API key for gateway endpoints.
- `$MODEL_NAME`: Embedding model to use.
- `$INPUT_PROMPT`: Text to embed (single string or list).

### Image generation

```bash
curl -X POST "https://apiv1.lexsi.ai/gateway/v1/images/generations" \
  -H "Authorization: Bearer $LEXSI_API_KEY" \
  -H "Content-Type: application/json" \
  -d "{
    "model": "<$$MODEL_NAME>",
    "prompt": "<$INPUT_PROMPT>"
  }"
```

Placeholders:
- `$LEXSI_API_KEY`: Your Lexsi API key for gateway endpoints.
- `$MODEL_NAME`: Image generation model to use.
- `$INPUT_PROMPT`: Text prompt describing the image to generate.