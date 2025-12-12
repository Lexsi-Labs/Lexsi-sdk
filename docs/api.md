# Lexsi REST API (curl snippets)

Use these ready-to-run curl examples with your Lexsi API key. Replace `$LEXSI_API_KEY` and payload values as needed.

## Basic

### Health check

```bash
curl -X GET "https://apiv1.lexsi.ai/healthcheck"
```
## Text Modality API's

### Text generation with explainability

```bash
curl -X GET "https://apiv1.lexsi.ai/v2/users/case-register" \
  -H "Authorization: Bearer $LEXSI_API_KEY"
```

### Chat completions

```bash
curl --request POST 'https://apiv1.lexsi.ai/gateway/v1/chat/completions' \
  --header 'x-api-token: <$API_TOKEN>' \
  --header 'Content-Type: application/json' \
  --data '{
    "provider": "<$MODEL-PROVIDER-NAME>",
    "api_key": "<$RANDOM-TEXT>",
    "client_id": "<$USERNAME>",
    "max_tokens": <$MAX-NEW-TOKENS>,
    "project_name": "<$PROJECT-NAME>",
    "model": "<$MODEL-NAME>",
    "messages": [
      {
        "role": "<$ROLE>",
        "content": "<$PROMPT>"
      }
    ],
    "stream": <$STREAM_BOOL>
  }'

```

### Completions

```bash
curl --request POST 'https://apiv1.lexsi.ai/gateway/v1/completions' \
  --header 'x-api-token: <$API_TOKEN>' \
  --header 'Content-Type: application/json' \
  --data '{
    "provider": "<$MODEL-PROVIDER-NAME>",
    "api_key": "<$RANDOM-TEXT>",
    "client_id": "<$USERNAME>",
    "max_tokens": <$MAX-NEW-TOKENS>,
    "project_name": "<$PROJECT-NAME>",
    "model": "<$MODEL-NAME>",
    "prompt": "<$PROMPT>",
    "stream": <$STREAM_BOOL>
  }'

```

### Embeddings

```bash
curl -X POST "https://apiv1.lexsi.ai/gateway/v1/embeddings" \
  -H "Authorization: Bearer $LEXSI_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "text-embedding-3-small",
    "input": ["Lexsi makes observability easy"]
  }'
```

### Image generation

```bash
curl -X POST "https://apiv1.lexsi.ai/gateway/v1/images/generations" \
  -H "Authorization: Bearer $LEXSI_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "stability/stable-diffusion-xl-base-1.0",
    "prompt": "A futuristic city skyline at dusk"
  }'
```
