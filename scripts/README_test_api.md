# test_api.py - Summary and Workflow

## Short Summary

`test_api.py` is a CLI script that validates API keys and endpoints for LLM (chat) and image generation. It supports **OpenAI**, **Groq**, and **Azure** providers.

**What it does:**
1. Sends a text prompt to the chat completions API and checks the response
2. Optionally tests image generation (DALL-E) and saves output images
3. Reads URL and key from args or `.env`

---

## Workflow

```
START
  |
  v
+------------------+
| Parse args/env   |  --url, --key, --provider, --prompt, --deployment, etc.
+--------+---------+
         |
         v
+------------------+
| Resolve URL      |  Use default per provider if not given
| - Azure: base +  |  Build full URL from base + deployment for Azure
|   deployment     |
+--------+---------+
         |
         v
+------------------+
| Chat test        |  POST to chat/completions
| test_api()       |  - Azure: api-key header
|                  |  - OpenAI/Groq: Bearer token
+--------+---------+
         |
         v
    [Success?]
    /       \
   No        Yes
   |          |
   v          v
 EXIT(1)   Print reply
              |
              v
+------------------+
| --test-image?    |
+--------+---------+
         |
    No   |   Yes
    |    |
    v    v
   END   +------------------+
         | Image test       |  POST to images/generations
         | test_image_gen() |  - Azure: use image-deployment (dall-e-3, etc.)
         |                  |  - Groq: not supported
         +--------+---------+
                  |
                  v
             [Success?]
             /       \
            No        Yes
            |          |
            v          v
         Print FAIL   save_generated_images()
                          |  Parse b64_json or url from response
                          |  Save to output_dir
                          v
                       Print saved paths
                          |
                          v
                         END
```

---

## Functions

| Function | Purpose |
|---------|---------|
| `test_api()` | Chat completions test; returns success/reply or error |
| `test_image_generation()` | Image generation test; returns success/image_count or error |
| `_build_azure_chat_url()` | Builds full Azure chat URL from base + deployment |
| `_get_image_url()` | Derives image URL from chat URL (or uses image-deployment for Azure) |
| `save_generated_images()` | Parses response (b64_json or url) and saves to disk |

---

## Provider Auth

| Provider | Auth header |
|----------|-------------|
| Azure | `api-key: <key>` |
| OpenAI | `Authorization: Bearer <key>` |
| Groq | `Authorization: Bearer <key>` |

---

## Key Options

| Option | Purpose |
|--------|---------|
| `--provider` | openai | groq | azure |
| `--url` | Full chat endpoint (or base for Azure) |
| `--key` | API key |
| `--deployment` | Azure chat deployment (required when URL is base) |
| `--image-deployment` | Azure image deployment (dall-e-3, gpt-image-1, etc.) |
| `--prompt` | Custom text for chat test |
| `--test-image` | Also run image generation test |
| `--output-dir` | Where to save generated images |
