# Environment Variable Reference

All settings can be overridden via environment variables. The config file (`~/.pi/knowledge-search.json`) is checked first, then env vars override individual fields.

| Variable                      | Description                                                       | Default                              |
| ----------------------------- | ----------------------------------------------------------------- | ------------------------------------ |
| `KNOWLEDGE_SEARCH_CONFIG`     | Path to config file                                               | `~/.pi/knowledge-search.json`        |
| `KNOWLEDGE_SEARCH_DIRS`       | Comma-separated directories to index                              | _(from config file)_                 |
| `KNOWLEDGE_SEARCH_EXTENSIONS` | Comma-separated file extensions                                   | `.md,.txt`                           |
| `KNOWLEDGE_SEARCH_EXCLUDE`    | Comma-separated directory names to skip                           | `node_modules,.git,.obsidian,.trash` |
| `KNOWLEDGE_SEARCH_DIMENSIONS` | Embedding vector dimensions                                       | `512` (`768` for `transformers`)     |
| `KNOWLEDGE_SEARCH_INDEX_DIR`  | Where to store the index                                          | `~/.pi/knowledge-search`             |
| `KNOWLEDGE_SEARCH_PROVIDER`   | Provider type: `openai`, `openai-compatible`, `bedrock`, `ollama`, `transformers` | `openai` |

### OpenAI

| Variable                                              | Default                  |
| ----------------------------------------------------- | ------------------------ |
| `OPENAI_API_KEY` or `KNOWLEDGE_SEARCH_OPENAI_API_KEY` | _(required)_             |
| `KNOWLEDGE_SEARCH_OPENAI_MODEL`                       | `text-embedding-3-small` |

### Bedrock

| Variable                           | Default                        |
| ---------------------------------- | ------------------------------ |
| `KNOWLEDGE_SEARCH_BEDROCK_PROFILE` | `default`                      |
| `KNOWLEDGE_SEARCH_BEDROCK_REGION`  | `us-east-1`                    |
| `KNOWLEDGE_SEARCH_BEDROCK_MODEL`   | `amazon.titan-embed-text-v2:0` |

### OpenAI-compatible (local/self-hosted)

Use any server that exposes an OpenAI-compatible `/v1/embeddings` endpoint, such as [llama.cpp](https://github.com/ggml-org/llama.cpp), [vLLM](https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html), or [Ollama](https://ollama.ai).

| Variable                           | Default                                                 |
| ---------------------------------- | ------------------------------------------------------- |
| `KNOWLEDGE_SEARCH_COMPAT_BASE_URL` | _(required — e.g. `http://127.0.0.1:8080`) _            |
| `KNOWLEDGE_SEARCH_COMPAT_API_KEY`  | _(optional — omit for runners that don't require auth)_ |
| `KNOWLEDGE_SEARCH_COMPAT_MODEL`    | `text-embedding-3-small`                                |

The `baseUrl` should point to your API root **without** the trailing `/v1` path — the embedder appends `/v1/embeddings` automatically.

> **Note:** `openai-compatible` intentionally does **not** fall back to `OPENAI_API_KEY`. If you point `baseUrl` at a third-party service, we don't want to silently ship your real OpenAI key to it. Set `KNOWLEDGE_SEARCH_COMPAT_API_KEY` explicitly if auth is required.

```bash
# Example using llama-cpp-python on port 8080:
export KNOWLEDGE_SEARCH_PROVIDER=openai-compatible
export KNOWLEDGE_SEARCH_COMPAT_BASE_URL=http://127.0.0.1:8080
export KNOWLEDGE_SEARCH_COMPAT_API_KEY=my-local-key
export KNOWLEDGE_SEARCH_COMPAT_MODEL=qwen3-embeddings
```

Or via config file (`~/.pi/knowledge-search.json`):

```json
{
  "provider": {
    "type": "openai-compatible",
    "baseUrl": "http://127.0.0.1:8080",
    "apiKey": "my-local-key",
    "model": "qwen3-embeddings"
  }
}
```

### Ollama

| Variable                          | Default                    |
| --------------------------------- | -------------------------- |
| `KNOWLEDGE_SEARCH_OLLAMA_URL`     | `http://localhost:11434`   |
| `KNOWLEDGE_SEARCH_OLLAMA_MODEL`   | `nomic-embed-text`         |

### Transformers.js (local ONNX)

| Variable                                | Default                            |
| --------------------------------------- | ---------------------------------- |
| `KNOWLEDGE_SEARCH_TRANSFORMERS_MODEL`   | `nomic-ai/nomic-embed-text-v1.5`   |

Model weights are cached in the shared HuggingFace cache (`PI_RAG_MODEL_CACHE` > `TRANSFORMERS_CACHE` > `HF_HOME/transformers` > `~/.cache/huggingface/transformers`), so pi-local-rag and pi-knowledge-search reuse the same download.

## Overview injection

| Variable                                  | Default | Description                                                     |
| ----------------------------------------- | ------- | --------------------------------------------------------------- |
| `KNOWLEDGE_SEARCH_OVERVIEW_INJECT`        | `true`  | Inject a folder+keyword overview as a custom message on start.  |
| `KNOWLEDGE_SEARCH_OVERVIEW_MAX_DEPTH`     | `2`     | Folder depth used to group files in the overview.               |
| `KNOWLEDGE_SEARCH_OVERVIEW_MAX_FOLDERS`   | `20`    | Max folders surfaced per source dir (ranked by note count).     |
| `KNOWLEDGE_SEARCH_OVERVIEW_MAX_KEYWORDS`  | `5`     | Max TF-IDF keywords shown per folder.                           |

Boolean values accept `true`/`false`, `1`/`0`, `yes`/`no`, `on`/`off` (case-insensitive).
