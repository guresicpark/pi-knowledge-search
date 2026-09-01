# Environment Variable Reference

All settings can be overridden via environment variables. The config file (`{cwd}/.pi/knowledge-search.json`, project-local) is checked first, then env vars override individual fields.

| Variable                      | Description                                                       | Default                              |
| ----------------------------- | ---------------------------------------------------------------- | ------------------------------------ |
| `KNOWLEDGE_SEARCH_CONFIG`     | Path to config file                                               | `{cwd}/.pi/knowledge-search.json`    |
| `KNOWLEDGE_SEARCH_DIRS`       | Comma-separated directories to index                              | _(from config file)_                 |
| `KNOWLEDGE_SEARCH_EXTENSIONS` | Comma-separated file extensions                                   | `.md,.txt`                           |
| `KNOWLEDGE_SEARCH_EXCLUDE`    | Comma-separated directory names to skip                           | `node_modules,.git,.obsidian,.trash` |
| `KNOWLEDGE_SEARCH_DIMENSIONS` | Embedding vector dimensions                                       | `768` (with a provider; unused FTS-only) |
| `KNOWLEDGE_SEARCH_INDEX_DIR`  | Where to store the index                                          | `{cwd}/.pi/knowledge-search`         |

### Transformers.js (local ONNX)

| Variable                                | Default                          |
| --------------------------------------- | -------------------------------- |
| `KNOWLEDGE_SEARCH_TRANSFORMERS_MODEL`   | `nomic-ai/nomic-embed-text-v1.5` |

Model weights are cached in the shared HuggingFace cache (`PI_RAG_MODEL_CACHE` > `TRANSFORMERS_CACHE` > `HF_HOME/transformers` > `~/.cache/huggingface/transformers`), so pi-local-rag and pi-knowledge-search reuse the same download.

## Overview injection

| Variable                                  | Default | Description                                                     |
| ----------------------------------------- | ------- | --------------------------------------------------------------- |
| `KNOWLEDGE_SEARCH_OVERVIEW_INJECT`        | `true`  | Inject a folder+keyword overview as a custom message on start.  |
| `KNOWLEDGE_SEARCH_OVERVIEW_MAX_DEPTH`     | `2`     | Folder depth used to group files in the overview.               |
| `KNOWLEDGE_SEARCH_OVERVIEW_MAX_FOLDERS`   | `20`    | Max folders surfaced per source dir (ranked by note count).     |
| `KNOWLEDGE_SEARCH_OVERVIEW_MAX_KEYWORDS`  | `5`     | Max TF-IDF keywords shown per folder.                           |

Boolean values accept `true`/`false`, `1`/`0`, `yes`/`no`, `on`/`off` (case-insensitive).
