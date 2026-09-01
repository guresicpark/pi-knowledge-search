# Environment Variable Reference

All settings can be overridden via environment variables. The config file (`{cwd}/.pi/knowledge-search.json`, project-local) is checked first, then env vars override individual fields.

| Variable                      | Description                                                       | Default                              |
| ----------------------------- | ---------------------------------------------------------------- | ------------------------------------ |
| `KNOWLEDGE_SEARCH_CONFIG`     | Path to config file                                               | `{cwd}/.pi/knowledge-search.json`    |
| `KNOWLEDGE_SEARCH_DIRS`       | Comma-separated directories to index                              | _(from config file)_                 |
| `KNOWLEDGE_SEARCH_EXTENSIONS` | Comma-separated file extensions                                   | nomic text group (see [README](../README.md#config-file)) |
| `KNOWLEDGE_SEARCH_EXCLUDE`    | Comma-separated directory names to skip                           | `node_modules,.git,.obsidian,.trash` |
| `KNOWLEDGE_SEARCH_INDEX_DIR`  | Where to store the index                                          | `{cwd}/.pi/knowledge-search`         |
| `KNOWLEDGE_SEARCH_AUTO_INJECT` | Per-turn knowledge lookup injection (re-enabled at startup while vectors exist) | `true` |

The embedding engine is fixed — `nomic-ai/nomic-embed-text-v1.5` via local ONNX — and is not configurable. Its weights are cached in the shared HuggingFace cache (`PI_RAG_MODEL_CACHE` > `TRANSFORMERS_CACHE` > `HF_HOME/transformers` > `~/.cache/huggingface/transformers`), so pi-local-rag and pi-knowledge-search reuse the same download.

## Overview injection

| Variable                                  | Default | Description                                                     |
| ----------------------------------------- | ------- | --------------------------------------------------------------- |
| `KNOWLEDGE_SEARCH_OVERVIEW_INJECT`        | `true`  | Inject a vault overview (source dirs + note counts) on start.   |

Boolean values accept `true`/`false`, `1`/`0`, `yes`/`no`, `on`/`off` (case-insensitive).
