# pi-knowledge-search

Hybrid search over local files for [pi](https://github.com/badlogic/pi). Indexes directories of text/markdown files using vector embeddings **and** SQLite FTS5 keyword search, and exposes `knowledge_search` + `kb_read` tools the LLM can call. Indexing runs on session startup (file changes mid-session are picked up on next restart or via `/knowledge-sync`).

On session start, injects a folder+keyword overview of the indexed vault as a custom message so the model knows what’s worth searching for before it asks.

## How search works

Every query runs against two backends in parallel and fuses the results via Reciprocal Rank Fusion (k=60):

- **Vector cosine similarity** — good for conceptual/fuzzy queries ("how did we handle X")
- **BM25 full-text** via SQLite FTS5 — good for exact matches, proper nouns, error strings, file paths, code identifiers

Docs that both backends agree on get boosted; either backend alone still surfaces relevant hits. If the embedder fails transiently, search falls back to pure BM25; if the FTS side-car is empty, it falls back to pure vector. Existing users upgrade seamlessly — the FTS side-car is backfilled from the vector index on first load with no re-embedding needed.

## Tools

The extension registers two LLM-facing tools:

| Tool | What it does |
|------|--------------|
| `knowledge_search` | Hybrid vector + BM25 search over indexed files (and any Bedrock Knowledge Bases). Returns passage-level excerpts ranked by Reciprocal Rank Fusion. |
| `kb_read` | Resolve a note reference — `[[wikilink]]`, basename, or relative path — to an indexed file and return its full content. Use when the model knows a note's name but not its full path, instead of running find/grep first. |

`kb_read` handles `[[Foo]]`, `[[Foo|alias]]`, `[[Foo#Heading]]`, bare names with or without extension (`Foo`, `Foo.md`), and relative paths (`evergreen/foo`). Multi-match references get a disambiguation prompt instead of guessing.

## Overview injection

On session start, pi-knowledge-search injects a one-shot folder+keyword summary of the indexed vault as a custom message. This gives the model a prior on what's in the knowledge base without having to discover the structure through trial-and-error searches.

The overview is built from whatever the index has loaded from disk — no extra scan — and includes:

- Folders grouped at configurable depth, sorted by note count
- Top keywords per folder (TF-IDF over filenames and headings)
- Optional `NAPKIN.md` / `README.md` / `_about.md` body as folder context

Override settings in the config file:

```json
{
  "overview": {
    "inject": true,
    "maxDepth": 2,
    "maxFoldersPerDir": 20,
    "maxKeywordsPerFolder": 5
  }
}
```

Or via env vars: `KNOWLEDGE_SEARCH_OVERVIEW_INJECT=false` disables injection, `KNOWLEDGE_SEARCH_OVERVIEW_MAX_DEPTH=3` deepens bucketing, etc.

## Install

**Recommended:** Install [pi-total-recall](https://github.com/samfoy/pi-total-recall) to get the complete context stack — persistent memory, session history search, and local knowledge search in one package:

```bash
pi install pi-total-recall
```

Or install pi-knowledge-search standalone:

```bash
pi install git:github.com/samfoy/pi-knowledge-search
```

Or try without installing:

```bash
pi -e git:github.com/samfoy/pi-knowledge-search
```

## Setup

Run the interactive setup command inside pi:

```
/knowledge-search-setup
```

This walks you through:

1. **Directories** to index (comma-separated paths)
2. **File extensions** to include (default: `.md, .txt`)
3. **Directories to exclude** (default: `node_modules, .git, .obsidian, .trash`)
4. **Embedding provider** — OpenAI, OpenAI-compatible (local/self-hosted), AWS Bedrock, or Ollama

Config is saved to `~/.pi/knowledge-search.json`. Run `/reload` to activate.

### Config file

You can also edit the config file directly:

```json
{
  "dirs": ["~/notes", "~/docs"],
  "fileExtensions": [".md", ".txt"],
  "excludeDirs": ["node_modules", ".git", ".obsidian", ".trash"],
  "provider": {
    "type": "openai",
    "model": "text-embedding-3-small"
  }
}
```

The API key for OpenAI can be set in the config file (`"apiKey": "sk-..."`) or via the `OPENAI_API_KEY` environment variable.

<details>
<summary>Bedrock config</summary>

```json
{
  "dirs": ["~/vault"],
  "provider": {
    "type": "bedrock",
    "profile": "my-aws-profile",
    "region": "us-west-2",
    "model": "amazon.titan-embed-text-v2:0"
  }
}
```

Requires the AWS SDK and valid credentials for the specified profile.

</details>

<details>
<summary>Ollama config (free, local)</summary>

```json
{
  "dirs": ["~/notes"],
  "provider": {
    "type": "ollama",
    "url": "http://localhost:11434",
    "model": "nomic-embed-text"
  }
}
```

Requires [Ollama](https://ollama.ai) running locally:

```bash
ollama serve
ollama pull nomic-embed-text
```

</details>

<details>
<summary>OpenAI-compatible config (free, local/self-hosted)</summary>

Any server that exposes an OpenAI-compatible `/v1/embeddings` endpoint works:
[llama.cpp](https://github.com/ggml-org/llama.cpp), [vLLM](https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html),
[litellm](https://docs.litellm.ai/), [Ollama's OpenAI-compatibility mode](https://ollama.com/blog/openai-compatibility), etc.

```json
{
  "dirs": ["~/notes"],
  "provider": {
    "type": "openai-compatible",
    "baseUrl": "http://127.0.0.1:8080",
    "apiKey": "your-local-key",
    "model": "qwen3-embeddings"
  }
}
```

The `baseUrl` should be your server root **without** a trailing `/v1` path — the embedder appends `/v1/embeddings` automatically.

For example with llama-cpp-python:

```bash
python -m llama_cpp.server --model ./models/qwen3-embedding.gguf --port 8080
```

Then configure knowledge-search to point at `http://127.0.0.1:8080` as shown above.

The `apiKey` field is optional; omit it if your runner doesn't require authentication.

</details>

### Bedrock Knowledge Bases

You can add [Amazon Bedrock Knowledge Bases](https://docs.aws.amazon.com/bedrock/latest/userguide/knowledge-base.html) as additional search sources. These are managed RAG services — Amazon handles chunking, embedding, and vector storage. pi-knowledge-search queries them at search time and merges results with local file results.

Add via command:

```
/knowledge-add-kb
```

Or add directly to the config file:

```json
{
  "dirs": ["~/notes"],
  "provider": { "type": "openai" },
  "knowledgeBases": [
    {
      "id": "XXXXXXXXXX",
      "region": "us-east-1",
      "profile": "default",
      "label": "Team docs"
    }
  ]
}
```

You can use Knowledge Bases alongside local file indexing, or on their own (omit `dirs` and `provider` for KB-only mode).

KB-only config:

```json
{
  "knowledgeBases": [
    {
      "id": "XXXXXXXXXX",
      "region": "us-east-1",
      "profile": "my-work-profile",
      "label": "Engineering wiki"
    }
  ]
}
```

Requires the AWS SDK and valid credentials with `bedrock:Retrieve` permissions.

### Environment variable overrides

Every config field can be overridden via environment variables. This is useful for CI or when you want different settings per shell session. See [env-vars.md](docs/env-vars.md) for the full list.

## How it works

1. On session start, loads the index from disk and incrementally syncs — only re-embeds new or modified files
2. Registers a `knowledge_search` tool the LLM calls with natural language queries
3. Returns ranked results with file paths, relevance scores, and content excerpts

Sync runs on session startup only. Files changed mid-session won't be searchable until the next session start or a manual `/knowledge-reindex`.

The index is stored at `~/.pi/knowledge-search/index.json`.

## Commands

| Command                   | Description                                     |
| ------------------------- | ----------------------------------------------- |
| `/knowledge-search-setup` | Interactive setup wizard                        |
| `/knowledge-add-kb`       | Add a Bedrock Knowledge Base as a search source |
| `/knowledge-overview`     | Force-rebuild and re-inject the vault overview  |
| `/knowledge-reindex`      | Force a full re-index                           |

## Performance

Typical numbers for ~500 markdown files (~20MB):

| Operation                     | Time   |
| ----------------------------- | ------ |
| Full index build              | ~7s    |
| Incremental sync (no changes) | ~12ms  |
| Search query                  | ~250ms |
| Index file size               | ~5MB   |

## Project-local storage

By default, config lives at `~/.pi/knowledge-search.json` and the index at `~/.pi/knowledge-search/`. To relocate per-project, add one of the following to `{project}/.pi/settings.json`:

```jsonc
{
  "pi-knowledge-search": {
    "localPath": ".pi/knowledge-search"   // config.json + index/ under this path
  }
}
```

Or via the [`pi-total-recall`](https://github.com/samfoy/pi-total-recall) cascade:

```jsonc
{
  "pi-total-recall": {
    "localPath": ".pi/total-recall"
    // pi-knowledge-search → {project}/.pi/total-recall/knowledge-search/
  }
}
```

**Resolution order (highest priority first):**

1. `KNOWLEDGE_SEARCH_CONFIG` / `KNOWLEDGE_SEARCH_INDEX_DIR` env vars
2. `pi-knowledge-search.localPath` in `{cwd}/.pi/settings.json`
3. `pi-total-recall.localPath` cascade → `{localPath}/knowledge-search/`
4. Global default: `~/.pi/knowledge-search.json` + `~/.pi/knowledge-search/`

Per-project indexes are particularly useful for vault- or doc-tree-scoped embeddings where you don't want cross-project bleed.

## License

MIT
