# pi-knowledge-search

Hybrid **local** search over local files for [pi](https://github.com/badlogic/pi). Indexes directories of text/markdown files using local ONNX vector embeddings **and** SQLite FTS5 keyword search, exposes `knowledge_search` + `kb_read` tools the LLM can call, and auto-injects a knowledge lookup on every prompt (like pi-local-rag's RAG lookup). Everything runs on your machine — no embedding APIs, no cloud services. Indexing runs on session startup; mid-session file changes are picked up with `/knowledge-search index`.

On session start, injects a folder+keyword overview of the indexed vault as a custom message so the model knows what’s worth searching for before it asks.

## Knowledge lookup

Like pi-local-rag's RAG lookup, every user prompt triggers an automatic **knowledge lookup**: the prompt runs through hybrid search (top 5 chunks, min score 0.1) and the hits are injected as a message right after the prompt — full excerpts for the model, and a green `Knowledge lookup — file.md (2), …` summary box for you (red on failure).

Injection is automatically enabled whenever the index holds vectors — at session start and after `/knowledge-search index` — so `/knowledge-search off` acts as a per-session kill-switch that the next startup flips back on (`autoInject` in the config).

## How search works

Every query runs against two backends in parallel and fuses the results via Reciprocal Rank Fusion (k=60):

- **Vector cosine similarity** — good for conceptual/fuzzy queries ("how did we handle X")
- **BM25 full-text** via SQLite FTS5 — good for exact matches, proper nouns, error strings, file paths, code identifiers

Docs that both backends agree on get boosted; either backend alone still surfaces relevant hits. If the embedder fails transiently, search falls back to pure BM25; if the FTS side-car is empty, it falls back to pure vector. Existing vector indexes upgrade with a one-time full re-embed (the index now records which engine built its vectors — see [Embedding-engine changes](#embedding-engine-changes-remove-existing-embeddings)); FTS-only installs upgrade seamlessly with no re-embedding.

## Tools

The extension registers two LLM-facing tools:

| Tool | What it does |
|------|--------------|
| `knowledge_search` | Hybrid vector + BM25 search over indexed files. Returns passage-level excerpts ranked by Reciprocal Rank Fusion. |
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

```bash
pi install git:github.com/samfoy/pi-knowledge-search
```

Or try without installing:

```bash
pi -e git:github.com/samfoy/pi-knowledge-search
```

Requires **Node 24+** — `node:sqlite` must include FTS5, which Node 22's bundled SQLite does not. On Node 22 you'll get `Error: no such table: chunks` at startup because the FTS5 virtual table never gets created.

## Setup

Everything is driven by the `/knowledge-search` command (mirroring pi-local-rag's `/rag`):

```
/knowledge-search                 # show status: indexed dirs, excludes, extensions, engine
/knowledge-search add ~/notes     # add directories to index (space- or comma-separated)
/knowledge-search exclude build   # add an excluded directory name (-<name> removes, bare lists)
/knowledge-search index           # incrementally index new/changed files (progress bar, like /rag)
/knowledge-search clear           # clear ALL project data + reset settings to defaults (confirm first)
/knowledge-search on | off        # enable/disable the per-turn knowledge lookup injection
/knowledge-search help            # list all subcommands
```

The first `add` on a fresh config defaults the embedding engine to local Transformers.js. Config is saved to `{cwd}/.pi/knowledge-search.json` — project-local, relative to the directory pi was started in. Directories added mid-session are picked up by `/knowledge-search index` without a reload.

### Config file

You can also edit the config file directly:

```json
{
  "dirs": ["~/notes", "~/docs"],
  "fileExtensions": [".md", ".txt"],
  "excludeDirs": ["node_modules", ".git", ".obsidian", ".trash"],
  "autoInject": true,
  "provider": {
    "type": "transformers",
    "model": "nomic-ai/nomic-embed-text-v1.5"
  }
}
```

The `model` field is optional — omit it for the nomic default. `autoInject` (default `true`) controls the per-turn knowledge lookup; it is re-enabled automatically at startup while the index holds vectors. Omit the whole `provider` block for **FTS-only mode**: zero-config pure BM25 keyword search, no model download.

> **Migrating from remote providers:** OpenAI, OpenAI-compatible, Bedrock, and Ollama embedding providers were removed — this extension is local-only now. A config naming a removed provider throws a migration error at startup; switch to `"transformers"` (or remove the `provider` block). Switching engines removes all existing embeddings and re-embeds once on the next sync.

### Transformers.js engine (local ONNX)

The default (and only) embedding engine runs fully locally via [Transformers.js](https://huggingface.co/docs/transformers.js) ONNX inference — no API key, no server. Defaults to `nomic-ai/nomic-embed-text-v1.5` (768-dim, q8 quantized) with the model's `search_query:` / `search_document:` task prefixes, mirroring pi-local-rag's text pipeline. Model weights are downloaded once (~111 MB) into a shared HuggingFace cache (`~/.cache/huggingface/transformers` by default, or `PI_RAG_MODEL_CACHE` / `TRANSFORMERS_CACHE` / `HF_HOME`), so pi-knowledge-search and pi-local-rag reuse the same download.

Any Transformers.js-compatible feature-extraction model id works, but the `search_query:`/`search_document:` prefixes are nomic-specific; other models simply ignore-tolerate them.

### Embedding-engine changes remove existing embeddings

Vectors from different engines, models, or dimensionalities are not comparable. The index records the signature of the engine that built it (`type:model:dimensions`); on load, a mismatch removes all existing embeddings and the next sync re-embeds everything with the new engine. Changing the `model` in your config therefore costs one full re-embed, after which incremental sync resumes. FTS-only installs are unaffected — keyword search doesn't depend on the embedder.

### Environment variable overrides

Every config field can be overridden via environment variables. This is useful for CI or when you want different settings per shell session. See [env-vars.md](docs/env-vars.md) for the full list.

## How it works

1. On session start, loads the index from disk and incrementally syncs — only re-embeds new or modified files (or everything, once, after an embedding-engine change)
2. Registers a `knowledge_search` tool the LLM calls with natural language queries
3. Before every agent turn, runs an automatic knowledge lookup on the prompt and injects the top hits as a message right after it (see [Knowledge lookup](#knowledge-lookup))
4. Returns ranked results with file paths, relevance scores, and content excerpts

Sync runs on session startup. Files changed mid-session can be picked up with `/knowledge-search index`.

The index is stored at `{cwd}/.pi/knowledge-search/index.json` (project-local; see [Project-local storage](#project-local-storage)).

## Commands

| Command                              | Description                                          |
| ------------------------------------ | ---------------------------------------------------- |
| `/knowledge-search`                  | Show status: dirs, excludes, extensions, engine      |
| `/knowledge-search add <dir>`        | Add directories to the index                         |
| `/knowledge-search exclude <name>`   | Manage excluded directory names (`-<name>` removes)  |
| `/knowledge-search index`            | Incrementally index new/changed files                |
| `/knowledge-search clear`            | Clear all project data; reset settings to defaults  |
| `/knowledge-search on` / `off`       | Toggle per-turn knowledge lookup injection           |
| `/knowledge-search help`             | List all subcommands                                 |
| `/knowledge-overview`                | Force-rebuild and re-inject the vault overview       |

## Performance

Indicative numbers for local ONNX embeddings (nomic-embed-text-v1.5, q8, Apple Silicon, warm model cache) on ~500 markdown files (~20 MB, ~500 chunks):

| Operation                     | Time            |
| ----------------------------- | --------------- |
| Full index build              | ~40s (chunked, batched ONNX) |
| Incremental sync (no changes) | ~15ms           |
| Search query                  | ~300ms          |
| Index file size               | ~5MB            |

First run per machine also downloads the nomic model (~111 MB, shared with pi-local-rag) — `/knowledge-search index` shows a notice before the download starts.

## Project-local storage

Config and index are **project-local by default**: config lives at `{cwd}/.pi/knowledge-search.json` and the index at `{cwd}/.pi/knowledge-search/`, where `{cwd}` is the directory pi was started in. Each project gets its own config and index — no cross-project bleed.

To relocate them elsewhere within a project, add the following to `{project}/.pi/settings.json`:

```jsonc
{
  "pi-knowledge-search": {
    "localPath": ".pi/knowledge-search"   // config.json + index/ under this path
  }
}
```

**Resolution order (highest priority first):**

1. `KNOWLEDGE_SEARCH_CONFIG` / `KNOWLEDGE_SEARCH_INDEX_DIR` env vars
2. `pi-knowledge-search.localPath` in `{cwd}/.pi/settings.json`
3. Project default: `{cwd}/.pi/knowledge-search.json` + `{cwd}/.pi/knowledge-search/`

**Migration from the global config:** versions prior to this change stored config at `~/.pi/knowledge-search.json` and the index at `~/.pi/knowledge-search/`. That location is no longer read — move the files into your project's `.pi/` directory to migrate, or re-run `/knowledge-search add <dir>` in the project.

**What `/knowledge-search clear` resets:** both databases (`index.json` vector index + `kb-fts.db` keyword side-car), the config file, and any `pi-knowledge-search.localPath` override in `{project}/.pi/settings.json` — leaving the project in a fresh-install state. The HuggingFace model cache (`~/.cache/huggingface/transformers`) is machine-wide and shared with pi-local-rag, so it is intentionally not touched.

## License

MIT
