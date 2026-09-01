# pi-knowledge-search

Hybrid **local** search over local files for [pi](https://github.com/badlogic/pi). Indexes directories of text/markdown files using local ONNX vector embeddings (always `nomic-ai/nomic-embed-text-v1.5` — the engine is fixed, not configurable) **and** SQLite FTS5 keyword search, exposes `knowledge_search` + `knowledge_kb_read` tools the LLM can call, and auto-injects a knowledge lookup on every prompt (like pi-local-rag's RAG lookup). Everything runs on your machine — no embedding APIs, no cloud services. Indexing runs on session startup; mid-session file changes are picked up with `/knowledge-search index`.

On session start, injects a folder+keyword overview of the indexed vault as a custom message so the model knows what’s worth searching for before it asks.

## Knowledge lookup

Like pi-local-rag's RAG lookup, every user prompt triggers an automatic **knowledge lookup**: the prompt runs through hybrid search (top 5 chunks, min score 0.1) and the hits are injected as a message right after the prompt — full excerpts for the model, and a green `Knowledge lookup — event_dispatcher.rst:1-8,9-37,44-72, …` summary box for you (red on failure). Each `file:start-end,…` entry lists the exact line ranges in that file where the hits live; a bare number is a single-line hit. Line ranges only appear on entries indexed by the current format (index version 4) — entries from an older-but-compatible index are kept as-is on load (no re-embedding) and fall back to a `file (n)` hit count until the file is next re-indexed.

Injection is automatically enabled whenever the index holds vectors — at session start and after `/knowledge-search index` — so `/knowledge-search off` acts as a per-session kill-switch that the next startup flips back on (`autoInject` in the config).

## How search works

Every query runs against two backends and blends the results exactly like pi-local-rag's hybrid search — `0.4 × BM25 + 0.6 × vector`:

- **Vector cosine similarity** — good for conceptual/fuzzy queries ("how did we handle X"). Raw cosine on the unit-normalized nomic embeddings, clamped at 0.
- **BM25 full-text** via SQLite FTS5 — good for exact matches, proper nouns, error strings, file paths, code identifiers. Raw BM25 scores are min-max normalized across the candidate set and get a 1.5× boost (capped at 1) when the first meaningful query term appears in the file path.

Multi-term queries use implicit AND (space-separated quoted phrases), matching pi-local-rag, so chunks missing a term are excluded rather than diluting the result set. Either backend alone still surfaces relevant hits: if the embedder fails transiently, search falls back to pure BM25; if the FTS side-car is empty, it falls back to pure vector. Indexes predating the fixed engine are re-embedded once on upgrade (see [Engine-signature invalidation](#engine-signature-invalidation)).

## Tools

The extension registers two LLM-facing tools:

| Tool | What it does |
|------|--------------|
| `knowledge_search` | Hybrid vector + BM25 search over indexed files (pi-local-rag's alpha blend: 0.4 × BM25 + 0.6 × vector). Returns passage-level excerpts. |
| `knowledge_kb_read` | Resolve a note reference — `[[wikilink]]`, basename, or relative path — to an indexed file and return its full content. Use when the model knows a note's name but not its full path, instead of running find/grep first. |

`knowledge_kb_read` handles `[[Foo]]`, `[[Foo|alias]]`, `[[Foo#Heading]]`, bare names with or without extension (`Foo`, `Foo.md`), and relative paths (`evergreen/foo`). Multi-match references get a disambiguation prompt instead of guessing.

## Overview injection

On session start, pi-knowledge-search injects a one-shot summary of the indexed vault as a custom message. This gives the model a prior on what's in the knowledge base without having to discover the structure through trial-and-error searches.

The overview is a compact list — one line per configured source dir with its note count:

```markdown
- **~/dir/docs1** — 42 notes
- **~/dir/docs2** — 7 notes
```

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

The first `add` writes the config to `{cwd}/.pi/knowledge-search.json` — project-local, relative to the directory pi was started in. Directories added mid-session are picked up by `/knowledge-search index` without a reload.

### Config file

You can also edit the config file directly:

```json
{
  "dirs": ["~/notes", "~/docs"],
  "excludeDirs": ["node_modules", ".git", ".obsidian", ".trash"],
  "autoInject": true
}
```

All fields except `dirs` are optional — the example shows the defaults for `excludeDirs` and `autoInject`. Omit `fileExtensions` to get the full default list below, or set it explicitly to narrow down (e.g. `[".md", ".txt"]` for a notes-only vault).

`autoInject` (default `true`) controls the per-turn knowledge lookup; it is re-enabled automatically at startup while the index holds vectors.

Default `fileExtensions` mirror the extensions pi-local-rag's nomic model indexes:

```
.md .mdx .txt .rst .html .htm .json .jsonc .yaml .yml .toml .ini .xml .csv .tsv .env .gitignore .dockerfile
```

(.pdf/.docx also go to nomic in pi-local-rag but need extraction libraries and are not indexed here.) Extension matching is case-insensitive; code extensions (`.ts`, `.py`, …) are not in nomic's group — set `fileExtensions` explicitly if you want them.

Files larger than **500 KB** are skipped during scanning (mirroring pi-local-rag's `TEXT_MAX_BYTES`) — this keeps minified bundles, dumped JSON/CSV, and base64 blobs from blowing up read time, chunk count, and embedding wall time. Files that exceed the cap after having been indexed are dropped on the next sync.

> **Migrating from older configs:** legacy `provider` and `dimensions` keys are ignored — the embedding engine is always nomic and not configurable. Vectors previously built by any other engine are removed on the first load after upgrading, and the next sync re-embeds everything with nomic.

### Embedding engine (always nomic)

The embedding engine is fixed: `nomic-ai/nomic-embed-text-v1.5` (768-dim, q8 quantized) via [Transformers.js](https://huggingface.co/docs/transformers.js) local ONNX inference — no API key, no server, no configuration. It uses the model's `search_query:` / `search_document:` task prefixes, mirroring pi-local-rag's text pipeline. Model weights are downloaded once (~111 MB) into a shared HuggingFace cache (`~/.cache/huggingface/transformers` by default, or `PI_RAG_MODEL_CACHE` / `TRANSFORMERS_CACHE` / `HF_HOME`), so pi-knowledge-search and pi-local-rag reuse the same download. `/knowledge-search index` shows a notice before the first download.

Transformers.js pulls in `sharp` (for vision preprocessing); since pi loads pi-local-rag alongside this extension in the same process, `sharp` is pinned to exactly the same version pi-local-rag resolves (0.35.3 → libvips 8.18.3) via npm `overrides`. Loading two different libvips dylibs into one process makes macOS objc emit a duplicate-class warning (`GNotificationCenterDelegate implemented in both …`) that can cause spurious casting failures and mysterious crashes.

### Engine-signature invalidation

Vectors from different engines, models, or dimensionalities are not comparable. The index records the signature of the engine that built it — now the constant `transformers:nomic-ai/nomic-embed-text-v1.5:768`. On load, a mismatch (any index predating the fixed engine, or built by a removed remote provider) removes all existing embeddings and the next sync re-embeds everything with nomic.

### Environment variable overrides

Every config field can be overridden via environment variables. This is useful for CI or when you want different settings per shell session. See [env-vars.md](docs/env-vars.md) for the full list.

## How it works

1. On session start, loads the index from disk and incrementally syncs — only re-embeds new or modified files. Older-but-compatible indexes (e.g. a pre-line-tracking version 3) are adopted as-is without a full re-embed; only an embedding-engine change (or a pre-chunk flat index) forces re-embedding everything. Files at or above 500 KB are skipped (see [Setup](#setup))
2. Registers two LLM-facing tools: `knowledge_search` for hybrid ranked search and `knowledge_kb_read` for resolving a note reference to a full file (see [Tools](#tools))
3. Before every agent turn, runs an automatic knowledge lookup on the prompt and injects the top hits as a message right after it (see [Knowledge lookup](#knowledge-lookup))
4. Returns ranked results with file paths, relevance scores, content excerpts, and the exact line ranges of each hit

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

Chunks are embedded in batches of **16** — one padded ONNX forward pass per batch, mirroring pi-local-rag's `BATCH_SIZE` — so the progress bar ticks per pass instead of every 50 chunks.

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
