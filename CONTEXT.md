# pi-knowledge-search

Hybrid local-file search for pi. Indexes text/markdown **and source code** with both vector embeddings (dual spaces: nomic for prose, jina for code) and SQLite FTS5 keyword search, blends results with an alpha-weighted score, and exposes a `knowledge_search` tool to the LLM.

## Language

**Chunk**:
The unit of indexed text. A file is split into one or more Chunks by the **Chunker** (typically on heading boundaries). Each Chunk is keyed by `${absPath}#${chunkIndex}` — that key is the primary identifier across both the vector and FTS sides.
_Avoid_: segment, block, passage.

**Chunker**:
Module that splits a file into Chunks. Preserves heading context (each Chunk knows which section heading it falls under).

**Vector index**:
The embedding-backed similarity store. Lives alongside the FTS index — same directory, different storage.
_Avoid_: embedding store (too generic).

**FTS5 side-car**:
A SQLite FTS5 database that sits *next to* the vector index and holds the same Chunks indexed for keyword search. "Side-car" is deliberate — it rides alongside the primary vector store, not instead of it. Can be **Backfilled** from an existing vector index without re-embedding.
_Avoid_: keyword index (too generic), secondary index.

**RRF** (Reciprocal Rank Fusion):
Historical term for the fusion strategy. Replaced by the **alpha blend**: both backends produce a ranked list for a query and each chunk's final score is `0.4 × normalized BM25 + 0.6 × cosine` (alpha = 0.4, like pi-local-rag).
_Avoid_: RRF, rank fusion (except when discussing the removed design).

**Result quota**:
The per-group slot count in a hybrid result list. The **total** is 7 slots when both Embed groups store vectors, else 5 — capped by the caller's limit. Slots are **split in proportion to each space's stored vector count** (integer quotas, min 1 per group, code group first). A group that can't fill its quota yields the slack to the other group.

**Relevance floor**:
The minimum hybrid score per Embed group — **0.35 for jina-code hits, 0.4 for nomic/BM25 hits**. Anything below is treated as unrelated and omitted.

**Hybrid search**:
The default search mode — runs the vector spaces and FTS in parallel and blends with `0.4 × BM25 + 0.6 × vector`. Falls back to pure FTS if the embedder errors, pure vector if the FTS side-car is empty.

**Embed group**:
Which embedding model a file's Chunks belong to: **code** (jina) or **text** (nomic). Decided by file extension (`codeExtensions` config, defaulting to pi-local-rag's code list). Each group is a separate vector space with its own query embedding, relevance floor, and result quota.

**Embedder**:
The local ONNX (Transformers.js) engine turning text into vectors — one pipeline per Embed group: **nomic-embed-text-v1.5** for prose, **jina-embeddings-v2-base-code** for source code. Not configurable.

**Provider**:
Historical term for a swappable Embedder backend. Removed — the engine is fixed (local nomic + jina).

**Sync worker**:
Background watcher that re-indexes files on change. Keeps both the Vector index and the FTS5 side-car in sync.

**Backfill**:
One-time migration that populates the FTS5 side-car from an existing Vector index without re-embedding anything. Runs on first load after upgrade.

**`knowledge_search`**:
The public tool exposed to the LLM — takes a query, returns ranked results. This is the only thing most callers see.

## Relationships

- A file produces one or more **Chunks** via the **Chunker**.
- Each **Chunk** is stored in both the **Vector index** and the **FTS5 side-car**, keyed by `${absPath}#${chunkIndex}`.
- **Hybrid search** fans a query out to both indices, then merges with the **alpha blend** (`0.4 × BM25 + 0.6 × cosine`).
- Each **Embed group** is its own vector space: its query is embedded only when that space holds vectors, and result slots are allocated by the **Result quota** split under per-group **Relevance floors**.
- The **Embedder** is only used at write time (and for the query-side vector) — the **FTS5 side-car** doesn't use it.
- The **Sync worker** keeps both indices consistent as files change.
- **Backfill** populates the FTS5 side-car from the Vector index without re-running the **Embedder**.

## Flagged ambiguities

- **"Search"** alone can mean Vector, FTS, or Hybrid — always qualify.
- **"Index"** is ambiguous between Vector and FTS — always say "Vector index" or "FTS5 side-car."
- **Zero-config FTS-only mode** exists for users without an embedding API — pure BM25, no vector side. This is a distinct configuration, not the fallback behaviour.

## Example dialogue

> **Sam:** "Why is a query for `ERR_REQUIRE_CYCLE_MODULE` not returning anything?"
> **Agent:** "Likely the **FTS5 side-car** is empty for that directory. **Vector search** treats the error code as noise; **FTS** handles it exactly. Check that the **Sync worker** has re-indexed the directory, or run a **Backfill** if the index predates FTS."
