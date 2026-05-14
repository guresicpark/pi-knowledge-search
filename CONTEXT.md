# pi-knowledge-search

Hybrid local-file search for pi. Indexes text/markdown with both vector embeddings and SQLite FTS5 keyword search, fuses results via RRF, and exposes a `knowledge_search` tool to the LLM.

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
The fusion strategy. Both backends produce a ranked list for a query; each doc's final score is `Σ 1/(k + rank)` across lists, with `k = 60`. Docs both backends agree on get boosted; docs either backend finds alone still surface.

**Hybrid search**:
The default search mode — runs vector and FTS in parallel and fuses via RRF. Falls back to pure FTS if the embedder errors, pure vector if the FTS side-car is empty.

**Embedder**:
The provider-agnostic interface for turning text into vectors. Providers: **OpenAI**, **OpenAI-compatible** (local/self-hosted), **Bedrock**, **Ollama**. Chosen per config at startup.

**Provider**:
A concrete Embedder implementation. Swappable via config.

**Sync worker**:
Background watcher that re-indexes files on change. Keeps both the Vector index and the FTS5 side-car in sync.

**Backfill**:
One-time migration that populates the FTS5 side-car from an existing Vector index without re-embedding anything. Runs on first load after upgrade.

**`knowledge_search`**:
The public tool exposed to the LLM — takes a query, returns ranked results. This is the only thing most callers see.

## Relationships

- A file produces one or more **Chunks** via the **Chunker**.
- Each **Chunk** is stored in both the **Vector index** and the **FTS5 side-car**, keyed by `${absPath}#${chunkIndex}`.
- **Hybrid search** fans a query out to both indices, then merges with **RRF (k=60)**.
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
