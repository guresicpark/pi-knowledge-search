import * as fs from "node:fs";
import * as path from "node:path";
import Assembler from "stream-json/assembler.js";
import makeParser from "stream-json/index.js";
import type { Config } from "./config.js";
import { classifyFileGroup } from "./config.js";
import { EMBEDDING_MODEL, EMBEDDING_DIMENSIONS, type EmbedGroup } from "./embedder.js";
import type { Embedder } from "./embedder.js";
import { chunkMarkdown, type Chunk } from "./chunker.js";
import { FtsChunkIndex, type FtsChunk } from "./fts-index.js";

/**
 * Which engine surfaced a chunk as a candidate: BM25 (FTS5 keyword),
 * nomic (prose vector space), or jina-code (code vector space) — mirrors
 * pi-local-rag's RetrievalSource.
 */
export type RetrievalSource = "bm25" | "nomic" | "jina-code";

interface IndexEntry {
  /** Relative path from its source directory root */
  relPath: string;
  /** Which source directory this belongs to */
  sourceDir: string;
  /** File mtime (ms) at time of indexing */
  mtime: number;
  /** Embedding vector (empty array when running FTS-only without an embedder) */
  vector: number[];
  /**
   * Embedding group the vector was produced with (`"code"` = jina,
   * `"text"` = nomic). Optional only for entries from pre-dual-model
   * indexes — those are derived from the file extension on load.
   */
  group?: EmbedGroup;
  /** This chunk's content for excerpt display */
  excerpt: string;
  /** Section heading this chunk falls under */
  heading: string;
  /** Chunk index (0, 1, 2... for multi-chunk files) */
  chunkIndex: number;
  /** 0-indexed line in the source file where this chunk starts */
  startLine: number;
  /** 0-indexed line in the source file where this chunk ends (inclusive) */
  endLine: number;
}

interface IndexData {
  version: number;
  dimensions: number;
  /**
   * Signature of the embedding engine (`type:model:dimensions`) that produced
   * the stored vectors. Null when unknown (pre-signature indexes, FTS-only
   * installs). A mismatch on load removes all existing embeddings and forces
   * a full re-embed — vectors from different engines/models are not
   * comparable.
   */
  embeddingModel: string | null;
  entries: Record<string, IndexEntry>; // keyed by "absPath#chunkIndex"
}

export interface SearchResult {
  /** Absolute file path */
  path: string;
  /** Cosine similarity score (0-1) */
  score: number;
  /** Content excerpt (the matched chunk) */
  excerpt: string;
  /** Section heading for context */
  heading: string;
  /**
   * Number of chunks in this file that matched the query, regardless of
   * deduplication or the limit. Lets callers show "file (N hits)" even
   * though only the best chunk per file is returned.
   */
  matches: number;
  /**
   * 1-indexed inclusive line ranges in the source file of every matching
   * chunk, regardless of deduplication or the limit. Empty when the stored
   * entry predates line-range indexing (fall back to `matches`).
   */
  lineRanges: Array<[number, number]>;
  /**
   * Which backend drove this hit's ranking in the hybrid blend — `"vector"`
   * when the cosine term contributed at least as much to the blended score
   * as the normalized BM25 term, `"bm25"` otherwise. Always `"bm25"` in the
   * pure-keyword fallback (no vectors). Undefined outside hybrid search
   * (e.g. plain `vectorSearch()`).
   */
  source?: "vector" | "bm25";
  /**
   * Every engine that surfaced this hit as a candidate — `"bm25"` (FTS5
   * side-car), `"nomic"` (prose vector space), `"jina-code"` (code vector
   * space). A hit found by several engines lists them all, mirroring
   * pi-local-rag's per-engine provenance. Empty outside hybrid search
   * (e.g. plain `vectorSearch()`).
   */
  sources: RetrievalSource[];
  /**
   * Embedding group the hit's chunk was indexed with — `"code"` hits are
   * embedded/searched in the jina space, `"text"` in the nomic space.
   * Derived from the file extension for entries predating the field.
   */
  group: EmbedGroup;
}

// ---------------------------------------------------------------------------
// Result quotas — pi-local-rag's ratio-based split across embedding spaces
// ---------------------------------------------------------------------------

/**
 * Total number of result slots when at most one embedding space has
 * stored vectors (or a single group fills the result alone). `limit` can
 * only shrink this, never grow it.
 */
export const RESULT_TOTAL_QUOTA = 5;

/**
 * Result-slot total for stores with vectors in both embedding spaces —
 * a mixed corpus has two groups to fill, so more hits feed the
 * ratio-based quota split. Same rule as RESULT_TOTAL_QUOTA otherwise:
 * `limit` can only shrink it.
 */
export const RESULT_TOTAL_QUOTA_DUAL_SPACE = 7;

/**
 * Minimum hybrid score per embedding space — anything below is treated as
 * unrelated and omitted, so a query with one related file returns just that
 * file instead of padding the list with noise. Calibrated on real indexes
 * (see pi-local-rag's MIN_HYBRID_SCORE_CODE/TEXT):
 *
 * - jina-code space: unrelated queries top out at ~0.28-0.33, related code
 *   hits start at ~0.38 (vague) and run to 0.63+ (keyword matches)
 * - nomic space + BM25-only hits: unrelated tops out at ~0.31-0.39, related
 *   starts at ~0.65. 0.4 is also exactly the ceiling of a pure keyword-only
 *   match (alpha × bm25=1 + (1-alpha) × cos~0), so those always survive.
 */
export const MIN_HYBRID_SCORE_CODE = 0.35;
export const MIN_HYBRID_SCORE_TEXT = 0.4;

/**
 * Split `total` result slots between the code and prose groups in
 * proportion to the store's stored vector counts (the corpus
 * composition ratio). Integer quotas that sum exactly to `total`, each
 * at least 1 — except `total < 2`, where both minimums can't hold and
 * the single slot goes to the code group (code-first policy).
 */
export function splitResultQuotas(
  total: number,
  codeVectorCount: number,
  proseVectorCount: number,
): { codeQuota: number; proseQuota: number } {
  if (total < 2) return { codeQuota: total, proseQuota: 0 };
  const vectorTotal = codeVectorCount + proseVectorCount;
  const codeShare = vectorTotal > 0 ? codeVectorCount / vectorTotal : 0.5;
  const codeQuota = Math.min(total - 1, Math.max(1, Math.round(total * codeShare)));
  return { codeQuota, proseQuota: total - codeQuota };
}

/**
 * A file hit by the FTS5 side-car (BM25 keyword search) before fusion — the
 * keyword-side counterpart of `SearchResult`, without content. Lets callers
 * show what pure keyword matching found alongside the fused ranking.
 */
export interface Bm25FileHit {
  /** Absolute file path */
  path: string;
  /** 1-indexed inclusive line ranges of the BM25-matched chunks, sorted. */
  lineRanges: Array<[number, number]>;
}

/** Result of `searchWithBm25` — fused hybrid hits plus the raw BM25 file hits. */
export interface HybridSearchWithBm25 {
  /** Fused (BM25 + vector) results, same as `hybridSearch()` returns. */
  results: SearchResult[];
  /** Files matched by the FTS5 side-car, best keyword match first. */
  bm25Files: Bm25FileHit[];
}

/**
 * Progress events emitted by sync() for UI rendering.
 *
 * - scan: directory scan finished — reports how many files need
 *   (re)embedding, how many are unchanged, the total chunk count, and the
 *   chunk split per embedding group (code → jina, text → nomic)
 * - embed: one embed batch completed — done/total chunks plus the file
 *   the latest batch ended in (best effort), and per-group cumulative
 *   progress for the per-model progress lines
 * - save: the index is being persisted to disk
 */
export type SyncProgress =
  | {
      phase: "scan";
      filesToProcess: number;
      unchanged: number;
      totalChunks: number;
      chunksByGroup: { code: number; text: number };
    }
  | {
      phase: "embed";
      done: number;
      total: number;
      currentFile?: string;
      doneByGroup: { code: number; text: number };
      totalByGroup: { code: number; text: number };
    }
  | { phase: "save" };

const INDEX_VERSION = 5; // Bumped from 4 for per-chunk embedding groups (nomic/jina dual space)

/**
 * Signature of the pre-dual-model engine (nomic-only). Indexes built by it
 * hold valid nomic vectors for text-group files — only code-group files
 * were embedded with the wrong model. On load they are adopted and the
 * code-group entries dropped (re-embedded by sync), instead of discarding
 * every vector.
 */
const LEGACY_TEXT_ONLY_SIGNATURE = `transformers:${EMBEDDING_MODEL}:${EMBEDDING_DIMENSIONS}`;

/**
 * Oldest on-disk index format we can adopt without re-embedding. v3
 * introduced the per-chunk key format (`${absPath}#${chunkIndex}`) and the
 * current IndexEntry shape; v4 only added the optional startLine/endLine
 * line-range fields. Indexes at or above this version are structurally
 * compatible — their vectors are still valid, so they load as-is. Older
 * (v1/v2) flat whole-file indexes use bare absPath keys and a different
 * entry shape, so they still force a full rebuild.
 */
const MIN_LOADABLE_INDEX_VERSION = 3;

const MAX_EXCERPT_LENGTH = 3500; // Safety cap for stored excerpts

/**
 * Plain-text files larger than this are skipped during scanning — mirrors
 * pi-local-rag's TEXT_MAX_BYTES. Keeps pathological files (minified bundles,
 * dumped JSON/CSV, base64 blobs) from blowing up read time, chunk count, and
 * the embedding batch's padded wall time.
 */
const TEXT_MAX_BYTES = 500_000;

export class KnowledgeIndex {
  private config: Config;
  /** Embedder may be null in FTS-only mode (no provider configured). */
  private embedder: Embedder | null;
  private data: IndexData;
  private dirty = false;
  private saveTimer: ReturnType<typeof setTimeout> | null = null;
  private fts: FtsChunkIndex;

  constructor(config: Config, embedder: Embedder | null) {
    this.config = config;
    this.embedder = embedder;
    this.data = {
      version: INDEX_VERSION,
      dimensions: config.dimensions,
      embeddingModel: config.modelSignature,
      entries: {},
    };
    this.fts = new FtsChunkIndex(config.indexDir);
  }

  /** True when no embedder is configured — search runs pure BM25. */
  get isFtsOnly(): boolean {
    return this.embedder === null;
  }

  size(): number {
    // Count unique file paths (not chunks)
    const paths = new Set<string>();
    for (const entry of Object.values(this.data.entries)) {
      paths.add(`${entry.sourceDir}/${entry.relPath}`);
    }
    return paths.size;
  }

  chunkCount(): number {
    return Object.keys(this.data.entries).length;
  }

  /**
   * Aggregate all chunks into a per-file view: one entry per indexed file with
   * the merged list of section headings found across its chunks. Used by the
   * overview builder and the knowledge_kb_read resolver — both want file-level data, not
   * chunk-level.
   */
  listFiles(): Array<{
    absPath: string;
    relPath: string;
    sourceDir: string;
    headings: string[];
    /** Embedding group of the file's chunks (all chunks of a file share it). */
    group: EmbedGroup;
  }> {
    const byPath = new Map<
      string,
      { absPath: string; relPath: string; sourceDir: string; headings: string[]; group: EmbedGroup }
    >();
    for (const [key, entry] of Object.entries(this.data.entries)) {
      const absPath = this.absPathFromKey(key);
      let agg = byPath.get(absPath);
      if (!agg) {
        agg = {
          absPath,
          relPath: entry.relPath,
          sourceDir: entry.sourceDir,
          headings: [],
          group: this.entryGroup(key, entry),
        };
        byPath.set(absPath, agg);
      }
      if (entry.heading && entry.heading !== "intro" && !agg.headings.includes(entry.heading)) {
        agg.headings.push(entry.heading);
      }
    }
    return Array.from(byPath.values());
  }

  /**
   * Threshold above which the load/save paths switch to streaming. V8's
   * string length limit is ~512MB (2^29 - 24 bytes on 64-bit). A single
   * call to `readFileSync(path, "utf-8")` or `JSON.stringify(hugeObject)`
   * throws `RangeError: Invalid string length` once that limit is hit.
   *
   * Below this threshold we use the straightforward sync paths since they
   * are an order of magnitude faster. Above it we switch to streaming.
   *
   * Set to 256MB to give a generous safety margin below the hard cliff.
   */
  private static readonly STREAMING_THRESHOLD_BYTES = 256 * 1024 * 1024;

  /**
   * Load the index from disk.
   *
   * Uses a fast sync path (`readFileSync` + `JSON.parse`) for normal-sized
   * indexes and automatically falls back to a streaming reader for files
   * large enough to risk V8's string length limit (`RangeError: Invalid
   * string length`).
   *
   * If the file is missing, corrupt, or from an incompatible version, falls
   * back to an empty index and returns — callers will then trigger a full
   * re-index. Never throws.
   */
  async load(): Promise<void> {
    this.fts.load();

    // Yield before the (potentially 99 MB) JSON parse — keeps the heavy
    // I/O off the same microtask drain as session_start /
    // before_agent_start, so pi's outbound model HTTP request fires
    // before we touch the index file.
    await new Promise<void>((r) => setImmediate(r));

    const indexFile = path.join(this.config.indexDir, "index.json");
    if (fs.existsSync(indexFile)) {
      try {
        let parsed: IndexData | null = null;
        const size = fs.statSync(indexFile).size;
        if (size >= KnowledgeIndex.STREAMING_THRESHOLD_BYTES) {
          parsed = await this.streamLoadJson(indexFile);
        } else {
          const raw = fs.readFileSync(indexFile, "utf-8");
          parsed = JSON.parse(raw) as IndexData;
        }
        // Accept the on-disk index when:
        //  - version is a known-compatible chunked format (current or an
        //    older one — v3/v4 differ only by optional fields, so an older
        //    index is migrated, not re-embedded) AND
        //  - dimensions match OR we're in FTS-only mode (dimensions are a
        //    vector-only concern; FTS-only installs should never invalidate
        //    a perfectly good entry map over them) AND
        //  - the embedding-engine signature matches OR we're in FTS-only
        //    mode. Vectors built by a different engine/model are not
        //    comparable — drop them and re-embed everything. The one
        //    exception is the legacy nomic-only signature: its text-group
        //    vectors are still valid under the current dual-model engine,
        //    so the index is adopted with only code-group entries removed
        //    (they were embedded with the wrong model).
        const dimsOk =
          this.isFtsOnly || parsed?.dimensions === this.config.dimensions;
        const sigOk =
          this.isFtsOnly ||
          parsed?.embeddingModel === this.config.modelSignature ||
          parsed?.embeddingModel === LEGACY_TEXT_ONLY_SIGNATURE;
        if (
          parsed &&
          parsed.version >= MIN_LOADABLE_INDEX_VERSION &&
          parsed.version <= INDEX_VERSION &&
          dimsOk &&
          sigOk
        ) {
          // Normalize the version so the next save persists the current
          // format. Entries missing the optional startLine/endLine fields
          // fall back to the per-file match count in search — no re-embed
          // needed to keep their vectors.
          this.data = { ...parsed, version: INDEX_VERSION };
          if (!this.isFtsOnly && parsed.embeddingModel === LEGACY_TEXT_ONLY_SIGNATURE) {
            // Migrate nomic-only → dual-model: drop code-group entries
            // (their vectors came from nomic and must be re-embedded with
            // jina); text-group vectors stay valid. Record the current
            // signature so the migration runs only once.
            const staleCodePaths = new Set<string>();
            for (const key of Object.keys(this.data.entries)) {
              const entry = this.data.entries[key];
              if (this.entryGroup(key, entry) === "code") {
                staleCodePaths.add(this.absPathFromKey(key));
              }
            }
            for (const key of Object.keys(this.data.entries)) {
              if (staleCodePaths.has(this.absPathFromKey(key))) {
                delete this.data.entries[key];
              }
            }
            this.data.embeddingModel = this.config.modelSignature;
            // The FTS side-car is loaded by this point (top of load()) —
            // drop the stale code files' keyword rows so sync() re-indexes
            // them cleanly.
            for (const absPath of staleCodePaths) {
              this.fts.deleteByAbsPath(absPath);
            }
          }
        }
        // Version (older than v3), dimension, or signature mismatch → keep
        // fresh data, caller will re-index.
      } catch {
        // Corrupt file / partial write / IO error → fresh index.
      }
    }

    // Backfill FTS side-car from the vector index when it's empty but the
    // JSON index is populated. Handles first-run upgrades from pre-hybrid
    // versions without forcing a full re-embed.
    const chunkCount = this.chunkCount();
    if (chunkCount > 0 && this.fts.count() === 0) {
      this.rebuildFtsFromEntries();
    }
  }

  /**
   * Repopulate the FTS side-car from the in-memory JSON entries. Used on
   * first load after upgrading to hybrid search so existing users don't
   * pay the cost of re-embedding just to get keyword search.
   */
  private rebuildFtsFromEntries(): void {
    const chunks: FtsChunk[] = [];
    for (const [key, entry] of Object.entries(this.data.entries)) {
      chunks.push({
        key,
        absPath: this.absPathFromKey(key),
        relPath: entry.relPath,
        sourceDir: entry.sourceDir,
        heading: entry.heading,
        content: entry.excerpt,
        chunkIndex: entry.chunkIndex,
        mtime: entry.mtime,
      });
    }
    if (chunks.length > 0) this.fts.upsertMany(chunks);
  }

  private streamLoadJson(file: string): Promise<IndexData | null> {
    return new Promise((resolve, reject) => {
      const stream = fs.createReadStream(file, { highWaterMark: 256 * 1024 });
      const parser = makeParser();

      let settled = false;
      const settle = (ok: () => void, err?: (e: Error) => void) => {
        if (settled) return;
        settled = true;
        if (err) err(new Error("assembler failed"));
        else ok();
      };

      // stream-json 3.x replaced the 2.x 'done' event with the onDone option.
      // The assembler handle itself is not needed — errors surface via the
      // parser stream, completion via onDone.
      Assembler.connectTo<IndexData>(parser, {
        onDone: (asm) => settle(() => resolve(asm.current)),
      });
      stream.on("error", (e) => settle(() => resolve(null), () => reject(e)));
      parser.on("error", (e) => settle(() => resolve(null), () => reject(e)));

      stream.pipe(parser);
    });
  }


  /**
   * Persist the index to disk.
   *
   * Fast path: `JSON.stringify` + `writeFile`, wrapped in an atomic rename
   * from `index.json.tmp`. This handles all normal-sized indexes in one shot.
   *
   * Fallback path: if `JSON.stringify` throws `RangeError: Invalid string
   * length` (V8's ~512MB string limit), fall back to streaming the JSON out
   * block by block via `createWriteStream`. This path never materialises the
   * full serialised form as a single string.
   *
   * Either way the write is atomic: content goes to `index.json.tmp` first,
   * then renamed over `index.json` once fully flushed. A crash mid-write
   * leaves the previous `index.json` intact.
   */
  private async save(): Promise<void> {
    fs.mkdirSync(this.config.indexDir, { recursive: true });
    const finalFile = path.join(this.config.indexDir, "index.json");
    const tmpFile = finalFile + ".tmp";

    try {
      let serialised: string;
      try {
        serialised = JSON.stringify(this.data);
      } catch (err) {
        if (err instanceof RangeError) {
          await this.saveStreaming(tmpFile);
          await fs.promises.rename(tmpFile, finalFile);
          this.dirty = false;
          return;
        }
        throw err;
      }
      await fs.promises.writeFile(tmpFile, serialised);
      await fs.promises.rename(tmpFile, finalFile);
      this.dirty = false;
    } catch (err) {
      try {
        fs.unlinkSync(tmpFile);
      } catch {
        // best-effort cleanup
      }
      throw err;
    }
  }

  /**
   * Streaming fallback used when the index is too big for `JSON.stringify`
   * to produce a single string. Writes key-by-key through a write stream so
   * no intermediate giant string is ever materialised.
   */
  private async saveStreaming(tmpFile: string): Promise<void> {
    const stream = fs.createWriteStream(tmpFile);
    let streamError: Error | null = null;
    stream.once("error", (err) => {
      streamError = err;
    });

    const write = (chunk: string): Promise<void> =>
      new Promise((resolve, reject) => {
        if (streamError) {
          reject(streamError);
          return;
        }
        if (stream.write(chunk)) {
          resolve();
        } else {
          stream.once("drain", () => (streamError ? reject(streamError) : resolve()));
        }
      });

    try {
      await write(
        `{"version":${JSON.stringify(this.data.version)},` +
          `"dimensions":${JSON.stringify(this.data.dimensions)},` +
          `"embeddingModel":${JSON.stringify(this.data.embeddingModel ?? null)},` +
          `"entries":{`
      );
      let first = true;
      for (const key of Object.keys(this.data.entries)) {
        const entry = this.data.entries[key];
        const prefix = first ? "" : ",";
        first = false;
        await write(`${prefix}${JSON.stringify(key)}:${JSON.stringify(entry)}`);
      }
      await write("}}");
    } catch (err) {
      stream.destroy();
      throw err;
    }

    await new Promise<void>((resolve, reject) => {
      stream.end((err?: Error | null) => (err ? reject(err) : resolve()));
    });
  }

  scheduleSave(): void {
    if (this.saveTimer) return;
    this.dirty = true;
    this.saveTimer = setTimeout(() => {
      this.saveTimer = null;
      if (this.dirty) {
        void this.save().catch((err) => {
          console.error(`knowledge-search: scheduled save failed: ${(err as Error).message}`);
        });
      }
    }, 5000);
  }

  /**
   * Build the entry key for a file chunk.
   */
  private entryKey(absPath: string, chunkIndex: number): string {
    return `${absPath}#${chunkIndex}`;
  }

  /**
   * Get the absolute path from an entry key (strip #chunkIndex).
   */
  private absPathFromKey(key: string): string {
    const hashIdx = key.lastIndexOf("#");
    return hashIdx >= 0 ? key.slice(0, hashIdx) : key;
  }

  /**
   * Embedding group an entry belongs to — the group recorded at index time,
   * or derived from the file extension for entries predating the field
   * (all-text legacy indexes; those entries' code-group counterparts were
   * dropped on load).
   */
  private entryGroup(key: string, entry: IndexEntry): EmbedGroup {
    return entry.group ?? classifyFileGroup(this.absPathFromKey(key), this.config.codeExtensions);
  }

  /**
   * Stored vector counts per embedding space (entries with an actual
   * vector). Drives which spaces get a query embedding and the ratio-based
   * result quota split — pi-local-rag's per-table embeddedCount.
   */
  vectorCountsByGroup(): { code: number; text: number } {
    let code = 0;
    let text = 0;
    for (const [key, entry] of Object.entries(this.data.entries)) {
      if (!entry.vector || entry.vector.length === 0) continue;
      if (this.entryGroup(key, entry) === "code") code += 1;
      else text += 1;
    }
    return { code, text };
  }

  /**
   * Remove all chunks for a given absolute file path from both the vector
   * store and the FTS side-car.
   */
  private removeAllChunks(absPath: string): number {
    const prefix = absPath + "#";
    const toRemove: string[] = [];
    for (const key of Object.keys(this.data.entries)) {
      if (key.startsWith(prefix)) {
        toRemove.push(key);
      }
    }
    for (const key of toRemove) {
      delete this.data.entries[key];
    }
    // Always clear FTS rows for this path too — FTS may hold entries even
    // when the vector side doesn't (e.g. if a previous embed batch failed).
    try {
      this.fts.deleteByAbsPath(absPath);
    } catch {
      // FTS not loaded yet — nothing to remove.
    }
    return toRemove.length;
  }

  /**
   * Prepare embedding text for a chunk, per embedding group:
   *
   * - text (nomic): title/heading context line, as before.
   * - code (jina): the file basename as a context line — pi-local-rag's
   *   file-context scheme. jina-code was trained on code-with-context
   *   pairs (docstring/question → code); a bare slice loses its file
   *   identity, and the basename anchors filename-oriented queries without
   *   touching the stored chunk content or FTS text.
   */
  private chunkEmbedText(group: EmbedGroup, relPath: string, heading: string, chunkText: string): string {
    if (group === "code") {
      return `${path.basename(relPath)}\n${chunkText}`;
    }
    const title = relPath.replace(/\.[^.]+$/, "").replace(/\//g, " > ");
    const sectionContext = heading && heading !== "intro" ? ` > ${heading}` : "";
    return `Title: ${title}${sectionContext}\n\n${chunkText}`;
  }

  /**
   * 0-indexed inclusive line range a chunk occupies in its source file,
   * as tracked by the chunker.
   */
  private chunkLineRange(chunk: Chunk): { startLine: number; endLine: number } {
    return {
      startLine: chunk.startLine,
      endLine: Math.max(chunk.endLine ?? chunk.startLine, chunk.startLine),
    };
  }

  /**
   * Scan all configured directories, find new/changed/removed files, update index.
   */
  async sync(
    opts?: { onProgress?: (progress: SyncProgress) => void }
  ): Promise<{ added: number; updated: number; removed: number }> {
    const allFiles = this.scanAllFiles();
    const currentPaths = new Set(allFiles.map((f) => f.absPath));

    // Remove entries for files that no longer exist
    let removed = 0;
    const seenRemoved = new Set<string>();
    for (const key of Object.keys(this.data.entries)) {
      const absPath = this.absPathFromKey(key);
      if (!currentPaths.has(absPath) && !seenRemoved.has(absPath)) {
        seenRemoved.add(absPath);
        removed += 1;
        this.removeAllChunks(absPath);
      }
    }

    // Find new or updated files
    const toProcess: {
      absPath: string;
      relPath: string;
      sourceDir: string;
      mtime: number;
      content: string;
      chunks: Chunk[];
    }[] = [];

    for (const file of allFiles) {
      // Check if any chunk exists for this file with current mtime
      const existingKey = this.entryKey(file.absPath, 0);
      const existing = this.data.entries[existingKey];
      if (existing && existing.mtime >= file.mtime) continue;

      const content = this.readFileContent(file.absPath);
      if (!content || content.trim().length <= 20) continue;

      const chunks = chunkMarkdown(content);
      if (chunks.length === 0) continue;

      toProcess.push({ ...file, content, chunks });
    }

    let added = 0;
    let updated = 0;
    const report = opts?.onProgress;

    if (toProcess.length > 0) {
      // Flatten all chunks for batch embedding, tracking each chunk's
      // embedding group (code files → jina, everything else → nomic).
      const allChunkTexts: string[] = [];
      const allChunkGroups: EmbedGroup[] = [];
      const chunkMeta: { fileIdx: number; chunkIdx: number }[] = [];

      for (let fi = 0; fi < toProcess.length; fi++) {
        const file = toProcess[fi];
        const group = classifyFileGroup(file.absPath, this.config.codeExtensions);
        for (let ci = 0; ci < file.chunks.length; ci++) {
          const chunk = file.chunks[ci];
          allChunkTexts.push(this.chunkEmbedText(group, file.relPath, chunk.heading, chunk.text));
          allChunkGroups.push(group);
          chunkMeta.push({ fileIdx: fi, chunkIdx: ci });
        }
      }

      const chunksByGroup = { code: 0, text: 0 };
      for (const group of allChunkGroups) chunksByGroup[group] += 1;

      report?.({
        phase: "scan",
        filesToProcess: toProcess.length,
        // Files scanned that need no reprocessing. `removed` files are NOT
        // subtracted — they were deleted from disk, so the fresh scan never
        // saw them; subtracting them here produced negative "unchanged" counts.
        unchanged: allFiles.length - toProcess.length,
        totalChunks: allChunkTexts.length,
        chunksByGroup,
      });

      // Embed in batches — skipped entirely in FTS-only mode. Each group is
      // embedded with its own model (text → nomic, code → jina), mirroring
      // pi-local-rag's per-group pipelines; progress reports cumulative
      // per-group counts so the UI can render one line per model.
      const allVectors: (number[] | null)[] = new Array(allChunkTexts.length).fill(null);
      if (this.embedder) {
        // One chunk per sync step matches pi-local-rag's BATCH_SIZE (16 texts
        // per ONNX forward pass), so each step is a single padded forward pass
        // and progress ticks per pass instead of every 50 chunks.
        const BATCH_SIZE = 16;
        const doneByGroup = { code: 0, text: 0 };
        const emitEmbedProgress = (upto: number) => {
          const lastMeta = chunkMeta[Math.min(upto, chunkMeta.length) - 1];
          report?.({
            phase: "embed",
            done: Math.min(upto, allChunkTexts.length),
            total: allChunkTexts.length,
            currentFile: lastMeta ? toProcess[lastMeta.fileIdx].relPath : undefined,
            doneByGroup: { ...doneByGroup },
            totalByGroup: { ...chunksByGroup },
          });
        };
        for (const group of ["text", "code"] as const) {
          const groupIndexes: number[] = [];
          for (let i = 0; i < allChunkGroups.length; i++) {
            if (allChunkGroups[i] === group) groupIndexes.push(i);
          }
          for (let g = 0; g < groupIndexes.length; g += BATCH_SIZE) {
            const batchIndexes = groupIndexes.slice(g, g + BATCH_SIZE);
            const batchTexts = batchIndexes.map((i) => allChunkTexts[i]);
            const vectors = await this.embedder.embedBatch(batchTexts, group);
            for (let j = 0; j < batchIndexes.length; j++) {
              allVectors[batchIndexes[j]] = vectors[j];
              if (vectors[j]) doneByGroup[group] += 1;
            }
            const lastIdx = batchIndexes[batchIndexes.length - 1];
            emitEmbedProgress(lastIdx + 1);
          }
        }
        // Cover a group with zero chunks so both model lines always render.
        emitEmbedProgress(allChunkTexts.length);
      } else {
        // FTS-only: no embedding work, jump the bar straight to complete so
        // the UI doesn't sit at 0% through the store loop.
        const lastMeta = chunkMeta[chunkMeta.length - 1];
        report?.({
          phase: "embed",
          done: allChunkTexts.length,
          total: allChunkTexts.length,
          currentFile: lastMeta ? toProcess[lastMeta.fileIdx].relPath : undefined,
          doneByGroup: { ...chunksByGroup },
          totalByGroup: { ...chunksByGroup },
        });
      }

      // Store results, grouped by file
      const processedFiles = new Set<number>();

      for (let i = 0; i < chunkMeta.length; i++) {
        const { fileIdx, chunkIdx } = chunkMeta[i];
        const vector = allVectors[i];
        // In FTS-only mode vector is null — store an empty array placeholder
        // so the rest of the pipeline keeps its shape.
        const storedVector: number[] = vector ?? [];
        // Skip if we expected a vector (embedder was present) and got none
        // — an embed failure shouldn't produce a silently vectorless entry.
        if (this.embedder && !vector) continue;

        const file = toProcess[fileIdx];

        // On first chunk of a file, remove old chunks and track add/update
        if (!processedFiles.has(fileIdx)) {
          processedFiles.add(fileIdx);
          const hadExisting = this.removeAllChunks(file.absPath) > 0;
          if (hadExisting) updated++;
          else added++;
        }

        const chunk = file.chunks[chunkIdx];
        const key = this.entryKey(file.absPath, chunkIdx);
        const excerpt = chunk.text.slice(0, MAX_EXCERPT_LENGTH);
        const { startLine, endLine } = this.chunkLineRange(chunk);
        this.data.entries[key] = {
          relPath: file.relPath,
          sourceDir: file.sourceDir,
          mtime: file.mtime,
          vector: storedVector,
          group: allChunkGroups[i],
          excerpt,
          heading: chunk.heading,
          chunkIndex: chunkIdx,
          startLine,
          endLine,
        };
        this.fts.upsert({
          key,
          absPath: file.absPath,
          relPath: file.relPath,
          sourceDir: file.sourceDir,
          heading: chunk.heading,
          content: excerpt,
          chunkIndex: chunkIdx,
          mtime: file.mtime,
        });
      }
    }

    if (added + updated + removed > 0) {
      report?.({ phase: "save" });
      await this.save();
    }

    return { added, updated, removed };
  }

  async rebuild(): Promise<void> {
    this.data.entries = {};
    try {
      this.fts.clear();
    } catch {
      // FTS not loaded — sync will populate it.
    }
    await this.sync();
  }

  /**
   * Pure vector search. Retained as an escape hatch for callers that
   * explicitly want cosine-only ranking (tests, A/B comparisons).
   *
   * In the `knowledge_search` tool path we call `search()` below, which
   * delegates to `hybridSearch()` by default.
   *
   * Throws if called in FTS-only mode — use `search()` or `hybridSearch()`
   * which degrade gracefully.
   */
  async vectorSearch(
    query: string,
    limit: number,
    signal?: AbortSignal,
  ): Promise<SearchResult[]> {
    if (!this.embedder) {
      throw new Error(
        "vectorSearch() requires an embedder — configure a provider or use search()/hybridSearch() instead.",
      );
    }
    // One query embedding per group that actually has stored vectors — a
    // chunk is only ever compared against its own space's query vector
    // (nomic vectors against the nomic query, jina vectors against the
    // jina query; cross-space similarities are meaningless).
    const queryVectorByGroup = new Map<EmbedGroup, number[]>();

    const scored: { key: string; absPath: string; score: number; group: EmbedGroup }[] = [];
    for (const [key, entry] of Object.entries(this.data.entries)) {
      if (!entry.vector || entry.vector.length === 0) continue;
      const group = this.entryGroup(key, entry);
      let queryVector = queryVectorByGroup.get(group);
      if (!queryVector) {
        queryVector = await this.embedder.embed(query, group, signal);
        queryVectorByGroup.set(group, queryVector);
      }
      const score = dotProduct(queryVector, entry.vector);
      scored.push({ key, absPath: this.absPathFromKey(key), score, group });
    }

    scored.sort((a, b) => b.score - a.score);

    // Collect per-file match counts and line ranges (pre-dedup) so callers
    // can show "file (N hits, L..-L..)" even though only the best chunk per
    // file is returned.
    const matchesByFile = new Map<string, number>();
    const rangesByFile = new Map<string, Array<[number, number]>>();
    for (const item of scored) {
      matchesByFile.set(item.absPath, (matchesByFile.get(item.absPath) ?? 0) + 1);
      const entry = this.data.entries[item.key];
      if (entry && typeof entry.startLine === "number") {
        const ranges = rangesByFile.get(item.absPath) ?? [];
        ranges.push([entry.startLine + 1, (entry.endLine ?? entry.startLine) + 1]);
        rangesByFile.set(item.absPath, ranges);
      }
    }

    // Deduplicate: keep only the best-scoring chunk per file
    const seenPaths = new Set<string>();
    const deduped: { key: string; absPath: string; score: number; group: EmbedGroup }[] = [];

    for (const item of scored) {
      if (seenPaths.has(item.absPath)) continue;
      seenPaths.add(item.absPath);
      deduped.push(item);
      if (deduped.length >= limit) break;
    }

    return deduped
      .filter((s) => s.score > 0.15)
      .map((s) => {
        const entry = this.data.entries[s.key];
        return {
          path: s.absPath,
          score: s.score,
          excerpt: entry.excerpt,
          heading: entry.heading,
          matches: matchesByFile.get(s.absPath) ?? 1,
          lineRanges: (rangesByFile.get(s.absPath) ?? []).sort((a, b) => a[0] - b[0]),
          sources: [] as RetrievalSource[],
          group: s.group,
        };
      });
  }

  /**
   * Default search path used by the `knowledge_search` tool. Delegates to
   * hybrid (vector + BM25 blended like pi-local-rag's hybridSearch).
   */
  async search(query: string, limit: number, signal?: AbortSignal): Promise<SearchResult[]> {
    return this.hybridSearch(query, limit, signal);
  }

  /**
   * Hybrid search — mirrors pi-local-rag's `hybridSearch` over both
   * embedding spaces (nomic prose + jina code): FTS5 BM25 + cosine
   * embeddings blended as `alpha * bm25 + (1 - alpha) * vector` with
   * alpha = 0.4, instead of RRF.
   *
   *  - BM25 raw scores are min-max normalized across the FTS candidate set
   *    (range 0 → all candidates score 1, so ties stay rankable)
   *  - vector similarity is the raw cosine on unit-normalized embeddings,
   *    clamped at 0 (no per-space min-max, which would pin the top chunk at
   *    1.0 and create structural ties)
   *  - BM25 gets a 1.5× filename boost (capped at 1) when the first
   *    meaningful query term appears in the file path
   *  - a chunk found only by one backend keeps its raw other-side score of 0
   *
   * Falls back gracefully:
   *   - embedding call fails or the store has no vectors → pure BM25
   *   - no FTS hits or empty side-car → pure vector
   *   - both fail → empty
   *
   * Deduplicates so only the best chunk per file is returned.
   */
  async hybridSearch(
    query: string,
    limit: number,
    signal?: AbortSignal,
  ): Promise<SearchResult[]> {
    return (await this.searchWithBm25(query, limit, signal)).results;
  }

  /**
   * Hybrid search (see `hybridSearch`) that additionally returns the raw
   * BM25 (FTS5 side-car) file hits harvested from the candidate set before
   * fusion — the keyword-side view of the same query, so callers can show
   * what pure keyword matching found alongside the blended ranking.
   *
   * Dual-space search, mirroring pi-local-rag: chunks live in two
   * independent embedding spaces (text → nomic, code → jina). Each space's
   * query is embedded only when that space has stored vectors; result
   * selection is pi-local-rag's ratio-based quota split — the total
   * (RESULT_TOTAL_QUOTA_DUAL_SPACE = 7 when both spaces store vectors, else
   * RESULT_TOTAL_QUOTA = 5, capped by `limit`) is divided between code and
   * prose in proportion to each space's stored vector count (integer
   * quotas, min 1 per group, code group first), and each group enforces its
   * own relevance floor (MIN_HYBRID_SCORE_CODE = 0.35 for hits the jina
   * space surfaced, MIN_HYBRID_SCORE_TEXT = 0.4 for everything else).
   */
  async searchWithBm25(
    query: string,
    limit: number,
    signal?: AbortSignal,
  ): Promise<HybridSearchWithBm25> {
    const ALPHA = 0.4;
    const ftsCandidateLimit = Math.max(limit * 20, 200);
    const vectorCandidateLimit = Math.max(limit * 10, 100);

    // BM25 candidates with raw (negated, bigger-is-better) scores.
    let ftsCandidates: { key: string; absPath: string; score: number }[];
    try {
      ftsCandidates = this.fts.searchScores(query, ftsCandidateLimit);
    } catch {
      ftsCandidates = [];
    }

    // Per-space stored vector counts — they gate the query embeddings and
    // drive the ratio-based result quota split.
    const { code: codeVectorCount, text: proseVectorCount } = this.vectorCountsByGroup();

    // Vector candidates per space: raw cosine similarity on unit-normalized
    // embeddings (dot product), clamped at 0, top-K by similarity. Each
    // model's query is only embedded when its own space has stored vectors
    // (skips loading a model the store can't use); a space whose embedding
    // fails is skipped without taking down the other.
    const vectorSimilarityByKey = new Map<string, number>();
    const vectorSourceByKey = new Map<string, RetrievalSource>();
    if (this.embedder) {
      const spaceFor = (group: EmbedGroup): RetrievalSource =>
        group === "code" ? "jina-code" : "nomic";
      await Promise.all(
        (["text", "code"] as const).map(async (group) => {
          const storedCount = group === "code" ? codeVectorCount : proseVectorCount;
          if (!storedCount) return;
          try {
            const queryVector = await this.embedder!.embed(query, group, signal);
            const scored: { key: string; sim: number }[] = [];
            for (const [key, entry] of Object.entries(this.data.entries)) {
              if (!entry.vector || entry.vector.length === 0) continue;
              if (this.entryGroup(key, entry) !== group) continue;
              scored.push({ key, sim: Math.max(0, dotProduct(queryVector, entry.vector)) });
            }
            scored.sort((a, b) => b.sim - a.sim);
            for (const s of scored.slice(0, vectorCandidateLimit)) {
              vectorSimilarityByKey.set(s.key, s.sim);
              vectorSourceByKey.set(s.key, spaceFor(group));
            }
          } catch (err) {
            // Surface a readable hint on first failure; swallow otherwise.
            if (process.env.KNOWLEDGE_SEARCH_DEBUG) {
              console.error(
                `knowledge-search: ${spaceFor(group)} vector search failed: ${(err as Error).message}`
              );
            }
          }
        }),
      );
    }

    // Union of candidate keys from all engines.
    const candidateKeys = new Set<string>([
      ...ftsCandidates.map((c) => c.key),
      ...vectorSimilarityByKey.keys(),
    ]);

    // Per-file BM25 hits, harvested straight from the ranked FTS candidates
    // (best keyword match first — Map iteration preserves insertion order).
    // These are the pre-fusion keyword hits, reported independently of the
    // blended ranking so callers can show the FTS5 side-car's own view.
    const bm25RangesByFile = new Map<string, Array<[number, number]>>();
    for (const c of ftsCandidates) {
      const entry = this.data.entries[c.key];
      if (!entry || typeof entry.startLine !== "number") continue;
      const ranges = bm25RangesByFile.get(c.absPath) ?? [];
      ranges.push([entry.startLine + 1, (entry.endLine ?? entry.startLine) + 1]);
      bm25RangesByFile.set(c.absPath, ranges);
    }
    const bm25Files: Bm25FileHit[] = [...bm25RangesByFile.entries()].map(([bm25Path, ranges]) => ({
      path: bm25Path,
      lineRanges: ranges.sort((a, b) => a[0] - b[0]),
    }));

    if (candidateKeys.size === 0) return { results: [], bm25Files };

    // Min-max normalize BM25 across the FTS candidate set (constant 1 when
    // every candidate ties, so ties stay rankable rather than zeroed).
    const bm25ByKey = new Map<string, number>();
    if (ftsCandidates.length > 0) {
      let bm25Max = -Infinity;
      let bm25Min = Infinity;
      for (const c of ftsCandidates) {
        if (c.score > bm25Max) bm25Max = c.score;
        if (c.score < bm25Min) bm25Min = c.score;
      }
      const bm25Range = bm25Max - bm25Min;
      for (const c of ftsCandidates) {
        bm25ByKey.set(c.key, bm25Range === 0 ? 1 : (c.score - bm25Min) / bm25Range);
      }
    }

    const hasAnyVectors = vectorSimilarityByKey.size > 0;
    const meaningfulQueryTerms = query.toLowerCase().split(/\s+/).filter((t) => t.length > 1);
    const firstQueryTerm = meaningfulQueryTerms[0];

    // Per-engine provenance for every candidate: which engine(s) surfaced
    // it (BM25, nomic space, jina-code space) — pi-local-rag's sources.
    const sourcesByKey = new Map<string, RetrievalSource[]>();
    const attributeSource = (key: string, source: RetrievalSource) => {
      const attributed = sourcesByKey.get(key) ?? [];
      if (!attributed.includes(source)) attributed.push(source);
      sourcesByKey.set(key, attributed);
    };
    for (const c of ftsCandidates) attributeSource(c.key, "bm25");
    for (const [key, source] of vectorSourceByKey) attributeSource(key, source);

    // Score every candidate: normalized BM25 (with a filename boost) blended
    // with the vector similarity, exactly pi-local-rag's alpha blend.
    const scored: { key: string; score: number }[] = [];
    for (const key of candidateKeys) {
      let bm25Normalized = bm25ByKey.get(key) ?? 0;
      // Boost when the first meaningful query term appears in the file path
      // (the absolute path — pi-local-rag's file_path). Guarded on the term
      // so an empty query can't spuriously boost every result.
      const absPath = this.absPathFromKey(key);
      if (firstQueryTerm && absPath.toLowerCase().includes(firstQueryTerm)) {
        bm25Normalized = Math.min(1, bm25Normalized * 1.5);
      }
      const vectorSimilarity = vectorSimilarityByKey.get(key) ?? 0;
      const hybridScore = hasAnyVectors
        ? ALPHA * bm25Normalized + (1 - ALPHA) * vectorSimilarity
        : bm25Normalized;
      scored.push({ key, score: hybridScore });
    }
    scored.sort((a, b) => b.score - a.score);

    // Result selection is a ratio-based quota split (see the method doc):
    // each candidate is floored by its own space's relevance floor — a
    // candidate's space is whichever embedding engine surfaced it; BM25-only
    // hits ride the text floor (their keyword-match ceiling is exactly 0.4).
    // The ranked survivors are split into code hits (surfaced by jina) and
    // prose hits; quotas are allocated proportionally to each space's stored
    // vector count (integer quotas summing exactly to the total, at least 1
    // per group when both qualify, code group first — a code hit outranks a
    // prose hit at equal quota rank). A group with fewer qualifying hits
    // than its quota yields the slack to the other group's next-best hits
    // (best hybrid first), so the total stays filled. Chunks found only by
    // BM25 rank with the prose group. Within a group, order is by hybrid
    // score. When only one group qualifies, it takes the whole total.
    const isCodeHit = (key: string) => sourcesByKey.get(key)?.includes("jina-code") ?? false;
    const minScoreFor = (key: string) =>
      isCodeHit(key) ? MIN_HYBRID_SCORE_CODE : MIN_HYBRID_SCORE_TEXT;
    const ranked = scored.filter(
      (s) => s.score > 0 && s.score >= minScoreFor(s.key),
    );
    const codeHits = ranked.filter((s) => isCodeHit(s.key));
    const proseHits = ranked.filter((s) => !isCodeHit(s.key));

    const bothSpacesHaveVectors = codeVectorCount > 0 && proseVectorCount > 0;
    const total = Math.min(
      limit,
      bothSpacesHaveVectors ? RESULT_TOTAL_QUOTA_DUAL_SPACE : RESULT_TOTAL_QUOTA,
    );

    let selected: { key: string; score: number }[];
    if (codeHits.length > 0 && proseHits.length > 0) {
      const { codeQuota, proseQuota } = splitResultQuotas(total, codeVectorCount, proseVectorCount);
      const codePrimary = codeHits.slice(0, codeQuota);
      const prosePrimary = proseHits.slice(0, proseQuota);
      const shortfall = total - (codePrimary.length + prosePrimary.length);
      let codeExtra: typeof codeHits = [];
      let proseExtra: typeof proseHits = [];
      if (shortfall > 0) {
        const filler = [
          ...codeHits.slice(codePrimary.length),
          ...proseHits.slice(prosePrimary.length),
        ]
          .sort((a, b) => b.score - a.score)
          .slice(0, shortfall);
        codeExtra = filler.filter((s) => isCodeHit(s.key));
        proseExtra = filler.filter((s) => !isCodeHit(s.key));
      }
      selected = [...codePrimary, ...codeExtra, ...prosePrimary, ...proseExtra];
    } else {
      selected = (codeHits.length > 0 ? codeHits : proseHits).slice(0, total);
    }

    // Collect per-file match counts and line ranges (pre-dedup) so callers
    // can show "file (N hits, L..-L..)" even though only the best chunk per
    // file is returned.
    const matchesByFile = new Map<string, number>();
    const rangesByFile = new Map<string, Array<[number, number]>>();
    for (const { key } of ranked) {
      const absPath = this.absPathFromKey(key);
      matchesByFile.set(absPath, (matchesByFile.get(absPath) ?? 0) + 1);
      const entry = this.data.entries[key];
      if (entry && typeof entry.startLine === "number") {
        const ranges = rangesByFile.get(absPath) ?? [];
        ranges.push([entry.startLine + 1, (entry.endLine ?? entry.startLine) + 1]);
        rangesByFile.set(absPath, ranges);
      }
    }

    // Dedup: keep only the best chunk per file, in quota order.
    const seen = new Set<string>();
    const out: SearchResult[] = [];
    for (const { key, score } of selected) {
      const entry = this.data.entries[key];
      // Key might exist in FTS but not in vector store if vector side is
      // stale. Look up excerpt/heading via FTS fallback in that case.
      const absPath = this.absPathFromKey(key);
      if (seen.has(absPath)) continue;
      seen.add(absPath);
      const finalScore = Math.min(score, 1);
      // Attribute the hit to whichever blend term contributed more — the
      // vector cosine or the normalized BM25. In the pure-keyword fallback
      // (no vectors at all) everything is BM25-driven by definition.
      const bm25Norm = bm25ByKey.get(key) ?? 0;
      const sim = vectorSimilarityByKey.get(key) ?? 0;
      const source: "vector" | "bm25" =
        hasAnyVectors && (1 - ALPHA) * sim >= ALPHA * bm25Norm ? "vector" : "bm25";
      const group = entry
        ? this.entryGroup(key, entry)
        : classifyFileGroup(absPath, this.config.codeExtensions);
      if (entry) {
        out.push({
          path: absPath,
          score: finalScore,
          excerpt: entry.excerpt,
          heading: entry.heading,
          matches: matchesByFile.get(absPath) ?? 1,
          lineRanges: (rangesByFile.get(absPath) ?? []).sort((a, b) => a[0] - b[0]),
          source,
          sources: sourcesByKey.get(key) ?? [],
          group,
        });
      } else {
        // Vector-less — synthesise from whatever FTS has.
        out.push({
          path: absPath,
          score: finalScore,
          excerpt: "",
          heading: "",
          matches: matchesByFile.get(absPath) ?? 1,
          lineRanges: [],
          source,
          sources: sourcesByKey.get(key) ?? [],
          group,
        });
      }
      if (out.length >= limit) break;
    }
    return { results: out, bm25Files };
  }

  /**
   * Update a single file in the index (called by watcher).
   */
  async updateFile(absPath: string, sourceDir: string): Promise<void> {
    if (!fs.existsSync(absPath)) {
      this.removeFile(absPath);
      return;
    }

    const relPath = path.relative(sourceDir, absPath);
    if (this.shouldSkip(relPath, path.basename(absPath))) return;

    const stat = fs.statSync(absPath);
    if (stat.size >= TEXT_MAX_BYTES) {
      this.removeFile(absPath);
      return;
    }
    const content = this.readFileContent(absPath);
    if (!content || content.trim().length <= 20) {
      this.removeFile(absPath);
      return;
    }

    const chunks = chunkMarkdown(content);
    if (chunks.length === 0) {
      this.removeFile(absPath);
      return;
    }

    // Remove old chunks for this file
    this.removeAllChunks(absPath);

    // Embed and store each chunk (vectors remain empty in FTS-only mode).
    // The whole file goes to one embedding group — code extension → jina,
    // everything else → nomic.
    let vectors: (number[] | null)[];
    if (this.embedder) {
      const group = classifyFileGroup(absPath, this.config.codeExtensions);
      const texts = chunks.map((c) => this.chunkEmbedText(group, relPath, c.heading, c.text));
      vectors = await this.embedder.embedBatch(texts, group);
    } else {
      vectors = new Array(chunks.length).fill(null);
    }

    const fileGroup = classifyFileGroup(absPath, this.config.codeExtensions);
    for (let i = 0; i < chunks.length; i++) {
      const vector = vectors[i];
      if (this.embedder && !vector) continue;
      const storedVector: number[] = vector ?? [];

      const key = this.entryKey(absPath, i);
      const excerpt = chunks[i].text.slice(0, MAX_EXCERPT_LENGTH);
      const { startLine, endLine } = this.chunkLineRange(chunks[i]);
      this.data.entries[key] = {
        relPath,
        sourceDir,
        mtime: stat.mtimeMs,
        vector: storedVector,
        group: fileGroup,
        excerpt,
        heading: chunks[i].heading,
        chunkIndex: i,
        startLine,
        endLine,
      };
      this.fts.upsert({
        key,
        absPath,
        relPath,
        sourceDir,
        heading: chunks[i].heading,
        content: excerpt,
        chunkIndex: i,
        mtime: stat.mtimeMs,
      });
    }
    this.scheduleSave();
  }

  removeFile(absPath: string): void {
    const removed = this.removeAllChunks(absPath);
    if (removed > 0) {
      this.scheduleSave();
    }
  }

  /**
   * Remove every trace of files indexed from the given source dirs — used
   * when a whole directory is dropped from the config. Complements
   * removeFile(): entries are matched by their stored sourceDir (so keys
   * pointing elsewhere are still caught) and FTS rows are swept by
   * sourceDir even when no vector entry references them anymore.
   * Returns the number of distinct files removed from the vector store.
   */
  removeBySourceDirs(dirs: string[]): number {
    const targets = new Set(dirs);
    const touched = new Set<string>();
    for (const key of Object.keys(this.data.entries)) {
      if (targets.has(this.data.entries[key].sourceDir)) {
        touched.add(this.absPathFromKey(key));
      }
    }
    for (const absPath of touched) {
      this.removeAllChunks(absPath);
    }
    // Sweep FTS rows under the removed dirs even when the vector side has
    // no matching entries (orphaned by a previously failed/locked write).
    this.fts.deleteBySourceDirs(dirs);
    if (touched.size > 0) {
      this.scheduleSave();
    }
    return touched.size;
  }

  /** Alias for removeFile — removes all data for a file path. */
  deleteFile(absPath: string): void {
    this.removeFile(absPath);
  }

  /** Flush pending saves and release resources. Awaits any in-flight save. */
  async close(): Promise<void> {
    if (this.saveTimer) {
      clearTimeout(this.saveTimer);
      this.saveTimer = null;
    }
    if (this.dirty) {
      await this.save();
    }
    try {
      this.fts.close();
    } catch {
      // already closed
    }
  }

  // -----------------------------------------------------------------------
  // Scanning
  // -----------------------------------------------------------------------

  private scanAllFiles(): {
    absPath: string;
    relPath: string;
    sourceDir: string;
    mtime: number;
  }[] {
    const results: {
      absPath: string;
      relPath: string;
      sourceDir: string;
      mtime: number;
    }[] = [];

    for (const dir of this.config.dirs) {
      this.walkDir(dir, dir, results);
    }
    return results;
  }

  private walkDir(
    currentDir: string,
    sourceDir: string,
    results: {
      absPath: string;
      relPath: string;
      sourceDir: string;
      mtime: number;
    }[]
  ): void {
    let entries: fs.Dirent[];
    try {
      entries = fs.readdirSync(currentDir, { withFileTypes: true });
    } catch {
      return;
    }

    for (const entry of entries) {
      const absPath = path.join(currentDir, entry.name);

      if (entry.isDirectory()) {
        if (this.config.excludeDirs.includes(entry.name) || entry.name.startsWith(".")) {
          continue;
        }
        this.walkDir(absPath, sourceDir, results);
      } else if (entry.isFile()) {
        // Case-insensitive extension match (like pi-local-rag) so README.MD
        // and config.JSON index the same as their lowercase forms.
        const ext = path.extname(entry.name).toLowerCase();
        if (!this.config.fileExtensions.includes(ext)) continue;
        const relPath = path.relative(sourceDir, absPath);
        if (this.shouldSkip(relPath, entry.name)) continue;
        try {
          const stat = fs.statSync(absPath);
          if (stat.size >= TEXT_MAX_BYTES) continue;
          results.push({ absPath, relPath, sourceDir, mtime: stat.mtimeMs });
        } catch {
          // Skip unreadable
        }
      }
    }
  }

  private shouldSkip(relPath: string, _basename: string): boolean {
    const parts = relPath.split(path.sep);
    for (const part of parts) {
      if (this.config.excludeDirs.includes(part) || part.startsWith(".")) {
        return true;
      }
    }
    return false;
  }

  private readFileContent(absPath: string): string | null {
    try {
      const content = fs.readFileSync(absPath, "utf-8");
      // Strip YAML frontmatter if present (common in markdown)
      return content.replace(/^---\n[\s\S]*?\n---\n?/, "");
    } catch {
      return null;
    }
  }
}

/** Dot product — works as cosine similarity when vectors are pre-normalized. */
export function dotProduct(a: number[], b: number[]): number {
  let sum = 0;
  const len = Math.min(a.length, b.length);
  for (let i = 0; i < len; i++) {
    sum += a[i] * b[i];
  }
  return sum;
}
