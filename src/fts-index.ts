import { DatabaseSync } from "node:sqlite";
import { mkdirSync } from "node:fs";
import { join } from "node:path";
import { assertFts5Available } from "./fts5-probe.js";

/**
 * Entry shape passed into `upsert`. Mirrors the fields of an `IndexEntry`
 * in `index-store.ts` that are useful for BM25 keyword search — chunk
 * content, heading, and the path metadata needed to dedupe by file.
 */
export interface FtsChunk {
  /** Entry key: `${absPath}#${chunkIndex}` */
  key: string;
  /** Absolute file path */
  absPath: string;
  /** Path relative to sourceDir */
  relPath: string;
  /** Source directory this chunk belongs to */
  sourceDir: string;
  /** Section heading this chunk falls under */
  heading: string;
  /** Chunk text — the full body, indexed as FTS5 content */
  content: string;
  /** Chunk index within the file */
  chunkIndex: number;
  /** File mtime (ms) at time of indexing */
  mtime: number;
}

export interface FtsSearchHit {
  /** Entry key */
  key: string;
  /** Absolute file path */
  absPath: string;
  /** Relative path from sourceDir */
  relPath: string;
  /** Section heading */
  heading: string;
  /** BM25-derived 0..1-ish relevance score (normalised; higher = better) */
  score: number;
  /** Chunk content (used as excerpt) */
  content: string;
}

/**
 * SQLite FTS5 side-car for keyword search over chunks. Complements the
 * vector index in `index-store.ts` — FTS handles exact-match, proper nouns,
 * error strings, file paths, and anything else where semantic similarity
 * misses.
 *
 * Writes are idempotent (INSERT OR REPLACE by primary key column `key`),
 * so re-indexing the same chunk is safe.
 */
export class FtsChunkIndex {
  private db: DatabaseSync | null = null;
  private readonly dbPath: string;

  constructor(indexDir: string) {
    mkdirSync(indexDir, { recursive: true });
    this.dbPath = join(indexDir, "kb-fts.db");
  }

  load(): void {
    if (this.db) return;
    // Fail fast with an actionable message on Node runtimes without FTS5
    // (Node 22's bundled SQLite omits it). Without this probe the
    // CREATE VIRTUAL TABLE below leaves the DB file with no tables and
    // later queries surface "no such table: chunks".
    assertFts5Available();
    this.db = new DatabaseSync(this.dbPath);
    this.db.exec("PRAGMA busy_timeout = 5000;");
    this.db.exec(`
      CREATE VIRTUAL TABLE IF NOT EXISTS chunks USING fts5(
        key UNINDEXED,
        absPath UNINDEXED,
        relPath UNINDEXED,
        sourceDir UNINDEXED,
        chunkIndex UNINDEXED,
        mtime UNINDEXED,
        heading,
        content,
        tokenize='porter unicode61'
      );
    `);
  }

  private requireDb(): DatabaseSync {
    if (!this.db) throw new Error("FtsChunkIndex: load() not called");
    return this.db;
  }

  count(): number {
    const db = this.requireDb();
    const row = db.prepare("SELECT COUNT(*) AS n FROM chunks").get() as
      | { n: number | bigint }
      | undefined;
    return Number(row?.n ?? 0);
  }

  /** Number of distinct absPaths in the index. */
  fileCount(): number {
    const db = this.requireDb();
    const row = db.prepare("SELECT COUNT(DISTINCT absPath) AS n FROM chunks").get() as
      | { n: number | bigint }
      | undefined;
    return Number(row?.n ?? 0);
  }

  /**
   * Insert or replace a single chunk. Safe to call repeatedly — matches by
   * `key`, which is unique per `${absPath}#${chunkIndex}` pair.
   */
  upsert(chunk: FtsChunk): void {
    const db = this.requireDb();
    // FTS5 virtual tables have no UNIQUE constraint, so we delete-then-insert.
    db.prepare("DELETE FROM chunks WHERE key = ?").run(chunk.key);
    db.prepare(
      `INSERT INTO chunks (key, absPath, relPath, sourceDir, chunkIndex, mtime, heading, content)
       VALUES (?, ?, ?, ?, ?, ?, ?, ?)`,
    ).run(
      chunk.key,
      chunk.absPath,
      chunk.relPath,
      chunk.sourceDir,
      chunk.chunkIndex,
      Math.round(chunk.mtime),
      chunk.heading ?? "",
      chunk.content ?? "",
    );
  }

  /** Bulk upsert inside a single transaction for much higher throughput. */
  upsertMany(chunks: Iterable<FtsChunk>): void {
    const db = this.requireDb();
    const del = db.prepare("DELETE FROM chunks WHERE key = ?");
    const ins = db.prepare(
      `INSERT INTO chunks (key, absPath, relPath, sourceDir, chunkIndex, mtime, heading, content)
       VALUES (?, ?, ?, ?, ?, ?, ?, ?)`,
    );
    db.exec("BEGIN");
    try {
      for (const c of chunks) {
        del.run(c.key);
        ins.run(
          c.key,
          c.absPath,
          c.relPath,
          c.sourceDir,
          c.chunkIndex,
          Math.round(c.mtime),
          c.heading ?? "",
          c.content ?? "",
        );
      }
      db.exec("COMMIT");
    } catch (err) {
      db.exec("ROLLBACK");
      throw err;
    }
  }

  /** Delete a single chunk by its key. */
  delete(key: string): void {
    this.requireDb().prepare("DELETE FROM chunks WHERE key = ?").run(key);
  }

  /** Delete every chunk belonging to a given absolute file path. */
  deleteByAbsPath(absPath: string): number {
    const res = this.requireDb()
      .prepare("DELETE FROM chunks WHERE absPath = ?")
      .run(absPath);
    return Number(res.changes ?? 0);
  }

  /** Remove all entries. */
  clear(): void {
    this.requireDb().exec("DELETE FROM chunks");
  }

  /**
   * Plain keyword search. Returns hits with BM25 normalised into a 0..1-ish
   * relevance score (higher is better) for display alongside vector scores.
   */
  search(query: string, limit = 20): FtsSearchHit[] {
    const fts = toFtsQuery(query);
    if (!fts) return [];
    const db = this.requireDb();
    const rows = db
      .prepare(
        `SELECT key, absPath, relPath, heading, content, bm25(chunks) AS score
           FROM chunks
          WHERE chunks MATCH ?
          ORDER BY score
          LIMIT ?`,
      )
      .all(fts, limit) as Array<{
      key: string;
      absPath: string;
      relPath: string;
      heading: string;
      content: string;
      score: number;
    }>;
    return rows.map((r) => {
      const raw = Number(r.score);
      // BM25: lower is better. Normalise into a rough 0..1 for display.
      const score = 1 / (1 + Math.abs(raw));
      return {
        key: String(r.key),
        absPath: String(r.absPath),
        relPath: String(r.relPath),
        heading: String(r.heading ?? ""),
        content: String(r.content ?? ""),
        score,
      };
    });
  }

  /**
   * Return a Map<entryKey, rank> for RRF fusion. Rank is 1-based (best = 1).
   * Optionally restrict to a subset of candidate keys.
   */
  searchRanks(query: string, limit = 200, allowedKeys?: Set<string>): Map<string, number> {
    const fts = toFtsQuery(query);
    const out = new Map<string, number>();
    if (!fts) return out;
    const db = this.requireDb();
    const rows = db
      .prepare(
        `SELECT key FROM chunks WHERE chunks MATCH ? ORDER BY bm25(chunks) LIMIT ?`,
      )
      .all(fts, limit) as Array<{ key: string }>;
    let r = 1;
    for (const row of rows) {
      const key = String(row.key);
      if (allowedKeys && !allowedKeys.has(key)) continue;
      out.set(key, r++);
    }
    return out;
  }

  /**
   * Raw BM25 scores for FTS candidates, for min-max normalization in hybrid
   * search (mirrors pi-local-rag). FTS5's bm25() is lower-is-better (more
   * negative = better), so the returned score is negated → bigger-is-better.
   */
  searchScores(query: string, limit = 200): { key: string; absPath: string; score: number }[] {
    const fts = toFtsQuery(query);
    if (!fts) return [];
    const db = this.requireDb();
    const rows = db
      .prepare(
        `SELECT key, absPath, bm25(chunks) AS score
           FROM chunks
          WHERE chunks MATCH ?
          ORDER BY score
          LIMIT ?`,
      )
      .all(fts, limit) as Array<{ key: string; absPath: string; score: number }>;
    return rows.map((r) => ({
      key: String(r.key),
      absPath: String(r.absPath),
      score: -Number(r.score),
    }));
  }

  close(): void {
    if (!this.db) return;
    this.db.close();
    this.db = null;
  }
}

// ─── Helpers ─────────────────────────────────────────────────────────

/**
 * Turn a user query into a safe FTS5 MATCH expression.
 * Strips FTS syntax characters, quotes each term, and joins with implicit AND
 * (space-separated quoted phrases) — exactly pi-local-rag's hybrid search.
 *
 * AND is more precise than OR — BM25 ranks multi-term matches highest, and
 * chunks missing a term are excluded rather than diluting the result set.
 *
 * Ported verbatim from pi-session-search/src/fts-index.ts (join aligned with
 * pi-local-rag's search.ts).
 */
export function toFtsQuery(q: string): string {
  const terms = q
    .replace(/["^*():{}[\]]/g, " ")
    .split(/\s+/)
    .map((t) => t.trim())
    .filter((t) => t.length > 0)
    .map((t) => `"${t}"`);
  return terms.join(" ");
}
