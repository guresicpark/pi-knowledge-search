/**
 * One-line summary for the knowledge-lookup TUI box, grouped per engine —
 * pi-local-rag's per-engine display:
 *
 *   Knowledge lookup — nomic (a.md:1-8,notes/b.md:12-25), jina-code (src/auth.ts:1-9) — bm25 (dir/file1:2-10,dir2/file:22-50)
 *
 * Every fused hit is listed under each engine that surfaced it
 * (`SearchResult.sources`): a chunk found by both its vector space and BM25
 * appears in both groups, exactly like pi-local-rag's engine summaries. The
 * `bm25` group additionally holds raw FTS5 side-car files that never made
 * the fused cut (below the score floor or beyond the result limit). A group
 * is omitted entirely when empty, so a pure-keyword fallback renders as
 * `Knowledge lookup — bm25 (…)`.
 */

import type { Bm25FileHit, RetrievalSource, SearchResult } from "./index-store.js";

/** Max entries per group (mirrors the lookup top-K). */
export const LOOKUP_TOP_K = 5;

/** Display order of engines in the summary (mirrors pi-local-rag). */
const ENGINE_DISPLAY_ORDER: RetrievalSource[] = ["nomic", "jina-code", "bm25"];

/**
 * Max line ranges rendered per file before truncating with `,…` — a file
 * matching in dozens of chunks would otherwise blow up the one-line box.
 * The full hit set still goes to the model via the injected excerpts.
 */
const MAX_RANGES_PER_FILE = 3;

/**
 * `dir/file.md:2-10,22-50` — one entry per file with the matching chunks'
 * 1-indexed line ranges (a bare number is a single-line hit), capped at
 * `MAX_RANGES_PER_FILE` entries. Falls back to a `file (n)` hit count when
 * the stored entry predates line-range indexing, or to the bare path when
 * no ranges are known at all.
 */
function formatFileHit(
  absPath: string,
  lineRanges: Array<[number, number]>,
  displayPath: (path: string) => string,
  matches?: number,
): string {
  const display = displayPath(absPath);
  if (lineRanges.length > 0) {
    const shown = lineRanges
      .slice(0, MAX_RANGES_PER_FILE)
      .map(([s, e]) => (s === e ? `${s}` : `${s}-${e}`))
      .join(",");
    const fmt = lineRanges.length > MAX_RANGES_PER_FILE ? `${shown},…` : shown;
    return `${display}:${fmt}`;
  }
  return matches === undefined ? display : `${display} (${matches})`;
}

/**
 * Build the summary line from the fused hybrid results and the raw BM25
 * (FTS5 side-car) file hits. `displayPath` renders absolute paths for
 * display (cwd-relative when inside the session cwd, else ~-shortened).
 *
 * Results without `sources` (e.g. from plain vectorSearch) are attributed
 * by the legacy blend-dominant `source` field, defaulting to nomic.
 */
export function formatLookupSummary(
  results: SearchResult[],
  bm25Files: Bm25FileHit[],
  displayPath: (absPath: string) => string,
): string {
  // file:ranges entries per engine, in result order. A hit surfaces under
  // every engine that returned it as a candidate.
  const entriesByEngine = new Map<RetrievalSource, string[]>();
  const addEntry = (engine: RetrievalSource, entry: string) => {
    const list = entriesByEngine.get(engine) ?? [];
    list.push(entry);
    entriesByEngine.set(engine, list);
  };

  for (const r of results) {
    const engines: RetrievalSource[] =
      r.sources && r.sources.length > 0
        ? r.sources
        : r.source === "bm25"
          ? ["bm25"]
          : ["nomic"];
    // The formatted hit (with ranges) renders in its primary engine groups;
    // secondary attributions get the bare path so a double-listed file
    // doesn't repeat its whole range list in the one-line box.
    for (const engine of engines) {
      const primary =
        engine === engines[0]
          ? formatFileHit(r.path, r.lineRanges ?? [], displayPath, r.matches ?? 1)
          : displayPath(r.path);
      addEntry(engine, primary);
    }
  }

  // Raw keyword hits that never surfaced in the fused results — the files
  // only the FTS5 side-car found — ride the bm25 group.
  const resultPaths = new Set(results.map((r) => r.path));
  for (const f of bm25Files.filter((f) => !resultPaths.has(f.path)).slice(0, LOOKUP_TOP_K)) {
    addEntry("bm25", formatFileHit(f.path, f.lineRanges, displayPath));
  }

  const parts = ENGINE_DISPLAY_ORDER.filter((engine) => entriesByEngine.has(engine)).map(
    (engine) => `${engine} (${entriesByEngine.get(engine)!.join(",")})`,
  );
  if (parts.length === 0) return "Knowledge lookup";
  return `Knowledge lookup — ${parts.join(" — ")}`;
}
