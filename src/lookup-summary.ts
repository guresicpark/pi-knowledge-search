/**
 * One-line summary for the knowledge-lookup TUI box, grouped per backend:
 *
 *   Knowledge lookup — nomic (event_dispatcher.rst:1-8,notes/a.md:12-25) — bm25 (dir/file1:2-10,dir2/file:22-50)
 *
 * The `nomic` group holds fused hybrid hits the vector side drove into the
 * ranking (`SearchResult.source === "vector"`). The `bm25` group holds the
 * keyword side's view: fused hits the BM25 term dominated, plus raw FTS5
 * side-car files that never made the fused cut (below the score floor or
 * beyond the result limit). A group is omitted entirely when empty, so a
 * pure-keyword fallback renders as `Knowledge lookup — bm25 (…)`.
 */

import type { Bm25FileHit, SearchResult } from "./index-store.js";

/** Max entries per group (mirrors the lookup top-K). */
export const LOOKUP_TOP_K = 5;

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
 */
export function formatLookupSummary(
  results: SearchResult[],
  bm25Files: Bm25FileHit[],
  displayPath: (absPath: string) => string,
): string {
  const nomicSummaries = results
    .filter((r) => r.source !== "bm25")
    .map((r) => formatFileHit(r.path, r.lineRanges ?? [], displayPath, r.matches ?? 1));

  const resultPaths = new Set(results.map((r) => r.path));
  const bm25Summaries = [
    ...results
      .filter((r) => r.source === "bm25")
      .map((r) => formatFileHit(r.path, r.lineRanges ?? [], displayPath)),
    // Raw keyword hits that never surfaced in the fused results — the files
    // only the FTS5 side-car found.
    ...bm25Files
      .filter((f) => !resultPaths.has(f.path))
      .slice(0, LOOKUP_TOP_K)
      .map((f) => formatFileHit(f.path, f.lineRanges, displayPath)),
  ];

  const parts: string[] = [];
  if (nomicSummaries.length > 0) parts.push(`nomic (${nomicSummaries.join(",")})`);
  if (bm25Summaries.length > 0) parts.push(`bm25 (${bm25Summaries.join(",")})`);
  if (parts.length === 0) return "Knowledge lookup";
  return `Knowledge lookup — ${parts.join(" — ")}`;
}
