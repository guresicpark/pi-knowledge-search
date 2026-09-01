import { describe, it } from "node:test";
import assert from "node:assert/strict";
import { formatLookupSummary } from "./lookup-summary.js";
import type { Bm25FileHit, SearchResult } from "./index-store.js";

const identity = (p: string) => p;

function result(overrides: Partial<SearchResult> & { path: string }): SearchResult {
  return {
    score: 0.8,
    excerpt: "…",
    heading: "",
    matches: 1,
    lineRanges: [],
    ...overrides,
  };
}

describe("formatLookupSummary", () => {
  it("nomic only — vector-driven fused hits, no bm25 segment", () => {
    const results: SearchResult[] = [
      result({ path: "/v/a.md", source: "vector", lineRanges: [[1, 8], [9, 37]] }),
      result({ path: "/v/b.md", source: "vector", lineRanges: [[12, 25]] }),
    ];
    const bm25Files: Bm25FileHit[] = [];
    assert.equal(
      formatLookupSummary(results, bm25Files, identity),
      "Knowledge lookup — nomic (/v/a.md:1-8,9-37,/v/b.md:12-25)",
    );
  });

  it("bm25 only — pure-keyword fallback renders just the bm25 segment", () => {
    const results: SearchResult[] = [
      result({ path: "/v/kw.md", source: "bm25", lineRanges: [[2, 10]] }),
    ];
    const bm25Files: Bm25FileHit[] = [
      { path: "/v/kw.md", lineRanges: [[2, 10]] },
      { path: "/v/extra.md", lineRanges: [[22, 50]] },
    ];
    assert.equal(
      formatLookupSummary(results, bm25Files, identity),
      "Knowledge lookup — bm25 (/v/kw.md:2-10,/v/extra.md:22-50)",
    );
  });

  it("mix — vector-driven and bm25-driven hits in separate groups, plus side-car-only files", () => {
    const results: SearchResult[] = [
      result({ path: "/v/semantic.md", source: "vector", lineRanges: [[1, 5]] }),
      result({ path: "/v/keyword.md", source: "bm25", lineRanges: [[30, 44]] }),
    ];
    const bm25Files: Bm25FileHit[] = [
      { path: "/v/keyword.md", lineRanges: [[30, 44]] },
      // Below the fused score floor — only the FTS side-car saw it.
      { path: "/v/weak-kw.md", lineRanges: [[7, 7]] },
    ];
    assert.equal(
      formatLookupSummary(results, bm25Files, identity),
      "Knowledge lookup — nomic (/v/semantic.md:1-5) — bm25 (/v/keyword.md:30-44,/v/weak-kw.md:7)",
    );
  });

  it("entries without source (non-hybrid search) land in the nomic group", () => {
    const results: SearchResult[] = [result({ path: "/v/legacy.md", lineRanges: [[3, 4]] })];
    assert.equal(
      formatLookupSummary(results, [], identity),
      "Knowledge lookup — nomic (/v/legacy.md:3-4)",
    );
  });

  it("falls back to a hit count when line ranges are missing (pre-v4 entries)", () => {
    const results: SearchResult[] = [result({ path: "/v/old.md", source: "vector", matches: 3 })];
    assert.equal(formatLookupSummary(results, [], identity), "Knowledge lookup — nomic (/v/old.md (3))");
  });

  it("single-line ranges render as a bare number; empty bm25 group is omitted", () => {
    const results: SearchResult[] = [
      result({ path: "/v/a.md", source: "vector", lineRanges: [[6, 6]] }),
    ];
    const bm25Files: Bm25FileHit[] = [{ path: "/v/a.md", lineRanges: [[6, 6]] }];
    assert.equal(formatLookupSummary(results, bm25Files, identity), "Knowledge lookup — nomic (/v/a.md:6)");
  });

  it("caps line ranges per file at 3 with a trailing ellipsis (chunk-flood guard)", () => {
    const ranges: Array<[number, number]> = [
      [1, 91],
      [101, 191],
      [201, 291],
      [301, 391],
      [401, 491],
      [501, 591],
      [601, 691],
      [701, 791],
    ];
    const results: SearchResult[] = [result({ path: "/v/big.md", source: "bm25", lineRanges: ranges })];
    const summary = formatLookupSummary(results, [], identity);
    assert.equal(summary, "Knowledge lookup — bm25 (/v/big.md:1-91,101-191,201-291,…)");
    assert.ok(summary.length < 80, "a many-chunk file must not blow up the one-line box");
  });

  it("returns a bare label when neither group has entries (defensive)", () => {
    assert.equal(formatLookupSummary([], [], identity), "Knowledge lookup");
  });
});
