import { describe, it, before, after, beforeEach } from "node:test";
import assert from "node:assert/strict";
import * as fs from "node:fs";
import * as os from "node:os";
import * as path from "node:path";
import { KnowledgeIndex, dotProduct, type SyncProgress } from "./index-store.js";
import type { Config } from "./config.js";
import type { Embedder } from "./embedder.js";

describe("dotProduct", () => {
  it("returns 0 for orthogonal vectors", () => {
    assert.equal(dotProduct([1, 0, 0], [0, 1, 0]), 0);
  });

  it("returns 1 for identical unit vectors", () => {
    const v = [1 / Math.sqrt(3), 1 / Math.sqrt(3), 1 / Math.sqrt(3)];
    const result = dotProduct(v, v);
    assert.ok(Math.abs(result - 1.0) < 1e-10, `Expected ~1.0, got ${result}`);
  });

  it("returns -1 for opposite unit vectors", () => {
    const v1 = [1, 0, 0];
    const v2 = [-1, 0, 0];
    assert.equal(dotProduct(v1, v2), -1);
  });

  it("computes correct dot product", () => {
    assert.equal(dotProduct([1, 2, 3], [4, 5, 6]), 32); // 4+10+18
  });

  it("handles empty vectors", () => {
    assert.equal(dotProduct([], []), 0);
  });

  it("handles mismatched lengths (uses shorter)", () => {
    assert.equal(dotProduct([1, 2], [3, 4, 5]), 11); // 3+8
  });

  it("works with high-dimensional vectors", () => {
    const dim = 512;
    const a = new Array(dim).fill(1 / Math.sqrt(dim));
    const b = new Array(dim).fill(1 / Math.sqrt(dim));
    const result = dotProduct(a, b);
    assert.ok(
      Math.abs(result - 1.0) < 1e-10,
      `Expected ~1.0 for normalized vectors, got ${result}`
    );
  });
});

// ---------------------------------------------------------------------------
// Streaming load/save round-trip
//
// These tests exercise the streaming JSON reader (stream-json based) and the
// manual streaming JSON writer used to persist the index. The streaming paths
// exist so that very large indexes (>500MB of serialised state) don't trip
// V8's "Invalid string length" limit that `readFileSync` + `JSON.parse` and
// `JSON.stringify` + `writeFileSync` would hit.
// ---------------------------------------------------------------------------

class StubEmbedder implements Embedder {
  async embed(): Promise<number[]> {
    throw new Error("not used in these tests");
  }
  async embedBatch(): Promise<(number[] | null)[]> {
    throw new Error("not used in these tests");
  }
}

function makeConfig(dir: string, dimensions = 4): Config {
  return {
    dirs: ["/tmp/does-not-matter"],
    fileExtensions: [".md"],
    excludeDirs: [],
    dimensions,
    modelSignature: "transformers:nomic-ai/nomic-embed-text-v1.5:768",
    indexDir: dir,
    autoInject: false,
    overview: { inject: false, maxDepth: 2, maxFoldersPerDir: 20, maxKeywordsPerFolder: 5 },
  };
}

describe("KnowledgeIndex streaming load/save", () => {
  let tmpDir: string;

  before(() => {
    tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), "ks-index-store-"));
  });

  beforeEach(() => {
    for (const f of fs.readdirSync(tmpDir)) {
      fs.rmSync(path.join(tmpDir, f), { force: true });
    }
  });

  after(() => {
    fs.rmSync(tmpDir, { recursive: true, force: true });
  });

  function seed(index: KnowledgeIndex, count: number, dims = 4): void {
    const internal = index as unknown as {
      data: {
        version: number;
        dimensions: number;
        entries: Record<string, unknown>;
      };
    };
    for (let i = 0; i < count; i++) {
      internal.data.entries[`/vault/file-${i}.md#0`] = {
        relPath: `file-${i}.md`,
        sourceDir: "/vault",
        mtime: 1_700_000_000_000 + i,
        vector: Array.from({ length: dims }, (_, k) => Math.sin(i + k)),
        excerpt: `Excerpt for file ${i}. It contains UTF-8 content including \u00e9\u00e1\u00f1 and emoji \ud83d\udcdd and newlines\nacross\nlines.`,
        heading: i % 3 === 0 ? "intro" : `Section ${i}`,
        chunkIndex: 0,
      };
    }
  }

  it("save + load round-trips entries unchanged", async () => {
    const config = makeConfig(tmpDir);
    const writer = new KnowledgeIndex(config, new StubEmbedder());
    seed(writer, 42);

    const saveMethod = (writer as unknown as { save: () => Promise<void> }).save;
    await saveMethod.call(writer);

    const reader = new KnowledgeIndex(config, new StubEmbedder());
    await reader.load();

    const writerData = (writer as unknown as { data: { entries: Record<string, unknown> } }).data;
    const readerData = (reader as unknown as { data: { entries: Record<string, unknown> } }).data;

    assert.equal(reader.chunkCount(), writer.chunkCount());
    assert.deepStrictEqual(readerData, writerData);
  });

  it("load returns an empty index when no file exists", async () => {
    const config = makeConfig(tmpDir);
    const reader = new KnowledgeIndex(config, new StubEmbedder());
    await reader.load();
    assert.equal(reader.chunkCount(), 0);
  });

  it("load discards a corrupt index file instead of throwing", async () => {
    const config = makeConfig(tmpDir);
    fs.writeFileSync(path.join(tmpDir, "index.json"), "{ this is not json !!");
    const reader = new KnowledgeIndex(config, new StubEmbedder());
    await reader.load();
    assert.equal(reader.chunkCount(), 0);
  });

  it("load discards an index with a mismatched version", async () => {
    const config = makeConfig(tmpDir);
    fs.writeFileSync(
      path.join(tmpDir, "index.json"),
      JSON.stringify({ version: 1, dimensions: 4, entries: { "a#0": { vector: [1, 0, 0, 0] } } })
    );
    const reader = new KnowledgeIndex(config, new StubEmbedder());
    await reader.load();
    assert.equal(reader.chunkCount(), 0);
  });

  it("loads an older-but-compatible chunked index (v3) without re-embedding", async () => {
    const config = makeConfig(tmpDir, 4);
    fs.writeFileSync(
      path.join(tmpDir, "index.json"),
      JSON.stringify({
        version: 3,
        dimensions: 4,
        embeddingModel: config.modelSignature,
        entries: {
          "/vault/a.md#0": {
            relPath: "a.md",
            sourceDir: "/vault",
            mtime: 1_700_000_000_000,
            vector: [1, 0, 0, 0],
            excerpt: "Excerpt A",
            heading: "intro",
            chunkIndex: 0,
          },
        },
      })
    );
    const reader = new KnowledgeIndex(config, new StubEmbedder());
    await reader.load();
    assert.equal(reader.chunkCount(), 1);
    const internal = reader as unknown as { data: { version: number; entries: Record<string, unknown> } };
    assert.equal(internal.data.version, 4, "version should normalize to the current format on load");
    const entry = internal.data.entries["/vault/a.md#0"] as { vector: number[] };
    assert.deepStrictEqual(entry.vector, [1, 0, 0, 0], "existing vectors must be preserved");
    await reader.close();
  });

  it("load discards a pre-chunk (v2) index to force a rebuild", async () => {
    const config = makeConfig(tmpDir, 4);
    fs.writeFileSync(
      path.join(tmpDir, "index.json"),
      JSON.stringify({
        version: 2,
        dimensions: 4,
        embeddingModel: config.modelSignature,
        entries: {
          "/vault/a.md": {
            relPath: "a.md",
            sourceDir: "/vault",
            mtime: 1_700_000_000_000,
            vector: [1, 0, 0, 0],
            excerpt: "A",
          },
        },
      })
    );
    const reader = new KnowledgeIndex(config, new StubEmbedder());
    await reader.load();
    assert.equal(reader.chunkCount(), 0);
    await reader.close();
  });

  it("load discards an index with mismatched dimensions (triggers re-index)", async () => {
    const config = makeConfig(tmpDir, 4);
    fs.writeFileSync(
      path.join(tmpDir, "index.json"),
      JSON.stringify({
        version: 4,
        dimensions: 1024,
        entries: { "a#0": { vector: new Array(1024).fill(0) } },
      })
    );
    const reader = new KnowledgeIndex(config, new StubEmbedder());
    await reader.load();
    assert.equal(reader.chunkCount(), 0);
  });

  it("save writes atomically via a .tmp file + rename", async () => {
    const config = makeConfig(tmpDir);
    const writer = new KnowledgeIndex(config, new StubEmbedder());
    seed(writer, 5);

    const saveMethod = (writer as unknown as { save: () => Promise<void> }).save;
    await saveMethod.call(writer);

    // After save, the .tmp file must not linger
    assert.ok(fs.existsSync(path.join(tmpDir, "index.json")));
    assert.ok(!fs.existsSync(path.join(tmpDir, "index.json.tmp")));
  });

  it("round-trips many entries without materialising a single giant string", async () => {
    // This test is about the streaming path — it doesn't verify memory use
    // directly (hard to do in pure node:test) but it does confirm the writer
    // can emit a sizeable index (~5MB of vectors) and the reader can restore
    // it byte-for-byte. With the old readFileSync/writeFileSync path this
    // would still work; the real benefit of streaming kicks in above ~500MB,
    // which is impractical to allocate in CI. So this is a smoke test that
    // the streaming code path behaves correctly on non-trivial input.
    const config = makeConfig(tmpDir, 256);
    const writer = new KnowledgeIndex(config, new StubEmbedder());
    seed(writer, 500, 256); // 500 entries x 256 dims ≈ a few MB serialised

    const saveMethod = (writer as unknown as { save: () => Promise<void> }).save;
    await saveMethod.call(writer);

    const reader = new KnowledgeIndex(config, new StubEmbedder());
    await reader.load();

    assert.equal(reader.chunkCount(), 500);
    const writerData = (writer as unknown as { data: { entries: Record<string, unknown> } }).data;
    const readerData = (reader as unknown as { data: { entries: Record<string, unknown> } }).data;
    assert.deepStrictEqual(readerData, writerData);
  });

  it("falls back to streaming writer when JSON.stringify would exceed V8's string limit", async () => {
    // Force the streaming fallback by monkey-patching JSON.stringify to throw
    // RangeError the way V8 does on strings >= 2^29 bytes. The writer should
    // catch that specific error and re-emit via createWriteStream instead.
    const config = makeConfig(tmpDir);
    const writer = new KnowledgeIndex(config, new StubEmbedder());
    seed(writer, 30);

    const realStringify = JSON.stringify;
    let fastPathCalled = false;
    let streamingPathWorked = false;
    // Only throw on the *first* top-level stringify of the whole data object.
    // The streaming path still uses JSON.stringify for individual keys/values
    // which must keep working.
    (JSON as unknown as { stringify: (v: unknown, ...rest: unknown[]) => string }).stringify = (
      value: unknown,
      ...rest: unknown[]
    ): string => {
      if (!fastPathCalled && value && typeof value === "object" && "version" in value && "entries" in value) {
        fastPathCalled = true;
        throw new RangeError("Invalid string length");
      }
      return realStringify(value, ...(rest as [any, any]));
    };

    try {
      const saveMethod = (writer as unknown as { save: () => Promise<void> }).save;
      await saveMethod.call(writer);
      streamingPathWorked = true;
    } finally {
      (JSON as unknown as { stringify: typeof realStringify }).stringify = realStringify;
    }

    assert.ok(fastPathCalled, "expected fast path to be attempted first");
    assert.ok(streamingPathWorked, "expected streaming fallback to complete");

    // Confirm the written file parses correctly with the regular loader.
    const reader = new KnowledgeIndex(config, new StubEmbedder());
    await reader.load();
    assert.equal(reader.chunkCount(), 30);
  });
});

// ---------------------------------------------------------------------------
// Embedding-engine signature invalidation
//
// Vectors from different engines/models are not comparable. The index
// persists the signature of the engine that built it; a mismatch on load
// removes all existing embeddings so sync() re-embeds everything.
// ---------------------------------------------------------------------------

describe("KnowledgeIndex embedding signature", () => {
  let tmpDir: string;

  before(() => {
    tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), "ks-sig-"));
  });

  beforeEach(() => {
    for (const f of fs.readdirSync(tmpDir)) {
      fs.rmSync(path.join(tmpDir, f), { force: true });
    }
  });

  after(() => {
    fs.rmSync(tmpDir, { recursive: true, force: true });
  });

  function makeSigConfig(dir: string, signature: string, dimensions = 4): Config {
    return {
      dirs: ["/tmp/does-not-matter"],
      fileExtensions: [".md"],
      excludeDirs: [],
      dimensions,
        modelSignature: signature,
      indexDir: dir,
      autoInject: false,
    overview: { inject: false, maxDepth: 2, maxFoldersPerDir: 20, maxKeywordsPerFolder: 5 },
    };
  }

  function writeIndex(entries: number, embeddingModel: string | null): void {
    const map: Record<string, unknown> = {};
    for (let i = 0; i < entries; i++) {
      map[`/vault/file-${i}.md#0`] = {
        relPath: `file-${i}.md`,
        sourceDir: "/vault",
        mtime: 1_700_000_000_000 + i,
        vector: [1, 0, 0, 0],
        excerpt: `Excerpt ${i}`,
        heading: "intro",
        chunkIndex: 0,
      };
    }
    fs.writeFileSync(
      path.join(tmpDir, "index.json"),
      JSON.stringify({ version: 4, dimensions: 4, embeddingModel, entries: map })
    );
  }

  it("keeps entries when the signature matches the current engine", async () => {
    writeIndex(3, "transformers:nomic-ai/nomic-embed-text-v1.5:4");
    const idx = new KnowledgeIndex(makeSigConfig(tmpDir, "transformers:nomic-ai/nomic-embed-text-v1.5:4"), new StubEmbedder());
    await idx.load();
    assert.equal(idx.chunkCount(), 3);
    await idx.close();
  });

  it("removes all existing embeddings when the engine changes", async () => {
    writeIndex(3, "transformers:Xenova/all-MiniLM-L6-v2:4");
    const idx = new KnowledgeIndex(
      makeSigConfig(tmpDir, "transformers:nomic-ai/nomic-embed-text-v1.5:4"),
      new StubEmbedder()
    );
    await idx.load();
    assert.equal(idx.chunkCount(), 0, "vectors from another engine must be dropped");
    await idx.close();
  });

  it("removes legacy vectors built before signatures existed", async () => {
    writeIndex(3, null);
    const idx = new KnowledgeIndex(makeSigConfig(tmpDir, "transformers:nomic-ai/nomic-embed-text-v1.5:4"), new StubEmbedder());
    await idx.load();
    assert.equal(idx.chunkCount(), 0, "signature-less vectors cannot be trusted");
    await idx.close();
  });

  it("FTS-only load ignores a signature mismatch (vectors unused)", async () => {
    writeIndex(3, "transformers:nomic-ai/nomic-embed-text-v1.5:4");
    const idx = new KnowledgeIndex(makeSigConfig(tmpDir, "other-engine:whatever:4"), null);
    await idx.load();
    assert.equal(idx.chunkCount(), 3);
    await idx.close();
  });

  it("persists the current signature on save", async () => {
    const idx = new KnowledgeIndex(
      makeSigConfig(tmpDir, "transformers:nomic-ai/nomic-embed-text-v1.5:4"),
      new StubEmbedder()
    );
    const saveMethod = (idx as unknown as { save: () => Promise<void> }).save;
    await saveMethod.call(idx);
    const raw = JSON.parse(fs.readFileSync(path.join(tmpDir, "index.json"), "utf-8"));
    assert.equal(raw.embeddingModel, "transformers:nomic-ai/nomic-embed-text-v1.5:4");
    await idx.close();
  });
});

// ---------------------------------------------------------------------------
// Text file size cap (mirrors pi-local-rag's TEXT_MAX_BYTES = 500_000)
// ---------------------------------------------------------------------------

describe("KnowledgeIndex text file size cap", () => {
  let tmpDir: string;

  before(() => {
    tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), "ks-sizecap-"));
  });

  beforeEach(() => {
    for (const f of fs.readdirSync(tmpDir)) {
      fs.rmSync(path.join(tmpDir, f), { recursive: true, force: true });
    }
  });

  after(() => {
    fs.rmSync(tmpDir, { recursive: true, force: true });
  });

  /** Embedder that returns fixed-dim vectors without hitting the model. */
  function stubEmbedder(dim = 4): Embedder {
    return {
      embed: async () => new Array(dim).fill(0.25),
      embedBatch: async (texts: string[]) => texts.map(() => new Array(dim).fill(0.25)),
    };
  }

  it("skips files at or above 500,000 bytes during sync", async () => {
    const vault = path.join(tmpDir, "vault");
    fs.mkdirSync(vault, { recursive: true });

    const smallFile = path.join(vault, "small.md");
    fs.writeFileSync(smallFile, "# Small\n\nThis is a small note that fits.\n");

    const bigFile = path.join(vault, "big.md");
    fs.writeFileSync(bigFile, "x".repeat(500_000));

    const indexDir = path.join(tmpDir, "index");
    const config = makeConfig(indexDir, 4);
    config.dirs = [vault];
    const idx = new KnowledgeIndex(config, stubEmbedder(4));
    await idx.load();
    const { added } = await idx.sync();
    assert.equal(added, 1, "only the small file should be indexed");
    assert.equal(idx.size(), 1);
    await idx.close();
  });

  it("skips oversized files in updateFile", async () => {
    const vault = path.join(tmpDir, "vault2");
    fs.mkdirSync(vault, { recursive: true });

    const bigFile = path.join(vault, "big.md");
    fs.writeFileSync(bigFile, "y".repeat(500_000));

    const indexDir = path.join(tmpDir, "index2");
    const config = makeConfig(indexDir, 4);
    const idx = new KnowledgeIndex(config, stubEmbedder(4));
    await idx.load();
    await idx.updateFile(bigFile, vault);
    assert.equal(idx.size(), 0, "oversized file must not be indexed");
    await idx.close();
  });
});

// ---------------------------------------------------------------------------
// sync() scan progress — the "unchanged" count must never go negative
// ---------------------------------------------------------------------------

describe("KnowledgeIndex sync scan progress", () => {
  let tmpDir: string;

  before(() => {
    tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), "ks-progress-"));
  });

  beforeEach(() => {
    for (const f of fs.readdirSync(tmpDir)) {
      fs.rmSync(path.join(tmpDir, f), { recursive: true, force: true });
    }
  });

  after(() => {
    fs.rmSync(tmpDir, { recursive: true, force: true });
  });

  /** Embedder that returns fixed-dim vectors without hitting the model. */
  function stubEmbedder(dim = 4): Embedder {
    return {
      embed: async () => new Array(dim).fill(0.25),
      embedBatch: async (texts: string[]) => texts.map(() => new Array(dim).fill(0.25)),
    };
  }

  it("reports a non-negative unchanged count when files were removed", async () => {
    const vault = path.join(tmpDir, "vault");
    fs.mkdirSync(vault, { recursive: true });
    fs.writeFileSync(path.join(vault, "a.md"), "# A\n\nContent for a that is long enough.\n");
    fs.writeFileSync(path.join(vault, "b.md"), "# B\n\nContent for b that is long enough.\n");

    const indexDir = path.join(tmpDir, "index");
    const config = makeConfig(indexDir, 4);
    config.dirs = [vault];
    const idx = new KnowledgeIndex(config, stubEmbedder(4));
    await idx.load();

    const scans: Extract<SyncProgress, { phase: "scan" }>[] = [];
    await idx.sync({
      onProgress: (p) => {
        if (p.phase === "scan") scans.push(p);
      },
    });
    assert.equal(scans[0].filesToProcess, 2);
    assert.equal(scans[0].unchanged, 0);

    // Add one file, modify one, delete one — then sync again.
    fs.writeFileSync(path.join(vault, "c.md"), "# C\n\nBrand new content for c.\n");
    fs.writeFileSync(path.join(vault, "b.md"), "# B v2\n\nRewritten content for b.\n");
    fs.rmSync(path.join(vault, "a.md"));

    scans.length = 0;
    const { removed } = await idx.sync({
      onProgress: (p) => {
        if (p.phase === "scan") scans.push(p);
      },
    });

    assert.equal(removed, 1);
    assert.equal(scans.length, 1, "scan event must fire when there is work");
    // a.md was deleted from disk, so the fresh scan (2 files: b.md + c.md)
    // never saw it — the old formula subtracted `removed` here and produced -1.
    assert.equal(scans[0].filesToProcess, 2);
    assert.equal(
      scans[0].unchanged,
      0,
      "unchanged must stay non-negative when deletions accompany re-indexing"
    );
    await idx.close();
  });

  it("reports all files as unchanged on a no-op sync", async () => {
    const vault = path.join(tmpDir, "vault2");
    fs.mkdirSync(vault, { recursive: true });
    fs.writeFileSync(path.join(vault, "a.md"), "# A\n\nSteady content that will not change.\n");

    const indexDir = path.join(tmpDir, "index2");
    const config = makeConfig(indexDir, 4);
    config.dirs = [vault];
    const idx = new KnowledgeIndex(config, stubEmbedder(4));
    await idx.load();
    await idx.sync();

    const scans: Extract<SyncProgress, { phase: "scan" }>[] = [];
    const counts = await idx.sync({
      onProgress: (p) => {
        if (p.phase === "scan") scans.push(p);
      },
    });
    assert.equal(counts.added + counts.updated + counts.removed, 0);
    assert.equal(scans.length, 0, "no scan event fires when nothing to process");
    await idx.close();
  });
});

// ---------------------------------------------------------------------------
// Directory removal — /knowledge remove <dir> must flush the removed dir's
// config entry, vector entries, and FTS side-car rows (including FTS rows
// orphaned without a vector counterpart), while leaving other dirs intact.
// Mirrors handleRemove's purge: per-file removeFile for paths under the
// removed dir, then removeBySourceDirs for anything still keyed to it.
// ---------------------------------------------------------------------------

describe("KnowledgeIndex source-dir removal flush", () => {
  let tmpDir: string;

  before(() => {
    tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), "ks-remove-dir-"));
  });

  after(() => {
    fs.rmSync(tmpDir, { recursive: true, force: true });
  });

  /** Embedder that returns fixed-dim vectors without hitting the model. */
  function stubEmbedder(dim = 4): Embedder {
    return {
      embed: async () => new Array(dim).fill(0.25),
      embedBatch: async (texts: string[]) => texts.map(() => new Array(dim).fill(0.25)),
    };
  }

  function isUnderDir(abs: string, dir: string): boolean {
    if (abs === dir) return true;
    const rel = path.relative(dir, abs);
    return rel !== "" && !rel.startsWith("..") && !path.isAbsolute(rel);
  }

  /** handleRemove's purge sequence, extracted verbatim. */
  async function purgeDir(indexDir: string, dirs: string[], removed: string[]): Promise<number> {
    const config = makeConfig(indexDir, 4);
    config.dirs = dirs;
    const idx = new KnowledgeIndex(config, stubEmbedder(4));
    await idx.load();
    let purged = 0;
    for (const f of idx.listFiles()) {
      if (removed.some((d) => isUnderDir(f.absPath, d))) {
        idx.removeFile(f.absPath);
        purged += 1;
      }
    }
    purged += idx.removeBySourceDirs(removed);
    await idx.close();
    return purged;
  }

  it("flushes vector + FTS data for the removed dir and keeps other dirs", async () => {
    const vaultA = path.join(tmpDir, "vault-a");
    const vaultB = path.join(tmpDir, "vault-b");
    fs.mkdirSync(vaultA, { recursive: true });
    fs.mkdirSync(vaultB, { recursive: true });
    fs.writeFileSync(path.join(vaultA, "alpha.md"), "# Alpha\n\nXylophone quarantine content lives here.\n");
    fs.writeFileSync(path.join(vaultB, "beta.md"), "# Beta\n\nCompletely different keepsake content.\n");

    // Index both dirs.
    const config = makeConfig(path.join(tmpDir, "idx"), 4);
    config.dirs = [vaultA, vaultB];
    const writer = new KnowledgeIndex(config, stubEmbedder(4));
    await writer.load();
    const { added } = await writer.sync();
    assert.equal(added, 2);
    const fts = writer as unknown as { fts: { count(): number } };
    assert.equal(fts.fts.count(), 2, "FTS side-car holds both files' chunks");
    await writer.close();

    // Remove vault A — config + purge, like /knowledge remove.
    const purged = await purgeDir(path.join(tmpDir, "idx"), [vaultA, vaultB], [vaultA]);
    assert.ok(purged >= 1, "at least vault A's file is purged");

    // Reopen: no trace of vault A on either side; vault B untouched.
    const reader = new KnowledgeIndex(config, stubEmbedder(4));
    await reader.load();
    const paths = reader.listFiles().map((f) => f.absPath);
    assert.deepEqual(paths, [path.join(vaultB, "beta.md")]);
    assert.equal((reader as unknown as typeof fts).fts.count(), 1, "FTS holds only vault B's chunk");
    const hits = await reader.search("Xylophone quarantine", 10);
    assert.equal(hits.length, 0, "removed dir's content is unsearchable");
    const kept = await reader.search("keepsake", 10);
    assert.equal(kept.length, 1, "kept dir still searchable");
    await reader.close();
  });

  it("sweeps FTS rows orphaned without a vector entry", async () => {
    const vaultA = path.join(tmpDir, "orphan-a");
    const vaultB = path.join(tmpDir, "orphan-b");
    fs.mkdirSync(vaultA, { recursive: true });
    fs.mkdirSync(vaultB, { recursive: true });
    fs.writeFileSync(path.join(vaultA, "a1.md"), "# A1\n\nWobble content for a1.\n");
    fs.writeFileSync(path.join(vaultA, "a2.md"), "# A2\n\nGrommet content for a2.\n");
    fs.writeFileSync(path.join(vaultB, "b1.md"), "# B1\n\nPlinth content for b1.\n");

    const config = makeConfig(path.join(tmpDir, "idx-orphan"), 4);
    config.dirs = [vaultA, vaultB];
    const writer = new KnowledgeIndex(config, stubEmbedder(4));
    await writer.load();
    await writer.sync();

    // Simulate an orphaned FTS row: a2.md's vector entries vanish (e.g. a
    // previously failed write left the FTS rows behind) — the per-file
    // removeFile pass driven by listFiles() can no longer see a2.md, but
    // removeBySourceDirs must still sweep its FTS rows.
    const internal = writer as unknown as {
      data: { entries: Record<string, unknown> };
      fts: { count(): number; deleteByAbsPath(p: string): number };
      save(): Promise<void>;
    };
    for (const key of Object.keys(internal.data.entries)) {
      if (key.startsWith(path.join(vaultA, "a2.md") + "#")) {
        delete internal.data.entries[key];
      }
    }
    await internal.save();
    const ftsCountWithOrphan = internal.fts.count();
    assert.equal(ftsCountWithOrphan, 3, "a2.md's FTS row is orphaned but present");
    await writer.close();

    const purged = await purgeDir(path.join(tmpDir, "idx-orphan"), [vaultA, vaultB], [vaultA]);
    assert.equal(purged, 1, "only a1.md is still visible to the per-file pass");

    const reader = new KnowledgeIndex(config, stubEmbedder(4));
    await reader.load();
    const fts = reader as unknown as { fts: { count(): number } };
    assert.equal(fts.fts.count(), 1, "orphaned FTS row swept — only vault B's chunk remains");
    const hits = await reader.search("Grommet Wobble", 10);
    assert.equal(hits.length, 0, "neither a1 nor orphaned a2 content is searchable");
    await reader.close();
  });
});
