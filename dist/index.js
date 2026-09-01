// src/index.ts
import { Type } from "@sinclair/typebox";
import { Box, Text } from "@earendil-works/pi-tui";
import { fork } from "node:child_process";
import * as fs5 from "node:fs";
import { isAbsolute as isAbsolute2, join as join6, relative as relative2, resolve as resolve2 } from "node:path";

// src/config.ts
import * as fs from "node:fs";
import * as path from "node:path";

// src/embedder.ts
import { join } from "node:path";
import { existsSync } from "node:fs";
import { homedir } from "node:os";
var EMBEDDING_MODEL = "nomic-ai/nomic-embed-text-v1.5";
var EMBEDDING_DIMENSIONS = 768;
function createEmbedder() {
  return new TransformersEmbedder(EMBEDDING_MODEL);
}
function truncate(text, maxChars = 1e4) {
  return text.length > maxChars ? text.slice(0, maxChars) : text;
}
function summarizeErrors(errs, max = 3) {
  const list = [...errs];
  const shown = list.slice(0, max).join("; ");
  return list.length > max ? `${shown} (+${list.length - max} more)` : shown;
}
var TRANSFORMERS_QUERY_PREFIX = "search_query: ";
var TRANSFORMERS_DOC_PREFIX = "search_document: ";
var TRANSFORMERS_BATCH_SIZE = 16;
function resolveTransformersCacheDir() {
  if (process.env.PI_RAG_MODEL_CACHE) return process.env.PI_RAG_MODEL_CACHE;
  if (process.env.TRANSFORMERS_CACHE) return process.env.TRANSFORMERS_CACHE;
  if (process.env.HF_HOME) return join(process.env.HF_HOME, "transformers");
  return join(homedir(), ".cache", "huggingface", "transformers");
}
function isTransformersModelCached(model) {
  return existsSync(join(resolveTransformersCacheDir(), model, "onnx", "model_quantized.onnx"));
}
var TransformersEmbedder = class {
  model;
  pipelinePromise = null;
  constructor(model) {
    this.model = model;
  }
  /**
   * Lazily load the ONNX feature-extraction pipeline (q8 quantized weights).
   * The load promise is cached so concurrent first calls share a single
   * download; a failed load is evicted so the next call retries.
   */
  getPipeline() {
    if (!this.pipelinePromise) {
      this.pipelinePromise = (async () => {
        const { pipeline, env } = await import("@huggingface/transformers");
        env.cacheDir = resolveTransformersCacheDir();
        return pipeline("feature-extraction", this.model, { dtype: "q8" });
      })();
      this.pipelinePromise.catch(() => {
        this.pipelinePromise = null;
      });
    }
    return this.pipelinePromise;
  }
  async embed(text, signal) {
    if (signal?.aborted) throw new Error("Aborted");
    const pipe = await this.getPipeline();
    const input = TRANSFORMERS_QUERY_PREFIX + text.replace(/\s+/g, " ").trim();
    const output = await pipe(truncate(input), { pooling: "mean", normalize: true });
    return Array.from(output.data);
  }
  async embedBatch(texts, signal, _concurrency) {
    const results = new Array(texts.length).fill(null);
    if (texts.length === 0) return results;
    let failed = 0;
    const errs = /* @__PURE__ */ new Set();
    try {
      const pipe = await this.getPipeline();
      for (let start = 0; start < texts.length; start += TRANSFORMERS_BATCH_SIZE) {
        if (signal?.aborted) throw new Error("Aborted");
        const batch = texts.slice(start, start + TRANSFORMERS_BATCH_SIZE).map((t) => TRANSFORMERS_DOC_PREFIX + truncate(t));
        const output = await pipe(batch, { pooling: "mean", normalize: true });
        const flattened = output.data;
        const dim = flattened.length / batch.length;
        for (let i = 0; i < batch.length; i++) {
          results[start + i] = Array.from(flattened.slice(i * dim, (i + 1) * dim));
        }
      }
    } catch (err) {
      failed = results.filter((v) => v === null).length;
      errs.add(err.message);
      if (failed > 0) {
        console.error(
          `Transformers embedding failed for ${failed}/${texts.length} chunks: ${summarizeErrors(errs)}`
        );
      }
    }
    return results;
  }
};

// src/config.ts
var DEFAULT_FILE_EXTENSIONS = [
  ".md",
  ".mdx",
  ".txt",
  ".rst",
  ".html",
  ".htm",
  ".json",
  ".jsonc",
  ".yaml",
  ".yml",
  ".toml",
  ".ini",
  ".xml",
  ".csv",
  ".tsv",
  ".env",
  ".gitignore",
  ".dockerfile"
];
function defaultConfigFile(cwd) {
  return path.join(cwd || process.cwd(), ".pi", "knowledge-search.json");
}
function defaultIndexDir(cwd) {
  return path.join(cwd || process.cwd(), ".pi", "knowledge-search");
}
function warnUnknownKeys(block, blockName, knownKeys) {
  if (!block || typeof block !== "object") return;
  const unknown = Object.keys(block).filter((k) => !knownKeys.includes(k));
  if (unknown.length === 0) return;
  console.error(
    `pi-knowledge-search: ignoring unknown key(s) in settings.json "${blockName}" block: ${unknown.join(", ")} (expected: ${knownKeys.join(", ")})`
  );
}
var PI_KNOWLEDGE_SEARCH_SETTINGS_KEYS = ["localPath"];
function resolveLocalBase(cwd) {
  if (!cwd) return null;
  try {
    const raw = fs.readFileSync(path.join(cwd, ".pi", "settings.json"), "utf-8");
    const settings = JSON.parse(raw) ?? {};
    const ks = settings["pi-knowledge-search"];
    warnUnknownKeys(ks, "pi-knowledge-search", PI_KNOWLEDGE_SEARCH_SETTINGS_KEYS);
    if (ks && typeof ks === "object" && typeof ks.localPath === "string" && ks.localPath) {
      return ks.localPath;
    }
  } catch {
  }
  return null;
}
function getConfigPath(cwd) {
  if (process.env.KNOWLEDGE_SEARCH_CONFIG) return process.env.KNOWLEDGE_SEARCH_CONFIG;
  const base = resolveLocalBase(cwd);
  if (base) return path.join(base, "config.json");
  return defaultConfigFile(cwd);
}
function getIndexDir(cwd) {
  if (process.env.KNOWLEDGE_SEARCH_INDEX_DIR) return process.env.KNOWLEDGE_SEARCH_INDEX_DIR;
  const base = resolveLocalBase(cwd);
  if (base) return path.join(base, "index");
  return defaultIndexDir(cwd);
}
function loadConfig(cwd) {
  const configPath = getConfigPath(cwd);
  let file = null;
  if (fs.existsSync(configPath)) {
    try {
      file = JSON.parse(fs.readFileSync(configPath, "utf-8"));
    } catch {
    }
  }
  const envDirs = process.env.KNOWLEDGE_SEARCH_DIRS;
  if (!file && !envDirs) {
    return null;
  }
  const home = process.env.HOME || "/tmp";
  const resolvePath = (p) => p.replace(/^~/, home);
  const dirs = (envDirs ? envDirs.split(",").map((d) => d.trim()) : file?.dirs ?? []).map(resolvePath).filter(Boolean);
  if (dirs.length === 0) return null;
  const fileExtensions = (envStr("KNOWLEDGE_SEARCH_EXTENSIONS")?.split(",").map((e) => e.trim().toLowerCase()) ?? file?.fileExtensions?.map((e) => e.toLowerCase()) ?? DEFAULT_FILE_EXTENSIONS).filter(Boolean);
  const excludeDirs = envStr("KNOWLEDGE_SEARCH_EXCLUDE")?.split(",").map((d) => d.trim()) ?? file?.excludeDirs ?? ["node_modules", ".git", ".obsidian", ".trash"];
  const legacy = file;
  if (legacy && (legacy.provider !== void 0 || legacy.dimensions !== void 0)) {
    console.error(
      'pi-knowledge-search: ignoring "provider"/"dimensions" config keys \u2014 the embedding engine is always nomic-embed-text-v1.5 (local ONNX).'
    );
  }
  const indexDir = getIndexDir(cwd);
  const overviewFile = file?.overview ?? {};
  const overview = {
    inject: envBool("KNOWLEDGE_SEARCH_OVERVIEW_INJECT") ?? overviewFile.inject ?? true,
    maxDepth: envInt("KNOWLEDGE_SEARCH_OVERVIEW_MAX_DEPTH") ?? overviewFile.maxDepth ?? 2,
    maxFoldersPerDir: envInt("KNOWLEDGE_SEARCH_OVERVIEW_MAX_FOLDERS") ?? overviewFile.maxFoldersPerDir ?? 20,
    maxKeywordsPerFolder: envInt("KNOWLEDGE_SEARCH_OVERVIEW_MAX_KEYWORDS") ?? overviewFile.maxKeywordsPerFolder ?? 5
  };
  return {
    dirs,
    fileExtensions,
    excludeDirs,
    dimensions: EMBEDDING_DIMENSIONS,
    modelSignature: `transformers:${EMBEDDING_MODEL}:${EMBEDDING_DIMENSIONS}`,
    indexDir,
    autoInject: envBool("KNOWLEDGE_SEARCH_AUTO_INJECT") ?? file?.autoInject ?? true,
    overview
  };
}
function saveConfig(config, cwd) {
  const configPath = getConfigPath(cwd);
  const dir = path.dirname(configPath);
  fs.mkdirSync(dir, { recursive: true });
  fs.writeFileSync(configPath, JSON.stringify(config, null, 2) + "\n");
}
function envStr(key) {
  const v = process.env[key]?.trim();
  return v || void 0;
}
function envInt(key) {
  const v = envStr(key);
  return v ? parseInt(v, 10) : void 0;
}
function envBool(key) {
  const v = envStr(key)?.toLowerCase();
  if (!v) return void 0;
  if (["1", "true", "yes", "on"].includes(v)) return true;
  if (["0", "false", "no", "off"].includes(v)) return false;
  return void 0;
}

// src/index-store.ts
import * as fs2 from "node:fs";
import * as path2 from "node:path";
import Assembler from "stream-json/assembler.js";
import makeParser from "stream-json/index.js";

// src/chunker.ts
import remarkFrontmatter from "remark-frontmatter";
import remarkParse from "remark-parse";
import { unified } from "unified";
var markdownProcessor = unified().use(remarkParse).use(remarkFrontmatter, ["yaml", "toml"]);
var LARGE_FILE_FAST_PATH_CHARS = 12e4;
function chunkMarkdown(content, maxChunkSize = 3e3, minChunkSize = 200) {
  if (!content || content.trim().length === 0) return [];
  if (content.length >= LARGE_FILE_FAST_PATH_CHARS) {
    return chunkMarkdownFast(content, maxChunkSize, minChunkSize);
  }
  const sections = splitByHeadings(content);
  if (sections.length === 0) return [];
  if (content.length <= maxChunkSize) {
    const starts = lineStartOffsets(content);
    return [
      {
        text: content.trim(),
        heading: sections[0]?.heading ?? "intro",
        startLine: 0,
        endLine: lineFromOffset(content.length - 1, starts),
        charOffset: 0
      }
    ];
  }
  let rawChunks = [];
  for (const section of sections) {
    if (section.text.trim().length === 0) continue;
    if (section.text.length <= maxChunkSize) {
      rawChunks.push(section);
    } else {
      rawChunks.push(...splitByBlocks(section, maxChunkSize));
    }
  }
  rawChunks = rawChunks.flatMap(
    (chunk) => chunk.text.length <= maxChunkSize ? [chunk] : hardSplit(chunk, maxChunkSize, 200)
  );
  return mergeTiny(rawChunks, minChunkSize, maxChunkSize);
}
function chunkMarkdownFast(content, maxChunkSize, minChunkSize) {
  if (content.length <= maxChunkSize) {
    const starts2 = lineStartOffsets(content);
    return [
      {
        text: content.trim(),
        heading: "intro",
        startLine: 0,
        endLine: lineFromOffset(content.length - 1, starts2),
        charOffset: 0
      }
    ];
  }
  const starts = lineStartOffsets(content);
  const headingRegex = /^##+\s+(.+)$/gm;
  const headingMatches = [];
  for (const match of content.matchAll(headingRegex)) {
    const start = match.index ?? 0;
    headingMatches.push({
      start,
      startLine: lineFromOffset(start, starts),
      heading: (match[1] ?? "intro").trim() || "intro"
    });
  }
  const sections = [];
  if (headingMatches.length === 0) {
    sections.push({
      text: content,
      heading: "intro",
      startLine: 0,
      endLine: lineFromOffset(content.length - 1, starts),
      charOffset: 0
    });
  } else {
    if (headingMatches[0].start > 0) {
      sections.push({
        text: content.slice(0, headingMatches[0].start),
        heading: "intro",
        startLine: 0,
        endLine: lineFromOffset(headingMatches[0].start - 1, starts),
        charOffset: 0
      });
    }
    for (let i = 0; i < headingMatches.length; i++) {
      const start = headingMatches[i].start;
      const end = i + 1 < headingMatches.length ? headingMatches[i + 1].start : content.length;
      sections.push({
        text: content.slice(start, end),
        heading: headingMatches[i].heading,
        startLine: headingMatches[i].startLine,
        endLine: lineFromOffset(end - 1, starts),
        charOffset: start
      });
    }
  }
  let rawChunks = [];
  for (const section of sections) {
    if (section.text.trim().length === 0) continue;
    if (section.text.length <= maxChunkSize) {
      rawChunks.push(section);
      continue;
    }
    rawChunks.push(...splitByParagraphsFallback(section, maxChunkSize));
  }
  rawChunks = rawChunks.flatMap(
    (chunk) => chunk.text.length <= maxChunkSize ? [chunk] : hardSplit(chunk, maxChunkSize, 200)
  );
  return mergeTiny(rawChunks, minChunkSize, maxChunkSize);
}
function lineStartOffsets(text) {
  const starts = [0];
  for (let i = 0; i < text.length; i++) {
    if (text.charCodeAt(i) === 10) starts.push(i + 1);
  }
  return starts;
}
function offsetFromLine(line, starts) {
  if (!line || line <= 1) return 0;
  return starts[Math.min(line - 1, starts.length - 1)] ?? 0;
}
function lineFromOffset(offset, starts) {
  let low = 0;
  let high = starts.length - 1;
  while (low <= high) {
    const mid = low + high >> 1;
    if (starts[mid] <= offset) low = mid + 1;
    else high = mid - 1;
  }
  return Math.max(0, low - 1);
}
function headingText(node) {
  if (!node) return "";
  if (typeof node.value === "string") return node.value;
  if (!Array.isArray(node.children)) return "";
  return node.children.map((child) => headingText(child)).join("");
}
function splitByHeadings(content) {
  const tree = markdownProcessor.parse(content);
  const starts = lineStartOffsets(content);
  const headings = (tree.children ?? []).filter((node) => node.type === "heading" && node.depth >= 2).map((node) => {
    const line = node.position?.start?.line;
    const start = offsetFromLine(line, starts);
    return {
      start,
      startLine: lineFromOffset(start, starts),
      heading: headingText(node).trim() || "intro"
    };
  }).sort((a, b) => a.start - b.start);
  if (headings.length === 0) {
    return [
      {
        text: content,
        heading: "intro",
        startLine: 0,
        endLine: lineFromOffset(content.length - 1, starts),
        charOffset: 0
      }
    ];
  }
  const sections = [];
  if (headings[0].start > 0) {
    sections.push({
      text: content.slice(0, headings[0].start),
      heading: "intro",
      startLine: 0,
      endLine: lineFromOffset(headings[0].start - 1, starts),
      charOffset: 0
    });
  }
  for (let i = 0; i < headings.length; i++) {
    const start = headings[i].start;
    const end = i + 1 < headings.length ? headings[i + 1].start : content.length;
    sections.push({
      text: content.slice(start, end),
      heading: headings[i].heading,
      startLine: headings[i].startLine,
      endLine: lineFromOffset(end - 1, starts),
      charOffset: start
    });
  }
  return sections;
}
function splitByBlocks(section, maxChunkSize) {
  const text = section.text;
  const tree = markdownProcessor.parse(text);
  const starts = lineStartOffsets(text);
  const blocks = (tree.children ?? []).map((node) => {
    const startLine = node.position?.start?.line;
    const endLine = node.position?.end?.line;
    if (!startLine || !endLine) return null;
    return {
      start: offsetFromLine(startLine, starts),
      end: offsetFromLine(endLine + 1, starts)
    };
  }).filter((x) => Boolean(x)).sort((a, b) => a.start - b.start);
  if (blocks.length === 0) {
    return splitByParagraphsFallback(section, maxChunkSize);
  }
  const units = blocks.map((block, i) => ({
    start: i === 0 ? 0 : block.start,
    end: i + 1 < blocks.length ? blocks[i + 1].start : text.length
  }));
  const chunks = [];
  let currentText = "";
  let currentOffset = section.charOffset;
  let currentStartLine = section.startLine;
  let currentEndLine = section.startLine;
  for (const unit of units) {
    const unitText = text.slice(unit.start, unit.end);
    const unitEndLine = section.startLine + lineFromOffset(Math.max(unit.end - 1, 0), starts);
    if (currentText.length > 0 && currentText.length + unitText.length > maxChunkSize) {
      chunks.push({
        text: currentText.trim(),
        heading: section.heading,
        startLine: currentStartLine,
        endLine: currentEndLine,
        charOffset: currentOffset
      });
      currentText = unitText;
      currentOffset = section.charOffset + unit.start;
      currentStartLine = section.startLine + lineFromOffset(unit.start, starts);
    } else {
      currentText += unitText;
    }
    currentEndLine = unitEndLine;
  }
  if (currentText.trim().length > 0) {
    chunks.push({
      text: currentText.trim(),
      heading: section.heading,
      startLine: currentStartLine,
      endLine: currentEndLine,
      charOffset: currentOffset
    });
  }
  return chunks;
}
function splitByParagraphsFallback(section, maxChunkSize) {
  const paragraphs = section.text.split(/\n\n+/);
  const chunks = [];
  let currentText = "";
  let currentOffset = section.charOffset;
  let currentStartLine = section.startLine;
  for (const para of paragraphs) {
    if (currentText.length > 0 && currentText.length + para.length + 2 > maxChunkSize) {
      chunks.push({
        text: currentText.trim(),
        heading: section.heading,
        startLine: currentStartLine,
        endLine: currentStartLine + currentText.split("\n").length - 1,
        charOffset: currentOffset
      });
      currentOffset = currentOffset + currentText.length + 2;
      currentStartLine += currentText.split("\n").length + 1;
      currentText = para;
    } else {
      currentText = currentText ? currentText + "\n\n" + para : para;
    }
  }
  if (currentText.trim().length > 0) {
    chunks.push({
      text: currentText.trim(),
      heading: section.heading,
      startLine: currentStartLine,
      endLine: currentStartLine + currentText.split("\n").length - 1,
      charOffset: currentOffset
    });
  }
  return chunks;
}
function hardSplit(chunk, maxSize, overlap) {
  const { text, heading, startLine, charOffset } = chunk;
  const chunks = [];
  let pos = 0;
  while (pos < text.length) {
    const end = Math.min(pos + maxSize, text.length);
    const pieceText = text.slice(pos, end);
    const pieceStartLine = startLine + text.slice(0, pos).split("\n").length - 1;
    chunks.push({
      text: pieceText,
      heading,
      startLine: pieceStartLine,
      endLine: pieceStartLine + pieceText.split("\n").length - 1,
      charOffset: charOffset + pos
    });
    pos = end - (end < text.length ? overlap : 0);
    if (pos <= chunks[chunks.length - 1].charOffset - charOffset) {
      pos = end;
    }
  }
  return chunks;
}
function mergeTiny(chunks, minSize, maxSize) {
  if (chunks.length <= 1) return chunks;
  const merged = [chunks[0]];
  for (let i = 1; i < chunks.length; i++) {
    const prev = merged[merged.length - 1];
    const curr = chunks[i];
    if (curr.text.length < minSize && prev.text.length + curr.text.length + 2 <= maxSize) {
      prev.text = prev.text + "\n\n" + curr.text;
      prev.endLine = curr.endLine;
    } else if (prev.text.length < minSize && prev.text.length + curr.text.length + 2 <= maxSize) {
      prev.text = prev.text + "\n\n" + curr.text;
      prev.heading = curr.heading;
      prev.endLine = curr.endLine;
    } else {
      merged.push(curr);
    }
  }
  return merged;
}

// src/fts-index.ts
import { DatabaseSync as DatabaseSync2 } from "node:sqlite";
import { mkdirSync as mkdirSync2 } from "node:fs";
import { join as join3 } from "node:path";

// src/fts5-probe.ts
import { DatabaseSync } from "node:sqlite";
var cached = null;
function assertFts5Available() {
  if (cached === true) return;
  if (cached === false) throw new Error(fts5ErrorMessage());
  const db = new DatabaseSync(":memory:");
  try {
    db.exec("CREATE VIRTUAL TABLE _fts5_probe USING fts5(x)");
    cached = true;
  } catch {
    cached = false;
    throw new Error(fts5ErrorMessage());
  } finally {
    try {
      db.close();
    } catch {
    }
  }
}
function fts5ErrorMessage() {
  return `SQLite FTS5 is not available in this Node runtime. pi-knowledge-search requires Node 24+ (where node:sqlite ships with FTS5 compiled in). Current: Node ${process.versions.node}. Upgrade Node and restart pi.`;
}

// src/fts-index.ts
var FtsChunkIndex = class {
  db = null;
  dbPath;
  constructor(indexDir) {
    mkdirSync2(indexDir, { recursive: true });
    this.dbPath = join3(indexDir, "kb-fts.db");
  }
  load() {
    if (this.db) return;
    assertFts5Available();
    this.db = new DatabaseSync2(this.dbPath);
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
  requireDb() {
    if (!this.db) throw new Error("FtsChunkIndex: load() not called");
    return this.db;
  }
  count() {
    const db = this.requireDb();
    const row = db.prepare("SELECT COUNT(*) AS n FROM chunks").get();
    return Number(row?.n ?? 0);
  }
  /** Number of distinct absPaths in the index. */
  fileCount() {
    const db = this.requireDb();
    const row = db.prepare("SELECT COUNT(DISTINCT absPath) AS n FROM chunks").get();
    return Number(row?.n ?? 0);
  }
  /**
   * Insert or replace a single chunk. Safe to call repeatedly — matches by
   * `key`, which is unique per `${absPath}#${chunkIndex}` pair.
   */
  upsert(chunk) {
    const db = this.requireDb();
    db.prepare("DELETE FROM chunks WHERE key = ?").run(chunk.key);
    db.prepare(
      `INSERT INTO chunks (key, absPath, relPath, sourceDir, chunkIndex, mtime, heading, content)
       VALUES (?, ?, ?, ?, ?, ?, ?, ?)`
    ).run(
      chunk.key,
      chunk.absPath,
      chunk.relPath,
      chunk.sourceDir,
      chunk.chunkIndex,
      Math.round(chunk.mtime),
      chunk.heading ?? "",
      chunk.content ?? ""
    );
  }
  /** Bulk upsert inside a single transaction for much higher throughput. */
  upsertMany(chunks) {
    const db = this.requireDb();
    const del = db.prepare("DELETE FROM chunks WHERE key = ?");
    const ins = db.prepare(
      `INSERT INTO chunks (key, absPath, relPath, sourceDir, chunkIndex, mtime, heading, content)
       VALUES (?, ?, ?, ?, ?, ?, ?, ?)`
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
          c.content ?? ""
        );
      }
      db.exec("COMMIT");
    } catch (err) {
      db.exec("ROLLBACK");
      throw err;
    }
  }
  /** Delete a single chunk by its key. */
  delete(key) {
    this.requireDb().prepare("DELETE FROM chunks WHERE key = ?").run(key);
  }
  /** Delete every chunk belonging to a given absolute file path. */
  deleteByAbsPath(absPath) {
    const res = this.requireDb().prepare("DELETE FROM chunks WHERE absPath = ?").run(absPath);
    return Number(res.changes ?? 0);
  }
  /** Remove all entries. */
  clear() {
    this.requireDb().exec("DELETE FROM chunks");
  }
  /**
   * Plain keyword search. Returns hits with BM25 normalised into a 0..1-ish
   * relevance score (higher is better) for display alongside vector scores.
   */
  search(query, limit = 20) {
    const fts = toFtsQuery(query);
    if (!fts) return [];
    const db = this.requireDb();
    const rows = db.prepare(
      `SELECT key, absPath, relPath, heading, content, bm25(chunks) AS score
           FROM chunks
          WHERE chunks MATCH ?
          ORDER BY score
          LIMIT ?`
    ).all(fts, limit);
    return rows.map((r) => {
      const raw = Number(r.score);
      const score = 1 / (1 + Math.abs(raw));
      return {
        key: String(r.key),
        absPath: String(r.absPath),
        relPath: String(r.relPath),
        heading: String(r.heading ?? ""),
        content: String(r.content ?? ""),
        score
      };
    });
  }
  /**
   * Return a Map<entryKey, rank> for RRF fusion. Rank is 1-based (best = 1).
   * Optionally restrict to a subset of candidate keys.
   */
  searchRanks(query, limit = 200, allowedKeys) {
    const fts = toFtsQuery(query);
    const out = /* @__PURE__ */ new Map();
    if (!fts) return out;
    const db = this.requireDb();
    const rows = db.prepare(
      `SELECT key FROM chunks WHERE chunks MATCH ? ORDER BY bm25(chunks) LIMIT ?`
    ).all(fts, limit);
    let r = 1;
    for (const row of rows) {
      const key = String(row.key);
      if (allowedKeys && !allowedKeys.has(key)) continue;
      out.set(key, r++);
    }
    return out;
  }
  close() {
    if (!this.db) return;
    this.db.close();
    this.db = null;
  }
};
function toFtsQuery(q) {
  const terms = q.replace(/["^*():{}[\]]/g, " ").split(/\s+/).map((t) => t.trim()).filter((t) => t.length > 0).map((t) => `"${t}"`);
  return terms.join(" OR ");
}

// src/index-store.ts
var INDEX_VERSION = 4;
var MAX_EXCERPT_LENGTH = 3500;
var KnowledgeIndex = class _KnowledgeIndex {
  config;
  /** Embedder may be null in FTS-only mode (no provider configured). */
  embedder;
  data;
  dirty = false;
  saveTimer = null;
  fts;
  constructor(config, embedder) {
    this.config = config;
    this.embedder = embedder;
    this.data = {
      version: INDEX_VERSION,
      dimensions: config.dimensions,
      embeddingModel: config.modelSignature,
      entries: {}
    };
    this.fts = new FtsChunkIndex(config.indexDir);
  }
  /** True when no embedder is configured — search runs pure BM25. */
  get isFtsOnly() {
    return this.embedder === null;
  }
  size() {
    const paths = /* @__PURE__ */ new Set();
    for (const entry of Object.values(this.data.entries)) {
      paths.add(`${entry.sourceDir}/${entry.relPath}`);
    }
    return paths.size;
  }
  chunkCount() {
    return Object.keys(this.data.entries).length;
  }
  /**
   * Aggregate all chunks into a per-file view: one entry per indexed file with
   * the merged list of section headings found across its chunks. Used by the
   * overview builder and the knowledge_kb_read resolver — both want file-level data, not
   * chunk-level.
   */
  listFiles() {
    const byPath = /* @__PURE__ */ new Map();
    for (const [key, entry] of Object.entries(this.data.entries)) {
      const absPath = this.absPathFromKey(key);
      let agg = byPath.get(absPath);
      if (!agg) {
        agg = {
          absPath,
          relPath: entry.relPath,
          sourceDir: entry.sourceDir,
          headings: []
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
  static STREAMING_THRESHOLD_BYTES = 256 * 1024 * 1024;
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
  async load() {
    this.fts.load();
    await new Promise((r) => setImmediate(r));
    const indexFile = path2.join(this.config.indexDir, "index.json");
    if (fs2.existsSync(indexFile)) {
      try {
        let parsed = null;
        const size = fs2.statSync(indexFile).size;
        if (size >= _KnowledgeIndex.STREAMING_THRESHOLD_BYTES) {
          parsed = await this.streamLoadJson(indexFile);
        } else {
          const raw = fs2.readFileSync(indexFile, "utf-8");
          parsed = JSON.parse(raw);
        }
        const dimsOk = this.isFtsOnly || parsed?.dimensions === this.config.dimensions;
        const sigOk = this.isFtsOnly || parsed?.embeddingModel === this.config.modelSignature;
        if (parsed && parsed.version === INDEX_VERSION && dimsOk && sigOk) {
          this.data = parsed;
        }
      } catch {
      }
    }
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
  rebuildFtsFromEntries() {
    const chunks = [];
    for (const [key, entry] of Object.entries(this.data.entries)) {
      chunks.push({
        key,
        absPath: this.absPathFromKey(key),
        relPath: entry.relPath,
        sourceDir: entry.sourceDir,
        heading: entry.heading,
        content: entry.excerpt,
        chunkIndex: entry.chunkIndex,
        mtime: entry.mtime
      });
    }
    if (chunks.length > 0) this.fts.upsertMany(chunks);
  }
  streamLoadJson(file) {
    return new Promise((resolve3, reject) => {
      const stream = fs2.createReadStream(file, { highWaterMark: 256 * 1024 });
      const parser = makeParser();
      const assembler = Assembler.connectTo(parser);
      let settled = false;
      const settle = (ok, err) => {
        if (settled) return;
        settled = true;
        if (err) err(new Error("assembler failed"));
        else ok();
      };
      assembler.on("done", (asm) => {
        settle(() => resolve3(asm.current));
      });
      stream.on("error", (e) => settle(() => resolve3(null), () => reject(e)));
      parser.on("error", (e) => settle(() => resolve3(null), () => reject(e)));
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
  async save() {
    fs2.mkdirSync(this.config.indexDir, { recursive: true });
    const finalFile = path2.join(this.config.indexDir, "index.json");
    const tmpFile = finalFile + ".tmp";
    try {
      let serialised;
      try {
        serialised = JSON.stringify(this.data);
      } catch (err) {
        if (err instanceof RangeError) {
          await this.saveStreaming(tmpFile);
          await fs2.promises.rename(tmpFile, finalFile);
          this.dirty = false;
          return;
        }
        throw err;
      }
      await fs2.promises.writeFile(tmpFile, serialised);
      await fs2.promises.rename(tmpFile, finalFile);
      this.dirty = false;
    } catch (err) {
      try {
        fs2.unlinkSync(tmpFile);
      } catch {
      }
      throw err;
    }
  }
  /**
   * Streaming fallback used when the index is too big for `JSON.stringify`
   * to produce a single string. Writes key-by-key through a write stream so
   * no intermediate giant string is ever materialised.
   */
  async saveStreaming(tmpFile) {
    const stream = fs2.createWriteStream(tmpFile);
    let streamError = null;
    stream.once("error", (err) => {
      streamError = err;
    });
    const write = (chunk) => new Promise((resolve3, reject) => {
      if (streamError) {
        reject(streamError);
        return;
      }
      if (stream.write(chunk)) {
        resolve3();
      } else {
        stream.once("drain", () => streamError ? reject(streamError) : resolve3());
      }
    });
    try {
      await write(
        `{"version":${JSON.stringify(this.data.version)},"dimensions":${JSON.stringify(this.data.dimensions)},"embeddingModel":${JSON.stringify(this.data.embeddingModel ?? null)},"entries":{`
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
    await new Promise((resolve3, reject) => {
      stream.end((err) => err ? reject(err) : resolve3());
    });
  }
  scheduleSave() {
    if (this.saveTimer) return;
    this.dirty = true;
    this.saveTimer = setTimeout(() => {
      this.saveTimer = null;
      if (this.dirty) {
        void this.save().catch((err) => {
          console.error(`knowledge-search: scheduled save failed: ${err.message}`);
        });
      }
    }, 5e3);
  }
  /**
   * Build the entry key for a file chunk.
   */
  entryKey(absPath, chunkIndex) {
    return `${absPath}#${chunkIndex}`;
  }
  /**
   * Get the absolute path from an entry key (strip #chunkIndex).
   */
  absPathFromKey(key) {
    const hashIdx = key.lastIndexOf("#");
    return hashIdx >= 0 ? key.slice(0, hashIdx) : key;
  }
  /**
   * Remove all chunks for a given absolute file path from both the vector
   * store and the FTS side-car.
   */
  removeAllChunks(absPath) {
    const prefix = absPath + "#";
    const toRemove = [];
    for (const key of Object.keys(this.data.entries)) {
      if (key.startsWith(prefix)) {
        toRemove.push(key);
      }
    }
    for (const key of toRemove) {
      delete this.data.entries[key];
    }
    try {
      this.fts.deleteByAbsPath(absPath);
    } catch {
    }
    return toRemove.length;
  }
  /**
   * Prepare embedding text for a chunk with title context.
   */
  chunkEmbedText(relPath, heading, chunkText) {
    const title = relPath.replace(/\.[^.]+$/, "").replace(/\//g, " > ");
    const sectionContext = heading && heading !== "intro" ? ` > ${heading}` : "";
    return `Title: ${title}${sectionContext}

${chunkText}`;
  }
  /**
   * 0-indexed inclusive line range a chunk occupies in its source file,
   * as tracked by the chunker.
   */
  chunkLineRange(chunk) {
    return {
      startLine: chunk.startLine,
      endLine: Math.max(chunk.endLine ?? chunk.startLine, chunk.startLine)
    };
  }
  /**
   * Scan all configured directories, find new/changed/removed files, update index.
   */
  async sync(opts) {
    const allFiles = this.scanAllFiles();
    const currentPaths = new Set(allFiles.map((f) => f.absPath));
    let removed = 0;
    const seenRemoved = /* @__PURE__ */ new Set();
    for (const key of Object.keys(this.data.entries)) {
      const absPath = this.absPathFromKey(key);
      if (!currentPaths.has(absPath) && !seenRemoved.has(absPath)) {
        seenRemoved.add(absPath);
        removed += 1;
        this.removeAllChunks(absPath);
      }
    }
    const toProcess = [];
    for (const file of allFiles) {
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
      const allChunkTexts = [];
      const chunkMeta = [];
      for (let fi = 0; fi < toProcess.length; fi++) {
        const file = toProcess[fi];
        for (let ci = 0; ci < file.chunks.length; ci++) {
          const chunk = file.chunks[ci];
          allChunkTexts.push(this.chunkEmbedText(file.relPath, chunk.heading, chunk.text));
          chunkMeta.push({ fileIdx: fi, chunkIdx: ci });
        }
      }
      report?.({
        phase: "scan",
        filesToProcess: toProcess.length,
        unchanged: allFiles.length - toProcess.length - removed,
        totalChunks: allChunkTexts.length
      });
      const allVectors = new Array(allChunkTexts.length).fill(null);
      if (this.embedder) {
        const BATCH_SIZE = 50;
        for (let i = 0; i < allChunkTexts.length; i += BATCH_SIZE) {
          const batchTexts = allChunkTexts.slice(i, i + BATCH_SIZE);
          const vectors = await this.embedder.embedBatch(batchTexts);
          for (let j = 0; j < vectors.length; j++) {
            allVectors[i + j] = vectors[j];
          }
          const lastMeta = chunkMeta[Math.min(i + vectors.length, chunkMeta.length) - 1];
          report?.({
            phase: "embed",
            done: Math.min(i + vectors.length, allChunkTexts.length),
            total: allChunkTexts.length,
            currentFile: lastMeta ? toProcess[lastMeta.fileIdx].relPath : void 0
          });
        }
      } else {
        const lastMeta = chunkMeta[chunkMeta.length - 1];
        report?.({
          phase: "embed",
          done: allChunkTexts.length,
          total: allChunkTexts.length,
          currentFile: lastMeta ? toProcess[lastMeta.fileIdx].relPath : void 0
        });
      }
      const processedFiles = /* @__PURE__ */ new Set();
      for (let i = 0; i < chunkMeta.length; i++) {
        const { fileIdx, chunkIdx } = chunkMeta[i];
        const vector = allVectors[i];
        const storedVector = vector ?? [];
        if (this.embedder && !vector) continue;
        const file = toProcess[fileIdx];
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
          excerpt,
          heading: chunk.heading,
          chunkIndex: chunkIdx,
          startLine,
          endLine
        };
        this.fts.upsert({
          key,
          absPath: file.absPath,
          relPath: file.relPath,
          sourceDir: file.sourceDir,
          heading: chunk.heading,
          content: excerpt,
          chunkIndex: chunkIdx,
          mtime: file.mtime
        });
      }
    }
    if (added + updated + removed > 0) {
      report?.({ phase: "save" });
      await this.save();
    }
    return { added, updated, removed };
  }
  async rebuild() {
    this.data.entries = {};
    try {
      this.fts.clear();
    } catch {
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
  async vectorSearch(query, limit, signal) {
    if (!this.embedder) {
      throw new Error(
        "vectorSearch() requires an embedder \u2014 configure a provider or use search()/hybridSearch() instead."
      );
    }
    const queryVector = await this.embedder.embed(query, signal);
    const scored = [];
    for (const [key, entry] of Object.entries(this.data.entries)) {
      if (!entry.vector) continue;
      const score = dotProduct(queryVector, entry.vector);
      scored.push({ key, absPath: this.absPathFromKey(key), score });
    }
    scored.sort((a, b) => b.score - a.score);
    const matchesByFile = /* @__PURE__ */ new Map();
    const rangesByFile = /* @__PURE__ */ new Map();
    for (const item of scored) {
      matchesByFile.set(item.absPath, (matchesByFile.get(item.absPath) ?? 0) + 1);
      const entry = this.data.entries[item.key];
      if (entry && typeof entry.startLine === "number") {
        const ranges = rangesByFile.get(item.absPath) ?? [];
        ranges.push([entry.startLine + 1, (entry.endLine ?? entry.startLine) + 1]);
        rangesByFile.set(item.absPath, ranges);
      }
    }
    const seenPaths = /* @__PURE__ */ new Set();
    const deduped = [];
    for (const item of scored) {
      if (seenPaths.has(item.absPath)) continue;
      seenPaths.add(item.absPath);
      deduped.push(item);
      if (deduped.length >= limit) break;
    }
    return deduped.filter((s) => s.score > 0.15).map((s) => {
      const entry = this.data.entries[s.key];
      return {
        path: s.absPath,
        score: s.score,
        excerpt: entry.excerpt,
        heading: entry.heading,
        matches: matchesByFile.get(s.absPath) ?? 1,
        lineRanges: (rangesByFile.get(s.absPath) ?? []).sort((a, b) => a[0] - b[0])
      };
    });
  }
  /**
   * Default search path used by the `knowledge_search` tool. Delegates to
   * hybrid (vector + BM25 fused via Reciprocal Rank Fusion).
   */
  async search(query, limit, signal) {
    return this.hybridSearch(query, limit, signal);
  }
  /**
   * Hybrid search: cosine embeddings + FTS5 BM25, fused via Reciprocal Rank
   * Fusion (k=60). Falls back gracefully:
   *   - no FTS hits or empty side-car → pure vector
   *   - embedding call fails (network blip, rate limit) → pure BM25
   *   - both fail → empty
   *
   * Deduplicates so only the best chunk per file is returned.
   */
  async hybridSearch(query, limit, signal) {
    const K = 60;
    const poolSize = Math.max(limit * 5, 50);
    const vecPromise = this.runVectorRanks(query, poolSize, signal).catch((err) => {
      if (process.env.KNOWLEDGE_SEARCH_DEBUG) {
        console.error(`knowledge-search: vector search failed: ${err.message}`);
      }
      return /* @__PURE__ */ new Map();
    });
    let ftsRanks;
    try {
      ftsRanks = this.fts.searchRanks(query, poolSize);
    } catch {
      ftsRanks = /* @__PURE__ */ new Map();
    }
    const vecRanks = await vecPromise;
    if (vecRanks.size === 0 && ftsRanks.size === 0) return [];
    const activeBackends = (vecRanks.size > 0 ? 1 : 0) + (ftsRanks.size > 0 ? 1 : 0);
    const fused = /* @__PURE__ */ new Map();
    for (const [key, r] of vecRanks) {
      fused.set(key, (fused.get(key) ?? 0) + 1 / (K + r));
    }
    for (const [key, r] of ftsRanks) {
      fused.set(key, (fused.get(key) ?? 0) + 1 / (K + r));
    }
    const displayScale = (K + 1) / Math.max(activeBackends, 1);
    const sorted = [...fused.entries()].sort((a, b) => b[1] - a[1]);
    const matchesByFile = /* @__PURE__ */ new Map();
    const rangesByFile = /* @__PURE__ */ new Map();
    for (const [key] of sorted) {
      const absPath = this.absPathFromKey(key);
      matchesByFile.set(absPath, (matchesByFile.get(absPath) ?? 0) + 1);
      const entry = this.data.entries[key];
      if (entry && typeof entry.startLine === "number") {
        const ranges = rangesByFile.get(absPath) ?? [];
        ranges.push([entry.startLine + 1, (entry.endLine ?? entry.startLine) + 1]);
        rangesByFile.set(absPath, ranges);
      }
    }
    const seen = /* @__PURE__ */ new Set();
    const out = [];
    for (const [key, score] of sorted) {
      const entry = this.data.entries[key];
      const absPath = this.absPathFromKey(key);
      if (seen.has(absPath)) continue;
      seen.add(absPath);
      const scaledScore = Math.min(score * displayScale, 1);
      if (entry) {
        out.push({
          path: absPath,
          score: scaledScore,
          excerpt: entry.excerpt,
          heading: entry.heading,
          matches: matchesByFile.get(absPath) ?? 1,
          lineRanges: (rangesByFile.get(absPath) ?? []).sort((a, b) => a[0] - b[0])
        });
      } else {
        out.push({
          path: absPath,
          score: scaledScore,
          excerpt: "",
          heading: "",
          matches: matchesByFile.get(absPath) ?? 1,
          lineRanges: []
        });
      }
      if (out.length >= limit) break;
    }
    return out;
  }
  /**
   * Run the vector side of hybrid search and return ranked keys as a
   * Map<key, rank> (1-based). Kept internal — `vectorSearch()` is the
   * public escape hatch.
   */
  async runVectorRanks(query, poolSize, signal) {
    if (!this.embedder) return /* @__PURE__ */ new Map();
    const queryVector = await this.embedder.embed(query, signal);
    const scored = [];
    for (const [key, entry] of Object.entries(this.data.entries)) {
      if (!entry.vector) continue;
      const score = dotProduct(queryVector, entry.vector);
      if (score <= 0.15) continue;
      scored.push({ key, score });
    }
    scored.sort((a, b) => b.score - a.score);
    const out = /* @__PURE__ */ new Map();
    const n = Math.min(scored.length, poolSize);
    for (let i = 0; i < n; i++) out.set(scored[i].key, i + 1);
    return out;
  }
  /**
   * Update a single file in the index (called by watcher).
   */
  async updateFile(absPath, sourceDir) {
    if (!fs2.existsSync(absPath)) {
      this.removeFile(absPath);
      return;
    }
    const relPath = path2.relative(sourceDir, absPath);
    if (this.shouldSkip(relPath, path2.basename(absPath))) return;
    const stat = fs2.statSync(absPath);
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
    this.removeAllChunks(absPath);
    let vectors;
    if (this.embedder) {
      const texts = chunks.map((c) => this.chunkEmbedText(relPath, c.heading, c.text));
      vectors = await this.embedder.embedBatch(texts);
    } else {
      vectors = new Array(chunks.length).fill(null);
    }
    for (let i = 0; i < chunks.length; i++) {
      const vector = vectors[i];
      if (this.embedder && !vector) continue;
      const storedVector = vector ?? [];
      const key = this.entryKey(absPath, i);
      const excerpt = chunks[i].text.slice(0, MAX_EXCERPT_LENGTH);
      const { startLine, endLine } = this.chunkLineRange(chunks[i]);
      this.data.entries[key] = {
        relPath,
        sourceDir,
        mtime: stat.mtimeMs,
        vector: storedVector,
        excerpt,
        heading: chunks[i].heading,
        chunkIndex: i,
        startLine,
        endLine
      };
      this.fts.upsert({
        key,
        absPath,
        relPath,
        sourceDir,
        heading: chunks[i].heading,
        content: excerpt,
        chunkIndex: i,
        mtime: stat.mtimeMs
      });
    }
    this.scheduleSave();
  }
  removeFile(absPath) {
    const removed = this.removeAllChunks(absPath);
    if (removed > 0) {
      this.scheduleSave();
    }
  }
  /** Alias for removeFile — removes all data for a file path. */
  deleteFile(absPath) {
    this.removeFile(absPath);
  }
  /** Flush pending saves and release resources. Awaits any in-flight save. */
  async close() {
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
    }
  }
  // -----------------------------------------------------------------------
  // Scanning
  // -----------------------------------------------------------------------
  scanAllFiles() {
    const results = [];
    for (const dir of this.config.dirs) {
      this.walkDir(dir, dir, results);
    }
    return results;
  }
  walkDir(currentDir, sourceDir, results) {
    let entries;
    try {
      entries = fs2.readdirSync(currentDir, { withFileTypes: true });
    } catch {
      return;
    }
    for (const entry of entries) {
      const absPath = path2.join(currentDir, entry.name);
      if (entry.isDirectory()) {
        if (this.config.excludeDirs.includes(entry.name) || entry.name.startsWith(".")) {
          continue;
        }
        this.walkDir(absPath, sourceDir, results);
      } else if (entry.isFile()) {
        const ext = path2.extname(entry.name).toLowerCase();
        if (!this.config.fileExtensions.includes(ext)) continue;
        const relPath = path2.relative(sourceDir, absPath);
        if (this.shouldSkip(relPath, entry.name)) continue;
        try {
          const stat = fs2.statSync(absPath);
          results.push({ absPath, relPath, sourceDir, mtime: stat.mtimeMs });
        } catch {
        }
      }
    }
  }
  shouldSkip(relPath, _basename) {
    const parts = relPath.split(path2.sep);
    for (const part of parts) {
      if (this.config.excludeDirs.includes(part) || part.startsWith(".")) {
        return true;
      }
    }
    return false;
  }
  readFileContent(absPath) {
    try {
      const content = fs2.readFileSync(absPath, "utf-8");
      return content.replace(/^---\n[\s\S]*?\n---\n?/, "");
    } catch {
      return null;
    }
  }
};
function dotProduct(a, b) {
  let sum = 0;
  const len = Math.min(a.length, b.length);
  for (let i = 0; i < len; i++) {
    sum += a[i] * b[i];
  }
  return sum;
}

// src/overview.ts
import * as fs3 from "node:fs";
import * as path3 from "node:path";
var STOPWORDS = /* @__PURE__ */ new Set([
  // Common English stopwords.
  "the",
  "and",
  "for",
  "are",
  "but",
  "not",
  "you",
  "all",
  "can",
  "had",
  "her",
  "was",
  "one",
  "our",
  "out",
  "day",
  "get",
  "has",
  "him",
  "his",
  "how",
  "its",
  "may",
  "new",
  "now",
  "old",
  "see",
  "two",
  "way",
  "who",
  "boy",
  "did",
  "use",
  "man",
  "with",
  "this",
  "that",
  "from",
  "they",
  "them",
  "were",
  "have",
  "what",
  "when",
  "your",
  "which",
  "their",
  "there",
  "about",
  "into",
  "than",
  "more",
  "some",
  "just",
  "like",
  "will",
  "also",
  "been",
  "over",
  "only",
  "then",
  "well",
  "other",
  "these",
  "would",
  "could",
  "should",
  "being",
  "where",
  "while",
  "still",
  "very",
  "most",
  "much",
  "such",
  "many",
  "even",
  "make",
  "made",
  "used",
  "using",
  "does",
  "each",
  "both",
  "here",
  "want",
  "need",
  "back",
  "take",
  "come",
  "came",
  "give",
  "given",
  "find",
  "found",
  "last",
  "same",
  "work",
  "works",
  // Markdown/note boilerplate.
  "note",
  "notes",
  "readme",
  "about",
  "index",
  "todo",
  "todos",
  "done",
  "draft",
  "drafts",
  "markdown",
  "file",
  "files",
  "folder",
  "folders",
  "section",
  "sections",
  // Generic doc words that rarely disambiguate folders.
  "page",
  "pages",
  "link",
  "links",
  "item",
  "items",
  "list",
  "lists",
  "content",
  "contents",
  "overview",
  "summary",
  "intro",
  "introduction",
  "details",
  "detail",
  "example",
  "examples"
]);
function tokenize(text) {
  const out = [];
  const parts = text.toLowerCase().split(/[^a-z0-9]+/);
  for (const p of parts) {
    if (p.length < 3 || p.length > 24) continue;
    if (STOPWORDS.has(p)) continue;
    if (/^\d+$/.test(p)) continue;
    out.push(p);
  }
  return out;
}
function bucketFolder(relPath, maxDepth) {
  const posix = relPath.split(path3.sep).join("/");
  const parts = posix.split("/");
  const dirParts = parts.slice(0, -1);
  if (dirParts.length === 0) return "";
  return dirParts.slice(0, maxDepth).join("/");
}
function extractKeywords(filesByFolder, maxKeywords) {
  const folderTf = /* @__PURE__ */ new Map();
  const df = /* @__PURE__ */ new Map();
  const totalFolders = filesByFolder.size;
  for (const [folder, files] of filesByFolder) {
    const tf = /* @__PURE__ */ new Map();
    for (const f of files) {
      const basename4 = path3.basename(f.relPath, path3.extname(f.relPath));
      for (const tok of tokenize(basename4)) {
        tf.set(tok, (tf.get(tok) ?? 0) + 2);
      }
      const headings = f.headings.slice(0, 6);
      for (const h of headings) {
        for (const tok of tokenize(h)) {
          tf.set(tok, (tf.get(tok) ?? 0) + 1);
        }
      }
    }
    folderTf.set(folder, tf);
    for (const term of tf.keys()) {
      df.set(term, (df.get(term) ?? 0) + 1);
    }
  }
  const result = /* @__PURE__ */ new Map();
  for (const [folder, tf] of folderTf) {
    const scored = [];
    for (const [term, count] of tf) {
      const docFreq = df.get(term) ?? 1;
      const idf = Math.log((totalFolders + 1) / (docFreq + 1)) + 1;
      const score = count * idf;
      scored.push({ term, score });
    }
    scored.sort((a, b) => b.score - a.score || a.term.localeCompare(b.term));
    result.set(
      folder,
      scored.slice(0, maxKeywords).map((s) => s.term)
    );
  }
  return result;
}
var CONTEXT_NOTE_CANDIDATES = ["NAPKIN.md", "README.md", "_about.md", "ABOUT.md"];
var FOLDER_ABOUT_CANDIDATES = ["_about.md", "README.md"];
function readTrimmed(absPath, maxChars) {
  try {
    const raw = fs3.readFileSync(absPath, "utf-8");
    const body = raw.replace(/^---\n[\s\S]*?\n---\n?/, "").trim();
    if (!body) return void 0;
    if (body.length <= maxChars) return body;
    return body.slice(0, maxChars).trimEnd() + "\u2026";
  } catch {
    return void 0;
  }
}
function findContextNote(sourceDir) {
  for (const name of CONTEXT_NOTE_CANDIDATES) {
    const p = path3.join(sourceDir, name);
    if (fs3.existsSync(p)) {
      const text = readTrimmed(p, 400);
      if (text) return text;
    }
  }
  return void 0;
}
function findFolderAbout(sourceDir, folder) {
  if (!folder) return void 0;
  for (const name of FOLDER_ABOUT_CANDIDATES) {
    const p = path3.join(sourceDir, folder, name);
    if (fs3.existsSync(p)) {
      const text = readTrimmed(p, 240);
      if (text) return text;
    }
  }
  return void 0;
}
function buildOverview(files, sourceDirs, opts = {}) {
  const maxDepth = Math.max(1, opts.maxDepth ?? 2);
  const maxFolders = Math.max(1, opts.maxFoldersPerDir ?? 20);
  const maxKeywords = Math.max(1, opts.maxKeywordsPerFolder ?? 5);
  const bySource = /* @__PURE__ */ new Map();
  for (const sd of sourceDirs) bySource.set(sd, /* @__PURE__ */ new Map());
  for (const f of files) {
    let bucket = bySource.get(f.sourceDir);
    if (!bucket) {
      bucket = /* @__PURE__ */ new Map();
      bySource.set(f.sourceDir, bucket);
    }
    const folder = bucketFolder(f.relPath, maxDepth);
    let arr = bucket.get(folder);
    if (!arr) {
      arr = [];
      bucket.set(folder, arr);
    }
    arr.push(f);
  }
  const sources = [];
  let totalNotes = 0;
  for (const [sourceDir, folders] of bySource) {
    const keywords = extractKeywords(folders, maxKeywords);
    const folderList = [];
    for (const [folder, fileList] of folders) {
      folderList.push({
        path: folder,
        noteCount: fileList.length,
        keywords: keywords.get(folder) ?? [],
        aboutText: findFolderAbout(sourceDir, folder)
      });
    }
    folderList.sort(
      (a, b) => b.noteCount - a.noteCount || a.path.localeCompare(b.path)
    );
    const trimmedFolders = folderList.slice(0, maxFolders);
    const sourceTotal = folderList.reduce((s, f) => s + f.noteCount, 0);
    totalNotes += sourceTotal;
    sources.push({
      dir: sourceDir,
      displayName: path3.basename(sourceDir) || sourceDir,
      contextNote: findContextNote(sourceDir),
      folders: trimmedFolders
    });
  }
  return { sources, totalNotes };
}
function formatOverview(overview) {
  if (overview.totalNotes === 0) return "";
  const lines = [];
  lines.push("## Knowledge-search vault overview");
  lines.push(
    `You have a local knowledge base indexed by pi-knowledge-search. Use the \`knowledge_search\` tool for semantic/keyword lookup and \`knowledge_kb_read\` to pull a note by name or \`[[wikilink]]\`.`
  );
  lines.push("");
  const home = process.env.HOME || "";
  for (const src of overview.sources) {
    const dirDisplay = home && src.dir.startsWith(home) ? src.dir.replace(home, "~") : src.dir;
    lines.push(`### ${dirDisplay}`);
    if (src.contextNote) {
      lines.push("");
      lines.push(src.contextNote);
    }
    lines.push("");
    if (src.folders.length === 0) {
      lines.push("_(no indexed files)_");
      lines.push("");
      continue;
    }
    for (const folder of src.folders) {
      const label = folder.path || "(root)";
      lines.push(`- **${label}/** \u2014 ${folder.noteCount} note${folder.noteCount === 1 ? "" : "s"}`);
      if (folder.keywords.length > 0) {
        lines.push(`  - keywords: ${folder.keywords.join(", ")}`);
      }
      if (folder.aboutText) {
        const oneLine = folder.aboutText.replace(/\s+/g, " ").slice(0, 180);
        lines.push(`  - ${oneLine}`);
      }
    }
    lines.push("");
  }
  return lines.join("\n").trimEnd();
}

// src/kb-reader.ts
import * as fs4 from "node:fs";
import * as path4 from "node:path";
function normalizeRef(ref) {
  let s = ref.trim();
  s = s.replace(/^\[\[/, "").replace(/\]\]$/, "");
  const pipeIdx = s.indexOf("|");
  if (pipeIdx !== -1) s = s.slice(0, pipeIdx);
  s = s.trim();
  let subheading;
  const hashIdx = s.indexOf("#");
  if (hashIdx !== -1) {
    subheading = s.slice(hashIdx + 1).trim() || void 0;
    s = s.slice(0, hashIdx).trim();
  }
  return { ref: s, subheading };
}
var DEFAULT_EXTS = [".md", ".txt"];
function normBasename(p, exts) {
  const base = path4.basename(p);
  for (const ext of exts) {
    if (base.toLowerCase().endsWith(ext)) {
      return base.slice(0, base.length - ext.length).toLowerCase();
    }
  }
  return base.toLowerCase();
}
function hasKnownExt(s, exts) {
  const lower = s.toLowerCase();
  return exts.some((e) => lower.endsWith(e));
}
function resolveNote(rawRef, indexedFiles, opts = {}) {
  const exts = (opts.fileExtensions ?? DEFAULT_EXTS).map(
    (e) => e.startsWith(".") ? e : "." + e
  );
  const maxMatches = opts.maxMatches ?? 10;
  const { ref, subheading } = normalizeRef(rawRef);
  if (!ref) {
    return { matches: [], unique: false, normalizedRef: "", subheading };
  }
  const matches = [];
  const seen = /* @__PURE__ */ new Set();
  function push(m) {
    if (seen.has(m.absPath)) return;
    seen.add(m.absPath);
    matches.push(m);
  }
  if (path4.isAbsolute(ref)) {
    const hit = indexedFiles.find((f) => f.absPath === ref);
    if (hit) push({ ...hit, reason: "absolute" });
  }
  if (opts.cwd && !path4.isAbsolute(ref) && (ref.includes("/") || ref.includes(path4.sep))) {
    const abs = path4.resolve(opts.cwd, ref);
    const hit = indexedFiles.find((f) => f.absPath === abs);
    if (hit) push({ ...hit, reason: "absolute" });
  }
  for (const f of indexedFiles) {
    const rel = f.relPath;
    if (rel === ref) {
      push({ ...f, reason: "relative-to-source" });
      continue;
    }
    if (hasKnownExt(ref, exts)) continue;
    for (const ext of exts) {
      if (rel === ref + ext) {
        push({ ...f, reason: "relative-to-source" });
        break;
      }
    }
  }
  const refLower = ref.toLowerCase();
  const refBaseLower = normBasename(ref, exts);
  for (const f of indexedFiles) {
    const b = path4.basename(f.relPath);
    if (b === ref || !hasKnownExt(ref, exts) && exts.some((e) => b === ref + e)) {
      push({ ...f, reason: "basename-exact" });
    }
  }
  for (const f of indexedFiles) {
    const bLower = normBasename(f.relPath, exts);
    if (bLower === refBaseLower) {
      push({ ...f, reason: "basename-ci" });
    }
  }
  if (ref.includes("/")) {
    const targets = hasKnownExt(ref, exts) ? [ref] : exts.map((e) => ref + e);
    const targetsLower = targets.map((t) => t.toLowerCase());
    for (const f of indexedFiles) {
      const rLower = f.relPath.toLowerCase();
      for (const t of targetsLower) {
        if (rLower.endsWith("/" + t) || rLower === t) {
          push({ ...f, reason: "relpath-suffix" });
          break;
        }
      }
    }
  }
  if (matches.length === 0) {
    for (const f of indexedFiles) {
      const bLower = normBasename(f.relPath, exts);
      if (bLower.includes(refBaseLower) || refBaseLower.includes(bLower)) {
        push({ ...f, reason: "substring" });
      } else if (f.relPath.toLowerCase().includes(refLower)) {
        push({ ...f, reason: "substring" });
      }
      if (matches.length >= maxMatches) break;
    }
  }
  const trimmed = matches.slice(0, maxMatches);
  const highConfidenceReasons = [
    "absolute",
    "relative-to-source",
    "basename-exact",
    "basename-ci",
    "relpath-suffix"
  ];
  const highConfidence = trimmed.filter((m) => highConfidenceReasons.includes(m.reason));
  const unique = highConfidence.length === 1;
  return { matches: trimmed, unique, normalizedRef: ref, subheading };
}
function readNote(absPath, opts = {}) {
  const maxBytes = opts.maxBytes ?? 64 * 1024;
  const buf = fs4.readFileSync(absPath);
  const total = buf.length;
  if (buf.length <= maxBytes) {
    return {
      path: absPath,
      content: buf.toString("utf-8"),
      truncated: false,
      totalBytes: total
    };
  }
  let end = maxBytes;
  while (end > 0 && (buf[end] & 192) === 128) {
    end--;
  }
  const slice = buf.subarray(0, end);
  const text = slice.toString("utf-8");
  const lastNewline = text.lastIndexOf("\n");
  const newlineNearby = lastNewline >= 0 && lastNewline > end - 2048;
  const safe = newlineNearby ? text.slice(0, lastNewline) : text;
  return {
    path: absPath,
    content: safe,
    truncated: true,
    totalBytes: total
  };
}

// src/index.ts
function renderProgressBar(current, total, width = 24) {
  const filled = total > 0 ? Math.round(current / total * width) : width;
  return `\x1B[36m${"\u2588".repeat(filled)}\x1B[2m${"\u2591".repeat(width - filled)}\x1B[0m`;
}
function index_default(pi) {
  let index = null;
  let currentConfig = null;
  let sessionCwd;
  let syncDone = false;
  let workerExitExpected = false;
  let activeWorker = null;
  function injectOverview(ctx, force) {
    if (!index || !currentConfig) return { status: "skipped", reason: "not configured" };
    if (!force && !currentConfig.overview.inject) {
      return { status: "skipped", reason: "overview.inject=false" };
    }
    if (index.size() === 0) return { status: "skipped", reason: "index is empty" };
    if (!force) {
      const alreadyInjected = ctx.sessionManager.getEntries().some(
        (e) => e.type === "custom_message" && e.customType === "knowledge-overview"
      );
      if (alreadyInjected) return { status: "skipped", reason: "already injected" };
    }
    const overview = buildOverview(index.listFiles(), currentConfig.dirs, {
      maxDepth: currentConfig.overview.maxDepth,
      maxFoldersPerDir: currentConfig.overview.maxFoldersPerDir,
      maxKeywordsPerFolder: currentConfig.overview.maxKeywordsPerFolder
    });
    const text = formatOverview(overview);
    if (!text) return { status: "skipped", reason: "empty overview" };
    pi.sendMessage({
      customType: "knowledge-overview",
      content: text,
      display: true,
      details: {
        totalNotes: overview.totalNotes,
        sourceCount: overview.sources.length,
        forced: force
      }
    });
    return {
      status: "injected",
      totalNotes: overview.totalNotes,
      sourceCount: overview.sources.length
    };
  }
  pi.registerMessageRenderer(
    "knowledge-lookup",
    (message, { outputPad }, theme) => {
      const details = message.details;
      const summary = details?.summary;
      if (!summary) return void 0;
      const isError = details?.error === true;
      const box = new Box(outputPad, 1, (boxTheme) => theme.bg(isError ? "toolErrorBg" : "toolSuccessBg", boxTheme));
      const [label, ...rest] = summary.split("\u2014");
      const rendered = rest.length ? `${theme.fg(isError ? "error" : "success", label.trim())} \u2014${theme.fg("text", rest.join("\u2014"))}` : theme.fg("text", summary);
      box.addChild(new Text(rendered, 0, 0));
      return box;
    }
  );
  function lookupDisplayPath(absPath) {
    if (sessionCwd) {
      const rel = relative2(sessionCwd, absPath);
      if (rel !== "" && !rel.startsWith("..") && !isAbsolute2(rel)) return rel;
    }
    return absPath.replace(process.env.HOME || "", "~");
  }
  const LOOKUP_TOP_K = 5;
  const LOOKUP_MIN_SCORE = 0.1;
  const LOOKUP_PREVIEW_CHARS = 600;
  pi.on("before_agent_start", async (event) => {
    if (!currentConfig?.autoInject) return;
    if (!index || index.size() === 0) return;
    if (!event.prompt.trim()) return;
    let results;
    try {
      results = (await index.search(event.prompt, LOOKUP_TOP_K)).filter(
        (r) => r.score >= LOOKUP_MIN_SCORE
      );
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      return {
        message: {
          customType: "knowledge-lookup",
          content: `[pi-knowledge-search] Knowledge lookup failed: ${msg}`,
          display: true,
          details: { summary: `Knowledge lookup failed \u2014 ${msg}`, error: true }
        }
      };
    }
    if (results.length === 0) return;
    const contextBlock = results.map((r) => {
      const heading = r.heading && r.heading !== "intro" ? ` > ${r.heading}` : "";
      return `### ${lookupDisplayPath(r.path)}${heading}

${r.excerpt.slice(0, LOOKUP_PREVIEW_CHARS)}`;
    }).join("\n\n");
    const fileSummaries = results.map((r) => {
      const display = lookupDisplayPath(r.path);
      const ranges = r.lineRanges ?? [];
      if (ranges.length > 0) {
        const fmt = ranges.map(([s, e]) => s === e ? `${s}` : `${s}-${e}`).join(",");
        return `${display}:${fmt}`;
      }
      return `${display} (${r.matches ?? 1})`;
    });
    const summary = `Knowledge lookup \u2014 ${fileSummaries.join(", ")}`;
    return {
      message: {
        customType: "knowledge-lookup",
        content: `[pi-knowledge-search] Automatic knowledge lookup triggered by the user's message above.
Retrieved ${results.length} chunk${results.length === 1 ? "" : "s"} via hybrid search (BM25 + vector). These are search hits, not statements from the user.

` + contextBlock,
        display: true,
        details: { summary, error: false }
      }
    };
  });
  pi.on("session_start", async (event, ctx) => {
    sessionCwd = ctx.cwd;
    try {
      currentConfig = loadConfig(sessionCwd);
    } catch {
      return;
    }
    if (!currentConfig) return;
    index = new KnowledgeIndex(currentConfig, createEmbedder());
    const indexLoaded = index.load();
    indexLoaded.then(() => {
      if (event.reason === "startup" && currentConfig && index && index.chunkCount() > 0 && !currentConfig.autoInject) {
        const file = readRawConfig();
        file.autoInject = true;
        saveConfig(file, sessionCwd);
        currentConfig.autoInject = true;
        ctx.ui.notify("Knowledge lookup auto-injection enabled", "info");
      }
      try {
        injectOverview(ctx, false);
      } catch (err) {
        const msg = err instanceof Error ? err.message : String(err);
        console.error(`knowledge-search: overview injection failed: ${msg}`);
      }
    }).catch((err) => {
      const msg = err instanceof Error ? err.message : String(err);
      console.error(`knowledge-search: index load failed: ${msg}`);
    });
    const MAX_WORKER_RESTARTS = 3;
    const RESTART_WINDOW_MS = 6e4;
    let workerRestartCount = 0;
    let workerRestartWindowStart = Date.now();
    function spawnWorker() {
      const workerPath = join6(import.meta.dirname, "..", "dist", "sync-worker.mjs");
      const worker = fork(workerPath, [], {
        stdio: ["ignore", "pipe", "pipe", "ipc"],
        // Suppress "node:sqlite is experimental" warning — node:sqlite is stable
        // enough for our read/write usage and the warning pollutes pi startup.
        execArgv: ["--no-warnings=ExperimentalWarning"],
        // Forward sessionCwd so the worker resolves the same project-local
        // settings.json (pi-knowledge-search.localPath).
        env: { ...process.env, KNOWLEDGE_SEARCH_CWD: sessionCwd ?? process.env.KNOWLEDGE_SEARCH_CWD ?? "" }
      });
      activeWorker = worker;
      let stdout = "";
      let stderrBuf = "";
      const report = (msg, level = "error") => {
        if (ctx.hasUI) {
          ctx.ui.notify(msg, level);
        } else {
          console.error(msg);
        }
      };
      worker.stdout?.on("data", (chunk) => {
        stdout += chunk.toString();
      });
      worker.stderr?.on("data", (chunk) => {
        stderrBuf += chunk.toString();
      });
      worker.on("error", (err) => {
        report(`knowledge-search: worker error: ${err.message}`);
      });
      worker.on("exit", async (code, signal) => {
        syncDone = true;
        if (activeWorker === worker) activeWorker = null;
        if (code === 0 && stdout) {
          try {
            const result = JSON.parse(stdout);
            await index.load();
            const changes = result.added + result.updated + result.removed;
            if (changes > 0) {
              ctx.ui.setStatus(
                "knowledge-search",
                `Index: +${result.added} ~${result.updated} -${result.removed} (${result.size} files, ${result.chunks} chunks)`
              );
              setTimeout(() => ctx.ui.setStatus("knowledge-search", ""), 5e3);
            }
          } catch {
          }
          const stderrTail = stderrBuf.trim().split("\n").filter(Boolean).pop() ?? "";
          if (stderrTail) {
            report(`knowledge-search: ${stderrTail}`, "warning");
          }
        } else if (code !== 0 && !workerExitExpected) {
          const now = Date.now();
          if (now - workerRestartWindowStart > RESTART_WINDOW_MS) {
            workerRestartCount = 0;
            workerRestartWindowStart = now;
          }
          workerRestartCount++;
          const stderrTail = stderrBuf.trim().split("\n").filter(Boolean).pop() ?? "";
          const detail = stderrTail ? ` (${stderrTail})` : "";
          if (workerRestartCount > MAX_WORKER_RESTARTS) {
            report(
              `knowledge-search: indexing worker crashed ${workerRestartCount}x within ${RESTART_WINDOW_MS / 1e3}s, giving up${detail}`
            );
          } else {
            report(
              `knowledge-search: indexing worker failed (code=${code}, signal=${signal}), retrying ${workerRestartCount}/${MAX_WORKER_RESTARTS}${detail}`,
              "warning"
            );
            setTimeout(() => {
              if (!workerExitExpected) spawnWorker();
            }, 2e3);
          }
        }
        stderrBuf = "";
      });
      worker.unref();
    }
    spawnWorker();
  });
  pi.on("session_shutdown", async () => {
    workerExitExpected = true;
    await index?.close();
  });
  const KS_SUBCOMMANDS = [
    { value: "add", label: "add", description: "Add directories to the index" },
    { value: "exclude", label: "exclude", description: "Manage excluded directory names (-<name> removes)" },
    { value: "index", label: "index", description: "Incrementally index new/changed files" },
    { value: "clear", label: "clear", description: "Clear the index and reset config to defaults" },
    { value: "on", label: "on", description: "Enable per-turn knowledge lookup injection" },
    { value: "off", label: "off", description: "Disable per-turn knowledge lookup injection" },
    { value: "help", label: "help", description: "Show all /knowledge-search commands" }
  ];
  function getSubcommandCompletions(prefix) {
    const matches = KS_SUBCOMMANDS.filter((s) => s.value.startsWith(prefix)).map((s) => ({
      value: s.value,
      label: s.label,
      description: s.description
    }));
    return matches.length > 0 ? matches : null;
  }
  function readRawConfig() {
    try {
      return JSON.parse(fs5.readFileSync(getConfigPath(sessionCwd), "utf-8"));
    } catch {
      return {};
    }
  }
  function resolveUserPath(p) {
    const home = process.env.HOME || "";
    const expanded = p.startsWith("~") ? home + p.slice(1) : p;
    return resolve2(sessionCwd ?? process.cwd(), expanded);
  }
  async function ensureIndexLoaded() {
    if (index) return;
    index = new KnowledgeIndex(currentConfig, createEmbedder());
    await index.load();
  }
  function renderStatusWidget(ctx) {
    const theme = ctx.ui.theme;
    const label = (text) => theme.fg("dim", text.padEnd(20));
    const lines = [theme.bold("pi-knowledge-search"), ""];
    lines.push(
      "  " + label("Embedding engine:") + theme.fg("success", EMBEDDING_MODEL) + theme.fg("dim", "  (local ONNX via Transformers.js \u2014 fixed, not configurable)")
    );
    if (index) {
      lines.push("  " + label("Indexed:") + theme.fg("success", `${index.size()} files \xB7 ${index.chunkCount()} chunks`));
    } else {
      lines.push("  " + label("Indexed:") + theme.fg("dim", "0 files (run /knowledge-search index)"));
    }
    lines.push("", "  " + theme.bold("Directories indexed:"));
    const dirs = currentConfig?.dirs ?? [];
    if (dirs.length) {
      for (const dir of dirs) lines.push("    " + theme.fg("muted", dir));
    } else {
      lines.push("    " + theme.fg("dim", "(none \u2014 add with /knowledge-search add <dir>)"));
    }
    lines.push("", "  " + theme.bold("Excluded directories:"));
    const excludes = currentConfig?.excludeDirs ?? [];
    if (excludes.length) {
      for (const name of excludes) lines.push("    " + theme.fg("muted", name));
    } else {
      lines.push("    " + theme.fg("dim", "(none \u2014 add with /knowledge-search exclude <name>)"));
    }
    lines.push("", "  " + theme.bold("File extensions:"));
    const exts = currentConfig?.fileExtensions ?? [];
    lines.push("    " + theme.fg("muted", exts.join(" ")));
    lines.push(
      "",
      "  " + label("Lookup inject:") + (currentConfig?.autoInject ? theme.fg("success", "enabled") : theme.fg("warning", "disabled")) + theme.fg("dim", `  topK=${5}  minScore=${0.1}`)
    );
    lines.push(
      "",
      "  " + label("Config:") + theme.fg("dim", getConfigPath(sessionCwd)),
      "  " + label("Index:") + theme.fg("dim", currentConfig?.indexDir ?? "")
    );
    ctx.ui.setWidget("knowledge-search-status", lines);
  }
  async function handleAdd(parts, ctx) {
    const raw = parts.slice(1).join(" ").split(/[\s,]+/).map((p) => p.trim()).filter(Boolean);
    if (raw.length === 0) {
      ctx.ui.notify("Usage: /knowledge-search add <dir> [<dir>...]", "warning");
      return;
    }
    const resolved = raw.map(resolveUserPath);
    for (const dir of resolved) {
      if (!fs5.existsSync(dir)) {
        ctx.ui.notify(`Path not found: ${dir}`, "error");
        return;
      }
    }
    const file = readRawConfig();
    const dirs = /* @__PURE__ */ new Set([...file.dirs ?? [], ...resolved]);
    file.dirs = [...dirs];
    saveConfig(file, sessionCwd);
    const added = resolved.filter((d) => !(currentConfig?.dirs ?? []).includes(d));
    if (currentConfig) {
      currentConfig.dirs = [...dirs];
    } else {
      currentConfig = loadConfig(sessionCwd);
    }
    const newCount = added.length;
    ctx.ui.notify(
      `Added ${newCount} director${newCount === 1 ? "y" : "ies"} \xB7 ${dirs.size} total. Run /knowledge-search index to index them.`,
      "info"
    );
  }
  function handleExclude(parts, ctx) {
    const expression = parts.slice(1).join(" ").trim();
    const file = readRawConfig();
    if (!expression) {
      const excludes2 = file.excludeDirs ?? [];
      if (!excludes2.length) {
        ctx.ui.notify("No excluded directories. Add one with: /knowledge-search exclude <name>", "info");
        return;
      }
      const theme = ctx.ui.theme;
      const lines = [theme.bold(`Excluded directories (${excludes2.length})`), ""];
      for (const name of excludes2) lines.push("  " + theme.fg("muted", name));
      lines.push("", theme.fg("dim", "Remove with: /knowledge-search exclude -<name>"));
      ctx.ui.setWidget("knowledge-search-exclude", lines);
      return;
    }
    if (expression.startsWith("-")) {
      const target = expression.slice(1);
      const before = (file.excludeDirs ?? []).length;
      file.excludeDirs = (file.excludeDirs ?? []).filter((n) => n !== target);
      if ((file.excludeDirs ?? []).length === before) {
        ctx.ui.notify(`Not excluded: ${target}`, "warning");
        return;
      }
      saveConfig(file, sessionCwd);
      if (currentConfig) currentConfig.excludeDirs = file.excludeDirs;
      ctx.ui.notify(
        `Removed exclude: ${target} \xB7 ${file.excludeDirs.length} remain. Run /knowledge-search index to re-apply.`,
        "info"
      );
      return;
    }
    const excludes = file.excludeDirs ?? [];
    if (excludes.includes(expression)) {
      ctx.ui.notify(`Already excluded: ${expression}`, "warning");
      return;
    }
    excludes.push(expression);
    file.excludeDirs = excludes;
    saveConfig(file, sessionCwd);
    if (currentConfig) currentConfig.excludeDirs = excludes;
    ctx.ui.notify(
      `Added exclude: ${expression} \xB7 ${excludes.length} total. Run /knowledge-search index to re-apply.`,
      "info"
    );
  }
  async function handleIndex(ctx) {
    if (!currentConfig || currentConfig.dirs.length === 0) {
      ctx.ui.notify("No directories configured. Run /knowledge-search add <dir> first.", "warning");
      return;
    }
    const clearProgressUI = () => {
      ctx.ui.setStatus("knowledge-search", void 0);
      ctx.ui.setWidget("knowledge-search", void 0);
    };
    try {
      await ensureIndexLoaded();
      if (!isTransformersModelCached(EMBEDDING_MODEL)) {
        ctx.ui.notify(
          `\u23F3 Loading embedding model: ${EMBEDDING_MODEL} \u2014 first run downloads it (~111 MB, this can take a few minutes)`,
          "info"
        );
      }
      const theme = ctx.ui.theme;
      const verb = theme.fg("accent", "Indexing");
      const { added, updated, removed } = await index.sync({
        onProgress: (p) => {
          if (p.phase === "scan") {
            const label = p.filesToProcess > 0 ? `Found ${p.filesToProcess} file(s) to index \xB7 ${p.unchanged} unchanged \xB7 ${p.totalChunks} chunks` : `Nothing to index \xB7 ${p.unchanged} files unchanged`;
            ctx.ui.setStatus("knowledge-search", `\u25A0 Scanning\u2026 ${label}`);
            ctx.ui.setWidget("knowledge-search", [verb, theme.fg("dim", label)]);
            return;
          }
          if (p.phase === "embed") {
            const percent = p.total ? Math.round(p.done / p.total * 100) : 100;
            const bar = renderProgressBar(p.done, p.total);
            ctx.ui.setStatus(
              "knowledge-search",
              `\u25A0 Indexing ${percent}% \u2502 ${p.done}/${p.total} chunks`
            );
            ctx.ui.setWidget("knowledge-search", [
              `${verb}  ${bar}  ${theme.fg("success", `${percent}%`)}`,
              `${theme.fg("dim", "file:    ")}${p.currentFile ?? "\u2026"}`,
              `${theme.fg("dim", "chunks:  ")}${theme.fg("success", String(p.done))}/${p.total}`
            ]);
            return;
          }
          ctx.ui.setStatus("knowledge-search", "\u25A0 Saving index...");
        }
      });
      syncDone = true;
      clearProgressUI();
      if (currentConfig && index.chunkCount() > 0 && !currentConfig.autoInject) {
        const file = readRawConfig();
        file.autoInject = true;
        saveConfig(file, sessionCwd);
        currentConfig.autoInject = true;
        ctx.ui.notify("Knowledge lookup auto-injection enabled", "info");
      }
      const changed = added + updated + removed;
      ctx.ui.notify(
        changed === 0 ? `\u2705 Up to date \xB7 ${index.size()} files (${index.chunkCount()} chunks) indexed` : `\u2705 Indexed ${added} new \xB7 ${updated} changed \xB7 ${removed} removed \xB7 ${index.size()} files (${index.chunkCount()} chunks) total`,
        "info"
      );
    } catch (err) {
      clearProgressUI();
      ctx.ui.notify(`Index failed: ${err.message}`, "error");
    }
  }
  async function handleClear(ctx) {
    const confirmed = await ctx.ui.confirm(
      "Clear knowledge search?",
      "Deletes all project data (vector index + keyword side-car + config) and resets project settings to defaults, including any localPath override in .pi/settings.json. Re-index afterwards with /knowledge-search add + index. The shared HuggingFace model cache is not touched."
    );
    if (!confirmed) {
      ctx.ui.notify("Clear cancelled.", "info");
      return;
    }
    const indexDir = currentConfig?.indexDir ?? getIndexDir(sessionCwd);
    const configPath = getConfigPath(sessionCwd);
    if (activeWorker) {
      workerExitExpected = true;
      activeWorker.kill();
      activeWorker = null;
    }
    try {
      await index?.close();
    } catch {
    }
    index = null;
    currentConfig = null;
    syncDone = false;
    try {
      if (fs5.existsSync(indexDir)) {
        for (const entry of fs5.readdirSync(indexDir)) {
          fs5.rmSync(join6(indexDir, entry), { recursive: true, force: true });
        }
      }
    } catch (err) {
      ctx.ui.notify(`Clear failed while deleting ${indexDir}: ${err.message}`, "error");
      return;
    }
    try {
      fs5.rmSync(configPath, { force: true });
    } catch (err) {
      ctx.ui.notify(`Clear failed while deleting ${configPath}: ${err.message}`, "error");
      return;
    }
    const settingsFile = join6(sessionCwd ?? process.cwd(), ".pi", "settings.json");
    try {
      if (fs5.existsSync(settingsFile)) {
        const settings = JSON.parse(fs5.readFileSync(settingsFile, "utf-8"));
        if (settings && typeof settings === "object" && "pi-knowledge-search" in settings) {
          delete settings["pi-knowledge-search"];
          fs5.writeFileSync(settingsFile, JSON.stringify(settings, null, 2) + "\n");
        }
      }
    } catch {
    }
    saveConfig(
      {
        dirs: [],
        fileExtensions: DEFAULT_FILE_EXTENSIONS,
        excludeDirs: ["node_modules", ".git", ".obsidian", ".trash"]
      },
      sessionCwd
    );
    statusWidgetVisible = false;
    ctx.ui.setWidget("knowledge-search-status", void 0);
    ctx.ui.notify(
      `\u2705 Cleared all project data (${indexDir}) and reset settings to defaults`,
      "info"
    );
  }
  function handleHelp(ctx) {
    const theme = ctx.ui.theme;
    const lines = [theme.bold("/knowledge-search commands"), ""];
    for (const s of KS_SUBCOMMANDS) {
      lines.push("  " + theme.fg("accent", s.label.padEnd(10)) + theme.fg("dim", s.description));
    }
    lines.push("", theme.fg("dim", "Bare /knowledge-search shows the current status."));
    ctx.ui.setWidget("knowledge-search-help", lines);
  }
  let statusWidgetVisible = false;
  pi.registerCommand("knowledge-search", {
    description: "knowledge-search: (status) | add <dir> | exclude <name> | index | clear | on | off | help",
    getArgumentCompletions: (prefix) => getSubcommandCompletions(prefix),
    handler: async (args, ctx) => {
      const parts = (args || "").trim().split(/\s+/);
      const subcommand = parts[0] || "";
      if (subcommand === "add") {
        await handleAdd(parts, ctx);
        return;
      }
      if (subcommand === "exclude") {
        handleExclude(parts, ctx);
        return;
      }
      if (subcommand === "index") {
        await handleIndex(ctx);
        return;
      }
      if (subcommand === "clear") {
        await handleClear(ctx);
        return;
      }
      if (subcommand === "on" || subcommand === "off") {
        const enabled = subcommand === "on";
        const file = readRawConfig();
        file.autoInject = enabled;
        saveConfig(file, sessionCwd);
        if (currentConfig) currentConfig.autoInject = enabled;
        ctx.ui.notify(
          enabled ? "Knowledge lookup injection enabled" : "Knowledge lookup injection disabled (re-enabled automatically at next startup while the index has vectors)",
          "info"
        );
        return;
      }
      if (subcommand === "help") {
        handleHelp(ctx);
        return;
      }
      if (subcommand) {
        ctx.ui.notify(`Unknown /knowledge-search command: ${subcommand}. Try /knowledge-search help`, "error");
        return;
      }
      if (statusWidgetVisible) {
        statusWidgetVisible = false;
        ctx.ui.setWidget("knowledge-search-status", void 0);
        return;
      }
      renderStatusWidget(ctx);
      statusWidgetVisible = true;
    }
  });
  pi.registerCommand("knowledge-overview", {
    description: "Rebuild and re-inject the knowledge-search vault overview (use after config changes or vault growth)",
    handler: async (_args, ctx) => {
      try {
        const result = injectOverview(ctx, true);
        if (result.status === "injected") {
          ctx.ui.notify(
            `Overview re-injected: ${result.totalNotes} notes, ${result.sourceCount} source dir(s).`,
            "info"
          );
        } else {
          ctx.ui.notify(`Overview not injected: ${result.reason}.`, "warning");
        }
      } catch (err) {
        const msg = err instanceof Error ? err.message : String(err);
        ctx.ui.notify(`Overview injection failed: ${msg}`, "error");
      }
    }
  });
  const searchParams = Type.Object({
    query: Type.String({ description: "Natural language search query" }),
    limit: Type.Optional(
      Type.Number({
        description: "Max results to return (default 8, max 20)"
      })
    )
  });
  pi.registerTool({
    name: "knowledge_search",
    label: "Knowledge Search",
    description: "Semantic search over local knowledge files. Returns the most relevant file excerpts for a natural language query. Use for finding past notes, investigations, decisions, documentation, and context. Prefer this over grep when you need conceptual or fuzzy matching rather than exact text.",
    promptGuidelines: [
      'Use knowledge_search for conceptual queries (e.g. "how did we handle X", "what was decided about Y"). Use grep/read for exact text or known filenames.'
    ],
    parameters: searchParams,
    async execute(toolCallId, params, signal) {
      if (!index || index.size() === 0) {
        const msg = !index ? "knowledge-search is not configured. The user can run /knowledge-search add <dir> to set it up." : !syncDone ? "Index is still syncing in the background. Try again in a moment." : "Index is empty.";
        return { content: [{ type: "text", text: msg }], details: {} };
      }
      const limit = Math.min(params.limit ?? 8, 20);
      try {
        const results = await index.search(params.query, limit, signal);
        if (results.length === 0) {
          return {
            content: [
              {
                type: "text",
                text: `No relevant results found for: "${params.query}"`
              }
            ],
            details: {}
          };
        }
        const home = process.env.HOME || "";
        const output = results.map((r, i) => {
          const displayPath = r.path.replace(home, "~");
          const score = (r.score * 100).toFixed(1);
          const heading = r.heading && r.heading !== "intro" ? ` > ${r.heading}` : "";
          return `### ${i + 1}. ${displayPath}${heading} (${score}% match)

${r.excerpt}`;
        }).join("\n\n---\n\n");
        const sourceInfo = `${index.size()} files, ${index.chunkCount()} chunks indexed`;
        const header = `Found ${results.length} results for "${params.query}" (${sourceInfo}):

`;
        return {
          content: [{ type: "text", text: header + output }],
          details: { resultCount: results.length, indexSize: index?.size() ?? 0 }
        };
      } catch (err) {
        throw new Error(`knowledge-search failed: ${err.message}`);
      }
    }
  });
  const readParams = Type.Object({
    name: Type.String({
      description: "Note reference: filename, basename, relative path, or [[wikilink]]. Examples: 'evergreen/hybrid-search', 'Hybrid search.md', '[[Hybrid search]]', '[[evergreen/hybrid-search|alias]]'."
    }),
    max_bytes: Type.Optional(
      Type.Number({
        description: "Truncate output to at most this many bytes (default 65536)."
      })
    )
  });
  pi.registerTool({
    name: "knowledge_kb_read",
    label: "KB Read",
    description: "Read a note from the knowledge base by name, relative path, or [[wikilink]]. Resolves fuzzy references without needing an absolute path \u2014 use this when you know the note's title/filename but not its full path on disk.",
    promptGuidelines: [
      "Use knowledge_kb_read when a note is referenced by name or [[wikilink]] \u2014 don't run find/grep first.",
      "Use the standard `read` tool for non-indexed files or when you already have an absolute path."
    ],
    parameters: readParams,
    async execute(_toolCallId, params) {
      if (!index || index.size() === 0) {
        const msg = !index ? "knowledge-search is not configured. Run /knowledge-search add <dir> to set it up." : !syncDone ? "Index is still syncing in the background. Try again in a moment." : "Index is empty.";
        return { content: [{ type: "text", text: msg }], details: {} };
      }
      const result = resolveNote(params.name, index.listFiles(), {
        fileExtensions: currentConfig?.fileExtensions,
        cwd: sessionCwd
      });
      if (result.matches.length === 0) {
        return {
          content: [
            {
              type: "text",
              text: `No note matched "${result.normalizedRef}". Try knowledge_search with a topic query to find related notes.`
            }
          ],
          details: {}
        };
      }
      if (!result.unique && result.matches.length > 1) {
        const home2 = process.env.HOME || "";
        const listed = result.matches.map((m, i) => {
          const display2 = home2 && m.absPath.startsWith(home2) ? m.absPath.replace(home2, "~") : m.absPath;
          return `${i + 1}. ${display2}  _(${m.reason})_`;
        }).join("\n");
        return {
          content: [
            {
              type: "text",
              text: `"${result.normalizedRef}" is ambiguous. ${result.matches.length} candidates:

${listed}

Call knowledge_kb_read again with a more specific path (e.g. the exact relative path) to disambiguate.`
            }
          ],
          details: { candidates: result.matches.map((m) => m.absPath) }
        };
      }
      const match = result.matches[0];
      let note;
      try {
        note = readNote(match.absPath, {
          maxBytes: params.max_bytes
        });
      } catch (err) {
        const msg = err instanceof Error ? err.message : String(err);
        return {
          content: [{ type: "text", text: `Failed to read ${match.absPath}: ${msg}` }],
          details: {}
        };
      }
      const home = process.env.HOME || "";
      const display = home && note.path.startsWith(home) ? note.path.replace(home, "~") : note.path;
      const truncNote = note.truncated ? `

_(truncated: showing first ${note.content.length} of ${note.totalBytes} bytes)_` : "";
      const section = result.subheading ? ` \u2014 section "${result.subheading}"` : "";
      const fuzzyNote = !result.unique ? `

_(fuzzy match via ${match.reason} \u2014 if this isn't the note you meant, re-run knowledge_kb_read with a more specific path)_` : "";
      const header = `# ${display}${section}${truncNote}${fuzzyNote}

`;
      return {
        content: [{ type: "text", text: header + note.content }],
        details: {
          resolvedPath: match.absPath,
          truncated: note.truncated
        }
      };
    }
  });
}
export {
  index_default as default
};
//# sourceMappingURL=index.js.map
