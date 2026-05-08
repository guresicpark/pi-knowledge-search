/**
 * Folder + keyword overview of indexed knowledge directories.
 *
 * Inspired by napkin's `napkin overview` — gives the agent a one-shot map
 * of the knowledge base at session start, so it knows what's worth searching
 * for before it asks.
 *
 * Keyword extraction is corpus-local TF-IDF over a lightweight feature set
 * (relative path segments + section headings). No embeddings, no LLM calls —
 * this runs on whatever the index already has loaded.
 */

import * as fs from "node:fs";
import * as path from "node:path";

export interface FileFacts {
  /** Absolute path — used only for disambiguation / de-dup. */
  absPath: string;
  /** Path relative to its source dir, POSIX-style. */
  relPath: string;
  /** Which configured source dir this file came from. */
  sourceDir: string;
  /** Section headings collected from all chunks of this file. */
  headings: string[];
}

export interface OverviewFolder {
  /** Display path, POSIX-style, relative to the source dir. "" means the root of the source dir. */
  path: string;
  /** Unique note count in this folder (recursive: includes nested files at deeper paths). */
  noteCount: number;
  /** Top keywords surfaced by TF-IDF over folder contents. */
  keywords: string[];
  /**
   * Optional description read from an `_about.md` / `README.md` file at this folder's
   * root. First ~240 chars, trimmed.
   */
  aboutText?: string;
}

export interface OverviewSource {
  /** Absolute path of the configured source directory. */
  dir: string;
  /** Short display name (basename of `dir`, with `~` substitution done upstream). */
  displayName: string;
  /**
   * Root-level context note if present (`NAPKIN.md`, `README.md`, or `_about.md`).
   * First ~400 chars, trimmed.
   */
  contextNote?: string;
  /** Folders grouped at the configured depth, sorted by noteCount desc. */
  folders: OverviewFolder[];
}

export interface Overview {
  sources: OverviewSource[];
  /** Total unique note count across all sources. */
  totalNotes: number;
}

export interface BuildOverviewOptions {
  /**
   * Max folder depth from each source dir root. A file at `a/b/c/d.md` with
   * depth=2 is grouped under `a/b`. Files shallower than the depth use their
   * actual depth. Default: 2.
   */
  maxDepth?: number;
  /** Max folders to include per source dir (ranked by noteCount desc). Default: 20. */
  maxFoldersPerDir?: number;
  /** Max keywords per folder. Default: 5. */
  maxKeywordsPerFolder?: number;
}

const STOPWORDS = new Set<string>([
  // Common English stopwords.
  "the", "and", "for", "are", "but", "not", "you", "all", "can", "had", "her",
  "was", "one", "our", "out", "day", "get", "has", "him", "his", "how", "its",
  "may", "new", "now", "old", "see", "two", "way", "who", "boy", "did", "use",
  "man", "with", "this", "that", "from", "they", "them", "were", "have", "what",
  "when", "your", "which", "their", "there", "about", "into", "than", "more",
  "some", "just", "like", "will", "also", "been", "over", "only", "then", "well",
  "other", "these", "would", "could", "should", "being", "where", "while",
  "still", "very", "most", "much", "such", "many", "even", "make", "made",
  "used", "using", "does", "each", "both", "here", "want", "need", "back",
  "take", "come", "came", "give", "given", "find", "found", "last", "same",
  "work", "works",
  // Markdown/note boilerplate.
  "note", "notes", "readme", "about", "index", "todo", "todos", "done", "draft",
  "drafts", "markdown", "file", "files", "folder", "folders", "section",
  "sections",
  // Generic doc words that rarely disambiguate folders.
  "page", "pages", "link", "links", "item", "items", "list", "lists", "content",
  "contents", "overview", "summary", "intro", "introduction", "details",
  "detail", "example", "examples",
]);

/** Tokenize a string into lowercased alphanumeric tokens suitable for keyword scoring. */
function tokenize(text: string): string[] {
  const out: string[] = [];
  // Split on anything that isn't a letter, digit, or within-word apostrophe.
  const parts = text.toLowerCase().split(/[^a-z0-9]+/);
  for (const p of parts) {
    if (p.length < 3 || p.length > 24) continue;
    if (STOPWORDS.has(p)) continue;
    if (/^\d+$/.test(p)) continue; // pure numbers
    out.push(p);
  }
  return out;
}

/** Bucket a file into a folder path given a max depth. POSIX-style paths. */
export function bucketFolder(relPath: string, maxDepth: number): string {
  const posix = relPath.split(path.sep).join("/");
  const parts = posix.split("/");
  // parts[parts.length - 1] is the file itself.
  const dirParts = parts.slice(0, -1);
  if (dirParts.length === 0) return "";
  return dirParts.slice(0, maxDepth).join("/");
}

/**
 * Extract top keywords for each folder using corpus-local TF-IDF.
 * - TF: raw count of term in folder (from filename tokens + heading tokens)
 * - IDF: log((N + 1) / (df + 1)) where N = total folders, df = folders containing term
 *
 * Returns a map from folderPath -> keywords (highest score first, capped at maxKeywords).
 */
function extractKeywords(
  filesByFolder: Map<string, FileFacts[]>,
  maxKeywords: number
): Map<string, string[]> {
  const folderTf = new Map<string, Map<string, number>>();
  const df = new Map<string, number>();
  const totalFolders = filesByFolder.size;

  for (const [folder, files] of filesByFolder) {
    const tf = new Map<string, number>();
    for (const f of files) {
      // Filename basename (without extension), then folder name components
      // closest to the file. These often contain the most topical tokens.
      const basename = path.basename(f.relPath, path.extname(f.relPath));
      for (const tok of tokenize(basename)) {
        tf.set(tok, (tf.get(tok) ?? 0) + 2); // filename weighted higher
      }
      // Headings (first few — later sections tend to be boilerplate like "References").
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

  const result = new Map<string, string[]>();
  for (const [folder, tf] of folderTf) {
    const scored: Array<{ term: string; score: number }> = [];
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

const CONTEXT_NOTE_CANDIDATES = ["NAPKIN.md", "README.md", "_about.md", "ABOUT.md"];
const FOLDER_ABOUT_CANDIDATES = ["_about.md", "README.md"];

function readTrimmed(absPath: string, maxChars: number): string | undefined {
  try {
    const raw = fs.readFileSync(absPath, "utf-8");
    // Strip frontmatter.
    const body = raw.replace(/^---\n[\s\S]*?\n---\n?/, "").trim();
    if (!body) return undefined;
    if (body.length <= maxChars) return body;
    return body.slice(0, maxChars).trimEnd() + "…";
  } catch {
    return undefined;
  }
}

function findContextNote(sourceDir: string): string | undefined {
  for (const name of CONTEXT_NOTE_CANDIDATES) {
    const p = path.join(sourceDir, name);
    if (fs.existsSync(p)) {
      const text = readTrimmed(p, 400);
      if (text) return text;
    }
  }
  return undefined;
}

function findFolderAbout(sourceDir: string, folder: string): string | undefined {
  if (!folder) return undefined; // root already covered by contextNote
  for (const name of FOLDER_ABOUT_CANDIDATES) {
    const p = path.join(sourceDir, folder, name);
    if (fs.existsSync(p)) {
      const text = readTrimmed(p, 240);
      if (text) return text;
    }
  }
  return undefined;
}

export function buildOverview(
  files: FileFacts[],
  sourceDirs: string[],
  opts: BuildOverviewOptions = {}
): Overview {
  const maxDepth = Math.max(1, opts.maxDepth ?? 2);
  const maxFolders = Math.max(1, opts.maxFoldersPerDir ?? 20);
  const maxKeywords = Math.max(1, opts.maxKeywordsPerFolder ?? 5);

  // Group files by source dir, then by folder bucket.
  const bySource = new Map<string, Map<string, FileFacts[]>>();
  for (const sd of sourceDirs) bySource.set(sd, new Map());

  for (const f of files) {
    let bucket = bySource.get(f.sourceDir);
    if (!bucket) {
      // File's source dir isn't in the configured list — still include it so we
      // don't silently drop data. Use the file's sourceDir as its own group.
      bucket = new Map();
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

  const sources: OverviewSource[] = [];
  let totalNotes = 0;

  for (const [sourceDir, folders] of bySource) {
    // Run TF-IDF per source dir (keeps keyword comparison local to each vault).
    const keywords = extractKeywords(folders, maxKeywords);

    const folderList: OverviewFolder[] = [];
    for (const [folder, fileList] of folders) {
      folderList.push({
        path: folder,
        noteCount: fileList.length,
        keywords: keywords.get(folder) ?? [],
        aboutText: findFolderAbout(sourceDir, folder),
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
      displayName: path.basename(sourceDir) || sourceDir,
      contextNote: findContextNote(sourceDir),
      folders: trimmedFolders,
    });
  }

  return { sources, totalNotes };
}

export function formatOverview(overview: Overview): string {
  if (overview.totalNotes === 0) return "";
  const lines: string[] = [];
  lines.push("## Knowledge-search vault overview");
  lines.push(
    `You have a local knowledge base indexed by pi-knowledge-search. Use the ` +
      `\`knowledge_search\` tool for semantic/keyword lookup and \`kb_read\` to ` +
      `pull a note by name or \`[[wikilink]]\`.`
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
      lines.push(`- **${label}/** — ${folder.noteCount} note${folder.noteCount === 1 ? "" : "s"}`);
      if (folder.keywords.length > 0) {
        lines.push(`  - keywords: ${folder.keywords.join(", ")}`);
      }
      if (folder.aboutText) {
        // Compact the about text to one line for the list form.
        const oneLine = folder.aboutText.replace(/\s+/g, " ").slice(0, 180);
        lines.push(`  - ${oneLine}`);
      }
    }
    lines.push("");
  }
  return lines.join("\n").trimEnd();
}
