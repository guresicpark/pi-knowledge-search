import * as fs from "node:fs";
import * as path from "node:path";
import {
  EMBEDDING_DIMENSIONS,
  EMBEDDING_MODEL,
  CODE_EMBEDDING_MODEL,
  type EmbedGroup,
} from "./embedder.js";

export interface Config {
  /** Directories to index */
  dirs: string[];
  /** File extensions to index (with dots) */
  fileExtensions: string[];
  /**
   * Extensions routed to the code embedding group (jina); everything else
   * in `fileExtensions` goes to the text group (nomic). Mirrors
   * pi-local-rag's code/text extension split.
   */
  codeExtensions: string[];
  /** Directory names to skip */
  excludeDirs: string[];
  /** Embedding dimensions — both models' fixed 768 */
  dimensions: number;
  /**
   * Signature of the engine that produces the embeddings
   * (`transformers:nomic-ai/nomic-embed-text-v1.5+jinaai/jina-embeddings-v2-base-code:768`).
   * The index persists the signature its vectors were built with; a mismatch
   * on load drops incompatible embeddings and forces a re-embed (text-only
   * nomic vectors from the legacy single-model engine are kept — see
   * index-store's legacy migration). Constant — the engine is not
   * configurable.
   */
  modelSignature: string;
  /** Where to store the index */
  indexDir: string;
  /** Inject a knowledge lookup into the conversation on every user prompt. Default: true. */
  autoInject: boolean;
  /** Session-start overview injection settings */
  overview: OverviewConfig;
}

export interface OverviewConfig {
  /** Inject a folder+keyword summary as a custom message on session start. Default: true. */
  inject: boolean;
  /** Max folder depth to group files into. Default: 2. */
  maxDepth: number;
  /** Max folders shown per source dir (ranked by note count). Default: 20. */
  maxFoldersPerDir: number;
  /** Max keywords surfaced per folder. Default: 5. */
  maxKeywordsPerFolder: number;
}

/**
 * Default file extensions — the union of pi-local-rag's two embedding
 * groups (DEFAULT_TEXT_EXTS): code extensions are embedded by
 * jina-embeddings-v2-base-code, everything else by nomic. Binary document
 * types (.pdf, .docx) also go to nomic there but require extraction
 * libraries (unpdf/mammoth) and are therefore not included here.
 */
export const DEFAULT_CODE_EXTENSIONS = [
  ".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs",
  ".py", ".rs", ".go", ".java", ".kt", ".kts", ".scala",
  ".c", ".cc", ".cpp", ".cxx", ".h", ".hpp", ".hxx",
  ".cs", ".fs", ".vb",
  ".swift", ".m", ".mm",
  ".rb", ".php", ".pl", ".lua", ".dart", ".ex", ".exs", ".erl", ".clj", ".cljs", ".edn",
  ".vue", ".svelte", ".astro", ".twig",
  ".css", ".scss", ".sass", ".less",
  ".sh", ".bash", ".zsh", ".fish", ".ps1",
  ".sql", ".graphql", ".gql", ".proto",
  ".tf", ".hcl",
];

/** Text/prose + data/config extensions → embedded by nomic (DEFAULT_DOC_EXTS). */
export const DEFAULT_DOC_EXTENSIONS = [
  ".md", ".mdx", ".txt", ".rst",
  ".html", ".htm",
  ".json", ".jsonc", ".yaml", ".yml", ".toml", ".ini", ".xml", ".csv", ".tsv",
  ".env", ".gitignore", ".dockerfile",
];

export const DEFAULT_FILE_EXTENSIONS = [...DEFAULT_CODE_EXTENSIONS, ...DEFAULT_DOC_EXTENSIONS];

/**
 * Which embedding group a file belongs to: code extensions route to the
 * jina code model, everything else to nomic — pi-local-rag's classifyFile.
 * `codeExtensions` defaults to the built-in code list.
 */
export function classifyFileGroup(filePath: string, codeExtensions?: string[]): EmbedGroup {
  const ext = path.extname(filePath).toLowerCase();
  if (!ext) return "text";
  const codeExts = codeExtensions ?? DEFAULT_CODE_EXTENSIONS;
  return codeExts.includes(ext) ? "code" : "text";
}

/**
 * Raw shape stored in the config file. There is no embedding-engine
 * configuration — nomic (text) + jina (code) are always used — so a legacy
 * `provider` or `dimensions` key is ignored (with a one-time warning).
 */
export interface ConfigFile {
  dirs?: string[];
  fileExtensions?: string[];
  codeExtensions?: string[];
  excludeDirs?: string[];
  autoInject?: boolean;
  overview?: Partial<OverviewConfig>;
}

// Storage is project-local: config lives at {cwd}/.pi/knowledge-search.json and
// the index at {cwd}/.pi/knowledge-search/. Evaluated lazily per call so cwd
// changes at runtime (tests, sandboxes) are honored.
function defaultConfigFile(cwd?: string): string {
  return path.join(cwd || process.cwd(), ".pi", "knowledge-search.json");
}
function defaultIndexDir(cwd?: string): string {
  return path.join(cwd || process.cwd(), ".pi", "knowledge-search");
}

/**
 * Resolve a project-local base directory for pi-knowledge-search storage.
 *
 * Resolution order (highest priority first):
 *   1. {cwd}/.pi/settings.json → "pi-knowledge-search".localPath
 *
 * When set, config is stored at {base}/config.json and index at {base}/index.
 * Environment variables (KNOWLEDGE_SEARCH_CONFIG / KNOWLEDGE_SEARCH_INDEX_DIR)
 * take precedence.
 *
 * Returns null when no project-local override is configured.
 */
/**
 * Emit a warning when a settings block contains keys outside a known
 * schema. Catches silent typos like `LocalPath` vs `localPath` — an unknown
 * key is usually a misspelled known key that got silently ignored, leaving
 * the user wondering why their config didn't take effect.
 */
function warnUnknownKeys(block: unknown, blockName: string, knownKeys: readonly string[]): void {
  if (!block || typeof block !== "object") return;
  const unknown = Object.keys(block as Record<string, unknown>).filter((k) => !knownKeys.includes(k));
  if (unknown.length === 0) return;
  console.error(
    `pi-knowledge-search: ignoring unknown key(s) in settings.json "${blockName}" block: ${unknown.join(", ")} (expected: ${knownKeys.join(", ")})`,
  );
}

// Keys pi-knowledge-search reads from settings.json. The bulk of config lives
// in a separate config.json (see getConfigPath) — only localPath comes from
// the settings.json block directly.
const PI_KNOWLEDGE_SEARCH_SETTINGS_KEYS = ["localPath"] as const;

export function resolveLocalBase(cwd?: string): string | null {
  if (!cwd) return null;
  try {
    const raw = fs.readFileSync(path.join(cwd, ".pi", "settings.json"), "utf-8");
    const settings = JSON.parse(raw) ?? {};

    // Package-specific override.
    const ks = settings["pi-knowledge-search"];
    warnUnknownKeys(ks, "pi-knowledge-search", PI_KNOWLEDGE_SEARCH_SETTINGS_KEYS);
    if (ks && typeof ks === "object" && typeof ks.localPath === "string" && ks.localPath) {
      return ks.localPath;
    }
  } catch {
    // No settings file, unreadable, or malformed — fall through to the default.
  }
  return null;
}

/**
 * Resolve the config file path. Priority:
 *   1. KNOWLEDGE_SEARCH_CONFIG env var (explicit override)
 *   2. Project-local base ({base}/config.json)
 *   3. Project default ({cwd}/.pi/knowledge-search.json; process.cwd() when no cwd)
 */
export function getConfigPath(cwd?: string): string {
  if (process.env.KNOWLEDGE_SEARCH_CONFIG) return process.env.KNOWLEDGE_SEARCH_CONFIG;
  const base = resolveLocalBase(cwd);
  if (base) return path.join(base, "config.json");
  return defaultConfigFile(cwd);
}

/**
 * Resolve the index directory. Priority matches getConfigPath().
 */
export function getIndexDir(cwd?: string): string {
  if (process.env.KNOWLEDGE_SEARCH_INDEX_DIR) return process.env.KNOWLEDGE_SEARCH_INDEX_DIR;
  const base = resolveLocalBase(cwd);
  if (base) return path.join(base, "index");
  return defaultIndexDir(cwd);
}

/**
 * Load config from file, with env var overrides.
 * Returns null if no config file exists (needs setup).
 *
 * @param cwd - Optional working directory; enables project-local resolution.
 */
export function loadConfig(cwd?: string): Config | null {
  const configPath = getConfigPath(cwd);

  // Try config file first
  let file: ConfigFile | null = null;
  if (fs.existsSync(configPath)) {
    try {
      file = JSON.parse(fs.readFileSync(configPath, "utf-8"));
    } catch {
      // Corrupted file
    }
  }

  // Check env var fallback for dirs
  const envDirs = process.env.KNOWLEDGE_SEARCH_DIRS;

  if (!file && !envDirs) {
    return null; // Not configured yet
  }

  // Build config: file values, then env overrides
  const home = process.env.HOME || "/tmp";
  const resolvePath = (p: string) => p.replace(/^~/, home);

  const dirs = (envDirs ? envDirs.split(",").map((d) => d.trim()) : (file?.dirs ?? []))
    .map(resolvePath)
    .filter(Boolean);

  if (dirs.length === 0) return null;

  // Legacy-config migration: versions before the dual-model engine shipped
  // the text-group-only default as an explicit `fileExtensions` list. A
  // config still carrying exactly that list is treated as stale — left
  // alone it would silently hide every code extension (.ts, .sql, …) from
  // the jina side, so it upgrades to the current union default. Any other
  // explicit list is a deliberate narrowing and stays authoritative.
  let fileExtensionsFromFile = file?.fileExtensions?.map((e) => e.toLowerCase());
  if (fileExtensionsFromFile && sameExtensionSet(fileExtensionsFromFile, DEFAULT_DOC_EXTENSIONS)) {
    console.error(
      "pi-knowledge-search: upgrading legacy fileExtensions (text-only default) to the dual-group default — code extensions (.ts, .py, .sql, …) are now indexed with jina-code. Re-run /knowledge index to pick them up."
    );
    fileExtensionsFromFile = undefined;
  }

  const fileExtensions = (envStr("KNOWLEDGE_SEARCH_EXTENSIONS")
    ?.split(",")
    .map((e) => e.trim().toLowerCase()) ??
    fileExtensionsFromFile ??
    DEFAULT_FILE_EXTENSIONS)
    .filter(Boolean);

  const excludeDirs = envStr("KNOWLEDGE_SEARCH_EXCLUDE")
    ?.split(",")
    .map((d) => d.trim()) ??
    file?.excludeDirs ?? ["node_modules", ".git", ".obsidian", ".trash"];

  // Code-group extensions (jina model). User-configurable, defaulting to
  // pi-local-rag's code list; fileExtensions governs what is scanned, this
  // governs which embedding model a scanned file goes to.
  const codeExtensions =
    envStr("KNOWLEDGE_SEARCH_CODE_EXTENSIONS")
      ?.split(",")
      .map((e) => e.trim().toLowerCase())
      .filter(Boolean) ??
      file?.codeExtensions?.map((e) => e.toLowerCase()) ??
      DEFAULT_CODE_EXTENSIONS;

  // The embedding engine is not configurable — always nomic (text) + jina
  // (code). Legacy `provider` / `dimensions` keys in old configs are
  // ignored with a one-time notice so their owners aren't left wondering.
  const legacy = file as Record<string, unknown> | null;
  if (legacy && (legacy.provider !== undefined || legacy.dimensions !== undefined)) {
    console.error(
      "pi-knowledge-search: ignoring \"provider\"/\"dimensions\" config keys — the embedding engine is always nomic-embed-text-v1.5 (text) + jina-embeddings-v2-base-code (code), local ONNX."
    );
  }

  const indexDir = getIndexDir(cwd);

  // Overview config — cheap and usually wanted, so defaults lean on.
  const overviewFile = file?.overview ?? {};
  const overview = {
    inject: envBool("KNOWLEDGE_SEARCH_OVERVIEW_INJECT") ?? overviewFile.inject ?? true,
    maxDepth:
      envInt("KNOWLEDGE_SEARCH_OVERVIEW_MAX_DEPTH") ?? overviewFile.maxDepth ?? 2,
    maxFoldersPerDir:
      envInt("KNOWLEDGE_SEARCH_OVERVIEW_MAX_FOLDERS") ??
      overviewFile.maxFoldersPerDir ??
      20,
    maxKeywordsPerFolder:
      envInt("KNOWLEDGE_SEARCH_OVERVIEW_MAX_KEYWORDS") ??
      overviewFile.maxKeywordsPerFolder ??
      5,
  };

  return {
    dirs,
    fileExtensions,
    codeExtensions,
    excludeDirs: excludeDirs,
    dimensions: EMBEDDING_DIMENSIONS,
    modelSignature: `transformers:${EMBEDDING_MODEL}+${CODE_EMBEDDING_MODEL}:${EMBEDDING_DIMENSIONS}`,
    indexDir,
    autoInject: envBool("KNOWLEDGE_SEARCH_AUTO_INJECT") ?? file?.autoInject ?? true,
    overview,
  };
}

/**
 * Save config to file.
 *
 * @param config - Config data to write.
 * @param cwd - Optional working directory; enables project-local resolution.
 */
export function saveConfig(config: ConfigFile, cwd?: string): void {
  const configPath = getConfigPath(cwd);
  const dir = path.dirname(configPath);
  fs.mkdirSync(dir, { recursive: true });
  fs.writeFileSync(configPath, JSON.stringify(config, null, 2) + "\n");
}

function envStr(key: string): string | undefined {
  const v = process.env[key]?.trim();
  return v || undefined;
}

/** Order-insensitive comparison of two lowercase extension lists. */
function sameExtensionSet(a: string[], b: string[]): boolean {
  if (a.length !== b.length) return false;
  const sortedA = [...a].sort();
  const sortedB = [...b].sort();
  return sortedA.every((ext, i) => ext === sortedB[i]);
}

function envInt(key: string): number | undefined {
  const v = envStr(key);
  return v ? parseInt(v, 10) : undefined;
}

/** Boolean env: 1/true/yes/on -> true, 0/false/no/off -> false, else undefined. */
function envBool(key: string): boolean | undefined {
  const v = envStr(key)?.toLowerCase();
  if (!v) return undefined;
  if (["1", "true", "yes", "on"].includes(v)) return true;
  if (["0", "false", "no", "off"].includes(v)) return false;
  return undefined;
}
