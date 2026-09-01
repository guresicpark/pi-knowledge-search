import * as fs from "node:fs";
import * as path from "node:path";

export interface Config {
  /** Directories to index */
  dirs: string[];
  /** File extensions to index (with dots) */
  fileExtensions: string[];
  /** Directory names to skip */
  excludeDirs: string[];
  /** Embedding dimensions */
  dimensions: number;
  /** Embedding provider config; null = FTS-only keyword search */
  provider: ProviderConfig | null;
  /**
   * Signature of the engine that produces the embeddings (`type:model:dimensions`).
   * The index persists the signature its vectors were built with; a mismatch
   * on load removes all existing embeddings and forces a full re-embed.
   * Null in FTS-only mode.
   */
  modelSignature: string | null;
  /** Where to store the index */
  indexDir: string;
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

/** The only embedding engine: local ONNX inference via Transformers.js. */
export type ProviderConfig = { type: "transformers"; model: string };

/**
 * Raw shape stored in the config file. `provider.type` is kept as a plain
 * string so legacy configs naming a removed provider (openai, bedrock, …)
 * surface a helpful migration error instead of failing silently.
 */
export interface ConfigFile {
  dirs?: string[];
  fileExtensions?: string[];
  excludeDirs?: string[];
  dimensions?: number;
  overview?: Partial<OverviewConfig>;
  provider?: { type: string; model?: string };
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

  const fileExtensions = envStr("KNOWLEDGE_SEARCH_EXTENSIONS")
    ?.split(",")
    .map((e) => e.trim()) ??
    file?.fileExtensions ?? [".md", ".txt"];

  const excludeDirs = envStr("KNOWLEDGE_SEARCH_EXCLUDE")
    ?.split(",")
    .map((d) => d.trim()) ??
    file?.excludeDirs ?? ["node_modules", ".git", ".obsidian", ".trash"];

  let provider: ProviderConfig | null = null;
  if (file?.provider) {
    if (file.provider.type !== "transformers") {
      throw new Error(
        `Unsupported embedding provider "${file.provider.type}". Only local search is supported: use { "type": "transformers" } (local ONNX embeddings via Transformers.js), or remove the provider block entirely for FTS-only keyword search.`
      );
    }
    provider = {
      type: "transformers",
      model:
        envStr("KNOWLEDGE_SEARCH_TRANSFORMERS_MODEL") ??
        file.provider.model ??
        "nomic-ai/nomic-embed-text-v1.5",
    };
  }

  // Dimensions resolve after the provider — the transformers models are
  // fixed at 768 dims and cannot be truncated.
  const dimensions = envInt("KNOWLEDGE_SEARCH_DIMENSIONS") ??
    file?.dimensions ??
    (provider ? 768 : 512);

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
    excludeDirs: excludeDirs,
    dimensions,
    provider,
    modelSignature: provider ? `${provider.type}:${provider.model ?? ""}:${dimensions}` : null,
    indexDir,
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
