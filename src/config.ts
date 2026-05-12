import * as fs from "node:fs";
import * as path from "node:path";
import type { KnowledgeBaseConfig } from "./kb-searcher.js";

export interface Config {
  /** Directories to index */
  dirs: string[];
  /** File extensions to index (with dots) */
  fileExtensions: string[];
  /** Directory names to skip */
  excludeDirs: string[];
  /** Embedding dimensions */
  dimensions: number;
  /** Embedding provider config (required for local file indexing) */
  provider: ProviderConfig | null;
  /** Where to store the index */
  indexDir: string;
  /** Optional Bedrock Knowledge Bases to search */
  knowledgeBases: KnowledgeBaseConfig[];
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

export type ProviderConfig =
  | { type: "openai"; apiKey: string; model: string }
  | { type: "openai-compatible"; apiKey?: string; model: string; baseUrl: string }
  | { type: "bedrock"; profile: string; region: string; model: string }
  | { type: "ollama"; url: string; model: string };

/** Raw shape stored in the config file. */
export interface ConfigFile {
  dirs?: string[];
  fileExtensions?: string[];
  excludeDirs?: string[];
  dimensions?: number;
  knowledgeBases?: KnowledgeBaseConfig[];
  overview?: Partial<OverviewConfig>;
  provider?:
    | { type: "openai"; apiKey?: string; model?: string }
    | { type: "openai-compatible"; apiKey?: string; model?: string; baseUrl?: string }
    | { type: "bedrock"; profile?: string; region?: string; model?: string }
    | { type: "ollama"; url?: string; model?: string };
}

// Lazy so HOME changes at runtime (tests, sandboxes) are honored.
function globalConfigFile(): string {
  return path.join(process.env.HOME || "/tmp", ".pi", "knowledge-search.json");
}
function globalIndexDir(): string {
  return path.join(process.env.HOME || "/tmp", ".pi", "knowledge-search");
}

/**
 * Resolve a project-local base directory for pi-knowledge-search storage.
 *
 * Resolution order (highest priority first):
 *   1. {cwd}/.pi/settings.json → "pi-knowledge-search".localPath
 *   2. {cwd}/.pi/settings.json → "pi-total-recall".localPath → {localPath}/knowledge-search
 *
 * When set, config is stored at {base}/config.json and index at {base}/index.
 * Environment variables (KNOWLEDGE_SEARCH_CONFIG / KNOWLEDGE_SEARCH_INDEX_DIR)
 * take precedence over both.
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
const PI_TOTAL_RECALL_KNOWN_KEYS = ["localPath"] as const;

export function resolveLocalBase(cwd?: string): string | null {
  if (!cwd) return null;
  try {
    const raw = fs.readFileSync(path.join(cwd, ".pi", "settings.json"), "utf-8");
    const settings = JSON.parse(raw) ?? {};

    // Package-specific override wins.
    const ks = settings["pi-knowledge-search"];
    warnUnknownKeys(ks, "pi-knowledge-search", PI_KNOWLEDGE_SEARCH_SETTINGS_KEYS);
    if (ks && typeof ks === "object" && typeof ks.localPath === "string" && ks.localPath) {
      return ks.localPath;
    }

    // pi-total-recall cascade.
    const tr = settings["pi-total-recall"];
    warnUnknownKeys(tr, "pi-total-recall", PI_TOTAL_RECALL_KNOWN_KEYS);
    if (tr && typeof tr === "object" && typeof tr.localPath === "string" && tr.localPath) {
      return path.join(tr.localPath, "knowledge-search");
    }
  } catch {
    // No settings file, unreadable, or malformed — fall through to global.
  }
  return null;
}

/**
 * Resolve the config file path. Priority:
 *   1. KNOWLEDGE_SEARCH_CONFIG env var (explicit override)
 *   2. Project-local base ({base}/config.json)
 *   3. Global default (~/.pi/knowledge-search.json)
 */
export function getConfigPath(cwd?: string): string {
  if (process.env.KNOWLEDGE_SEARCH_CONFIG) return process.env.KNOWLEDGE_SEARCH_CONFIG;
  const base = resolveLocalBase(cwd);
  if (base) return path.join(base, "config.json");
  return globalConfigFile();
}

/**
 * Resolve the index directory. Priority matches getConfigPath().
 */
export function getIndexDir(cwd?: string): string {
  if (process.env.KNOWLEDGE_SEARCH_INDEX_DIR) return process.env.KNOWLEDGE_SEARCH_INDEX_DIR;
  const base = resolveLocalBase(cwd);
  if (base) return path.join(base, "index");
  return globalIndexDir();
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

  const hasKBs = (file?.knowledgeBases?.length ?? 0) > 0;

  if (!file && !envDirs) {
    return null; // Not configured yet
  }

  // Build config: file values, then env overrides
  const home = process.env.HOME || "/tmp";
  const resolvePath = (p: string) => p.replace(/^~/, home);

  const dirs = (envDirs ? envDirs.split(",").map((d) => d.trim()) : (file?.dirs ?? []))
    .map(resolvePath)
    .filter(Boolean);

  if (dirs.length === 0 && !hasKBs) return null;

  const fileExtensions = envStr("KNOWLEDGE_SEARCH_EXTENSIONS")
    ?.split(",")
    .map((e) => e.trim()) ??
    file?.fileExtensions ?? [".md", ".txt"];

  const excludeDirs = envStr("KNOWLEDGE_SEARCH_EXCLUDE")
    ?.split(",")
    .map((d) => d.trim()) ??
    file?.excludeDirs ?? ["node_modules", ".git", ".obsidian", ".trash"];

  const dimensions = envInt("KNOWLEDGE_SEARCH_DIMENSIONS") ?? file?.dimensions ?? 512;

  const providerType =
    envStr("KNOWLEDGE_SEARCH_PROVIDER") ??
    file?.provider?.type ??
    // Convenience default: if OPENAI_API_KEY is exported and nothing else
    // is configured, assume the user wants the openai provider.
    (process.env.OPENAI_API_KEY ? "openai" : undefined);

  let provider: ProviderConfig | null = null;
  if (providerType) {
    switch (providerType) {
      case "openai": {
        // Helpful migration error: if someone set a custom baseUrl on `openai`,
        // it used to be silently ignored. Steer them to openai-compatible.
        if (file?.provider?.type === "openai" && (file.provider as { baseUrl?: unknown }).baseUrl) {
          throw new Error(
            'Custom baseUrl is not supported on provider type "openai" (it would be silently ignored and requests would hit api.openai.com). Change "type" to "openai-compatible" to use a custom endpoint.'
          );
        }
        const apiKey =
          envStr("KNOWLEDGE_SEARCH_OPENAI_API_KEY") ??
          process.env.OPENAI_API_KEY ??
          (file?.provider?.type === "openai" ? file.provider.apiKey : undefined);
        if (!apiKey) {
          throw new Error(
            "OpenAI API key required. Run /knowledge-search-setup or set OPENAI_API_KEY."
          );
        }
        provider = {
          type: "openai",
          apiKey,
          model:
            envStr("KNOWLEDGE_SEARCH_OPENAI_MODEL") ??
            (file?.provider?.type === "openai" ? file.provider.model : undefined) ??
            "text-embedding-3-small",
        };
        break;
      }
      case "openai-compatible": {
        // Intentionally do NOT fall back to OPENAI_API_KEY here — an openai-
        // compatible endpoint may be a third-party service, and silently sending
        // the user's real OpenAI key to a foreign host would be a credential leak.
        // Users must set KNOWLEDGE_SEARCH_COMPAT_API_KEY explicitly (or leave
        // unset for runners like llama.cpp that don't require auth).
        const compatApiKey =
          envStr("KNOWLEDGE_SEARCH_COMPAT_API_KEY") ??
          (file?.provider?.type === "openai-compatible" ? file.provider.apiKey : undefined);
        const compatBaseUrl =
          envStr("KNOWLEDGE_SEARCH_COMPAT_BASE_URL") ??
          (file?.provider?.type === "openai-compatible" ? file.provider.baseUrl : undefined);
        if (!compatBaseUrl) {
          throw new Error(
            "OpenAI-compatible requires baseUrl. Set KNOWLEDGE_SEARCH_COMPAT_BASE_URL or provide it in your knowledge-search.json config."
          );
        }
        provider = {
          type: "openai-compatible",
          apiKey: compatApiKey,
          model:
            envStr("KNOWLEDGE_SEARCH_COMPAT_MODEL") ??
            (file?.provider?.type === "openai-compatible" ? file.provider.model : undefined) ??
            "text-embedding-3-small",
          baseUrl: compatBaseUrl,
        };
        break;
      }
      case "bedrock":
        provider = {
          type: "bedrock",
          profile:
            envStr("KNOWLEDGE_SEARCH_BEDROCK_PROFILE") ??
            (file?.provider?.type === "bedrock" ? file.provider.profile : undefined) ??
            "default",
          region:
            envStr("KNOWLEDGE_SEARCH_BEDROCK_REGION") ??
            (file?.provider?.type === "bedrock" ? file.provider.region : undefined) ??
            "us-east-1",
          model:
            envStr("KNOWLEDGE_SEARCH_BEDROCK_MODEL") ??
            (file?.provider?.type === "bedrock" ? file.provider.model : undefined) ??
            "amazon.titan-embed-text-v2:0",
        };
        break;
      case "ollama":
        provider = {
          type: "ollama",
          url:
            envStr("KNOWLEDGE_SEARCH_OLLAMA_URL") ??
            (file?.provider?.type === "ollama" ? file.provider.url : undefined) ??
            "http://localhost:11434",
          model:
            envStr("KNOWLEDGE_SEARCH_OLLAMA_MODEL") ??
            (file?.provider?.type === "ollama" ? file.provider.model : undefined) ??
            "nomic-embed-text",
        };
        break;
      default:
        throw new Error(
          `Unknown provider: "${providerType}". Use "openai", "openai-compatible", "bedrock", or "ollama".`
        );
    }
  } // end if (providerType)

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
    indexDir,
    knowledgeBases: file?.knowledgeBases ?? [],
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
