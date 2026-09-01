import type {
  ExtensionAPI,
  ExtensionCommandContext,
  SessionEntry,
} from "@earendil-works/pi-coding-agent";
import { Type } from "@sinclair/typebox";
import type { ChildProcess } from "node:child_process";
import { fork } from "node:child_process";
import * as fs from "node:fs";
import { join, resolve } from "node:path";
import { loadConfig, saveConfig, getConfigPath, getIndexDir, type Config, type ConfigFile } from "./config.js";
import { createEmbedder, isTransformersModelCached } from "./embedder.js";
import { KnowledgeIndex, type SyncProgress } from "./index-store.js";
import { buildOverview, formatOverview } from "./overview.js";
import { resolveNote, readNote } from "./kb-reader.js";

/** Render a 24-cell block progress bar (cyan filled / dim empty), like /rag's. */
function renderProgressBar(current: number, total: number, width = 24): string {
  const filled = total > 0 ? Math.round((current / total) * width) : width;
  return `\x1b[36m${"█".repeat(filled)}\x1b[2m${"░".repeat(width - filled)}\x1b[0m`;
}

export default function (pi: ExtensionAPI) {
  let index: KnowledgeIndex | null = null;
  let currentConfig: Config | null = null;
  let sessionCwd: string | undefined;
  let syncDone = false;
  let workerExitExpected = false;
  /** In-flight sync worker, so `/knowledge-search clear` can kill it before wiping storage. */
  let activeWorker: ChildProcess | null = null;

  /**
   * Build and inject the folder+keyword overview as a custom message.
   * @param force When true, inject even if one is already present.
   *              Used by /knowledge-overview after config changes or vault growth.
   * @returns Information about what happened for user-facing feedback.
   */
  function injectOverview(
    ctx: {
      sessionManager: { getEntries: () => SessionEntry[] };
    },
    force: boolean
  ):
    | { status: "skipped"; reason: string }
    | { status: "injected"; totalNotes: number; sourceCount: number } {
    if (!index || !currentConfig) return { status: "skipped", reason: "not configured" };
    if (!force && !currentConfig.overview.inject) {
      return { status: "skipped", reason: "overview.inject=false" };
    }
    if (index.size() === 0) return { status: "skipped", reason: "index is empty" };

    if (!force) {
      const alreadyInjected = ctx.sessionManager
        .getEntries()
        .some(
          (e: SessionEntry) =>
            e.type === "custom_message" && e.customType === "knowledge-overview"
        );
      if (alreadyInjected) return { status: "skipped", reason: "already injected" };
    }

    const overview = buildOverview(index.listFiles(), currentConfig.dirs, {
      maxDepth: currentConfig.overview.maxDepth,
      maxFoldersPerDir: currentConfig.overview.maxFoldersPerDir,
      maxKeywordsPerFolder: currentConfig.overview.maxKeywordsPerFolder,
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
        forced: force,
      },
    });
    return {
      status: "injected",
      totalNotes: overview.totalNotes,
      sourceCount: overview.sources.length,
    };
  }

  // ------------------------------------------------------------------
  // Lifecycle
  // ------------------------------------------------------------------

  pi.on("session_start", async (_event, ctx) => {
    sessionCwd = ctx.cwd;
    try {
      currentConfig = loadConfig(sessionCwd);
    } catch {
      return;
    }
    if (!currentConfig) return;

    let indexLoaded: Promise<void> = Promise.resolve();
    if (currentConfig.provider) {
      const embedder = createEmbedder(currentConfig.provider, currentConfig.dimensions);
      index = new KnowledgeIndex(currentConfig, embedder);
      // Fire-and-forget: don't block session_start on the (potentially
      // 99 MB) JSON.parse. injectOverview below awaits this promise; the
      // outbound model HTTP request can fire as soon as session_start
      // returns. See plan: Slice B'.
      indexLoaded = index.load();
    } else if (currentConfig.dirs.length > 0) {
      // FTS-only mode — no embedder, keyword search still works zero-config.
      index = new KnowledgeIndex(currentConfig, null);
      indexLoaded = index.load();
    }

    if (!index) {
      syncDone = true;
      return; // Nothing to sync
    }

    // ----------------------------------------------------------------
    // Inject a folder+keyword overview of the vault as a custom message,
    // unless one is already in the session or the user disabled it.
    // Runs off whatever the index has loaded from disk — the worker's
    // incremental sync below will update the store for future sessions.
    //
    // Gated on indexLoaded so the synchronous JSON.parse stays off the
    // session_start critical path; injectOverview itself runs in a
    // microtask after load() resolves, while pi's outbound model HTTP
    // already fired during session_start's earlier return.
    // ----------------------------------------------------------------
    indexLoaded
      .then(() => {
        try {
          injectOverview(ctx, false);
        } catch (err: unknown) {
          // Overview is a nice-to-have — never let it break startup.
          const msg = err instanceof Error ? err.message : String(err);
          console.error(`knowledge-search: overview injection failed: ${msg}`);
        }
      })
      .catch((err: unknown) => {
        const msg = err instanceof Error ? err.message : String(err);
        console.error(`knowledge-search: index load failed: ${msg}`);
      });

    // Sync in a child process so it never blocks the main event loop
    const MAX_WORKER_RESTARTS = 3;
    const RESTART_WINDOW_MS = 60_000;
    let workerRestartCount = 0;
    let workerRestartWindowStart = Date.now();

    function spawnWorker() {
      // Use pre-compiled worker to avoid ESM/CJS cycle with tsx on Node 25+
      // Rebuild with: npx esbuild src/sync-worker.ts --bundle --platform=node --format=esm --outfile=dist/sync-worker.mjs --external:better-sqlite3 --packages=external
      const workerPath = join(import.meta.dirname, "..", "dist", "sync-worker.mjs");
      const worker = fork(workerPath, [], {
        stdio: ["ignore", "pipe", "pipe", "ipc"],
        // Suppress "node:sqlite is experimental" warning — node:sqlite is stable
        // enough for our read/write usage and the warning pollutes pi startup.
        execArgv: ["--no-warnings=ExperimentalWarning"],
        // Forward sessionCwd so the worker resolves the same project-local
        // settings.json (pi-knowledge-search.localPath).
        env: { ...process.env, KNOWLEDGE_SEARCH_CWD: sessionCwd ?? process.env.KNOWLEDGE_SEARCH_CWD ?? "" },
      });
      activeWorker = worker;

      let stdout = "";
      let stderrBuf = "";
      // Surface worker status through the managed UI when a TUI is present.
      // Writing directly to the terminal (console.error) from these async
      // callbacks paints outside pi's render region and corrupts the input
      // box; only fall back to console.error in headless (-p/json) modes.
      const report = (msg: string, level: "info" | "warning" | "error" = "error") => {
        if (ctx.hasUI) {
          ctx.ui.notify(msg, level);
        } else {
          console.error(msg);
        }
      };
      worker.stdout?.on("data", (chunk: Buffer) => {
        stdout += chunk.toString();
      });
      worker.stderr?.on("data", (chunk: Buffer) => {
        // Buffer worker stderr and surface a single summarized line on exit
        // instead of echoing every chunk raw to the terminal.
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
            // Reload the index from disk since the worker updated it
            await index!.load();
            const changes = result.added + result.updated + result.removed;
            if (changes > 0) {
              ctx.ui.setStatus(
                "knowledge-search",
                `Index: +${result.added} ~${result.updated} -${result.removed} (${result.size} files, ${result.chunks} chunks)`
              );
              setTimeout(() => ctx.ui.setStatus("knowledge-search", ""), 5000);
            }
          } catch {
            // ignore parse errors
          }
          // A clean exit can still carry non-fatal diagnostics (e.g. an
          // aggregated "embedding failed for N/M chunks" line) the worker
          // wrote to stderr. Surface the last such line as a warning so it
          // isn't silently dropped, without treating the run as failed.
          const stderrTail = stderrBuf.trim().split("\n").filter(Boolean).pop() ?? "";
          if (stderrTail) {
            report(`knowledge-search: ${stderrTail}`, "warning");
          }
        } else if (code !== 0 && !workerExitExpected) {
          const now = Date.now();
          // Reset counter if outside the time window
          if (now - workerRestartWindowStart > RESTART_WINDOW_MS) {
            workerRestartCount = 0;
            workerRestartWindowStart = now;
          }
          workerRestartCount++;

          const stderrTail = stderrBuf.trim().split("\n").filter(Boolean).pop() ?? "";
          const detail = stderrTail ? ` (${stderrTail})` : "";
          if (workerRestartCount > MAX_WORKER_RESTARTS) {
            report(
              `knowledge-search: indexing worker crashed ${workerRestartCount}x within ${RESTART_WINDOW_MS / 1000}s, giving up${detail}`
            );
          } else {
            report(
              `knowledge-search: indexing worker failed (code=${code}, signal=${signal}), retrying ${workerRestartCount}/${MAX_WORKER_RESTARTS}${detail}`,
              "warning"
            );
            setTimeout(() => {
              if (!workerExitExpected) spawnWorker();
            }, 2000);
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
    // watcher removed (d38a81f) — caused UI freezes. Rely on sync-on-startup only.
    await index?.close();
  });

  // ------------------------------------------------------------------
  // /knowledge-search command: status (bare, toggling), add, exclude,
  // index (incremental), clear, and help.
  // ------------------------------------------------------------------

  /** Subcommand table for autocomplete and /knowledge-search help. */
  const KS_SUBCOMMANDS: { value: string; label: string; description: string }[] = [
    { value: "add", label: "add", description: "Add directories to the index" },
    { value: "exclude", label: "exclude", description: "Manage excluded directory names (-<name> removes)" },
    { value: "index", label: "index", description: "Incrementally index new/changed files" },
    { value: "clear", label: "clear", description: "Clear the index and reset config to defaults" },
    { value: "help", label: "help", description: "Show all /knowledge-search commands" },
  ];

  function getSubcommandCompletions(prefix: string) {
    const matches = KS_SUBCOMMANDS.filter((s) => s.value.startsWith(prefix)).map((s) => ({
      value: s.value,
      label: s.label,
      description: s.description,
    }));
    return matches.length > 0 ? matches : null;
  }

  /**
   * Read the raw config file (preserving unknown fields), or an empty
   * object when missing/corrupt — the base for add/exclude mutations.
   */
  function readRawConfig(): ConfigFile & Record<string, unknown> {
    try {
      return JSON.parse(fs.readFileSync(getConfigPath(sessionCwd), "utf-8"));
    } catch {
      return {};
    }
  }

  /** Expand ~ and resolve a user-typed path against the session cwd. */
  function resolveUserPath(p: string): string {
    const home = process.env.HOME || "";
    const expanded = p.startsWith("~") ? home + p.slice(1) : p;
    return resolve(sessionCwd ?? process.cwd(), expanded);
  }

  /**
   * Ensure an in-memory KnowledgeIndex exists for the current config.
   * Mutating `currentConfig` in place keeps an existing index's config
   * reference in sync; a fresh index is only built when there was none
   * (e.g. the session started unconfigured and `add` just created one).
   */
  async function ensureIndexLoaded(): Promise<void> {
    if (index) return;
    const embedder = currentConfig?.provider
      ? createEmbedder(currentConfig.provider, currentConfig.dimensions)
      : null;
    index = new KnowledgeIndex(currentConfig!, embedder);
    await index.load();
  }

  /** Bare-command status widget, mirroring /rag's layout. */
  function renderStatusWidget(ctx: ExtensionCommandContext): void {
    const theme = ctx.ui.theme;
    const label = (text: string) => theme.fg("dim", text.padEnd(20));
    const lines: string[] = [theme.bold("pi-knowledge-search"), ""];

    lines.push("  " + label("Embedding engine:") + (currentConfig?.provider
      ? theme.fg("success", currentConfig.provider.model) + theme.fg("dim", "  (local ONNX via Transformers.js)")
      : theme.fg("warning", "none") + theme.fg("dim", "  (FTS-only keyword search)")));

    if (index) {
      lines.push("  " + label("Indexed:") + theme.fg("success", `${index.size()} files · ${index.chunkCount()} chunks`));
    } else {
      lines.push("  " + label("Indexed:") + theme.fg("dim", "0 files (run /knowledge-search index)"));
    }

    lines.push("", "  " + theme.bold("Directories indexed:"));
    const dirs = currentConfig?.dirs ?? [];
    if (dirs.length) {
      for (const dir of dirs) lines.push("    " + theme.fg("muted", dir));
    } else {
      lines.push("    " + theme.fg("dim", "(none — add with /knowledge-search add <dir>)"));
    }

    lines.push("", "  " + theme.bold("Excluded directories:"));
    const excludes = currentConfig?.excludeDirs ?? [];
    if (excludes.length) {
      for (const name of excludes) lines.push("    " + theme.fg("muted", name));
    } else {
      lines.push("    " + theme.fg("dim", "(none — add with /knowledge-search exclude <name>)"));
    }

    lines.push("", "  " + theme.bold("File extensions:"));
    const exts = currentConfig?.fileExtensions ?? [];
    lines.push("    " + theme.fg("muted", exts.join(" ")));

    lines.push(
      "",
      "  " + label("Config:") + theme.fg("dim", getConfigPath(sessionCwd)),
      "  " + label("Index:") + theme.fg("dim", currentConfig?.indexDir ?? ""),
    );

    ctx.ui.setWidget("knowledge-search-status", lines);
  }

  /** /knowledge-search add <dir> [<dir>...] — track directories to index. */
  async function handleAdd(parts: string[], ctx: ExtensionCommandContext): Promise<void> {
    const raw = parts
      .slice(1)
      .join(" ")
      .split(/[\s,]+/)
      .map((p) => p.trim())
      .filter(Boolean);

    if (raw.length === 0) {
      ctx.ui.notify("Usage: /knowledge-search add <dir> [<dir>...]", "warning");
      return;
    }

    const resolved = raw.map(resolveUserPath);
    for (const dir of resolved) {
      if (!fs.existsSync(dir)) {
        ctx.ui.notify(`Path not found: ${dir}`, "error");
        return;
      }
    }

    const file = readRawConfig();
    const dirs = new Set([...(file.dirs ?? []), ...resolved]);
    const before = file.dirs?.length ?? 0;
    file.dirs = [...dirs];

    // Fresh configs default to the local ONNX engine; an existing provider
    // block (or its absence, FTS-only) is preserved as-is.
    if (!file.provider && before === 0) {
      file.provider = { type: "transformers" };
    }

    saveConfig(file as ConfigFile, sessionCwd);

    // Keep the in-session config/index in sync so a follow-up `index`
    // picks the new dirs up without a reload.
    const added = resolved.filter((d) => !(currentConfig?.dirs ?? []).includes(d));
    if (currentConfig) {
      currentConfig.dirs = [...dirs];
      if (!currentConfig.provider && file.provider) currentConfig.provider = { type: "transformers", model: "nomic-ai/nomic-embed-text-v1.5" };
    } else {
      currentConfig = loadConfig(sessionCwd);
    }

    const newCount = added.length;
    ctx.ui.notify(
      `Added ${newCount} director${newCount === 1 ? "y" : "ies"} · ${dirs.size} total. Run /knowledge-search index to index them.`,
      "info"
    );
  }

  /** /knowledge-search exclude [<name>|-<name>] — manage excluded directory names. */
  function handleExclude(parts: string[], ctx: ExtensionCommandContext): void {
    const expression = parts.slice(1).join(" ").trim();
    const file = readRawConfig();

    if (!expression) {
      const excludes = file.excludeDirs ?? [];
      if (!excludes.length) {
        ctx.ui.notify("No excluded directories. Add one with: /knowledge-search exclude <name>", "info");
        return;
      }
      const theme = ctx.ui.theme;
      const lines: string[] = [theme.bold(`Excluded directories (${excludes.length})`), ""];
      for (const name of excludes) lines.push("  " + theme.fg("muted", name));
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
      saveConfig(file as ConfigFile, sessionCwd);
      if (currentConfig) currentConfig.excludeDirs = file.excludeDirs!;
      ctx.ui.notify(
        `Removed exclude: ${target} · ${file.excludeDirs!.length} remain. Run /knowledge-search index to re-apply.`,
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
    saveConfig(file as ConfigFile, sessionCwd);
    if (currentConfig) currentConfig.excludeDirs = excludes;
    ctx.ui.notify(
      `Added exclude: ${expression} · ${excludes.length} total. Run /knowledge-search index to re-apply.`,
      "info"
    );
  }

  /**
   * /knowledge-search index — incremental sync of new/changed files, with
   * pi-local-rag-style progress: footer status line + widget with a block
   * progress bar, plus a cold-start notice when the embedding model needs
   * downloading.
   */
  async function handleIndex(ctx: ExtensionCommandContext): Promise<void> {
    if (!currentConfig || currentConfig.dirs.length === 0) {
      ctx.ui.notify("No directories configured. Run /knowledge-search add <dir> first.", "warning");
      return;
    }

    const clearProgressUI = () => {
      ctx.ui.setStatus("knowledge-search", undefined);
      ctx.ui.setWidget("knowledge-search", undefined);
    };

    try {
      await ensureIndexLoaded();

      // Cold-start notice: a missing model triggers a ~111 MB download that
      // can stall the first index for minutes — say so before it happens
      // (mirrors pi-local-rag's onModelLoad).
      if (
        currentConfig.provider?.type === "transformers" &&
        !isTransformersModelCached(currentConfig.provider.model)
      ) {
        ctx.ui.notify(
          `⏳ Loading embedding model: ${currentConfig.provider.model} — first run downloads it (~111 MB, this can take a few minutes)`,
          "info"
        );
      }

      const theme = ctx.ui.theme;
      const verb = theme.fg("accent", "Indexing");
      const { added, updated, removed } = await index!.sync({
        onProgress: (p: SyncProgress) => {
          if (p.phase === "scan") {
            const label =
              p.filesToProcess > 0
                ? `Found ${p.filesToProcess} file(s) to index · ${p.unchanged} unchanged · ${p.totalChunks} chunks`
                : `Nothing to index · ${p.unchanged} files unchanged`;
            ctx.ui.setStatus("knowledge-search", `■ Scanning… ${label}`);
            ctx.ui.setWidget("knowledge-search", [verb, theme.fg("dim", label)]);
            return;
          }
          if (p.phase === "embed") {
            const percent = p.total ? Math.round((p.done / p.total) * 100) : 100;
            const bar = renderProgressBar(p.done, p.total);
            ctx.ui.setStatus(
              "knowledge-search",
              `■ Indexing ${percent}% │ ${p.done}/${p.total} chunks`
            );
            ctx.ui.setWidget("knowledge-search", [
              `${verb}  ${bar}  ${theme.fg("success", `${percent}%`)}`,
              `${theme.fg("dim", "file:    ")}${p.currentFile ?? "…"}`,
              `${theme.fg("dim", "chunks:  ")}${theme.fg("success", String(p.done))}/${p.total}`,
            ]);
            return;
          }
          ctx.ui.setStatus("knowledge-search", "■ Saving index...");
        },
      });
      syncDone = true;
      clearProgressUI();

      const changed = added + updated + removed;
      ctx.ui.notify(
        changed === 0
          ? `✅ Up to date · ${index!.size()} files (${index!.chunkCount()} chunks) indexed`
          : `✅ Indexed ${added} new · ${updated} changed · ${removed} removed · ${index!.size()} files (${index!.chunkCount()} chunks) total`,
        "info"
      );
    } catch (err: any) {
      clearProgressUI();
      ctx.ui.notify(`Index failed: ${err.message}`, "error");
    }
  }

  /**
   * /knowledge-search clear — empty the index (vectors + FTS side-car) and
   * reset the config to fresh defaults, mirroring pi-local-rag's /rag clear.
   */
  async function handleClear(ctx: ExtensionCommandContext): Promise<void> {
    const confirmed = await ctx.ui.confirm(
      "Clear knowledge search?",
      "Deletes all project data (vector index + keyword side-car + config) and resets project settings to defaults, including any localPath override in .pi/settings.json. Re-index afterwards with /knowledge-search add + index. The shared HuggingFace model cache is not touched."
    );
    if (!confirmed) {
      ctx.ui.notify("Clear cancelled.", "info");
      return;
    }

    // Resolve storage before resetting state — from the live config when
    // present, otherwise from the same path-resolution defaults. The config
    // path is captured BEFORE the settings.json override is stripped below,
    // so a localPath-relocated config file gets deleted too.
    const indexDir = currentConfig?.indexDir ?? getIndexDir(sessionCwd);
    const configPath = getConfigPath(sessionCwd);

    // Kill an in-flight sync worker first so it can't re-write the index
    // files we're about to delete. workerExitExpected suppresses the crash
    // retry loop and the post-exit index reload.
    if (activeWorker) {
      workerExitExpected = true;
      activeWorker.kill();
      activeWorker = null;
    }

    // Close the in-memory index (flushes the FTS handle) before deleting.
    try {
      await index?.close();
    } catch {
      // best-effort — files are deleted next regardless
    }
    index = null;
    currentConfig = null;
    syncDone = false;

    // Wipe everything inside the index directory (index.json, kb-fts.db,
    // .tmp leftovers), keeping the directory itself — like /rag clear's
    // resetStore().
    try {
      if (fs.existsSync(indexDir)) {
        for (const entry of fs.readdirSync(indexDir)) {
          fs.rmSync(join(indexDir, entry), { recursive: true, force: true });
        }
      }
    } catch (err: any) {
      ctx.ui.notify(`Clear failed while deleting ${indexDir}: ${err.message}`, "error");
      return;
    }

    // Delete the config file (it is recreated at the default location
    // below — deleting first also removes a localPath-relocated file that
    // would otherwise be orphaned by the settings reset).
    try {
      fs.rmSync(configPath, { force: true });
    } catch (err: any) {
      ctx.ui.notify(`Clear failed while deleting ${configPath}: ${err.message}`, "error");
      return;
    }

    // Reset the project settings.json too: drop the pi-knowledge-search
    // block (localPath override) so storage resolution returns to the
    // default {cwd}/.pi location. Everything else in settings.json is
    // left untouched.
    const settingsFile = join(sessionCwd ?? process.cwd(), ".pi", "settings.json");
    try {
      if (fs.existsSync(settingsFile)) {
        const settings = JSON.parse(fs.readFileSync(settingsFile, "utf-8"));
        if (settings && typeof settings === "object" && "pi-knowledge-search" in settings) {
          delete settings["pi-knowledge-search"];
          fs.writeFileSync(settingsFile, JSON.stringify(settings, null, 2) + "\n");
        }
      }
    } catch {
      // Unreadable/malformed settings — leave as-is; the default-location
      // config written next still applies.
    }

    // Reset the config to fresh install defaults at the default location
    // (no provider — the next `add` re-defaults the engine to local
    // Transformers.js). The HuggingFace model cache is machine-wide and
    // shared with pi-local-rag, so it is intentionally NOT touched.
    saveConfig(
      {
        dirs: [],
        fileExtensions: [".md", ".txt"],
        excludeDirs: ["node_modules", ".git", ".obsidian", ".trash"],
      },
      sessionCwd
    );

    statusWidgetVisible = false;
    ctx.ui.setWidget("knowledge-search-status", undefined);
    ctx.ui.notify(
      `✅ Cleared all project data (${indexDir}) and reset settings to defaults`,
      "info"
    );
  }

  /** /knowledge-search help — subcommand list widget. */
  function handleHelp(ctx: ExtensionCommandContext): void {
    const theme = ctx.ui.theme;
    const lines: string[] = [theme.bold("/knowledge-search commands"), ""];
    for (const s of KS_SUBCOMMANDS) {
      lines.push("  " + theme.fg("accent", s.label.padEnd(10)) + theme.fg("dim", s.description));
    }
    lines.push("", theme.fg("dim", "Bare /knowledge-search shows the current status."));
    ctx.ui.setWidget("knowledge-search-help", lines);
  }

  let statusWidgetVisible = false;

  pi.registerCommand("knowledge-search", {
    description: "knowledge-search: (status) | add <dir> | exclude <name> | index | clear | help",
    getArgumentCompletions: (prefix: string) => getSubcommandCompletions(prefix),
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
      if (subcommand === "help") {
        handleHelp(ctx);
        return;
      }
      if (subcommand) {
        ctx.ui.notify(`Unknown /knowledge-search command: ${subcommand}. Try /knowledge-search help`, "error");
        return;
      }

      // Bare /knowledge-search toggles the status widget (like /rag).
      if (statusWidgetVisible) {
        statusWidgetVisible = false;
        ctx.ui.setWidget("knowledge-search-status", undefined);
        return;
      }
      renderStatusWidget(ctx);
      statusWidgetVisible = true;
    },
  });

  // ------------------------------------------------------------------
  // Overview command
  // ------------------------------------------------------------------

  pi.registerCommand("knowledge-overview", {
    description:
      "Rebuild and re-inject the knowledge-search vault overview (use after config changes or vault growth)",
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
      } catch (err: unknown) {
        const msg = err instanceof Error ? err.message : String(err);
        ctx.ui.notify(`Overview injection failed: ${msg}`, "error");
      }
    },
  });

  // ------------------------------------------------------------------
  // Search tool
  // ------------------------------------------------------------------

  const searchParams = Type.Object({
    query: Type.String({ description: "Natural language search query" }),
    limit: Type.Optional(
      Type.Number({
        description: "Max results to return (default 8, max 20)",
      })
    ),
  });
  type SearchDetails = { resultCount?: number; indexSize?: number };

  pi.registerTool<typeof searchParams, SearchDetails>({
    name: "knowledge_search",
    label: "Knowledge Search",
    description:
      "Semantic search over local knowledge files. Returns the most relevant file excerpts for a natural language query. Use for finding past notes, investigations, decisions, documentation, and context. Prefer this over grep when you need conceptual or fuzzy matching rather than exact text.",
    promptGuidelines: [
      'Use knowledge_search for conceptual queries (e.g. "how did we handle X", "what was decided about Y"). Use grep/read for exact text or known filenames.',
    ],
    parameters: searchParams,
    async execute(toolCallId, params, signal) {
      if (!index || index.size() === 0) {
        const msg =
          !index
            ? "knowledge-search is not configured. The user can run /knowledge-search add <dir> to set it up."
            : !syncDone
              ? "Index is still syncing in the background. Try again in a moment."
              : "Index is empty.";
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
                text: `No relevant results found for: "${params.query}"`,
              },
            ],
            details: {},
          };
        }

        const home = process.env.HOME || "";
        const output = results
          .map((r: any, i: number) => {
            const displayPath = r.path.replace(home, "~");
            const score = (r.score * 100).toFixed(1);
            const heading = r.heading && r.heading !== "intro" ? ` > ${r.heading}` : "";
            return `### ${i + 1}. ${displayPath}${heading} (${score}% match)\n\n${r.excerpt}`;
          })
          .join("\n\n---\n\n");

        const sourceInfo = `${index.size()} files, ${index.chunkCount()} chunks indexed`;
        const header = `Found ${results.length} results for "${params.query}" (${sourceInfo}):\n\n`;

        return {
          content: [{ type: "text", text: header + output }],
          details: { resultCount: results.length, indexSize: index?.size() ?? 0 },
        };
      } catch (err: any) {
        throw new Error(`knowledge-search failed: ${err.message}`);
      }
    },
  });

  // ------------------------------------------------------------------
  // Read tool — resolve a note reference (wikilink, basename, fuzzy name)
  // to a file in the indexed vault and return its content. Complements
  // knowledge_search by letting the agent pull a known note without first
  // running grep/find to get an absolute path.
  // ------------------------------------------------------------------
  const readParams = Type.Object({
    name: Type.String({
      description:
        "Note reference: filename, basename, relative path, or [[wikilink]]. Examples: 'evergreen/hybrid-search', 'Hybrid search.md', '[[Hybrid search]]', '[[evergreen/hybrid-search|alias]]'.",
    }),
    max_bytes: Type.Optional(
      Type.Number({
        description: "Truncate output to at most this many bytes (default 65536).",
      })
    ),
  });
  type ReadDetails = { resolvedPath?: string; candidates?: string[]; truncated?: boolean };

  pi.registerTool<typeof readParams, ReadDetails>({
    name: "kb_read",
    label: "KB Read",
    description:
      "Read a note from the knowledge base by name, relative path, or [[wikilink]]. Resolves fuzzy references without needing an absolute path — use this when you know the note's title/filename but not its full path on disk.",
    promptGuidelines: [
      "Use kb_read when a note is referenced by name or [[wikilink]] — don't run find/grep first.",
      "Use the standard `read` tool for non-indexed files or when you already have an absolute path.",
    ],
    parameters: readParams,
    async execute(_toolCallId, params) {
      if (!index || index.size() === 0) {
        const msg = !index
          ? "knowledge-search is not configured. Run /knowledge-search add <dir> to set it up."
          : !syncDone
            ? "Index is still syncing in the background. Try again in a moment."
            : "Index is empty.";
        return { content: [{ type: "text", text: msg }], details: {} };
      }

      const result = resolveNote(params.name, index.listFiles(), {
        fileExtensions: currentConfig?.fileExtensions,
        cwd: sessionCwd,
      });

      if (result.matches.length === 0) {
        return {
          content: [
            {
              type: "text",
              text: `No note matched "${result.normalizedRef}". Try knowledge_search with a topic query to find related notes.`,
            },
          ],
          details: {},
        };
      }

      if (!result.unique && result.matches.length > 1) {
        const home = process.env.HOME || "";
        const listed = result.matches
          .map((m, i) => {
            const display = home && m.absPath.startsWith(home) ? m.absPath.replace(home, "~") : m.absPath;
            return `${i + 1}. ${display}  _(${m.reason})_`;
          })
          .join("\n");
        return {
          content: [
            {
              type: "text",
              text:
                `"${result.normalizedRef}" is ambiguous. ${result.matches.length} candidates:\n\n${listed}\n\n` +
                `Call kb_read again with a more specific path (e.g. the exact relative path) to disambiguate.`,
            },
          ],
          details: { candidates: result.matches.map((m) => m.absPath) },
        };
      }

      const match = result.matches[0];
      let note;
      try {
        note = readNote(match.absPath, {
          maxBytes: params.max_bytes,
        });
      } catch (err: unknown) {
        const msg = err instanceof Error ? err.message : String(err);
        return {
          content: [{ type: "text", text: `Failed to read ${match.absPath}: ${msg}` }],
          details: {},
        };
      }

      const home = process.env.HOME || "";
      const display = home && note.path.startsWith(home) ? note.path.replace(home, "~") : note.path;
      const truncNote = note.truncated
        ? `\n\n_(truncated: showing first ${note.content.length} of ${note.totalBytes} bytes)_`
        : "";
      const section = result.subheading ? ` — section "${result.subheading}"` : "";
      // When a single low-confidence match slips through (fuzzy substring), flag
      // the reason so the agent can decide whether to trust the result or refine
      // the reference. High-confidence tiers are resolved silently.
      const fuzzyNote = !result.unique
        ? `\n\n_(fuzzy match via ${match.reason} — if this isn't the note you meant, re-run kb_read with a more specific path)_`
        : "";
      const header = `# ${display}${section}${truncNote}${fuzzyNote}\n\n`;

      return {
        content: [{ type: "text", text: header + note.content }],
        details: {
          resolvedPath: match.absPath,
          truncated: note.truncated,
        },
      };
    },
  });
}
