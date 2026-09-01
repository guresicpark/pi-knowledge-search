import { describe, it, before, beforeEach, afterEach, after } from "node:test";
import assert from "node:assert/strict";
import * as fs from "node:fs";
import * as path from "node:path";
import * as os from "node:os";

// CONFIG_PATH is evaluated at module load time from KNOWLEDGE_SEARCH_CONFIG env var.
// ESM hoists imports before top-level code, so we must use dynamic import().
// We set env var first, then dynamically import config.ts.

const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), "ks-config-test-"));
const configFile = path.join(tmpDir, "config.json");

// These env vars need to be saved/restored
const envKeys = [
  "KNOWLEDGE_SEARCH_CONFIG",
  "KNOWLEDGE_SEARCH_DIRS",
  "KNOWLEDGE_SEARCH_EXTENSIONS",
  "KNOWLEDGE_SEARCH_EXCLUDE",
  "KNOWLEDGE_SEARCH_DIMENSIONS",
  "KNOWLEDGE_SEARCH_TRANSFORMERS_MODEL",
  "KNOWLEDGE_SEARCH_AUTO_INJECT",
  "KNOWLEDGE_SEARCH_INDEX_DIR",
];

let loadConfig: (typeof import("./config.js"))["loadConfig"];
let getConfigPath: (typeof import("./config.js"))["getConfigPath"];
let saveConfig: (typeof import("./config.js"))["saveConfig"];

const originalEnv: Record<string, string | undefined> = {};

describe("config", () => {
  before(async () => {
    // Save ALL env state
    for (const key of envKeys) {
      originalEnv[key] = process.env[key];
    }
    // Set config path BEFORE importing config module
    process.env.KNOWLEDGE_SEARCH_CONFIG = configFile;
    // Clear interfering env vars
    for (const key of envKeys) {
      if (key !== "KNOWLEDGE_SEARCH_CONFIG" && key !== "HOME") {
        delete process.env[key];
      }
    }

    // Dynamic import so CONFIG_PATH picks up our env var
    const configModule = await import("./config.js");
    loadConfig = configModule.loadConfig;
    getConfigPath = configModule.getConfigPath;
    saveConfig = configModule.saveConfig;
  });

  beforeEach(() => {
    // Clear all knowledge search env vars except CONFIG
    for (const key of envKeys) {
      if (key !== "KNOWLEDGE_SEARCH_CONFIG") {
        delete process.env[key];
      }
    }
    // Remove config file if exists
    try {
      fs.unlinkSync(configFile);
    } catch {}
  });

  after(() => {
    // Restore env
    for (const key of envKeys) {
      if (originalEnv[key] === undefined) {
        delete process.env[key];
      } else {
        process.env[key] = originalEnv[key];
      }
    }
    fs.rmSync(tmpDir, { recursive: true, force: true });
  });

  it("getConfigPath returns the env-configured path", () => {
    assert.equal(getConfigPath(), configFile);
  });

  it("returns null when no config file and no env vars", () => {
    const config = loadConfig();
    assert.equal(config, null);
  });

  it("loads valid config from file", () => {
    fs.writeFileSync(
      configFile,
      JSON.stringify({
        dirs: ["/tmp/test-docs"],
        fileExtensions: [".md"],
        excludeDirs: ["node_modules"],
        dimensions: 256,
        provider: {
          type: "transformers",
          model: "Xenova/all-MiniLM-L6-v2",
        },
      })
    );

    const config = loadConfig();
    assert.ok(config);
    assert.ok(config.provider);
    assert.deepStrictEqual(config.dirs, ["/tmp/test-docs"]);
    assert.deepStrictEqual(config.fileExtensions, [".md"]);
    assert.equal(config.dimensions, 256);
    assert.equal(config.provider.type, "transformers");
    if (config.provider.type === "transformers") {
      assert.equal(config.provider.model, "Xenova/all-MiniLM-L6-v2");
    }
  });

  it("returns null for corrupt JSON config file", () => {
    fs.writeFileSync(configFile, "{ this is not valid json }}}}");
    const config = loadConfig();
    assert.equal(config, null);
  });

  it("uses env var KNOWLEDGE_SEARCH_DIRS as fallback (FTS-only)", () => {
    process.env.KNOWLEDGE_SEARCH_DIRS = "/tmp/dir-a, /tmp/dir-b";

    const config = loadConfig();
    assert.ok(config);
    assert.equal(config.provider, null);
    assert.deepStrictEqual(config.dirs, ["/tmp/dir-a", "/tmp/dir-b"]);
  });

  it("applies default values for optional fields", () => {
    fs.writeFileSync(
      configFile,
      JSON.stringify({
        dirs: ["/tmp/docs"],
        provider: { type: "transformers" },
      })
    );

    const config = loadConfig();
    assert.ok(config);
    assert.deepStrictEqual(config.fileExtensions, [".md", ".txt"]);
    assert.ok(config.excludeDirs.includes("node_modules"));
    assert.ok(config.excludeDirs.includes(".git"));
    assert.ok(config.excludeDirs.includes(".obsidian"));
    assert.ok(config.excludeDirs.includes(".trash"));
    assert.equal(config.dimensions, 768);
  });

  it("resolves ~ in directory paths", () => {
    const originalHome = process.env.HOME;
    process.env.HOME = "/home/testuser";

    fs.writeFileSync(
      configFile,
      JSON.stringify({
        dirs: ["~/Documents/notes"],
        provider: { type: "transformers" },
      })
    );

    try {
      const config = loadConfig();
      assert.ok(config);
      assert.deepStrictEqual(config.dirs, ["/home/testuser/Documents/notes"]);
    } finally {
      process.env.HOME = originalHome;
    }
  });

  it("configures transformers provider with nomic defaults and 768 dims", () => {
    fs.writeFileSync(
      configFile,
      JSON.stringify({
        dirs: ["/tmp/docs"],
        provider: { type: "transformers" },
      })
    );

    const config = loadConfig();
    assert.ok(config);
    assert.ok(config.provider);
    assert.equal(config.provider.type, "transformers");
    if (config.provider.type === "transformers") {
      assert.equal(config.provider.model, "nomic-ai/nomic-embed-text-v1.5");
    }
    assert.equal(config.dimensions, 768);
    assert.equal(config.modelSignature, "transformers:nomic-ai/nomic-embed-text-v1.5:768");
  });

  it("transformers provider honors custom model from file and env", () => {
    fs.writeFileSync(
      configFile,
      JSON.stringify({
        dirs: ["/tmp/docs"],
        provider: { type: "transformers", model: "Xenova/all-MiniLM-L6-v2" },
      })
    );
    process.env.KNOWLEDGE_SEARCH_TRANSFORMERS_MODEL = "Xenova/bge-small-en-v1.5";

    const config = loadConfig();
    assert.ok(config);
    assert.ok(config.provider);
    if (config.provider.type === "transformers") {
      assert.equal(config.provider.model, "Xenova/bge-small-en-v1.5");
    }
    assert.equal(config.modelSignature, "transformers:Xenova/bge-small-en-v1.5:768");
  });

  it("modelSignature is null in FTS-only mode", () => {
    fs.writeFileSync(
      configFile,
      JSON.stringify({
        dirs: ["/tmp/docs"],
      })
    );

    const config = loadConfig();
    assert.ok(config);
    assert.equal(config.provider, null);
    assert.equal(config.modelSignature, null);
  });

  it("modelSignature includes provider type, model, and dimensions", () => {
    fs.writeFileSync(
      configFile,
      JSON.stringify({
        dirs: ["/tmp/docs"],
        dimensions: 256,
        provider: { type: "transformers", model: "Xenova/all-MiniLM-L6-v2" },
      })
    );

    const config = loadConfig();
    assert.ok(config);
    assert.equal(config.modelSignature, "transformers:Xenova/all-MiniLM-L6-v2:256");
  });

  it("autoInject defaults to true and can be overridden by file and env", () => {
    fs.writeFileSync(
      configFile,
      JSON.stringify({
        dirs: ["/tmp/docs"],
        provider: { type: "transformers" },
      })
    );
    assert.equal(loadConfig()?.autoInject, true);

    fs.writeFileSync(
      configFile,
      JSON.stringify({
        dirs: ["/tmp/docs"],
        autoInject: false,
      })
    );
    assert.equal(loadConfig()?.autoInject, false);

    process.env.KNOWLEDGE_SEARCH_AUTO_INJECT = "1";
    assert.equal(loadConfig()?.autoInject, true);
  });

  it("throws a helpful migration error for removed provider types", () => {
    for (const removed of ["openai", "openai-compatible", "bedrock", "ollama"]) {
      fs.writeFileSync(
        configFile,
        JSON.stringify({
          dirs: ["/tmp/docs"],
          provider: { type: removed },
        })
      );

      assert.throws(
        () => loadConfig(),
        (err: unknown) => {
          assert.ok(err instanceof Error);
          assert.match(err.message, /Unsupported embedding provider/);
          assert.match(err.message, new RegExp(`"${removed}"`));
          assert.match(err.message, /transformers/);
          assert.match(err.message, /FTS-only/);
          return true;
        },
        `expected loadConfig() to throw for provider "${removed}"`
      );
    }
  });

  it("throws for unknown provider type", () => {
    fs.writeFileSync(
      configFile,
      JSON.stringify({
        dirs: ["/tmp/docs"],
        provider: { type: "unknown-provider" },
      })
    );

    assert.throws(() => loadConfig(), /Unsupported embedding provider/);
  });

  it("env vars override config file values", () => {
    fs.writeFileSync(
      configFile,
      JSON.stringify({
        dirs: ["/tmp/file-dirs"],
        dimensions: 256,
        provider: { type: "transformers" },
      })
    );
    process.env.KNOWLEDGE_SEARCH_DIRS = "/tmp/env-dirs";
    process.env.KNOWLEDGE_SEARCH_DIMENSIONS = "1024";

    const config = loadConfig();
    assert.ok(config);
    assert.deepStrictEqual(config.dirs, ["/tmp/env-dirs"]);
    assert.equal(config.dimensions, 1024);
  });

  it("saveConfig writes valid JSON to config path", () => {
    const configData = {
      dirs: ["/tmp/saved"],
      provider: { type: "transformers" as const, model: "nomic-ai/nomic-embed-text-v1.5" },
    };
    saveConfig(configData);

    const raw = fs.readFileSync(configFile, "utf-8");
    const parsed = JSON.parse(raw);
    assert.deepStrictEqual(parsed.dirs, ["/tmp/saved"]);
    assert.equal(parsed.provider.type, "transformers");
  });

  it("returns null when dirs resolve to empty", () => {
    fs.writeFileSync(
      configFile,
      JSON.stringify({
        dirs: [],
        provider: { type: "transformers" },
      })
    );

    const config = loadConfig();
    assert.equal(config, null);
  });
});
