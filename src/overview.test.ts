import { describe, it } from "node:test";
import assert from "node:assert/strict";
import * as fs from "node:fs";
import * as os from "node:os";
import * as path from "node:path";
import { buildOverview, formatOverview, bucketFolder } from "./overview.js";

describe("bucketFolder", () => {
  it("returns empty string for a root-level file", () => {
    assert.equal(bucketFolder("note.md", 2), "");
  });

  it("buckets single-nested files at their actual folder", () => {
    assert.equal(bucketFolder("evergreen/note.md", 2), "evergreen");
  });

  it("truncates deeper paths to the requested depth", () => {
    assert.equal(bucketFolder("Notes/Evergreen/sub/note.md", 2), "Notes/Evergreen");
    assert.equal(bucketFolder("Notes/Evergreen/sub/note.md", 1), "Notes");
  });

  it("keeps shallower paths intact when depth exceeds nesting", () => {
    assert.equal(bucketFolder("a/b/note.md", 5), "a/b");
  });

  it("normalizes path separators to POSIX", () => {
    // Even on posix-only test runners, verify it handles the native separator cleanly.
    const sep = path.sep;
    assert.equal(bucketFolder(`a${sep}b${sep}note.md`, 2), "a/b");
  });
});

describe("buildOverview", () => {
  it("groups files by bucketed folder and counts notes", () => {
    const files = [
      { absPath: "/v/evergreen/a.md", relPath: "evergreen/a.md", sourceDir: "/v", headings: [] },
      { absPath: "/v/evergreen/b.md", relPath: "evergreen/b.md", sourceDir: "/v", headings: [] },
      { absPath: "/v/taskei/x.md", relPath: "taskei/x.md", sourceDir: "/v", headings: [] },
    ];
    const o = buildOverview(files, ["/v"], { maxDepth: 2, maxKeywordsPerFolder: 0 });
    assert.equal(o.sources.length, 1);
    assert.equal(o.totalNotes, 3);
    assert.equal(o.sources[0].noteCount, 3);
    const byPath = new Map(o.sources[0].folders.map((f) => [f.path, f.noteCount]));
    assert.equal(byPath.get("evergreen"), 2);
    assert.equal(byPath.get("taskei"), 1);
  });

  it("counts code-group files as sources, not notes", () => {
    const files = [
      { absPath: "/v/doc.md", relPath: "doc.md", sourceDir: "/v", headings: [] },
      { absPath: "/v/schema.sql", relPath: "schema.sql", sourceDir: "/v", headings: [], group: "code" as const },
      { absPath: "/v/sub/auth.ts", relPath: "sub/auth.ts", sourceDir: "/v", headings: [], group: "code" as const },
    ];
    const o = buildOverview(files, ["/v"], { maxDepth: 2, maxKeywordsPerFolder: 0 });
    assert.equal(o.totalNotes, 1, "only prose files count as notes");
    assert.equal(o.totalSources, 2, "code files count as sources");
    assert.equal(o.sources[0].noteCount, 1);
    assert.equal(o.sources[0].sourceCount, 2);
    const root = o.sources[0].folders.find((f) => f.path === "");
    assert.equal(root?.noteCount, 1);
    assert.equal(root?.sourceCount, 1);
    const sub = o.sources[0].folders.find((f) => f.path === "sub");
    assert.equal(sub?.noteCount, 0);
    assert.equal(sub?.sourceCount, 1);
  });

  it("sorts folders by combined note + source count", () => {
    const files = [
      { absPath: "/v/notes/a.md", relPath: "notes/a.md", sourceDir: "/v", headings: [] },
      { absPath: "/v/code/a.ts", relPath: "code/a.ts", sourceDir: "/v", headings: [], group: "code" as const },
      { absPath: "/v/code/b.sql", relPath: "code/b.sql", sourceDir: "/v", headings: [], group: "code" as const },
      { absPath: "/v/code/c.ts", relPath: "code/c.ts", sourceDir: "/v", headings: [], group: "code" as const },
    ];
    const o = buildOverview(files, ["/v"], { maxDepth: 1, maxKeywordsPerFolder: 0 });
    assert.equal(o.sources[0].folders[0].path, "code", "code folder (3 files) outranks notes folder (1)");
  });

  it("derives keywords from filenames weighted higher than headings", () => {
    const files = [
      {
        absPath: "/v/vault/rosie-rca-investigation.md",
        relPath: "vault/rosie-rca-investigation.md",
        sourceDir: "/v",
        headings: ["Overview", "Timeline"],
      },
      {
        absPath: "/v/vault/rosie-deployment-notes.md",
        relPath: "vault/rosie-deployment-notes.md",
        sourceDir: "/v",
        headings: ["Deploys"],
      },
      // A folder that never mentions "rosie" — gives the IDF something to chew on.
      {
        absPath: "/v/misc/cat-food.md",
        relPath: "misc/cat-food.md",
        sourceDir: "/v",
        headings: ["Brands"],
      },
    ];
    const o = buildOverview(files, ["/v"], { maxDepth: 1, maxKeywordsPerFolder: 4 });
    const vault = o.sources[0].folders.find((f) => f.path === "vault");
    assert.ok(vault, "expected vault folder");
    assert.ok(vault!.keywords.includes("rosie"), `expected 'rosie' in ${vault!.keywords}`);
  });

  it("picks up a NAPKIN.md / README.md context note", () => {
    const tmp = fs.mkdtempSync(path.join(os.tmpdir(), "ks-overview-"));
    try {
      fs.writeFileSync(path.join(tmp, "NAPKIN.md"), "# Project vault\n\nShort context note.\n");
      const files = [
        { absPath: `${tmp}/notes/a.md`, relPath: "notes/a.md", sourceDir: tmp, headings: [] },
      ];
      const o = buildOverview(files, [tmp]);
      assert.match(o.sources[0].contextNote ?? "", /Project vault/);
    } finally {
      fs.rmSync(tmp, { recursive: true, force: true });
    }
  });

  it("reads _about.md from individual folders", () => {
    const tmp = fs.mkdtempSync(path.join(os.tmpdir(), "ks-overview-about-"));
    try {
      const folder = path.join(tmp, "evergreen");
      fs.mkdirSync(folder, { recursive: true });
      fs.writeFileSync(path.join(folder, "_about.md"), "Atomic evergreen notes live here.");
      const files = [
        {
          absPath: path.join(folder, "a.md"),
          relPath: "evergreen/a.md",
          sourceDir: tmp,
          headings: [],
        },
      ];
      const o = buildOverview(files, [tmp], { maxDepth: 1 });
      const folderEntry = o.sources[0].folders.find((f) => f.path === "evergreen");
      assert.match(folderEntry?.aboutText ?? "", /Atomic evergreen notes/);
    } finally {
      fs.rmSync(tmp, { recursive: true, force: true });
    }
  });

  it("caps folders per source dir by noteCount descending", () => {
    const files = Array.from({ length: 25 }, (_, i) => ({
      absPath: `/v/f${i}/note.md`,
      relPath: `f${i}/note.md`,
      sourceDir: "/v",
      // Variable file count by duplicating — folder f0 has 10 notes, f1 has 9, ...
      headings: [],
    }));
    // Add extras so f0 dominates.
    for (let i = 0; i < 9; i++) {
      files.push({
        absPath: `/v/f0/extra${i}.md`,
        relPath: `f0/extra${i}.md`,
        sourceDir: "/v",
        headings: [],
      });
    }
    const o = buildOverview(files, ["/v"], {
      maxDepth: 1,
      maxFoldersPerDir: 5,
      maxKeywordsPerFolder: 0,
    });
    assert.equal(o.sources[0].folders.length, 5);
    // f0 should be first since it has the most notes (10 total).
    assert.equal(o.sources[0].folders[0].path, "f0");
    assert.equal(o.sources[0].folders[0].noteCount, 10);
  });
});

describe("formatOverview", () => {
  it("returns empty string when there are no notes", () => {
    const text = formatOverview({ sources: [], totalNotes: 0 });
    assert.equal(text, "");
  });

  it("renders one bullet per source dir with its note count", () => {
    const out = formatOverview({
      totalNotes: 3,
      sources: [
        {
          dir: "/v",
          displayName: "v",
          noteCount: 3,
          contextNote: "Hello",
          folders: [
            { path: "evergreen", noteCount: 2, keywords: ["retrieval", "hybrid"] },
            { path: "taskei", noteCount: 1, keywords: [] },
          ],
        },
      ],
    });
    assert.match(out, /Knowledge-search vault overview/);
    assert.match(out, /- \*\*\/v\*\* — 3 notes/);
  });

  it("uses the singular form for a single note", () => {
    const out = formatOverview({
      totalNotes: 1,
      sources: [
        {
          dir: "/v",
          displayName: "v",
          noteCount: 1,
          folders: [{ path: "notes", noteCount: 1, keywords: [] }],
        },
      ],
    });
    assert.match(out, /- \*\*\/v\*\* — 1 note\b/);
  });

  it("omits folders, keywords, and context notes from the output", () => {
    const out = formatOverview({
      totalNotes: 2,
      sources: [
        {
          dir: "/v",
          displayName: "v",
          noteCount: 2,
          contextNote: "LONG README BODY THAT MUST NOT APPEAR",
          folders: [
            {
              path: "notes",
              noteCount: 2,
              keywords: ["kw-one", "kw-two"],
              aboutText: "This folder holds the daily log entries.",
            },
          ],
        },
      ],
    });
    assert.doesNotMatch(out, /LONG README BODY/);
    assert.doesNotMatch(out, /kw-one/);
    assert.doesNotMatch(out, /daily log entries/);
  });

  it("renders code-only dirs as sources and mixed dirs as notes · sources", () => {
    const out = formatOverview({
      totalNotes: 2,
      totalSources: 5,
      sources: [
        { dir: "/docs", displayName: "docs", noteCount: 2, folders: [] },
        { dir: "/tables", displayName: "tables", noteCount: 0, sourceCount: 5, folders: [] },
        {
          dir: "/mixed",
          displayName: "mixed",
          noteCount: 1,
          sourceCount: 1,
          folders: [],
        },
      ],
    });
    assert.match(out, /- \*\*\/docs\*\* — 2 notes/);
    assert.match(out, /- \*\*\/tables\*\* — 5 sources\b/, "code-only dir reads as sources");
    assert.match(out, /- \*\*\/mixed\*\* — 1 note · 1 source\b/, "mixed dir shows both segments");
    assert.doesNotMatch(out, /0 notes/, "zero segments are omitted");
    assert.doesNotMatch(out, /0 sources/, "zero segments are omitted");
  });
});
