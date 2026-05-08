import { describe, it } from "node:test";
import assert from "node:assert/strict";
import * as fs from "node:fs";
import * as os from "node:os";
import * as path from "node:path";
import { normalizeRef, resolveNote, readNote, type IndexedFile } from "./kb-reader.js";

describe("normalizeRef", () => {
  it("strips [[ ]] wrappers", () => {
    assert.deepEqual(normalizeRef("[[Foo bar]]"), { ref: "Foo bar", subheading: undefined });
  });

  it("takes the left side of a piped alias", () => {
    assert.deepEqual(normalizeRef("[[a/b|Display name]]"), { ref: "a/b", subheading: undefined });
  });

  it("extracts #subheading", () => {
    assert.deepEqual(normalizeRef("[[Foo#Heading]]"), { ref: "Foo", subheading: "Heading" });
  });

  it("handles plain bare references", () => {
    assert.deepEqual(normalizeRef("   just-a-name   "), {
      ref: "just-a-name",
      subheading: undefined,
    });
  });

  it("handles forgiving unterminated wikilinks", () => {
    assert.deepEqual(normalizeRef("[[foo"), { ref: "foo", subheading: undefined });
    assert.deepEqual(normalizeRef("foo]]"), { ref: "foo", subheading: undefined });
  });
});

describe("resolveNote", () => {
  const files: IndexedFile[] = [
    { absPath: "/v/Notes/Evergreen/hybrid-search.md", relPath: "Notes/Evergreen/hybrid-search.md", sourceDir: "/v" },
    { absPath: "/v/Notes/Evergreen/Rosie architecture.md", relPath: "Notes/Evergreen/Rosie architecture.md", sourceDir: "/v" },
    { absPath: "/v/TaskNotes/Tasks/foo.md", relPath: "TaskNotes/Tasks/foo.md", sourceDir: "/v" },
    { absPath: "/other/foo.md", relPath: "foo.md", sourceDir: "/other" },
  ];

  it("resolves an absolute path that matches an indexed file", () => {
    const r = resolveNote("/v/Notes/Evergreen/hybrid-search.md", files);
    assert.ok(r.unique);
    assert.equal(r.matches[0].absPath, "/v/Notes/Evergreen/hybrid-search.md");
    assert.equal(r.matches[0].reason, "absolute");
  });

  it("ignores absolute paths that aren't indexed", () => {
    const r = resolveNote("/random/elsewhere.md", files);
    assert.equal(r.matches.length, 0);
  });

  it("resolves a relative path to a source dir", () => {
    const r = resolveNote("Notes/Evergreen/hybrid-search.md", files);
    assert.ok(r.unique);
    assert.equal(r.matches[0].reason, "relative-to-source");
  });

  it("resolves a relative path without extension", () => {
    const r = resolveNote("Notes/Evergreen/hybrid-search", files);
    assert.ok(r.unique);
    assert.equal(r.matches[0].absPath, "/v/Notes/Evergreen/hybrid-search.md");
  });

  it("resolves an exact basename match", () => {
    const r = resolveNote("hybrid-search", files);
    assert.ok(r.unique);
    assert.equal(r.matches[0].reason, "basename-exact");
  });

  it("resolves a case-insensitive basename", () => {
    const r = resolveNote("HYBRID-SEARCH", files);
    assert.ok(r.unique);
  });

  it("resolves wikilink forms", () => {
    const r = resolveNote("[[hybrid-search]]", files);
    assert.ok(r.unique);
    const r2 = resolveNote("[[Notes/Evergreen/hybrid-search|Hybrid]]", files);
    assert.ok(r2.unique);
  });

  it("extracts subheading from wikilink", () => {
    const r = resolveNote("[[hybrid-search#RRF]]", files);
    assert.ok(r.unique);
    assert.equal(r.subheading, "RRF");
  });

  it("handles filenames with spaces and mixed case", () => {
    const r = resolveNote("Rosie architecture", files);
    assert.ok(r.unique);
    assert.equal(r.matches[0].absPath, "/v/Notes/Evergreen/Rosie architecture.md");
  });

  it("treats a unique relpath suffix as high confidence", () => {
    const r = resolveNote("Evergreen/hybrid-search", files);
    assert.ok(r.unique, `expected unique for relpath suffix, got matches=${JSON.stringify(r.matches)}`);
    assert.equal(r.matches[0].absPath, "/v/Notes/Evergreen/hybrid-search.md");
  });

  it("returns multiple matches when the name is ambiguous", () => {
    const r = resolveNote("foo", files);
    assert.equal(r.matches.length, 2);
    assert.equal(r.unique, false);
  });

  it("falls back to substring match as a last resort", () => {
    const r = resolveNote("Evergreen", files);
    // No file is literally named "Evergreen"; substring hits both evergreen notes.
    assert.ok(r.matches.length >= 2);
    assert.equal(r.matches.every((m) => m.reason === "substring"), true);
    assert.equal(r.unique, false);
  });

  it("returns empty result for an empty reference", () => {
    const r = resolveNote("", files);
    assert.equal(r.matches.length, 0);
  });

  it("returns empty result when nothing matches", () => {
    const r = resolveNote("nonexistent-note-xyz", files);
    assert.equal(r.matches.length, 0);
  });

  it("normalizes fileExtensions that lack a leading dot", () => {
    // Users who set fileExtensions to ["md"] (no dot) in the config file
    // used to break extension-stripping logic. resolveNote should tolerate it.
    const r = resolveNote("hybrid-search", files, { fileExtensions: ["md"] });
    assert.ok(r.unique);
    assert.equal(r.matches[0].absPath, "/v/Notes/Evergreen/hybrid-search.md");
  });
});

describe("readNote", () => {
  it("reads a file in full when under the byte cap", () => {
    const tmp = fs.mkdtempSync(path.join(os.tmpdir(), "ks-read-"));
    try {
      const p = path.join(tmp, "n.md");
      fs.writeFileSync(p, "Hello world\nSecond line\n");
      const r = readNote(p);
      assert.equal(r.truncated, false);
      assert.match(r.content, /Hello world/);
      assert.match(r.content, /Second line/);
    } finally {
      fs.rmSync(tmp, { recursive: true, force: true });
    }
  });

  it("truncates large files and reports total bytes", () => {
    const tmp = fs.mkdtempSync(path.join(os.tmpdir(), "ks-read-big-"));
    try {
      const p = path.join(tmp, "big.md");
      const body = "line\n".repeat(20000); // 100_000 bytes
      fs.writeFileSync(p, body);
      const r = readNote(p, { maxBytes: 1024 });
      assert.equal(r.truncated, true);
      assert.ok(r.content.length <= 1024);
      assert.equal(r.totalBytes, body.length);
    } finally {
      fs.rmSync(tmp, { recursive: true, force: true });
    }
  });

  it("never emits a partial UTF-8 codepoint at the truncation boundary", () => {
    const tmp = fs.mkdtempSync(path.join(os.tmpdir(), "ks-read-utf8-"));
    try {
      const p = path.join(tmp, "utf8.md");
      // 4-byte codepoint (📚 = U+1F4DA) repeated enough to push past any byte cap.
      // No newlines so the "cut at last newline" fallback doesn't help.
      const body = "📚".repeat(4000);
      fs.writeFileSync(p, body);
      // Pick a maxBytes that lands mid-codepoint (4-byte char starts at multiples of 4,
      // so 1023 is guaranteed to be inside a codepoint).
      const r = readNote(p, { maxBytes: 1023 });
      assert.equal(r.truncated, true);
      // No replacement character at the end — we backed off to a valid boundary.
      assert.equal(r.content.includes("\uFFFD"), false, "content contains U+FFFD replacement char");
      // Every codepoint is still the full book emoji.
      for (const c of r.content) {
        assert.equal(c, "📚");
      }
    } finally {
      fs.rmSync(tmp, { recursive: true, force: true });
    }
  });
});
