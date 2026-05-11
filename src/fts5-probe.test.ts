import { describe, it } from "node:test";
import assert from "node:assert/strict";
import { assertFts5Available, _resetFts5ProbeCache } from "./fts5-probe.js";

describe("assertFts5Available", () => {
  it("returns without throwing on a Node runtime with FTS5 (this CI env)", () => {
    _resetFts5ProbeCache();
    assert.doesNotThrow(() => assertFts5Available());
  });

  it("caches the positive result (second call is a fast no-op)", () => {
    _resetFts5ProbeCache();
    assertFts5Available();
    assert.doesNotThrow(() => assertFts5Available());
  });
});
