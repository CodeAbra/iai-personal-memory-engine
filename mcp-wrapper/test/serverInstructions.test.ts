import { strict as assert } from "node:assert";
import { describe, it } from "node:test";

import {
  CONTEXT_EDITING_CONFIG,
  HOT_TOOLS,
  SERVER_INSTRUCTIONS,
} from "../src/index.js";
import { toolSchemas } from "../src/tools.js";

describe("server instructions routing prose", () => {
  it("orders recall before repository search, never as a replacement", () => {
    assert.match(SERVER_INSTRUCTIONS, /memory_recall BEFORE searching/);
    assert.match(SERVER_INSTRUCTIONS, /never replaces/);
    assert.match(SERVER_INSTRUCTIONS, /file search/);
    assert.doesNotMatch(SERVER_INSTRUCTIONS, /instead of grep/i);
  });

  it("carries no hardcoded token-cost figures", () => {
    assert.doesNotMatch(SERVER_INSTRUCTIONS, /~?\d[\d,]* tokens/);
  });

  it("marks recalled code claims as historical", () => {
    assert.match(SERVER_INSTRUCTIONS, /historical/);
    assert.match(SERVER_INSTRUCTIONS, /valid_from\/valid_to/);
  });

  it("still carries the machine config, parseable", () => {
    const marker = "config: ";
    const at = SERVER_INSTRUCTIONS.indexOf(marker);
    assert.ok(at > 0, "config marker present after the prose");
    const parsed = JSON.parse(SERVER_INSTRUCTIONS.slice(at + marker.length));
    assert.deepEqual(parsed.context_editing, CONTEXT_EDITING_CONFIG);
    assert.deepEqual(parsed.hot_tools, HOT_TOOLS);
  });
});

describe("tool descriptions carry the routing within the 30-token ceiling", () => {
  it("memory_recall names its question class and the ordering", () => {
    const d = toolSchemas.memory_recall.description;
    assert.match(d, /decisions, preferences, prior discussion, rationale/);
    assert.match(d, /before a repository search/);
  });

  it("memory_search keeps hints-to-verify and rejects substitution", () => {
    const d = toolSchemas.memory_search.description;
    assert.match(d, /returns hints to verify/);
    assert.match(d, /never replaces a repository search/);
  });
});
