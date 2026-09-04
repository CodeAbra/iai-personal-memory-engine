import { strict as assert } from "node:assert";
import { describe, it } from "node:test";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import path from "node:path";

import { toolSchemas } from "../src/tools.js";

const EXPECTED_ENUM = ["fact", "estimate", "hypothesis", "opinion", "unknown"];
const EXPECTED_SALIENCE_ENUM = ["unflagged", "notable", "critical"];

describe("epistemic_status tool schema", () => {
  it("memory_capture exposes the 5-value enum, optional", () => {
    const schema = toolSchemas.memory_capture.inputSchema as {
      properties: Record<string, { enum?: string[] }>;
      required: string[];
    };
    assert.deepEqual(schema.properties.epistemic_status.enum, EXPECTED_ENUM);
    assert.ok(!schema.required.includes("epistemic_status"));
  });

  it("memory_contradict exposes the 5-value enum, optional", () => {
    const schema = toolSchemas.memory_contradict.inputSchema as {
      properties: Record<string, { enum?: string[] }>;
      required: string[];
    };
    assert.deepEqual(schema.properties.epistemic_status.enum, EXPECTED_ENUM);
    assert.ok(!schema.required.includes("epistemic_status"));
  });

  it("dist-bundle/index.js is rebuilt and not stale", () => {
    const here = path.dirname(fileURLToPath(import.meta.url));
    const bundlePath = path.join(here, "..", "dist-bundle", "index.js");
    const bundle = readFileSync(bundlePath, "utf-8");
    const idx = bundle.indexOf("epistemic_status");
    assert.notEqual(
      idx, -1,
      "dist-bundle/index.js does not contain epistemic_status — rebuild with `npm run bundle`",
    );
    // A stale bundle built before this enum landed would still match a bare
    // substring check (e.g. from an unrelated occurrence). Requiring every
    // enum member to appear within a short window of the key detects drift
    // between the live schema and the checked-out bundle, not just presence.
    const window = bundle.slice(idx, idx + 200);
    for (const member of EXPECTED_ENUM) {
      assert.ok(
        window.includes(`"${member}"`),
        `dist-bundle/index.js is stale — missing enum member "${member}" near epistemic_status`,
      );
    }
  });
});

describe("salience_level tool schema", () => {
  it("memory_capture exposes the 3-value enum, optional, defaulting to unflagged", () => {
    const schema = toolSchemas.memory_capture.inputSchema as {
      properties: Record<string, { enum?: string[]; default?: string }>;
      required: string[];
    };
    assert.deepEqual(schema.properties.salience_level.enum, EXPECTED_SALIENCE_ENUM);
    assert.equal(schema.properties.salience_level.default, "unflagged");
    assert.ok(!schema.required.includes("salience_level"));
  });

  it("dist-bundle/index.js is rebuilt and not stale", () => {
    const here = path.dirname(fileURLToPath(import.meta.url));
    const bundlePath = path.join(here, "..", "dist-bundle", "index.js");
    const bundle = readFileSync(bundlePath, "utf-8");
    const idx = bundle.indexOf("salience_level");
    assert.notEqual(
      idx, -1,
      "dist-bundle/index.js does not contain salience_level — rebuild with `npm run bundle`",
    );
    // A stale bundle built before this enum landed would still match a bare
    // substring check (e.g. from an unrelated occurrence). Requiring every
    // enum member to appear within a short window of the key detects drift
    // between the live schema and the checked-out bundle, not just presence.
    const window = bundle.slice(idx, idx + 200);
    for (const member of EXPECTED_SALIENCE_ENUM) {
      assert.ok(
        window.includes(`"${member}"`),
        `dist-bundle/index.js is stale — missing enum member "${member}" near salience_level`,
      );
    }
  });
});
