// Parity: the JS port must reproduce the Python interpreter's behavior on
// the golden programs. Regenerate golden.json with tools/make_golden.py.
// Run: node --test docs/playground/test/
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { test } from "node:test";
import { fileURLToPath } from "node:url";

import { assemble } from "../asm.js";
import { Machine, fromDict } from "../versor.js";

const golden = JSON.parse(
  readFileSync(join(dirname(fileURLToPath(import.meta.url)), "golden.json")),
);

function check(program, expected) {
  const m = new Machine(program, { trace: true });
  const res = m.run();
  assert.equal(res.haltReason, expected.haltReason);
  assert.equal(res.steps, expected.steps);
  assert.deepEqual(m.trace.map((r) => r.opcode), expected.opcodes);
  assert.equal(res.out.length, expected.out.length);
  for (let i = 0; i < res.out.length; i++) {
    if (typeof expected.out[i] === "string") {
      assert.equal(res.out[i], expected.out[i]);
    } else {
      assert.ok(Math.abs(res.out[i] - expected.out[i]) < 1e-9,
        `out[${i}]: ${res.out[i]} != ${expected.out[i]}`);
    }
  }
  for (let i = 0; i < 3; i++) {
    assert.ok(Math.abs(res.displacement[i] - expected.displacement[i]) < 1e-9,
      `displacement[${i}]`);
  }
}

for (const entry of golden) {
  test(`golden: ${entry.name}`, () => {
    check(fromDict(entry.program), entry.expected);
  });
  if (entry.vasm) {
    test(`golden (assembled in JS): ${entry.name}`, () => {
      check(fromDict(toPlain(assemble(entry.vasm))), entry.expected);
    });
  }
}

// assemble() returns the program shape directly; normalize via fromDict's
// input contract (vertices as arrays)
function toPlain(prog) {
  return {
    name: prog.name,
    decoder: prog.decoder,
    chains: prog.chains.map((c) => ({
      id: c.id,
      vertices: Object.entries(c.vertices).map(([vid, out]) => ({
        id: Number(vid),
        out: out.map((e) => ({
          seg: e.seg, to: e.to,
          ...(e.guard ? { guard: e.guard } : {}),
        })),
      })),
    })),
  };
}
