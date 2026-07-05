/* Versor core, ported from the Python reference implementation
 * (versor/quat.py, decode.py, isa.py, machine.py, builder.py, asm.py).
 * Semantics parity is enforced by test/parity.test.mjs against golden
 * outputs generated from the Python interpreter. Plain ES module, no deps.
 */

export const EPS = 1e-6;

// ---------- vectors & quaternions ----------
export const vadd = (a, b) => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
export const vsub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
export const vscale = (a, s) => [a[0] * s, a[1] * s, a[2] * s];
export const vdot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
export const vcross = (a, b) => [
  a[1] * b[2] - a[2] * b[1],
  a[2] * b[0] - a[0] * b[2],
  a[0] * b[1] - a[1] * b[0],
];
export const vnorm = (a) => Math.sqrt(vdot(a, a));

export const QID = [1, 0, 0, 0]; // [w, x, y, z]

export function qmul(a, b) {
  const [w1, x1, y1, z1] = a, [w2, x2, y2, z2] = b;
  return [
    w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
    w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
    w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
    w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
  ];
}
export const qconj = (q) => [q[0], -q[1], -q[2], -q[3]];
export function qnormalize(q) {
  const n = Math.hypot(q[0], q[1], q[2], q[3]);
  return [q[0] / n, q[1] / n, q[2] / n, q[3] / n];
}
export function qrot(q, v) {
  const u = [q[1], q[2], q[3]];
  const t = vscale(vcross(u, v), 2);
  return vadd(vadd(v, vscale(t, q[0])), vcross(u, t));
}
export function axisAngle(axis, angle) {
  const n = vnorm(axis);
  const h = angle / 2, s = Math.sin(h) / n;
  return [Math.cos(h), axis[0] * s, axis[1] * s, axis[2] * s];
}

// ---------- faults ----------
export class VersorFault extends Error {
  constructor(kind, message, loc) {
    super(`${kind}: ${message}${loc ? ` [step ${loc.step}, chain ${loc.chain}, vertex ${loc.vertex}]` : ""}`);
    this.kind = kind;
    this.shortMessage = message;
  }
}

// ---------- decoders ----------
const THRESHOLD = 0.35, DEAD_ZONE = 0.05, NN_MARGIN = 0.01;
const PHI = (1 + Math.sqrt(5)) / 2;

const key = (t) => t.join(",");

class Cubic26 {
  constructor() { this.name = "cubic26"; }
  decode(v) {
    const s = [];
    for (const c of v) {
      if (Math.abs(Math.abs(c) - THRESHOLD) < DEAD_ZONE) {
        throw new VersorFault("AmbiguousDirection",
          `component ${c.toFixed(4)} within dead zone of threshold ±${THRESHOLD}`);
      }
      s.push(c > THRESHOLD ? 1 : c < -THRESHOLD ? -1 : 0);
    }
    if (!s[0] && !s[1] && !s[2]) {
      throw new VersorFault("AmbiguousDirection", "vector quantized to (0,0,0)");
    }
    return s;
  }
  directions() {
    const out = {};
    for (const t of ALL_TRIPLES) {
      const n = vnorm(t);
      out[key(t)] = vscale(t, 1 / n);
    }
    return out;
  }
}

class NearestNeighbor {
  constructor(name, entries) {
    this.name = name;
    this.entries = entries; // [{v, triple|null}]
  }
  decode(v) {
    let best = -1, bi = -1, second = -1;
    for (let i = 0; i < this.entries.length; i++) {
      const d = vdot(this.entries[i].v, v);
      if (d > best) { second = best; best = d; bi = i; }
      else if (d > second) { second = d; }
    }
    if (best - second < NN_MARGIN) {
      throw new VersorFault("AmbiguousDirection",
        `within ${NN_MARGIN} of a ${this.name} cone boundary`);
    }
    const t = this.entries[bi].triple;
    if (t === null) {
      throw new VersorFault("ReservedDirection",
        `direction is a reserved ${this.name} cone`);
    }
    return t.slice();
  }
  directions() {
    const out = {};
    for (const e of this.entries) {
      if (e.triple !== null) out[key(e.triple)] = e.v.slice();
    }
    return out;
  }
}

function icosaEntries() {
  const e = [];
  const unit = (v) => vscale(v, 1 / vnorm(v));
  for (const sx of [-1, 1]) for (const sy of [-1, 1]) for (const sz of [-1, 1]) {
    e.push({ v: unit([sx, sy, sz]), triple: [sx, sy, sz] });
  }
  for (const s1 of [-1, 1]) for (const s2 of [-1, 1]) {
    e.push({ v: unit([0, s1, s2 * PHI]), triple: [0, s1, s2] });
    e.push({ v: unit([s1, s2 * PHI, 0]), triple: [s1, s2, 0] });
    e.push({ v: unit([s1 * PHI, 0, s2]), triple: [s1, 0, s2] });
  }
  for (const s of [-1, 1]) {
    e.push({ v: unit([s * PHI, s / PHI, 0]), triple: [s, 0, 0] });
    e.push({ v: unit([0, s * PHI, s / PHI]), triple: [0, s, 0] });
    e.push({ v: unit([s / PHI, 0, s * PHI]), triple: [0, 0, s] });
    e.push({ v: unit([s * PHI, -s / PHI, 0]), triple: null });
    e.push({ v: unit([0, s * PHI, -s / PHI]), triple: null });
    e.push({ v: unit([-s / PHI, 0, s * PHI]), triple: null });
  }
  return e;
}

// frozen table from tools/optimize_sphere26.py (see versor/decode.py)
const SPHERE26_TABLE = [
  [[-0.600301876388214, -0.598848634433877, +0.530111280998122], [-1, -1, 1]],
  [[-0.446406946572806, +0.184324847647420, -0.875639873801610], [-1, 0, -1]],
  [[-0.869932147354444, +0.045068511287929, +0.491107817377789], [-1, 0, 1]],
  [[-0.691661090487099, -0.713721202769234, -0.110485205452290], [-1, -1, 0]],
  [[+0.049935126077074, -0.699471611273205, +0.712913703197331], [0, -1, 1]],
  [[-0.747469029906257, +0.558081339546148, -0.360326612646605], [-1, 1, -1]],
  [[-0.223030638816671, +0.964599211521904, -0.140732708637151], [-1, 1, 0]],
  [[+0.199787421253120, -0.492401221198910, -0.847128103459477], [0, -1, -1]],
  [[-0.647304747219036, +0.683174579798115, +0.338037065638325], [-1, 1, 1]],
  [[+0.290837759716615, +0.152210094231070, -0.944587468018282], [0, 0, -1]],
  [[-0.971777467495341, -0.082639230854734, -0.220950924850019], [-1, 0, 0]],
  [[-0.111562925436732, -0.901734944353629, -0.417645548042304], [0, -1, 0]],
  [[-0.528894555907088, -0.449926282823061, -0.719608844273656], [-1, -1, -1]],
];

function sphere26Entries() {
  const e = [];
  for (const [v, t] of SPHERE26_TABLE) {
    e.push({ v: v.slice(), triple: t.slice() });
    e.push({ v: vscale(v, -1), triple: t.map((c) => -c) });
  }
  return e;
}

export const DECODERS = {
  cubic26: () => new Cubic26(),
  icosa32: () => new NearestNeighbor("icosa32", icosaEntries()),
  sphere26: () => new NearestNeighbor("sphere26", sphere26Entries()),
};

const ALL_TRIPLES = [];
for (const x of [-1, 0, 1]) for (const y of [-1, 0, 1]) for (const z of [-1, 0, 1]) {
  if (x || y || z) ALL_TRIPLES.push([x, y, z]);
}

// ---------- ISA ----------
const regIndex = (n) => Math.floor(n) % 4;
const localX = (m) => qrot(qconj(m.F), m.A)[0];
const setLocalX = (m, s) => { m.A = qrot(m.F, [s, 0, 0]); };

export const OPCODES = new Map(); // "sx,sy,sz" -> {mnemonic, klass, handler}
function op(triple, mnemonic, klass, handler) {
  OPCODES.set(key(triple), { mnemonic, klass, handler, triple });
}

op([1, 0, 0], "LOADI", "data", (m, n) => setLocalX(m, n));
op([-1, 0, 0], "STORE", "data", (m) => { m.M.set(m.cellKey(), m.A.slice()); });
op([0, 1, 0], "LOAD", "data", (m) => { m.A = (m.M.get(m.cellKey()) || [0, 0, 0]).slice(); });
op([0, -1, 0], "MOVR", "data", (m, n) => { m.R[regIndex(n)] = m.A.slice(); });
op([0, 0, 1], "MOVA", "data", (m, n) => { m.A = m.R[regIndex(n)].slice(); });
op([0, 0, -1], "HALT", "control", (m) => m.halt("HALT"));
op([1, 1, 0], "ADD", "arithmetic", (m, n) => { m.A = vadd(m.A, m.R[regIndex(n)]); });
op([1, -1, 0], "SUB", "arithmetic", (m, n) => { m.A = vsub(m.A, m.R[regIndex(n)]); });
op([-1, 1, 0], "SCALE", "arithmetic", (m, n) => { m.A = vscale(m.A, n); });
op([-1, -1, 0], "DOT", "arithmetic", (m, n) => setLocalX(m, vdot(m.A, m.R[regIndex(n)])));
op([1, 0, 1], "CROSS", "arithmetic", (m, n) => { m.A = vcross(m.A, m.R[regIndex(n)]); });
op([1, 0, -1], "NORM", "arithmetic", (m) => {
  const a = vnorm(m.A);
  if (a < EPS) throw new VersorFault("DivisionByZero", "NORM of zero-magnitude accumulator");
  m.A = vscale(m.A, 1 / a);
});
function projVec(m, n) {
  const r = m.R[regIndex(n)];
  const rr = vdot(r, r);
  if (rr < EPS * EPS) throw new VersorFault("DivisionByZero", "projection onto zero-magnitude register");
  return vscale(r, vdot(m.A, r) / rr);
}
op([-1, 0, 1], "PROJ", "arithmetic", (m, n) => { m.A = projVec(m, n); });
op([-1, 0, -1], "REJ", "arithmetic", (m, n) => { m.A = vsub(m.A, projVec(m, n)); });
op([0, 1, 1], "ROTF", "frame", (m, n) => { m.F = qnormalize(qmul(m.F, axisAngle([1, 0, 0], n))); });
op([0, 1, -1], "ROTG", "frame", (m, n) => { m.F = qnormalize(qmul(m.F, axisAngle([0, 1, 0], n))); });
op([0, -1, 1], "ROTH", "frame", (m, n) => { m.F = qnormalize(qmul(m.F, axisAngle([0, 0, 1], n))); });
op([0, -1, -1], "OUT", "data", (m, n) => {
  const x = localX(m);
  if (n >= 2.0 - 1e-9) {
    const c = Math.round(x);
    if (!Number.isFinite(x) || c < 0 || c >= 0x110000) {
      throw new VersorFault("InvalidCharCode", `OUT char mode with A.x = ${x}`);
    }
    m.OUT.push(String.fromCodePoint(c));
  } else {
    m.OUT.push(x);
  }
});
op([1, 1, 1], "CALL", "control", (m, n) => {
  const cid = Math.floor(n) % m.program.chains.length;
  if (m.CS.length >= m.maxCallDepth) {
    throw new VersorFault("CallStackOverflow", `call depth ${m.maxCallDepth} exceeded`);
  }
  m.CS.push([m.chain, m.vertex, m.F, m.P.slice()]);
  m.chain = cid;
  m.vertex = 0;
});
op([1, 1, -1], "RET", "control", (m) => m.doRet());
op([1, -1, 1], "JMPZ", "control", (m) => { if (vnorm(m.A) < EPS) m.skip = true; });
op([1, -1, -1], "JMPP", "control", (m) => { if (localX(m) > EPS) m.skip = true; });
op([-1, 1, 1], "PUSHF", "frame", (m) => { m.AUX.push([m.F, m.P.slice()]); });
op([-1, 1, -1], "POPF", "frame", (m) => {
  if (!m.AUX.length) throw new VersorFault("StackUnderflow", "POPF on empty aux stack");
  m.F = m.AUX.pop()[0];
});
op([-1, -1, 1], "NOP", "control", () => {});
op([-1, -1, -1], "FAULT", "control", (m, n) => {
  throw new VersorFault("ExplicitFault", `FAULT opcode, operand ${n}`);
});

export const MNEMONIC_TO_TRIPLE = {};
for (const { mnemonic, triple } of OPCODES.values()) MNEMONIC_TO_TRIPLE[mnemonic] = triple;

// ---------- machine ----------
export class Machine {
  constructor(program, opts = {}) {
    this.program = program;
    this.decoder = DECODERS[opts.decoder || program.decoder || "cubic26"]();
    this.stepBudget = opts.stepBudget ?? 1_000_000;
    this.maxCallDepth = opts.maxCallDepth ?? 1024;
    this.P = [0, 0, 0];
    this.F = opts.F0 ? opts.F0.slice() : QID.slice();
    this.A = [0, 0, 0];
    this.R = [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]];
    this.M = new Map();
    this.CS = [];
    this.AUX = [];
    this.OUT = [];
    this.chain = 0;
    this.vertex = 0;
    this.steps = 0;
    this.skip = false;
    this.halted = false;
    this.haltReason = "";
    this.trace = opts.trace ? [] : null;
  }

  cellKey() {
    return `${Math.floor(this.P[0])},${Math.floor(this.P[1])},${Math.floor(this.P[2])}`;
  }

  halt(reason) { this.halted = true; this.haltReason = reason; }

  fault(kind, message) {
    this.halt(`fault: ${kind}`);
    throw new VersorFault(kind, message,
      { step: this.steps, chain: this.chain, vertex: this.vertex });
  }

  doRet() {
    if (!this.CS.length) this.fault("StackUnderflow", "RET with empty call stack");
    const [chain, vertex, f, p] = this.CS.pop();
    this.A = vsub(this.P, p);
    this.F = f;
    this.chain = chain;
    this.vertex = vertex;
  }

  pickBranch(edges) {
    const na = vnorm(this.A);
    const a = na >= EPS ? vscale(this.A, 1 / na) : [0, 0, 0];
    let best = null, bestDot = -Infinity;
    for (const e of edges) {
      const d = vdot(qrot(this.F, e.guard), a);
      if (d > bestDot + 1e-12) { best = e; bestDot = d; }
    }
    return best;
  }

  step() {
    if (this.halted) return;
    if (this.steps >= this.stepBudget) {
      this.fault("StepBudgetExhausted", `budget of ${this.stepBudget} steps`);
    }
    const edges = this.program.chains[this.chain].vertices[this.vertex];
    if (!edges || !edges.length) {
      if (!this.CS.length) { this.halt("end of root chain"); return; }
      const frm = this.vertex;
      this.doRet();
      this.steps += 1;
      if (this.trace) {
        this.trace.push({
          step: this.steps, chain: this.chain, frm, to: this.vertex,
          P0: this.P.slice(), P1: this.P.slice(), F: this.F.slice(),
          opcode: "RET*", klass: "control", n: 0, A: this.A.slice(),
          skipped: false, branch: false, outLen: this.OUT.length,
        });
      }
      return;
    }
    const isBranch = edges.length > 1;
    const edge = isBranch ? this.pickBranch(edges) : edges[0];
    const vraw = edge.seg;
    const n = vnorm(vraw);
    if (n < EPS) this.fault("ZeroLengthSegment", "cannot decode a zero-length segment");
    const vlocal = qrot(qconj(this.F), vraw);
    let triple;
    try {
      triple = this.decoder.decode(vscale(vlocal, 1 / n));
    } catch (f) {
      if (f instanceof VersorFault) this.fault(f.kind, f.shortMessage);
      throw f;
    }
    const o = OPCODES.get(key(triple));
    const frm = this.vertex;
    const p0 = this.P.slice();
    this.P = vadd(this.P, vraw);
    this.vertex = edge.to;
    const skipped = this.skip;
    if (skipped) {
      this.skip = false;
    } else {
      try {
        o.handler(this, n);
      } catch (f) {
        if (f instanceof VersorFault && f.shortMessage !== undefined && !f.located) {
          this.halt(`fault: ${f.kind}`);
          const nf = new VersorFault(f.kind, f.shortMessage,
            { step: this.steps, chain: this.chain, vertex: this.vertex });
          nf.located = true;
          throw nf;
        }
        throw f;
      }
    }
    this.steps += 1;
    if (this.trace) {
      this.trace.push({
        step: this.steps, chain: this.chain, frm, to: this.vertex,
        P0: p0, P1: vadd(p0, vraw), F: this.F.slice(), opcode: o.mnemonic,
        klass: o.klass, n, A: this.A.slice(), skipped, branch: isBranch,
        outLen: this.OUT.length,
      });
    }
  }

  run() {
    while (!this.halted) this.step();
    return {
      out: this.OUT, displacement: this.P.slice(),
      haltReason: this.haltReason, steps: this.steps,
    };
  }
}

// ---------- program model ----------
export function fromDict(data) {
  const chains = [];
  for (const rc of data.chains) {
    const vertices = {};
    for (const rv of rc.vertices) {
      vertices[rv.id] = (rv.out || []).map((e) => ({
        seg: e.seg.slice(), to: e.to,
        guard: e.guard ? vscale(e.guard, 1 / vnorm(e.guard)) : null,
      }));
    }
    chains[rc.id] = { id: rc.id, vertices, comment: rc.comment || "" };
  }
  return { name: data.name || "", decoder: data.decoder || "cubic26", chains };
}
