/* .vasm assembler, ported from versor/asm.py + the builder core it rides on
 * (authoring-frame tracking, labels with forward references, branch arms).
 * Parity with Python is covered by test/parity.test.mjs.
 */
import {
  DECODERS, MNEMONIC_TO_TRIPLE, axisAngle, qmul, qnormalize, qrot,
  vnorm, vscale,
} from "./versor.js";

const GUARDS = {
  x: [1, 0, 0], "+x": [1, 0, 0], "-x": [-1, 0, 0],
  y: [0, 1, 0], "+y": [0, 1, 0], "-y": [0, -1, 0],
  z: [0, 0, 1], "+z": [0, 0, 1], "-z": [0, 0, -1],
};
const REG_OPS = new Set(["MOVR", "MOVA", "ADD", "SUB", "DOT", "CROSS", "PROJ", "REJ"]);
const FLOAT_OPS = new Set(["LOADI", "SCALE"]);
const ANGLE_OPS = new Set(["ROTF", "ROTG", "ROTH"]);
const ROT_AXES = { ROTF: [1, 0, 0], ROTG: [0, 1, 0], ROTH: [0, 0, 1] };
const DEFAULT_N = {
  HALT: 1, NOP: 1, RET: 1, JMPZ: 1, JMPP: 1, PUSHF: 1, POPF: 1,
  NORM: 1, STORE: 1, LOAD: 1, FAULT: 1, OUT: 1, OUTC: 2, EXEC: 2,
};
const PSEUDO = { OUTC: "OUT", EXEC: "LOAD" };
const LABEL_RE = /^([A-Za-z_]\w*):\s*(.*)$/;
const REG_RE = /^[rR]([0-3])$/;
const PI_RE = /^(\d+\.?\d*)?\s*\*?\s*pi\s*(?:\/\s*(\d+\.?\d*))?$/i;

export class AsmError extends Error {}
const err = (ln, msg) => new AsmError(`line ${ln}: ${msg}`);

const strip = (line) => line.split(";")[0].split("#")[0].trim();

function parseNum(s, ln, what) {
  const v = Number(s);
  if (!Number.isFinite(v) || s.trim() === "") throw err(ln, `${what}: expected a number, got '${s}'`);
  return v;
}

function parseAngle(s, ln) {
  const m = PI_RE.exec(s.trim());
  if (m) {
    const k = m[1] ? Number(m[1]) : 1;
    const d = m[2] ? Number(m[2]) : 1;
    return (k * Math.PI) / d;
  }
  return parseNum(s, ln, "angle");
}

function parseVec(s, ln, what) {
  s = s.trim();
  if (!s.startsWith("(") || !s.endsWith(")")) throw err(ln, `${what}: expected (x, y, z)`);
  const parts = s.slice(1, -1).split(",");
  if (parts.length !== 3) throw err(ln, `${what}: expected 3 components, got ${parts.length}`);
  return parts.map((p) => parseNum(p, ln, what));
}

function parseGuard(s, ln) {
  const g = GUARDS[s.trim().toLowerCase()];
  return g ? g.slice() : parseVec(s, ln, "guard");
}

function splitTop(s) {
  const parts = [];
  let depth = 0, cur = "";
  for (const ch of s) {
    if (ch === "(") depth++;
    else if (ch === ")") depth--;
    if (ch === "," && depth === 0) { parts.push(cur); cur = ""; }
    else cur += ch;
  }
  parts.push(cur);
  return parts.map((p) => p.trim()).filter(Boolean);
}

function operand(mnemonic, args, ln, chainIds, nChains) {
  args = args.trim();
  if (REG_OPS.has(mnemonic)) {
    if (!args) throw err(ln, `${mnemonic} needs a register (r0..r3) or magnitude`);
    const m = REG_RE.exec(args);
    if (m) return Number(m[1]) + 0.5;
    if (/^[rR]/.test(args) && !/^\d/.test(args.slice(1, 2))) {
      throw err(ln, `${mnemonic}: bad register '${args}' (valid: r0..r3)`);
    }
    return parseNum(args, ln, mnemonic);
  }
  if (FLOAT_OPS.has(mnemonic)) {
    if (!args) throw err(ln, `${mnemonic} needs a numeric operand`);
    return parseNum(args, ln, mnemonic);
  }
  if (ANGLE_OPS.has(mnemonic)) {
    if (!args) throw err(ln, `${mnemonic} needs an angle`);
    return parseAngle(args, ln);
  }
  if (mnemonic === "CALL") {
    if (!args) throw err(ln, "CALL needs a chain index or name");
    let cid;
    if (args in chainIds) cid = chainIds[args];
    else {
      cid = Number(args);
      if (!Number.isInteger(cid)) throw err(ln, `CALL: unknown chain '${args}'`);
    }
    if (cid < 0 || cid >= nChains) throw err(ln, `CALL: chain ${cid} out of range (0..${nChains - 1})`);
    return cid + 0.5;
  }
  if (mnemonic in DEFAULT_N) {
    const n = args ? parseNum(args, ln, mnemonic) : DEFAULT_N[mnemonic];
    if (mnemonic === "EXEC" && n < 2.0) {
      throw err(ln, `EXEC magnitude must be >= 2 (got ${n}); below 2 it is a plain LOAD`);
    }
    return n;
  }
  throw err(ln, `unknown mnemonic '${mnemonic}'`);
}

class ChainBuilder {
  constructor(id, comment, dirs) {
    this.id = id;
    this.comment = comment;
    this.dirs = dirs;
    this.edges = { 0: [] };
    this.nextVid = 1;
    this.cursor = 0;
    this.labels = {};
    this.pending = []; // [edge, labelName]
    this.Fa = [1, 0, 0, 0];
  }
  newVertex() {
    const v = this.nextVid++;
    this.edges[v] = [];
    return v;
  }
  requireCursor(ln) {
    if (this.cursor === null) throw err(ln, `chain ${this.id}: no cursor after branch; use a label`);
    return this.cursor;
  }
  label(name, ln) {
    if (name in this.labels) throw err(ln, `duplicate label '${name}'`);
    this.labels[name] = this.requireCursor(ln);
  }
  at(name) {
    if (!(name in this.labels)) this.labels[name] = this.newVertex();
    this.cursor = this.labels[name];
  }
  resolve(target) {
    if (target == null) return this.newVertex();
    if (target in this.labels) return this.labels[target];
    return null;
  }
  opVec(mnemonic, n, ln) {
    const d = this.dirs[MNEMONIC_TO_TRIPLE[mnemonic]?.join(",")];
    if (!d) throw err(ln, `unknown mnemonic '${mnemonic}'`);
    if (n <= 1e-9) throw err(ln, `${mnemonic}: operand magnitude must be positive, got ${n}`);
    return vscale(d, n);
  }
  emit(localVec, to, ln, raw = false) {
    const frm = this.requireCursor(ln);
    const seg = raw ? localVec.slice() : qrot(this.Fa, localVec);
    const edge = { seg, to: -1 };
    const resolved = this.resolve(to);
    if (resolved === null) this.pending.push([edge, to]);
    else edge.to = resolved;
    this.edges[frm].push(edge);
    this.cursor = resolved;
  }
  branch(arms, ln) {
    const frm = this.requireCursor(ln);
    for (const a of arms) {
      const edge = { seg: qrot(this.Fa, a.localSeg), guard: a.guard, to: -1 };
      const resolved = this.resolve(a.to);
      if (resolved === null) this.pending.push([edge, a.to]);
      else edge.to = resolved;
      this.edges[frm].push(edge);
    }
    this.cursor = null;
  }
  rot(mnemonic, angle, ln) {
    const a = ((angle % (2 * Math.PI)) + 2 * Math.PI) % (2 * Math.PI);
    if (a < 1e-9) throw err(ln, `${mnemonic}: rotation angle is zero mod 2*pi`);
    this.emit(this.opVec(mnemonic, a, ln), null, ln);
    this.Fa = qnormalize(qmul(this.Fa, axisAngle(ROT_AXES[mnemonic], a)));
  }
  finish() {
    for (const [edge, name] of this.pending) {
      if (!(name in this.labels)) this.labels[name] = this.newVertex();
      edge.to = this.labels[name];
    }
    this.pending = [];
    const vertices = {};
    for (const vid of Object.keys(this.edges)) vertices[vid] = this.edges[vid];
    return { id: this.id, vertices, comment: this.comment };
  }
}

function prescan(lines) {
  let name = "", decoder = "cubic26", nChains = 0, sawCode = false;
  const chainIds = {};
  lines.forEach((raw, i) => {
    const ln = i + 1;
    const line = strip(raw);
    if (!line) return;
    if (line.startsWith(".")) {
      const sp = line.indexOf(" ");
      const directive = (sp === -1 ? line : line.slice(0, sp)).toLowerCase();
      const rest = sp === -1 ? "" : line.slice(sp + 1).trim();
      if (directive === ".name") name = rest;
      else if (directive === ".decoder") decoder = rest;
      else if (directive === ".chain") {
        const cname = rest.split(/\s+/)[0] || "";
        if (cname) {
          if (cname in chainIds) throw err(ln, `duplicate chain name '${cname}'`);
          chainIds[cname] = nChains;
        }
        nChains++;
      } else throw err(ln, `unknown directive '${directive}'`);
    } else {
      if (nChains === 0 && !sawCode) nChains = 1;
      sawCode = true;
    }
  });
  if (nChains === 0) throw new AsmError("no instructions found");
  return { name, decoder, chainIds, nChains };
}

export function assemble(text) {
  const lines = text.split("\n");
  const { name, decoder, chainIds, nChains } = prescan(lines);
  if (!(decoder in DECODERS)) throw new AsmError(`unknown decoder '${decoder}'`);
  const dirs = DECODERS[decoder]().directions();
  const chains = [];
  let c = null;

  lines.forEach((raw, i) => {
    const ln = i + 1;
    let line = strip(raw);
    if (!line) return;

    if (line.startsWith(".")) {
      const sp = line.indexOf(" ");
      const directive = (sp === -1 ? line : line.slice(0, sp)).toLowerCase();
      if (directive === ".chain") {
        c = new ChainBuilder(chains.length, sp === -1 ? "" : line.slice(sp + 1).trim(), dirs);
        chains.push(c);
      }
      return;
    }
    if (c === null) {
      c = new ChainBuilder(0, "entry", dirs);
      chains.push(c);
    }

    const lm = LABEL_RE.exec(line);
    if (lm) {
      if (c.cursor === null) c.at(lm[1]);
      else c.label(lm[1], ln);
      line = lm[2].trim();
      if (!line) return;
    }

    const sp = line.search(/\s/);
    const mnemonic = (sp === -1 ? line : line.slice(0, sp)).toUpperCase();
    let args = sp === -1 ? "" : line.slice(sp + 1).trim();

    if (mnemonic === "BR") {
      const arms = splitTop(args).map((spec) => {
        const ai = spec.lastIndexOf("->");
        if (ai === -1) throw err(ln, `branch arm needs '-> target': '${spec}'`);
        const target = spec.slice(ai + 2).trim();
        const left = spec.slice(0, ai);
        const ci = left.indexOf(":");
        if (ci === -1) throw err(ln, `branch arm needs 'guard: OP': '${spec}'`);
        const guard = parseGuard(left.slice(0, ci), ln);
        const g = vscale(guard, 1 / vnorm(guard));
        const opParts = left.slice(ci + 1).trim().split(/\s+/);
        const mn = opParts[0].toUpperCase();
        const n = operand(mn, opParts.slice(1).join(" "), ln, chainIds, nChains);
        const real = PSEUDO[mn] || mn;
        return { localSeg: c.opVec(real, n, ln), guard: g, to: target };
      });
      if (arms.length < 2) throw err(ln, "branch needs 2+ arms");
      c.branch(arms, ln);
      return;
    }

    let to = null;
    const ai = args.lastIndexOf("->");
    if (ai !== -1) {
      to = args.slice(ai + 2).trim();
      args = args.slice(0, ai).trim();
    }

    if (mnemonic === "SEG" || mnemonic === "SEGRAW") {
      const vec = parseVec(args, ln, mnemonic);
      c.emit(vec, to, ln, mnemonic === "SEGRAW");
      return;
    }
    if (mnemonic === "OP") {
      const parts = args.split(/\s+/);
      if (parts.length !== 2) throw err(ln, "OP needs: OP MNEMONIC magnitude");
      c.emit(c.opVec(parts[0].toUpperCase(), parseNum(parts[1], ln, "OP"), ln), to, ln);
      return;
    }

    const n = operand(mnemonic, args, ln, chainIds, nChains);
    const real = PSEUDO[mnemonic] || mnemonic;
    if (ANGLE_OPS.has(real)) {
      if (to !== null) throw err(ln, `${real}: '->' jump not supported on rotations`);
      c.rot(real, n, ln);
    } else {
      c.emit(c.opVec(real, n, ln), to, ln);
    }
  });

  return {
    name, decoder,
    chains: chains.map((cb) => cb.finish()),
  };
}
