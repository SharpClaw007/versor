import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

import { AsmError, assemble } from "./asm.js";
import { EXAMPLE_INPUTS, EXAMPLES } from "./examples.js";
import { Machine, VersorFault, qrot } from "./versor.js";

const CLASS_COLORS = {
  data: 0x14b8a6, arithmetic: 0xff7f50, frame: 0xa855f7, control: 0x6b7280,
};
const TRIAD_COLORS = [0xef4444, 0x22c55e, 0x3b82f6];

const $ = (id) => document.getElementById(id);

// ---------- three.js scene ----------
const view = $("view");
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x14141c);
const camera = new THREE.PerspectiveCamera(50, 1, 0.01, 5000);
const renderer = new THREE.WebGLRenderer({ antialias: true });
view.appendChild(renderer.domElement);
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;

const grid = new THREE.GridHelper(40, 40, 0x30304a, 0x22222f);
scene.add(grid);
scene.add(new THREE.AxesHelper(2));

function resize() {
  const w = view.clientWidth, h = view.clientHeight;
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
  renderer.setSize(w, h);
  renderer.setPixelRatio(window.devicePixelRatio);
}
window.addEventListener("resize", resize);

const cursor = new THREE.Mesh(
  new THREE.SphereGeometry(0.16, 24, 16),
  new THREE.MeshBasicMaterial({ color: 0xffffff }),
);
scene.add(cursor);

const startMark = new THREE.Mesh(
  new THREE.SphereGeometry(0.14, 16, 12),
  new THREE.MeshBasicMaterial({ color: 0x16a34a }),
);
scene.add(startMark);

const triad = TRIAD_COLORS.map((c) => {
  const g = new THREE.BufferGeometry().setFromPoints(
    [new THREE.Vector3(), new THREE.Vector3()]);
  const line = new THREE.Line(g,
    new THREE.LineBasicMaterial({ color: c, linewidth: 2 }));
  scene.add(line);
  return line;
});

let pathGroup = new THREE.Group();
scene.add(pathGroup);

// ---------- run state ----------
let records = [];
let outBuf = [];
let cursorIdx = 0;        // how many records are shown
let playing = false;
let lastTick = 0;
let runError = "";

function setStatus(text, err = false) {
  const el = $("status");
  el.textContent = text;
  el.className = err ? "err" : "";
}

function fmtOut(buf) {
  return buf.map((o) => (typeof o === "string"
    ? o : (Math.round(o * 1e6) / 1e6).toString() + " ")).join("");
}

function compileAndRun() {
  records = [];
  outBuf = [];
  runError = "";
  let prog;
  try {
    prog = assemble($("editor").value);
  } catch (e) {
    if (e instanceof AsmError) { setStatus(e.message, true); return false; }
    throw e;
  }
  // assemble() emits vertices keyed by id already; adapt to Machine's shape
  const chains = [];
  for (const c of prog.chains) chains[c.id] = c;
  const raw = $("stdin").value.trim();
  const tokens = raw ? raw.split(",").map((t) => t.trim()) : [];
  const input = tokens.length && tokens.every((t) => t !== "" && Number.isFinite(Number(t)))
    ? tokens.map(Number)
    : raw;  // non-numeric input feeds INP as char codes
  const m = new Machine({ ...prog, chains },
    { trace: true, stepBudget: 60000, input });
  try {
    m.run();
  } catch (e) {
    if (e instanceof VersorFault) runError = e.message;
    else throw e;
  }
  records = m.trace || [];
  outBuf = m.OUT;
  buildPath();
  setStatus(runError ||
    `${m.haltReason} after ${m.steps} steps — decoder ${prog.decoder}`,
    Boolean(runError));
  return true;
}

function buildPath() {
  scene.remove(pathGroup);
  pathGroup = new THREE.Group();
  scene.add(pathGroup);

  const box = new THREE.Box3();
  box.expandByPoint(new THREE.Vector3(0, 0, 0));
  for (const r of records) {
    box.expandByPoint(new THREE.Vector3(...r.P0));
    box.expandByPoint(new THREE.Vector3(...r.P1));
    const zero = r.P0.every((c, i) => Math.abs(c - r.P1[i]) < 1e-12);
    if (zero) { r._line = null; continue; }
    const g = new THREE.BufferGeometry().setFromPoints(
      [new THREE.Vector3(...r.P0), new THREE.Vector3(...r.P1)]);
    const mat = r.skipped
      ? new THREE.LineDashedMaterial({ color: CLASS_COLORS[r.klass], dashSize: 0.15, gapSize: 0.1 })
      : new THREE.LineBasicMaterial({ color: CLASS_COLORS[r.klass] });
    const line = new THREE.Line(g, mat);
    if (r.skipped) line.computeLineDistances();
    line.visible = false;
    r._line = line;
    pathGroup.add(line);
    if (r.branch) {
      const d = new THREE.Mesh(
        new THREE.OctahedronGeometry(0.12),
        new THREE.MeshBasicMaterial({ color: 0xd8d8e8 }));
      d.position.set(...r.P0);
      d.visible = false;
      r._mark = d;
      pathGroup.add(d);
    }
  }

  const size = box.getSize(new THREE.Vector3());
  const center = box.getCenter(new THREE.Vector3());
  const span = Math.max(size.x, size.y, size.z, 4);
  camera.position.set(center.x + span * 0.9, center.y + span * 0.7,
    center.z + span * 0.9);
  controls.target.copy(center);
  camera.near = span / 1000;
  camera.far = span * 50;
  camera.updateProjectionMatrix();

  startMark.position.set(0, 0, 0);
  const s = span / 40;
  cursor.scale.setScalar(s / 0.16 * 0.16 || 1);
  setCursor(0, span);
  cursorIdx = 0;
}

function setCursor(k, spanHint) {
  cursorIdx = Math.max(0, Math.min(k, records.length));
  for (let i = 0; i < records.length; i++) {
    const r = records[i];
    if (r._line) r._line.visible = i < cursorIdx;
    if (r._mark) r._mark.visible = i < cursorIdx;
  }
  const r = cursorIdx > 0 ? records[cursorIdx - 1] : null;
  const P = r ? r.P1 : [0, 0, 0];
  const F = r ? r.F : [1, 0, 0, 0];
  cursor.position.set(...P);
  const span = spanHint ||
    controls.target.distanceTo(camera.position) || 10;
  const len = Math.max(span * 0.06, 0.4);
  triad.forEach((line, i) => {
    const axis = [[1, 0, 0], [0, 1, 0], [0, 0, 1]][i];
    const tip = qrot(F, axis).map((c, j) => P[j] + c * len);
    line.geometry.setFromPoints(
      [new THREE.Vector3(...P), new THREE.Vector3(...tip)]);
  });
  $("hud").textContent = r
    ? `step ${r.step}/${records.length}  chain ${r.chain}  ${r.opcode}  n=${r.n.toFixed(3)}${r.skipped ? " (skipped)" : ""}\n`
      + `A = (${r.A.map((c) => c.toFixed(2)).join(", ")})\n`
      + `P = (${P.map((c) => c.toFixed(2)).join(", ")})`
    : "step 0 — press Run";
  $("out").textContent = fmtOut(outBuf.slice(0, r ? r.outLen : 0));
}

// ---------- controls ----------
const exampleSel = $("example");
for (const name of Object.keys(EXAMPLES)) {
  const opt = document.createElement("option");
  opt.value = name;
  opt.textContent = name;
  exampleSel.appendChild(opt);
}
exampleSel.addEventListener("change", () => {
  $("editor").value = EXAMPLES[exampleSel.value];
  $("stdin").value = EXAMPLE_INPUTS[exampleSel.value] || "";
  playing = false;
  compileAndRun();
});

$("run").addEventListener("click", () => {
  if (compileAndRun()) { setCursor(records.length); playing = false; }
});
$("step").addEventListener("click", () => {
  if (!records.length) compileAndRun();
  playing = false;
  setCursor(cursorIdx + 1);
});
$("play").addEventListener("click", () => {
  if (!records.length) compileAndRun();
  if (cursorIdx >= records.length) setCursor(0);
  playing = !playing;
  $("play").textContent = playing ? "Pause" : "Play";
});
$("reset").addEventListener("click", () => {
  playing = false;
  $("play").textContent = "Play";
  setCursor(0);
});
$("speed").addEventListener("input", () => {
  $("speedval").textContent = `${$("speed").value}/s`;
});

$("editor").value = EXAMPLES.countdown;
exampleSel.value = "countdown";
resize();
compileAndRun();

// ---------- animation loop ----------
function tick(t) {
  requestAnimationFrame(tick);
  if (playing && records.length) {
    const interval = 1000 / Number($("speed").value);
    if (t - lastTick >= interval) {
      lastTick = t;
      if (cursorIdx < records.length) setCursor(cursorIdx + 1);
      else { playing = false; $("play").textContent = "Play"; }
    }
  }
  controls.update();
  renderer.render(scene, camera);
}
requestAnimationFrame(tick);
