# Design: the three deferred extensions

> Status: **planned, not implemented.** Targets spec open questions Q2 and
> Q4 plus the reserved icosa32 cones. Ordering: EXEC (v0.3a) → Sim(3)
> scale (v0.3b) → extended ISA "Versor-32" (v0.4). Each lands with a spec
> addendum here (the original `versor-design.md` stays as written), a
> whitepaper section, JS-port parity + golden regeneration, and README
> updates.

## 1. Sim(3) scale channel (spec Q2) — v0.3b

**Decision: geometric scaling, encoded in CALL's fractional magnitude. No
new opcode; no operand scaling.**

### Semantics

- Machine pose becomes $(P, F, s) \in \mathrm{Sim}(3)$ with scale $s > 0$
  (initial 1.0).
- Movement scales: every instruction advances $P \mathrel{+}= s\,v_{raw}$.
  Decode direction and operand $n = \lVert v \rVert$ are **unchanged** —
  "magnitude = absolute operand" survives.
- `CALL n`: chain id $= \lfloor n \rfloor$ as today; the callee runs at
  $s' = s \cdot 2^{\,2\,\mathrm{frac}(n) - 1}$.
  The builder's existing `idx + 0.5` convention gives frac $= 0.5$ →
  factor 1: **all existing programs behave identically** (goldens must
  stay byte-identical — that is milestone SM1's acceptance test).
  frac → 0 halves, frac → 1 doubles; compose calls for wider range.
- `RET` restores the caller's $s$ (scale is callee-local, exactly like
  $F$); `PUSHF`/`POPF` push/restore $(F, s)$; trace records carry $s$.
- Return displacement of a callee at scale $s$ is $s \cdot d_{unscaled}$:
  scale is observable through geometry and returns, not arithmetic.
- Bounds: $s \in [2^{-32}, 2^{32}]$, else fault `ScaleOverflow`.
- Memory cells stay world-fixed ($\mathbb{Z}^3$ has no scale symmetry);
  the covariance theorem's memory caveat extends verbatim. Sub-unit-scale
  code shares cells — document, don't "fix".

### Rejected alternative

Scaling operands ($n \to s\,n$, the spec's literal phrasing) corrupts the
register-selector channel ($\lfloor s\,n \rfloor \bmod 4$ retargets
registers under scale) and would scale rotation angles, which Sim(3) does
not rotate... scale. If value scaling is ever wanted it requires a
per-opcode scaling-policy table — its own design round.

### Milestones

- **SM1** plumbing + backward compatibility: goldens unchanged; scale
  visible in trace/HUD; viz segment width ∝ $s$.
- **SM2** gauge tests: memory-free program under global similarity —
  outputs invariant, displacement scales.
- **SM3** showpiece `koch.vsr`: self-calling chain with frac < 0.5 draws a
  literal fractal; animation for the README.

Effort ≈ 1–2 days. Risk low (additive, default-invisible).

## 2. Executable memory (spec Q4) — v0.3a, build first

**Decision: `LOAD` with n ≥ 2 executes the cell (magnitude-band overload,
same precedent as OUT's char mode). No new opcode, works in every
decoder.**

### Semantics

- `LOAD`, $n \ge 2$ (**EXEC**): let $w = M[\mathrm{cell}(P)]$ at the
  arrival cell. Decode $w$ under the **live frame** and execute it as a
  full instruction — including its movement $P \mathrel{+}= w$ (or
  $s\,w$ once Sim(3) lands). Empty cell → fault `ExecEmptyCell` (loud;
  not a NOP).
- Because executed vectors move the machine, chained EXECs **walk stored
  code**: cells whose vectors point into the next cell form a program
  laid down in memory. STORE computes it; EXEC runs it — true
  self-modifying code with zero changes to the memory model (the spec
  kept $M$ as raw vec3 for exactly this).
- EXEC-of-EXEC allowed to depth 64 → fault `ExecDepthExceeded`.
  CALL/RET inside an EXEC'd instruction work unchanged (chain position is
  orthogonal state).

### Consequences to document

Behavior stops being piecewise-constant on $\mathbb{R}^{3m}$ alone
(whitepaper §8): with EXEC it is a function of program × memory
trajectory. `interp.py`/`synth.py` treat EXEC-using programs as a
separate class (detect at load: any LOAD-direction segment with
$n \ge 2$, plus a runtime marker in traces).

### Milestones

- **EM1** trampoline: store a LOADI vector, EXEC it, assert A.
- **EM2** stored program: chain writes a countdown into a cell trail,
  EXEC-walks it, asserts OUT.
- **EM3 (stretch)** `TRAP` mode via STORE n ≥ 2: auto-execute on cell
  entry. Only if EM2 leaves appetite.

Effort ≈ 1 day. Smallest, most self-contained — first.

## 3. Extended ISA for the reserved icosa32 cones — v0.4 "Versor-32"

The 6 reserved directions form **3 antipodal pairs** → 3 dual op-pairs,
chosen from pain actually felt writing programs:

| Pair | Ops | Rationale |
|---|---|---|
| 1 | `INP` / `SWAP` | The ISA has OUT but no input (`INP`: scalar into the frame-local slot, char mode at n ≥ 2; needs `Machine(input=...)`, CLI `--input`, playground text field). `SWAP` A↔R[idx] removes the 3-op register dances (see fib). |
| 2 | `PUSHA` / `POPA` | Data stack (aux currently carries frames only). Genuinely dual ops on an antipodal pair. Unlocks data recursion — the call stack carries no data besides displacement. |
| 3 | `MULR` / `LOADP` | `MULR`: A ×= frame-local x of R[idx] — variable×variable in one op (VHL v2 drops its loop idiom). `LOADP`: A = P, read-only position introspection. Its teleport dual is **rejected**: writing P breaks Proposition 6, same reason POPF restores frame only. |

### The real cost: opcode key-space refactor

Opcode keys are sign triples; reserved cones have none. Change the
decoder contract to return an **opcode key** (string), with
`OPCODES: key → handler`. Touches `decode.py`, `isa.py`, `machine.py`,
`builder.py`, `asm.py`, the JS port, and golden regeneration — mechanical,
~half a day, and it should land alone (hence v0.4).

- `cubic26` keeps base-26 forever (no free cones in its geometry).
- `icosa32` gains the 6 extended ops on its reserved cones.
- `sphere26` → regenerate as **`sphere32`** (16 antipodal pairs, rerun
  `tools/optimize_sphere26.py` generalized) so the extended ISA also has
  a max-margin decoder.
- Loader lint warns when extended ops appear in a `cubic26` program;
  runtime fault `UnavailableOpcode`.

### Milestones

- **X1** key-space refactor, zero behavior change (goldens identical).
- **X2** the 6 handlers + input plumbing + tests.
- **X3** `sphere32` regeneration + measurements (calcs.py, whitepaper).
- **X4** VHL v2: `input()` and native `a * b` via MULR.

## Interactions

- EXEC and scale compose: an EXEC'd vector moves $s\,w$ — a stored
  program stamped at scale. Fractal self-modifying code is a legal v0.3
  sentence.
- Neither v0.3 feature introduces opcodes, so the v0.4 refactor doesn't
  rework them.
- Every feature re-runs `tools/make_golden.py` and the parity suite; the
  playground must track semantics in the same commit.
