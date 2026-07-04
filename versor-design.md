# VERSOR

### *A language where the program is the path.*

**Versor** (n.) — in geometric algebra, a unit multivector that enacts a rotation. In this language: the machine's orientation frame, the thing that turns raw vectors into meaning.

Every instruction is a vector. Every program is a polyline through 3D space. Direction is the opcode, magnitude is the operand, and the frame decides what everything means.

---

> **Status:** v0.1 draft spec, ready for implementation.
> **Goal for the implementing agent (Claude Code):** build a working interpreter, serialization format, and execution visualizer in Python. Milestones are at the bottom. Everything above them is normative spec unless marked *open question*.

---

## 1. Vision

A programming language whose syntax and semantics are **geometry in 3D space**:

- Every **instruction is a vector**: its *direction* (quantized against the current frame) selects the opcode, and its *magnitude* is the immediate operand.
- A **program is a polyline**: instructions chained tip-to-tail. The instruction pointer walks the path segment by segment.
- The machine carries an **orientation frame** (a unit quaternion). Instructions are decoded *relative to the frame*, not global axes. Rotating the frame reinterprets all downstream instructions.
- A **function is a stored polyline shape**; calling it stamps the shape into space composed with the caller's frame. The call-site orientation and scale *are* the arguments.
- A function's **return value is its net displacement** (endpoint − start point). Path = implementation; endpoint = specification. Two programs with equal net displacement are extensionally equal.
- **Memory is space itself**: values are stored at spatial cells; the machine's position is the pointer.

### Non-goals for v0.1
- Performance. Interpret naively.
- Human ergonomics of authoring (no VR editor yet — programs are authored in the text serialization or generated programmatically).
- Formal Turing-completeness proof (design for it, prove later).
- Floating-point robustness at cone boundaries beyond the dead-zone rule in §3.3.

---

## 2. Terminology

| Term | Meaning |
|---|---|
| **Segment** | One instruction: a 3-vector `v ∈ ℝ³`. |
| **Chain** | An ordered list of segments forming a polyline = a program or function body. |
| **Vertex** | A point where segments meet. Branch vertices have >1 outgoing edge. |
| **Frame `F`** | Unit quaternion; the machine's current orientation. |
| **Position `P`** | Machine's current point in ℝ³; also the memory pointer. |
| **Net displacement** | Sum of executed segment vectors from chain entry to exit. |
| **Cone** | Angular region on the unit sphere mapped to one opcode. |
| **Cell** | Quantized spatial location used as a memory address. |

---

## 3. Machine model

### 3.1 State
```
P  : vec3        # position (also the memory pointer)
F  : quaternion  # orientation frame, unit norm, starts as identity
R  : vec3[4]     # registers R0..R3
A  : vec3        # accumulator (implicit destination of most data ops)
M  : dict[cell -> vec3]   # spatial memory, sparse, default (0,0,0)
CS : list[(chain_id, seg_index, F, P)]   # call stack
OUT: output buffer (list of scalars/chars)
```

### 3.2 Execution loop
```
1. Fetch next segment v_raw of the current chain.
2. Decode in the frame:  v_local = F⁻¹ · v_raw · F   (rotate v_raw into frame coords)
3. opcode  = quantize_direction(normalize(v_local))    # §3.3
   operand = |v_raw|                                    # magnitude, frame-independent
4. Execute opcode (§4). Most opcodes also advance:  P += v_raw
5. At a branch vertex, evaluate the guard (§5) to pick the outgoing edge.
6. Halt when the chain ends with no outgoing edge, or on HALT.
```
**Note:** step 4 — every instruction moves the machine along itself unless the opcode says otherwise. Movement is not optional; it is what makes memory addressing (§6) work.

### 3.3 Direction quantization (v0.1: cubic-26)
Use the 26 directions of the cube neighborhood — 6 face, 12 edge, 8 corner — because decode is trivial and hand-authoring is sane. (Icosahedral quantization is a v0.2 upgrade; keep the decoder pluggable.)

Decode: normalize `v_local`, then snap each component to {-1, 0, +1} using threshold `t = 0.35`:
```
s_i = +1 if v_i > t;  -1 if v_i < -t;  0 otherwise
```
The resulting sign triple (excluding (0,0,0)) indexes the opcode table.

**Dead-zone rule:** if `|v_i - t| < 0.05` for any component (vector sits near a cone boundary), decoding is an error → runtime fault `AmbiguousDirection`. Programs must commit to their cones. This kills boundary chaos at the cost of forbidding a thin shell of directions.

---

## 4. Opcode table v0.1

Sign triples are in **frame-local** coordinates. `n = operand` (the magnitude). Angles below are in radians. Where an op needs a register index, derive it from the operand: `idx = floor(n) mod 4`.

### Face directions (the 6 workhorses)
| Dir | Mnemonic | Effect |
|---|---|---|
| (+1, 0, 0) | `LOADI` | `A = (n, 0, 0)` in frame coords (i.e., `A = F·(n,0,0)·F⁻¹`) |
| (−1, 0, 0) | `STORE` | `M[cell(P)] = A` |
| (0, +1, 0) | `LOAD` | `A = M[cell(P)]` |
| (0, −1, 0) | `MOVR` | `R[idx] = A` |
| (0, 0, +1) | `MOVA` | `A = R[idx]` |
| (0, 0, −1) | `HALT` | halt; net displacement of the root chain is the program's result |

### Edge directions (arithmetic & geometry, 12)
| Dir | Mnemonic | Effect |
|---|---|---|
| (+1, +1, 0) | `ADD`  | `A = A + R[idx]` |
| (+1, −1, 0) | `SUB`  | `A = A − R[idx]` |
| (−1, +1, 0) | `SCALE`| `A = A * n` |
| (−1, −1, 0) | `DOT`  | `A = (A·R[idx], 0, 0)` (scalar in x-slot) |
| (+1, 0, +1) | `CROSS`| `A = A × R[idx]` |
| (+1, 0, −1) | `NORM` | `A = A / |A|` (fault on zero) |
| (−1, 0, +1) | `PROJ` | `A = proj of A onto R[idx]` |
| (−1, 0, −1) | `REJ`  | `A = A − proj(A, R[idx])` |
| (0, +1, +1) | `ROTF` | rotate frame: `F = F · axis_angle(x̂_local, n)` — see §4.1 |
| (0, +1, −1) | `ROTG` | rotate frame about ŷ_local by n |
| (0, −1, +1) | `ROTH` | rotate frame about ẑ_local by n |
| (0, −1, −1) | `OUT`  | append `A.x` to OUT (if `n ≥ 2`, emit as char `chr(round(A.x))`) |

### Corner directions (control & calls, 8)
| Dir | Mnemonic | Effect |
|---|---|---|
| (+1, +1, +1) | `CALL` | call chain `id = floor(n) mod chain_count` — see §7 |
| (+1, +1, −1) | `RET`  | return; write callee net displacement into `A` of caller |
| (+1, −1, +1) | `JMPZ` | if `|A| < ε` skip the next segment (still move along it) |
| (+1, −1, −1) | `JMPP` | if `A.x > 0` skip the next segment |
| (−1, +1, +1) | `PUSHF`| push `(F, P)` onto an aux stack |
| (−1, +1, −1) | `POPF` | pop and restore `(F, P)` (frame restore, *not* position — see open Q3) |
| (−1, −1, +1) | `NOP`  | move only |
| (−1, −1, −1) | `FAULT`| deliberate crash with message = operand |

`ε = 1e-6` for zero tests.

### 4.1 Frame rotations
`ROTF/ROTG/ROTH` rotate the frame about the frame's *own current* local axes (intrinsic rotations). Because SO(3) is non-commutative, `ROTF·ROTG ≠ ROTG·ROTF` — this is deliberate and is the expressive core of the language. After a frame rotation, the *same raw vectors* downstream decode to different opcodes.

---

## 5. Control flow

A chain is stored as a directed graph of vertices and segments (most vertices have exactly one outgoing segment → a plain polyline).

**Branch vertices** have 2+ outgoing segments, each tagged with a **guard normal** `g` (a unit vec3, stored in the program file, in frame-local coords). The machine takes the outgoing edge with the maximum `dot(F·g·F⁻¹, A_normalized)`; ties → first listed. Convention: a two-way branch uses `g` and `−g`, so the test is just the sign of one dot product.

**Loops** are closed cycles in the graph. Termination happens when a branch guard inside the cycle eventually routes execution out (typically because `A` shrinks/rotates each iteration). There is no special loop construct — a loop is literally a cycle.

**Halting:** `HALT` opcode, running off the end of the root chain, or faults (`AmbiguousDirection`, division by zero magnitude, call-stack overflow at depth 1024, step budget exhaustion at 1e6 steps — all configurable).

---

## 6. Memory model — space is the address bus

`cell(P) = (floor(P.x / c), floor(P.y / c), floor(P.z / c))` with cell size `c = 1.0`.

- `STORE` writes `A` to the cell the machine is standing in; `LOAD` reads it.
- Walking is pointer arithmetic. A jump in position is a pointer jump.
- Code and data share the space. v0.1 keeps them non-interacting (executing a cell's stored vector is **not** supported yet), but the layout should not preclude it — see open Q4 (self-modifying programs).

---

## 7. Functions

A program file contains multiple **chains**. Chain 0 is the entry point. Others are callable.

**CALL semantics:**
1. Push `(current chain, seg index, F, P)` onto CS.
2. Execution jumps to segment 0 of the callee. The callee's segments are raw vectors defined in *its own local space*; they are executed under the **caller's current frame** — i.e., decode uses the live `F`, so orientation at the call site parameterizes the callee's behavior. The caller's frame *is* the argument.
3. Optionally the caller pre-scales: there is no separate scale argument; magnitudes are absolute in v0.1 (*open Q2: frame scale as a second argument channel*).

**RET semantics:** compute `d = P_now − P_at_call`. Pop CS, restore chain/segment (P is NOT restored — the callee physically moved the machine; this is intentional). Set caller's `A = d`. The return value is the net displacement.

**Consequence to preserve in the implementation:** a pure function is one whose net displacement is invariant to where it's called from (translation invariance is automatic; frame-dependence is the argument mechanism).

---

## 8. Serialization format (`.vsr`)

JSON, one file per program:
```json
{
  "version": "0.1",
  "name": "countdown",
  "chains": [
    {
      "id": 0,
      "comment": "entry",
      "vertices": [
        {"id": 0, "out": [{"seg": [3.0, 0.0, 0.0], "to": 1}]},
        {"id": 1, "out": [
            {"seg": [1.0, 1.0, 0.0], "to": 2, "guard": [0.0, 1.0, 0.0]},
            {"seg": [0.0, 0.0, -1.0], "to": 3, "guard": [0.0, -1.0, 0.0]}
        ]},
        {"id": 2, "out": [{"seg": [-1.4, 1.4, 0.0], "to": 1}]},
        {"id": 3, "out": []}
      ]
    }
  ]
}
```
- `seg` is the raw ℝ³ vector. `guard` present only on branch edges.
- Provide a tiny Python builder API too (`Chain().loadi(3).branch(...)`) so tests don't hand-write JSON.

---

## 9. Implementation plan

Language: **Python 3.11+**. Dependencies: `numpy`, `matplotlib` (3D trace rendering); optional `pyquaternion` or hand-rolled quaternions (prefer hand-rolled, ~40 lines, no dependency).

```
versor/
  quat.py        # quaternion: mul, conj, rotate_vec, axis_angle, normalize
  decode.py      # quantize_direction (pluggable: cubic26 now, icosa later), dead-zone check
  isa.py         # opcode table: dict[sign_triple -> handler]; handlers are pure functions on MachineState
  machine.py     # MachineState dataclass + step() + run(budget)
  loader.py      # .vsr JSON -> chain graph; validation (dead-zone lint at load time!)
  builder.py     # fluent Python API for constructing programs
  trace.py       # execution trace recorder: list of (P, F, opcode, A) per step
  viz.py         # matplotlib 3D: draw chain geometry + executed path colored by opcode class
  cli.py         # `python -m versor run prog.vsr [--trace out.png] [--steps N]`
tests/
  test_quat.py test_decode.py test_isa.py test_programs.py
examples/
  countdown.vsr  add_two.vsr  hello.vsr  fib.vsr
```

**Load-time lint (important):** when loading a program, decode every segment under the *identity* frame and warn (not fault) if any segment is within the dead zone — but note this can't catch frame-dependent ambiguity, so the runtime check in §3.3 stays.

**Trace/viz requirements:** color segments by opcode class (data=teal, arithmetic=coral, frame=purple, control=gray), draw the net-displacement dashed line, mark branch vertices, animate optionally (`FuncAnimation`). This is the language's debugger; treat it as a first-class deliverable.

---

## 10. Milestones & acceptance tests

**M0 — Quaternions & decode.** Unit tests: rotation composition non-commutativity; decode of all 26 canonical directions; dead-zone faults.

**M1 — Straight-line programs.** Run: `LOADI 5 → SCALE 2 → OUT` prints `10.0`. Run under a pre-rotated frame and confirm identical output (frame covariance of a chain rotated together with its frame).

**M2 — Branch & loop.** `countdown.vsr`: loads N into `A`, loops subtracting 1 (via `SUB` with `R0=(1,0,0)`), `OUT`s each value, exits the cycle when `A.x ≤ 0` via guard. Expect `N, N-1, ..., 1` in OUT and the trace showing a literal closed cycle.

**M3 — Functions.** `add_two.vsr`: chain 1 computes a fixed displacement; chain 0 CALLs it twice under two different frame rotations and OUTs both return displacements — outputs must differ, demonstrating orientation-as-argument.

**M4 — Memory.** Program stores a value in a cell, walks away, walks back (different route), LOADs it — proving position-addressed memory and path-independence of addressing.

**M5 — Showpieces.** `hello.vsr` (OUT in char mode) and `fib.vsr` (iterative Fibonacci using two registers + a loop). Plus `viz.py` renders of each for the README.

**Stretch — M6:** icosahedral decoder; program interpolation demo (lerp all segments between two M2-equivalent programs, plot which fraction of interpolants still halt/decode).

---

## 11. Open questions (do not block v0.1)

1. **Turing-completeness.** Position-addressed memory over an infinite lattice + branching should suffice (sketch: encode a Minsky machine with two registers as distances along two axes). Write the reduction after M4.
2. **Scale as an argument channel.** Should the frame carry a scale factor multiplying operand magnitudes inside callees? Powerful, but breaks "magnitude = absolute operand." Decide at M3.
3. **POPF position restore.** Restoring `P` teleports the machine, which breaks the "return value = physical displacement" story. Current spec restores frame only. Revisit if function composition feels wrong.
4. **Executable memory.** Letting the machine execute stored vectors when the path crosses a written cell = true self-modifying code. Deferred; keep `M` values as raw vec3 so it stays possible.
5. **Effect regions & homotopy.** Long-term: define IO/memory regions in space so effect-equivalence = path homotopy in the complement. Research-flavored; do not implement in v0.1.
