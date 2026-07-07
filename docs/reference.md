# Versor language reference

One page: everything you need to write Versor. Formal treatment in the
[whitepaper](whitepaper.md); original spec in
[`versor-design.md`](../versor-design.md).

## The machine

| State | Meaning |
|---|---|
| `P ∈ ℝ³` | Position — also the memory pointer. |
| `F` (unit quaternion) | Orientation frame; instructions decode relative to it. |
| `s > 0` | Sim(3) scale; movement is `P += s·v`. Starts 1. |
| `A ∈ ℝ³` | Accumulator. |
| `R0..R3 ∈ ℝ³` | Registers. |
| `M : ℤ³ → ℝ³` | Memory, one vector per unit cell; `cell(P) = ⌊P⌋`. |
| call stack | `(chain, vertex, F, P, s)` per CALL; depth ≤ 1024. |
| aux stack | `(F, P, s)` per PUSHF. |
| data stack | vectors, PUSHA/POPA (Versor-32). |
| OUT / IN | Output buffer; input buffer for INP (Versor-32). |

**Execution step:** pick the next segment `v` (at a branch, by guards),
decode `F⁻¹vF`'s direction to an opcode and read `|v|` as the operand `n`,
move `P += s·v`, then run the opcode. *Every instruction moves.* Halting:
`HALT`, running off the root chain, or a fault. Step budget defaults to 10⁶.

**Scalars** live in the frame-local x slot: LOADI writes it, OUT/JMPP/DOT
read or write it as `(F⁻¹AF).x`.

**Branches:** each arm has a frame-local guard normal; the machine takes the
arm maximizing `dot(F·g·F⁻¹, Â)`. First-listed wins ties; `|A| < ε` makes
every dot zero, so the zero accumulator takes the first arm — list the exit
arm first.

**Functions:** CALL runs another chain under the caller's live frame
(orientation is the argument) and scale (frac of the CALL magnitude, factor
`2^(2f−1)` in `[0.5, 2)`). RET (or chain end) returns the callee's net
displacement in A and restores F and s — never P.

**Register indices** encode as magnitude: `idx = ⌊n⌋ mod 4`. Chain ids in
CALL the same way. Operands are magnitudes, hence strictly positive; there
are no negative immediates (use SUB).

## Opcodes

Directions are frame-local sign triples (cubic26 cone centers); the four
decoders place their cones differently but keep these names.

| Dir | Op | Effect |
|---|---|---|
| (+1, 0, 0) | `LOADI n` | A = n in the frame-local x slot |
| (−1, 0, 0) | `STORE` | M[cell(P)] = A (arrival cell) |
| (0, +1, 0) | `LOAD` | A = M[cell(P)]; **n ≥ 2 = EXEC**: execute the cell's vector (movement included, chains walk stored code, depth ≤ 64) |
| (0, −1, 0) | `MOVR r` | R[idx] = A |
| (0, 0, +1) | `MOVA r` | A = R[idx] |
| (0, 0, −1) | `HALT` | halt; root net displacement is the result |
| (+1, +1, 0) | `ADD r` | A += R[idx] |
| (+1, −1, 0) | `SUB r` | A −= R[idx] |
| (−1, +1, 0) | `SCALE n` | A ·= n |
| (−1, −1, 0) | `DOT r` | A = A·R[idx] (scalar, x slot) |
| (+1, 0, +1) | `CROSS r` | A = A × R[idx] |
| (+1, 0, −1) | `NORM` | A = A/\|A\| (fault on zero) |
| (−1, 0, +1) | `PROJ r` | A = proj(A, R[idx]) |
| (−1, 0, −1) | `REJ r` | A −= proj(A, R[idx]) |
| (0, +1, +1) | `ROTF θ` | rotate frame about its own x by θ |
| (0, +1, −1) | `ROTG θ` | ... about its own y |
| (0, −1, +1) | `ROTH θ` | ... about its own z |
| (0, −1, −1) | `OUT` | emit frame-local A.x; **n ≥ 2**: as chr(round) |
| (+1, +1, +1) | `CALL c [scale]` | call chain ⌊n⌋; frac(n) scales (Sim(3)) |
| (+1, +1, −1) | `RET` | return displacement; restore F, s (not P) |
| (+1, −1, +1) | `JMPZ` | \|A\| < ε → skip next segment (it still moves) |
| (+1, −1, −1) | `JMPP` | frame-local A.x > ε → skip next |
| (−1, +1, +1) | `PUSHF` | push (F, P, s) |
| (−1, +1, −1) | `POPF` | restore F and s — not P |
| (−1, −1, +1) | `NOP` | move only |
| (−1, −1, −1) | `FAULT n` | deliberate fault |

**Versor-32 extended opcodes** (need `.decoder icosa32` or `sphere32`;
they live on the six extra icosahedral cones, keyed by name):

| Op | Effect |
|---|---|
| `INP` | next input scalar into the x slot (fault when exhausted) |
| `SWAP r` | A ↔ R[idx] |
| `PUSHA` / `POPA` | data stack push/pop of A |
| `MULR r` | A ·= frame-local x of R[idx] (variable × variable) |
| `LOADP` | A = P (read-only) |

## Decoders

| Name | Cones | Usable sphere | Notes |
|---|---|---|---|
| `cubic26` | 26 | 72.2 % | default; component thresholds at ±0.35, dead band ±0.05 |
| `icosa32` | 32 | 91.3 % | icosahedron vertices + face normals; full Versor-32 ISA |
| `sphere26` | 26 | 92.4 % | optimized base-26 packing (min separation 38.17°) |
| `sphere32` | 32 | 91.3 % | optimized Versor-32 packing (converges to the icosahedral 37.38°) |

Decode faults (`AmbiguousDirection`) rather than guesses near cone walls.
Programs are authored on cone centers — the builder, assembler, and VHL do
this for you, tracking the authoring frame through rotations.

## Assembly (`.vasm`)

```asm
; comment            # also a comment
.name countdown
.decoder cubic26     ; optional

.chain entry         ; first chain = entry point
        LOADI 1
        MOVR r0              ; registers r0..r3, or a raw magnitude
        LOADI 5
loop:   OUT                  ; label: names the vertex before the next op
        SUB r0
        BR -x: HALT -> end, +x: NOP -> loop
.chain fn            ; CALL fn | CALL 1 | CALL fn 0.55 (scaled)
        NOP 2                ; unlabeled branch targets = dead ends
```

- Guards: `+x -x +y -y +z -z` or explicit `(gx,gy,gz)`.
- Angles: floats or `pi`, `pi/2`, `3pi/4`.
- `-> label` on any instruction jumps (cycles without guards).
- `SEG (x,y,z)` / `SEGRAW (x,y,z)` / `OP MNEMONIC n` escape hatches.
- Pseudo-ops: `OUTC` (char OUT), `EXEC` (executing LOAD).

## VHL (`.vhl`)

```
fn fib(n) {                  # functions compile to chains; recursion works
    if n - 2 {               # if/while test: expr > 0
        return fib(n - 1) + fib(n - 2)
    }
    if n { return 1 }        # (block braces per line: `if n {` ... `}`)
    return 0                 # falling off the end returns 0
}
let i = 1
while 10 - i {
    print fib(i)
    let i = i + 1
}
repeat 2 { print 0 }         # repeat: counted-loop sugar, runs ceil(n) times
```

Statements: `let`, `print`, `return`, `repeat/while/if expr { ... }`
(`} else {` on one line). Expressions: `+ − * ( )` with constant folding,
`input()`, and calls; `*` by a constant compiles to SCALE,
variable×variable to MULR. Functions use a data-stack calling convention
(PUSHA/POPA; caller saves live registers), so any of fn/input()/var×var
selects the icosa32 dialect. Three general registers — no spilling.

## Faults

`AmbiguousDirection` · `ZeroLengthSegment` · `DivisionByZero` ·
`CallStackOverflow` · `StackUnderflow` · `StepBudgetExhausted` ·
`ExplicitFault` · `InvalidCharCode` · `ExecEmptyCell` · `ExecDepthExceeded`
· `ScaleOverflow` · `InputExhausted`. All carry step/chain/vertex.

## CLI

```
versor run FILE[.vsr|.vasm|.vhl] [--decoder D] [--input 6,7|--input-text s]
           [--trace out.png] [--animate out.gif] [--steps N]
versor asm FILE.vasm [-o out.vsr]      versor vhl FILE.vhl [-o out.vsr]
versor export FILE --gcode/--obj/--stl  versor lint FILE
```
