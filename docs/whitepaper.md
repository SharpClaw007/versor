# The Mathematics of Versor

*A geometric account of a language whose programs are paths.*

**Version 0.1 · July 2026 · companion to [`versor-design.md`](../versor-design.md)
and the reference implementation in [`versor/`](../versor)**

Every numeric claim in this paper is reproduced by
[`docs/calcs.py`](calcs.py); every behavioral claim is pinned by the test
suite. Where the implementation had to resolve an ambiguity in the spec, the
mathematics that forced the resolution is given here.

---

## Contents

1. [The machine is a moving frame](#1-the-machine-is-a-moving-frame)
2. [Quaternions and the rotation action](#2-quaternions-and-the-rotation-action)
3. [Decoding is a partition of the sphere](#3-decoding-is-a-partition-of-the-sphere)
4. [Semantics: the step map and frame covariance](#4-semantics-the-step-map-and-frame-covariance)
5. [Functions: a displacement calculus](#5-functions-a-displacement-calculus)
6. [Memory: the integer lattice quotient](#6-memory-the-integer-lattice-quotient)
7. [Loops are screw motions](#7-loops-are-screw-motions)
8. [Program space and interpolation](#8-program-space-and-interpolation)
9. [Computability sketch](#9-computability-sketch)
10. [Numerical design](#10-numerical-design)
11. [Implementation correspondence](#11-implementation-correspondence)
12. [Open problems](#12-open-problems)

---

## 1. The machine is a moving frame

The core of the Versor machine is a **pose**: a position $P \in \mathbb{R}^3$
together with an orientation, represented by a unit quaternion
$F \in S^3 \subset \mathbb{H}$. Up to the sign ambiguity of §2 this is an
element of the special Euclidean group

$$ (P, F) \;\longleftrightarrow\; g \in SE(3) = \mathbb{R}^3 \rtimes SO(3), $$

i.e. the machine is a rigid body. The remaining state — accumulator
$A \in \mathbb{R}^3$, registers $R_0..R_3 \in \mathbb{R}^3$, sparse memory
$M : \mathbb{Z}^3 \to \mathbb{R}^3$, control position, call stack, output
buffer — rides along on this pose. Three design commitments follow from
taking the pose seriously:

- **Instructions are decoded in body coordinates.** A raw segment
  $v \in \mathbb{R}^3$ means whatever it means *relative to* $F$ (§3–4).
- **Every instruction translates the pose along itself**: $P \mathrel{+}= v$.
  Movement is not an effect some opcodes have; it is the substrate all
  opcodes share (§5 exploits this).
- **The pose is the interface.** A function receives the caller's $F$ as its
  argument and returns a displacement of $P$ (§5).

## 2. Quaternions and the rotation action

$\mathbb{H}$ is the real algebra spanned by $1, i, j, k$ with
$i^2 = j^2 = k^2 = ijk = -1$; the Hamilton product is bilinear and
associative but not commutative. Writing $q = (w, \mathbf{u})$ with scalar
part $w$ and vector part $\mathbf{u} \in \mathbb{R}^3$:

$$ q_1 q_2 = \big(w_1 w_2 - \mathbf{u}_1 \cdot \mathbf{u}_2,\;
   w_1\mathbf{u}_2 + w_2\mathbf{u}_1 + \mathbf{u}_1 \times \mathbf{u}_2\big). $$

Unit quaternions form the 3-sphere $S^3$, a Lie group. It acts on
$\mathbb{R}^3$ (identified with the pure imaginaries) by conjugation:

$$ \rho_q(v) = q\,v\,q^{-1} = q\,v\,\bar q \quad (|q| = 1), $$

and $\rho : S^3 \to SO(3)$ is a surjective 2-to-1 homomorphism with kernel
$\{\pm 1\}$: $q$ and $-q$ enact the same rotation (the implementation's
`Quat.approx` compares rotations, not representatives, for exactly this
reason). The rotation by angle $\theta$ about unit axis $\hat n$ is
$q = (\cos\tfrac\theta2,\ \hat n \sin\tfrac\theta2)$.

**Evaluation.** `quat.py` rotates by the expanded form

$$ \rho_q(v) = v + w\,t + \mathbf{u} \times t, \qquad t = 2\,\mathbf{u} \times v, $$

which is the conjugation product with the scalar-part cancellations already
performed — two cross products and one scale instead of two full Hamilton
products, and no conversion to a matrix.

**Intrinsic composition.** Versor's frame rotations `ROTF/ROTG/ROTH` are
*intrinsic*: they rotate about the frame's own current axes. In quaternion
terms this is **right** multiplication.

> **Lemma 1.** Let $F' = F q_{\hat e,\theta}$ where $\hat e$ is a coordinate
> axis. Then the world-space axis about which the frame turned is
> $\rho_F(\hat e)$ — the frame's *local* $\hat e$ — whereas $F' = q F$ would
> have turned it about the fixed world axis $\hat e$.
>
> *Proof.* $\rho_{F'} = \rho_F \circ \rho_q$, so the new decode of a world
> vector $v$ is $\rho_q^{-1}(\rho_F^{-1}(v))$: the extra rotation is applied
> in body coordinates. ∎

Because $SO(3)$ is non-abelian, $F q_1 q_2 \ne F q_2 q_1$ in general:
reordering two frame rotations changes the meaning of *every* downstream
instruction. This non-commutativity is not an implementation hazard — it is
the language's expressive core, and `test_isa.py::test_rot_non_commutative`
pins it.

## 3. Decoding is a partition of the sphere

A nonzero instruction vector factors uniquely (polar decomposition) as

$$ v = n\,\hat v, \qquad n = \lVert v \rVert \in \mathbb{R}_{>0}, \quad \hat v \in S^2 . $$

The magnitude $n$ is the operand — rotation-invariant, so it can be read
before decoding. The direction is first pulled into body coordinates,
$\hat v_{\text{local}} = \rho_F^{-1}(\hat v)$, then quantized by a **decoder**

$$ Q : S^2 \longrightarrow K \cup \{\bot\}, $$

where $K$ is the opcode key set (sign triples) and $\bot$ is a fault. A
decoder is exactly a partition of $S^2$ into labeled cells plus a forbidden
set; the implementation treats it as a plugin with two methods: the
partition map `decode` and a section `directions` $: K \to S^2$ choosing a
safe interior point of each cell (the *cone center*, which the builder and
assembler use as the authoring target).

Operands are further stratified by the register ops: the index is
$\lfloor n \rfloor \bmod 4$, so the magnitude channel carries both a real
operand and two bits of register selector. Note $n > 0$ always — there are
no negative immediates; sign is reached through `SUB`, a consequence of
using the polar factorization rather than a signed encoding.

### 3.1 Cubic-26

$Q_{c}(\hat v)$ thresholds each component at $t = 0.35$ into
$\{-1, 0, +1\}$, faulting inside the **dead zone**
$\big|\,|\hat v_i| - t\,\big| < 0.05$ for any $i$. The cells are the
intersections of $S^2$ with slabs bounded by the twelve small circles
$\hat v_i = \pm t$; the dead zone is a shell around those circles. Since the
largest component of a unit vector is at least $1/\sqrt3 > 0.35 + 0.05$, the
all-zero triple is unreachable and every non-fault decode is well defined.

Monte Carlo measure of the partition (2,000,000 uniform samples,
`docs/calcs.py`):

| region | cells | fraction of $S^2$ | per cell |
|---|---|---|---|
| face $(\pm1,0,0)\ldots$ | 6 | 17.8 % | 2.96 % |
| edge $(\pm1,\pm1,0)\ldots$ | 12 | 42.1 % | 3.51 % |
| corner $(\pm1,\pm1,\pm1)$ | 8 | 12.4 % | 1.55 % |
| **dead zone** | — | **27.8 %** | — |

Two honest observations. The cells are far from equal-area (edge cells are
2.3× corner cells), and the dead zone consumes over a quarter of the sphere.
That is the price of a decode rule that is trivially computable and — more
importantly — **stable**:

> **Lemma 2 (robustness).** If $Q_c(\hat v) = k \ne \bot$, then every
> $\hat v'$ with $\lVert \hat v' - \hat v \rVert_\infty < 0.05$ satisfies
> $Q_c(\hat v') \in \{k, \bot\}$: a perturbation smaller than the dead-zone
> half-width can *fault* a program but can never *silently retarget* it to a
> different opcode.
>
> *Proof.* Success at $\hat v$ means every component is at distance
> $\ge 0.05$ from $\pm t$, so no component's sign class changes under the
> perturbation. ∎

This is the whole function of the dead zone: it converts the
measure-zero decode discontinuity into a measure-positive, *detected*
failure — a stability margin for both floating-point noise and the program
interpolation of §8.

### 3.2 Icosa-32 and the impossibility of icosahedral 26

The spec's stretch goal asks for an "icosahedral decoder." Taken literally
it cannot exist:

> **Proposition 3.** No set of 26 directions in $S^2$ is invariant under the
> icosahedral rotation group $I$.
>
> *Proof.* An invariant direction set is a union of $I$-orbits. The orbit of
> a point on $S^2$ under $I$ (order 60) has size $60/|\mathrm{Stab}|$, and
> the point stabilizers of the icosahedral action are cyclic of order 5, 3,
> 2, or 1 — giving orbit sizes 12 (vertices), 20 (face centers), 30 (edge
> midpoints), or 60 (generic). No sum of elements of $\{12, 20, 30, 60\}$
> equals 26. ∎

The implemented `icosa32` therefore uses the smallest icosahedral direction
set that can host the ISA: the $12 + 20 = 32$ directions given by the
icosahedron's vertices and face normals, decoded by nearest neighbor
(equivalently, the spherical Voronoi partition), with 26 cones carrying the
existing opcodes and 6 left **reserved** (decoding into one is a fault, held
for a future ISA extension). The assignment uses golden-ratio arithmetic,
$\varphi = \tfrac{1+\sqrt5}{2}$, $\varphi^2 = \varphi + 1$:

- **Corners, exact.** The face normals of the icosahedron are the vertices
  of the dodecahedron, which include all eight $(\pm1,\pm1,\pm1)/\sqrt3$ —
  the cubic corner directions are icosahedral *verbatim*.
- **Edges, 13.28°.** Each cubic edge direction, e.g. $(1,1,0)/\sqrt2$, maps
  to the unique icosahedron vertex with the same sign pattern,
  $(1,\varphi,0)/\sqrt{1+\varphi^2}$, at
  $\cos\theta = \frac{1+\varphi}{\sqrt2\sqrt{1+\varphi^2}} = 0.9732$.
- **Faces, 20.91°.** Using $\varphi^2 + \varphi^{-2} = 3$ (from
  $\varphi^2 = \varphi + 1$ and hence $\varphi^{-2} = (\varphi-1)^2 = 2 - \varphi$),
  the axis-heavy dodecahedron vertices such as
  $(\varphi, \varphi^{-1}, 0)$ have norm exactly $\sqrt3$, and
  $\cos\theta = \varphi/\sqrt3 = 0.9342$ against $+\hat x$. Each axis has
  *two* such candidates, $(\varphi, \pm\varphi^{-1}, 0)$; one is assigned
  (minor-component sign matching the major), its mirror is reserved.

A caution that cost this project a bug: the 20 dodecahedral directions must
be the face normals *of the icosahedron orientation actually used* — the
cyclic family $(0, \varphi, \varphi^{-1})$, since e.g. the face
$\{(1,\varphi,0), (-1,\varphi,0), (0,1,\varphi)\}$ has centroid
$\propto (0, \varphi^2, 1) = \varphi\,(0, \varphi, \varphi^{-1})$. The
mirrored standard family $(0, \varphi^{-1}, \varphi)$ is the dual of a
*rotated* icosahedron; mixing the two orientations produces near-coincident
direction pairs and silently collapses the set's minimum pairwise
separation from 37.38° to 10.81°
(`test_m6.py::test_direction_set_is_a_true_dual_pair` pins the correct
geometry).

> **Proposition 4.** The six cubic face directions lie on exact Voronoi
> boundaries of `icosa32` and are therefore ambiguous under it.
>
> *Proof.* $\hat x \cdot (\varphi,\pm\varphi^{-1},0)/\sqrt3 = \varphi/\sqrt3$
> for both signs: an exact two-way tie, i.e. a boundary point of the
> partition, inside any positive dead margin. ∎

So `cubic26` and `icosa32` are *dialects*: corner- and edge-authored
programs are portable (pinned by
`test_m6.py::test_corner_and_edge_cubic_directions_agree_across_decoders`),
face-authored ones must be re-aimed — which the builder does automatically
from the decoder's `directions` section. The ambiguity rule is a relative
margin on cosine similarity: fault when the best and second-best dot
products are within 0.01. Near a wall between directions separated by
$\theta$ the gap grows at rate $2\sin(\theta/2)$ per radian of motion, so
with the set's minimum separation of 37.38° (the uniform
vertex-to-adjacent-face angle,
$\arccos(\varphi^2 / (\sqrt3\sqrt{1+\varphi^2}))$) the fault shell is about
0.9° on each side of every wall. Measured partition: 73.6 % assigned,
17.7 % reserved, 8.8 % dead — less than a third of cubic-26's forbidden
area, at the cost of six unusable cones.

## 4. Semantics: the step map and frame covariance

A program is a finite family of directed graphs (chains); each edge carries
a raw segment $v$ and, at branch vertices, a frame-local unit guard $g$. The
machine state is

$$ \sigma = (P, F, A, R, M, \kappa), $$

with $\kappa$ the control data (current chain and vertex, call stack, skip
flag). One step: select the outgoing edge (by guard if branching), decode
$(k, n) = \big(Q(\rho_F^{-1}(\hat v)),\ \lVert v\rVert\big)$, advance
$P \mathrel{+}= v$, then apply the opcode's handler
$h_k(\cdot\,; n) : \Sigma \to \Sigma$. Branch selection maximizes
$\rho_F(g) \cdot \hat A$ over the outgoing edges, first-listed winning ties,
with $\hat A := 0$ when $\lVert A \rVert < \varepsilon$ (so the zero
accumulator degenerates to "take the first edge" — countdown's exit
convention).

The central symmetry of the language:

> **Theorem 5 (frame covariance).** Let $q$ be a unit quaternion,
> $R = \rho_q$. Given program $\mathcal P$, let $R\mathcal P$ be the program
> with every segment rotated ($v \mapsto Rv$) and guards unchanged (guards
> are stored frame-locally). Run $\mathcal P$ from
> $(P_0, F_0) = (\mathbf 0, 1)$ and $R\mathcal P$ from $(\mathbf 0, q)$.
> If no `STORE`/`LOAD` executes, then at every step the two executions
> satisfy
> $$ P' = RP, \quad F' = qF, \quad A' = RA, \quad R_i' = R R_i, $$
> take identical branches, execute identical opcode sequences, and produce
> **identical output**; the final displacement is rotated by $R$.
>
> *Proof.* Induction over steps. Decode: with $F' = qF$ and $v' = Rv$,
> $\rho_{F'}^{-1}(v') = \rho_F^{-1}\rho_q^{-1}\rho_q(v) = \rho_F^{-1}(v)$ —
> same key, same magnitude. Branch: $\rho_{F'}(g)\cdot \hat{A'} =
> R\rho_F(g) \cdot R\hat A = \rho_F(g)\cdot\hat A$ by orthogonality of $R$.
> Handlers: `LOADI` sets $A' = \rho_{F'}(n,0,0) = R\,\rho_F(n,0,0)$;
> `ADD/SUB/SCALE/CROSS/NORM/PROJ/REJ` are equivariant vector algebra
> ($R(a \times b) = Ra \times Rb$ for $R \in SO(3)$, dot products invariant);
> `ROT*` right-multiplies, $(qF)q_e = q(Fq_e)$; `OUT`/`JMPP`/`DOT` read or
> write the frame-local slot $\rho_{F}^{-1}(A)_x$, which the induction
> hypothesis makes equal on both sides; `CALL/RET` push and restore
> covariant pairs, and returned displacements rotate with $R$. Every case
> preserves the invariant, and all observable scalars coincide. ∎

Two corollaries worth making explicit:

- **The theorem forced an implementation decision.** If `OUT` read the
  *world* $A_x$ instead of the frame-local slot, the proof breaks at the
  `OUT` case and milestone M1's covariance test fails. Frame-local scalar
  I/O is not a style choice; it is what makes rotation a symmetry of the
  language. (Design decision 1 in the README.)
- **Memory breaks full covariance, deliberately.** `cell(P) = \lfloor P/c \rfloor`
  is defined on a fixed lattice $\mathbb{Z}^3$ that only a finite subgroup
  of $SO(3)$ respects. For $R$ in the orientation-preserving hyperoctahedral
  group (the 24 rotations of the cube), $\mathrm{cell}(RP) = \pi(\mathrm{cell}(P))$
  for a bijection $\pi$ of $\mathbb{Z}^3$ (away from the measure-zero set of
  cell boundaries), and the theorem extends with $M' = M \circ \pi^{-1}$.
  For generic $R$ it does not: space itself — the memory — is the one
  world-anchored structure in the language.

## 5. Functions: a displacement calculus

Every opcode advances $P$ by its raw segment, *including* skipped segments,
`CALL`, and `RET`. This uniformity yields the language's cleanest algebraic
fact:

> **Proposition 6.** Along any fixed control path $e_1 e_2 \cdots e_m$, the
> net displacement is $\sum_i v_{e_i}$ — independent of the frame, the data
> state, and the opcodes the segments decode to. Consequently the frame
> influences a run's displacement **only** by changing which edges are
> taken (guard outcomes, skip flags, fault/halt timing).
>
> *Proof.* $P$ is modified exclusively by the uniform pre-handler update
> $P \mathrel{+}= v$; no handler writes $P$. (`POPF` restores the frame
> only — the one spec ambiguity resolved specifically to keep this
> proposition true.) ∎

A call executes the callee's edges under the caller's live $F$ and returns
$d = P_{\text{ret}} - P_{\text{call}}$, with move-then-execute ordering
arranged so $d$ is *exactly* the callee's swept segments. A callee is thus a
map

$$ d : SO(3) \longrightarrow \mathbb{R}^3, \qquad F \mapsto \sum_{e \in \pi(F)} v_e, $$

where $\pi(F)$ is the control path the frame selects. Proposition 6 says
this map is **piecewise constant**: a branch-free callee is a constant
function (its displacement cannot depend on its argument), and
orientation-as-argument (milestone M3) requires at least one guard. The
`add_two` example is the minimal witness: one branch whose two arms sweep
$0.6\hat x$ versus $2.5\hat x$, selected by whether the caller's frame has
been turned through $\pi$. Extensional equality of programs — "equal net
displacement" in the spec — is equality of these maps.

`RET` restores the caller's $F$ (frame changes are callee-local; the frame
is an argument, not a channel) but never restores $P$: the callee's movement
*is* its return value, and undoing it would make Proposition 6 vacuous.

## 6. Memory: the integer lattice quotient

Addressing is the quotient map

$$ \mathrm{cell} : \mathbb{R}^3 \to \mathbb{Z}^3, \qquad
   \mathrm{cell}(P) = \big(\lfloor P_x/c \rfloor, \lfloor P_y/c \rfloor, \lfloor P_z/c \rfloor\big), \quad c = 1. $$

`STORE`/`LOAD` act at the cell of the *arrival* point of their own segment.
Because $\mathrm{cell}$ depends only on the endpoint, addressing is
**path-independent**: any route that lands in $[6,7) \times [0,1) \times [0,1)$
reads the same cell — milestone M4's store/walk-away/return-differently test
is exactly this quotient property. Walking is pointer arithmetic in the
literal sense: the reachable address set is the image of the walk.

The quotient's fibers have boundaries (a measure-zero grid of planes), and
floating point makes "on the boundary" a real state; the examples are
authored to land in cell *interiors* (e.g. `memory.vsr` targets
$(6.5, 0.5, 0)$), and §10 records this as a numerical design rule rather
than a semantic one.

## 7. Loops are screw motions

A loop is a cycle in the chain graph — there is no loop construct. What does
iteration look like *in space*?

> **Proposition 7 (staircases and helices).** Let $C$ be a graph cycle whose
> traversal leaves the frame unchanged (no net frame rotation) and let
> $\delta = \sum_{e \in C} v_e$. Then $k$ traversals translate the pose by
> $k\,\delta$: the executed path is the loop-body shape repeated at the
> points of an arithmetic progression — a discrete *staircase* (a
> zero-pitch screw). If instead each iteration repeats the same
> **frame-local** motion $h = (R_h, \delta_h) \in SE(3)$ with $R_h \neq \mathrm{id}$
> (rotations that accumulate), the pose evolves by right multiplication,
> $g_k = g_0 h^k$, and by Chasles' theorem $h$ is a screw motion: the
> iterates lie on a discrete **circular helix** about $h$'s screw axis
> (a circle when the pitch is zero).
>
> *Proof.* First claim: with $F$ restored after each traversal, decode and
> guards repeat verbatim, so each lap executes the same edges and adds the
> same $\delta$ (Prop. 6). Second: right multiplication composes body-frame
> motions; $g_k = g_0 h^k$ by induction, and the orbit of a point under the
> cyclic group $\langle h \rangle \le SE(3)$ generated by a proper screw is
> a helix with rotation angle $\theta_h$ per step and translation
> $\mathrm{pitch}(h)$ along the axis. ∎

Both regimes are visible in the repository's renders: `countdown.vsr` is
the staircase case (identical loop-body copies marching along $\delta$),
and `helix.vsr` is the proper screw — the same three frame-local
instructions repeated under an accumulating $\pi/4$ turn, closing one full
revolution in eight laps. The animated GIFs show the frame triad realizing
$R_h$ lap by lap.

## 8. Program space and interpolation

Fix a program topology: a chain graph with $m$ segment slots (guards and
targets frozen). The geometry of such programs is a point of
$\mathbb{R}^{3m}$, and behavior is a function on that space. Decoding makes
it piecewise constant: pulled back through
$v \mapsto Q(\rho_F^{-1}(\hat v))$, each segment slot partitions its
$\mathbb{R}^3$ factor into open opcode cones, fault shells, and (for the
register ops) magnitude shells at integer radii. On the open cells of the
product partition, the entire opcode sequence — hence, for terminating
programs, the entire behavior — is locally constant.

Straight-line interpolation probes this cell structure along one line:
$h(t) = (1-t)\,a + t\,b$ for topology-equal programs $a, b$. The repository
instance interpolates two extensionally equal countdowns whose only
direction-changing slot lerps the loop filler from
$\mathrm{NOP} = (-1,-1,1)/\sqrt3$ to $\mathrm{PUSHF} = (-1,1,1)/\sqrt3$.
Writing $s = 2t - 1$, the lerped direction has normalized $y$-component

$$ \hat y(s) = \frac{s}{\sqrt{2 + s^2}}, $$

which crosses the cubic dead band $|\hat y| \in (0.30, 0.40)$ for
$|s| \in (0.4447, 0.6172)$. Solving back to $t$ **predicts** fault bands

$$ t \in (0.1914,\ 0.2776) \cup (0.7224,\ 0.8086), $$

a viable fraction of $0.8275$ — against the measured $0.825$ over 401
interpolants (`examples/interpolate.py`; the strip chart in the README is
this computation). Between the bands, $|\hat y| < 0.30$: the segment has
left NOP's cone entirely and decodes as $\mathrm{PROJ} = (-1,0,1)/\sqrt2$ —
a *different program with identical behavior*, since projecting the
already-axis-aligned accumulator onto $R_0 = \hat x$ is the identity there.
The interpolation thus exhibits, in one dimension, all three phenomena the
cell structure allows: behavior-preserving cells, detected fault walls
(Lemma 2 guarantees walls, never silent retargeting), and neighboring cells
that happen to compute the same function — the germ of the spec's open
question 5, where extensional equivalence classes would be studied as
unions of cells and homotopy classes of paths between them.

### 8.1 Robustness maps and evolutionary search

`versor/synth.py` operationalizes the cell picture. **Per-segment
tolerance** — the largest perturbation radius of one segment that preserves
output in every sampled direction — empirically measures the distance to
the nearest behavior wall, and on `countdown` it cleanly separates two
kinds of geometry: *structure-carrying* segments (branch arms, OUT, the
register-indexed ops) tolerate 0.15–0.31 before anything changes, while
*value-carrying* segments (the LOADI whose magnitudes are the output
values) have tolerance zero — the magnitude channel is continuous, so its
exact-output cells are measure-zero slices. A Versor program is thus a
mixed discrete/continuous object, and the tolerance map is its local
decomposition.

The same structure supports **synthesis by continuous search**: a
(1+λ) evolution strategy over all segment vectors with a behavioral
fitness (output distance, shaped so that producing more of the target
sequence always dominates value error, with fault and non-halting
penalties) repairs a countdown whose every segment was scrambled by
Gaussian noise — from wrong output back to printing 3 2 1 *exactly* (to
the 10⁻⁴ behavioral tolerance) in 75 generations of 16 children
(`examples/synthesize.py`). Discrete cell hops and continuous value polish
alternate: sigma adapts by a 1/5-style rule and re-expands on stall,
because value polish shrinks the mutation scale below what a cell hop
needs. Program space being ℝ^{3m} is what makes this trivially available —
no AST mutation operators, no grammar; just noise on geometry.

## 9. Computability sketch

Two-counter Minsky machines are Turing complete, and Versor hosts them
directly (spec open question 1). The construction below is **mechanized**:
`versor/minsky.py` compiles arbitrary `Inc`/`DecJz` programs to chains, and
`tests/test_minsky.py` runs transfer, doubling, and multiplication machines
end to end. Keep $R_0 = (1,0,0)$ as the unit, and counters as
$R_1 = (c_1, 0, 0)$, $R_2 = (c_2, 0, 0)$.

- **INC($c_j$):** `MOVA j; ADD 0; MOVR j` — four segments of straight-line
  chain, then an unconditional edge (`-> label`) to the next gadget.
- **DECJZ($c_j$, zero-target):** `MOVA j`, then a branch whose *first* arm
  has guard $-\hat x$ and routes to the zero-target, and whose second arm
  has guard $+\hat x$ and continues into `SUB 0; MOVR j`. For $c_j \ge 1$
  the dot products are $-1$ and $+1$: the decrement arm wins. For $c_j = 0$
  the accumulator is zero, all guard dots vanish, and the tie rule selects
  the first-listed (zero) arm — the same convention countdown's exit uses.
- **Control flow** is the chain graph itself; **HALT** is `HALT`.

Counters live in vector magnitudes, so the mathematical machine is
unbounded; the interpreter bounds runs only by its configurable step budget
and floating point (integers are exact in binary floating point up to
$2^{53}$, far beyond any budget-respecting run). Positions drift
unboundedly during a long computation — harmless, since the gadgets above
never touch `STORE`/`LOAD` and Proposition 6 makes drift behaviorally
inert.

## 10. Numerical design

The implementation's numerical rules, in terms of the structures above:

- **Stay on $S^3$.** Every `ROT*` renormalizes $F$ after the Hamilton
  product; drift off the unit sphere would make $\rho_F$ a rotation-plus-
  scale and silently corrupt every decode. Norm preservation of $\rho_F$
  also guarantees the operand $n = \lVert v\rVert$ equals
  $\lVert v_{\text{local}}\rVert$ to rounding.
- **Margins, not exactness.** All decode boundaries carry finite margins
  (component band $0.05$; cosine gap $0.01$), so decode is stable under
  perturbations smaller than the margin (Lemma 2) — this absorbs both
  floating-point noise and authoring imprecision, and it is why the
  builder targets cone *centers*.
- **Zero tests** use $\varepsilon = 10^{-6}$ ($\lVert A\rVert < \varepsilon$
  for `JMPZ` and branch degeneracy). `JMPP` tests the frame-local slot
  against $+\varepsilon$ rather than $0$ so that an exactly-cancelled
  accumulator carrying $10^{-16}$ of rounding residue does not skip.
- **Author to cell interiors.** The lattice quotient's boundaries are
  measure-zero but floating-point-reachable; example programs land at
  half-integer coordinates. Exact binary fractions ($0.5, 0.75, 1.75$)
  are used so that walk arithmetic cancels without residue.
- **Char output** rounds the frame-local scalar, tolerating the rotation
  round-trip's $\sim 10^{-15}$ relative error before `chr`.

## 11. Implementation correspondence

| Mathematics | Implementation |
|---|---|
| $S^3$, Hamilton product, $\rho_q$, axis-angle | `versor/quat.py` |
| Sphere partitions $Q$, cone-center sections, margins | `versor/decode.py` (`Cubic26`, `Icosa32`) |
| Opcode handlers $h_k$, frame-local scalar slot | `versor/isa.py` |
| Step map, branch rule, covariance-critical orderings | `versor/machine.py` |
| Chain graphs, $\mathbb{Z}^3$ memory model, serialization | `versor/loader.py` |
| Authoring section $K \to S^2$ composed with tracked $\tilde F$ | `versor/builder.py`, `versor/asm.py` |
| Runs as discrete curves in $\mathbb{R}^3$; screw orbits | `versor/trace.py`, `versor/viz.py` |
| $\mathbb{R}^{3m}$ program space, line probes | `versor/interp.py`, `examples/interpolate.py` |
| Wall distances & evolutionary search (§8.1) | `versor/synth.py`, `examples/synthesize.py` |
| Minsky embedding (§9), mechanized | `versor/minsky.py`, `tests/test_minsky.py` |
| Antipodal line packing (open problem 6) | `tools/optimize_sphere26.py` → `Sphere26` |
| JS semantics parity (golden files) | `docs/playground/`, `tools/make_golden.py` |
| Theorem 5 / Prop. 4 / Prop. 6 / §8 numbers as tests | `tests/` (M1, M3, M6 suites) |
| Every number in this paper | `docs/calcs.py` |

## 12. Open problems

1. ~~Scale as an argument channel~~ (spec Q2) — shipped in v0.3b. The pose
   group is now $\mathrm{Sim}(3)$: a scalar $s$ multiplies *movement*
   ($P \mathrel{+}= s\,v$) but not operands, sidestepping the register-index
   obstruction ($\lfloor s\,n \rfloor \bmod 4$ would retarget registers,
   which is why operand scaling was rejected). Scale rides `CALL`'s
   fractional magnitude, $s' = s\cdot 2^{\,2\,\mathrm{frac}(n)-1}$, with the
   builder's `idx + 0.5` convention giving factor exactly 1; `RET`/`POPF`
   restore it. One consequence discovered in implementation deserves
   emphasis: **frame-covariant recursion is impossible with plain chains.**
   A callee's raw vectors decode under the caller's live frame, so a chain
   behaves identically only across call sites sharing an orientation —
   scaled self-similar recursion works straight off (`zoom.vsr`), but
   fractals with *turns* (Koch, Lévy) require orientation-specialized chain
   clones, one per call-site frame class. That cloning transform is
   exactly what a compiler targeting Versor would have to do, and remains
   open.
2. **Executable memory** (spec Q4): stored vectors as code would make the
   step map depend on $M$ along the path, breaking the piecewise-constant
   structure of §8 in interesting ways (program space becomes
   self-referential).
3. **Effect regions and homotopy** (spec Q5): define I/O and memory regions
   as subsets of $\mathbb{R}^3$ and ask when two paths are equivalent
   through path homotopy in the complement — §8's cell decomposition is the
   finite shadow of that question.
4. ~~Mechanize §9~~ — done: `versor/minsky.py` compiles Minsky programs to
   chains, tested end to end. The remaining formal gap is a machine-checked
   proof that the gadget semantics match (e.g. in Lean), plus lifting the
   interpreter's finite step budget into the statement.
5. **The six reserved cones** of `icosa32` await opcodes; candidates should
   respect the antipodal symmetry of the current assignment.
6. ~~Equal-area decoding~~ — shipped as `sphere26`: an antipodal 13-line
   packing found by annealed Riesz repulsion (`tools/optimize_sphere26.py`;
   the cubic seed and every random restart converge to the same 38.17°
   minimum separation, so it is plausibly the optimal antipodal packing).
   All 26 opcodes, no reserved cones, near-uniform margins: 92.5 % of the
   sphere decodes, versus 72.2 % for cubic-26 and 73.7 % for icosa32. What
   remains open is a *proof* of packing optimality and a decoder whose cell
   *areas* (not just margins) are equalized.
