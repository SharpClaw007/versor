# Esolang wiki draft

MediaWiki-formatted draft for an [esolangs.org](https://esolangs.org) article.
Posting requires a wiki account — copy the source below when ready.

---

```mediawiki
'''Versor''' is a [[three-dimensional]] programming language by Juan Reyes
(2026) in which the program is literally a path: every instruction is a
vector in R³ whose ''direction'' selects the opcode and whose ''magnitude''
is the operand, decoded relative to a quaternion orientation frame the
program itself rotates. A program is a polyline (more generally a graph of
polylines); memory is space itself, addressed by the machine's position;
a function's return value is its net displacement.

== Semantics ==

The machine is a rigid frame: a position P ∈ R³ and a unit quaternion F.
Each step decodes the next segment v in body coordinates (F⁻¹vF), quantizes
its direction to one of 26 opcodes (6 face + 12 edge + 8 corner directions
of the cube neighborhood, with a dead zone around cone boundaries that
faults instead of guessing), reads |v| as the operand, and always advances
P += v — movement is the substrate all instructions share.

Rotating the frame (ROTF/ROTG/ROTH, about the frame's ''own'' axes)
reinterprets every downstream instruction; since 3D rotations do not
commute, instruction order matters geometrically. Branches choose among
outgoing edges by comparing guard normals against the accumulator's
direction. Calls execute a stored chain under the caller's live frame —
''orientation is the argument'' — and return the callee's net displacement.

Two-counter [[Minsky machine]]s embed directly (counters as register
vector magnitudes, decrement-jump-if-zero via a branch tie rule), so the
language is Turing-complete modulo the usual unbounded-storage caveat.

== Continuous program space ==

Because a program with fixed topology is a point of R^(3m), programs can be
linearly interpolated, perturbed, and evolved. The reference implementation
demonstrates: interpolation between two extensionally equal countdowns
(82.5% of the straight line between them still works, and a middle stretch
mutates NOP into PROJ while computing the same function); per-segment
robustness maps; and a (1+λ) evolution strategy that repairs a scrambled
program back to exact behavior. Loops are helices: a cycle whose body
repeats a fixed local motion sweeps a discrete screw orbit (Chasles).

== Example ==

Countdown from 5, in the .vasm assembly front-end:

<pre>
.chain entry
        LOADI 1
        MOVR r0                          ; R0 = unit decrement
        LOADI 5                          ; A = counter
loop:   OUT
        SUB r0
        BR -x: HALT -> end, +x: NOP -> loop
</pre>

== External resources ==

* [https://github.com/SharpClaw007/versor Reference implementation]
  (Python interpreter, assembler, visualizer, G-code/STL export)
* [https://sharpclaw007.github.io/versor/playground/ Browser playground]
* [https://github.com/SharpClaw007/versor/blob/main/docs/whitepaper.md
  Mathematics whitepaper]

[[Category:Languages]]
[[Category:2026]]
[[Category:Turing complete]]
[[Category:Multi-dimensional languages]]
```
