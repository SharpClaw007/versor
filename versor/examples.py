"""The showpiece programs from the spec milestones, built with the builder.

Kept inside the package so tests and examples/make_examples.py share one
source of truth.
"""
from __future__ import annotations

import math

from .builder import ProgramBuilder, arm, arm_seg
from .loader import Program


def straightline() -> ProgramBuilder:
    """M1: LOADI 5 -> SCALE 2 -> OUT prints 10."""
    b = ProgramBuilder("straightline")
    b.chain("entry").loadi(5).scale(2).out().halt()
    return b


def countdown(n: int = 5, decoder: str = "cubic26") -> ProgramBuilder:
    """M2: prints n, n-1, ..., 1 via a literal cycle in the chain graph.

    R0 holds the unit decrement. The branch lists the exit edge first so the
    zero-accumulator tie rule routes execution out when A reaches (0,0,0).

    `decoder` re-aims every segment at that decoder's cone centers, so the
    same program authors cleanly under cubic26 or icosa32 (M6).
    """
    b = ProgramBuilder("countdown", decoder=decoder)
    c = b.chain("entry")
    c.loadi(1).movr(0)          # R0 = unit decrement
    c.loadi(n)                  # A = counter
    c.label("loop")
    c.out().sub(0)              # print, then A -= R0
    c.branch(
        arm("HALT", 1.0, guard=(-1, 0, 0), to="end"),  # first: wins the A=0 tie
        arm("NOP", 1.0, guard=(1, 0, 0), to="loop"),   # A.x > 0: keep cycling
    )
    return b


def countdown_b(n: int = 5) -> ProgramBuilder:
    """A second countdown implementation for the M6 interpolation demo.

    Same graph topology and guards as countdown(), same output, different
    geometry: free magnitudes moved (register indices and OUT's float mode
    survive because floor(n) and the n >= 2 threshold don't care), and the
    loop-back filler is PUSHF instead of NOP — a different corner direction,
    so interpolants sweep across cone boundaries on the way from A to B.
    """
    b = ProgramBuilder("countdown-b")
    c = b.chain("entry")
    c.loadi(1).op("MOVR", 0.9)          # R0 = unit decrement (idx still 0)
    c.loadi(n)
    c.label("loop")
    c.op("OUT", 1.9).op("SUB", 0.75)    # still float mode, still R0
    c.branch(
        arm("HALT", 2.5, guard=(-1, 0, 0), to="end"),
        arm("PUSHF", 1.0, guard=(1, 0, 0), to="loop"),  # harmless filler
    )
    return b


def add_two() -> ProgramBuilder:
    """M3: orientation is the argument.

    Chain 1 branches on the *world* accumulator against frame-local guards:
    rotating the caller's frame flips which arm wins, so the two calls sweep
    different displacements (0.6 vs 2.5 along x) — and the caller reads them
    through its own frame, printing 0.6 then -2.5.
    """
    b = ProgramBuilder("add_two")

    c0 = b.chain("entry")
    c0.loadi(1)                 # A = world +x: the branch input for both calls
    c0.call(1).out()            # frame = identity: short arm, prints 0.6
    c0.roth(math.pi)            # rotate frame; builder re-aims later segments
    c0.call(1).out()            # same chain, flipped guards: prints -2.5
    c0.halt()

    c1 = b.chain("fork: displacement depends on caller frame")
    c1.branch(
        arm_seg((0.6, 0, 0), guard=(1, 0, 0)),    # LOADI under identity
        arm_seg((2.5, 0, 0), guard=(-1, 0, 0)),   # STORE under a pi z-rotation
    )
    return b


def memory() -> ProgramBuilder:
    """M4: store at a cell, walk away, return by a different route, load.

    Every instruction moves, so the route is chosen op-by-op: MOVR walks -y,
    DOT and SCALE zig-zag back (their data effects are dead weight), and the
    LOAD segment itself lands inside the stored cell (6,0,0).
    """
    b = ProgramBuilder("memory")
    c = b.chain("entry")
    c.loadi(7)                          # A = 7, P = (7, 0, 0)
    c.store(0.5)                        # P = (6.5, 0, 0): M[(6,0,0)] = A
    c.op("MOVR", 2.0)                   # walk away: P = (6.5, -2, 0)
    c.loadi(3)                          # further away: P = (9.5, -2, 0), A clobbered
    c.op("DOT", 0.75 * math.sqrt(2))    # zig:  P = (8.75, -2.75, 0)
    c.op("SCALE", 2.25 * math.sqrt(2))  # zag:  P = (6.5, -0.5, 0)
    c.load(1)                           # land at (6.5, 0.5, 0) = cell (6,0,0)
    c.out()                             # prints 7
    c.halt()
    return b


def hello(text: str = "Hello, world!\n") -> ProgramBuilder:
    """M5: OUT in char mode. The program is the skyline of the char codes."""
    b = ProgramBuilder("hello")
    c = b.chain("entry")
    for ch in text:
        c.loadi(float(ord(ch))).outc()
    c.halt()
    return b


def fib(count: int = 8) -> ProgramBuilder:
    """M5: iterative Fibonacci. Prints 1, 2, 3, 5, 8, ... (`count` values).

    R0 = a, R1 = b, R2 = scratch, R3 = loop counter. The counter's decrement
    unit is minted each lap by NORMing the counter itself, so four registers
    suffice.
    """
    b = ProgramBuilder("fib")
    c = b.chain("entry")
    c.loadi(1).movr(1)                  # b = 1 (a = R0 defaults to 0)
    c.loadi(count).movr(3)              # counter
    c.label("loop")
    c.mova(0).add(1).movr(2)            # R2 = t = a + b
    c.mova(1).movr(0)                   # a = b
    c.mova(2).movr(1)                   # b = t   (A still holds t)
    c.out()
    c.mova(3).norm().movr(2)            # R2 = unit(counter)
    c.mova(3).sub(2).movr(3)            # counter -= 1
    c.branch(
        arm("HALT", 1.0, guard=(-1, 0, 0), to="end"),
        arm("NOP", 1.0, guard=(1, 0, 0), to="loop"),
    )
    return b


def helix(laps: int = 8) -> ProgramBuilder:
    """Showpiece: the frame IS the geometry.

    Every lap emits the same three frame-local intents — LOADI 2, MOVR, and a
    ROTG twist — but because ROTG accumulates, the identical local program
    corkscrews through world space. No branches, no loop: the helix is the
    unrolled consequence of frame rotation.

    8 laps (45-degree steps) keeps every accumulated direction clear of the
    identity-frame dead zone, so the example lints clean; finer steps would
    run fine but trip false-positive lint warnings (the lint decodes under
    identity and cannot see the accumulated frame).
    """
    b = ProgramBuilder("helix")
    c = b.chain("entry")
    for _ in range(laps):
        c.loadi(2).movr(0).rotg(2 * math.pi / laps)
    c.halt()
    return b


def zoom(depth: int = 8) -> ProgramBuilder:
    """Sim(3) showpiece (v0.3b): telescoping recursion by pure scale.

    Each level draws the same two-segment motif and calls the next level at
    0.55 scale; RET restores the caller's scale, so the trace is the motif
    at 0.55^k for k = 0..depth-1 — geometric self-similarity from the call
    stack alone.

    No frame rotations, deliberately: a callee's raw vectors decode under
    the caller's live frame, so a chain only behaves identically across
    call sites that share an orientation. Fractals with turns (Koch, Lévy)
    need orientation-specialized chain clones — see the whitepaper's
    Sim(3) notes.
    """
    b = ProgramBuilder("zoom")
    b.chain("entry").call(1).halt()
    for k in range(1, depth + 1):
        c = b.chain(f"level {k}")
        c.nop(2.0)
        c.op("MOVR", 2.0)          # walk -y (R2 write is harmless)
        if k < depth:
            c.call(k + 1, scale=0.55)
        c.nop(2.0)
    return b


ALL = {
    "straightline": straightline,
    "countdown": countdown,
    "countdown_b": countdown_b,
    "add_two": add_two,
    "memory": memory,
    "hello": hello,
    "fib": fib,
    "helix": helix,
    "zoom": zoom,
}


def build_all() -> dict[str, Program]:
    return {name: fn().build() for name, fn in ALL.items()}
