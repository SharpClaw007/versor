"""Mechanized Minsky-machine reduction (whitepaper §9, spec open question 1).

A two-counter Minsky machine is compiled to a single Versor chain:

- ``R0`` holds the unit vector (1, 0, 0);
- counters live as ``R1 = (c1, 0, 0)`` and ``R2 = (c2, 0, 0)``;
- ``Inc``  is  MOVA j / ADD 0 / MOVR j  and an unconditional jump;
- ``DecJz`` loads the counter and branches: the **first-listed** arm (guard
  -x) routes to the zero target — when the counter is 0 the accumulator is
  the zero vector, every guard dot vanishes, and the tie rule selects it —
  while the +x arm falls through to SUB 0 / MOVR j and the next line;
- instruction index ``len(program)`` (or anything out of range) halts.

Two-counter Minsky machines are Turing complete (Minsky 1967), so this
module is the constructive half of the reduction; the simulation tests in
``tests/test_minsky.py`` are the evidence that the gadgets implement the
Minsky semantics exactly.
"""
from __future__ import annotations

from dataclasses import dataclass

from .builder import ProgramBuilder, arm


@dataclass(frozen=True)
class Inc:
    counter: int  # 1 or 2
    next: int


@dataclass(frozen=True)
class DecJz:
    counter: int
    next: int  # taken after a successful decrement
    zero: int  # taken when the counter is zero


Instruction = Inc | DecJz


def _check(program: list[Instruction]) -> None:
    for i, ins in enumerate(program):
        if ins.counter not in (1, 2):
            raise ValueError(f"instruction {i}: counter must be 1 or 2, "
                             f"got {ins.counter}")


def compile_minsky(program: list[Instruction], c1: int = 0, c2: int = 0,
                   emit_counters: bool = False) -> ProgramBuilder:
    """Compile a Minsky program to a Versor chain.

    Initial counter values are loaded into R1/R2 (0 uses the register
    default). With ``emit_counters`` the halt gadget OUTs c1 then c2 before
    halting, so the result is observable in the OUT buffer as well as in
    the registers.
    """
    _check(program)
    if min(c1, c2) < 0:
        raise ValueError("counters are non-negative")
    n = len(program)

    def target(k: int) -> str:
        return f"i{k}" if 0 <= k < n else "halt"

    b = ProgramBuilder("minsky")
    c = b.chain("compiled 2-counter Minsky machine")

    c.loadi(1).movr(0)  # R0 = unit
    if c1 > 0:
        c.loadi(c1).movr(1)
    if c2 > 0:
        c.loadi(c2).movr(2)
    c.op("NOP", 1.0, to=target(0))

    for i, ins in enumerate(program):
        c.at(f"i{i}")
        if isinstance(ins, Inc):
            c.mova(ins.counter).add(0).movr(ins.counter)
            c.op("NOP", 1.0, to=target(ins.next))
        else:
            c.mova(ins.counter)
            c.branch(
                arm("NOP", 1.0, guard=(-1, 0, 0), to=target(ins.zero)),
                arm("NOP", 1.0, guard=(1, 0, 0), to=f"d{i}"),
            )
            c.at(f"d{i}")
            c.sub(0).movr(ins.counter)
            c.op("NOP", 1.0, to=target(ins.next))

    c.at("halt")
    if emit_counters:
        c.mova(1).out().mova(2).out()
    c.halt()
    return b
