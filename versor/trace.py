"""Execution trace recorder — the raw material for the visualizer."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class StepRecord:
    step: int
    chain: int
    frm: int                 # vertex the segment left from
    to: int                  # vertex the segment arrived at
    P0: np.ndarray           # position before the move
    P1: np.ndarray           # position after the move
    F: tuple                 # frame (w, x, y, z) after execution
    opcode: str
    klass: str               # data | arithmetic | frame | control
    n: float                 # operand magnitude
    A: np.ndarray            # accumulator after execution
    skipped: bool = False    # segment moved but was not executed (JMPZ/JMPP)
    branch: bool = False     # segment was chosen at a branch vertex
    out_len: int = 0         # length of the OUT buffer after this step
    s: float = 1.0           # Sim(3) scale after this step (v0.3b)


class Trace:
    def __init__(self):
        self.records: list[StepRecord] = []

    def record(self, **kw) -> None:
        self.records.append(StepRecord(**kw))

    def __len__(self) -> int:
        return len(self.records)

    def __iter__(self):
        return iter(self.records)

    def opcodes(self) -> list[str]:
        return [r.opcode for r in self.records]
