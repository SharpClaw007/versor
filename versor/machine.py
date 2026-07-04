"""The Versor machine: state + step() + run().

Semantics decisions locked for v0.1 (spec ambiguities resolved):

1. Scalar reads/writes of A.x are frame-local (see isa.py docstring).
2. Running off the end of a non-root chain is an implicit RET; only the
   root chain halts the program that way (spec 5 wins over spec 3.2).
3. Move-then-execute: P += v_raw happens before the handler runs, so
   STORE/LOAD address the arrival cell and the returned displacement is
   exactly the callee's swept segments (incl. its RET segment, excl. the
   caller's CALL segment).
4. Branch with |A| < eps: A_normalized is taken as the zero vector, every
   guard dot is 0, and the existing tie rule picks the first listed edge.
5. RET restores the caller's frame F (that is why CALL pushes it); position
   P is deliberately NOT restored — the callee physically moved the machine.
6. RET with an empty call stack is a StackUnderflow fault.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .decode import get_decoder
from .errors import VersorFault
from .isa import EPS, OPCODES
from .loader import Program
from .quat import Quat
from .trace import Trace

CELL_SIZE = 1.0
DEFAULT_STEP_BUDGET = 1_000_000
DEFAULT_MAX_CALL_DEPTH = 1024


@dataclass
class RunResult:
    out: list                     # OUT buffer: floats and 1-char strings
    displacement: np.ndarray      # net displacement of the root chain
    halt_reason: str
    steps: int

    def out_text(self) -> str:
        """OUT buffer rendered as text: chars joined, floats one per line."""
        lines, run = [], []
        for item in self.out:
            if isinstance(item, str):
                run.append(item)
            else:
                if run:
                    lines.append("".join(run))
                    run = []
                lines.append(f"{item:.6g}")
        if run:
            lines.append("".join(run))
        return "\n".join(lines)


class Machine:
    def __init__(self, program: Program, *, decoder: str | None = None,
                 step_budget: int = DEFAULT_STEP_BUDGET,
                 max_call_depth: int = DEFAULT_MAX_CALL_DEPTH,
                 F0: Quat | None = None, P0=None, trace: Trace | None = None):
        self.program = program
        self.decoder = get_decoder(decoder or program.decoder)
        self.step_budget = step_budget
        self.max_call_depth = max_call_depth
        self.trace = trace

        self.P = np.zeros(3) if P0 is None else np.asarray(P0, dtype=float).copy()
        self._P_start = self.P.copy()
        self.F = F0 if F0 is not None else Quat.identity()
        self.A = np.zeros(3)
        self.R = [np.zeros(3) for _ in range(4)]
        self.M: dict[tuple[int, int, int], np.ndarray] = {}
        self.CS: list[tuple[int, int, Quat, np.ndarray]] = []
        self.AUX: list[tuple[Quat, np.ndarray]] = []
        self.OUT: list = []

        self.chain = 0
        self.vertex = program.chains[0].entry
        self.steps = 0
        self.skip = False
        self.halted = False
        self.halt_reason = ""

    # --- helpers used by opcode handlers ---

    def cell(self) -> tuple[int, int, int]:
        return tuple(int(math.floor(c / CELL_SIZE)) for c in self.P)

    def halt(self, reason: str) -> None:
        self.halted = True
        self.halt_reason = reason

    def fault(self, kind: str, message: str) -> None:
        self.halt(f"fault: {kind}")
        raise VersorFault(kind, message, step=self.steps, chain=self.chain,
                          vertex=self.vertex)

    def do_ret(self) -> None:
        """Shared by the RET opcode and implicit RET at chain end."""
        if not self.CS:
            self.fault("StackUnderflow", "RET with empty call stack")
        chain, vertex, f_saved, p_saved = self.CS.pop()
        self.A = self.P - p_saved      # return value = callee net displacement
        self.F = f_saved               # frame changes are callee-local
        self.chain, self.vertex = chain, vertex

    # --- execution ---

    def _pick_branch(self, edges):
        na = float(np.linalg.norm(self.A))
        a = self.A / na if na >= EPS else np.zeros(3)
        best, best_dot = None, -math.inf
        for e in edges:
            d = float(np.dot(self.F.rotate(e.guard), a))
            if d > best_dot + 1e-12:  # strict: ties keep the first listed edge
                best, best_dot = e, d
        return best

    def step(self) -> None:
        if self.halted:
            return
        if self.steps >= self.step_budget:
            self.fault("StepBudgetExhausted", f"budget of {self.step_budget} steps")

        edges = self.program.chains[self.chain].vertices[self.vertex]
        if not edges:
            if not self.CS:
                self.halt("end of root chain")
            else:
                frm = self.vertex
                self.do_ret()
                self.steps += 1
                if self.trace is not None:
                    self.trace.record(
                        step=self.steps, chain=self.chain, frm=frm, to=self.vertex,
                        P0=self.P.copy(), P1=self.P.copy(), F=self.F.as_tuple(),
                        opcode="RET*", klass="control", n=0.0, A=self.A.copy(),
                        out_len=len(self.OUT))
            return

        is_branch = len(edges) > 1
        edge = self._pick_branch(edges) if is_branch else edges[0]

        v_raw = edge.seg
        n = float(np.linalg.norm(v_raw))
        if n < EPS:
            self.fault("ZeroLengthSegment", "cannot decode a zero-length segment")
        v_local = self.F.conj().rotate(v_raw)
        try:
            triple = self.decoder.decode(v_local / n)
        except VersorFault as f:
            self.fault(f.kind, f.message)
        op = OPCODES[triple]

        frm = self.vertex
        p0 = self.P.copy()
        self.P = self.P + v_raw          # move ...
        self.vertex = edge.to
        skipped = self.skip
        if skipped:
            self.skip = False
        else:
            try:
                op.handler(self, n)      # ... then execute
            except VersorFault as f:
                if f.step is None:       # annotate handler faults with location
                    self.halt(f"fault: {f.kind}")
                    raise VersorFault(f.kind, f.message, step=self.steps,
                                      chain=self.chain, vertex=self.vertex) from None
                raise
        self.steps += 1

        if self.trace is not None:
            self.trace.record(
                step=self.steps, chain=self.chain, frm=frm, to=self.vertex,
                P0=p0, P1=p0 + v_raw, F=self.F.as_tuple(), opcode=op.mnemonic,
                klass=op.klass, n=n, A=self.A.copy(), skipped=skipped,
                branch=is_branch, out_len=len(self.OUT))

    def run(self) -> RunResult:
        while not self.halted:
            self.step()
        return RunResult(out=self.OUT, displacement=self.P - self._P_start,
                         halt_reason=self.halt_reason, steps=self.steps)


def run_program(program: Program, **kw) -> RunResult:
    return Machine(program, **kw).run()
