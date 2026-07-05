"""Interactive REPL: line-at-a-time .vasm with a live machine.

Replay model: the session's accepted lines form a growing program; each new
line re-assembles and re-runs it from scratch (cheap at REPL scale, and it
keeps every invariant of batch execution — authoring-frame tracking,
branches, labels, even .chain directives all work). A line that fails to
assemble is rejected and reported; a line that assembles but faults at
runtime is kept (use :undo to drop it).

Commands: :help :state :mem :list :undo :reset :decoder NAME :input v,v,...
          :save FILE :trace FILE.png
"""
from __future__ import annotations

import math

import numpy as np

from .asm import assemble
from .errors import LoadError, VersorFault
from .machine import Machine
from .trace import Trace


def _fmt_vec(v) -> str:
    return f"({v[0]:.3g}, {v[1]:.3g}, {v[2]:.3g})"


def _fmt_out_item(o) -> str:
    return repr(o) if isinstance(o, str) else f"{o:.6g}"


class Repl:
    def __init__(self, decoder: str = "cubic26", input_buf=None):
        self.decoder = decoder
        self.input_buf = list(input_buf) if input_buf else []
        self.lines: list[str] = []
        self.machine: Machine | None = None
        self.trace: Trace | None = None

    # --- program assembly & replay ---

    def source(self) -> str:
        return f".decoder {self.decoder}\n" + "\n".join(self.lines) + "\n"

    def _replay(self) -> Machine:
        prog = assemble(self.source()).build()
        trace = Trace()
        m = Machine(prog, trace=trace, step_budget=100_000,
                    input=list(self.input_buf))
        try:
            m.run()
        except VersorFault:
            pass  # machine state up to the fault is still inspectable
        self.machine, self.trace = m, trace
        return m

    # --- one REPL interaction; returns lines to display ---

    def feed(self, line: str) -> list[str]:
        line = line.rstrip()
        if not line.strip():
            return []
        if line.strip().startswith(":"):
            return self._command(line.strip())

        prev_out = len(self.machine.OUT) if self.machine else 0
        self.lines.append(line)
        try:
            m = self._replay()
        except LoadError as e:
            self.lines.pop()
            msg = str(e)
            # replay wraps the source with a .decoder line: shift line nums
            return [f"error: {self._reline(msg)}"]

        echo = []
        if m.halt_reason.startswith("fault"):
            echo.append(f"FAULT: {m.halt_reason.removeprefix('fault: ')} "
                        "(line kept; :undo to drop)")
        for o in m.OUT[prev_out:]:
            echo.append(f"OUT: {_fmt_out_item(o)}")
        last = self.trace.records[-1] if self.trace.records else None
        if last is not None:
            state = (f"{last.opcode:<6} A={_fmt_vec(m.A)} P={_fmt_vec(m.P)}")
            if abs(m.S - 1.0) > 1e-12:
                state += f" s={m.S:.4g}"
            echo.append(state)
        return echo

    def _reline(self, msg: str) -> str:
        import re
        def shift(match):
            return f"line {int(match.group(1)) - 1}"
        return re.sub(r"line (\d+)", shift, msg)

    # --- commands ---

    def _command(self, cmd: str) -> list[str]:
        parts = cmd.split(None, 1)
        name, arg = parts[0], (parts[1].strip() if len(parts) > 1 else "")
        if name == ":help":
            return [":state :mem :list :undo :reset :decoder NAME "
                    ":input v,v,... :save FILE :trace FILE.png"]
        if name == ":state":
            if self.machine is None:
                return ["(no instructions yet)"]
            m = self.machine
            w, x, y, z = m.F.as_tuple()
            angle = 2 * math.degrees(math.acos(max(-1.0, min(1.0, w))))
            out = [f"P = {_fmt_vec(m.P)}   A = {_fmt_vec(m.A)}",
                   f"F = ({w:.3g}, {x:.3g}, {y:.3g}, {z:.3g})"
                   f"  [{angle:.1f}° from identity]   s = {m.S:.4g}",
                   f"R0={_fmt_vec(m.R[0])} R1={_fmt_vec(m.R[1])} "
                   f"R2={_fmt_vec(m.R[2])} R3={_fmt_vec(m.R[3])}",
                   f"steps = {m.steps}   OUT = "
                   f"[{', '.join(_fmt_out_item(o) for o in m.OUT)}]"]
            if m.DS:
                out.append(f"data stack: {len(m.DS)} deep")
            return out
        if name == ":mem":
            if self.machine is None or not self.machine.M:
                return ["(memory empty)"]
            return [f"M{cell} = {_fmt_vec(v)}"
                    for cell, v in sorted(self.machine.M.items())]
        if name == ":list":
            return self.lines or ["(empty)"]
        if name == ":undo":
            if not self.lines:
                return ["(nothing to undo)"]
            dropped = self.lines.pop()
            if self.lines:
                self._replay()
            else:
                self.machine = self.trace = None
            return [f"dropped: {dropped.strip()}"]
        if name == ":reset":
            self.lines.clear()
            self.machine = self.trace = None
            return ["reset"]
        if name == ":decoder":
            if arg not in __import__("versor.decode",
                                     fromlist=["DECODERS"]).DECODERS:
                return [f"unknown decoder {arg!r}"]
            self.decoder = arg
            if self.lines:
                try:
                    self._replay()
                except LoadError as e:
                    return [f"decoder set, but program no longer assembles: "
                            f"{self._reline(str(e))}"]
            return [f"decoder = {arg}"]
        if name == ":input":
            try:
                self.input_buf = [float(v) for v in arg.split(",") if v.strip()]
            except ValueError:
                self.input_buf = arg  # text: char codes
            if self.lines:
                self._replay()
            return [f"input = {self.input_buf}"]
        if name == ":save":
            if not arg:
                return ["usage: :save FILE.vasm"]
            with open(arg, "w") as f:
                f.write(self.source())
            return [f"saved {arg}"]
        if name == ":trace":
            if not arg:
                return ["usage: :trace FILE.png"]
            if self.trace is None or not len(self.trace):
                return ["(no trace yet)"]
            from .viz import render
            return [f"rendered {render(self.trace, arg, title='repl session')}"]
        return [f"unknown command {name} (:help)"]


def main() -> int:  # pragma: no cover — terminal loop
    try:
        import readline  # noqa: F401 — history/editing side effect
    except ImportError:
        pass
    repl = Repl()
    print("versor repl — type .vasm instructions; :help for commands; "
          "ctrl-d to exit")
    while True:
        try:
            line = input("vsr> ")
        except EOFError:
            print()
            return 0
        except KeyboardInterrupt:
            print()
            continue
        for out in repl.feed(line):
            print(out)
