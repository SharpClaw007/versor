"""Fluent Python API for constructing Versor programs.

The builder tracks an *authoring frame*: every helper takes its arguments in
frame-local intent ("LOADI 5", "rotate about local z by pi") and emits the
raw world-space vector the runtime will decode back to that opcode — assuming
execution reaches it with the same frame history. After rotf/rotg/roth the
authoring frame updates exactly like the runtime frame, so straight-line
authoring stays sane even mid-rotation. Guards are stored frame-local per the
spec, so they are NOT rotated by the authoring frame.

Register-indexed ops encode idx as magnitude idx + 0.5 (floor mod 4 = idx);
CALL encodes chain id the same way. Magnitude is the operand, so ops like
SCALE cannot encode non-positive factors — that is the language, not the
builder.

Branching:

    c.label("loop")
    c.out().sub(0)
    c.branch(
        arm("HALT", 1.0, guard=(-1, 0, 0), to="end"),   # listed first: wins ties
        arm("NOP", 1.0, guard=(1, 0, 0), to="loop"),
    )

Targets may be labels defined earlier, later, or never (never = a fresh
dead-end vertex). After branch() the cursor is unset; continue with at(label).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .decode import get_decoder
from .errors import LoadError
from .isa import MNEMONIC_TO_TRIPLE
from .loader import Program, from_dict
from .quat import Quat

TWO_PI = 2.0 * math.pi


@dataclass
class Arm:
    """One outgoing edge of a branch. Opcode intent is resolved against the
    program's decoder direction table at branch() time, so `arm()` itself
    stays decoder-agnostic."""
    guard: np.ndarray
    to: str | int | None
    op: str | None = None
    n: float = 0.0
    local_seg: np.ndarray | None = None


def arm(op: str, n: float, *, guard, to=None) -> Arm:
    """One outgoing edge of a branch: opcode intent + guard + target."""
    if op not in MNEMONIC_TO_TRIPLE:
        raise LoadError(f"unknown mnemonic {op!r}")
    g = np.asarray(guard, dtype=float)
    gn = float(np.linalg.norm(g))
    if gn < 1e-9:
        raise LoadError("arm: zero guard")
    return Arm(guard=g / gn, to=to, op=op, n=float(n))


def arm_seg(local_seg, *, guard, to=None) -> Arm:
    """Branch arm from an explicit frame-local segment vector."""
    g = np.asarray(guard, dtype=float)
    return Arm(guard=g / float(np.linalg.norm(g)), to=to,
               local_seg=np.asarray(local_seg, dtype=float))


def _op_vec(mnemonic: str, n: float,
            dirs: dict[str, np.ndarray]) -> np.ndarray:
    if mnemonic not in dirs:
        if mnemonic in MNEMONIC_TO_TRIPLE:
            raise LoadError(
                f"{mnemonic} is a Versor-32 extended opcode with no cone "
                "under this decoder — use decoder='icosa32' or 'sphere32'")
        raise LoadError(f"unknown mnemonic {mnemonic!r}")
    if n <= 1e-9:
        raise LoadError(f"{mnemonic}: operand magnitude must be positive, got {n}")
    return dirs[mnemonic] * float(n)


def _reg_n(idx: int) -> float:
    if not 0 <= idx <= 3:
        raise LoadError(f"register index {idx} out of range 0..3")
    return idx + 0.5


class ChainBuilder:
    def __init__(self, cid: int, comment: str = "",
                 dirs: dict[str, np.ndarray] | None = None):
        self.id = cid
        self.comment = comment
        self._dirs = dirs if dirs is not None else _mnemonic_dirs("cubic26")
        self._edges: dict[int, list[dict]] = {0: []}
        self._next_vid = 1
        self._cursor: int | None = 0
        self._labels: dict[str, int] = {}
        self._pending: list[tuple[dict, str]] = []  # edges awaiting a label
        self._Fa = Quat.identity()

    # --- structure ---

    def _new_vertex(self) -> int:
        vid = self._next_vid
        self._next_vid += 1
        self._edges[vid] = []
        return vid

    def _require_cursor(self) -> int:
        if self._cursor is None:
            raise LoadError(
                f"chain {self.id}: no cursor after branch(); use at(label) first")
        return self._cursor

    def label(self, name: str) -> "ChainBuilder":
        """Name the current vertex."""
        if name in self._labels:
            raise LoadError(f"chain {self.id}: duplicate label {name!r}")
        self._labels[name] = self._require_cursor()
        return self

    def at(self, name: str) -> "ChainBuilder":
        """Move the cursor to a labeled vertex (creating it if new)."""
        if name not in self._labels:
            self._labels[name] = self._new_vertex()
        self._cursor = self._labels[name]
        return self

    def _resolve(self, target) -> int | None:
        """Vertex id for a target, or None if it is an unresolved label."""
        if target is None:
            return self._new_vertex()
        if isinstance(target, int):
            if target not in self._edges:
                raise LoadError(f"chain {self.id}: unknown target vertex {target}")
            return target
        if target in self._labels:
            return self._labels[target]
        return None  # forward label reference

    def _emit(self, local_vec: np.ndarray, to=None) -> "ChainBuilder":
        frm = self._require_cursor()
        raw = self._Fa.rotate(local_vec)
        edge = {"seg": [float(c) for c in raw], "to": -1}
        resolved = self._resolve(to)
        if resolved is None:
            self._pending.append((edge, to))
        else:
            edge["to"] = resolved
        self._edges[frm].append(edge)
        self._cursor = resolved  # None while pending; at() repositions
        return self

    def seg(self, local_vec, to=None) -> "ChainBuilder":
        """Emit an explicit frame-local segment (rotated by the authoring frame)."""
        return self._emit(np.asarray(local_vec, dtype=float), to)

    def seg_raw(self, raw_vec, to=None) -> "ChainBuilder":
        """Emit a raw world-space segment, bypassing the authoring frame."""
        frm = self._require_cursor()
        edge = {"seg": [float(c) for c in np.asarray(raw_vec, dtype=float)], "to": -1}
        resolved = self._resolve(to)
        if resolved is None:
            self._pending.append((edge, to))
        else:
            edge["to"] = resolved
        self._edges[frm].append(edge)
        self._cursor = resolved
        return self

    def op(self, mnemonic: str, n: float, to=None) -> "ChainBuilder":
        return self._emit(_op_vec(mnemonic, n, self._dirs), to)

    def branch(self, *arms: Arm) -> "ChainBuilder":
        if len(arms) < 2:
            raise LoadError(f"chain {self.id}: branch needs 2+ arms")
        frm = self._require_cursor()
        for a in arms:
            local = (a.local_seg if a.local_seg is not None
                     else _op_vec(a.op, a.n, self._dirs))
            raw = self._Fa.rotate(local)
            edge = {"seg": [float(c) for c in raw],
                    "guard": [float(c) for c in a.guard], "to": -1}
            resolved = self._resolve(a.to)
            if resolved is None:
                self._pending.append((edge, a.to))
            else:
                edge["to"] = resolved
            self._edges[frm].append(edge)
        self._cursor = None
        return self

    # --- opcode helpers (frame-local intent) ---

    def loadi(self, n: float):  return self.op("LOADI", n)
    def store(self, n: float = 1.0):  return self.op("STORE", n)
    def load(self, n: float = 1.0):  return self.op("LOAD", n)

    def exec_cell(self, n: float = 2.0):
        """EXEC: LOAD with n >= 2 executes the arrival cell's vector."""
        if n < 2.0:
            raise LoadError(f"exec_cell: magnitude must be >= 2, got {n}")
        return self.op("LOAD", n)
    def movr(self, idx: int):  return self.op("MOVR", _reg_n(idx))
    def mova(self, idx: int):  return self.op("MOVA", _reg_n(idx))
    def halt(self, n: float = 1.0):  return self.op("HALT", n)

    def add(self, idx: int):  return self.op("ADD", _reg_n(idx))
    def sub(self, idx: int):  return self.op("SUB", _reg_n(idx))
    def scale(self, k: float):  return self.op("SCALE", k)
    def dot(self, idx: int):  return self.op("DOT", _reg_n(idx))
    def cross(self, idx: int):  return self.op("CROSS", _reg_n(idx))
    def norm(self, n: float = 1.0):  return self.op("NORM", n)
    def proj(self, idx: int):  return self.op("PROJ", _reg_n(idx))
    def rej(self, idx: int):  return self.op("REJ", _reg_n(idx))

    def _rot(self, mnemonic: str, axis, angle: float):
        a = angle % TWO_PI
        if a < 1e-9:
            raise LoadError(f"{mnemonic}: rotation angle is zero mod 2*pi "
                            "(magnitude must be positive)")
        self.op(mnemonic, a)
        self._Fa = (self._Fa * Quat.axis_angle(axis, a)).normalized()
        return self

    def rotf(self, angle: float):  return self._rot("ROTF", (1, 0, 0), angle)
    def rotg(self, angle: float):  return self._rot("ROTG", (0, 1, 0), angle)
    def roth(self, angle: float):  return self._rot("ROTH", (0, 0, 1), angle)

    def out(self):  return self.op("OUT", 1.0)
    def outc(self):  return self.op("OUT", 2.0)

    def call(self, cid: int, scale: float = 1.0):
        """CALL chain `cid`; `scale` multiplies the callee's Sim(3) scale,
        encoded in the fractional magnitude as 2^(2*frac - 1). One call can
        carry a factor in (0.5, 2); compose calls for more."""
        if scale == 1.0:
            frac = 0.5
        else:
            frac = 0.5 + math.log2(scale) / 2.0
            if not -1e-9 <= frac < 1.0 - 1e-6:
                raise LoadError(
                    f"call: scale {scale} outside a single call's range "
                    "[0.5, 2) — compose nested calls for larger factors")
            frac = max(frac, 0.0)
        return self.op("CALL", cid + frac)
    def ret(self, n: float = 1.0):  return self.op("RET", n)
    def jmpz(self, n: float = 1.0):  return self.op("JMPZ", n)
    def jmpp(self, n: float = 1.0):  return self.op("JMPP", n)
    def pushf(self, n: float = 1.0):  return self.op("PUSHF", n)
    def popf(self, n: float = 1.0):  return self.op("POPF", n)
    def nop(self, n: float = 1.0):  return self.op("NOP", n)
    def fault(self, code: float = 1.0):  return self.op("FAULT", code)

    # extended Versor-32 opcodes (icosa32/sphere32 only)
    def inp(self, n: float = 1.0):  return self.op("INP", n)
    def swap(self, idx: int):  return self.op("SWAP", _reg_n(idx))
    def pusha(self, n: float = 1.0):  return self.op("PUSHA", n)
    def popa(self, n: float = 1.0):  return self.op("POPA", n)
    def mulr(self, idx: int):  return self.op("MULR", _reg_n(idx))
    def loadp(self, n: float = 1.0):  return self.op("LOADP", n)

    # --- output ---

    def _finish(self) -> dict:
        for edge, name in self._pending:
            if name not in self._labels:
                # never-defined forward label: fresh dead-end vertex
                self._labels[name] = self._new_vertex()
            edge["to"] = self._labels[name]
        self._pending.clear()
        return {
            "id": self.id,
            **({"comment": self.comment} if self.comment else {}),
            "vertices": [{"id": vid, "out": edges}
                         for vid, edges in sorted(self._edges.items())],
        }


def _mnemonic_dirs(decoder: str) -> dict[str, np.ndarray]:
    """Cone-center authoring direction per mnemonic for the given decoder.
    Extended Versor-32 mnemonics only appear when the decoder assigns them
    a cone (icosa32/sphere32); cubic26 stays base-26."""
    by_key = get_decoder(decoder).directions()
    return {mn: by_key[k] for mn, k in MNEMONIC_TO_TRIPLE.items()
            if k in by_key}


class ProgramBuilder:
    def __init__(self, name: str = "", decoder: str = "cubic26"):
        self.name = name
        self.decoder = decoder
        self._dirs = _mnemonic_dirs(decoder)  # validates the decoder name
        self.chains: list[ChainBuilder] = []

    def chain(self, comment: str = "") -> ChainBuilder:
        cb = ChainBuilder(len(self.chains), comment, self._dirs)
        self.chains.append(cb)
        return cb

    def to_dict(self) -> dict:
        return {"version": "0.1", "name": self.name, "decoder": self.decoder,
                "chains": [c._finish() for c in self.chains]}

    def build(self) -> Program:
        """Build and validate (runs the same validation + lint as the loader)."""
        return from_dict(self.to_dict())

    def save(self, path: str) -> Program:
        import json
        prog = self.build()
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
            f.write("\n")
        return prog
