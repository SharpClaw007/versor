"""Opcode table v0.1: sign triple -> handler.

Handlers run *after* the machine has advanced along the segment
(move-then-execute), so STORE/LOAD address the arrival cell, CALL pushes the
post-move position (the CALL segment belongs to the caller's path), and RET
computes displacement after its own move (the RET segment belongs to the
callee's shape). Net effect: a function's return value is the displacement
swept by exactly the callee's segments.

Scalar convention: the accumulator A lives in world coordinates. Anything
that reads or writes "A.x as a scalar" (LOADI, DOT, OUT, JMPP) does so in
*frame-local* coordinates, consistent with LOADI's definition
A = F·(n,0,0)·F⁻¹. This is what makes a chain rotated together with its
frame produce identical output (M1 frame covariance).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import numpy as np

from .errors import VersorFault
from .quat import Quat

EPS = 1e-6

X_HAT = np.array([1.0, 0.0, 0.0])
Y_HAT = np.array([0.0, 1.0, 0.0])
Z_HAT = np.array([0.0, 0.0, 1.0])


def reg_index(n: float) -> int:
    return int(math.floor(n)) % 4


def _local_x(m) -> float:
    """Frame-local x component of A."""
    return float(m.F.conj().rotate(m.A)[0])


def _set_local_x(m, s: float) -> None:
    """A = scalar s in the frame-local x slot."""
    m.A = m.F.rotate(np.array([s, 0.0, 0.0]))


# --- face directions: data ---

def _loadi(m, n):
    _set_local_x(m, n)


def _store(m, n):
    m.M[m.cell()] = m.A.copy()


def _load(m, n):
    m.A = m.M.get(m.cell(), np.zeros(3)).copy()


def _movr(m, n):
    m.R[reg_index(n)] = m.A.copy()


def _mova(m, n):
    m.A = m.R[reg_index(n)].copy()


def _halt(m, n):
    m.halt("HALT")


# --- edge directions: arithmetic & geometry ---

def _add(m, n):
    m.A = m.A + m.R[reg_index(n)]


def _sub(m, n):
    m.A = m.A - m.R[reg_index(n)]


def _scale(m, n):
    m.A = m.A * n


def _dot(m, n):
    _set_local_x(m, float(np.dot(m.A, m.R[reg_index(n)])))


def _cross(m, n):
    m.A = np.cross(m.A, m.R[reg_index(n)])


def _norm(m, n):
    a = float(np.linalg.norm(m.A))
    if a < EPS:
        raise VersorFault("DivisionByZero", "NORM of zero-magnitude accumulator")
    m.A = m.A / a


def _proj_vec(m, n) -> np.ndarray:
    r = m.R[reg_index(n)]
    rr = float(np.dot(r, r))
    if rr < EPS * EPS:
        raise VersorFault("DivisionByZero", "projection onto zero-magnitude register")
    return (float(np.dot(m.A, r)) / rr) * r


def _proj(m, n):
    m.A = _proj_vec(m, n)


def _rej(m, n):
    m.A = m.A - _proj_vec(m, n)


def _rotf(m, n):
    m.F = (m.F * Quat.axis_angle(X_HAT, n)).normalized()


def _rotg(m, n):
    m.F = (m.F * Quat.axis_angle(Y_HAT, n)).normalized()


def _roth(m, n):
    m.F = (m.F * Quat.axis_angle(Z_HAT, n)).normalized()


def _out(m, n):
    x = _local_x(m)
    if n >= 2.0 - 1e-9:
        m.OUT.append(chr(round(x)))
    else:
        m.OUT.append(x)


# --- corner directions: control & calls ---

def _call(m, n):
    cid = int(math.floor(n)) % len(m.program.chains)
    if len(m.CS) >= m.max_call_depth:
        raise VersorFault("CallStackOverflow", f"call depth {m.max_call_depth} exceeded")
    # m.vertex was already advanced to the CALL edge's target = return address.
    m.CS.append((m.chain, m.vertex, m.F, m.P.copy()))
    m.chain = cid
    m.vertex = m.program.chains[cid].entry


def _ret(m, n):
    m.do_ret()


def _jmpz(m, n):
    if float(np.linalg.norm(m.A)) < EPS:
        m.skip = True


def _jmpp(m, n):
    if _local_x(m) > EPS:
        m.skip = True


def _pushf(m, n):
    m.AUX.append((m.F, m.P.copy()))


def _popf(m, n):
    if not m.AUX:
        raise VersorFault("StackUnderflow", "POPF on empty aux stack")
    f, _p = m.AUX.pop()
    m.F = f  # frame only; position is NOT restored (spec open Q3)


def _nop(m, n):
    pass


def _fault(m, n):
    raise VersorFault("ExplicitFault", f"FAULT opcode, operand {n:.6g}")


@dataclass(frozen=True)
class Op:
    mnemonic: str
    klass: str  # data | arithmetic | frame | control
    handler: Callable


OPCODES: dict[tuple[int, int, int], Op] = {
    # face
    (+1, 0, 0): Op("LOADI", "data", _loadi),
    (-1, 0, 0): Op("STORE", "data", _store),
    (0, +1, 0): Op("LOAD", "data", _load),
    (0, -1, 0): Op("MOVR", "data", _movr),
    (0, 0, +1): Op("MOVA", "data", _mova),
    (0, 0, -1): Op("HALT", "control", _halt),
    # edge
    (+1, +1, 0): Op("ADD", "arithmetic", _add),
    (+1, -1, 0): Op("SUB", "arithmetic", _sub),
    (-1, +1, 0): Op("SCALE", "arithmetic", _scale),
    (-1, -1, 0): Op("DOT", "arithmetic", _dot),
    (+1, 0, +1): Op("CROSS", "arithmetic", _cross),
    (+1, 0, -1): Op("NORM", "arithmetic", _norm),
    (-1, 0, +1): Op("PROJ", "arithmetic", _proj),
    (-1, 0, -1): Op("REJ", "arithmetic", _rej),
    (0, +1, +1): Op("ROTF", "frame", _rotf),
    (0, +1, -1): Op("ROTG", "frame", _rotg),
    (0, -1, +1): Op("ROTH", "frame", _roth),
    (0, -1, -1): Op("OUT", "data", _out),
    # corner
    (+1, +1, +1): Op("CALL", "control", _call),
    (+1, +1, -1): Op("RET", "control", _ret),
    (+1, -1, +1): Op("JMPZ", "control", _jmpz),
    (+1, -1, -1): Op("JMPP", "control", _jmpp),
    (-1, +1, +1): Op("PUSHF", "frame", _pushf),
    (-1, +1, -1): Op("POPF", "frame", _popf),
    (-1, -1, +1): Op("NOP", "control", _nop),
    (-1, -1, -1): Op("FAULT", "control", _fault),
}

# canonical unit direction for each mnemonic (for the builder)
DIRECTIONS: dict[str, np.ndarray] = {
    op.mnemonic: np.array(triple, dtype=float) / np.linalg.norm(triple)
    for triple, op in OPCODES.items()
}
