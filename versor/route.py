"""Harmless-mover routing: reach a target displacement with instructions
that compute nothing.

The compiler's spill problem (VHL) needs the machine to physically stand in
a chosen memory cell at the moment a STORE or LOAD executes — but every
instruction moves, so "go there" must itself be spelled in instructions
whose *effects* are disposable. Discipline: the accumulator is parked on the
data stack around a route (PUSHA ... POPA), making A dead; then these ops
become pure movement:

    LOADI · SCALE · ADD · SUB · DOT · CROSS      any magnitude
    REJ                                          magnitude < 1 (register
                                                 index 0 → R0, the reserved
                                                 unit; REJ faults on a zero
                                                 register and R0 never is)

Register *reads* can't fault and A is dead, so junk register contents are
irrelevant; only the movement matters. The mover directions are the
decoder's actual cone centers (they differ per dialect — icosa32's DOT is
not cubic26's DOT), so the router solves the decomposition numerically:
the seven movers positively span ℝ³ in every shipped decoder, hence some
3-subset's cone contains any target and yields an exact nonnegative
solution. REJ totals are chunked into sub-unit magnitudes to keep the
register index at 0.

Contract: R0 holds the unit vector (VHL's prelude guarantees it), and A is
parked. Routes are exact to ~1e-9 — far inside the ±0.5 cell tolerance the
spiller needs.
"""
from __future__ import annotations

from itertools import combinations

import numpy as np

from .decode import get_decoder
from .isa import MNEMONIC_TO_KEY

MOVERS = ("LOADI", "SCALE", "ADD", "SUB", "DOT", "CROSS", "REJ")
_MIN = 1e-9
_DIRS_CACHE: dict[str, dict[str, np.ndarray]] = {}


def mover_dirs(decoder: str = "cubic26") -> dict[str, np.ndarray]:
    """Cone-center direction per mover mnemonic, for the given decoder."""
    if decoder not in _DIRS_CACHE:
        by_key = get_decoder(decoder).directions()
        _DIRS_CACHE[decoder] = {
            mn: np.asarray(by_key[MNEMONIC_TO_KEY[mn]], dtype=float)
            for mn in MOVERS
        }
    return _DIRS_CACHE[decoder]


def route(delta, decoder: str = "cubic26") -> list[tuple[str, float]]:
    """Instruction list [(mnemonic, magnitude), ...] whose net movement is
    `delta` (to ~1e-9), using only A-dead-harmless movers."""
    target = np.asarray(delta, dtype=float)
    if np.linalg.norm(target) < _MIN:
        return []
    dirs = mover_dirs(decoder)

    best = None
    for names in combinations(MOVERS, 3):
        basis = np.column_stack([dirs[n] for n in names])
        if abs(np.linalg.det(basis)) < 1e-9:
            continue
        coeffs = np.linalg.solve(basis, target)
        if np.all(coeffs >= -1e-12):
            cost = float(np.sum(np.clip(coeffs, 0, None)))
            if best is None or cost < best[0]:
                best = (cost, names, np.clip(coeffs, 0, None))
    if best is None:  # pragma: no cover — movers positively span R^3
        raise ValueError(f"unroutable delta {target.tolist()} "
                         f"under decoder {decoder!r}")

    _cost, names, coeffs = best
    out: list[tuple[str, float]] = []
    for name, n in zip(names, coeffs):
        if n < _MIN:
            continue
        if name == "REJ":
            while n > 0.9:
                out.append(("REJ", 0.9))
                n -= 0.9
            if n > _MIN:
                out.append(("REJ", float(n)))
        else:
            out.append((name, float(n)))
    return out


def route_displacement(ops: list[tuple[str, float]],
                       decoder: str = "cubic26") -> np.ndarray:
    """Net movement of a mover list (verification and bookkeeping)."""
    dirs = mover_dirs(decoder)
    total = np.zeros(3)
    for mnemonic, n in ops:
        total += dirs[mnemonic] * n
    return total
