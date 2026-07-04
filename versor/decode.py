"""Direction quantization: unit vector -> opcode sign triple.

Two decoders ship:

- ``cubic26`` (v0.1 default): snap components to {-1, 0, +1} at threshold 0.35.
- ``icosa32`` (M6): nearest-neighbor over the 32 directions of icosahedral
  symmetry (12 icosahedron vertices + 20 face normals).

Icosahedral symmetry has orbit sizes 12/20/30 — there is no 26-direction
icosahedral set, so icosa32 keeps the 26-opcode ISA by *assigning* each opcode
a cone and leaving 6 directions reserved (decoding into one is a
``ReservedDirection`` fault, parked for a future ISA extension):

- the 8 corner triples map exactly: (±1,±1,±1)/√3 ARE dodecahedron vertices;
- the 12 edge triples map to the icosahedron vertex with the same sign
  pattern (φ in place of one 1), 13.3° away;
- the 6 face triples map to the axis-heavy dodecahedron vertex with matching
  signs (e.g. +x -> (φ, 0, 1/φ)/√3), 20.9° away — its mirror twin
  (φ, 0, -1/φ) is one of the 6 reserved directions.

Consequences: programs authored on corner/edge cone centers decode the same
under both decoders; cubic face directions sit on an exact Voronoi tie under
icosa32 and are therefore ambiguous — icosa32 programs author face ops on the
icosa cone centers instead (the builder handles this via its ``decoder``
parameter).

Dead-zone rules:

- cubic26 (spec 3.3, corrected): the spec writes `|v_i - t| < 0.05`, which
  misses the negative boundary (v_i = -0.35 passes the literal formula).
  Intent is distance of the component's *absolute value* from the threshold:
  `||v_i| - t| < 0.05`. Checked on the normalized frame-local vector.
- icosa32: ambiguous when the best and second-best cosine similarities are
  within ``ICOSA_MARGIN`` of each other — a shell of ≈2° around each Voronoi
  boundary, comparable in spirit to the cubic band.
"""
from __future__ import annotations

import itertools
import math

import numpy as np

from .errors import VersorFault

THRESHOLD = 0.35
DEAD_ZONE = 0.05

PHI = (1.0 + math.sqrt(5.0)) / 2.0
ICOSA_MARGIN = 0.01  # min gap between best & runner-up cosine similarity

ALL_TRIPLES = [t for t in itertools.product((-1, 0, 1), repeat=3)
               if t != (0, 0, 0)]


class Cubic26:
    """Snap each component of a unit vector to {-1, 0, +1} at threshold 0.35."""

    name = "cubic26"

    def decode(self, unit_v: np.ndarray) -> tuple[int, int, int]:
        s = []
        for c in unit_v:
            if abs(abs(c) - THRESHOLD) < DEAD_ZONE:
                raise VersorFault(
                    "AmbiguousDirection",
                    f"component {c:.4f} within dead zone of threshold "
                    f"±{THRESHOLD} (unit vector {np.round(unit_v, 4).tolist()})",
                )
            if c > THRESHOLD:
                s.append(1)
            elif c < -THRESHOLD:
                s.append(-1)
            else:
                s.append(0)
        triple = (s[0], s[1], s[2])
        if triple == (0, 0, 0):
            # Unreachable for true unit vectors (max component >= 1/sqrt(3)),
            # but guard against non-normalized input.
            raise VersorFault("AmbiguousDirection", "vector quantized to (0,0,0)")
        return triple

    def directions(self) -> dict[tuple[int, int, int], np.ndarray]:
        """Cone-center unit vector per opcode triple (authoring targets)."""
        return {t: np.array(t, dtype=float) / np.linalg.norm(t)
                for t in ALL_TRIPLES}


def _icosa_assignment() -> list[tuple[np.ndarray, tuple[int, int, int] | None]]:
    """The 32 icosahedral directions with their opcode triple (None = reserved)."""
    entries: list[tuple[np.ndarray, tuple[int, int, int] | None]] = []

    # 8 dodecahedron corner vertices: exact match with the cubic corners
    for sx, sy, sz in itertools.product((-1, 1), repeat=3):
        entries.append((np.array([sx, sy, sz], dtype=float), (sx, sy, sz)))

    # 12 icosahedron vertices <-> the 12 cubic edge triples (phi -> 1)
    for sy, sz in itertools.product((-1, 1), repeat=2):
        entries.append((np.array([0, sy, sz * PHI]), (0, sy, sz)))
    for sx, sy in itertools.product((-1, 1), repeat=2):
        entries.append((np.array([sx, sy * PHI, 0]), (sx, sy, 0)))
    for sx, sz in itertools.product((-1, 1), repeat=2):
        entries.append((np.array([sx * PHI, 0, sz]), (sx, 0, sz)))

    # 12 axis-heavy dodecahedron vertices: 6 carry the face opcodes
    # (minor-component sign matches the major one), 6 are reserved.
    for s in (-1, 1):
        entries.append((np.array([s * PHI, 0, s / PHI]), (s, 0, 0)))
        entries.append((np.array([s / PHI, s * PHI, 0]), (0, s, 0)))
        entries.append((np.array([0, s / PHI, s * PHI]), (0, 0, s)))
        entries.append((np.array([s * PHI, 0, -s / PHI]), None))
        entries.append((np.array([-s / PHI, s * PHI, 0]), None))
        entries.append((np.array([0, s / PHI, -s * PHI]), None))

    return [(v / np.linalg.norm(v), triple) for v, triple in entries]


class Icosa32:
    """Nearest-neighbor over the 32 icosahedral directions (M6)."""

    name = "icosa32"

    def __init__(self):
        entries = _icosa_assignment()
        self._matrix = np.array([v for v, _ in entries])   # 32 x 3
        self._triples = [t for _, t in entries]

    def decode(self, unit_v: np.ndarray) -> tuple[int, int, int]:
        dots = self._matrix @ np.asarray(unit_v, dtype=float)
        order = np.argsort(dots)
        best, second = int(order[-1]), int(order[-2])
        if dots[best] - dots[second] < ICOSA_MARGIN:
            raise VersorFault(
                "AmbiguousDirection",
                f"within {ICOSA_MARGIN} of an icosa32 cone boundary "
                f"(unit vector {np.round(unit_v, 4).tolist()})",
            )
        triple = self._triples[best]
        if triple is None:
            raise VersorFault(
                "ReservedDirection",
                f"direction {np.round(self._matrix[best], 4).tolist()} is one "
                "of the 6 unassigned icosa32 cones",
            )
        return triple

    def directions(self) -> dict[tuple[int, int, int], np.ndarray]:
        return {t: v.copy() for v, t in zip(self._matrix, self._triples)
                if t is not None}


DECODERS = {"cubic26": Cubic26, "icosa32": Icosa32}


def get_decoder(name: str):
    try:
        return DECODERS[name]()
    except KeyError:
        raise ValueError(f"unknown decoder {name!r}; available: {sorted(DECODERS)}")
