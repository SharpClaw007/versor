"""Direction quantization: unit vector -> opcode sign triple.

Four decoders ship:

- ``cubic26`` (v0.1 default): snap components to {-1, 0, +1} at threshold
  0.35. Base-26 ISA only.
- ``icosa32`` (M6): nearest-neighbor over the 32 directions of icosahedral
  symmetry (12 icosahedron vertices + 20 face normals). Since v0.4 the six
  formerly reserved cones carry the extended Versor-32 opcodes
  (INP/SWAP, PUSHA/POPA, MULR/LOADP), keyed by name rather than triple.
- ``sphere26``: optimized antipodal packing of the 26 base directions
  (tools/optimize_sphere26.py).
- ``sphere32``: optimized antipodal packing of the full 32-key Versor-32
  ISA (tools/optimize_sphere32.py).

Icosahedral symmetry has orbit sizes 12/20/30 — there is no 26-direction
icosahedral set, so icosa32 keeps the 26-opcode ISA by *assigning* each opcode
a cone and leaving 6 directions reserved (decoding into one is a
``ReservedDirection`` fault, parked for a future ISA extension):

- the 8 corner triples map exactly: (±1,±1,±1)/√3 ARE dodecahedron vertices;
- the 12 edge triples map to the icosahedron vertex with the same sign
  pattern (φ in place of one 1), 13.3° away;
- the 6 face triples map to the axis-heavy dodecahedron vertex with matching
  signs (e.g. +x -> (φ, 1/φ, 0)/√3), 20.9° away — its mirror twin
  (φ, -1/φ, 0) is one of the 6 reserved directions.

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
    # (minor-component sign matches the major one), 6 carry the extended
    # "Versor-32" opcodes (v0.4) on the mirror twins — three antipodal
    # pairs: INP/SWAP, PUSHA/POPA, MULR/LOADP.
    # These are cyclic permutations of (phi, 1/phi, 0) — the TRUE dual of the
    # icosahedron orientation above (each is the centroid of three mutually
    # adjacent icosahedron vertices). The mirrored family (phi, 0, 1/phi)
    # belongs to a rotated dodecahedron; using it mixes two incompatible
    # polyhedron orientations and collapses the minimum pairwise separation
    # of the 32 directions from 37.38 to 10.81 degrees.
    for s in (-1, 1):
        entries.append((np.array([s * PHI, s / PHI, 0]), (s, 0, 0)))
        entries.append((np.array([0, s * PHI, s / PHI]), (0, s, 0)))
        entries.append((np.array([s / PHI, 0, s * PHI]), (0, 0, s)))
        entries.append((np.array([s * PHI, -s / PHI, 0]),
                        "INP" if s > 0 else "SWAP"))
        entries.append((np.array([0, s * PHI, -s / PHI]),
                        "PUSHA" if s > 0 else "POPA"))
        entries.append((np.array([-s / PHI, 0, s * PHI]),
                        "MULR" if s > 0 else "LOADP"))

    return [(v / np.linalg.norm(v), triple) for v, triple in entries]


class _NearestNeighbor:
    """Nearest-neighbor decoding over a fixed direction set with a relative
    cosine-gap ambiguity margin. Subclasses set name and provide entries.

    Opcode keys are sign triples for the base-26 ISA and short strings for
    the extended "Versor-32" opcodes (v0.4); None marks a reserved cone."""

    name = "abstract"
    margin = ICOSA_MARGIN

    def __init__(self, entries):
        self._matrix = np.array([v for v, _ in entries])
        self._triples = [t for _, t in entries]

    def decode(self, unit_v: np.ndarray):
        dots = self._matrix @ np.asarray(unit_v, dtype=float)
        order = np.argsort(dots)
        best, second = int(order[-1]), int(order[-2])
        if dots[best] - dots[second] < self.margin:
            raise VersorFault(
                "AmbiguousDirection",
                f"within {self.margin} of a {self.name} cone boundary "
                f"(unit vector {np.round(unit_v, 4).tolist()})",
            )
        triple = self._triples[best]
        if triple is None:
            raise VersorFault(
                "ReservedDirection",
                f"direction {np.round(self._matrix[best], 4).tolist()} is a "
                f"reserved {self.name} cone",
            )
        return triple

    def directions(self) -> dict:
        return {t: v.copy() for v, t in zip(self._matrix, self._triples)
                if t is not None}


class Icosa32(_NearestNeighbor):
    """Nearest-neighbor over the 32 icosahedral directions (M6)."""

    name = "icosa32"

    def __init__(self):
        super().__init__(_icosa_assignment())


# Antipodal 13-line packing found by tools/optimize_sphere26.py (Riesz s=12,
# annealed, 40 random restarts + cubic seed all converge here): minimum
# pairwise separation 38.17 deg vs cubic-26's 35.26, with near-uniform
# margins and no reserved directions. Labels are matched to the nearest
# cubic triples (max drift ~33 deg) but the assignment is semantics-neutral.
_SPHERE26_TABLE = [
    ((-0.600301876388214, -0.598848634433877, +0.530111280998122), (-1, -1, 1)),
    ((-0.446406946572806, +0.184324847647420, -0.875639873801610), (-1, 0, -1)),
    ((-0.869932147354444, +0.045068511287929, +0.491107817377789), (-1, 0, 1)),
    ((-0.691661090487099, -0.713721202769234, -0.110485205452290), (-1, -1, 0)),
    ((+0.049935126077074, -0.699471611273205, +0.712913703197331), (0, -1, 1)),
    ((-0.747469029906257, +0.558081339546148, -0.360326612646605), (-1, 1, -1)),
    ((-0.223030638816671, +0.964599211521904, -0.140732708637151), (-1, 1, 0)),
    ((+0.199787421253120, -0.492401221198910, -0.847128103459477), (0, -1, -1)),
    ((-0.647304747219036, +0.683174579798115, +0.338037065638325), (-1, 1, 1)),
    ((+0.290837759716615, +0.152210094231070, -0.944587468018282), (0, 0, -1)),
    ((-0.971777467495341, -0.082639230854734, -0.220950924850019), (-1, 0, 0)),
    ((-0.111562925436732, -0.901734944353629, -0.417645548042304), (0, -1, 0)),
    ((-0.528894555907088, -0.449926282823061, -0.719608844273656), (-1, -1, -1)),
]


class Sphere26(_NearestNeighbor):
    """Optimized antipodal packing: all 26 opcodes, near-uniform margins."""

    name = "sphere26"

    def __init__(self):
        entries = []
        for v, t in _SPHERE26_TABLE:
            vec = np.array(v)
            entries.append((vec, t))
            entries.append((-vec, tuple(-c for c in t)))
        super().__init__(entries)


# Versor-32 antipodal 16-line packing (tools/optimize_sphere32.py; seeded
# and random starts converge to 37.38 deg minimum separation). String keys
# are extended opcodes; each line's antipode carries the partner opcode
# (or the negated triple).
_EXT_PARTNER = {"INP": "SWAP", "PUSHA": "POPA", "MULR": "LOADP"}
_SPHERE32_TABLE = [
    ((+0.144541519227640, +0.880028878315152, -0.452390232598739), "PUSHA"),
    ((-0.980834676069467, -0.182230567689861, +0.068959106861502), (-1, 0, 0)),
    ((+0.880854070848405, -0.443264921404343, -0.166169537888849), "INP"),
    ((-0.019885070504030, -0.985709983869711, -0.167273463737148), (0, -1, 0)),
    ((+0.351317087629498, -0.584852352854011, +0.731111502645564), (0, -1, 1)),
    ((-0.728894994596138, +0.089842846944058, +0.678704906204222), (-1, 0, 1)),
    ((-0.844633354881627, +0.284324514537852, -0.453601219414391), (-1, 1, -1)),
    ((-0.436988436068475, +0.724542485435285, +0.532990894426620), (-1, 1, 1)),
    ((-0.212034862061437, -0.341420288051529, +0.915681933958183), "MULR"),
    ((-0.676013506817563, -0.527392762459030, +0.514648047413039), (-1, -1, 1)),
    ((+0.180790694289701, -0.298307937362607, -0.937191068760643), (0, 0, -1)),
    ((-0.781407672422826, -0.350298099364822, -0.516423557809028), (-1, -1, -1)),
    ((-0.508516760084874, +0.844740010813678, -0.166808329657992), (-1, 1, 0)),
    ((-0.260282707989636, -0.685544396102580, -0.679913077454707), (0, -1, -1)),
    ((-0.399306876852523, -0.026742125301063, -0.916427234881563), (-1, 0, -1)),
    ((-0.619686155068403, -0.781639867525860, -0.070909708155471), (-1, -1, 0)),
]


class Sphere32(_NearestNeighbor):
    """Optimized antipodal packing carrying the full Versor-32 ISA."""

    name = "sphere32"

    def __init__(self):
        entries = []
        for v, k in _SPHERE32_TABLE:
            vec = np.array(v)
            partner = (_EXT_PARTNER[k] if isinstance(k, str)
                       else tuple(-c for c in k))
            entries.append((vec, k))
            entries.append((-vec, partner))
        super().__init__(entries)


DECODERS = {"cubic26": Cubic26, "icosa32": Icosa32, "sphere26": Sphere26,
            "sphere32": Sphere32}


def get_decoder(name: str):
    try:
        return DECODERS[name]()
    except KeyError:
        raise ValueError(f"unknown decoder {name!r}; available: {sorted(DECODERS)}")
