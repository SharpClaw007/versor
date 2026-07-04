"""Direction quantization: unit vector -> opcode sign triple.

v0.1 ships cubic-26 (6 face + 12 edge + 8 corner directions). The decoder is
pluggable; an icosahedral decoder is a planned v0.2 upgrade.

Dead-zone rule (spec 3.3, corrected): the spec writes `|v_i - t| < 0.05`,
which misses the negative boundary (v_i = -0.35 passes the literal formula).
Intent is distance of the component's *absolute value* from the threshold:
`||v_i| - t| < 0.05`. Checked on the normalized frame-local vector.
"""
from __future__ import annotations

import numpy as np

from .errors import VersorFault

THRESHOLD = 0.35
DEAD_ZONE = 0.05


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


DECODERS = {"cubic26": Cubic26}


def get_decoder(name: str):
    try:
        return DECODERS[name]()
    except KeyError:
        raise ValueError(f"unknown decoder {name!r}; available: {sorted(DECODERS)}")
