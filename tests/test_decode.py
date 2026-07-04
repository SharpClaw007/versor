import itertools
import math

import numpy as np
import pytest

from versor.decode import Cubic26, get_decoder
from versor.errors import VersorFault

ALL_26 = [t for t in itertools.product((-1, 0, 1), repeat=3) if t != (0, 0, 0)]


def test_all_26_canonical_directions_roundtrip():
    dec = Cubic26()
    for triple in ALL_26:
        v = np.array(triple, dtype=float)
        assert dec.decode(v / np.linalg.norm(v)) == triple


def test_dead_zone_exact_threshold_faults():
    dec = Cubic26()
    v = np.array([0.35, math.sqrt(1 - 0.35 ** 2), 0.0])
    with pytest.raises(VersorFault) as e:
        dec.decode(v)
    assert e.value.kind == "AmbiguousDirection"


def test_dead_zone_negative_boundary_faults():
    # the spec's literal formula |v_i - t| misses this side; corrected decoder
    # must fault on components near -0.35 too
    dec = Cubic26()
    v = np.array([-0.36, math.sqrt(1 - 0.36 ** 2), 0.0])
    with pytest.raises(VersorFault):
        dec.decode(v)


def test_just_outside_dead_zone_decodes():
    dec = Cubic26()
    v = np.array([0.29, math.sqrt(1 - 0.29 ** 2), 0.0])
    assert dec.decode(v) == (0, 1, 0)
    v = np.array([0.41, math.sqrt(1 - 0.41 ** 2), 0.0])
    assert dec.decode(v) == (1, 1, 0)


def test_all_zero_triple_guard():
    dec = Cubic26()
    with pytest.raises(VersorFault):
        dec.decode(np.array([0.2, 0.2, 0.2]))  # deliberately non-unit


def test_decoder_registry():
    assert get_decoder("cubic26").name == "cubic26"
    with pytest.raises(ValueError):
        get_decoder("icosa")  # v0.2
