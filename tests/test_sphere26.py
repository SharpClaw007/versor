import itertools
import math

import numpy as np
import pytest

from versor import Machine, VersorFault
from versor.decode import Sphere26, get_decoder
from versor.examples import countdown

ALL_26 = [t for t in itertools.product((-1, 0, 1), repeat=3) if t != (0, 0, 0)]


def test_covers_all_26_opcodes_with_no_reserved():
    dec = Sphere26()
    dirs = dec.directions()
    assert set(dirs) == set(ALL_26)
    assert all(t is not None for t in dec._triples)


def test_cone_centers_roundtrip():
    dec = Sphere26()
    for triple, vec in dec.directions().items():
        assert dec.decode(vec) == triple


def test_antipodal_consistency():
    dec = Sphere26()
    for triple, vec in dec.directions().items():
        assert dec.decode(-vec) == tuple(-c for c in triple)


def test_min_pairwise_separation_beats_cubic():
    m = Sphere26()._matrix
    dots = np.clip(m @ m.T, -1, 1)
    np.fill_diagonal(dots, -1)
    min_angle = math.degrees(math.acos(float(dots.max())))
    assert min_angle > 38.0  # cubic-26 minimum is 35.26

    unit_norms = np.linalg.norm(m, axis=1)
    assert np.allclose(unit_norms, 1.0, atol=1e-12)


def test_boundary_faults():
    dec = Sphere26()
    a = dec._matrix[0]
    b = dec._matrix[2]  # rows 0/1 are antipodes; pick a distinct line
    mid = (a + b) / np.linalg.norm(a + b)
    # a midpoint between two centers is either ambiguous or decodes to a
    # third cone; on the wall between the two it must fault
    dots = sorted(dec._matrix @ mid)[-2:]
    if dots[1] - dots[0] < dec.margin:
        with pytest.raises(VersorFault):
            dec.decode(mid)


def test_countdown_runs_under_sphere26():
    prog = countdown(5, decoder="sphere26").build()
    assert prog.decoder == "sphere26"
    assert prog.warnings == []
    res = Machine(prog).run()
    assert res.out == pytest.approx([5.0, 4.0, 3.0, 2.0, 1.0])


def test_registry():
    assert get_decoder("sphere26").name == "sphere26"
