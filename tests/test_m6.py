"""M6: icosahedral decoder + program interpolation."""
import itertools
import math

import numpy as np
import pytest

from versor import Machine, VersorFault, classify, from_dict, lerp_programs, to_dict
from versor.decode import PHI, Cubic26, Icosa32, get_decoder
from versor.examples import countdown, countdown_b, straightline

CORNERS = [t for t in itertools.product((-1, 1), repeat=3)]
EDGES = [t for t in itertools.product((-1, 0, 1), repeat=3)
         if sum(abs(c) for c in t) == 2]
FACES = [t for t in itertools.product((-1, 0, 1), repeat=3)
         if sum(abs(c) for c in t) == 1]


def unit(v):
    v = np.asarray(v, dtype=float)
    return v / np.linalg.norm(v)


class TestIcosa32:
    def test_has_26_base_and_6_extended_directions(self):
        dec = Icosa32()
        dirs = dec.directions()
        assert len(dirs) == 32
        assert len(dec._triples) == 32
        base = [t for t in dec._triples if isinstance(t, tuple)]
        ext = [t for t in dec._triples if isinstance(t, str)]
        assert len(base) == 26
        assert sorted(ext) == ["INP", "LOADP", "MULR", "POPA", "PUSHA", "SWAP"]

    def test_assigned_directions_roundtrip(self):
        dec = Icosa32()
        for triple, vec in dec.directions().items():
            assert dec.decode(vec) == triple

    def test_corner_and_edge_cubic_directions_agree_across_decoders(self):
        cubic, icosa = Cubic26(), Icosa32()
        for triple in CORNERS + EDGES:
            v = unit(triple)
            assert icosa.decode(v) == cubic.decode(v) == triple

    def test_cubic_face_directions_are_ambiguous_under_icosa(self):
        # each cube face axis ties exactly between two axis-heavy
        # dodecahedron vertices — a Voronoi boundary
        icosa = Icosa32()
        for triple in FACES:
            with pytest.raises(VersorFault) as e:
                icosa.decode(unit(triple))
            assert e.value.kind == "AmbiguousDirection"

    def test_formerly_reserved_directions_carry_extended_opcodes(self):
        icosa = Icosa32()
        assert icosa.decode(unit((PHI, -1 / PHI, 0))) == "INP"
        assert icosa.decode(unit((-PHI, 1 / PHI, 0))) == "SWAP"
        assert icosa.decode(unit((0, PHI, -1 / PHI))) == "PUSHA"
        assert icosa.decode(unit((0, -PHI, 1 / PHI))) == "POPA"
        assert icosa.decode(unit((-1 / PHI, 0, PHI))) == "MULR"
        assert icosa.decode(unit((1 / PHI, 0, -PHI))) == "LOADP"

    def test_direction_set_is_a_true_dual_pair(self):
        # regression for the mixed-orientation bug: the 20 dodecahedral
        # directions must be the face normals of the 12-vertex icosahedron
        # actually used, giving minimum pairwise separation 37.38 degrees
        # (the mismatched mirror family collapses it to 10.81)
        m = Icosa32()._matrix
        dots = np.clip(m @ m.T, -1, 1)
        np.fill_diagonal(dots, -1)
        min_angle = math.degrees(math.acos(float(dots.max())))
        assert min_angle > 37.0

    def test_registry(self):
        assert get_decoder("icosa32").name == "icosa32"

    def test_countdown_runs_identically_under_icosa(self):
        prog = countdown(5, decoder="icosa32").build()
        assert prog.decoder == "icosa32"
        assert prog.warnings == []
        res = Machine(prog).run()
        assert res.out == pytest.approx([5.0, 4.0, 3.0, 2.0, 1.0])

    def test_decoder_field_survives_serialization(self):
        prog = countdown(3, decoder="icosa32").build()
        reloaded = from_dict(to_dict(prog))
        assert reloaded.decoder == "icosa32"
        assert Machine(reloaded).run().out == pytest.approx([3.0, 2.0, 1.0])

    def test_cubic_program_faults_under_icosa_override(self):
        # cubic face ops (LOADI etc.) sit on icosa32 Voronoi ties
        prog = countdown(5).build()
        with pytest.raises(VersorFault) as e:
            Machine(prog, decoder="icosa32").run()
        assert e.value.kind == "AmbiguousDirection"


@pytest.fixture(scope="module")
def ab():
    a = countdown(5).build()
    b = countdown_b(5).build()
    expected = Machine(countdown(5).build()).run().out
    return a, b, expected


class TestInterpolation:
    def test_endpoints_are_equivalent(self, ab):
        a, b, expected = ab
        for t in (0.0, 1.0):
            r = classify(lerp_programs(a, b, t), expected)
            assert r["status"] == "equivalent"

    def test_dead_zone_band_faults(self, ab):
        a, b, expected = ab
        r = classify(lerp_programs(a, b, 0.25), expected)
        assert r["status"] == "fault"
        assert r["detail"] == "AmbiguousDirection"

    def test_midpoint_mutates_nop_into_proj_but_stays_equivalent(self, ab):
        a, b, expected = ab
        r = classify(lerp_programs(a, b, 0.5), expected)
        assert r["status"] == "equivalent"
        assert "PROJ" in r["opcodes"]
        assert "NOP" not in r["opcodes"]

    def test_guards_and_targets_preserved(self, ab):
        a, b, _ = ab
        mid = lerp_programs(a, b, 0.5)
        va, vm = a.chains[0].vertices, mid.chains[0].vertices
        for vid in va:
            for ea, em in zip(va[vid], vm[vid]):
                assert ea.to == em.to
                if ea.guard is not None:
                    assert np.allclose(ea.guard, em.guard)

    def test_topology_mismatch_rejected(self, ab):
        a, _, _ = ab
        with pytest.raises(ValueError):
            lerp_programs(a, straightline().build(), 0.5)

    def test_decoder_mismatch_rejected(self):
        with pytest.raises(ValueError, match="decoder"):
            lerp_programs(countdown(5).build(),
                          countdown(5, decoder="icosa32").build(), 0.5)