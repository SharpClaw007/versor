"""v0.3b: Sim(3) scale channel — scale rides CALL's fractional magnitude."""
import numpy as np
import pytest

from versor import LoadError, Machine, ProgramBuilder, VersorFault
from versor.asm import assemble
from versor.examples import zoom
from versor.isa import DIRECTIONS


def run(builder, **kw):
    m = Machine(builder.build(), **kw)
    res = m.run()
    return m, res


class TestBackwardCompatibility:
    def test_default_call_factor_is_exactly_one(self):
        b = ProgramBuilder("t")
        b.chain().call(1).halt()
        b.chain("callee").op("NOP", 2.0)
        m, _ = run(b)
        assert m.S == 1.0
        assert np.allclose(m.A, DIRECTIONS["NOP"] * 2.0)


class TestScaledCalls:
    def test_callee_displacement_scales(self):
        b = ProgramBuilder("t")
        b.chain().call(1, scale=0.5).halt()
        b.chain("callee").op("NOP", 4.0)
        m, _ = run(b)
        assert np.allclose(m.A, DIRECTIONS["NOP"] * 4.0 * 0.5)

    def test_scale_composes_through_nesting(self):
        b = ProgramBuilder("t")
        b.chain().call(1, scale=0.5).halt()
        b.chain("mid").call(2, scale=0.5)
        b.chain("leaf").op("NOP", 8.0)
        m, _ = run(b)
        # mid's A after leaf returns: 8 * 0.25 = 2 along NOP's direction;
        # the entry's A is mid's displacement: leaf's plus mid's CALL segment
        leaf_move = DIRECTIONS["NOP"] * 8.0 * 0.25
        # mid's CALL: chain 2 at scale 0.5 -> frac 0 -> magnitude 2.0
        call_seg = DIRECTIONS["CALL"] * 2.0 * 0.5
        assert np.allclose(m.A, leaf_move + call_seg)

    def test_ret_restores_scale(self):
        b = ProgramBuilder("t")
        b.chain().call(1, scale=0.75).op("NOP", 2.0).halt()
        b.chain("callee").op("NOP", 2.0)
        m, _ = run(b)
        assert m.S == 1.0

    def test_popf_restores_scale(self):
        b = ProgramBuilder("t")
        c0 = b.chain()
        c0.pushf().call(1, scale=0.5).halt()
        c1 = b.chain("callee")
        c1.op("NOP", 2.0)   # moves 1.0 at s=0.5
        c1.popf()           # restores the caller's pushed (F, s=1)
        c1.op("NOP", 2.0)   # moves 2.0 at s=1
        m, _ = run(b)
        nop = DIRECTIONS["NOP"]
        popf_move = DIRECTIONS["POPF"] * 1.0 * 0.5  # POPF's own move, pre-restore
        assert np.allclose(m.A, nop * 1.0 + popf_move + nop * 2.0)

    def test_scale_overflow_faults(self):
        b = ProgramBuilder("t")
        b.chain().call(0, scale=0.55)  # entry calls itself, shrinking
        with pytest.raises(VersorFault) as e:
            run(b)
        assert e.value.kind == "ScaleOverflow"


class TestFrontEnds:
    def test_builder_rejects_out_of_range_scale(self):
        b = ProgramBuilder("t")
        with pytest.raises(LoadError, match="compose"):
            b.chain().call(1, scale=3.0)

    def test_asm_call_scale(self):
        src = """
.chain main
        CALL fn 0.5
        HALT
.chain fn
        NOP 4
"""
        m = Machine(assemble(src).build())
        m.run()
        assert np.allclose(m.A, DIRECTIONS["NOP"] * 2.0)

    def test_asm_scale_range_error(self):
        with pytest.raises(LoadError, match="compose"):
            assemble(".chain main\nCALL fn 2.5\nHALT\n.chain fn\nNOP 1\n")


class TestZoom:
    def test_runs_and_restores_scale(self):
        m, res = run(zoom(6))
        assert res.halt_reason == "HALT"
        assert m.S == 1.0

    def test_levels_shrink_geometrically(self):
        from versor import Trace
        trace = Trace()
        Machine(zoom(4).build(), trace=trace).run()
        # the first NOP of each level moves 2 * 0.55^k
        moves = [np.linalg.norm(r.P1 - r.P0) for r in trace
                 if r.opcode == "NOP"]
        # depth-first: the first NOP of each level comes first, descending
        for k in range(1, 4):
            assert moves[k] == pytest.approx(moves[0] * 0.55 ** k)
