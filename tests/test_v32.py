"""v0.4 Versor-32: extended opcodes on icosa32's formerly reserved cones."""
import numpy as np
import pytest

from versor import LoadError, Machine, ProgramBuilder, VersorFault
from versor.asm import assemble


def build(fn, decoder="icosa32"):
    b = ProgramBuilder("t", decoder=decoder)
    fn(b.chain())
    return b.build()


def run(fn, decoder="icosa32", **kw):
    m = Machine(build(fn, decoder), **kw)
    m.run()
    return m


class TestInp:
    def test_reads_scalars_in_order(self):
        m = run(lambda c: c.inp().out().inp().out().halt(), input=[7, 9])
        assert m.OUT == pytest.approx([7.0, 9.0])

    def test_string_input_is_char_codes(self):
        m = run(lambda c: c.inp().outc().halt(), input="A")
        assert m.OUT == ["A"]

    def test_exhausted_faults(self):
        with pytest.raises(VersorFault) as e:
            run(lambda c: c.inp().halt(), input=[])
        assert e.value.kind == "InputExhausted"

    def test_cli_style_echo_program(self):
        # echo three inputs: INP/OUT in a loop would need a counter; keep
        # it straight-line, the loop machinery is already covered elsewhere
        m = run(lambda c: c.inp().out().inp().out().inp().out().halt(),
                input=[1, 2, 3])
        assert m.OUT == pytest.approx([1.0, 2.0, 3.0])


class TestSwap:
    def test_swaps_accumulator_and_register(self):
        def prog(c):
            c.loadi(3).movr(1).loadi(8).swap(1).halt()
        m = run(prog)
        assert np.allclose(m.R[1], [8, 0, 0])
        # A picked up R1's old value (3 along x)... after the swap segment's
        # own movement, A is still the vector (3,0,0)
        assert np.allclose(m.A, [3, 0, 0])


class TestDataStack:
    def test_pusha_popa_roundtrip(self):
        def prog(c):
            c.loadi(5).pusha().loadi(9).pusha().loadi(1).popa().out() \
             .popa().out().halt()
        m = run(prog)
        assert m.OUT == pytest.approx([9.0, 5.0])  # LIFO

    def test_popa_empty_faults(self):
        with pytest.raises(VersorFault) as e:
            run(lambda c: c.popa().halt())
        assert e.value.kind == "StackUnderflow"


class TestMulr:
    def test_variable_times_variable(self):
        def prog(c):
            c.loadi(6).movr(1).loadi(7).mulr(1).out().halt()
        m = run(prog)
        assert m.OUT == pytest.approx([42.0])

    def test_reads_frame_local_slot(self):
        def prog(c):
            c.loadi(4).movr(1)
            c.roth(np.pi / 2)      # R1 stays world (4,0,0)
            c.loadi(3).mulr(1)     # frame-local x of R1 is now ~0
            c.out().halt()
        m = run(prog)
        assert m.OUT[0] == pytest.approx(0.0, abs=1e-9)


class TestLoadp:
    def test_reads_position(self):
        def prog(c):
            c.loadi(5).loadp().halt()
        m = run(prog)
        # A was set to P right after LOADP's own move; HALT moved once more
        halt_seg = build(prog).chains[0].vertices[2][0].seg
        assert np.allclose(m.A, m.P - halt_seg)


class TestAvailability:
    def test_cubic26_rejects_extended_ops(self):
        with pytest.raises(LoadError, match="Versor-32"):
            build(lambda c: c.inp().halt(), decoder="cubic26")

    def test_asm_extended_ops_under_icosa32(self):
        src = """
.decoder icosa32
        INP
        MOVR r1
        INP
        MULR r1
        OUT
        HALT
"""
        m = Machine(assemble(src).build(), input=[6, 7])
        m.run()
        assert m.OUT == pytest.approx([42.0])

    def test_asm_extended_ops_under_cubic26_error(self):
        with pytest.raises(LoadError, match="Versor-32"):
            assemble("INP\nHALT\n")
