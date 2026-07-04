import math

import numpy as np
import pytest

from versor import Machine, ProgramBuilder, VersorFault, arm
from versor.quat import Quat


def run(builder, **kw):
    m = Machine(builder.build(), **kw)
    m.run()
    return m


def one_chain(*ops):
    """Build a single straight chain: ops are (method_name, *args) tuples."""
    b = ProgramBuilder("t")
    c = b.chain()
    for name, *args in ops:
        getattr(c, name)(*args)
    c.halt()
    return b


class TestData:
    def test_loadi(self):
        m = run(one_chain(("loadi", 5)))
        assert np.allclose(m.A, [5, 0, 0])

    def test_loadi_under_rotated_frame_is_frame_local(self):
        q = Quat.axis_angle([0, 0, 1], math.pi / 2)
        b = ProgramBuilder("t")
        c = b.chain()
        # author under the rotated frame so segments decode correctly
        c._Fa = q
        c.loadi(5).halt()
        m = run(b, F0=q)
        assert np.allclose(m.A, [0, 5, 0])  # frame-x is world +y

    def test_movr_mova_roundtrip_and_index(self):
        m = run(one_chain(("loadi", 4), ("movr", 2), ("loadi", 1), ("mova", 2)))
        assert np.allclose(m.A, [4, 0, 0])
        assert np.allclose(m.R[2], [4, 0, 0])
        assert np.allclose(m.R[0], 0)

    def test_store_load_hit_arrival_cell(self):
        m = run(one_chain(("loadi", 3), ("store", 0.5)))
        # LOADI lands at (3,0,0); STORE moves to (2.5,0,0) -> cell (2,0,0)
        assert np.allclose(m.M[(2, 0, 0)], [3, 0, 0])

    def test_out_float_and_char(self):
        m = run(one_chain(("loadi", 65), ("outc",), ("out",)))
        assert m.OUT == ["A", pytest.approx(65.0)]


class TestArithmetic:
    def test_add_sub(self):
        m = run(one_chain(("loadi", 2), ("movr", 1), ("loadi", 7),
                          ("add", 1), ("sub", 1), ("sub", 1)))
        assert np.allclose(m.A, [5, 0, 0])

    def test_scale(self):
        m = run(one_chain(("loadi", 5), ("scale", 2)))
        assert np.allclose(m.A, [10, 0, 0])

    def test_dot_writes_frame_local_x_slot(self):
        m = run(one_chain(("loadi", 3), ("movr", 0), ("loadi", 4), ("dot", 0)))
        assert np.allclose(m.A, [12, 0, 0])

    def test_cross(self):
        b = ProgramBuilder("t")
        c = b.chain()
        c.loadi(2).movr(0)              # R0 = world (2,0,0)
        c.roth(math.pi / 2)             # frame-x now world +y
        c.loadi(3)                      # A = world (0,3,0)
        c.cross(0)                      # (0,3,0) x (2,0,0) = (0,0,-6)
        c.halt()
        m = run(b)
        assert np.allclose(m.A, [0, 0, -6])

    def test_norm_and_zero_fault(self):
        m = run(one_chain(("loadi", 5), ("norm",)))
        assert np.allclose(m.A, [1, 0, 0])
        with pytest.raises(VersorFault) as e:
            run(one_chain(("norm",)))  # A starts at zero
        assert e.value.kind == "DivisionByZero"

    def test_proj_rej(self):
        # handler-level math check on a fresh machine
        from versor.isa import _proj_vec, _rej
        m = run(one_chain())
        m.A = np.array([3.0, 4.0, 0.0])
        m.R[0] = np.array([2.0, 0.0, 0.0])
        assert np.allclose(_proj_vec(m, 0.5), [3, 0, 0])
        _rej(m, 0.5)
        assert np.allclose(m.A, [0, 4, 0])
        with pytest.raises(VersorFault):
            m.R[1] = np.zeros(3)
            _proj_vec(m, 1.5)


class TestFrame:
    def test_rot_non_commutative(self):
        ma = run(one_chain(("rotf", math.pi / 2), ("rotg", math.pi / 2)))
        mb = run(one_chain(("rotg", math.pi / 2), ("rotf", math.pi / 2)))
        v = np.array([0.0, 0.0, 1.0])
        assert not np.allclose(ma.F.rotate(v), mb.F.rotate(v))

    def test_pushf_popf_restores_frame_not_position(self):
        b = ProgramBuilder("t")
        c = b.chain()
        c.pushf().roth(math.pi / 2).popf()
        c._Fa = Quat.identity()  # runtime frame is restored; re-author identity
        c.halt()
        m = run(b)
        assert m.F.approx(Quat.identity())
        assert not np.allclose(m.P, 0)  # position kept

    def test_popf_empty_faults(self):
        with pytest.raises(VersorFault) as e:
            run(one_chain(("popf",)))
        assert e.value.kind == "StackUnderflow"


class TestControl:
    def test_jmpz_skips_but_moves(self):
        m = run(one_chain(("jmpz",), ("loadi", 5)))  # A=0 -> skip LOADI
        assert np.allclose(m.A, 0)
        # the skipped LOADI segment still moved the machine +5 in x
        assert m.P[0] > 5

    def test_jmpp_skips_on_positive_local_x(self):
        m = run(one_chain(("loadi", 1), ("jmpp",), ("loadi", 9)))
        assert np.allclose(m.A, [1, 0, 0])

    def test_jmpp_no_skip_on_zero(self):
        m = run(one_chain(("loadi", 1), ("movr", 0), ("sub", 0),
                          ("jmpp",), ("loadi", 9)))
        assert np.allclose(m.A, [9, 0, 0])

    def test_fault_opcode(self):
        with pytest.raises(VersorFault) as e:
            run(one_chain(("fault", 42)))
        assert e.value.kind == "ExplicitFault"
        assert "42" in e.value.message

    def test_call_ret_displacement_and_frame_restore(self):
        b = ProgramBuilder("t")
        c0 = b.chain("entry")
        c0.call(1).halt()
        c1 = b.chain("callee")
        c1.roth(math.pi / 2)  # frame change must not leak to caller
        c1.op("NOP", 2.0)
        c1.ret(1)
        m = run(b)
        prog = b.build()
        # returned A = callee net displacement: ROTH seg + NOP seg + RET seg
        callee = prog.chains[1]
        expected = sum((e.seg for edges in callee.vertices.values() for e in edges),
                       np.zeros(3))
        assert np.allclose(m.A, expected)
        assert m.F.approx(Quat.identity())

    def test_implicit_ret_at_chain_end(self):
        b = ProgramBuilder("t")
        b.chain("entry").call(1).halt()
        b.chain("callee").op("NOP", 3.0)  # no explicit RET
        m = run(b)
        nop_seg = b.build().chains[1].vertices[0][0].seg
        assert np.allclose(m.A, nop_seg)

    def test_ret_in_root_faults(self):
        with pytest.raises(VersorFault) as e:
            run(one_chain(("ret",)))
        assert e.value.kind == "StackUnderflow"

    def test_call_stack_overflow(self):
        b = ProgramBuilder("t")
        b.chain("entry").call(1).halt()
        c1 = b.chain("recurse")
        c1.call(1).halt()
        with pytest.raises(VersorFault) as e:
            run(b, max_call_depth=8)
        assert e.value.kind == "CallStackOverflow"

    def test_step_budget(self):
        b = ProgramBuilder("t")
        c = b.chain()
        c.label("a").op("NOP", 1.0, to="a")
        with pytest.raises(VersorFault) as e:
            run(b, step_budget=100)
        assert e.value.kind == "StepBudgetExhausted"

    def test_runtime_ambiguous_direction_under_rotated_frame(self):
        b = ProgramBuilder("t")
        c = b.chain()
        c.roth(0.36)             # sin(0.36) ~ 0.352: inside the dead zone
        c.seg_raw((1, 0, 0))     # decodes fine under identity, faults under F
        c.halt()
        prog = b.build()
        assert not prog.warnings  # load-time lint cannot see this
        with pytest.raises(VersorFault) as e:
            Machine(prog).run()
        assert e.value.kind == "AmbiguousDirection"

    def test_branch_zero_accumulator_takes_first_edge(self):
        b = ProgramBuilder("t")
        c = b.chain()
        c.branch(
            arm("LOADI", 7.0, guard=(0, 1, 0), to="first"),
            arm("LOADI", 9.0, guard=(0, -1, 0), to="second"),
        )
        c.at("first").halt()
        c.at("second").halt()
        m = run(b)  # A starts (0,0,0): tie, first listed wins
        assert np.allclose(m.A, [7, 0, 0])
