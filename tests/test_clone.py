"""Orientation-specialized chain cloning (versor/clone.py)."""
import math

import numpy as np
import pytest

from versor import Machine, ProgramBuilder, arm
from versor.clone import SpecializeError, specialize
from versor.examples import add_two, countdown, levy, zoom


def run(prog, **kw):
    m = Machine(prog, **kw)
    res = m.run()
    return m, res


class TestNoOpCases:
    def test_single_chain_unchanged_in_behavior(self):
        prog = countdown(4).build()
        spec = specialize(prog)
        assert len(spec.chains) == 1
        assert run(spec)[1].out == run(prog)[1].out

    def test_rotation_free_calls_one_clone_each(self):
        prog = zoom(4).build()
        spec = specialize(prog)
        assert len(spec.chains) == len(prog.chains)
        m0, r0 = run(prog)
        m1, r1 = run(spec)
        assert np.allclose(m0.P, m1.P)  # scale channel preserved too


class TestGuardsSurvive:
    def test_add_two_orientation_argument_preserved(self):
        # Guards are intentional frame-dependence and survive: the two calls
        # still take different arms (0.6 vs 2.5 — orientation is still the
        # argument). But each arm now executes AS AUTHORED, so the second
        # call's displacement is frame-covariant and reads +2.5 in the
        # caller's frame — the original's -2.5 came from the raw-segment
        # reinterpretation that specialization removes.
        prog = add_two().build()
        assert run(prog)[1].out == pytest.approx([0.6, -2.5])
        spec = specialize(prog)
        assert run(spec)[1].out == pytest.approx([0.6, 2.5])
        assert len(spec.chains) == 3  # entry + fork under two frames


class TestReinterpretationRemoved:
    def build_naive(self):
        b = ProgramBuilder("naive")
        c0 = b.chain("entry")
        c0.roth(math.pi / 2)
        c0.call(1)
        c0.halt()
        b.chain("emit").loadi(5).out()
        return b.build()

    def test_naive_program_misbehaves(self):
        # under the rotated frame the callee's segments reinterpret — here
        # OUT becomes REJ on a zero register and faults
        from versor import VersorFault
        with pytest.raises(VersorFault):
            run(self.build_naive())

    def test_specialized_program_behaves_as_authored(self):
        spec = specialize(self.build_naive())
        _, res = run(spec)
        assert res.out == pytest.approx([5.0])


class TestLevy:
    def test_specialized_levy_runs(self):
        spec = specialize(levy(6).build())
        m, res = run(spec, step_budget=100_000)
        assert res.halt_reason == "HALT"
        # 2^(depth-1) base segments were drawn
        from versor import Trace
        trace = Trace()
        Machine(spec, trace=trace, step_budget=100_000).run()
        assert trace.opcodes().count("NOP") == 2 ** 5

    def test_clone_count_bounded_by_frame_group(self):
        spec = specialize(levy(6).build())
        # rotations are z-axis multiples of 45deg: at most 8 frames/level
        assert len(spec.chains) <= 1 + 8 * 6

    def test_scale_still_composes(self):
        spec = specialize(levy(4).build())
        from versor import Trace
        trace = Trace()
        Machine(spec, trace=trace, step_budget=100_000).run()
        nops = [np.linalg.norm(r.P1 - r.P0) for r in trace
                if r.opcode == "NOP"]
        # base segments all execute at scale (1/sqrt(2))^(depth-1)
        assert np.allclose(nops, 3.0 * 2 ** -1.5)


class TestRefusals:
    def test_irrational_rotation_group_hits_budget(self):
        b = ProgramBuilder("irr")
        b.chain("entry").call(1).halt()
        c1 = b.chain("spin")
        c1.roth(1.0)  # 1 radian: infinite group
        c1.call(1)
        with pytest.raises(SpecializeError, match="budget"):
            specialize(b.build(), max_clones=64)

    def test_popf_refused(self):
        b = ProgramBuilder("t")
        b.chain().pushf().popf().halt()
        with pytest.raises(SpecializeError, match="POPF"):
            specialize(b.build())

    def test_skippable_rotation_refused(self):
        b = ProgramBuilder("t")
        b.chain().jmpz().roth(math.pi / 2).halt()
        with pytest.raises(SpecializeError, match="skipped"):
            specialize(b.build())

    def test_converging_frames_refused(self):
        b = ProgramBuilder("t")
        c = b.chain()
        c.loadi(1)
        c.branch(
            arm("ROTH", math.pi / 2, guard=(1, 0, 0), to="m"),
            arm("NOP", 1.0, guard=(-1, 0, 0), to="m"),
        )
        c.at("m").halt()
        with pytest.raises(SpecializeError, match="converging"):
            specialize(b.build())
