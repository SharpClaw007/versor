"""Milestone acceptance tests (M1-M5) on the example programs."""
import math

import numpy as np
import pytest

from versor import Machine, Trace, from_dict, to_dict
from versor.examples import (add_two, countdown, fib, helix, hello, memory,
                             straightline)
from versor.quat import Quat


def run(prog, **kw):
    m = Machine(prog, **kw)
    res = m.run()
    return m, res


class TestM1Straightline:
    def test_prints_ten(self):
        _, res = run(straightline().build())
        assert res.out == [pytest.approx(10.0)]
        assert res.halt_reason == "HALT"

    def test_frame_covariance(self):
        """Rotating every raw segment together with the initial frame must
        leave the output unchanged."""
        q = Quat.axis_angle([1, 2, -1], 1.234)
        prog = straightline().build()
        for ch in prog.chains:
            for edges in ch.vertices.values():
                for e in edges:
                    e.seg = q.rotate(e.seg)
        _, res = run(prog, F0=q)
        assert res.out == [pytest.approx(10.0)]


class TestM2Countdown:
    def test_output(self):
        _, res = run(countdown(5).build())
        assert res.out == pytest.approx([5.0, 4.0, 3.0, 2.0, 1.0])
        assert res.halt_reason == "HALT"

    def test_trace_shows_cycle(self):
        trace = Trace()
        prog = countdown(4).build()
        Machine(prog, trace=trace).run()
        # the loop body re-enters the same graph vertex each lap
        loop_entries = [r for r in trace if r.opcode == "NOP" and r.branch]
        assert len(loop_entries) == 3  # 4 laps, last one exits via HALT
        revisited = [r.to for r in loop_entries]
        assert len(set(revisited)) == 1  # a literal cycle in the graph

    def test_loop_body_is_a_helix(self):
        """Same graph vertex, translated in space each lap: loops are helices."""
        trace = Trace()
        Machine(countdown(4).build(), trace=trace).run()
        outs = [r.P0 for r in trace if r.opcode == "OUT"]
        deltas = np.diff(np.array(outs), axis=0)
        assert np.allclose(deltas, deltas[0])  # constant per-lap displacement


class TestM3AddTwo:
    def test_orientation_is_the_argument(self):
        _, res = run(add_two().build())
        assert res.out == pytest.approx([0.6, -2.5])

    def test_world_displacements_differ(self):
        trace = Trace()
        Machine(add_two().build(), trace=trace).run()
        rets = [r.A.copy() for r in trace if r.opcode == "RET*"]
        assert len(rets) == 2
        assert not np.allclose(rets[0], rets[1])


class TestM4Memory:
    def test_store_walk_away_walk_back_load(self):
        m, res = run(memory().build())
        assert res.out == [pytest.approx(7.0)]
        assert np.allclose(m.M[(6, 0, 0)], [7, 0, 0])

    def test_route_back_differs_from_route_out(self):
        trace = Trace()
        Machine(memory().build(), trace=trace).run()
        ops = trace.opcodes()
        # out via MOVR/-y, back via DOT/SCALE zig-zag: different ops, different path
        assert "MOVR" in ops and "DOT" in ops and "SCALE" in ops


class TestM5Showpieces:
    def test_hello(self):
        _, res = run(hello().build())
        assert "".join(res.out) == "Hello, world!\n"

    def test_fib(self):
        _, res = run(fib(8).build())
        assert res.out == pytest.approx([1, 2, 3, 5, 8, 13, 21, 34])


class TestSerialization:
    @pytest.mark.parametrize("builder_fn", [straightline, countdown, add_two,
                                            memory, hello, fib])
    def test_roundtrip_preserves_behavior(self, builder_fn):
        prog = builder_fn().build()
        _, direct = run(prog)
        _, reloaded = run(from_dict(to_dict(prog)))
        assert direct.out == reloaded.out
        assert np.allclose(direct.displacement, reloaded.displacement)

    def test_examples_are_lint_clean(self):
        for fn in (straightline, countdown, add_two, memory, hello, fib, helix):
            assert fn().build().warnings == []

    def test_helix_halts_after_full_turn(self):
        m = Machine(helix(8).build())
        res = m.run()
        assert res.halt_reason == "HALT"
        assert res.steps == 8 * 3 + 1
