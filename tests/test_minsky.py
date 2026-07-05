import numpy as np
import pytest

from versor import Machine
from versor.minsky import DecJz, Inc, compile_minsky


def run(program, c1=0, c2=0, **kw):
    m = Machine(compile_minsky(program, c1, c2, **kw).build())
    res = m.run()
    return m, res


def counters(m):
    return int(round(m.R[1][0])), int(round(m.R[2][0]))


class TestGadgets:
    def test_empty_program_halts(self):
        m, res = run([], c1=3, c2=1)
        assert res.halt_reason == "HALT"
        assert counters(m) == (3, 1)

    def test_inc(self):
        m, _ = run([Inc(1, next=1)], c1=0)
        assert counters(m) == (1, 0)

    def test_decjz_decrements_and_falls_through(self):
        m, _ = run([DecJz(1, next=1, zero=1)], c1=5)
        assert counters(m) == (4, 0)

    def test_decjz_takes_zero_branch_on_empty_counter(self):
        # zero branch jumps straight to halt, skipping the Inc at line 1
        m, _ = run([DecJz(1, next=1, zero=2), Inc(2, next=2)], c1=0)
        assert counters(m) == (0, 0)

    def test_counter_validation(self):
        with pytest.raises(ValueError):
            compile_minsky([Inc(3, next=1)])
        with pytest.raises(ValueError):
            compile_minsky([], c1=-1)


class TestPrograms:
    def test_transfer_adds_c2_into_c1(self):
        # 0: DECJZ c2 -> 1 (zero -> halt);  1: INC c1 -> 0
        prog = [DecJz(2, next=1, zero=2), Inc(1, next=0)]
        m, res = run(prog, c1=3, c2=4, emit_counters=True)
        assert counters(m) == (7, 0)
        assert res.out == pytest.approx([7.0, 0.0])

    def test_doubling(self):
        # 0: DECJZ c1 -> 1 (zero -> halt);  1: INC c2 -> 2;  2: INC c2 -> 0
        prog = [DecJz(1, next=1, zero=3), Inc(2, next=2), Inc(2, next=0)]
        m, _ = run(prog, c1=5)
        assert counters(m) == (0, 10)

    def test_multiply_by_repeated_transfer(self):
        # c2 = 3 * c1, unrolled as three INCs per decrement
        prog = [DecJz(1, next=1, zero=4),
                Inc(2, next=2), Inc(2, next=3), Inc(2, next=0)]
        m, _ = run(prog, c1=6)
        assert counters(m) == (0, 18)

    def test_counters_only_touch_x_axis(self):
        prog = [DecJz(1, next=1, zero=2), Inc(2, next=0)]
        m, _ = run(prog, c1=9)
        assert np.allclose(m.R[1][1:], 0) and np.allclose(m.R[2][1:], 0)

    def test_step_count_linear_in_counter(self):
        prog = [DecJz(1, next=1, zero=2), Inc(2, next=0)]
        _, small = run(prog, c1=10)
        _, large = run(prog, c1=100)
        assert large.steps < small.steps * 11  # no quadratic blowup
