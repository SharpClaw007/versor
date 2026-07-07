"""Routed register spilling: allocation as path planning."""
import numpy as np
import pytest

from versor import Machine, Trace
from versor.vhl import CompileError, compile_vhl


def run(src, budget=200_000, **kw):
    m = Machine(compile_vhl(src).build(), step_budget=budget, **kw)
    res = m.run()
    return m, res


SIX_VARS = """
let a = 1
let b = 2
let c = 3
let d = 4
let e = 5
let f = 6
"""


class TestStraightLine:
    def test_five_var_sum(self):
        src = ("let a = 10\nlet b = 20\nlet c = 30\nlet d = 40\nlet e = 50\n"
               "print a + b + c + d + e")
        _, res = run(src)
        assert res.out == pytest.approx([150.0])

    def test_six_vars_all_readable(self):
        src = SIX_VARS + "print f\nprint e\nprint d\nprint c\nprint b\nprint a"
        _, res = run(src)
        assert res.out == pytest.approx([6, 5, 4, 3, 2, 1])

    def test_spilled_var_reassignment(self):
        src = SIX_VARS + "let a = a + 100\nprint a + f"
        _, res = run(src)
        assert res.out == pytest.approx([107.0])

    def test_subtraction_order_with_spills(self):
        src = SIX_VARS + "print a - b - c\nprint f - e"
        _, res = run(src)
        assert res.out == pytest.approx([-4.0, 1.0])

    def test_products_with_spills(self):
        src = SIX_VARS + "print a * f\nprint b * e * 2"
        _, res = run(src)
        assert res.out == pytest.approx([6.0, 20.0])

    def test_spill_cells_live_in_the_spill_region(self):
        m, _ = run(SIX_VARS + "print a")
        for cell in m.M:
            assert cell[0] >= 300 and cell[1] == -300


class TestControlFlow:
    def test_spill_access_inside_while(self):
        src = SIX_VARS + """
let i = 3
while i {
    let a = a + f
    let i = i - 1
}
print a
"""
        _, res = run(src)
        assert res.out == pytest.approx([1 + 3 * 6])

    def test_loop_bodies_are_displacement_closed(self):
        # same static position every lap: the loop's OUT records must all
        # start from identical positions
        src = SIX_VARS + """
let i = 4
while i {
    print a + i
    let i = i - 1
}
"""
        trace = Trace()
        prog = compile_vhl(src).build()
        Machine(prog, trace=trace, step_budget=200_000).run()
        outs = [r.P0 for r in trace if r.opcode == "OUT"]
        assert len(outs) == 4
        for p in outs[1:]:
            assert np.allclose(p, outs[0], atol=1e-6)

    def test_spill_in_if_arms(self):
        src = SIX_VARS + """
if a {
    let b = b + 10
} else {
    let b = b + 100
}
print b + f
"""
        _, res = run(src)
        assert res.out == pytest.approx([18.0])

    def test_if_arms_converge_spatially(self):
        src = SIX_VARS + "if a - 10 {\nprint 1\n} else {\nprint 2\n}\nprint b"
        _, res = run(src)
        assert res.out == pytest.approx([2.0, 2.0])

    def test_repeat_with_spills(self):
        src = SIX_VARS + "repeat 3 {\nlet a = a + e\n}\nprint a"
        _, res = run(src)
        assert res.out == pytest.approx([16.0])


class TestInterplay:
    def test_input_and_spill(self):
        src = SIX_VARS + "let g = input()\nprint g + a + f"
        _, res = run(src, input=[100])
        assert res.out == pytest.approx([107.0])

    def test_call_then_spill_access_errors(self):
        src = SIX_VARS + ("fn id(x) {\nreturn x\n}\n"
                          "print id(1)\nprint a")
        with pytest.raises(CompileError, match="statically-known position"):
            compile_vhl(src)

    def test_spill_before_call_is_fine_if_unused_after(self):
        src = SIX_VARS + ("fn id(x) {\nreturn x\n}\n"
                          "print id(9)")
        _, res = run(src)
        assert res.out == pytest.approx([9.0])
