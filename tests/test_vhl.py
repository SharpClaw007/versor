import pytest

from versor import Machine
from versor.vhl import CompileError, compile_vhl


def run(src, **kw):
    return Machine(compile_vhl(src).build(), **kw).run().out


class TestExpressions:
    def test_print_constant(self):
        assert run("print 7") == pytest.approx([7.0])

    def test_arithmetic(self):
        assert run("print 2 + 3 * 4 - 1") == pytest.approx([13.0])

    def test_parens_and_precedence(self):
        assert run("print (2 + 3) * 4") == pytest.approx([20.0])

    def test_negative_constants_and_negation(self):
        assert run("print -5") == pytest.approx([-5.0])
        assert run("let x = 3\nprint -x") == pytest.approx([-3.0])
        assert run("print 2 - 7") == pytest.approx([-5.0])

    def test_zero(self):
        assert run("print 0") == pytest.approx([0.0])
        assert run("print 3 * 0") == pytest.approx([0.0])

    def test_variables(self):
        src = "let x = 5\nlet y = x + 3\nprint y\nprint y * 2\n"
        assert run(src) == pytest.approx([8.0, 16.0])

    def test_reassignment(self):
        assert run("let x = 1\nlet x = x + 1\nprint x") == pytest.approx([2.0])

    def test_subtraction_order(self):
        assert run("let a = 10\nlet b = 4\nprint a - b") == pytest.approx([6.0])

    def test_const_times_var(self):
        assert run("let x = 6\nprint 3 * x") == pytest.approx([18.0])


class TestRepeat:
    def test_countdown(self):
        src = """
let x = 3
repeat 3 {
    print x
    let x = x - 1
}
"""
        assert run(src) == pytest.approx([3.0, 2.0, 1.0])

    def test_repeat_zero_skips_body(self):
        assert run("repeat 0 {\nprint 9\n}\nprint 1") == pytest.approx([1.0])

    def test_repeat_negative_skips_body(self):
        assert run("repeat 0 - 2 {\nprint 9\n}\nprint 1") == pytest.approx([1.0])

    def test_repeat_variable_count(self):
        src = "let n = 4\nrepeat n {\nprint n\n}"
        assert run(src) == pytest.approx([4.0] * 4)

    def test_nested_repeat(self):
        src = """
repeat 2 {
    repeat 3 {
        print 1
    }
}
"""
        assert run(src) == pytest.approx([1.0] * 6)

    def test_multiplication_by_loop(self):
        # var * var, the VHL way
        src = """
let acc = 0
let y = 4
repeat 3 {
    let acc = acc + y
}
print acc
"""
        assert run(src) == pytest.approx([12.0])


class TestVersor32:
    """VHL v2: input() and var*var auto-select the icosa32 dialect."""

    def test_plain_programs_stay_cubic26(self):
        assert compile_vhl("print 1").build().decoder == "cubic26"

    def test_var_times_var(self):
        src = "let x = 6\nlet y = 7\nprint x * y"
        prog = compile_vhl(src).build()
        assert prog.decoder == "icosa32"
        assert Machine(prog).run().out == pytest.approx([42.0])

    def test_input(self):
        src = "let a = input()\nlet b = input()\nprint a + b"
        assert run(src, input=[30, 12]) == pytest.approx([42.0])

    def test_input_in_expression(self):
        assert run("print input() * 3", input=[5]) == pytest.approx([15.0])

    def test_square_via_product(self):
        src = "let x = input()\nprint x * x"
        assert run(src, input=[9]) == pytest.approx([81.0])

    def test_input_exhausted_faults(self):
        from versor import VersorFault
        with pytest.raises(VersorFault) as e:
            run("print input()", input=[])
        assert e.value.kind == "InputExhausted"


class TestErrors:
    @pytest.mark.parametrize("src,match", [
        ("print y", "undefined variable"),
        ("frobnicate 3", "unknown statement"),
        ("let 3 = 4", "expected: let"),
        ("repeat 3", "repeat expr"),
        ("}", "unmatched"),
        ("repeat 3 {\nprint 1", "unclosed"),
        ("print (1 + 2", "missing '[)]'"),
        ("print 1 @ 2", "cannot tokenize"),
    ])
    def test_error_cases(self, src, match):
        with pytest.raises(CompileError, match=match):
            compile_vhl(src)

    def test_out_of_registers(self):
        src = "let a = 1\nlet b = 2\nlet c = 3\nlet d = 4"
        with pytest.raises(CompileError, match="out of registers"):
            compile_vhl(src)

    def test_error_lines(self):
        with pytest.raises(CompileError, match="line 2"):
            compile_vhl("print 1\nprint zzz")
