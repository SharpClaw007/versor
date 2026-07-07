"""VHL v3: while, if/else, user functions, recursion."""
import pytest

from versor import Machine
from versor.vhl import CompileError, compile_vhl


def run(src, **kw):
    return Machine(compile_vhl(src).build(), **kw).run().out


class TestWhile:
    def test_countdown(self):
        src = "let n = 4\nwhile n {\nprint n\nlet n = n - 1\n}"
        assert run(src) == pytest.approx([4.0, 3.0, 2.0, 1.0])

    def test_condition_reevaluated(self):
        src = "let n = 5\nwhile n - 2 {\nprint n\nlet n = n - 1\n}"
        assert run(src) == pytest.approx([5.0, 4.0, 3.0])

    def test_never_entered(self):
        assert run("while 0 {\nprint 9\n}\nprint 1") == pytest.approx([1.0])

    def test_negative_condition_skips(self):
        assert run("while 0 - 3 {\nprint 9\n}\nprint 1") == pytest.approx([1.0])


class TestIf:
    def test_taken_and_not_taken(self):
        assert run("if 1 {\nprint 7\n}\nprint 8") == pytest.approx([7.0, 8.0])
        assert run("if 0 {\nprint 7\n}\nprint 8") == pytest.approx([8.0])

    def test_else(self):
        src = "if %s {\nprint 1\n} else {\nprint 2\n}\nprint 3"
        assert run(src % "5") == pytest.approx([1.0, 3.0])
        assert run(src % "0") == pytest.approx([2.0, 3.0])

    def test_nested(self):
        src = """
let x = 2
if x {
    if x - 1 {
        print 22
    } else {
        print 21
    }
} else {
    print 10
}
"""
        assert run(src) == pytest.approx([22.0])

    def test_if_inside_while(self):
        src = """
let n = 4
while n {
    if n - 2 {
        print n
    }
    let n = n - 1
}
"""
        assert run(src) == pytest.approx([4.0, 3.0])


class TestFunctions:
    def test_basic_call(self):
        src = "fn square(x) {\nreturn x * x\n}\nprint square(6)"
        assert run(src) == pytest.approx([36.0])

    def test_two_args_and_expression_args(self):
        src = ("fn diff(a, b) {\nreturn a - b\n}\n"
               "print diff(10, 3 + 3)")
        assert run(src) == pytest.approx([4.0])

    def test_call_in_expression(self):
        src = "fn inc(x) {\nreturn x + 1\n}\nprint inc(2) * inc(4)"
        assert run(src) == pytest.approx([15.0])

    def test_live_registers_survive_call(self):
        src = ("fn clobber(x) {\nlet a = 100\nlet b = 200\nreturn x\n}\n"
               "let u = 7\nlet v = 9\nprint clobber(1) + u + v")
        assert run(src) == pytest.approx([17.0])

    def test_fall_off_end_returns_zero(self):
        src = "fn noop(x) {\nprint x\n}\nprint noop(5) + 1"
        assert run(src) == pytest.approx([5.0, 1.0])

    def test_early_return(self):
        src = ("fn f(x) {\nif x {\nreturn 1\n}\nreturn 2\n}\n"
               "print f(1)\nprint f(0)")
        assert run(src) == pytest.approx([1.0, 2.0])

    def test_functions_calling_functions(self):
        src = ("fn double(x) {\nreturn x + x\n}\n"
               "fn quad(x) {\nreturn double(double(x))\n}\n"
               "print quad(3)")
        assert run(src) == pytest.approx([12.0])

    def test_recursion_factorial(self):
        src = """
fn fact(n) {
    if n - 1 {
        return fact(n - 1) * n
    }
    return 1
}
print fact(5)
"""
        assert run(src) == pytest.approx([120.0])

    def test_recursion_fibonacci(self):
        src = """
fn fib(n) {
    if n - 2 {
        return fib(n - 1) + fib(n - 2)
    }
    if n {
        return 1
    }
    return 0
}
print fib(10)
"""
        assert run(src) == pytest.approx([55.0])

    def test_function_with_input(self):
        src = "fn twice(x) {\nreturn x + x\n}\nprint twice(input())"
        assert run(src, input=[21]) == pytest.approx([42.0])

    def test_selects_icosa32(self):
        src = "fn f(x) {\nreturn x\n}\nprint f(1)"
        assert compile_vhl(src).build().decoder == "icosa32"


class TestErrors:
    @pytest.mark.parametrize("src,match", [
        ("print f(1)", "undefined function"),
        ("fn f(x) {\nreturn x\n}\nprint f(1, 2)", "takes 1 argument"),
        ("return 3", "return outside"),
        ("fn f(x) {\nfn g(y) {\nreturn y\n}\n}", "nested fn"),
        ("fn f(x, x) {\nreturn x\n}", "duplicate parameter"),
        ("fn f(x) {\nreturn x\n}\nfn f(y) {\nreturn y\n}", "duplicate function"),
        ("} else {", "without a matching if"),
        ("if 1 {\nprint 1\n} else {\nprint 2\n} else {\nprint 3\n}",
         "duplicate else"),
        ("while 1 {\nprint 1", "unclosed"),
    ])
    def test_error_cases(self, src, match):
        with pytest.raises(CompileError, match=match):
            compile_vhl(src)

    def test_call_with_full_registers_exhausts(self):
        src = ("fn f(x) {\nreturn x\n}\n"
               "let a = 1\nlet b = 2\nlet c = 3\nprint f(a) + b + c")
        with pytest.raises(CompileError, match="out of registers"):
            compile_vhl(src)
