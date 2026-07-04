import math
import os

import numpy as np
import pytest

from versor import AsmError, Machine, assemble, assemble_path
from versor.examples import add_two, countdown

EXAMPLES = os.path.join(os.path.dirname(__file__), "..", "examples")


def run_text(text, **kw):
    m = Machine(assemble(text).build(), **kw)
    res = m.run()
    return m, res


class TestPrograms:
    def test_countdown_vasm_matches_builder(self):
        prog = assemble_path(os.path.join(EXAMPLES, "countdown.vasm")).build()
        res = Machine(prog).run()
        expected = Machine(countdown(5).build()).run()
        assert res.out == expected.out
        assert np.allclose(res.displacement, expected.displacement)

    def test_add_two_vasm_matches_builder(self):
        prog = assemble_path(os.path.join(EXAMPLES, "add_two.vasm")).build()
        res = Machine(prog).run()
        expected = Machine(add_two().build()).run()
        assert res.out == pytest.approx(expected.out)

    def test_minimal_program(self):
        _, res = run_text("LOADI 5\nSCALE 2\nOUT\nHALT\n")
        assert res.out == [pytest.approx(10.0)]


class TestSyntax:
    def test_comments_and_blank_lines(self):
        _, res = run_text("""
        ; full-line comment
        LOADI 3   ; trailing comment
        OUT       # hash comment too

        HALT
        """)
        assert res.out == [pytest.approx(3.0)]

    def test_registers_and_raw_magnitudes(self):
        m, _ = run_text("LOADI 4\nMOVR r2\nMOVR 3.0\nHALT\n")
        assert np.allclose(m.R[2], [4, 0, 0])
        assert np.allclose(m.R[3], [4, 0, 0])  # raw 3.0 -> idx 3

    def test_angle_forms(self):
        for spec in ("pi/2", "0.5pi", "1.5707963267948966"):
            m, _ = run_text(f"ROTH {spec}\nSEG (0,0,-1)\n")
            v = m.F.rotate(np.array([1.0, 0, 0]))
            assert np.allclose(v, [0, 1, 0], atol=1e-9)

    def test_outc_char_mode(self):
        _, res = run_text("LOADI 72\nOUTC\nHALT\n")
        assert res.out == ["H"]

    def test_call_by_name_and_index(self):
        text = """
.chain main
        CALL fn
        CALL 1
        HALT
.chain fn
        NOP 2
"""
        m, _ = run_text(text)
        assert np.allclose(m.A, assemble(text).build().chains[1].vertices[0][0].seg)

    def test_decoder_directive(self):
        text = ".decoder icosa32\nLOADI 5\nOUT\nHALT\n"
        prog = assemble(text).build()
        assert prog.decoder == "icosa32"
        assert Machine(prog).run().out == [pytest.approx(5.0)]

    def test_jump_suffix_builds_cycle(self):
        from versor import VersorFault
        text = "a: NOP -> a\n"
        with pytest.raises(VersorFault) as e:
            run_text(text, step_budget=50)
        assert e.value.kind == "StepBudgetExhausted"

    def test_segraw_bypasses_authoring_frame(self):
        m, _ = run_text("ROTH pi\nSEGRAW (0,0,-1)\n")  # raw -z: HALT anyway
        assert m.halt_reason == "HALT"

    def test_guard_shorthand_equals_explicit(self):
        a = assemble("LOADI 2\nBR -x: HALT -> e, +x: NOP -> f\n").build()
        b = assemble("LOADI 2\nBR (-1,0,0): HALT -> e, (1,0,0): NOP -> f\n").build()
        ea = a.chains[0].vertices[1]
        eb = b.chains[0].vertices[1]
        for x, y in zip(ea, eb):
            assert np.allclose(x.guard, y.guard)

    def test_implicit_entry_chain_before_directive(self):
        text = "CALL fn\nHALT\n.chain fn\nNOP 2\n"
        m, _ = run_text(text)
        assert m.halt_reason == "HALT"


class TestErrors:
    @pytest.mark.parametrize("text,match", [
        ("BOGUS 1\n", "unknown mnemonic"),
        ("MOVR r7\n", "register"),
        ("LOADI\n", "numeric operand"),
        ("ROTF\n", "needs an angle"),
        ("CALL nope\nHALT\n", "unknown chain"),
        ("CALL 3\nHALT\n", "out of range"),
        ("BR -x: HALT\n", "'-> target'"),
        ("BR -x -> end, +x -> loop\n", "guard: OP"),
        ("LOADI 0\n", "must be positive"),
        ("SEG (1,2)\n", "3 components"),
        (".decoder martian\nHALT\n", "unknown decoder"),
        (".bogus x\nHALT\n", "unknown directive"),
    ])
    def test_error_cases(self, text, match):
        with pytest.raises(AsmError, match=match):
            assemble(text).build()

    def test_errors_carry_line_numbers(self):
        with pytest.raises(AsmError, match="line 3"):
            assemble("LOADI 1\nOUT\nBOGUS 9\n")

    def test_duplicate_label(self):
        with pytest.raises(AsmError, match="duplicate label"):
            assemble("a: LOADI 1\na: HALT\n")

    def test_empty_file(self):
        with pytest.raises(AsmError, match="no instructions"):
            assemble("; nothing here\n")
