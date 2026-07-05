"""LSP analysis functions (pure — no pygls needed)."""
from versor.lsp import (completions_for, definition_line, diagnostics_for,
                        hover_for, kind_of, word_at)

COUNTDOWN = """.name countdown
.chain entry
        LOADI 1
        MOVR r0
        LOADI 5
loop:   OUT
        SUB r0
        BR -x: HALT -> end, +x: NOP -> loop
"""


class TestDiagnostics:
    def test_clean_program_no_errors(self):
        assert diagnostics_for(COUNTDOWN, "vasm") == []

    def test_error_carries_line(self):
        diags = diagnostics_for("LOADI 1\nBOGUS 2\nHALT\n", "vasm")
        assert len(diags) == 1
        assert diags[0]["line"] == 1  # 0-based
        assert "BOGUS" in diags[0]["message"]
        assert diags[0]["severity"] == "error"

    def test_vhl_errors(self):
        diags = diagnostics_for("print 1\nprint zzz\n", "vhl")
        assert diags[0]["line"] == 1
        assert "undefined" in diags[0]["message"]

    def test_lint_warnings_surface(self):
        # a dead-zone segment: warning, not error
        diags = diagnostics_for("SEG (0.351, 0.9364, 0)\nHALT\n", "vasm")
        assert any(d["severity"] == "warning" for d in diags)

    def test_extended_op_under_cubic26(self):
        diags = diagnostics_for("INP\nHALT\n", "vasm")
        assert diags[0]["severity"] == "error"
        assert "Versor-32" in diags[0]["message"]


class TestHover:
    def test_opcodes_documented(self):
        assert "frame-local x slot" in hover_for("LOADI")
        assert "orientation is the argument" in hover_for("call")
        assert "Versor-32" in hover_for("MULR")

    def test_directives(self):
        assert "decoder" in hover_for(".decoder").lower()

    def test_unknown(self):
        assert hover_for("banana") is None

    def test_every_mnemonic_has_docs(self):
        from versor.isa import MNEMONIC_TO_TRIPLE
        from versor.opdocs import OP_DOCS
        assert set(MNEMONIC_TO_TRIPLE) <= set(OP_DOCS)


class TestNavigation:
    def test_label_definition(self):
        assert definition_line(COUNTDOWN, "loop") == 5

    def test_chain_definition(self):
        assert definition_line(COUNTDOWN, "entry") == 1

    def test_missing(self):
        assert definition_line(COUNTDOWN, "nope") is None

    def test_word_at(self):
        assert word_at("        BR -x: HALT -> end", 16) == "HALT"
        assert word_at(".decoder icosa32", 3) == ".decoder"
        assert word_at("", 0) is None


class TestCompletions:
    def test_vasm_has_everything(self):
        items = dict(completions_for(COUNTDOWN, "vasm"))
        assert items["LOADI"] == "opcode"
        assert items["INP"] == "opcode"
        assert items["r0"] == "register"
        assert items[".chain"] == "directive"
        assert items["loop"] == "label"
        assert items["entry"] == "chain"

    def test_vhl_keywords(self):
        items = dict(completions_for("", "vhl"))
        assert "repeat" in items and "input()" in items


def test_kind_of():
    assert kind_of("file:///x/a.vasm") == "vasm"
    assert kind_of("/x/a.vhl") == "vhl"
    assert kind_of("/x/a.py") is None
