from versor.repl import Repl


def feed_all(repl, *lines):
    out = []
    for ln in lines:
        out += repl.feed(ln)
    return out


class TestExecution:
    def test_instructions_execute_and_echo(self):
        r = Repl()
        out = r.feed("LOADI 5")
        assert any("A=(5, 0, 0)" in ln for ln in out)

    def test_out_items_echo_once_each(self):
        r = Repl()
        echo1 = feed_all(r, "LOADI 5", "OUT")
        assert sum("OUT: 5" in ln for ln in echo1) == 1
        echo2 = r.feed("OUT")
        assert sum("OUT: 5" in ln for ln in echo2) == 1  # only the new one

    def test_rotation_reaims_typed_intent(self):
        # after ROTH pi/2 the machine frame matches the authoring frame,
        # so a typed LOADI still decodes as LOADI
        r = Repl()
        feed_all(r, "ROTH pi/2", "LOADI 3")
        assert abs(r.machine.A[1] - 3.0) < 1e-9  # frame-x is world +y

    def test_labels_and_branches_work(self):
        r = Repl()
        feed_all(r, "LOADI 1", "MOVR r0", "LOADI 3",
                 "loop: OUT", "SUB r0",
                 "BR -x: NOP -> end, +x: NOP -> loop")
        assert [round(o) for o in r.machine.OUT] == [3, 2, 1]

    def test_bad_line_rejected_and_reported(self):
        r = Repl()
        r.feed("LOADI 5")
        out = r.feed("BOGUS 1")
        assert any("error" in ln for ln in out)
        assert r.lines == ["LOADI 5"]  # rejected line not kept

    def test_error_line_number_matches_user_view(self):
        r = Repl()
        r.feed("LOADI 5")
        out = r.feed("LOADI")  # missing operand, user line 2
        assert any("line 2" in ln for ln in out)

    def test_runtime_fault_kept_with_notice(self):
        r = Repl()
        out = r.feed("EXEC 2")  # empty cell
        assert any("FAULT" in ln and "ExecEmptyCell" in ln for ln in out)
        assert r.lines == ["EXEC 2"]


class TestCommands:
    def test_state_and_mem(self):
        r = Repl()
        feed_all(r, "LOADI 7", "STORE 0.5")
        state = r.feed(":state")
        assert any("P = (6.5, 0, 0)" in ln for ln in state)
        mem = r.feed(":mem")
        assert any("M(6, 0, 0)" in ln for ln in mem)

    def test_undo_restores(self):
        r = Repl()
        feed_all(r, "LOADI 5", "SCALE 2")
        assert abs(r.machine.A[0] - 10) < 1e-9
        r.feed(":undo")
        assert abs(r.machine.A[0] - 5) < 1e-9
        r.feed(":undo")
        assert r.machine is None
        assert r.feed(":undo") == ["(nothing to undo)"]

    def test_reset(self):
        r = Repl()
        r.feed("LOADI 5")
        assert r.feed(":reset") == ["reset"]
        assert r.lines == []

    def test_decoder_switch(self):
        r = Repl()
        assert r.feed(":decoder icosa32") == ["decoder = icosa32"]
        r.feed("INP")  # extended op now assembles
        assert r.lines == ["INP"]
        assert r.feed(":decoder martian") == ["unknown decoder 'martian'"]

    def test_input_buffer(self):
        r = Repl()
        r.feed(":decoder icosa32")
        r.feed(":input 6,7")
        feed_all(r, "INP", "MOVR r1", "INP", "MULR r1", "OUT")
        assert round(r.machine.OUT[-1]) == 42

    def test_save_and_list(self, tmp_path):
        r = Repl()
        r.feed("LOADI 5")
        assert r.feed(":list") == ["LOADI 5"]
        path = str(tmp_path / "s.vasm")
        r.feed(f":save {path}")
        assert "LOADI 5" in open(path).read()

    def test_unknown_command(self):
        assert "unknown command" in Repl().feed(":frobnicate")[0]
