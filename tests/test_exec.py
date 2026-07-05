"""v0.3a: executable memory — EXEC is LOAD with n >= 2."""
import numpy as np
import pytest

from versor import LoadError, Machine, ProgramBuilder, Trace, VersorFault
from versor.asm import assemble


def halt_only():
    b = ProgramBuilder("t")
    b.chain().halt()
    return b.build()


class TestTrampoline:
    def build(self):
        # EM1: the program writes an instruction into memory, walks away,
        # comes back, and executes it
        b = ProgramBuilder("trampoline")
        c = b.chain()
        c.loadi(9)                # A = (9,0,0), P = (9,0,0)
        c.store(0.5)              # P = (8.5,0,0): M[(8,0,0)] = (9,0,0)
        c.op("MOVR", 2.0)         # walk away: P = (8.5,-2,0)
        c.exec_cell(2.5)          # arrive (8.5,0.5,0) = cell (8,0,0): EXEC
        c.halt()
        return b

    def test_stored_instruction_executes(self):
        m = Machine(self.build().build())
        m.run()
        # the stored vector (9,0,0) decodes as LOADI 9
        assert np.allclose(m.A, [9, 0, 0])

    def test_exec_moves_along_the_stored_vector(self):
        m = Machine(self.build().build())
        m.run()
        # P after exec = (8.5,0.5,0) + (9,0,0), then HALT's (0,0,-1)
        assert np.allclose(m.P, [17.5, 0.5, -1.0])


class TestChainedExec:
    def test_exec_walks_a_trail_of_stored_code(self):
        # cells (0,3k,0) each hold (0,3,0): an exec-LOAD hop to the next
        # cell. Plant a LOADI at the end of the trail; one EXEC runs all.
        m = Machine(halt_only())
        for k in range(1, 4):
            m.M[(0, 3 * k, 0)] = np.array([0.0, 3.0, 0.0])
        m.M[(0, 12, 0)] = np.array([6.0, 0.0, 0.0])  # LOADI 6 at the end
        m.P = np.array([0.5, 3.5, 0.0])  # standing in cell (0,3,0)
        m.do_exec()
        assert np.allclose(m.A, [6, 0, 0])
        assert m.P[1] == pytest.approx(12.5)  # walked the whole trail

    def test_runaway_trail_hits_depth_cap(self):
        m = Machine(halt_only())
        for k in range(0, 80):
            m.M[(0, 3 * k, 0)] = np.array([0.0, 3.0, 0.0])
        m.P = np.array([0.5, 0.5, 0.0])
        with pytest.raises(VersorFault) as e:
            m.do_exec()
        assert e.value.kind == "ExecDepthExceeded"

    def test_nested_exec_counts_against_step_budget(self):
        m = Machine(halt_only(), step_budget=10)
        m.steps = 9
        for k in range(0, 20):
            m.M[(0, 3 * k, 0)] = np.array([0.0, 3.0, 0.0])
        m.P = np.array([0.5, 0.5, 0.0])
        with pytest.raises(VersorFault) as e:
            m.do_exec()
        assert e.value.kind == "StepBudgetExhausted"


class TestFaults:
    def test_exec_empty_cell_faults(self):
        b = ProgramBuilder("t")
        b.chain().exec_cell(2.0)
        with pytest.raises(VersorFault) as e:
            Machine(b.build()).run()
        assert e.value.kind == "ExecEmptyCell"

    def test_plain_load_below_2_unchanged(self):
        b = ProgramBuilder("t")
        c = b.chain()
        c.loadi(3).store(0.5).op("MOVR", 2.0).load(1.9).halt()
        m = Machine(b.build())
        m.run()
        assert m.halt_reason == "HALT"  # no exec, no fault

    def test_builder_and_asm_guardrails(self):
        with pytest.raises(LoadError):
            ProgramBuilder("t").chain().exec_cell(1.0)
        with pytest.raises(LoadError, match="must be >= 2"):
            assemble("EXEC 1\n")

    def test_asm_exec_assembles_as_load(self):
        prog = assemble("LOADI 9\nSTORE 0.5\nOP MOVR 2.0\nEXEC 2.5\nHALT\n").build()
        m = Machine(prog)
        m.run()
        assert np.allclose(m.A, [9, 0, 0])


class TestTrace:
    def test_exec_records_marked_and_chronological(self):
        b = ProgramBuilder("t")
        c = b.chain()
        c.loadi(9).store(0.5).op("MOVR", 2.0).exec_cell(2.5).halt()
        trace = Trace()
        Machine(b.build(), trace=trace).run()
        ops = trace.opcodes()
        i = ops.index("LOAD")
        assert ops[i + 1] == "@LOADI"  # executed-from-memory marker
        steps = [r.step for r in trace]
        assert steps == sorted(steps)
