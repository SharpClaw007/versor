"""Harmless-mover routing (versor/route.py)."""
import numpy as np
import pytest

from versor import Machine, ProgramBuilder
from versor.route import route, route_displacement


class TestClosedForm:
    @pytest.mark.parametrize("decoder", ["cubic26", "icosa32", "sphere26",
                                         "sphere32"])
    @pytest.mark.parametrize("delta", [
        (3, 0, 0), (-3, 0, 0), (0, 4, 0), (0, -4, 0), (0, 0, 2), (0, 0, -2),
        (1.5, -2.25, 0.75), (-10.1, 7.3, -3.9), (0.01, -0.02, 0.005),
        (200.0, -150.0, 12.0), (0, 0, 0),
    ])
    def test_route_is_exact(self, delta, decoder):
        ops = route(delta, decoder)
        assert np.allclose(route_displacement(ops, decoder), delta, atol=1e-9)

    def test_random_deltas_all_decoders(self):
        rng = np.random.default_rng(0)
        for decoder in ("cubic26", "icosa32", "sphere26", "sphere32"):
            for _ in range(50):
                d = rng.normal(scale=20, size=3)
                ops = route(d, decoder)
                assert np.allclose(route_displacement(ops, decoder), d,
                                   atol=1e-8)

    def test_rej_magnitudes_keep_index_zero(self):
        ops = route((0, 0, -5), "icosa32")
        for mnemonic, n in ops:
            if mnemonic == "REJ":
                assert 0 < n < 1

    def test_all_magnitudes_positive(self):
        for mnemonic, n in route((-7, 3, -2), "icosa32"):
            assert n > 1e-10


class TestOnMachine:
    def run_route(self, delta, *, a_value=9.0):
        """Execute a route with A parked on the data stack; the router's
        contract requires R0 to hold the reserved unit vector."""
        b = ProgramBuilder("t", decoder="icosa32")
        c = b.chain()
        c.loadi(1).movr(0)  # R0 = unit (VHL's prelude provides this)
        c.loadi(a_value)
        c.pusha()
        for mnemonic, n in route(delta, "icosa32"):
            c.op(mnemonic, n)
        c.popa()
        c.halt()
        m = Machine(b.build())
        m.run()
        return m

    @pytest.mark.parametrize("delta", [
        (5, 0, 0), (-4, 3, 0), (2.5, -1.5, 1.0), (-3, -3, -2),
    ])
    def test_machine_lands_on_target(self, delta):
        m = self.run_route(delta)
        m0 = self.run_route((0, 0, 0))
        assert np.allclose(m.P - m0.P, delta, atol=1e-9)

    def test_accumulator_survives(self):
        m = self.run_route((-6, 4, -1), a_value=7.0)
        assert np.allclose(m.A, [7, 0, 0])

    def test_no_faults_with_junk_registers(self):
        # movers read registers holding whatever; must never fault
        m = self.run_route((-8, -8, -3))
        assert m.halt_reason == "HALT"
