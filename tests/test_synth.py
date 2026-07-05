import numpy as np
import pytest

from versor import Machine
from versor.examples import countdown
from versor.synth import (evolve, get_vectors, output_fitness, run_out,
                          segment_keys, segment_tolerances, set_vectors)


@pytest.fixture(scope="module")
def prog():
    return countdown(3).build()


class TestVectors:
    def test_roundtrip(self, prog):
        x = get_vectors(prog)
        assert x.shape == (7, 3)
        clone = set_vectors(prog, x)
        assert run_out(clone) == run_out(prog)

    def test_set_does_not_mutate_original(self, prog):
        baseline = run_out(prog)
        set_vectors(prog, get_vectors(prog) * 0.0 + 1.0)
        assert run_out(prog) == baseline


class TestFitness:
    def test_exact_and_tolerance(self):
        fit = output_fitness([3.0, 2.0])
        assert fit([3.0, 2.0]) == 0.0
        assert fit([3.00000001, 2.0]) == 0.0
        assert fit([3.1, 2.0]) == pytest.approx(0.1)

    def test_fault_and_length_penalties(self):
        fit = output_fitness([1.0])
        assert fit(None) == 100.0
        assert fit([]) == 10.0
        assert fit([1.0, 1.0]) == 10.0

    def test_chars(self):
        fit = output_fitness(["H", "i"])
        assert fit(["H", "i"]) == 0.0
        assert fit(["H", "o"]) == 5.0


class TestTolerances:
    def test_value_channels_have_zero_tolerance(self, prog):
        tols = dict(segment_tolerances(prog, directions=6, iters=8))
        keys = segment_keys(prog)
        # segment 0 is LOADI 1 (the decrement unit): its magnitude IS the
        # output spacing, so any perturbation changes behavior
        assert tols[keys[0]] < 0.02
        # the branch arms only carry structure: comfortably positive
        assert tols[(0, 5, 0)] > 0.1
        assert tols[(0, 5, 1)] > 0.1

    def test_faulting_baseline_rejected(self, prog):
        bad = set_vectors(prog, get_vectors(prog) * 1e-12)
        with pytest.raises(ValueError):
            segment_tolerances(bad)


class TestEvolve:
    def test_repairs_scrambled_countdown(self, prog):
        rng = np.random.default_rng(5)
        x0 = get_vectors(prog)
        broken = set_vectors(prog, x0 + rng.normal(scale=0.15, size=x0.shape))
        assert run_out(broken) != run_out(prog)  # actually broken

        fit = output_fitness([3.0, 2.0, 1.0])
        best, hist = evolve(broken, fit, seed=3, lam=16, generations=250,
                            sigma=0.15, step_budget=2000)
        assert hist[-1] == 0.0
        assert hist[-1] < hist[0]
        out = run_out(best)
        assert out == pytest.approx([3.0, 2.0, 1.0], abs=1e-3)

    def test_already_perfect_returns_immediately(self, prog):
        best, hist = evolve(prog, output_fitness([3.0, 2.0, 1.0]),
                            generations=50)
        assert hist == [0.0]
        assert run_out(best) == run_out(prog)
