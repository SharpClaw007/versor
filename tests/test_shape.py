"""Shape-objective synthesis: compute Y while drawing X."""
import numpy as np
import pytest

from versor.examples import countdown
from versor.synth import (evolve, get_vectors, magnitude_locked_mutation,
                          resample_polyline, run_points, shape_distance,
                          shape_fitness, tolerance_mask)

L_SHAPE = np.array([[0, 2, 0], [0, 0, 0], [3, 0, 0]], float)


class TestResample:
    def test_endpoint_preserving(self):
        r = resample_polyline(L_SHAPE, 16)
        assert np.allclose(r[0], L_SHAPE[0])
        assert np.allclose(r[-1], L_SHAPE[-1])

    def test_uniform_spacing(self):
        r = resample_polyline(L_SHAPE, 11)
        gaps = np.linalg.norm(np.diff(r, axis=0), axis=1)
        assert np.allclose(gaps, gaps[0])

    def test_degenerate_polyline(self):
        r = resample_polyline(np.zeros((3, 3)), 8)
        assert r.shape == (8, 3)


class TestShapeDistance:
    def test_identity_is_zero(self):
        assert shape_distance(L_SHAPE, L_SHAPE) == pytest.approx(0.0)

    def test_translation_and_scale_invariant(self):
        moved = L_SHAPE * 7.0 + np.array([5, -2, 3])
        assert shape_distance(moved, L_SHAPE) == pytest.approx(0.0, abs=1e-9)

    def test_orientation_matters(self):
        flipped = L_SHAPE * np.array([-1, 1, 1])
        assert shape_distance(flipped, L_SHAPE) > 0.3

    def test_different_shapes_far(self):
        line = np.array([[0, 0, 0], [5, 0, 0]], float)
        assert shape_distance(line, L_SHAPE) > 0.2


class TestMutationMask:
    def test_tolerance_mask_finds_value_carriers(self):
        prog = countdown(3).build()
        locked = tolerance_mask(prog)
        assert locked.tolist().count(True) == 2  # the two LOADIs

    def test_locked_magnitudes_survive_mutation(self):
        prog = countdown(3).build()
        x = get_vectors(prog)
        locked = tolerance_mask(prog)
        mutate = magnitude_locked_mutation(x, locked)
        kids = mutate(x, 0.5, 8, np.random.default_rng(0))
        norms = np.linalg.norm(kids, axis=2)
        seed_norms = np.linalg.norm(x, axis=1)
        assert np.allclose(norms[:, locked], seed_norms[locked])
        assert not np.allclose(norms[:, ~locked], seed_norms[~locked])


class TestShapeEvolution:
    def test_shape_improves_while_output_exact(self):
        target = np.array([[-5.0, 0.6, 0], [0.0, -0.6, 0], [4.0, 3.0, 0]])
        expected = [3.0, 2.0, 1.0]
        prog = countdown(3).build()
        _, seed_pts = run_points(prog)
        seed_d = shape_distance(seed_pts, target)

        fit = shape_fitness(expected, target)
        best, hist = evolve(
            prog, fit, evaluator=lambda p: run_points(p, 2000),
            mutate=magnitude_locked_mutation(get_vectors(prog),
                                             tolerance_mask(prog)),
            sigma=0.25, lam=16, generations=120, seed=2, step_budget=2000)

        out, pts = run_points(best)
        assert out == pytest.approx(expected)          # Y still computed
        assert shape_distance(pts, target) < seed_d - 0.2  # X approached
        assert hist[-1] < hist[0]
