"""Compute Y while drawing X: evolve a countdown whose trace is a swoosh.

The behavioral objective (print 5 4 3 2 1, weighted to dominate) and the
aesthetic objective (trace shaped like a swoosh) are both functions of the
same geometry, so one evolution strategy optimizes them together. Mutation
is masked by the robustness map: value-carrying segments keep their
magnitudes (their length IS the printed value) and contribute direction
only.

Usage: python examples/shapewrite.py
Writes docs/screenshots/shapewrite.png.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np  # noqa: E402

from versor.examples import countdown  # noqa: E402
from versor.synth import (evolve, magnitude_locked_mutation,  # noqa: E402
                          get_vectors, run_points, shape_distance,
                          shape_fitness, tolerance_mask)

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_PNG = os.path.join(HERE, "..", "docs", "screenshots", "shapewrite.png")

TEAL = "#14b8a6"
GRAY = "#9ca3af"
INK = "#0b0b0b"
INK2 = "#52514e"
SURFACE = "#fcfcfb"

TARGET = np.array([[-5.0, 0.6, 0], [0.0, -0.6, 0], [4.0, 3.0, 0]])
EXPECTED = [5.0, 4.0, 3.0, 2.0, 1.0]


def norm2(p):
    p = p - p.mean(axis=0)
    return p / np.sqrt((p ** 2).sum(axis=1).mean())


def main():
    prog = countdown(5).build()
    locked = tolerance_mask(prog)
    print(f"magnitude-locked segments: {np.where(locked)[0].tolist()} "
          "(value-carrying, per the robustness map)")

    fit = shape_fitness(EXPECTED, TARGET)
    _, seed_pts = run_points(prog)
    print(f"seed shape distance: {shape_distance(seed_pts, TARGET):.3f}")

    best, hist = evolve(
        prog, fit, evaluator=lambda p: run_points(p, 2000),
        mutate=magnitude_locked_mutation(get_vectors(prog), locked),
        sigma=0.25, lam=32, generations=1000, seed=2, step_budget=2000)

    out, pts = run_points(best)
    print(f"evolved in {len(hist) - 1} generations: fitness {hist[-1]:.3f}")
    print(f"still prints: {[round(o, 6) for o in out]}")
    print(f"final shape distance: {shape_distance(pts, TARGET):.3f}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharex=True,
                             sharey=True)
    fig.patch.set_facecolor(SURFACE)
    for ax, p, title in ((axes[0], seed_pts, "before: countdown.vsr"),
                         (axes[1], pts, "after: same outputs, evolved shape")):
        ax.set_facecolor(SURFACE)
        t = norm2(TARGET)
        n = norm2(p)
        ax.plot(t[:, 0], t[:, 1], "--", lw=3.5, color=GRAY, label="target")
        ax.plot(n[:, 0], n[:, 1], "-o", ms=3.5, lw=1.9, color=TEAL,
                label="executed trace")
        ax.set_aspect("equal")
        ax.set_title(title, fontsize=11, color=INK)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.tick_params(colors=INK2, labelsize=8)
    axes[0].legend(frameon=False, fontsize=9, loc="upper left")
    fig.suptitle("Evolving the path without changing the program's output "
                 "— both are functions of the same geometry",
                 fontsize=11.5, color=INK)
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=140, facecolor=SURFACE, bbox_inches="tight")
    print(f"-> {os.path.normpath(OUT_PNG)}")


if __name__ == "__main__":
    main()
