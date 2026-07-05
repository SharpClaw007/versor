"""Continuous program-space demos: robustness map + evolutionary repair.

Usage: python examples/synthesize.py
Writes docs/screenshots/robustness.png and docs/screenshots/evolution.png.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np  # noqa: E402

from versor.decode import Cubic26  # noqa: E402
from versor.examples import countdown  # noqa: E402
from versor.isa import OPCODES  # noqa: E402
from versor.synth import (evolve, get_vectors, output_fitness,  # noqa: E402
                          run_out, segment_keys, segment_tolerances,
                          set_vectors)

HERE = os.path.dirname(os.path.abspath(__file__))
SHOTS = os.path.join(HERE, "..", "docs", "screenshots")

TEAL = "#14b8a6"
INK = "#0b0b0b"
INK2 = "#52514e"
SURFACE = "#fcfcfb"


def seg_label(prog, key):
    c, v, i = key
    seg = prog.chains[c].vertices[v][i].seg
    n = np.linalg.norm(seg)
    mnemonic = OPCODES[Cubic26().decode(seg / n)].mnemonic
    arm = f" arm{i}" if len(prog.chains[c].vertices[v]) > 1 else ""
    return f"{mnemonic}{arm} (v{v})"


def robustness_plot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    prog = countdown(5).build()
    tols = segment_tolerances(prog)
    labels = [seg_label(prog, k) for k, _ in tols]
    values = [t for _, t in tols]

    fig, ax = plt.subplots(figsize=(8, 3.6))
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    y = np.arange(len(values))
    ax.barh(y, values, height=0.62, color=TEAL)
    for yi, v in zip(y, values):
        ax.text(v + 0.01, yi, "0 — value-carrying" if v < 0.02 else f"{v:.2f}",
                va="center", fontsize=9,
                color=INK2 if v < 0.02 else INK)
    ax.set_yticks(y, labels, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("max perturbation radius preserving output", fontsize=9,
                  color=INK2)
    ax.set_title("countdown.vsr — per-segment robustness "
                 "(distance to the nearest behavior wall)",
                 fontsize=11, color=INK)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.tick_params(colors=INK2, length=0)
    fig.tight_layout()
    out = os.path.join(SHOTS, "robustness.png")
    fig.savefig(out, dpi=140, facecolor=SURFACE)
    plt.close(fig)
    print(f"-> {os.path.normpath(out)}")
    for lbl, v in zip(labels, values):
        print(f"   {lbl:16s} {v:.3f}")


def evolution_plot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    prog = countdown(3).build()
    rng = np.random.default_rng(5)
    x0 = get_vectors(prog)
    broken = set_vectors(prog, x0 + rng.normal(scale=0.15, size=x0.shape))
    print(f"scrambled program output: {run_out(broken, 2000)}")
    best, hist = evolve(broken, output_fitness([3.0, 2.0, 1.0]), seed=3,
                        lam=16, generations=250, sigma=0.15, step_budget=2000)
    print(f"evolved back in {len(hist) - 1} generations: "
          f"{run_out(best, 2000)}")

    fig, ax = plt.subplots(figsize=(8, 3.2))
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    ax.plot(hist, color=TEAL, lw=2)
    ax.set_xlabel("generation", fontsize=9, color=INK2)
    ax.set_ylabel("behavioral distance", fontsize=9, color=INK2)
    ax.set_title("(1+16) evolution strategy repairing a scrambled countdown "
                 "to printing 3 2 1", fontsize=11, color=INK)
    ax.annotate(f"exact behavior recovered\ngeneration {len(hist) - 1}",
                xy=(len(hist) - 1, hist[-1]), xytext=(-110, 28),
                textcoords="offset points", fontsize=9, color=INK,
                arrowprops={"arrowstyle": "->", "color": INK2})
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(colors=INK2)
    fig.tight_layout()
    out = os.path.join(SHOTS, "evolution.png")
    fig.savefig(out, dpi=140, facecolor=SURFACE)
    plt.close(fig)
    print(f"-> {os.path.normpath(out)}")


if __name__ == "__main__":
    robustness_plot()
    evolution_plot()
