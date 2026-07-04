"""M6 demo: lerp every segment between two extensionally-equal countdowns and
map what survives on the straight line between them.

Usage: python examples/interpolate.py [N_SAMPLES]

Writes docs/screenshots/interpolation.png and prints the outcome fractions.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np  # noqa: E402

from versor import Machine, Trace, classify, lerp_programs  # noqa: E402
from versor.examples import countdown, countdown_b  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_PNG = os.path.join(HERE, "..", "docs", "screenshots", "interpolation.png")

# status roles (dataviz reference palette): a color never carries meaning
# alone — every span also gets a direct label, and the legend names each role
STATUS = {
    "equivalent": ("#0ca30c", "equivalent, A's exact opcode path"),
    "mutated": ("#fab219", "equivalent, different opcode path"),
    "diverged": ("#ec835a", "halts, wrong output"),
    "fault": ("#d03b3b", "fault"),
}
INK = "#0b0b0b"
INK_2 = "#52514e"
SURFACE = "#fcfcfb"


def sample(n_samples: int):
    a = countdown(5).build()
    b = countdown_b(5).build()
    expected = Machine(a).run().out

    base_trace = Trace()
    Machine(countdown(5).build(), trace=base_trace).run()
    baseline_ops = base_trace.opcodes()

    ts = np.linspace(0.0, 1.0, n_samples)
    statuses = []
    for t in ts:
        r = classify(lerp_programs(a, b, float(t)), expected)
        if r["status"] == "equivalent":
            statuses.append("equivalent" if r["opcodes"] == baseline_ops
                            else "mutated")
        elif r["status"] == "diverged":
            statuses.append("diverged")
        else:
            statuses.append("fault")
    return ts, statuses


def runs(ts, statuses):
    """Contiguous same-status spans as (t_start, t_end, status)."""
    half = (ts[1] - ts[0]) / 2.0
    spans, start = [], 0
    for i in range(1, len(statuses) + 1):
        if i == len(statuses) or statuses[i] != statuses[start]:
            t0 = max(0.0, ts[start] - half)
            t1 = min(1.0, ts[i - 1] + half)
            spans.append((t0, t1, statuses[start]))
            start = i
    return spans


def plot(spans, fractions, n_samples):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    fig, ax = plt.subplots(figsize=(12, 3.0))
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    for t0, t1, st in spans:
        color, _ = STATUS[st]
        # hairline surface gap between spans: the boundary is a cone crossing
        ax.axvspan(t0 + 0.001, t1 - 0.001, ymin=0.28, ymax=0.72,
                   color=color, lw=0)
        # relief rule: every span carries a visible direct label
        mid = (t0 + t1) / 2.0
        wide = (t1 - t0) > 0.12
        label = f"{st}\n{100 * (t1 - t0):.0f}%" if wide else f"{100 * (t1 - t0):.0f}%"
        ax.text(mid, 0.80, label, ha="center", va="bottom",
                fontsize=9, color=INK, linespacing=1.3)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.3)
    ax.set_yticks([])
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(["t = 0\ncountdown A", "0.25", "0.5", "0.75",
                        "t = 1\ncountdown B"], fontsize=9, color=INK_2)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(colors=INK_2, length=3)

    ok = fractions.get("equivalent", 0) + fractions.get("mutated", 0)
    ax.set_title(
        f"Straight-line interpolation between two equivalent countdowns — "
        f"{100 * ok:.1f}% of the line still prints 5 4 3 2 1"
        f"  ({n_samples} samples)",
        fontsize=11, color=INK, pad=14)

    handles = [Patch(color=c, label=lbl) for st, (c, lbl) in STATUS.items()
               if any(s == st for _, _, s in spans)]
    ax.legend(handles=handles, loc="lower center", ncol=len(handles),
              frameon=False, fontsize=9, bbox_to_anchor=(0.5, -0.42))

    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=140, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


def main():
    n_samples = int(sys.argv[1]) if len(sys.argv) > 1 else 401
    ts, statuses = sample(n_samples)
    fractions = {st: statuses.count(st) / len(statuses)
                 for st in dict.fromkeys(statuses)}
    spans = runs(ts, statuses)

    for st, frac in fractions.items():
        print(f"{st:12s} {100 * frac:5.1f}%")
    print("spans:")
    for t0, t1, st in spans:
        print(f"  [{t0:.3f}, {t1:.3f}]  {st}")
    plot(spans, fractions, n_samples)
    print(f"-> {os.path.normpath(OUT_PNG)}")


if __name__ == "__main__":
    main()
