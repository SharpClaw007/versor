"""Regenerate the example .vsr files and their trace renders.

Usage: python examples/make_examples.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from versor import Machine, Trace  # noqa: E402
from versor.examples import ALL  # noqa: E402
from versor.viz import render  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RENDERS = os.path.join(HERE, "renders")
SCREENSHOTS = os.path.join(HERE, "..", "docs", "screenshots")


def main():
    os.makedirs(RENDERS, exist_ok=True)
    os.makedirs(SCREENSHOTS, exist_ok=True)
    for name, fn in ALL.items():
        builder = fn()
        path = os.path.join(HERE, f"{name}.vsr")
        prog = builder.save(path)
        trace = Trace()
        res = Machine(prog, trace=trace).run()
        png = render(trace, os.path.join(RENDERS, f"{name}.png"), title=name)
        out = "".join(str(o) if isinstance(o, str) else f"{o:.6g} "
                      for o in res.out).strip()
        print(f"{name:14s} {res.steps:4d} steps  out={out!r}  -> {png}")

    # wide hero shot for the README masthead
    trace = Trace()
    Machine(ALL["helix"]().build(), trace=trace).run()
    hero = render(trace, os.path.join(SCREENSHOTS, "hero-helix.png"),
                  title="helix.vsr — one local program, corkscrewed by its frame",
                  elev=16, azim=-55, figsize=(14, 6.5))
    print(f"{'hero':14s} -> {hero}")


if __name__ == "__main__":
    main()
