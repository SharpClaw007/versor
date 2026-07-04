"""CLI: python -m versor run prog.vsr [--trace out.png] [--animate out.gif] [--steps N]"""
from __future__ import annotations

import argparse
import sys

from .errors import LoadError, VersorFault
from .loader import load
from .machine import DEFAULT_STEP_BUDGET, Machine, RunResult
from .trace import Trace


def _print_result(res: RunResult) -> None:
    text = res.out_text()
    if text:
        print(text)
    d = res.displacement
    print(f"-- halted: {res.halt_reason} after {res.steps} steps", file=sys.stderr)
    print(f"-- net displacement: ({d[0]:.6g}, {d[1]:.6g}, {d[2]:.6g})", file=sys.stderr)


def cmd_run(args) -> int:
    try:
        prog = load(args.file)
    except LoadError as e:
        print(f"load error: {e}", file=sys.stderr)
        return 2
    for w in prog.warnings:
        print(f"warning: {w}", file=sys.stderr)

    trace = Trace() if (args.trace or args.animate) else None
    m = Machine(prog, step_budget=args.steps, trace=trace)
    code = 0
    try:
        res = m.run()
        _print_result(res)
    except VersorFault as f:
        if m.OUT:
            print(RunResult(m.OUT, m.P, "", m.steps).out_text())
        print(f"-- {f}", file=sys.stderr)
        code = 1

    if trace is not None and len(trace):
        from . import viz
        title = prog.name or args.file
        if args.trace:
            print(f"-- trace image: {viz.render(trace, args.trace, title=title)}",
                  file=sys.stderr)
        if args.animate:
            print(f"-- animation: {viz.animate(trace, args.animate, title=title)}",
                  file=sys.stderr)
    return code


def cmd_lint(args) -> int:
    try:
        prog = load(args.file)
    except LoadError as e:
        print(f"load error: {e}", file=sys.stderr)
        return 2
    for w in prog.warnings:
        print(f"warning: {w}")
    n_chains = len(prog.chains)
    n_segs = sum(len(es) for ch in prog.chains for es in ch.vertices.values())
    print(f"ok: {prog.name or args.file} — {n_chains} chain(s), {n_segs} segment(s), "
          f"{len(prog.warnings)} warning(s)")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="versor",
                                description="Versor: the program is the path.")
    sub = p.add_subparsers(dest="cmd", required=True)

    pr = sub.add_parser("run", help="run a .vsr program")
    pr.add_argument("file")
    pr.add_argument("--trace", metavar="OUT.png", help="render executed path")
    pr.add_argument("--animate", metavar="OUT.gif", help="animate executed path")
    pr.add_argument("--steps", type=int, default=DEFAULT_STEP_BUDGET,
                    help="step budget (default 1e6)")
    pr.set_defaults(fn=cmd_run)

    pl = sub.add_parser("lint", help="validate a .vsr program")
    pl.add_argument("file")
    pl.set_defaults(fn=cmd_lint)

    args = p.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
