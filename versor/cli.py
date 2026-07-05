"""CLI: python -m versor run prog.vsr [--trace out.png] [--animate out.gif] [--steps N]"""
from __future__ import annotations

import argparse
import sys

from .decode import DECODERS
from .errors import LoadError, VersorFault
from .loader import load
from .machine import DEFAULT_STEP_BUDGET, Machine, RunResult
from .trace import Trace


def _load_any(path: str):
    """Load .vsr directly; assemble .vasm / compile .vhl on the fly."""
    if path.endswith(".vasm"):
        from .asm import assemble_path
        return assemble_path(path).build()
    if path.endswith(".vhl"):
        from .vhl import compile_path
        return compile_path(path).build()
    return load(path)


def _print_result(res: RunResult) -> None:
    text = res.out_text()
    if text:
        print(text)
    d = res.displacement
    print(f"-- halted: {res.halt_reason} after {res.steps} steps", file=sys.stderr)
    print(f"-- net displacement: ({d[0]:.6g}, {d[1]:.6g}, {d[2]:.6g})", file=sys.stderr)


def cmd_run(args) -> int:
    try:
        prog = _load_any(args.file)
    except LoadError as e:
        print(f"load error: {e}", file=sys.stderr)
        return 2
    for w in prog.warnings:
        print(f"warning: {w}", file=sys.stderr)

    trace = Trace() if (args.trace or args.animate) else None
    inp = None
    if args.input is not None:
        inp = [float(v) for v in args.input.split(",") if v.strip()]
    elif args.input_text is not None:
        inp = args.input_text
    m = Machine(prog, step_budget=args.steps, trace=trace,
                decoder=args.decoder, input=inp)
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
            gif = viz.animate(trace, args.animate, title=title, out=m.OUT,
                              fps=args.fps, spin=args.spin)
            print(f"-- animation: {gif}", file=sys.stderr)
    return code


def cmd_lint(args) -> int:
    try:
        prog = _load_any(args.file)
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


def cmd_asm(args) -> int:
    import os

    from .asm import assemble_path
    try:
        pb = assemble_path(args.file)
        out = args.out or os.path.splitext(args.file)[0] + ".vsr"
        prog = pb.save(out)
    except LoadError as e:
        print(f"asm error: {e}", file=sys.stderr)
        return 2
    for w in prog.warnings:
        print(f"warning: {w}", file=sys.stderr)
    n_segs = sum(len(es) for ch in prog.chains for es in ch.vertices.values())
    print(f"{out}: {len(prog.chains)} chain(s), {n_segs} segment(s), "
          f"decoder {prog.decoder}")
    return 0


def cmd_vhl(args) -> int:
    import os

    from .vhl import compile_path
    try:
        pb = compile_path(args.file)
        out = args.out or os.path.splitext(args.file)[0] + ".vsr"
        prog = pb.save(out)
    except LoadError as e:
        print(f"vhl error: {e}", file=sys.stderr)
        return 2
    n_segs = sum(len(es) for ch in prog.chains for es in ch.vertices.values())
    print(f"{out}: {n_segs} segment(s)")
    return 0


def cmd_export(args) -> int:
    if not (args.gcode or args.obj or args.stl):
        print("export: pass at least one of --gcode/--obj/--stl", file=sys.stderr)
        return 2
    try:
        prog = _load_any(args.file)
    except LoadError as e:
        print(f"load error: {e}", file=sys.stderr)
        return 2
    from . import export
    trace = Trace()
    m = Machine(prog, step_budget=args.steps, trace=trace)
    try:
        m.run()
    except VersorFault as f:
        print(f"-- {f} (exporting the partial trace)", file=sys.stderr)
    if args.gcode:
        print(export.to_gcode(trace, args.gcode, feed=args.feed,
                              scale=args.scale))
    if args.obj:
        print(export.to_obj(trace, args.obj, scale=args.scale))
    if args.stl:
        print(export.to_stl(trace, args.stl, radius=args.radius,
                            scale=args.scale))
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="versor",
                                description="Versor: the program is the path.")
    sub = p.add_subparsers(dest="cmd", required=True)

    pr = sub.add_parser("run", help="run a .vsr program")
    pr.add_argument("file")
    pr.add_argument("--trace", metavar="OUT.png", help="render executed path")
    pr.add_argument("--animate", metavar="OUT.gif",
                    help="render an execution animation (cursor, frame triad, HUD)")
    pr.add_argument("--fps", type=int, default=12, help="animation frames/sec")
    pr.add_argument("--spin", type=float, default=0.0,
                    help="camera degrees per animation frame")
    pr.add_argument("--steps", type=int, default=DEFAULT_STEP_BUDGET,
                    help="step budget (default 1e6)")
    pr.add_argument("--decoder", choices=sorted(DECODERS), default=None,
                    help="override the program's decoder")
    pr.add_argument("--input", default=None,
                    help="comma-separated scalars for INP")
    pr.add_argument("--input-text", default=None,
                    help="string whose char codes feed INP")
    pr.set_defaults(fn=cmd_run)

    pl = sub.add_parser("lint", help="validate a .vsr/.vasm program")
    pl.add_argument("file")
    pl.set_defaults(fn=cmd_lint)

    pe = sub.add_parser("export", help="export an execution trace as a toolpath/mesh")
    pe.add_argument("file")
    pe.add_argument("--gcode", metavar="OUT.nc", help="G-code toolpath")
    pe.add_argument("--obj", metavar="OUT.obj", help="OBJ polyline")
    pe.add_argument("--stl", metavar="OUT.stl", help="tube mesh for printing")
    pe.add_argument("--scale", type=float, default=1.0, help="unit scale")
    pe.add_argument("--feed", type=float, default=600.0, help="G1 feed rate")
    pe.add_argument("--radius", type=float, default=0.15, help="STL tube radius")
    pe.add_argument("--steps", type=int, default=DEFAULT_STEP_BUDGET)
    pe.set_defaults(fn=cmd_export)

    pa = sub.add_parser("asm", help="assemble a .vasm file to .vsr")
    pa.add_argument("file")
    pa.add_argument("-o", "--out", default=None,
                    help="output path (default: input with .vsr extension)")
    pa.set_defaults(fn=cmd_asm)

    pl2 = sub.add_parser("lsp", help="start the language server (stdio)")
    pl2.set_defaults(fn=lambda args: __import__(
        "versor.lsp", fromlist=["main"]).main())

    pv = sub.add_parser("vhl", help="compile a .vhl file to .vsr")
    pv.add_argument("file")
    pv.add_argument("-o", "--out", default=None,
                    help="output path (default: input with .vsr extension)")
    pv.set_defaults(fn=cmd_vhl)

    args = p.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
