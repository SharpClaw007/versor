"""Regenerate docs/playground/test/golden.json: reference outputs from the
Python interpreter that the JS port must reproduce.

Usage: python tools/make_golden.py
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from versor import Machine, Trace, to_dict  # noqa: E402
from versor.examples import ALL, countdown  # noqa: E402
from versor.minsky import DecJz, Inc, compile_minsky  # noqa: E402

OUT = os.path.join(os.path.dirname(__file__), "..", "docs", "playground",
                   "test", "golden.json")
EXAMPLES = os.path.join(os.path.dirname(__file__), "..", "examples")


def record(name, prog, vasm=None):
    trace = Trace()
    m = Machine(prog, trace=trace)
    res = m.run()
    entry = {
        "name": name,
        "program": to_dict(prog),
        "expected": {
            "out": list(res.out),
            "haltReason": res.halt_reason,
            "steps": res.steps,
            "displacement": [float(c) for c in res.displacement],
            "opcodes": trace.opcodes(),
        },
    }
    if vasm is not None:
        entry["vasm"] = vasm
    return entry


def main():
    entries = []
    for name, fn in ALL.items():
        entries.append(record(name, fn().build()))
    entries.append(record("countdown-icosa32",
                          countdown(4, decoder="icosa32").build()))
    entries.append(record("countdown-sphere26",
                          countdown(4, decoder="sphere26").build()))
    entries.append(record(
        "minsky-transfer",
        compile_minsky([DecJz(2, next=1, zero=2), Inc(1, next=0)],
                       c1=3, c2=4, emit_counters=True).build()))

    from versor import ProgramBuilder
    b = ProgramBuilder("exec-trampoline")
    c = b.chain()
    c.loadi(9).store(0.5).op("MOVR", 2.0).exec_cell(2.5).out().halt()
    entries.append(record("exec-trampoline", b.build()))

    for vasm_name in ("countdown", "add_two"):
        path = os.path.join(EXAMPLES, f"{vasm_name}.vasm")
        with open(path) as f:
            src = f.read()
        from versor.asm import assemble
        entries.append(record(f"vasm-{vasm_name}", assemble(src).build(),
                              vasm=src))

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(entries, f)
    print(f"{OUT}: {len(entries)} golden programs")


if __name__ == "__main__":
    main()
