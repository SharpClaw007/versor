"""Program interpolation (M6): lerp segments between two topology-equal
programs and see which fraction of the in-betweens still work.

Two programs are *extensionally equal* when they produce the same output; the
interesting question is what lies on the straight line between two different
implementations. Each interpolant lerps every raw segment vector; guards and
graph structure must match exactly (they are the topology, not the geometry).
"""
from __future__ import annotations

import numpy as np

from .errors import VersorFault
from .loader import Chain, Edge, Program
from .machine import Machine
from .trace import Trace


def lerp_programs(a: Program, b: Program, t: float) -> Program:
    """Segment-wise linear interpolation: (1-t)*a + t*b.

    Raises ValueError unless a and b share decoder, chain/vertex/edge
    structure, and guards.
    """
    if a.decoder != b.decoder:
        raise ValueError(f"decoder mismatch: {a.decoder!r} vs {b.decoder!r}")
    if len(a.chains) != len(b.chains):
        raise ValueError(f"chain count mismatch: {len(a.chains)} vs {len(b.chains)}")

    chains = []
    for ca, cb in zip(a.chains, b.chains):
        if set(ca.vertices) != set(cb.vertices):
            raise ValueError(f"chain {ca.id}: vertex sets differ")
        vertices: dict[int, list[Edge]] = {}
        for vid in ca.vertices:
            ea, eb = ca.vertices[vid], cb.vertices[vid]
            if len(ea) != len(eb):
                raise ValueError(f"chain {ca.id} vertex {vid}: edge counts differ")
            edges = []
            for i, (x, y) in enumerate(zip(ea, eb)):
                if x.to != y.to:
                    raise ValueError(
                        f"chain {ca.id} vertex {vid} edge {i}: targets differ")
                gx, gy = x.guard is not None, y.guard is not None
                if gx != gy or (gx and not np.allclose(x.guard, y.guard)):
                    raise ValueError(
                        f"chain {ca.id} vertex {vid} edge {i}: guards differ")
                seg = (1.0 - t) * x.seg + t * y.seg
                edges.append(Edge(seg=seg, to=x.to,
                                  guard=None if x.guard is None else x.guard.copy()))
            vertices[vid] = edges
        chains.append(Chain(id=ca.id, vertices=vertices, comment=ca.comment))

    return Program(chains=chains, name=f"lerp({a.name}, {b.name}, {t:.4g})",
                   version=a.version, decoder=a.decoder)


def _out_equal(x: list, y: list, tol: float = 1e-9) -> bool:
    if len(x) != len(y):
        return False
    for p, q in zip(x, y):
        if isinstance(p, str) or isinstance(q, str):
            if p != q:
                return False
        elif abs(p - q) > tol:
            return False
    return True


def classify(prog: Program, expected_out: list, *,
             step_budget: int = 100_000) -> dict:
    """Run an interpolant and report what became of it.

    status: 'equivalent' (same output), 'diverged' (halted, different
    output), or 'fault' (with the fault kind in 'detail'). 'opcodes' carries
    the executed mnemonic sequence so callers can tell an equivalent-but-
    mutated path from a verbatim one.
    """
    trace = Trace()
    m = Machine(prog, trace=trace, step_budget=step_budget)
    try:
        res = m.run()
    except VersorFault as f:
        return {"status": "fault", "detail": f.kind, "opcodes": trace.opcodes()}
    ok = _out_equal(list(res.out), list(expected_out))
    return {"status": "equivalent" if ok else "diverged",
            "detail": res.halt_reason, "opcodes": trace.opcodes()}
