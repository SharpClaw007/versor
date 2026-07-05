"""Physical export: a trace is a polyline, so a run is a toolpath.

- ``to_gcode``: the executed path as G-code (G21/G90, G0 to the start, one
  G1 per segment) — run a program on a pen plotter or CNC and the machine
  physically walks the computation.
- ``to_obj``: the path as an OBJ polyline (``l`` element), for DCC tools.
- ``to_stl``: the path as a watertight-ish tube mesh (parallel-transport
  frames, so the tube doesn't twist), for 3D printing a run.

All exporters take the trace of an execution, not the program: what you
print is what actually ran, branches taken and loops unrolled.
"""
from __future__ import annotations

import struct

import numpy as np


def trace_points(trace) -> np.ndarray:
    """The executed path as an ordered point list (consecutive dupes merged)."""
    pts = []
    for r in trace:
        if not pts:
            pts.append(np.asarray(r.P0, dtype=float))
        if not np.allclose(r.P1, pts[-1]):
            pts.append(np.asarray(r.P1, dtype=float))
    if len(pts) < 2:
        raise ValueError("trace has no movement to export")
    return np.array(pts)


def to_gcode(trace, path: str, *, feed: float = 600.0, scale: float = 1.0,
             decimals: int = 4) -> str:
    pts = trace_points(trace) * scale
    fmt = f"{{:.{decimals}f}}"

    def xyz(p):
        return (f"X{fmt.format(p[0])} Y{fmt.format(p[1])} "
                f"Z{fmt.format(p[2])}")

    lines = [
        "; versor execution trace",
        "G21 ; units: mm",
        "G90 ; absolute coordinates",
        f"G0 {xyz(pts[0])}",
    ]
    lines += [f"G1 {xyz(p)} F{feed:g}" for p in pts[1:]]
    lines.append("M2 ; end of program")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    return path


def to_obj(trace, path: str, *, scale: float = 1.0) -> str:
    pts = trace_points(trace) * scale
    with open(path, "w") as f:
        f.write("# versor execution trace polyline\n")
        for p in pts:
            f.write(f"v {p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n")
        f.write("l " + " ".join(str(i + 1) for i in range(len(pts))) + "\n")
    return path


def _transport_frames(pts: np.ndarray):
    """Unit tangent + parallel-transported normal/binormal per vertex."""
    segs = np.diff(pts, axis=0)
    seg_t = segs / np.linalg.norm(segs, axis=1, keepdims=True)
    # vertex tangents: averaged at interior vertices
    t = np.vstack([seg_t[:1], seg_t[:-1] + seg_t[1:], seg_t[-1:]])
    norms = np.linalg.norm(t, axis=1, keepdims=True)
    # a hairpin (t_i = -t_{i+1}) averages to zero; fall back to the segment
    bad = norms[:, 0] < 1e-9
    t[bad] = np.vstack([seg_t[:1], seg_t])[bad]
    t /= np.linalg.norm(t, axis=1, keepdims=True)

    seed = np.array([0.0, 0.0, 1.0])
    if abs(float(t[0] @ seed)) > 0.9:
        seed = np.array([0.0, 1.0, 0.0])
    n = seed - (seed @ t[0]) * t[0]
    n /= np.linalg.norm(n)
    normals = [n]
    for i in range(1, len(t)):
        n = normals[-1] - (normals[-1] @ t[i]) * t[i]
        ln = np.linalg.norm(n)
        if ln < 1e-9:  # tangent flipped; re-seed
            s = np.array([0.0, 0.0, 1.0])
            if abs(float(t[i] @ s)) > 0.9:
                s = np.array([0.0, 1.0, 0.0])
            n = s - (s @ t[i]) * t[i]
            ln = np.linalg.norm(n)
        normals.append(n / ln)
    normals = np.array(normals)
    binormals = np.cross(t, normals)
    return t, normals, binormals


def to_stl(trace, path: str, *, radius: float = 0.15, sides: int = 8,
           scale: float = 1.0) -> str:
    pts = trace_points(trace) * scale
    t, n, b = _transport_frames(pts)

    theta = 2 * np.pi * np.arange(sides) / sides
    rings = (pts[:, None, :]
             + radius * (np.cos(theta)[None, :, None] * n[:, None, :]
                         + np.sin(theta)[None, :, None] * b[:, None, :]))

    tris = []
    for i in range(len(pts) - 1):
        for j in range(sides):
            k = (j + 1) % sides
            a0, a1 = rings[i, j], rings[i, k]
            b0, b1 = rings[i + 1, j], rings[i + 1, k]
            tris.append((a0, b0, b1))
            tris.append((a0, b1, a1))
    for j in range(sides):  # end caps (fans around the path endpoints)
        k = (j + 1) % sides
        tris.append((pts[0], rings[0, k], rings[0, j]))
        tris.append((pts[-1], rings[-1, j], rings[-1, k]))

    with open(path, "wb") as f:
        f.write(b"versor execution trace tube".ljust(80, b"\0"))
        f.write(struct.pack("<I", len(tris)))
        for v0, v1, v2 in tris:
            normal = np.cross(v1 - v0, v2 - v0)
            ln = np.linalg.norm(normal)
            normal = normal / ln if ln > 1e-12 else np.zeros(3)
            f.write(struct.pack("<3f", *normal))
            f.write(struct.pack("<9f", *v0, *v1, *v2))
            f.write(struct.pack("<H", 0))
    return path
