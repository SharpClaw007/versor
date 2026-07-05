import struct

import numpy as np
import pytest

from versor import Machine, Trace
from versor.examples import countdown, helix
from versor.export import to_gcode, to_obj, to_stl, trace_points


@pytest.fixture(scope="module")
def tr():
    trace = Trace()
    Machine(countdown(3).build(), trace=trace).run()
    return trace


def test_trace_points_continuous(tr):
    pts = trace_points(tr)
    assert len(pts) > 5
    assert np.allclose(pts[0], 0)  # starts at origin


def test_gcode(tr, tmp_path):
    out = str(tmp_path / "t.nc")
    to_gcode(tr, out, feed=1200, scale=2.0)
    lines = open(out).read().splitlines()
    assert lines[1] == "G21 ; units: mm"
    g1 = [ln for ln in lines if ln.startswith("G1 ")]
    assert len(g1) == len(trace_points(tr)) - 1
    assert all("F1200" in ln for ln in g1)
    # final coordinate matches the (scaled) trace endpoint
    last = trace_points(tr)[-1] * 2.0
    assert f"X{last[0]:.4f}" in g1[-1]
    assert lines[-1].startswith("M2")


def test_obj(tr, tmp_path):
    out = str(tmp_path / "t.obj")
    to_obj(tr, out)
    lines = open(out).read().splitlines()
    verts = [ln for ln in lines if ln.startswith("v ")]
    poly = [ln for ln in lines if ln.startswith("l ")]
    assert len(verts) == len(trace_points(tr))
    assert len(poly) == 1
    assert len(poly[0].split()) == len(verts) + 1  # 'l' + one index per vertex


def test_stl_structure(tr, tmp_path):
    out = str(tmp_path / "t.stl")
    to_stl(tr, out, radius=0.2, sides=8)
    data = open(out, "rb").read()
    (count,) = struct.unpack_from("<I", data, 80)
    n_pts = len(trace_points(tr))
    assert count == (n_pts - 1) * 8 * 2 + 2 * 8
    assert len(data) == 84 + 50 * count
    # every float in the body is finite
    floats = np.frombuffer(data[84:], dtype=np.uint8)
    tris = np.frombuffer(data, dtype=np.dtype("<12f4, <u2"), offset=84)
    assert np.isfinite(np.vstack([t[0] for t in tris])).all()


def test_stl_handles_frame_rotating_paths(tmp_path):
    trace = Trace()
    Machine(helix().build(), trace=trace).run()
    out = str(tmp_path / "h.stl")
    to_stl(trace, out)
    data = open(out, "rb").read()
    (count,) = struct.unpack_from("<I", data, 80)
    assert count > 0 and len(data) == 84 + 50 * count
