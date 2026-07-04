import os

from versor import Machine, Trace
from versor.examples import straightline
from versor.viz import animate, render


def _traced():
    trace = Trace()
    m = Machine(straightline().build(), trace=trace)
    m.run()
    return trace, m


def test_render_writes_png(tmp_path):
    trace, _ = _traced()
    out = str(tmp_path / "t.png")
    assert render(trace, out, figsize=(4, 3.5)) == out
    assert os.path.getsize(out) > 0


def test_animate_writes_gif(tmp_path):
    trace, m = _traced()
    out = str(tmp_path / "t.gif")
    assert animate(trace, out, out=m.OUT, fps=4, dpi=40,
                   figsize=(4, 3.5), hold=2, spin=1.0) == out
    assert os.path.getsize(out) > 0
