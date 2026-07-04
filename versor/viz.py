"""3D rendering of execution traces — the language's debugger.

Segments are colored by opcode class (data=teal, arithmetic=coral,
frame=purple, control=gray). The dashed black line is the net displacement.
Branch vertices are marked with black diamonds; skipped segments are dotted.
"""
from __future__ import annotations

import numpy as np

CLASS_COLORS = {
    "data": "#14b8a6",        # teal
    "arithmetic": "#ff7f50",  # coral
    "frame": "#a855f7",       # purple
    "control": "#6b7280",     # gray
}


def _axes3d(title: str, figsize=(9, 8)):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    return fig, ax


def _finalize(fig, ax, points: np.ndarray):
    import matplotlib.lines as mlines

    lo = points.min(axis=0)
    hi = points.max(axis=0)
    center = (lo + hi) / 2
    span = max(float((hi - lo).max()), 1.0) * 0.55
    ax.set_xlim(center[0] - span, center[0] + span)
    ax.set_ylim(center[1] - span, center[1] + span)
    ax.set_zlim(center[2] - span, center[2] + span)
    ax.set_box_aspect((1, 1, 1))

    handles = [mlines.Line2D([], [], color=c, lw=2.5, label=k)
               for k, c in CLASS_COLORS.items()]
    handles.append(mlines.Line2D([], [], color="black", lw=1.2, ls="--",
                                 label="net displacement"))
    handles.append(mlines.Line2D([], [], color="black", marker="D", ls="",
                                 label="branch vertex"))
    ax.legend(handles=handles, loc="upper left", fontsize=8)
    fig.tight_layout()


def _segments(trace):
    """(P0, P1, color, style) per executed step; skips zero-length RET*."""
    for r in trace:
        if np.allclose(r.P0, r.P1):
            continue
        yield r


def render(trace, out_path: str, title: str = "versor trace",
           elev: float = 22.0, azim: float = -60.0,
           figsize=(9, 8)) -> str:
    fig, ax = _axes3d(title, figsize)

    pts = [np.zeros(3)]
    for r in _segments(trace):
        color = CLASS_COLORS.get(r.klass, "black")
        style = ":" if r.skipped else "-"
        ax.plot(*zip(r.P0, r.P1), color=color, ls=style,
                lw=1.6 if r.skipped else 2.4, alpha=0.9)
        if r.branch:
            ax.scatter(*r.P0, color="black", marker="D", s=28, zorder=5)
        pts.extend([r.P0, r.P1])

    if trace.records:
        start = trace.records[0].P0
        end = trace.records[-1].P1
        ax.plot(*zip(start, end), color="black", ls="--", lw=1.2, alpha=0.8)
        ax.scatter(*start, color="#16a34a", s=60, zorder=6)   # start: green
        ax.scatter(*end, color="#dc2626", marker="s", s=60, zorder=6)  # end: red

    _finalize(fig, ax, np.array(pts))
    ax.view_init(elev=elev, azim=azim)
    fig.savefig(out_path, dpi=140)
    import matplotlib.pyplot as plt
    plt.close(fig)
    return out_path


def animate(trace, out_path: str, title: str = "versor trace",
            fps: int = 12) -> str:
    """Growing-path animation, saved as GIF (pillow writer)."""
    from matplotlib.animation import FuncAnimation, PillowWriter

    fig, ax = _axes3d(title)
    records = list(_segments(trace))
    if not records:
        raise ValueError("empty trace, nothing to animate")

    pts = [np.zeros(3)]
    for r in records:
        pts.extend([r.P0, r.P1])
    _finalize(fig, ax, np.array(pts))

    drawn = []

    def frame(i):
        r = records[i]
        color = CLASS_COLORS.get(r.klass, "black")
        line, = ax.plot(*zip(r.P0, r.P1), color=color,
                        ls=":" if r.skipped else "-", lw=2.4)
        drawn.append(line)
        if r.branch:
            drawn.append(ax.scatter(*r.P0, color="black", marker="D", s=28))
        return drawn

    anim = FuncAnimation(fig, frame, frames=len(records), interval=1000 / fps)
    anim.save(out_path, writer=PillowWriter(fps=fps))
    import matplotlib.pyplot as plt
    plt.close(fig)
    return out_path
