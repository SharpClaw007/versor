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


TRIAD_COLORS = ("#dc2626", "#16a34a", "#2563eb")  # frame x, y, z


def _fmt_vec(v) -> str:
    return f"({v[0]:6.2f}, {v[1]:6.2f}, {v[2]:6.2f})"


def _fmt_out(buf, width: int = 34) -> str:
    parts = []
    for item in buf:
        parts.append(item if isinstance(item, str) else f"{item:g} ")
    s = "".join(parts).replace("\n", "\\n")
    return s if len(s) <= width else "…" + s[-(width - 1):]


def animate(trace, out_path: str, title: str = "versor trace",
            out: list | None = None, fps: int = 12, dpi: int = 80,
            figsize=(8.0, 7.0), hold: int = 10, spin: float = 0.0,
            elev: float = 22.0, azim: float = -60.0) -> str:
    """Execution animation, saved as GIF.

    Per step: the path grows (colored by opcode class, current segment
    highlighted), a cursor marks the machine's position, a small RGB triad
    shows the live frame F's local axes, and a HUD reports step / opcode /
    accumulator / OUT buffer. `out` is the machine's final OUT buffer (each
    record's out_len indexes into it); `hold` repeats the final state; `spin`
    rotates the camera that many degrees per frame.
    """
    from matplotlib.animation import FuncAnimation, PillowWriter

    from .quat import Quat

    records = list(trace)
    if not records:
        raise ValueError("empty trace, nothing to animate")

    fig, ax = _axes3d(title, figsize)
    pts = [records[0].P0.copy()]
    for r in records:
        pts.extend([r.P0, r.P1])
    pts = np.array(pts)
    _finalize(fig, ax, pts)
    ax.view_init(elev=elev, azim=azim)
    triad_len = 0.09 * max(float((pts.max(axis=0) - pts.min(axis=0)).max()), 1.0)

    ax.scatter(*records[0].P0, color="#16a34a", s=60, zorder=6)
    cursor, = ax.plot([records[0].P0[0]], [records[0].P0[1]],
                      [records[0].P0[2]], marker="o", ms=9, color="black",
                      zorder=7)
    triad = [ax.plot([], [], [], color=c, lw=2.2, zorder=6)[0]
             for c in TRIAD_COLORS]
    hud = fig.text(0.98, 0.955, "", family="monospace", fontsize=9,
                   va="top", ha="right")
    finished = []   # artists added during the hold frames
    prev_seg = []   # the currently-highlighted segment line

    def frame(i):
        nonlocal prev_seg
        if spin:
            ax.view_init(elev=elev, azim=azim + spin * i)
        if i >= len(records):        # hold: show the ending
            if not finished:
                start, end = records[0].P0, records[-1].P1
                finished.append(ax.plot(*zip(start, end), color="black",
                                        ls="--", lw=1.2, alpha=0.8)[0])
                finished.append(ax.scatter(*end, color="#dc2626", marker="s",
                                           s=60, zorder=6))
            return []

        r = records[i]
        for line in prev_seg:        # de-highlight the previous segment
            line.set_linewidth(2.4)
            line.set_alpha(0.85)
        prev_seg = []
        if not np.allclose(r.P0, r.P1):
            color = CLASS_COLORS.get(r.klass, "black")
            line, = ax.plot(*zip(r.P0, r.P1), color=color,
                            ls=":" if r.skipped else "-", lw=4.2, alpha=1.0)
            prev_seg = [line]
            if r.branch:
                ax.scatter(*r.P0, color="black", marker="D", s=28, zorder=5)

        cursor.set_data_3d([r.P1[0]], [r.P1[1]], [r.P1[2]])
        f = Quat(*r.F)
        for axis_line, axis in zip(triad, np.eye(3)):
            tip = r.P1 + triad_len * f.rotate(axis)
            axis_line.set_data_3d(*zip(r.P1, tip))

        note = " (skipped)" if r.skipped else ""
        emitted = (out or [])[:r.out_len]
        hud.set_text(
            f"step {r.step:>4}/{len(records)}  chain {r.chain}  "
            f"{r.opcode:<6} n={r.n:<7.3f}{note}\n"
            f"A = {_fmt_vec(r.A)}   P = {_fmt_vec(r.P1)}\n"
            f"OUT: {_fmt_out(emitted)}")
        return []

    anim = FuncAnimation(fig, frame, frames=len(records) + hold,
                         interval=1000 / fps)
    anim.save(out_path, writer=PillowWriter(fps=fps), dpi=dpi)
    import matplotlib.pyplot as plt
    plt.close(fig)
    return out_path
