"""Generate the sphere32 direction table: the Versor-32 counterpart of
sphere26 — 16 antipodal lines carrying the 26 base opcodes plus the six
extended ones (INP/SWAP, PUSHA/POPA, MULR/LOADP as antipodal pairs).

Same method as tools/optimize_sphere26.py: annealed, direction-normalized
Riesz repulsion from many random starts plus a seeded start (cubic-26 cone
centers + icosa32's extended directions), then greedy + 2-opt |dot| label
matching against the seeds.

Usage: python tools/optimize_sphere32.py [restarts]
"""
import itertools
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np  # noqa: E402

from versor.decode import PHI  # noqa: E402

S = 12.0
STEPS = 15000
LR0 = 0.004

TRIPLES = [t for t in itertools.product((-1, 0, 1), repeat=3) if t != (0, 0, 0)]
REP_TRIPLES = []
_seen = set()
for _t in TRIPLES:
    if tuple(-c for c in _t) not in _seen:
        _seen.add(_t)
        REP_TRIPLES.append(_t)

EXT_SEEDS = [((PHI, -1 / PHI, 0), "INP"),
             ((0, PHI, -1 / PHI), "PUSHA"),
             ((-1 / PHI, 0, PHI), "MULR")]
N_LINES = 16


def unit(v):
    v = np.asarray(v, dtype=float)
    return v / np.linalg.norm(v)


def full(x):
    return np.vstack([x, -x])


def min_angle_deg(pts):
    d = np.clip(pts @ pts.T, -1, 1)
    np.fill_diagonal(d, -1)
    return math.degrees(math.acos(float(d.max())))


def anneal(x):
    for step in range(STEPS):
        pts = full(x)
        d = pts[:, None, :] - pts[None, :, :]
        r2 = np.einsum("ijk,ijk->ij", d, d)
        np.fill_diagonal(r2, np.inf)
        f = (d / (r2 ** ((S + 2) / 2))[:, :, None]).sum(axis=1)
        g = f[:N_LINES] - f[N_LINES:]
        g -= (np.einsum("ij,ij->i", g, x))[:, None] * x
        g /= np.maximum(np.linalg.norm(g, axis=1, keepdims=True), 1e-12)
        x = x + LR0 * (1.0 - step / STEPS) * g
        x /= np.linalg.norm(x, axis=1, keepdims=True)
    return x


def assign_labels(x, seeds, labels):
    score = np.abs(x @ seeds.T)
    perm = [-1] * N_LINES
    used = set()
    for i in np.argsort(-score.max(axis=1)):
        j = max((j for j in range(N_LINES) if j not in used),
                key=lambda j: score[i, j])
        perm[i] = j
        used.add(j)
    improved = True
    while improved:
        improved = False
        for a in range(N_LINES):
            for b in range(a + 1, N_LINES):
                if (score[a, perm[b]] + score[b, perm[a]]
                        > score[a, perm[a]] + score[b, perm[b]] + 1e-12):
                    perm[a], perm[b] = perm[b], perm[a]
                    improved = True
    out = []
    for i, j in enumerate(perm):
        v = x[i] if float(x[i] @ seeds[j]) >= 0 else -x[i]
        out.append((v, labels[j]))
    return out


def main():
    restarts = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    rng = np.random.default_rng(11)
    seeds = np.array([unit(t) for t in REP_TRIPLES]
                     + [unit(v) for v, _ in EXT_SEEDS])
    labels = list(REP_TRIPLES) + [lbl for _, lbl in EXT_SEEDS]
    starts = [seeds.copy()] + [rng.normal(size=(N_LINES, 3))
                               for _ in range(restarts)]

    best, best_angle = None, -1.0
    for k, s0 in enumerate(starts):
        x = anneal(s0 / np.linalg.norm(s0, axis=1, keepdims=True))
        a = min_angle_deg(full(x))
        if a > best_angle:
            best, best_angle = x, a
            print(f"{'seeded' if k == 0 else f'restart {k}'}: "
                  f"{a:.2f} deg  <- new best")

    labeled = assign_labels(best, seeds, labels)
    print(f"\nbest min pairwise angle: {best_angle:.2f} deg "
          f"(sphere26: 38.17; icosa32: 37.38)")
    print("\n# paste into versor/decode.py:")
    print("_SPHERE32_TABLE = [")
    for v, lbl in labeled:
        lbl_s = repr(lbl) if isinstance(lbl, str) else str(lbl)
        print(f"    (({v[0]:+.15f}, {v[1]:+.15f}, {v[2]:+.15f}), {lbl_s}),")
    print("]  # 16 antipodal representatives; negation gives the partner")


if __name__ == "__main__":
    main()
