"""Generate the sphere26 direction table (whitepaper open problem 6).

Searches for a good antipodal packing of 26 directions (13 lines in RP^2)
by annealed, direction-normalized Riesz repulsion from many random starts
(plus the cubic-26 seed), then assigns each line an opcode triple pair by
greedy best-|dot| matching against the cubic cone centers with 2-opt
improvement — the labeling is semantics-neutral, the matching just keeps
sphere26 directions roughly recognizable next to their cubic cousins.

Usage: python tools/optimize_sphere26.py [restarts]
Prints the frozen table to paste into versor/decode.py plus quality stats.
"""
import itertools
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np  # noqa: E402

S = 12.0         # Riesz exponent (higher biases toward max-min packing)
STEPS = 15000
LR0 = 0.004      # max angular step; forces are direction-normalized

TRIPLES = [t for t in itertools.product((-1, 0, 1), repeat=3) if t != (0, 0, 0)]
REP_TRIPLES = []
_seen = set()
for _t in TRIPLES:
    if tuple(-c for c in _t) not in _seen:
        _seen.add(_t)
        REP_TRIPLES.append(_t)


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
        g = f[:13] - f[13:]
        g -= (np.einsum("ij,ij->i", g, x))[:, None] * x
        g /= np.maximum(np.linalg.norm(g, axis=1, keepdims=True), 1e-12)
        x = x + LR0 * (1.0 - step / STEPS) * g
        x /= np.linalg.norm(x, axis=1, keepdims=True)
    return x


def assign_labels(x):
    """Match the 13 lines to the 13 cubic triple pairs, maximizing |dot|."""
    seeds = np.array([np.array(t, float) / np.linalg.norm(t)
                      for t in REP_TRIPLES])
    score = np.abs(x @ seeds.T)  # lines x triple-pairs
    # greedy
    perm = [-1] * 13
    used = set()
    for i in np.argsort(-score.max(axis=1)):
        j = max((j for j in range(13) if j not in used),
                key=lambda j: score[i, j])
        perm[i] = j
        used.add(j)
    # 2-opt improvement
    improved = True
    while improved:
        improved = False
        for a in range(13):
            for b in range(a + 1, 13):
                cur = score[a, perm[a]] + score[b, perm[b]]
                swp = score[a, perm[b]] + score[b, perm[a]]
                if swp > cur + 1e-12:
                    perm[a], perm[b] = perm[b], perm[a]
                    improved = True
    out = []
    for i, j in enumerate(perm):
        seed = seeds[j]
        v = x[i] if float(x[i] @ seed) >= 0 else -x[i]
        out.append((v, REP_TRIPLES[j]))
    return out


def main():
    restarts = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    rng = np.random.default_rng(7)
    cubic = np.array([np.array(t, float) / np.linalg.norm(t)
                      for t in REP_TRIPLES])
    starts = [cubic] + [rng.normal(size=(13, 3)) for _ in range(restarts)]

    best, best_angle = None, -1.0
    for k, s0 in enumerate(starts):
        x = s0 / np.linalg.norm(s0, axis=1, keepdims=True)
        x = anneal(x)
        a = min_angle_deg(full(x))
        tag = "cubic-seed" if k == 0 else f"restart {k}"
        if a > best_angle:
            best, best_angle = x, a
            print(f"{tag}: {a:.2f} deg  <- new best")
        elif k == 0:
            print(f"{tag}: {a:.2f} deg")

    labeled = assign_labels(best)
    drift = [math.degrees(math.acos(np.clip(
        float(v @ (np.array(t, float) / np.linalg.norm(t))), -1, 1)))
        for v, t in labeled]
    print(f"\nbest min pairwise angle: {best_angle:.2f} deg "
          f"(cubic-26 baseline: 35.26)")
    print(f"label drift from cubic centers: max {max(drift):.1f} deg, "
          f"mean {sum(drift) / 13:.1f} deg")

    print("\n# paste into versor/decode.py:")
    print("_SPHERE26_TABLE = [")
    for v, t in labeled:
        print(f"    (({v[0]:+.15f}, {v[1]:+.15f}, {v[2]:+.15f}), {t}),")
    print("]  # 13 antipodal representatives; negation gives the other 13")


if __name__ == "__main__":
    main()
