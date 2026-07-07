"""Continuous program-space tools: robustness maps and evolutionary search.

Versor's behavior is a piecewise-constant function on program space
R^{3m} (whitepaper §8). That makes two operations meaningful that no
discrete representation supports:

- **Robustness**: how far can one segment move before behavior changes?
  ``segment_tolerances`` measures, per segment, the largest perturbation
  radius that preserves the program's output in every sampled direction —
  an empirical probe of the distance to the nearest behavior wall.
- **Synthesis**: search program space directly. ``evolve`` runs a
  (1+lambda) evolution strategy over all segment vectors with a
  behavioral fitness; ``output_fitness`` scores a program by how close its
  OUT buffer is to a target sequence.
"""
from __future__ import annotations

import math

import numpy as np

from .errors import VersorFault
from .loader import Chain, Edge, Program
from .machine import Machine

FAULT_PENALTY = 50.0
NO_HALT_PENALTY = 50.0
MISSING_ITEM_PENALTY = 10.0
VALUE_ERROR_CAP = 5.0  # < MISSING_ITEM_PENALTY: more outputs always wins


def _clone(prog: Program) -> Program:
    chains = [Chain(id=ch.id,
                    vertices={vid: [Edge(seg=e.seg.copy(), to=e.to,
                                         guard=None if e.guard is None
                                         else e.guard.copy())
                                    for e in edges]
                              for vid, edges in ch.vertices.items()},
                    comment=ch.comment)
              for ch in prog.chains]
    return Program(chains=chains, name=prog.name, version=prog.version,
                   decoder=prog.decoder)


def segment_keys(prog: Program) -> list[tuple[int, int, int]]:
    """(chain, vertex, edge-index) for every segment, in stable order."""
    return [(ch.id, vid, i)
            for ch in prog.chains
            for vid, edges in sorted(ch.vertices.items())
            for i in range(len(edges))]


def get_vectors(prog: Program) -> np.ndarray:
    """All segment vectors flattened to shape (m, 3)."""
    return np.array([prog.chains[c].vertices[v][i].seg
                     for c, v, i in segment_keys(prog)])


def set_vectors(prog: Program, vecs: np.ndarray) -> Program:
    out = _clone(prog)
    for (c, v, i), vec in zip(segment_keys(prog), vecs):
        out.chains[c].vertices[v][i].seg = np.array(vec, dtype=float)
    return out


def run_out(prog: Program, step_budget: int = 20_000):
    """OUT buffer, or None on fault / budget exhaustion."""
    m = Machine(prog, step_budget=step_budget)
    try:
        with np.errstate(all="ignore"):  # evolved garbage overflows freely
            res = m.run()
    except VersorFault:
        return None
    return list(res.out)


def _out_equal(a, b, tol=1e-6) -> bool:
    if a is None or b is None or len(a) != len(b):
        return False
    for p, q in zip(a, b):
        if isinstance(p, str) or isinstance(q, str):
            if p != q:
                return False
        elif abs(p - q) > tol:
            return False
    return True


def segment_tolerances(prog: Program, *, directions: int = 12,
                       hi: float = 2.0, iters: int = 12,
                       step_budget: int = 20_000, seed: int = 0):
    """Per-segment perturbation tolerance.

    For each segment, bisect for the largest radius r such that adding
    r*u to that segment alone — for every u in a fixed sample of unit
    directions — leaves the program's output unchanged. Returns a list of
    (key, tolerance) in segment_keys order; a tolerance of `hi` means "no
    wall found within the search radius".
    """
    rng = np.random.default_rng(seed)
    baseline = run_out(prog, step_budget)
    if baseline is None:
        raise ValueError("baseline program faults; tolerances undefined")
    result = []
    for key in segment_keys(prog):
        u = rng.normal(size=(directions, 3))
        u /= np.linalg.norm(u, axis=1, keepdims=True)

        def survives(r: float) -> bool:
            for d in u:
                p = _clone(prog)
                c, v, i = key
                p.chains[c].vertices[v][i].seg = \
                    p.chains[c].vertices[v][i].seg + r * d
                if not _out_equal(run_out(p, step_budget), baseline):
                    return False
            return True

        lo, r_hi = 0.0, hi
        if survives(r_hi):
            result.append((key, r_hi))
            continue
        for _ in range(iters):
            mid = (lo + r_hi) / 2
            if survives(mid):
                lo = mid
            else:
                r_hi = mid
        result.append((key, lo))
    return result


def run_points(prog: Program, step_budget: int = 20_000):
    """(OUT buffer, executed-path points) — (None, None) on fault/budget."""
    from .trace import Trace
    trace = Trace()
    m = Machine(prog, trace=trace, step_budget=step_budget)
    try:
        with np.errstate(all="ignore"):
            m.run()
    except VersorFault:
        return None, None
    pts = [np.zeros(3)]
    for r in trace:
        if not np.allclose(r.P1, pts[-1]):
            pts.append(np.asarray(r.P1, dtype=float))
    return list(m.OUT), np.array(pts)


def resample_polyline(pts: np.ndarray, m: int = 64) -> np.ndarray:
    """m points spaced uniformly by arc length along the polyline."""
    seg = np.diff(pts, axis=0)
    lens = np.linalg.norm(seg, axis=1)
    cum = np.concatenate([[0.0], np.cumsum(lens)])
    if cum[-1] < 1e-12:
        return np.repeat(pts[:1], m, axis=0)
    out = np.empty((m, pts.shape[1]))
    for i, t in enumerate(np.linspace(0.0, cum[-1], m)):
        k = min(np.searchsorted(cum, t, side="right") - 1, len(seg) - 1)
        f = (t - cum[k]) / max(lens[k], 1e-12)
        out[i] = pts[k] + f * seg[k]
    return out


def _normalized_shape(pts: np.ndarray, m: int) -> np.ndarray:
    r = resample_polyline(pts, m)
    r = r - r.mean(axis=0)
    scale = np.sqrt((r ** 2).sum(axis=1).mean())
    return r / max(scale, 1e-12)


def shape_distance(pts: np.ndarray, target: np.ndarray, m: int = 64) -> float:
    """Mean pointwise distance between two polylines after arc-length
    resampling and translation/scale normalization. Orientation is NOT
    normalized: the target is drawn where it is aimed."""
    a = _normalized_shape(np.asarray(pts, dtype=float), m)
    b = _normalized_shape(np.asarray(target, dtype=float), m)
    return float(np.linalg.norm(a - b, axis=1).mean())


def shape_fitness(target_out: list, target_shape, *, weight: float = 100.0,
                  samples: int = 64):
    """Compute Y while drawing X: behavioral distance dominates (weighted),
    trace-shape distance breaks ties. Use with evaluator=run_points."""
    target_shape = np.asarray(target_shape, dtype=float)
    fit_out = output_fitness(target_out)

    def fit(res) -> float:
        out, pts = res
        b = fit_out(out)
        s = (shape_distance(pts, target_shape, samples)
             if pts is not None and len(pts) > 1 else 10.0)
        return weight * b + s

    return fit


def tolerance_mask(prog: Program, threshold: float = 0.02, **kw) -> np.ndarray:
    """Boolean mask over segments: True where perturbation tolerance is
    below `threshold` — the value-carrying, magnitude-frozen geometry."""
    tols = np.array([t for _, t in segment_tolerances(prog, **kw)])
    return tols < threshold


def magnitude_locked_mutation(seed_vectors: np.ndarray, locked: np.ndarray):
    """Mutation operator that renormalizes locked segments back to their
    seed magnitudes: direction stays free, the operand never changes. This
    is the robustness map acting as evolution's stiffness mask."""
    seed_norms = np.linalg.norm(seed_vectors, axis=1)

    def mutate(parent: np.ndarray, sigma: float, lam: int, rng) -> np.ndarray:
        kids = parent[None] + rng.normal(scale=sigma, size=(lam, *parent.shape))
        norms = np.linalg.norm(kids, axis=2)
        factor = np.where(locked[None, :],
                          seed_norms[None, :] / np.maximum(norms, 1e-12), 1.0)
        return kids * factor[:, :, None]

    return mutate


def output_fitness(target: list, tol: float = 1e-4):
    """Fitness (lower = better, 0 = match within tol): distance between OUT
    buffers, with penalties for faults, non-halting, and length mismatch."""

    def fit(out) -> float:
        if out is None:
            return FAULT_PENALTY + NO_HALT_PENALTY
        score = 0.0
        for p, q in zip(out, target):
            if isinstance(p, str) or isinstance(q, str):
                score += 0.0 if p == q else VALUE_ERROR_CAP
            else:
                d = abs(p - q)
                score += 0.0 if d <= tol else min(d, VALUE_ERROR_CAP)
        score += MISSING_ITEM_PENALTY * abs(len(out) - len(target))
        return score

    return fit


def evolve(topology: Program, fitness, *, sigma: float = 0.3,
           lam: int = 16, generations: int = 300, seed: int = 0,
           step_budget: int = 20_000, evaluator=None, mutate=None):
    """(1+lambda) evolution strategy over all segment vectors.

    `evaluator` maps a program to whatever `fitness` scores (default:
    run_out, so `fitness` sees the OUT buffer — use run_points +
    shape_fitness for trace-shape objectives). `mutate` overrides the
    default isotropic Gaussian (see magnitude_locked_mutation). Sigma
    adapts by a 1/5-success-style rule with a stall re-expansion. Returns
    (best_program, history); stops early at fitness 0.
    """
    rng = np.random.default_rng(seed)
    if evaluator is None:
        def evaluator(p):
            return run_out(p, step_budget)
    if mutate is None:
        def mutate(parent, s, k, r):
            return parent[None] + r.normal(scale=s, size=(k, *parent.shape))
    x = get_vectors(topology)
    best = fitness(evaluator(topology))
    history = [best]
    stall = 0
    for _ in range(generations):
        if best == 0.0:
            break
        children = mutate(x, sigma, lam, rng)
        scores = [fitness(evaluator(set_vectors(topology, c)))
                  for c in children]
        k = int(np.argmin(scores))
        if scores[k] < best:
            best, x = scores[k], children[k]
            sigma *= 1.15
            stall = 0
        else:
            sigma *= 0.9
            stall += 1
            if stall >= 50:  # value-polish shrank sigma; structural jumps
                sigma = 0.25  # need exploration again
                stall = 0
        sigma = float(np.clip(sigma, 1e-7, 2.0))
        history.append(best)
    return set_vectors(topology, x), history
