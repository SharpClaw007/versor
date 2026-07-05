"""Reproduces every numeric claim in docs/whitepaper.md.

Usage: python docs/calcs.py [N_SAMPLES]
"""
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np  # noqa: E402

from versor.decode import PHI, Cubic26, Icosa32, Sphere26, Sphere32  # noqa: E402
from versor.errors import VersorFault  # noqa: E402


def sphere_samples(n, seed=0):
    rng = np.random.default_rng(seed)
    v = rng.normal(size=(n, 3))
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def cubic26_partition(n):
    """Monte Carlo measure of the cubic-26 cells and dead zone."""
    dec = Cubic26()
    v = sphere_samples(n)
    counts = {"face": 0, "edge": 0, "corner": 0, "dead": 0}
    for x in v:
        try:
            triple = dec.decode(x)
        except VersorFault:
            counts["dead"] += 1
            continue
        counts[{1: "face", 2: "edge", 3: "corner"}[sum(map(abs, triple))]] += 1
    print("cubic-26 sphere partition (fraction of S^2):")
    for k, c in counts.items():
        print(f"  {k:>6}: {c / n:.4f} total"
              + (f"  ({c / n / {'face': 6, 'edge': 12, 'corner': 8}[k]:.4f} per cell)"
                 if k != "dead" else ""))


def nn_partition(dec, n):
    v = sphere_samples(n)
    ok = dead = reserved = 0
    for x in v:
        try:
            dec.decode(x)
            ok += 1
        except VersorFault as f:
            if f.kind == "ReservedDirection":
                reserved += 1
            else:
                dead += 1
    print(f"{dec.name} partition: assigned {ok / n:.4f}, "
          f"reserved {reserved / n:.4f}, dead zone {dead / n:.4f} "
          f"(margin {dec.margin})")


def icosa_angles():
    e_cos = (1 + PHI) / (math.sqrt(2) * math.sqrt(1 + PHI ** 2))
    f_cos = PHI / math.sqrt(3)
    print(f"phi^2 + phi^-2 = {PHI**2 + PHI**-2:.12f}  (exactly 3)")
    print(f"edge assignment angle:  cos = (1+phi)/(sqrt(2)sqrt(1+phi^2)) "
          f"= {e_cos:.6f} -> {math.degrees(math.acos(e_cos)):.2f} deg")
    print(f"face assignment angle:  cos = phi/sqrt(3) "
          f"= {f_cos:.6f} -> {math.degrees(math.acos(f_cos)):.2f} deg")
    # minimum angular gap within each nearest-neighbor direction set
    for dec in (Icosa32(), Sphere26(), Sphere32()):
        m = dec._matrix
        dots = np.clip(m @ m.T, -1, 1)
        np.fill_diagonal(dots, -1)
        print(f"closest pair of {dec.name} directions: "
              f"{math.degrees(math.acos(dots.max())):.2f} deg apart")


def interpolation_bands():
    """Closed-form fault bands for the countdown A->B lerp (whitepaper 7.2).

    The loop-back filler lerps NOP=(-1,-1,1)/sqrt(3) -> PUSHF=(-1,1,1)/sqrt(3);
    with s = 2t-1 the normalized y component is s/sqrt(2+s^2), and decoding
    faults when |y_hat| lands in the dead band (0.30, 0.40).
    """
    def s_for(y):
        return math.sqrt(y * y * 2 / (1 - y * y))
    bands = []
    for lo, hi in [(0.30, 0.40)]:
        s0, s1 = s_for(lo), s_for(hi)
        bands.append(((1 - s1) / 2, (1 - s0) / 2))
        bands.append(((1 + s0) / 2, (1 + s1) / 2))
    for t0, t1 in sorted(bands):
        print(f"predicted fault band: t in ({t0:.4f}, {t1:.4f})"
              f"  width {t1 - t0:.4f}")
    print(f"predicted viable fraction: {1 - sum(b - a for a, b in bands):.4f}")


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 2_000_000
    print(f"[{n:,} sphere samples]")
    cubic26_partition(n)
    nn_partition(Icosa32(), n // 4)
    nn_partition(Sphere26(), n // 4)
    nn_partition(Sphere32(), n // 4)
    icosa_angles()
    interpolation_bands()


if __name__ == "__main__":
    main()
