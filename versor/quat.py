"""Hand-rolled unit quaternions.

Convention: q = (w, x, y, z), Hamilton product.

A frame F maps frame-local coordinates to world coordinates:

    v_world = F * v_local * F^-1   ==  F.rotate(v_local)
    v_local = F^-1 * v_world * F   ==  F.conj().rotate(v_world)
"""
from __future__ import annotations

import math

import numpy as np


class Quat:
    __slots__ = ("w", "x", "y", "z")

    def __init__(self, w: float, x: float, y: float, z: float):
        self.w = float(w)
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)

    @staticmethod
    def identity() -> "Quat":
        return Quat(1.0, 0.0, 0.0, 0.0)

    @staticmethod
    def axis_angle(axis, angle: float) -> "Quat":
        """Rotation of `angle` radians about `axis` (need not be unit)."""
        a = np.asarray(axis, dtype=float)
        n = float(np.linalg.norm(a))
        if n == 0.0:
            raise ValueError("axis_angle: zero axis")
        a = a / n
        h = angle / 2.0
        s = math.sin(h)
        return Quat(math.cos(h), a[0] * s, a[1] * s, a[2] * s)

    def __mul__(self, o: "Quat") -> "Quat":
        w1, x1, y1, z1 = self.w, self.x, self.y, self.z
        w2, x2, y2, z2 = o.w, o.x, o.y, o.z
        return Quat(
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        )

    def conj(self) -> "Quat":
        return Quat(self.w, -self.x, -self.y, -self.z)

    def norm(self) -> float:
        return math.sqrt(self.w ** 2 + self.x ** 2 + self.y ** 2 + self.z ** 2)

    def normalized(self) -> "Quat":
        n = self.norm()
        if n == 0.0:
            raise ValueError("cannot normalize zero quaternion")
        return Quat(self.w / n, self.x / n, self.y / n, self.z / n)

    def rotate(self, v) -> np.ndarray:
        """Rotate vec3 `v` by this (unit) quaternion: q * v * q^-1."""
        v = np.asarray(v, dtype=float)
        u = np.array([self.x, self.y, self.z])
        t = 2.0 * np.cross(u, v)
        return v + self.w * t + np.cross(u, t)

    def as_tuple(self) -> tuple[float, float, float, float]:
        return (self.w, self.x, self.y, self.z)

    def approx(self, o: "Quat", tol: float = 1e-9) -> bool:
        """Equality as a rotation (q and -q are the same rotation)."""
        d1 = max(abs(self.w - o.w), abs(self.x - o.x), abs(self.y - o.y), abs(self.z - o.z))
        d2 = max(abs(self.w + o.w), abs(self.x + o.x), abs(self.y + o.y), abs(self.z + o.z))
        return min(d1, d2) < tol

    def __repr__(self) -> str:
        return f"Quat({self.w:.6g}, {self.x:.6g}, {self.y:.6g}, {self.z:.6g})"
