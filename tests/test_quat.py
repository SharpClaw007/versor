import math

import numpy as np
import pytest

from versor.quat import Quat

X, Y, Z = np.eye(3)


def test_identity_rotates_nothing():
    q = Quat.identity()
    v = np.array([1.2, -3.4, 5.6])
    assert np.allclose(q.rotate(v), v)


def test_axis_angle_quarter_turn_z():
    q = Quat.axis_angle(Z, math.pi / 2)
    assert np.allclose(q.rotate(X), Y, atol=1e-12)
    assert np.allclose(q.rotate(Y), -X, atol=1e-12)
    assert np.allclose(q.rotate(Z), Z, atol=1e-12)


def test_composition_matches_sequential_rotation():
    qa = Quat.axis_angle(X, 0.7)
    qb = Quat.axis_angle(Y, -1.1)
    v = np.array([0.3, 0.4, 0.5])
    assert np.allclose((qa * qb).rotate(v), qa.rotate(qb.rotate(v)))


def test_rotation_composition_non_commutative():
    qx = Quat.axis_angle(X, math.pi / 2)
    qy = Quat.axis_angle(Y, math.pi / 2)
    v = np.array([0.0, 0.0, 1.0])
    assert not np.allclose((qx * qy).rotate(v), (qy * qx).rotate(v))


def test_conjugate_is_inverse():
    q = Quat.axis_angle([1, 2, 3], 0.9)
    v = np.array([4.0, -5.0, 6.0])
    assert np.allclose(q.conj().rotate(q.rotate(v)), v)


def test_normalized_and_norm():
    q = Quat(2.0, 0.0, 0.0, 0.0)
    assert q.norm() == pytest.approx(2.0)
    assert q.normalized().as_tuple() == pytest.approx((1.0, 0.0, 0.0, 0.0))
    with pytest.raises(ValueError):
        Quat(0, 0, 0, 0).normalized()


def test_approx_sign_invariant():
    q = Quat.axis_angle(Z, 1.0)
    neg = Quat(-q.w, -q.x, -q.y, -q.z)
    assert q.approx(neg)


def test_axis_angle_zero_axis_rejected():
    with pytest.raises(ValueError):
        Quat.axis_angle([0, 0, 0], 1.0)
