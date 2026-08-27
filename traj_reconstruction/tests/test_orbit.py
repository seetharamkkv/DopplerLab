"""Unit tests for observer-centered orbit alignment."""

from __future__ import annotations

import numpy as np
import pytest

from traj_reconstruction.orbit import canonical_xy, orbit_align, orbit_family, xy_from_state


def _straight_flyby(n: int = 200, v: float = 20.0, h: float = 15.0, t_cpa: float = 2.0) -> np.ndarray:
    t = np.linspace(0.0, 4.0, n)
    x = v * (t - t_cpa)
    y = np.full_like(t, h)
    return np.column_stack([x, y])


def test_orbit_align_identity():
    xy = _straight_flyby()
    res = orbit_align(xy, xy)
    assert res.rms < 1e-10
    assert res.reflected is False


def test_orbit_align_rotation_invariance():
    xy = _straight_flyby()
    angle = np.deg2rad(73.0)
    c, s = np.cos(angle), np.sin(angle)
    rot = np.column_stack([c * xy[:, 0] - s * xy[:, 1], s * xy[:, 0] + c * xy[:, 1]])
    res = orbit_align(rot, xy)
    assert res.rms < 1e-8


def test_orbit_align_mirror_invariance():
    xy = _straight_flyby()
    mir = xy.copy()
    mir[:, 0] *= -1.0
    res = orbit_align(mir, xy)
    assert res.rms < 1e-8
    assert res.reflected is True


def test_orbit_align_shape_mismatch():
    a = _straight_flyby(50)
    b = _straight_flyby(60)
    with pytest.raises(ValueError, match="shape mismatch"):
        orbit_align(a, b)


def test_canonical_xy_places_cpa_on_plus_y():
    xy = _straight_flyby()
    can, meta = canonical_xy(xy)
    cpa = meta["cpa_index"]
    assert abs(can[cpa, 0]) < 1e-8
    assert can[cpa, 1] > 0


def test_xy_from_state():
    n = 10
    state = np.zeros((n, 4), dtype=np.float64)
    state[:, 0] = np.arange(n)
    state[:, 2] = 5.0
    xy = xy_from_state(state)
    assert xy.shape == (n, 2)
    assert np.allclose(xy[:, 1], 5.0)


def test_orbit_family_count():
    xy = _straight_flyby(40)
    fam = orbit_family(xy, n_rotations=12, include_mirror=True)
    assert len(fam) == 24
    # Every member aligns to the original with ~0 RMS.
    for member in fam[::3]:
        assert orbit_align(member, xy).rms < 1e-8
