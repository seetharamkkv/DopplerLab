"""Phase 5 orbit product tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from traj_reconstruction.path_families import make_s_curve
from traj_reconstruction.product import (
    export_orbit_product,
    predict_orbit,
    render_orbit_png,
    write_orbit_viewer_html,
)
from traj_reconstruction.synthesize import synthesize_tone_on_path


def test_predict_orbit_from_audio_only():
    xy = make_s_curve(cpa_distance_m=14.0, half_length_m=55.0, amplitude_m=7.0)
    synth = synthesize_tone_on_path(xy, speed_mps=18.0, f0_hz=500.0, pad_s=0.2)
    product = predict_orbit(audio=synth["audio"], sr=synth["sr"], method="flexible")
    assert product.heading_absolute is False
    assert product.mirror_ambiguous is True
    assert product.xy.shape[1] == 2
    assert len(product.r) == len(product.theta_rel) == len(product.frame_times)
    assert 0.0 < product.confidence <= 1.0


def test_rotation_family_members_differ_but_same_shape():
    xy = make_s_curve(cpa_distance_m=12.0, half_length_m=50.0, amplitude_m=6.0)
    synth = synthesize_tone_on_path(xy, speed_mps=16.0, f0_hz=480.0)
    product = predict_orbit(audio=synth["audio"], sr=synth["sr"], method="flexible")
    a = product.rotated(0.0)
    b = product.rotated(np.deg2rad(90.0))
    assert not np.allclose(a, b)
    # Radii unchanged under rotation.
    assert np.allclose(
        np.sqrt(a[:, 0] ** 2 + a[:, 1] ** 2),
        np.sqrt(b[:, 0] ** 2 + b[:, 1] ** 2),
        atol=1e-9,
    )


def test_export_artifacts(tmp_path: Path):
    xy = make_s_curve(cpa_distance_m=12.0, half_length_m=45.0, amplitude_m=5.0)
    synth = synthesize_tone_on_path(xy, speed_mps=15.0, f0_hz=500.0)
    product = predict_orbit(audio=synth["audio"], sr=synth["sr"], method="flexible")
    paths = export_orbit_product(product, tmp_path / "prod")
    assert paths["json"].is_file()
    assert paths["png"].is_file() and paths["png"].stat().st_size > 1000
    html = paths["html"].read_text()
    assert "Rotation about observer" in html
    assert "Mirror family" in html
    assert "Absolute world heading is not determined" in html
    assert '"mirror_ambiguous": true' in html.lower() or "mirror_ambiguous" in html
