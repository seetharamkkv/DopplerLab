"""Draw-eval scoring helpers (no DopplerSim required)."""

from __future__ import annotations

import numpy as np

from traj_reconstruction.draw_eval import _score, checkpoint_banner, dopplersim_root


def test_score_identity_path():
    t = np.linspace(0, 4, 80)
    xy = np.column_stack([20.0 * (t - 2.0), np.full_like(t, 12.0)])
    m = _score(xy, xy, duration_s=4.0)
    assert m["orbit_rms_m"] < 1e-8
    assert m["cpa_rel_err"] < 1e-8
    assert m["n_frames"] == 80


def test_dopplersim_sibling_exists():
    root = dopplersim_root()
    assert (root / "doppler_sim" / "path2d" / "synthesis.py").is_file()


def test_checkpoint_banner_shape():
    info = checkpoint_banner()
    assert "checkpoint" in info
    assert "exists" in info
    assert "training" in info


def test_app_serves_page():
    from traj_reconstruction.draw_eval import create_app, list_inference_models, resolve_inference_model

    client = create_app().test_client()
    home = client.get("/")
    assert home.status_code == 200
    assert b'name="model"' in home.data
    assert b"Inference model" in home.data
    boot = client.get("/api/bootstrap")
    assert boot.status_code == 200
    payload = boot.get_json()
    assert payload["checkpoint"]["exists"] is True
    ids = [m["id"] for m in payload["models"]]
    assert ids == ["cnn", "seq", "mlp", "flexible"]
    assert payload["default_model"] in ids
    assert any(m["default"] for m in payload["models"])
    cnn = resolve_inference_model("cnn")
    assert cnn["id"] == "cnn" and cnn["available"]
    flex = resolve_inference_model("flexible")
    assert flex["kind"] == "physics"
    listed = list_inference_models()
    assert {m["id"] for m in listed} == {"cnn", "seq", "mlp", "flexible"}
    bad = client.post("/api/jobs", data={"path_json": '[{"x":0,"y":1},{"x":2,"y":3}]', "model": "nope"})
    assert bad.status_code == 400
    ck = client.get("/api/checkpoint?model=cnn")
    assert ck.status_code == 200
    assert ck.get_json()["id"] == "cnn"
