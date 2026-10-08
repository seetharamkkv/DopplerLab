import numpy as np

from ctc_exp.catalog import load_straight_audio_only
from ctc_exp.track_geom import straight_passby_xy


def test_pythagoras_cpa_and_range():
    times = np.array([4.0, 5.0, 6.0])
    xy, vel = straight_passby_xy(times, speed_mps=20.0, cpa_s=5.0, cpa_m=0.5)
    assert np.allclose(xy[1], [0.0, 0.5])
    assert np.allclose(xy[2], [20.0, 0.5])
    rng = np.sqrt(xy[:, 0] ** 2 + xy[:, 1] ** 2)
    assert np.isclose(rng[1], 0.5)
    assert np.isclose(rng[2], np.hypot(20.0, 0.5))
    assert np.allclose(vel[:, 0], 20.0)
    assert np.allclose(vel[:, 1], 0.0)


def test_straight_loader_skips_sim_state(tmp_path):
    sid = "sample_0000001"
    sample = tmp_path / "audio_clips" / sid
    sample.mkdir(parents=True)
    (sample / "car.wav").write_bytes(b"not-a-real-wav")
    (sample / "metadata").mkdir()
    (sample / "metadata" / "state_frames.npy").write_bytes(b"LEAK")
    (sample / "metadata" / "range_m.npy").write_bytes(b"LEAK")
    (tmp_path / "dataset.csv").write_text(
        "sample_id,filename,vehicle_class,trajectory_type,speed_mps,"
        "cpa_distance_m,cpa_time_sec\n"
        f"{sid},car.wav,Toy,straight,20.0,0.5,5.0\n",
        encoding="utf-8",
    )
    clips = load_straight_audio_only(tmp_path)
    assert len(clips) == 1
    c = clips[0]
    assert c.xy_m is None and c.vel_mps is None and c.range_m is None
    assert c.times_s is None
    assert c.speed_mps == 20.0 and c.cpa_m == 0.5
