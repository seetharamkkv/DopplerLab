"""Flexible freehand orbit recovery (Phase 3b).

Two complementary tools (audio / STFT only — no metadata):

1. ``fit_flexible_orbit`` — physics fit with a Fourier-lateral freeform path
   (generalizes straight flybys to S-curves / bends).
2. ``OrbitMLP`` — small STFT→canonical-path network trained with orbit MSE
   on simulated Tier-1 batches (NumPy SGD; no metadata inputs).
"""

from __future__ import annotations

import json
import pickle
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import least_squares

from traj_reconstruction.dataset import Phase1Batch, Phase1Sample, to_inference_bundle
from traj_reconstruction.frontend import RidgeFeatures, extract_ridges
from traj_reconstruction.kinematics import (
    SPEED_OF_SOUND,
    resample_path_constant_speed,
    solve_retarded_time_path,
)
from traj_reconstruction.orbit import OrbitAlignResult, canonical_xy, orbit_align, xy_from_state
from traj_reconstruction.parametric import (
    FitResult,
    fit_parametric_orbit,
    init_params_from_ridges,
)


def _freeform_polyline(
    *,
    h: float,
    half_length: float,
    coeffs: np.ndarray,
    n_pts: int = 120,
) -> np.ndarray:
    """Canonical-gauge path: x along-track, y = h + Fourier lateral terms."""
    s = np.linspace(0.0, 1.0, int(n_pts))
    x = (s - 0.5) * 2.0 * float(half_length)
    y = np.full_like(s, float(h))
    # coeffs: [a1,a2,...,aM, b1,b2,...] sine / cosine lateral modes
    m = len(coeffs) // 2
    a = coeffs[:m]
    b = coeffs[m : 2 * m]
    for k, (ak, bk) in enumerate(zip(a, b), start=1):
        y = y + float(ak) * np.sin(k * np.pi * s) + float(bk) * np.cos(k * np.pi * s)
    return np.column_stack([x, y])


def _forward_freeform(
    params: dict[str, float],
    coeffs: np.ndarray,
    frame_times: np.ndarray,
    *,
    sr: int,
    pad_s: float,
    c: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    duration = float(frame_times[-1] - frame_times[0]) if len(frame_times) > 1 else 4.0
    v = max(float(params["v"]), 1.0)
    half_length = float(np.clip(0.5 * v * (duration + 2.0 * pad_s), 15.0, 400.0))
    poly = _freeform_polyline(
        h=float(params["h"]),
        half_length=half_length,
        coeffs=coeffs,
    )
    traj = resample_path_constant_speed(poly, speed_mps=v, sr=sr, pad_s=pad_s)
    t_path = traj["t"]
    range_inst = np.sqrt(traj["x"] ** 2 + traj["y"] ** 2)
    t_cpa_geom = float(t_path[int(np.argmin(range_inst))])
    t_src = t_path - t_cpa_geom + float(params["t_cpa"])
    t_obs = np.linspace(
        float(min(t_src[0], frame_times[0])),
        float(max(t_src[-1], frame_times[-1])),
        max(len(t_src), 512),
    )
    t_r, r = solve_retarded_time_path(t_obs, t_src, traj["x"], traj["y"], c=c)
    valid = np.isfinite(t_r) & np.isfinite(r) & (r > 1e-9)
    idx = np.clip(
        np.searchsorted(t_src, np.nan_to_num(t_r, nan=t_src[0]), side="left"),
        0,
        len(t_src) - 1,
    )
    r_xy = np.maximum(np.sqrt(traj["x"][idx] ** 2 + traj["y"][idx] ** 2), 1e-9)
    v_r = (traj["x"][idx] * traj["vx"][idx] + traj["y"][idx] * traj["vy"][idx]) / r_xy
    f_obs = params["f0"] * float(c) / (float(c) + v_r)
    f_obs = np.where(valid, f_obs, np.nan)
    A = np.where(valid, 1.0 / np.maximum(r, 1e-9), 0.0)
    f_dense = np.where(np.isfinite(f_obs), f_obs, float(params["f0"]))
    f_hat = np.interp(frame_times, t_obs, f_dense)
    A_raw = np.interp(frame_times, t_obs, A)
    A_hat = A_raw / max(float(np.max(A_raw)), 1e-12)
    x_i = np.interp(frame_times, t_src, traj["x"], left=traj["x"][0], right=traj["x"][-1])
    y_i = np.interp(frame_times, t_src, traj["y"], left=traj["y"][0], right=traj["y"][-1])
    return f_hat, A_hat, np.column_stack([x_i, y_i])


def fit_flexible_orbit(
    feats: RidgeFeatures,
    *,
    n_modes: int = 3,
    use_amplitude: bool = True,
    w_f: float = 1.0,
    w_A: float = 0.5,
    w_smooth: float = 0.05,
    pad_s: float = 0.25,
    c: float = SPEED_OF_SOUND,
    gt_xy: np.ndarray | None = None,
    gt_speed_mps: float | None = None,
    gt_cpa_distance_m: float | None = None,
) -> FitResult:
    """Fit a Fourier-lateral freeform path to ridge features (audio-only)."""
    base = init_params_from_ridges(feats, family="straight", c=c)
    # Warm-start from straight parametric when helpful.
    try:
        warm = fit_parametric_orbit(
            feats, family="straight", use_amplitude=use_amplitude, pad_s=pad_s, c=c
        )
        if warm.residual_rms_f < 40.0:
            base.update({k: warm.params[k] for k in ("v", "h", "t_cpa", "f0") if k in warm.params})
    except Exception:
        pass

    t = feats.frame_times
    f_obs = feats.f_obs_hz
    A_obs = feats.A_env / max(float(np.max(feats.A_env)), 1e-12)
    f_med = float(np.median(f_obs[feats.quality > 0.5])) if np.any(feats.quality > 0.5) else float(
        np.median(f_obs)
    )
    w = (
        np.clip(feats.quality, 0.0, 1.0)
        * np.clip(A_obs, 0.0, 1.0)
        * (np.abs(f_obs - f_med) < 0.35 * max(f_med, 1.0)).astype(np.float64)
    )
    w = np.maximum(w, 1e-3)
    f_scale = max(float(np.std(f_obs[w > 0.05])), 10.0) if np.any(w > 0.05) else 10.0

    n_coeff = 2 * int(n_modes)
    x0 = np.array(
        [base["v"], base["h"], base["t_cpa"], base["f0"], *np.zeros(n_coeff)],
        dtype=np.float64,
    )
    lo = np.array([3.0, 2.0, float(t[0]), 50.0, *([-80.0] * n_coeff)], dtype=np.float64)
    hi = np.array([60.0, 200.0, float(t[-1]), 4000.0, *([80.0] * n_coeff)], dtype=np.float64)
    x0 = np.minimum(np.maximum(x0, lo + 1e-6), hi - 1e-6)

    def unpack(x: np.ndarray) -> tuple[dict[str, float], np.ndarray]:
        params = {
            "v": float(x[0]),
            "h": float(x[1]),
            "t_cpa": float(x[2]),
            "f0": float(x[3]),
            "half_length": float(base.get("half_length", 50.0)),
        }
        return params, x[4:].astype(np.float64)

    def residual(x: np.ndarray) -> np.ndarray:
        params, coeffs = unpack(x)
        f_hat, A_hat, xy = _forward_freeform(
            params, coeffs, t, sr=feats.sr, pad_s=pad_s, c=c
        )
        rf = w * (f_hat - f_obs) / f_scale
        parts = [np.sqrt(w_f) * rf]
        if use_amplitude:
            parts.append(np.sqrt(w_A) * w * (A_hat - A_obs))
        # Smoothness on lateral coeffs + path curvature proxy.
        parts.append(np.sqrt(w_smooth) * coeffs)
        d2 = np.diff(xy, n=2, axis=0)
        parts.append(np.sqrt(w_smooth) * 0.01 * d2.ravel())
        return np.concatenate(parts)

    result = least_squares(
        residual, x0, bounds=(lo, hi), method="trf", max_nfev=160, ftol=1e-9, xtol=1e-9
    )
    params, coeffs = unpack(result.x)
    f_hat, A_hat, xy = _forward_freeform(
        params, coeffs, t, sr=feats.sr, pad_s=pad_s, c=c
    )
    rf = float(np.sqrt(np.mean((w * (f_hat - f_obs)) ** 2)))
    ra = float(np.sqrt(np.mean((w * (A_hat - A_obs)) ** 2)))
    params_out = {
        **params,
        **{f"c{i}": float(v) for i, v in enumerate(coeffs)},
    }

    orbit = None
    metrics: dict[str, float] | None = None
    if gt_xy is not None:
        n = min(len(xy), len(gt_xy))
        orbit = orbit_align(xy[:n], gt_xy[:n])
        metrics = {
            "orbit_rms_m": orbit.rms,
            "pred_speed_mps": params["v"],
            "pred_cpa_distance_m": params["h"],
        }
        if gt_speed_mps is not None:
            metrics["speed_rel_err"] = abs(params["v"] - float(gt_speed_mps)) / max(
                float(gt_speed_mps), 1e-6
            )
        if gt_cpa_distance_m is not None:
            metrics["cpa_rel_err"] = abs(params["h"] - float(gt_cpa_distance_m)) / max(
                float(gt_cpa_distance_m), 1e-6
            )

    return FitResult(
        family="flexible_fourier",
        params=params_out,
        xy_pred=xy,
        frame_times=t.copy(),
        f_hat_hz=f_hat,
        A_hat=A_hat,
        residual_rms_f=rf,
        residual_rms_A=ra,
        success=bool(result.success),
        message=str(result.message),
        orbit=orbit,
        metrics=metrics,
    )


def fit_flexible_from_audio(
    *,
    wav_path: Path | str | None = None,
    stft_db: np.ndarray | None = None,
    audio: np.ndarray | None = None,
    sr: int | None = None,
    gt_xy: np.ndarray | None = None,
    **kwargs: Any,
) -> FitResult:
    kw: dict[str, Any] = {"n_harmonics": 1}
    if stft_db is not None:
        kw["stft_db"] = stft_db
        if sr is not None:
            kw["sr"] = sr
    elif audio is not None:
        kw["audio"] = audio
        if sr is not None:
            kw["sr"] = sr
    elif wav_path is not None:
        kw["wav_path"] = wav_path
    else:
        raise ValueError("provide wav_path, audio, or stft_db")
    feats = extract_ridges(**kw)
    return fit_flexible_orbit(feats, gt_xy=gt_xy, **kwargs)


# ---------------------------------------------------------------------------
# Lightweight STFT → path MLP (NumPy)
# ---------------------------------------------------------------------------


def _resize_stft(stft_db: np.ndarray, out_f: int = 32, out_t: int = 48) -> np.ndarray:
    """Nearest-neighbor resize to fixed feature grid."""
    stft_db = np.asarray(stft_db, dtype=np.float64)
    f, t = stft_db.shape
    fi = (np.linspace(0, f - 1, out_f)).astype(int)
    ti = (np.linspace(0, t - 1, out_t)).astype(int)
    return stft_db[fi][:, ti]


def stft_to_features(stft_db: np.ndarray, out_f: int = 32, out_t: int = 48) -> np.ndarray:
    grid = _resize_stft(stft_db, out_f=out_f, out_t=out_t)
    # Normalize per-clip.
    mu, sig = float(np.mean(grid)), float(np.std(grid) + 1e-6)
    return ((grid - mu) / sig).ravel().astype(np.float64)


def _path_to_knots(xy: np.ndarray, n_knots: int = 32) -> np.ndarray:
    idx = (np.linspace(0, len(xy) - 1, n_knots)).astype(int)
    return xy[idx].astype(np.float64)


def _knots_to_path(knots: np.ndarray, n_frames: int) -> np.ndarray:
    knots = np.asarray(knots, dtype=np.float64).reshape(-1, 2)
    s = np.linspace(0.0, 1.0, len(knots))
    sq = np.linspace(0.0, 1.0, int(n_frames))
    x = np.interp(sq, s, knots[:, 0])
    y = np.interp(sq, s, knots[:, 1])
    return np.column_stack([x, y])


@dataclass
class OrbitMLP:
    """Tiny MLP: STFT features → canonical path knots."""

    w1: np.ndarray
    b1: np.ndarray
    w2: np.ndarray
    b2: np.ndarray
    w3: np.ndarray
    b3: np.ndarray
    n_knots: int = 32
    feat_f: int = 32
    feat_t: int = 48

    @classmethod
    def create(cls, rng: np.random.Generator | None = None, n_knots: int = 32) -> OrbitMLP:
        rng = rng or np.random.default_rng(0)
        in_dim = 32 * 48
        h1, h2 = 256, 128
        out = n_knots * 2

        def xavier(fan_in: int, fan_out: int) -> np.ndarray:
            lim = np.sqrt(6.0 / (fan_in + fan_out))
            return rng.uniform(-lim, lim, size=(fan_in, fan_out))

        return cls(
            w1=xavier(in_dim, h1),
            b1=np.zeros(h1),
            w2=xavier(h1, h2),
            b2=np.zeros(h2),
            w3=xavier(h2, out),
            b3=np.zeros(out),
            n_knots=n_knots,
        )

    def forward_knots(self, feat: np.ndarray) -> np.ndarray:
        x = np.asarray(feat, dtype=np.float64).ravel()
        h = np.tanh(x @ self.w1 + self.b1)
        h = np.tanh(h @ self.w2 + self.b2)
        return (h @ self.w3 + self.b3).reshape(self.n_knots, 2)

    def predict_xy(self, stft_db: np.ndarray, n_frames: int) -> np.ndarray:
        feat = stft_to_features(stft_db, out_f=self.feat_f, out_t=self.feat_t)
        knots = self.forward_knots(feat)
        return _knots_to_path(knots, n_frames)

    def save(self, path: Path | str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f".{path.name}.writing.npz")
        np.savez_compressed(
            tmp,
            w1=self.w1,
            b1=self.b1,
            w2=self.w2,
            b2=self.b2,
            w3=self.w3,
            b3=self.b3,
            n_knots=np.array([self.n_knots]),
            feat_f=np.array([self.feat_f]),
            feat_t=np.array([self.feat_t]),
        )
        tmp.replace(path)

    @classmethod
    def load(cls, path: Path | str) -> OrbitMLP:
        path = Path(path)
        last_err: Exception | None = None
        for attempt in range(8):
            try:
                data = np.load(path)
                return cls(
                    w1=data["w1"],
                    b1=data["b1"],
                    w2=data["w2"],
                    b2=data["b2"],
                    w3=data["w3"],
                    b3=data["b3"],
                    n_knots=int(data["n_knots"][0]),
                    feat_f=int(data["feat_f"][0]),
                    feat_t=int(data["feat_t"][0]),
                )
            except Exception as exc:  # noqa: BLE001 — retry while trainer replaces the file
                last_err = exc
                import time

                time.sleep(0.04 * (attempt + 1))
        assert last_err is not None
        raise last_err


def orbit_mse_loss(pred_xy: np.ndarray, gt_xy: np.ndarray) -> float:
    """Orbit-aligned MSE (eval / reporting)."""
    n = min(len(pred_xy), len(gt_xy))
    aligned = orbit_align(pred_xy[:n], gt_xy[:n]).aligned_pred
    return float(np.mean(np.sum((aligned - gt_xy[:n]) ** 2, axis=1)))


def canonical_mse_loss(pred_xy: np.ndarray, gt_xy: np.ndarray) -> float:
    """MSE after mapping both to canonical gauge (train target)."""
    p, _ = canonical_xy(pred_xy)
    g, _ = canonical_xy(gt_xy)
    n = min(len(p), len(g))
    return float(np.mean(np.sum((p[:n] - g[:n]) ** 2, axis=1)))


def _sample_target(sample: Phase1Sample) -> np.ndarray:
    src = sample.canonical_state_frames
    if src is None:
        src = sample.state_frames
        xy = xy_from_state(src)
        xy, _ = canonical_xy(xy)
        return xy
    return xy_from_state(src)


def _last_checkpoint_path(best_path: Path) -> Path:
    return best_path.with_name(best_path.stem + ".last.npz")


def _status_path(best_path: Path) -> Path:
    return best_path.with_name(best_path.stem + ".status.json")


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text)
    tmp.replace(path)


def _weights_tuple(model: OrbitMLP) -> tuple[np.ndarray, ...]:
    return (
        model.w1.copy(),
        model.b1.copy(),
        model.w2.copy(),
        model.b2.copy(),
        model.w3.copy(),
        model.b3.copy(),
    )


def _apply_weights(model: OrbitMLP, state: tuple[np.ndarray, ...]) -> None:
    model.w1, model.b1, model.w2, model.b2, model.w3, model.b3 = (
        state[0].copy(),
        state[1].copy(),
        state[2].copy(),
        state[3].copy(),
        state[4].copy(),
        state[5].copy(),
    )


def _save_best_atomic(model: OrbitMLP, best_state: tuple[np.ndarray, ...], path: Path) -> None:
    current = _weights_tuple(model)
    _apply_weights(model, best_state)
    model.save(path)
    _apply_weights(model, current)


def _save_last_atomic(
    path: Path,
    *,
    model: OrbitMLP,
    best_state: tuple[np.ndarray, ...] | None,
    epoch_next: int,
    best_val: float,
    best_epoch: int | None,
    history: list[dict[str, float]],
    train_idx: list[int],
    val_idx: list[int],
    seed: int,
    lr: float,
    rng: np.random.Generator,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.writing.npz")
    best = best_state or _weights_tuple(model)
    np.savez_compressed(
        tmp,
        w1=model.w1,
        b1=model.b1,
        w2=model.w2,
        b2=model.b2,
        w3=model.w3,
        b3=model.b3,
        best_w1=best[0],
        best_b1=best[1],
        best_w2=best[2],
        best_b2=best[3],
        best_w3=best[4],
        best_b3=best[5],
        n_knots=np.array([model.n_knots]),
        feat_f=np.array([model.feat_f]),
        feat_t=np.array([model.feat_t]),
        epoch_next=np.array([int(epoch_next)]),
        best_val=np.array([float(best_val)]),
        best_epoch=np.array([-1 if best_epoch is None else int(best_epoch)]),
        train_idx=np.asarray(train_idx, dtype=np.int64),
        val_idx=np.asarray(val_idx, dtype=np.int64),
        seed=np.array([int(seed)]),
        lr=np.array([float(lr)]),
        history_json=np.array(json.dumps(history)),
        rng_state=np.frombuffer(pickle.dumps(rng.bit_generator.state), dtype=np.uint8),
    )
    tmp.replace(path)


def _write_status(
    best_path: Path,
    *,
    epoch_completed: int,
    epochs: int,
    best_val: float,
    best_epoch: int | None,
    running: bool,
    n_train: int,
    n_val: int,
) -> None:
    payload = {
        "running": running,
        "epoch_completed": int(epoch_completed),
        "epochs_target": int(epochs),
        "best_val_orbit_rms": float(best_val) if np.isfinite(best_val) else None,
        "best_epoch": best_epoch,
        "best_checkpoint": str(best_path.resolve()),
        "last_checkpoint": str(_last_checkpoint_path(best_path).resolve()),
        "n_train": n_train,
        "n_val": n_val,
        "updated_at": time.time(),
        "note": "Frontend / infer should load best_checkpoint; it is replaced atomically.",
    }
    _atomic_write_text(_status_path(best_path), json.dumps(payload, indent=2))


def train_orbit_mlp(
    batch_dir: Path | str,
    *,
    epochs: int = 40,
    lr: float = 3e-3,
    seed: int = 0,
    holdout_families: tuple[str, ...] = ("u_turn",),
    checkpoint_path: Path | str | None = None,
    resume: bool = True,
) -> tuple[OrbitMLP, dict[str, Any]]:
    """Train OrbitMLP on a simulated Phase 1 batch (STFT → canonical path).

    ``checkpoint_path`` is the **best** weights file (frontend-safe, atomic).
    Resume state is ``<stem>.last.npz``. Inference can load the best file while
    training continues.
    """
    batch = Phase1Batch.from_dir(batch_dir)
    rng = np.random.default_rng(seed)
    model = OrbitMLP.create(rng)
    best_path = Path(checkpoint_path) if checkpoint_path is not None else None
    last_path = _last_checkpoint_path(best_path) if best_path is not None else None

    def _family(row: dict[str, str]) -> str:
        return str(row.get("path_family") or row.get("trajectory_type") or "")

    train_idx = [i for i, r in enumerate(batch.rows) if _family(r) not in holdout_families]
    val_idx = [i for i, r in enumerate(batch.rows) if _family(r) in holdout_families]
    if not val_idx:
        order = list(train_idx)
        rng.shuffle(order)
        n_val = max(1, len(order) // 5)
        val_idx = order[:n_val]
        train_idx = order[n_val:]

    history: list[dict[str, float]] = []
    best_val = float("inf")
    best_state: tuple[np.ndarray, ...] | None = None
    best_epoch: int | None = None
    start_epoch = 0

    if resume and last_path is not None and last_path.is_file():
        data = np.load(last_path, allow_pickle=True)
        model = OrbitMLP(
            w1=data["w1"],
            b1=data["b1"],
            w2=data["w2"],
            b2=data["b2"],
            w3=data["w3"],
            b3=data["b3"],
            n_knots=int(data["n_knots"][0]),
            feat_f=int(data["feat_f"][0]),
            feat_t=int(data["feat_t"][0]),
        )
        best_state = (
            data["best_w1"],
            data["best_b1"],
            data["best_w2"],
            data["best_b2"],
            data["best_w3"],
            data["best_b3"],
        )
        start_epoch = int(data["epoch_next"][0])
        best_val = float(data["best_val"][0])
        be = int(data["best_epoch"][0])
        best_epoch = None if be < 0 else be
        train_idx = [int(i) for i in data["train_idx"].tolist()]
        val_idx = [int(i) for i in data["val_idx"].tolist()]
        history = json.loads(data["history_json"].item())
        rng.bit_generator.state = pickle.loads(data["rng_state"].tobytes())
        print(
            f"Resumed last checkpoint at epoch {start_epoch}/{epochs} "
            f"(best_val={best_val:.4f} m)",
            flush=True,
        )
    elif resume and best_path is not None and best_path.is_file():
        loaded = OrbitMLP.load(best_path)
        model.w1, model.b1 = loaded.w1, loaded.b1
        model.w2, model.b2 = loaded.w2, loaded.b2
        model.w3, model.b3 = loaded.w3, loaded.b3
        model.n_knots, model.feat_f, model.feat_t = loaded.n_knots, loaded.feat_f, loaded.feat_t
        best_state = _weights_tuple(model)
        sidecar = best_path.with_suffix(".json")
        if sidecar.is_file():
            prev = json.loads(sidecar.read_text())
            history = list(prev.get("history") or [])
            start_epoch = len(history)
            best_val = float(prev.get("best_val_orbit_rms", float("inf")))
            if history:
                best_epoch = int(min(history, key=lambda h: h["val_orbit_rms"])["epoch"])
        print(
            f"Warm-started best checkpoint; continuing from epoch {start_epoch}/{epochs} "
            f"(best_val={best_val:.4f} m)",
            flush=True,
        )

    packed: list[tuple[np.ndarray, np.ndarray, np.ndarray, int]] = []
    for i in range(len(batch)):
        sample = batch.load(i)
        bundle = to_inference_bundle(sample)
        assert bundle.stft_db is not None
        feat = stft_to_features(bundle.stft_db)
        gt = _sample_target(sample)
        knots_gt = _path_to_knots(gt, model.n_knots)
        packed.append((feat, knots_gt, gt, sample.n_frames))

    def _report(running: bool, epoch_completed: int) -> dict[str, Any]:
        return {
            "batch_dir": str(Path(batch_dir).resolve()),
            "epochs": epochs,
            "epoch_completed": epoch_completed,
            "n_train": len(train_idx),
            "n_val": len(val_idx),
            "holdout_families": list(holdout_families),
            "best_val_orbit_rms": best_val,
            "best_epoch": best_epoch,
            "history": history,
            "data_scope": "simulated_only",
            "inputs": "spectrograms/stft.npy only",
            "source": "dopplersim_path2d_whiteboard_batch",
            "running": running,
            "best_checkpoint": str(best_path.resolve()) if best_path is not None else None,
        }

    if best_path is not None:
        _write_status(
            best_path,
            epoch_completed=start_epoch,
            epochs=epochs,
            best_val=best_val,
            best_epoch=best_epoch,
            running=True,
            n_train=len(train_idx),
            n_val=len(val_idx),
        )
        if best_state is not None and not best_path.is_file():
            _save_best_atomic(model, best_state, best_path)

    if start_epoch >= int(epochs):
        print(f"Already at {start_epoch}/{epochs} epochs; nothing to train.", flush=True)

    for epoch in range(start_epoch, int(epochs)):
        rng.shuffle(train_idx)
        train_losses = []
        for i in train_idx:
            feat, knots_gt, _gt, _n_frames = packed[i]
            x = feat
            z1 = x @ model.w1 + model.b1
            h1 = np.tanh(z1)
            z2 = h1 @ model.w2 + model.b2
            h2 = np.tanh(z2)
            out = h2 @ model.w3 + model.b3
            knots = out.reshape(model.n_knots, 2)

            err = knots - knots_gt
            d2 = np.diff(knots, n=2, axis=0)
            loss = float(np.mean(err**2) + 0.01 * np.mean(d2**2))
            train_losses.append(loss)

            dout = (2.0 / err.size) * err.ravel()
            g_w3 = np.outer(h2, dout)
            g_b3 = dout
            dh2 = (model.w3 @ dout) * (1.0 - h2**2)
            g_w2 = np.outer(h1, dh2)
            g_b2 = dh2
            dh1 = (model.w2 @ dh2) * (1.0 - h1**2)
            g_w1 = np.outer(x, dh1)
            g_b1 = dh1

            model.w3 -= lr * g_w3
            model.b3 -= lr * g_b3
            model.w2 -= lr * g_w2
            model.b2 -= lr * g_b2
            model.w1 -= lr * g_w1
            model.b1 -= lr * g_b1

        val_orbit = []
        for i in val_idx:
            feat, _knots_gt, gt, n_frames = packed[i]
            pred = _knots_to_path(model.forward_knots(feat), n_frames)
            val_orbit.append(orbit_align(pred, gt).rms)
        mean_train = float(np.mean(train_losses)) if train_losses else 0.0
        mean_val = float(np.mean(val_orbit)) if val_orbit else float("inf")
        history.append({"epoch": float(epoch), "train_mse": mean_train, "val_orbit_rms": mean_val})
        print(
            f"epoch {epoch + 1}/{epochs}  train_mse={mean_train:.4f}  "
            f"val_orbit_rms={mean_val:.4f} m",
            flush=True,
        )
        if mean_val < best_val:
            best_val = mean_val
            best_epoch = epoch
            best_state = _weights_tuple(model)
            if best_path is not None:
                _save_best_atomic(model, best_state, best_path)
                print(f"  wrote best checkpoint (frontend-safe): {best_path}", flush=True)

        if best_path is not None and last_path is not None:
            _save_last_atomic(
                last_path,
                model=model,
                best_state=best_state,
                epoch_next=epoch + 1,
                best_val=best_val,
                best_epoch=best_epoch,
                history=history,
                train_idx=train_idx,
                val_idx=val_idx,
                seed=seed,
                lr=lr,
                rng=rng,
            )
            report_live = _report(running=True, epoch_completed=epoch + 1)
            _atomic_write_text(best_path.with_suffix(".json"), json.dumps(report_live, indent=2))
            _write_status(
                best_path,
                epoch_completed=epoch + 1,
                epochs=epochs,
                best_val=best_val,
                best_epoch=best_epoch,
                running=True,
                n_train=len(train_idx),
                n_val=len(val_idx),
            )

    if best_state is not None:
        _apply_weights(model, best_state)

    report = _report(running=False, epoch_completed=max(start_epoch, int(epochs)))
    if best_path is not None:
        model.save(best_path)
        _atomic_write_text(best_path.with_suffix(".json"), json.dumps(report, indent=2))
        _write_status(
            best_path,
            epoch_completed=int(epochs),
            epochs=epochs,
            best_val=best_val,
            best_epoch=best_epoch,
            running=False,
            n_train=len(train_idx),
            n_val=len(val_idx),
        )
    return model, report


def infer_orbit_mlp(
    model: OrbitMLP,
    *,
    stft_db: np.ndarray | None = None,
    wav_path: Path | str | None = None,
) -> np.ndarray:
    """Audio-only inference → path about the observer (canonical-ish knots)."""
    if stft_db is None:
        if wav_path is None:
            raise ValueError("provide stft_db or wav_path")
        feats = extract_ridges(wav_path=wav_path, n_harmonics=1)
        stft_db = feats.stft_db
        n_frames = feats.frame_times.shape[0]
    else:
        n_frames = int(stft_db.shape[1])
    return model.predict_xy(stft_db, n_frames)
