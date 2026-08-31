"""Ridge-track 1D CNN: f_obs(t) + A_env(t) → canonical path (audio only).

Time is preserved (no pooling). The network never sees v, h, t_CPA, family, or
any Phase 1 metadata. Features come from ``extract_ridges`` only.
"""

from __future__ import annotations

import json
import pickle
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from traj_reconstruction.dataset import Phase1Batch, to_inference_bundle
from traj_reconstruction.flexible import _sample_target
from traj_reconstruction.frontend import RidgeFeatures, extract_ridges
from traj_reconstruction.orbit import orbit_align
from traj_reconstruction.orbit_cnn import _amp_grad, _he, _relu, _smooth_grad

ARCH_TAG = "ridge_seq_1d_v1"
N_CH = 6
SEQ_T = 128
N_HARMONICS = 1


def _resample_1d(x: np.ndarray, n: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    if x.size == int(n):
        return x
    if x.size < 2:
        return np.full(int(n), float(x[0]) if x.size else 0.0, dtype=np.float64)
    t = np.linspace(0.0, 1.0, x.size)
    tq = np.linspace(0.0, 1.0, int(n))
    return np.interp(tq, t, x)


def _resample_path(xy: np.ndarray, n: int) -> np.ndarray:
    xy = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
    return np.column_stack([_resample_1d(xy[:, 0], n), _resample_1d(xy[:, 1], n)])


def ridge_sequence_features(
    feats: RidgeFeatures,
    *,
    out_t: int = SEQ_T,
) -> np.ndarray:
    """(C, T) audio-only sequence. No metadata.

    Channels: fractional Doppler, df/dt, peak-norm amplitude, dA/dt, quality, time.
    """
    f = np.asarray(feats.f_obs_hz, dtype=np.float64)
    a = np.asarray(feats.A_env, dtype=np.float64)
    q = np.asarray(feats.quality, dtype=np.float64)
    if f.size < 2:
        raise ValueError("ridge track too short")
    good = q > 0.5
    f_med = float(np.median(f[good])) if np.any(good) else float(np.median(f))
    f_rel = (f - f_med) / max(abs(f_med), 1.0)
    a_n = a / max(float(np.max(a)), 1e-12)
    df = np.clip(np.gradient(f_rel), -0.25, 0.25)
    da = np.clip(np.gradient(a_n), -0.5, 0.5)
    t = np.linspace(0.0, 1.0, f.size)
    return np.stack(
        [
            _resample_1d(f_rel, out_t),
            _resample_1d(df, out_t),
            _resample_1d(a_n, out_t),
            _resample_1d(da, out_t),
            _resample_1d(q, out_t),
            _resample_1d(t, out_t),
        ],
        axis=0,
    ).astype(np.float32)


def ridges_from_audio(
    *,
    stft_db: np.ndarray | None = None,
    wav_path: Path | str | None = None,
    audio: np.ndarray | None = None,
    sr: int | None = None,
) -> RidgeFeatures:
    kwargs: dict[str, Any] = {"n_harmonics": N_HARMONICS}
    if stft_db is not None:
        return extract_ridges(stft_db=stft_db, **kwargs)
    if wav_path is not None:
        return extract_ridges(wav_path=wav_path, **kwargs)
    if audio is not None:
        if sr is None:
            raise ValueError("sr required with audio")
        return extract_ridges(audio=audio, sr=int(sr), **kwargs)
    raise ValueError("provide wav_path, audio, or stft_db")


def _conv1d(x: np.ndarray, w: np.ndarray, b: np.ndarray) -> np.ndarray:
    cin, t = x.shape
    cout, cin_w, k = w.shape
    if cin != cin_w:
        raise ValueError("conv1d channel mismatch")
    pad = k // 2
    xp = np.pad(x, ((0, 0), (pad, pad)))
    windows = sliding_window_view(xp, k, axis=1)
    return np.einsum("itk,oik->ot", windows, w, optimize=True) + b[:, None]


def _conv1d_backward(
    x: np.ndarray, w: np.ndarray, dout: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cout, cin, k = w.shape
    pad = k // 2
    xp = np.pad(x, ((0, 0), (pad, pad)))
    windows = sliding_window_view(xp, k, axis=1)
    dw = np.einsum("itk,ot->oik", windows, dout, optimize=True)
    db = dout.sum(axis=1)
    dout_p = np.pad(dout, ((0, 0), (k - 1, k - 1)))
    win_d = sliding_window_view(dout_p, k, axis=1)
    w_flip = np.flip(w, axis=2)
    dxp = np.einsum("otk,oik->it", win_d, w_flip, optimize=True)
    dx = dxp[:, pad : pad + x.shape[1]]
    return dx, dw, db


@dataclass
class OrbitSeq1D:
    """Time-preserving 1D CNN: ridge sequence → path at SEQ_T samples."""

    conv1_w: np.ndarray
    conv1_b: np.ndarray
    conv2_w: np.ndarray
    conv2_b: np.ndarray
    conv3_w: np.ndarray
    conv3_b: np.ndarray
    head_w: np.ndarray
    head_b: np.ndarray
    seq_t: int = SEQ_T
    n_ch: int = N_CH
    arch: str = ARCH_TAG

    @classmethod
    def create(cls, rng: np.random.Generator | None = None, seq_t: int = SEQ_T) -> OrbitSeq1D:
        rng = rng or np.random.default_rng(0)
        return cls(
            conv1_w=_he(rng, (32, N_CH, 9), N_CH * 9),
            conv1_b=np.zeros(32),
            conv2_w=_he(rng, (48, 32, 5), 32 * 5),
            conv2_b=np.zeros(48),
            conv3_w=_he(rng, (48, 48, 5), 48 * 5),
            conv3_b=np.zeros(48),
            head_w=rng.normal(0.0, 0.02, size=(2, 48, 1)).astype(np.float64),
            head_b=np.zeros(2),
            seq_t=int(seq_t),
        )

    def param_list(self) -> list[np.ndarray]:
        return [
            self.conv1_w, self.conv1_b, self.conv2_w, self.conv2_b,
            self.conv3_w, self.conv3_b, self.head_w, self.head_b,
        ]

    def _forward(self, feat: np.ndarray, cache: dict[str, Any]) -> np.ndarray:
        x = np.asarray(feat, dtype=np.float64)
        z1 = _conv1d(x, self.conv1_w, self.conv1_b)
        a1 = _relu(z1)
        z2 = _conv1d(a1, self.conv2_w, self.conv2_b)
        a2 = _relu(z2)
        z3 = _conv1d(a2, self.conv3_w, self.conv3_b)
        a3 = _relu(z3)
        out = _conv1d(a3, self.head_w, self.head_b)
        cache.update(x=x, z1=z1, a1=a1, z2=z2, a2=a2, z3=z3, a3=a3)
        return out.T

    def forward_path(self, feat: np.ndarray) -> np.ndarray:
        return self._forward(feat, {})

    def backward_from_cache(self, cache: dict[str, Any], dout_xy: np.ndarray) -> list[np.ndarray]:
        dout = np.asarray(dout_xy, dtype=np.float64).T
        d_a3, d_hw, d_hb = _conv1d_backward(cache["a3"], self.head_w, dout)
        d_z3 = d_a3 * (cache["z3"] > 0)
        d_a2, d_c3w, d_c3b = _conv1d_backward(cache["a2"], self.conv3_w, d_z3)
        d_z2 = d_a2 * (cache["z2"] > 0)
        d_a1, d_c2w, d_c2b = _conv1d_backward(cache["a1"], self.conv2_w, d_z2)
        d_z1 = d_a1 * (cache["z1"] > 0)
        _dx, d_c1w, d_c1b = _conv1d_backward(cache["x"], self.conv1_w, d_z1)
        return [d_c1w, d_c1b, d_c2w, d_c2b, d_c3w, d_c3b, d_hw, d_hb]

    def predict_xy(
        self,
        *,
        n_frames: int,
        wav_path: Path | str | None = None,
        audio: np.ndarray | None = None,
        sr: int | None = None,
        stft_db: np.ndarray | None = None,
        feats: RidgeFeatures | None = None,
    ) -> np.ndarray:
        if feats is None:
            feats = ridges_from_audio(stft_db=stft_db, wav_path=wav_path, audio=audio, sr=sr)
        seq = ridge_sequence_features(feats, out_t=self.seq_t)
        return _resample_path(self.forward_path(seq), int(n_frames))

    def _arrays(self) -> dict[str, np.ndarray]:
        return {
            "arch": np.array(self.arch),
            "conv1_w": self.conv1_w, "conv1_b": self.conv1_b,
            "conv2_w": self.conv2_w, "conv2_b": self.conv2_b,
            "conv3_w": self.conv3_w, "conv3_b": self.conv3_b,
            "head_w": self.head_w, "head_b": self.head_b,
            "seq_t": np.array([self.seq_t]),
            "n_ch": np.array([self.n_ch]),
        }

    def save(self, path: Path | str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f".{path.name}.writing.npz")
        np.savez_compressed(tmp, **self._arrays())
        tmp.replace(path)

    @classmethod
    def is_checkpoint(cls, path: Path | str) -> bool:
        path = Path(path)
        if not path.is_file():
            return False
        try:
            data = np.load(path, allow_pickle=True)
        except Exception:
            return False
        if "arch" in data.files and str(np.asarray(data["arch"]).reshape(-1)[0]) == ARCH_TAG:
            return True
        return (
            "head_w" in data.files
            and "conv1_w" in data.files
            and np.asarray(data["conv1_w"]).ndim == 3
        )

    @classmethod
    def load(cls, path: Path | str) -> OrbitSeq1D:
        path = Path(path)
        last_err: Exception | None = None
        for attempt in range(8):
            try:
                data = np.load(path, allow_pickle=True)
                if "head_w" not in data.files:
                    raise ValueError(f"{path} is not a ridge-seq checkpoint")
                return cls(
                    conv1_w=data["conv1_w"], conv1_b=data["conv1_b"],
                    conv2_w=data["conv2_w"], conv2_b=data["conv2_b"],
                    conv3_w=data["conv3_w"], conv3_b=data["conv3_b"],
                    head_w=data["head_w"], head_b=data["head_b"],
                    seq_t=int(data["seq_t"][0]) if "seq_t" in data.files else SEQ_T,
                    n_ch=int(data["n_ch"][0]) if "n_ch" in data.files else N_CH,
                    arch=str(np.asarray(data["arch"]).reshape(-1)[0]) if "arch" in data.files else ARCH_TAG,
                )
            except Exception as exc:  # noqa: BLE001
                last_err = exc
                time.sleep(0.04 * (attempt + 1))
        assert last_err is not None
        raise last_err


def _last_checkpoint_path(best_path: Path) -> Path:
    return best_path.with_name(best_path.stem + ".last.npz")


def _status_path(best_path: Path) -> Path:
    return best_path.with_name(best_path.stem + ".status.json")


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text)
    tmp.replace(path)


def _weights_tuple(model: OrbitSeq1D) -> tuple[np.ndarray, ...]:
    return tuple(p.copy() for p in model.param_list())


def _apply_weights(model: OrbitSeq1D, state: tuple[np.ndarray, ...]) -> None:
    for dst, src in zip(model.param_list(), state):
        dst[:] = src


def _save_best_atomic(model: OrbitSeq1D, best_state: tuple[np.ndarray, ...], path: Path) -> None:
    current = _weights_tuple(model)
    _apply_weights(model, best_state)
    model.save(path)
    _apply_weights(model, current)


def _is_seq_checkpoint_file(path: Path) -> bool:
    return OrbitSeq1D.is_checkpoint(path)


def _write_status(
    best_path: Path,
    *,
    epoch_completed: int,
    epochs: int,
    min_epochs: int,
    patience: int,
    best_val: float,
    best_epoch: int | None,
    running: bool,
    n_train: int,
    n_val: int,
) -> None:
    payload = {
        "running": running,
        "arch": ARCH_TAG,
        "inputs": "ridge_f_obs_and_A_env",
        "epoch_completed": int(epoch_completed),
        "epochs_target": int(epochs),
        "min_epochs": int(min_epochs),
        "patience": int(patience),
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


def _save_last_atomic(
    path: Path,
    *,
    model: OrbitSeq1D,
    best_state: tuple[np.ndarray, ...] | None,
    vel: list[np.ndarray],
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
    payload: dict[str, np.ndarray] = {
        **model._arrays(),
        "epoch_next": np.array([int(epoch_next)]),
        "best_val": np.array([float(best_val)]),
        "best_epoch": np.array([-1 if best_epoch is None else int(best_epoch)]),
        "train_idx": np.asarray(train_idx, dtype=np.int64),
        "val_idx": np.asarray(val_idx, dtype=np.int64),
        "seed": np.array([int(seed)]),
        "lr": np.array([float(lr)]),
        "history_json": np.array(json.dumps(history)),
        "rng_state": np.frombuffer(pickle.dumps(rng.bit_generator.state), dtype=np.uint8),
    }
    names = (
        "conv1_w", "conv1_b", "conv2_w", "conv2_b",
        "conv3_w", "conv3_b", "head_w", "head_b",
    )
    for name, arr in zip(names, best):
        payload[f"best_{name}"] = arr
    for i, v in enumerate(vel):
        payload[f"mom_{i}"] = v
    np.savez_compressed(tmp, **payload)
    tmp.replace(path)


def train_orbit_seq(
    batch_dir: Path | str,
    *,
    epochs: int = 250,
    min_epochs: int = 200,
    patience: int = 20,
    lr: float = 3e-4,
    momentum: float = 0.9,
    weight_decay: float = 1e-4,
    w_amp: float = 0.4,
    w_smooth: float = 0.03,
    seed: int = 0,
    holdout_families: tuple[str, ...] = ("u_turn",),
    checkpoint_path: Path | str | None = None,
    resume: bool = True,
) -> tuple[OrbitSeq1D, dict[str, Any]]:
    """Train ridge 1D CNN on simulated Phase 1 audio (no metadata inputs)."""
    batch = Phase1Batch.from_dir(batch_dir)
    rng = np.random.default_rng(seed)
    model = OrbitSeq1D.create(rng)
    best_path = Path(checkpoint_path) if checkpoint_path is not None else None
    last_path = _last_checkpoint_path(best_path) if best_path is not None else None
    vel = [np.zeros_like(p) for p in model.param_list()]

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
    resumed = False

    resume_src = None
    if resume and last_path is not None and _is_seq_checkpoint_file(last_path):
        resume_src = last_path
    elif resume and best_path is not None and _is_seq_checkpoint_file(best_path):
        resume_src = best_path
    elif resume and last_path is not None and last_path.is_file():
        print(
            f"Ignoring {last_path.name}: not a {ARCH_TAG} checkpoint (retraining from scratch).",
            flush=True,
        )
    elif resume and best_path is not None and best_path.is_file():
        print(
            f"Ignoring {best_path.name}: not a {ARCH_TAG} checkpoint (retraining from scratch).",
            flush=True,
        )

    if resume_src is not None:
        data = np.load(resume_src, allow_pickle=True)
        model = OrbitSeq1D.load(resume_src)
        names = (
            "conv1_w", "conv1_b", "conv2_w", "conv2_b",
            "conv3_w", "conv3_b", "head_w", "head_b",
        )
        if all(f"best_{n}" in data.files for n in names):
            best_state = tuple(np.asarray(data[f"best_{n}"]) for n in names)
        else:
            best_state = _weights_tuple(model)
        if "epoch_next" in data.files:
            start_epoch = int(data["epoch_next"][0])
            best_val = float(data["best_val"][0])
            be = int(data["best_epoch"][0])
            best_epoch = None if be < 0 else be
            train_idx = [int(i) for i in data["train_idx"].tolist()]
            val_idx = [int(i) for i in data["val_idx"].tolist()]
            history = json.loads(str(data["history_json"].item()))
            rng.bit_generator.state = pickle.loads(data["rng_state"].tobytes())
            vel = []
            for i, p in enumerate(model.param_list()):
                key = f"mom_{i}"
                vel.append(np.asarray(data[key]) if key in data.files else np.zeros_like(p))
            resumed = True
            print(
                f"Resumed {ARCH_TAG} at epoch {start_epoch}/{epochs} "
                f"(best_val={best_val:.4f} m)",
                flush=True,
            )
        else:
            best_state = _weights_tuple(model)
            print(
                f"Warm-started {ARCH_TAG} best weights; training from epoch 0/{epochs}.",
                flush=True,
            )

    print("Packing ridge sequences from STFT/WAV (no metadata)…", flush=True)
    packed: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]] = []
    for i in range(len(batch)):
        sample = batch.load(i)
        bundle = to_inference_bundle(sample)
        if bundle.stft_db is not None:
            ridges = ridges_from_audio(stft_db=bundle.stft_db)
        else:
            ridges = ridges_from_audio(wav_path=bundle.wav_path)
        feat = ridge_sequence_features(ridges, out_t=model.seq_t)
        gt = _sample_target(sample)
        gt_t = _resample_path(gt, model.seq_t)
        env = feat[2].astype(np.float64)
        packed.append((feat, gt_t, gt, env, sample.n_frames))
        if (i + 1) % 100 == 0 or i + 1 == len(batch):
            print(f"  packed {i + 1}/{len(batch)}", flush=True)

    def _report(running: bool, epoch_completed: int) -> dict[str, Any]:
        return {
            "batch_dir": str(Path(batch_dir).resolve()),
            "arch": ARCH_TAG,
            "epochs": epochs,
            "min_epochs": min_epochs,
            "patience": patience,
            "epoch_completed": epoch_completed,
            "n_train": len(train_idx),
            "n_val": len(val_idx),
            "holdout_families": list(holdout_families),
            "best_val_orbit_rms": best_val,
            "best_epoch": best_epoch,
            "history": history,
            "data_scope": "simulated_only",
            "inputs": "ridge_f_obs_A_env_quality",
            "forbidden_inputs": [
                "metadata/",
                "v",
                "h",
                "t_cpa",
                "path_family",
                "simulation_parameters",
            ],
            "source": "dopplersim_path2d_whiteboard_batch",
            "running": running,
            "resumed_ridge_seq": resumed,
            "best_checkpoint": str(best_path.resolve()) if best_path is not None else None,
        }

    if best_path is not None:
        _write_status(
            best_path,
            epoch_completed=start_epoch,
            epochs=epochs,
            min_epochs=min_epochs,
            patience=patience,
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

    stopped_at = start_epoch
    for epoch in range(start_epoch, int(epochs)):
        rng.shuffle(train_idx)
        train_losses = []
        params = model.param_list()
        for i in train_idx:
            feat, gt_t, _gt, env, _n_frames = packed[i]
            cache: dict[str, Any] = {}
            xy = model._forward(feat, cache)
            err = xy - gt_t
            loss_xy = float(np.mean(err**2))
            loss_s, g_s = _smooth_grad(xy)
            loss_a, g_a = _amp_grad(xy, env)
            loss = loss_xy + w_smooth * loss_s + w_amp * loss_a
            train_losses.append(loss)
            dout = (2.0 / err.size) * err.ravel() + w_smooth * g_s + w_amp * g_a
            grads = model.backward_from_cache(cache, dout.reshape(xy.shape))
            nrm = float(np.sqrt(sum(float(np.sum(g * g)) for g in grads)))
            if nrm > 5.0:
                scale = 5.0 / nrm
                grads = [g * scale for g in grads]
            for p, g, v in zip(params, grads, vel):
                if p.ndim > 1:
                    g = g + weight_decay * p
                v[:] = momentum * v + g
                p -= lr * v

        val_orbit = []
        for i in val_idx:
            feat, _gt_t, gt, _env, n_frames = packed[i]
            pred = _resample_path(model.forward_path(feat), n_frames)
            val_orbit.append(orbit_align(pred, gt).rms)
        mean_train = float(np.mean(train_losses)) if train_losses else 0.0
        mean_val = float(np.mean(val_orbit)) if val_orbit else float("inf")
        history.append(
            {"epoch": float(epoch), "train_mse": mean_train, "val_orbit_rms": mean_val}
        )
        print(
            f"epoch {epoch + 1}/{epochs}  train={mean_train:.4f}  "
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

        stopped_at = epoch + 1
        if best_path is not None and last_path is not None:
            _save_last_atomic(
                last_path,
                model=model,
                best_state=best_state,
                vel=vel,
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
                min_epochs=min_epochs,
                patience=patience,
                best_val=best_val,
                best_epoch=best_epoch,
                running=True,
                n_train=len(train_idx),
                n_val=len(val_idx),
            )

        if (
            epoch + 1 >= int(min_epochs)
            and best_epoch is not None
            and (epoch - best_epoch) >= int(patience)
        ):
            print(
                f"Early stop at epoch {epoch + 1}: no val improvement for {patience} "
                f"epochs after min_epochs={min_epochs} (best epoch {best_epoch + 1}).",
                flush=True,
            )
            break

    if best_state is not None:
        _apply_weights(model, best_state)

    report = _report(running=False, epoch_completed=stopped_at)
    if best_path is not None:
        model.save(best_path)
        _atomic_write_text(best_path.with_suffix(".json"), json.dumps(report, indent=2))
        _write_status(
            best_path,
            epoch_completed=stopped_at,
            epochs=epochs,
            min_epochs=min_epochs,
            patience=patience,
            best_val=best_val,
            best_epoch=best_epoch,
            running=False,
            n_train=len(train_idx),
            n_val=len(val_idx),
        )
    return model, report


def infer_orbit_seq(
    model: OrbitSeq1D,
    *,
    stft_db: np.ndarray | None = None,
    wav_path: Path | str | None = None,
    audio: np.ndarray | None = None,
    sr: int | None = None,
) -> np.ndarray:
    ridges = ridges_from_audio(stft_db=stft_db, wav_path=wav_path, audio=audio, sr=sr)
    n_frames = int(ridges.frame_times.shape[0])
    return model.predict_xy(n_frames=n_frames, feats=ridges)
