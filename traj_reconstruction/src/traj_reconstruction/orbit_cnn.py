"""2D CNN orbit model from complex STFT (audio only — no simulator metadata).

Inputs are a complex spectrogram built from WAV (or a magnitude STFT fallback):
relative dB (keeps 1/R drops), dB time-derivative, and peak-normalized real/imag.
The network never sees v, h, t_CPA, family labels, or any Phase 1 metadata.
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
from scipy.ndimage import zoom

from traj_reconstruction.contract import SPEC_SR_HZ, STFT_N_FFT
from traj_reconstruction.dataset import Phase1Batch, to_inference_bundle
from traj_reconstruction.flexible import _knots_to_path, _path_to_knots, _sample_target
from traj_reconstruction.frontend import _load_wav
from traj_reconstruction.kinematics import compute_stft_complex
from traj_reconstruction.orbit import orbit_align

ARCH_TAG = "complex_stft_cnn_v1"
N_CH = 4
FEAT_F = 48
FEAT_T = 96
N_KNOTS = 64
FMAX_HZ = 8000.0
REL_DB_SCALE = 40.0
FLAT_DIM = 48 * 6 * 12


def _resample_mono(audio: np.ndarray, sr: int, target_sr: int) -> np.ndarray:
    audio = np.asarray(audio, dtype=np.float64).reshape(-1)
    if int(sr) == int(target_sr) or audio.size < 2:
        return audio
    n_out = max(int(round(audio.size * float(target_sr) / float(sr))), 2)
    t_in = np.linspace(0.0, 1.0, audio.size, endpoint=False)
    t_out = np.linspace(0.0, 1.0, n_out, endpoint=False)
    return np.interp(t_out, t_in, audio)


def _resize2d(arr: np.ndarray, out_f: int, out_t: int) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float64)
    f, t = arr.shape
    if f == out_f and t == out_t:
        return arr
    return np.asarray(zoom(arr, (out_f / max(f, 1), out_t / max(t, 1)), order=1), dtype=np.float64)


def complex_stft_features(
    *,
    audio: np.ndarray | None = None,
    sr: int | None = None,
    wav_path: Path | str | None = None,
    stft_db: np.ndarray | None = None,
    out_f: int = FEAT_F,
    out_t: int = FEAT_T,
) -> np.ndarray:
    """(4, F, T) audio-only tensor. No metadata.

    Channels: relative dB (peak=0), dB time-derivative, peak-normalized real, imag.
    """
    spec: np.ndarray | None = None
    if wav_path is not None:
        audio, sr = _load_wav(Path(wav_path))
    if audio is not None:
        if sr is None:
            raise ValueError("sr required with audio")
        y = _resample_mono(audio, int(sr), SPEC_SR_HZ)
        spec = compute_stft_complex(y, sr=SPEC_SR_HZ)
        fmax_bin = int(FMAX_HZ / (SPEC_SR_HZ / STFT_N_FFT)) + 1
        spec = spec[: min(fmax_bin, spec.shape[0])]
    elif stft_db is not None:
        mag = np.power(10.0, np.asarray(stft_db, dtype=np.float64) / 20.0)
        spec = mag.astype(np.complex128)
    else:
        raise ValueError("provide wav/audio or stft_db")

    mag = np.abs(spec)
    peak = float(np.max(mag) + 1e-12)
    db = 20.0 * np.log10(np.maximum(mag, 1e-10))
    db = db - float(np.max(db))
    ddb = np.zeros_like(db)
    ddb[:, 1:] = db[:, 1:] - db[:, :-1]
    real = spec.real / peak
    imag = spec.imag / peak
    feat = np.stack(
        [
            _resize2d(db, out_f, out_t) / REL_DB_SCALE,
            _resize2d(ddb, out_f, out_t),
            _resize2d(real, out_f, out_t),
            _resize2d(imag, out_f, out_t),
        ],
        axis=0,
    ).astype(np.float32)
    feat[1:] = np.clip(feat[1:], -8.0, 8.0)
    return feat


def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0.0)


def _conv2d(x: np.ndarray, w: np.ndarray, b: np.ndarray) -> np.ndarray:
    cin, h, wdt = x.shape
    cout, cin_w, kh, kw = w.shape
    if cin != cin_w:
        raise ValueError("conv channel mismatch")
    ph, pw = kh // 2, kw // 2
    xp = np.pad(x, ((0, 0), (ph, ph), (pw, pw)))
    windows = sliding_window_view(xp, (kh, kw), axis=(1, 2))
    return np.einsum("ihwkl,oikl->ohw", windows, w, optimize=True) + b[:, None, None]


def _conv2d_backward(
    x: np.ndarray, w: np.ndarray, dout: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cout, cin, kh, kw = w.shape
    ph, pw = kh // 2, kw // 2
    xp = np.pad(x, ((0, 0), (ph, ph), (pw, pw)))
    windows = sliding_window_view(xp, (kh, kw), axis=(1, 2))
    dw = np.einsum("ihwkl,ohw->oikl", windows, dout, optimize=True)
    db = dout.reshape(cout, -1).sum(axis=1)
    dout_p = np.pad(dout, ((0, 0), (kh - 1, kh - 1), (kw - 1, kw - 1)))
    win_d = sliding_window_view(dout_p, (kh, kw), axis=(1, 2))
    w_flip = np.flip(w, axis=(2, 3))
    dxp = np.einsum("ohwkl,oikl->ihw", win_d, w_flip, optimize=True)
    dx = dxp[:, ph : ph + x.shape[1], pw : pw + x.shape[2]]
    return dx, dw, db


def _maxpool2(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    c, h, w = x.shape
    h2, w2 = h // 2 * 2, w // 2 * 2
    x = x[:, :h2, :w2]
    blocks = (
        x.reshape(c, h2 // 2, 2, w2 // 2, 2)
        .transpose(0, 1, 3, 2, 4)
        .reshape(c, h2 // 2, w2 // 2, 4)
    )
    idx = np.argmax(blocks, axis=-1).astype(np.int8)
    out = np.max(blocks, axis=-1)
    return out, idx


def _maxpool2_backward(idx: np.ndarray, dout: np.ndarray, like: np.ndarray) -> np.ndarray:
    c, h, w = like.shape
    h2, w2 = h // 2 * 2, w // 2 * 2
    dx = np.zeros((c, h2, w2), dtype=np.float64)
    c_ix, i_ix, j_ix = np.indices(idx.shape)
    di = idx // 2
    dj = idx % 2
    dx[c_ix, 2 * i_ix + di, 2 * j_ix + dj] = dout
    if (h, w) != (h2, w2):
        full = np.zeros_like(like, dtype=np.float64)
        full[:, :h2, :w2] = dx
        return full
    return dx


def _he(rng: np.random.Generator, shape: tuple[int, ...], fan_in: int) -> np.ndarray:
    return rng.normal(0.0, np.sqrt(2.0 / max(fan_in, 1)), size=shape).astype(np.float64)


@dataclass
class OrbitCNN:
    """Small 2D CNN: complex STFT → canonical path knots."""

    conv1_w: np.ndarray
    conv1_b: np.ndarray
    conv2_w: np.ndarray
    conv2_b: np.ndarray
    conv3_w: np.ndarray
    conv3_b: np.ndarray
    fc1_w: np.ndarray
    fc1_b: np.ndarray
    fc2_w: np.ndarray
    fc2_b: np.ndarray
    n_knots: int = N_KNOTS
    feat_f: int = FEAT_F
    feat_t: int = FEAT_T
    arch: str = ARCH_TAG

    @classmethod
    def create(cls, rng: np.random.Generator | None = None, n_knots: int = N_KNOTS) -> OrbitCNN:
        rng = rng or np.random.default_rng(0)
        flat = FLAT_DIM
        hidden = 256
        out = int(n_knots) * 2
        return cls(
            conv1_w=_he(rng, (16, N_CH, 5, 5), N_CH * 25),
            conv1_b=np.zeros(16),
            conv2_w=_he(rng, (32, 16, 3, 3), 16 * 9),
            conv2_b=np.zeros(32),
            conv3_w=_he(rng, (48, 32, 3, 3), 32 * 9),
            conv3_b=np.zeros(48),
            fc1_w=_he(rng, (flat, hidden), flat),
            fc1_b=np.zeros(hidden),
            fc2_w=_he(rng, (hidden, out), hidden),
            fc2_b=np.zeros(out),
            n_knots=int(n_knots),
        )

    def param_list(self) -> list[np.ndarray]:
        return [
            self.conv1_w, self.conv1_b, self.conv2_w, self.conv2_b,
            self.conv3_w, self.conv3_b, self.fc1_w, self.fc1_b, self.fc2_w, self.fc2_b,
        ]

    def _forward(self, feat: np.ndarray, cache: dict[str, Any]) -> np.ndarray:
        x = np.asarray(feat, dtype=np.float64)
        z1 = _conv2d(x, self.conv1_w, self.conv1_b)
        a1 = _relu(z1)
        p1, i1 = _maxpool2(a1)
        z2 = _conv2d(p1, self.conv2_w, self.conv2_b)
        a2 = _relu(z2)
        p2, i2 = _maxpool2(a2)
        z3 = _conv2d(p2, self.conv3_w, self.conv3_b)
        a3 = _relu(z3)
        p3, i3 = _maxpool2(a3)
        flat = p3.ravel()
        h = _relu(flat @ self.fc1_w + self.fc1_b)
        out = h @ self.fc2_w + self.fc2_b
        cache.update(
            x=x, z1=z1, a1=a1, p1=p1, i1=i1, z2=z2, a2=a2, p2=p2, i2=i2,
            z3=z3, a3=a3, p3=p3, i3=i3, flat=flat, h=h,
        )
        return out

    def forward_knots(self, feat: np.ndarray) -> np.ndarray:
        return self._forward(feat, {}).reshape(self.n_knots, 2)

    def backward(self, feat: np.ndarray, dout: np.ndarray) -> list[np.ndarray]:
        cache: dict[str, Any] = {}
        self._forward(feat, cache)
        return self.backward_from_cache(cache, dout)

    def backward_from_cache(self, cache: dict[str, Any], dout: np.ndarray) -> list[np.ndarray]:
        d_fc2_w = np.outer(cache["h"], dout)
        d_fc2_b = dout
        d_h = (self.fc2_w @ dout) * (cache["h"] > 0)
        d_fc1_w = np.outer(cache["flat"], d_h)
        d_fc1_b = d_h
        d_flat = self.fc1_w @ d_h
        d_p3 = d_flat.reshape(cache["p3"].shape)
        d_a3 = _maxpool2_backward(cache["i3"], d_p3, cache["a3"])
        d_z3 = d_a3 * (cache["z3"] > 0)
        d_p2, d_c3w, d_c3b = _conv2d_backward(cache["p2"], self.conv3_w, d_z3)
        d_a2 = _maxpool2_backward(cache["i2"], d_p2, cache["a2"])
        d_z2 = d_a2 * (cache["z2"] > 0)
        d_p1, d_c2w, d_c2b = _conv2d_backward(cache["p1"], self.conv2_w, d_z2)
        d_a1 = _maxpool2_backward(cache["i1"], d_p1, cache["a1"])
        d_z1 = d_a1 * (cache["z1"] > 0)
        _dx, d_c1w, d_c1b = _conv2d_backward(cache["x"], self.conv1_w, d_z1)
        return [d_c1w, d_c1b, d_c2w, d_c2b, d_c3w, d_c3b, d_fc1_w, d_fc1_b, d_fc2_w, d_fc2_b]

    def predict_xy(
        self,
        *,
        n_frames: int,
        wav_path: Path | str | None = None,
        audio: np.ndarray | None = None,
        sr: int | None = None,
        stft_db: np.ndarray | None = None,
    ) -> np.ndarray:
        feat = complex_stft_features(
            audio=audio, sr=sr, wav_path=wav_path, stft_db=stft_db,
            out_f=self.feat_f, out_t=self.feat_t,
        )
        return _knots_to_path(self.forward_knots(feat), n_frames)

    def _arrays(self) -> dict[str, np.ndarray]:
        return {
            "arch": np.array(self.arch),
            "conv1_w": self.conv1_w, "conv1_b": self.conv1_b,
            "conv2_w": self.conv2_w, "conv2_b": self.conv2_b,
            "conv3_w": self.conv3_w, "conv3_b": self.conv3_b,
            "fc1_w": self.fc1_w, "fc1_b": self.fc1_b,
            "fc2_w": self.fc2_w, "fc2_b": self.fc2_b,
            "n_knots": np.array([self.n_knots]),
            "feat_f": np.array([self.feat_f]),
            "feat_t": np.array([self.feat_t]),
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
        return "conv1_w" in data.files and np.asarray(data["conv1_w"]).ndim == 4

    @classmethod
    def load(cls, path: Path | str) -> OrbitCNN:
        path = Path(path)
        last_err: Exception | None = None
        for attempt in range(8):
            try:
                data = np.load(path, allow_pickle=True)
                if "conv1_w" not in data.files:
                    raise ValueError(f"{path} is not a complex-STFT CNN checkpoint")
                return cls(
                    conv1_w=data["conv1_w"], conv1_b=data["conv1_b"],
                    conv2_w=data["conv2_w"], conv2_b=data["conv2_b"],
                    conv3_w=data["conv3_w"], conv3_b=data["conv3_b"],
                    fc1_w=data["fc1_w"], fc1_b=data["fc1_b"],
                    fc2_w=data["fc2_w"], fc2_b=data["fc2_b"],
                    n_knots=int(data["n_knots"][0]),
                    feat_f=int(data["feat_f"][0]),
                    feat_t=int(data["feat_t"][0]),
                    arch=str(data["arch"]) if "arch" in data.files else ARCH_TAG,
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


def _weights_tuple(model: OrbitCNN) -> tuple[np.ndarray, ...]:
    return tuple(p.copy() for p in model.param_list())


def _apply_weights(model: OrbitCNN, state: tuple[np.ndarray, ...]) -> None:
    for dst, src in zip(model.param_list(), state):
        dst[:] = src


def _save_best_atomic(model: OrbitCNN, best_state: tuple[np.ndarray, ...], path: Path) -> None:
    current = _weights_tuple(model)
    _apply_weights(model, best_state)
    model.save(path)
    _apply_weights(model, current)


def _is_cnn_checkpoint_file(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        data = np.load(path, allow_pickle=True)
    except Exception:
        return False
    if "conv1_w" not in data.files or np.asarray(data["conv1_w"]).ndim != 4:
        return False
    if "arch" in data.files and str(np.asarray(data["arch"]).reshape(-1)[0]) not in (
        ARCH_TAG,
        "",
    ):
        return False
    return True


def _spec_amp_envelope(feat: np.ndarray, n_knots: int) -> np.ndarray:
    """Peak-normalized 1/R-like envelope from relative-dB channel (audio only)."""
    rel_db = np.asarray(feat[0], dtype=np.float64) * REL_DB_SCALE
    lin = np.power(10.0, rel_db / 20.0)
    env = lin.mean(axis=0)
    env = env / max(float(np.max(env)), 1e-12)
    t = np.linspace(0.0, 1.0, env.size)
    tq = np.linspace(0.0, 1.0, int(n_knots))
    return np.interp(tq, t, env)


def _amp_grad(knots: np.ndarray, env: np.ndarray) -> tuple[float, np.ndarray]:
    env = np.asarray(env, dtype=np.float64)
    env_n = env / max(float(np.mean(env)), 1e-12)
    r = np.sqrt(np.sum(knots**2, axis=1)).clip(1.0)
    inv = 1.0 / r
    mu = float(np.mean(inv) + 1e-12)
    inv_n = inv / mu
    err = inv_n - env_n
    loss = float(np.mean(err**2))
    n = float(err.size)
    d_inv_n = (2.0 / n) * err
    d_inv = d_inv_n / mu - (np.sum(d_inv_n * inv) / (mu * mu * n))
    d_r = -d_inv / (r**2)
    d_knots = (d_r / r)[:, None] * knots
    return loss, d_knots.ravel()


def _smooth_grad(knots: np.ndarray) -> tuple[float, np.ndarray]:
    d2 = np.diff(knots, n=2, axis=0)
    loss = float(np.mean(d2**2))
    d_d2 = (2.0 / d2.size) * d2
    g = np.zeros_like(knots)
    g[:-2] += d_d2
    g[1:-1] += -2.0 * d_d2
    g[2:] += d_d2
    return loss, g.ravel()


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
        "inputs": "complex_stft_from_wav",
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
    model: OrbitCNN,
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
        "conv1_w", "conv1_b", "conv2_w", "conv2_b", "conv3_w", "conv3_b",
        "fc1_w", "fc1_b", "fc2_w", "fc2_b",
    )
    for name, arr in zip(names, best):
        payload[f"best_{name}"] = arr
    for i, v in enumerate(vel):
        payload[f"mom_{i}"] = v
    np.savez_compressed(tmp, **payload)
    tmp.replace(path)


def train_orbit_cnn(
    batch_dir: Path | str,
    *,
    epochs: int = 250,
    min_epochs: int = 200,
    patience: int = 20,
    lr: float = 3e-4,
    momentum: float = 0.9,
    weight_decay: float = 1e-4,
    w_amp: float = 0.35,
    w_smooth: float = 0.02,
    seed: int = 0,
    holdout_families: tuple[str, ...] = ("u_turn",),
    checkpoint_path: Path | str | None = None,
    resume: bool = True,
) -> tuple[OrbitCNN, dict[str, Any]]:
    """Train 2D CNN on complex STFT → canonical path (WAV/STFT only; no metadata).

    Resume is allowed only from ``complex_stft_cnn_v1`` checkpoints. Log-mel / MLP
    files are ignored so we never warm-start from collapsed magnitude features.
    """
    batch = Phase1Batch.from_dir(batch_dir)
    rng = np.random.default_rng(seed)
    model = OrbitCNN.create(rng)
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
    if resume and last_path is not None and _is_cnn_checkpoint_file(last_path):
        resume_src = last_path
    elif resume and best_path is not None and _is_cnn_checkpoint_file(best_path):
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
        model = OrbitCNN.load(resume_src)
        names = (
            "conv1_w", "conv1_b", "conv2_w", "conv2_b", "conv3_w", "conv3_b",
            "fc1_w", "fc1_b", "fc2_w", "fc2_b",
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

    print("Packing complex-STFT features from WAV (no metadata)…", flush=True)
    packed: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]] = []
    for i in range(len(batch)):
        sample = batch.load(i)
        bundle = to_inference_bundle(sample)
        if bundle.wav_path is not None:
            feat = complex_stft_features(wav_path=bundle.wav_path)
        else:
            assert bundle.stft_db is not None
            feat = complex_stft_features(stft_db=bundle.stft_db)
        gt = _sample_target(sample)
        knots_gt = _path_to_knots(gt, model.n_knots)
        env = _spec_amp_envelope(feat, model.n_knots)
        packed.append((feat, knots_gt, gt, env, sample.n_frames))
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
            "inputs": "wav_complex_stft_or_stft_fallback",
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
            "resumed_complex_stft": resumed,
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
            feat, knots_gt, _gt, env, _n_frames = packed[i]
            cache: dict[str, Any] = {}
            out = model._forward(feat, cache)
            knots = out.reshape(model.n_knots, 2)
            err = knots - knots_gt
            loss_xy = float(np.mean(err**2))
            loss_s, g_s = _smooth_grad(knots)
            loss_a, g_a = _amp_grad(knots, env)
            loss = loss_xy + w_smooth * loss_s + w_amp * loss_a
            train_losses.append(loss)
            dout = (2.0 / err.size) * err.ravel() + w_smooth * g_s + w_amp * g_a
            grads = model.backward_from_cache(cache, dout)
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
            feat, _knots_gt, gt, _env, n_frames = packed[i]
            pred = _knots_to_path(model.forward_knots(feat), n_frames)
            val_orbit.append(orbit_align(pred, gt).rms)
        mean_train = float(np.mean(train_losses)) if train_losses else 0.0
        mean_val = float(np.mean(val_orbit)) if val_orbit else float("inf")
        history.append(
            {
                "epoch": float(epoch),
                "train_mse": mean_train,
                "val_orbit_rms": mean_val,
            }
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


def infer_orbit_cnn(
    model: OrbitCNN,
    *,
    stft_db: np.ndarray | None = None,
    wav_path: Path | str | None = None,
    audio: np.ndarray | None = None,
    sr: int | None = None,
) -> np.ndarray:
    """Audio-only CNN inference. Prefers WAV so phase and 1/R drops are kept."""
    if wav_path is not None or audio is not None:
        if wav_path is not None:
            audio, sr = _load_wav(Path(wav_path))
        if audio is None or sr is None:
            raise ValueError("audio and sr required")
        y = _resample_mono(audio, int(sr), SPEC_SR_HZ)
        n_frames = compute_stft_complex(y, sr=SPEC_SR_HZ).shape[1]
        return model.predict_xy(n_frames=n_frames, audio=audio, sr=sr)
    if stft_db is None:
        raise ValueError("provide wav_path/audio or stft_db")
    return model.predict_xy(n_frames=int(stft_db.shape[1]), stft_db=stft_db)


def load_orbit_model(path: Path | str):
    """Load ridge-seq, complex-STFT CNN, or legacy OrbitMLP from a checkpoint."""
    path = Path(path)
    from traj_reconstruction.orbit_seq import OrbitSeq1D

    if OrbitSeq1D.is_checkpoint(path):
        return OrbitSeq1D.load(path)
    if OrbitCNN.is_checkpoint(path):
        return OrbitCNN.load(path)
    from traj_reconstruction.flexible import OrbitMLP

    return OrbitMLP.load(path)


def infer_learned_orbit(
    model: Any,
    *,
    stft_db: np.ndarray | None = None,
    wav_path: Path | str | None = None,
    audio: np.ndarray | None = None,
    sr: int | None = None,
) -> np.ndarray:
    from traj_reconstruction.orbit_seq import OrbitSeq1D, infer_orbit_seq

    if isinstance(model, OrbitSeq1D):
        return infer_orbit_seq(
            model, stft_db=stft_db, wav_path=wav_path, audio=audio, sr=sr
        )
    if isinstance(model, OrbitCNN):
        return infer_orbit_cnn(
            model, stft_db=stft_db, wav_path=wav_path, audio=audio, sr=sr
        )
    from traj_reconstruction.flexible import infer_orbit_mlp

    return infer_orbit_mlp(model, stft_db=stft_db, wav_path=wav_path)

