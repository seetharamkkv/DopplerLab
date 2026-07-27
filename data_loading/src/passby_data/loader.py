"""Framework-agnostic loaders: paths + labels + optional mel tensors."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Sequence

import numpy as np

from .config import AudioConfig, DEFAULT_REAL_ROOT
from .leakage import assert_no_leakage
from .mel_stats import MelStats, compute_mel_stats, load_mel_stats
from .splits import SplitBundle, load_split, partition_paths


@dataclass(frozen=True)
class Partition:
    name: str
    uids: list[str]
    paths: list[str]
    speeds_kmh: np.ndarray


@dataclass
class ModelDataBundle:
    """Everything a trainer needs without leaking val/test into stats."""

    split: SplitBundle
    fit: Partition
    val: Partition
    test: Partition
    stats: MelStats
    domain_root: Path


def _partition(name: str, root: Path, uids: list[str]) -> Partition:
    paths, speeds = partition_paths(root, uids)
    return Partition(name=name, uids=list(uids), paths=paths, speeds_kmh=speeds)


def load_for_training(
    split: SplitBundle | str | Path,
    *,
    real_root: str | Path = DEFAULT_REAL_ROOT,
    synth_root: str | Path | None = None,
    mode: Literal["real", "synth", "mixed"] = "real",
    stats_path: str | Path | None = None,
    fit_stats: bool = True,
    cfg: AudioConfig | None = None,
) -> ModelDataBundle:
    """Build fit/val/test partitions with train-only mel stats.

    Parameters
    ----------
    mode
        ``real`` — real fit/val/test.
        ``synth`` — synth fit (+ synth val scenes); test still real by default
        via ``split.test_uids`` resolved under ``real_root``.
        ``mixed`` — real+synth fit; real val; real test.
    """
    if not isinstance(split, SplitBundle):
        split = load_split(split)

    real_root = Path(real_root)
    cfg = cfg or AudioConfig()

    fit_uids = list(split.fit_uids or split.train_uids)
    val_uids = list(split.val_uids)
    test_uids = list(split.test_uids)

    if mode == "real":
        fit = _partition("fit", real_root, fit_uids)
        val = _partition("val", real_root, val_uids) if val_uids else _partition(
            "val", real_root, []
        )
        test = _partition("test", real_root, test_uids)
        stats_uids = fit_uids
        stats_paths = fit.paths
        domain_root = real_root
    elif mode == "synth":
        if synth_root is None:
            raise ValueError("synth_root required for mode='synth'")
        synth_root = Path(synth_root)
        held = set(split.val_scene_keys) | set(split.test_scene_keys)
        from .catalog import condition_key

        synth_fit = [
            u for u in split.synth_train_uids if condition_key(u) not in held
        ]
        synth_val = [
            u
            for u in split.synth_train_uids
            if condition_key(u) in set(split.val_scene_keys)
        ]
        fit = _partition("fit", synth_root, synth_fit)
        val = _partition("val", synth_root, synth_val)
        test = _partition("test", real_root, test_uids)
        stats_uids = synth_fit
        stats_paths = fit.paths
        domain_root = synth_root
    else:  # mixed
        if synth_root is None:
            raise ValueError("synth_root required for mode='mixed'")
        synth_root = Path(synth_root)
        from .catalog import condition_key

        held = set(split.val_scene_keys) | set(split.test_scene_keys)
        synth_fit = [
            u for u in split.synth_train_uids if condition_key(u) not in held
        ]
        real_fit_part = _partition("fit_real", real_root, fit_uids)
        synth_fit_part = _partition("fit_synth", synth_root, synth_fit)
        fit = Partition(
            name="fit",
            uids=real_fit_part.uids + [f"synth:{u}" for u in synth_fit_part.uids],
            paths=real_fit_part.paths + synth_fit_part.paths,
            speeds_kmh=np.concatenate(
                [real_fit_part.speeds_kmh, synth_fit_part.speeds_kmh]
            ),
        )
        val = _partition("val", real_root, val_uids)
        test = _partition("test", real_root, test_uids)
        stats_uids = fit_uids + synth_fit
        stats_paths = fit.paths
        domain_root = real_root

    assert_no_leakage(
        split,
        stats_uids=[
            u.split(":", 1)[-1] if ":" in u else u for u in stats_uids
        ],
        eval_uids=test_uids,
        seresnet_train_uids=[
            u.split(":", 1)[-1] if ":" in u else u for u in fit.uids
        ],
    )

    if stats_path and Path(stats_path).is_file() and not fit_stats:
        stats = load_mel_stats(stats_path)
    else:
        stats = compute_mel_stats(
            stats_paths, cfg=cfg, save_path=stats_path, verbose=True
        )

    return ModelDataBundle(
        split=split,
        fit=fit,
        val=val,
        test=test,
        stats=stats,
        domain_root=domain_root,
    )


def preprocess_mel(
    wav_path: str | Path,
    stats: MelStats,
    *,
    cfg: AudioConfig | None = None,
    augment: bool = False,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Load wav → peak-norm → (optional augment) → mel-dB → z-score.

    Returns float32 array shaped ``(n_mels, n_frames, 1)``.
    """
    import librosa

    cfg = cfg or AudioConfig()
    rng = rng or np.random.default_rng()
    audio, _ = librosa.load(str(wav_path), sr=cfg.sample_rate, mono=True)
    n = cfg.audio_length_samples
    if len(audio) > n:
        audio = audio[:n]
    else:
        audio = np.pad(audio, (0, n - len(audio)), "constant")

    if augment:
        if rng.random() < 0.8:
            gain_db = float(rng.uniform(-6.0, 6.0))
            audio = audio * (10.0 ** (gain_db / 20.0))
            snr_db = float(rng.uniform(10.0, 25.0))
            power = float(np.sum(audio**2) / max(len(audio), 1))
            if power > 1e-6:
                noise_power = power / (10 ** (snr_db / 10))
                audio = audio + rng.normal(0.0, np.sqrt(noise_power), len(audio))

    if np.max(np.abs(audio)) > 0:
        audio = audio / np.max(np.abs(audio))

    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=cfg.sample_rate,
        n_fft=cfg.n_fft,
        hop_length=cfg.hop_length,
        n_mels=cfg.n_mels,
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_norm = (mel_db - stats.mean) / stats.std
    return np.expand_dims(mel_norm, axis=-1).astype(np.float32)


def iter_mels(
    partition: Partition,
    stats: MelStats,
    *,
    cfg: AudioConfig | None = None,
    augment: bool = False,
    seed: int = 42,
):
    """Yield ``(mel, speed_kmh, uid)`` for a partition."""
    rng = np.random.default_rng(seed)
    for path, speed, uid in zip(
        partition.paths, partition.speeds_kmh, partition.uids, strict=True
    ):
        yield preprocess_mel(path, stats, cfg=cfg, augment=augment, rng=rng), float(
            speed
        ), uid
