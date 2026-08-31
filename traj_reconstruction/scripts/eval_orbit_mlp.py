#!/usr/bin/env python3
"""Held-out OrbitMLP eval on a DopplerSim 2D whiteboard batch.

Reconstructs the training split (seed + OrbitMLP.create rng consumption),
assigns each clip to the product path-family subset from geometry, and
scores STFT-only predictions with orbit RMS (plus CPA / speed).

Also scores the same checkpoint on in-package labeled family prototypes
(tone acoustics — domain-shifted vs vehicle STFTs).
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from traj_reconstruction.contract import PATH_FAMILIES
from traj_reconstruction.dataset import Phase1Batch, to_inference_bundle
from traj_reconstruction.flexible import (
    OrbitMLP,
    _sample_target,
    infer_orbit_mlp,
)
from traj_reconstruction.kinematics import (
    canonical_state_frames,
    compute_stft_db,
    interpolate_state,
    polyline_arclength,
    stft_frame_times,
    stft_n_frames,
)
from traj_reconstruction.orbit import orbit_align, xy_from_state
from traj_reconstruction.path_families import generate_path
from traj_reconstruction.synthesize import synthesize_tone_on_path


FAMILIES = ("straight", "arc", "s_curve", "u_turn", "multi_cpa")


def _training_split_indices(
    n: int, *, seed: int, holdout_families: tuple[str, ...], families: list[str]
) -> tuple[list[int], list[int]]:
    """Match ``train_orbit_mlp``: create() consumes rng, then 20% val if no family holdout."""
    rng = np.random.default_rng(seed)
    OrbitMLP.create(rng)
    train_idx = [i for i, fam in enumerate(families) if fam not in holdout_families]
    val_idx = [i for i, fam in enumerate(families) if fam in holdout_families]
    if not val_idx:
        order = list(train_idx)
        rng.shuffle(order)
        n_val = max(1, len(order) // 5)
        val_idx = order[:n_val]
        train_idx = order[n_val:]
    return train_idx, val_idx


def assign_path_family(xy: np.ndarray) -> dict[str, Any]:
    """Map a mic-centric polyline onto the product family subset."""
    xy = np.asarray(xy, dtype=np.float64)
    if xy.ndim != 2 or xy.shape[1] != 2 or len(xy) < 3:
        return {
            "family": "straight",
            "max_turn_deg": 0.0,
            "net_heading_deg": 0.0,
            "n_inflections": 0,
            "n_cpa": 0,
        }

    dxy = np.diff(xy, axis=0)
    headings = np.arctan2(dxy[:, 1], dxy[:, 0])
    unwrapped = np.unwrap(headings)
    dhead = np.diff(unwrapped)
    max_turn_deg = float(np.rad2deg(np.max(np.abs(dhead)))) if len(dhead) else 0.0
    net_heading_deg = float(np.rad2deg(abs(unwrapped[-1] - unwrapped[0])))

    signs = np.sign(dhead)
    signs = signs[np.abs(dhead) > np.deg2rad(2.0)]
    n_inflect = int(np.sum(signs[1:] * signs[:-1] < 0)) if len(signs) > 1 else 0

    r = np.sqrt(xy[:, 0] ** 2 + xy[:, 1] ** 2)
    if len(r) >= 9:
        k = 7
        r_s = np.convolve(r, np.ones(k) / k, mode="same")
    else:
        r_s = r
    n_cpa = 0
    for i in range(2, len(r_s) - 2):
        if r_s[i] <= r_s[i - 1] and r_s[i] <= r_s[i + 1]:
            left = float(np.max(r_s[: i + 1]))
            right = float(np.max(r_s[i:]))
            if min(left, right) - float(r_s[i]) >= 2.0:
                n_cpa += 1
    n_cpa = max(n_cpa, 1)

    if n_cpa >= 2:
        family = "multi_cpa"
    elif net_heading_deg >= 135.0 or max_turn_deg >= 90.0:
        family = "u_turn"
    elif n_inflect >= 2:
        family = "s_curve"
    elif max_turn_deg >= 10.0 or net_heading_deg >= 20.0:
        family = "arc"
    else:
        family = "straight"

    return {
        "family": family,
        "max_turn_deg": max_turn_deg,
        "net_heading_deg": net_heading_deg,
        "n_inflections": n_inflect,
        "n_cpa": n_cpa,
    }


def _cpa_m(xy: np.ndarray) -> float:
    return float(np.min(np.sqrt(xy[:, 0] ** 2 + xy[:, 1] ** 2)))


def _speed_mps(xy: np.ndarray, duration_s: float) -> float:
    length = float(polyline_arclength(xy)[-1])
    return length / max(float(duration_s), 1e-6)


def _summarize(rms: list[float]) -> dict[str, float | None]:
    if not rms:
        return {"n": 0, "mean": None, "median": None, "p90": None, "max": None}
    a = np.asarray(rms, dtype=np.float64)
    return {
        "n": int(len(a)),
        "mean": float(np.mean(a)),
        "median": float(np.median(a)),
        "p90": float(np.percentile(a, 90)),
        "max": float(np.max(a)),
    }


def _score_pair(
    pred: np.ndarray, gt: np.ndarray, *, duration_s: float
) -> dict[str, float]:
    n = min(len(pred), len(gt))
    orb = orbit_align(pred[:n], gt[:n])
    gt_cpa = _cpa_m(gt[:n])
    pred_cpa = _cpa_m(orb.aligned_pred)
    gt_v = _speed_mps(gt[:n], duration_s)
    pred_v = _speed_mps(pred[:n], duration_s)
    return {
        "orbit_rms_m": orb.rms,
        "orbit_len_norm": orb.length_normalized_rms,
        "cpa_gt_m": gt_cpa,
        "cpa_pred_m": pred_cpa,
        "cpa_abs_err_m": abs(pred_cpa - gt_cpa),
        "cpa_rel_err": abs(pred_cpa - gt_cpa) / max(gt_cpa, 1e-6),
        "speed_gt_mps": gt_v,
        "speed_pred_mps": pred_v,
        "speed_rel_err": abs(pred_v - gt_v) / max(gt_v, 1e-6),
        "orbit_reflected": float(orb.reflected),
    }


def eval_batch(
    batch: Phase1Batch,
    model: OrbitMLP,
    indices: list[int],
    split: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for i in indices:
        sample = batch.load(i)
        bundle = to_inference_bundle(sample)
        assert bundle.stft_db is not None
        pred = infer_orbit_mlp(model, stft_db=bundle.stft_db)
        gt = _sample_target(sample)
        duration = float(sample.frame_times[-1] - sample.frame_times[0])
        metrics = _score_pair(pred, gt, duration_s=duration)
        geom = assign_path_family(gt)
        row_meta = batch.rows[i]
        rows.append(
            {
                "split": split,
                "sample_id": sample.sample_id,
                "vehicle": row_meta.get("vehicle_class"),
                "speed_kmph": float(row_meta.get("speed_kmph") or 0.0),
                "assigned_family": geom["family"],
                **{k: geom[k] for k in geom if k != "family"},
                **metrics,
            }
        )
    return rows


def eval_family_prototypes(
    model: OrbitMLP, *, n_per: int = 8, seed: int = 1
) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    for fam in FAMILIES:
        for k in range(int(n_per)):
            spec = generate_path(
                fam,
                speed_mps=float(rng.uniform(10.0, 28.0)),
                cpa_distance_m=float(rng.uniform(10.0, 22.0)),
                rng=rng,
            )
            synth = synthesize_tone_on_path(
                spec.polyline, speed_mps=spec.speed_mps, f0_hz=500.0
            )
            audio = synth["audio"]
            sr = int(synth["sr"])
            stft = compute_stft_db(audio, sr=sr)
            n = len(audio)
            times = stft_frame_times(stft_n_frames(n))
            state = interpolate_state(
                synth["trajectory"]["t"], synth["trajectory"]["state"], times
            )
            can, _ = canonical_state_frames(state)
            gt = xy_from_state(can)
            pred = model.predict_xy(stft, gt.shape[0])
            duration = float(times[-1] - times[0]) if len(times) > 1 else 1.0
            metrics = _score_pair(pred, gt, duration_s=duration)
            rows.append(
                {
                    "split": "family_prototypes_tone",
                    "family": fam,
                    "k": k,
                    "speed_mps": spec.speed_mps,
                    **metrics,
                    "note": "in-package pure-tone STFT; not vehicle DopplerSim acoustics",
                }
            )
    return rows


def _by_family(rows: list[dict[str, Any]], key: str = "assigned_family") -> dict[str, Any]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        grouped[str(r[key])].append(float(r["orbit_rms_m"]))
    return {fam: _summarize(grouped.get(fam, [])) for fam in FAMILIES}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--batch", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, default=Path("outputs/eval_orbit_mlp.json"))
    args = p.parse_args()

    batch = Phase1Batch.from_dir(args.batch)
    families = [
        str(r.get("path_family") or r.get("trajectory_type") or "") for r in batch.rows
    ]
    train_idx, val_idx = _training_split_indices(
        len(batch), seed=args.seed, holdout_families=("u_turn",), families=families
    )
    model = OrbitMLP.load(args.checkpoint)

    print(f"split train={len(train_idx)} val={len(val_idx)} (seed={args.seed})")
    val_rows = eval_batch(batch, model, val_idx, "val")
    train_rows = eval_batch(batch, model, train_idx, "train")
    proto_rows = eval_family_prototypes(model)

    coverage = Counter(r["assigned_family"] for r in train_rows + val_rows)
    report = {
        "batch_dir": str(Path(args.batch).resolve()),
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "headline_metric": "orbit_rms_m after Procrustes rotation +/- reflection about mic",
        "n_train": len(train_idx),
        "n_val": len(val_idx),
        "seed": args.seed,
        "family_coverage_n1000": dict(coverage),
        "family_coverage_frac": {k: v / 1000.0 for k, v in coverage.items()},
        "product_families": list(PATH_FAMILIES),
        "train": {
            "overall": _summarize([r["orbit_rms_m"] for r in train_rows]),
            "by_family": _by_family(train_rows),
            "cpa_rel_err_median": float(
                np.median([r["cpa_rel_err"] for r in train_rows])
            ),
            "speed_rel_err_median": float(
                np.median([r["speed_rel_err"] for r in train_rows])
            ),
        },
        "val": {
            "overall": _summarize([r["orbit_rms_m"] for r in val_rows]),
            "by_family": _by_family(val_rows),
            "cpa_rel_err_median": float(np.median([r["cpa_rel_err"] for r in val_rows])),
            "speed_rel_err_median": float(
                np.median([r["speed_rel_err"] for r in val_rows])
            ),
        },
        "family_prototypes_tone": {
            "overall": _summarize([r["orbit_rms_m"] for r in proto_rows]),
            "by_family": _by_family(proto_rows, key="family"),
            "n_per_family": 8,
            "domain": "in-package pure tone; MLP trained on vehicle 2D whiteboard STFTs",
        },
        "val_rows": val_rows,
        "proto_rows": proto_rows,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))

    slim = {k: report[k] for k in report if k not in ("val_rows", "proto_rows")}
    print(json.dumps(slim, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
