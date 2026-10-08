"""Position-model ablations. The selection metric is orbit RMS, then range error.

    position_ctc/.venv/bin/python position_ctc/scripts/ablate_position.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "position_ctc" / "src"))

from position_ctc.data import (  # noqa: E402
    CROP_FRAMES,
    augment_features,
    crop_pair,
    discover_clips,
    load_arrays,
    split_by_vehicle,
    standardize_mag,
)
from position_ctc.features import DATA_HOP, DATA_SR  # noqa: E402
from position_ctc.kinematics import kinematic_penalty, project_path, violation_fraction  # noqa: E402
from position_ctc.metrics import aggregate, score_path  # noqa: E402
from position_ctc.model import FrameEncoder  # noqa: E402

DT = DATA_HOP / float(DATA_SR)
MODES = (
    "xy",
    "polar",
    "range_polar",
    "integrate",
    "integrate_gt0",
    "delta",
    "multitask",
    "straight",
    "residual",
)


def _device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _xy(state: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(state[:, [0, 2]], dtype=np.float32)


def _vel(state: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(state[:, [1, 3]], dtype=np.float32)


def _prepare(clips, indices: list[int]) -> list[dict]:
    rows = []
    for count, index in enumerate(indices, start=1):
        clip = clips[index]
        feat, state = load_arrays(clip)
        rows.append(
            {
                "clip_id": clip.clip_id,
                "feat": feat.astype(np.float32),
                "xy": _xy(state),
                "vel": _vel(state),
            }
        )
        if count % 200 == 0:
            print(f"  loaded {count}/{len(indices)}", flush=True)
    return rows


def _upsample(values: torch.Tensor, length: int) -> torch.Tensor:
    return F.interpolate(values.transpose(1, 2), size=length, mode="linear", align_corners=True).transpose(1, 2)


def _integrate(vel: torch.Tensor, origin: torch.Tensor, dt: float) -> torch.Tensor:
    step = vel[:, :-1, :] * dt
    tail = torch.cumsum(step, dim=1)
    zeros = torch.zeros(vel.shape[0], 1, 2, device=vel.device, dtype=vel.dtype)
    return origin[:, None, :] + torch.cat([zeros, tail], dim=1)


def _masked_huber(pred: torch.Tensor, target: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    time = pred.shape[1]
    mask = torch.arange(time, device=pred.device)[None, :] < lengths[:, None]
    err = F.smooth_l1_loss(pred.contiguous(), target.contiguous(), reduction="none", beta=2.0).mean(dim=-1)
    mask_f = mask.float()
    return (err * mask_f).sum() / mask_f.sum().clamp(min=1.0)


class PositionModel(nn.Module):
    def __init__(self, mode: str, mean_path: np.ndarray | None = None):
        super().__init__()
        if mode not in MODES:
            raise ValueError(mode)
        self.mode = mode
        self.encoder = FrameEncoder()
        wide = mode in ("straight", "polar", "range_polar")
        self.head = nn.Linear(256, 3 if wide else 2)
        self.origin = nn.Linear(256, 2)
        if mean_path is not None:
            self.register_buffer("mean_path", torch.tensor(mean_path, dtype=torch.float32))

    def forward(
        self,
        feat: torch.Tensor,
        gt_xy: torch.Tensor | None,
        lengths: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.encoder(feat)
        raw = self.head(hidden)
        length = feat.shape[-1]
        aux = feat.new_zeros(())
        if self.mode == "straight":
            pooled = hidden.mean(dim=1)
            clip = self.head(pooled)
            speed = F.softplus(clip[:, 0]) + 4.0
            cpa = F.softplus(clip[:, 1]) + 1.0
            frac = torch.sigmoid(clip[:, 2])
            t = torch.arange(length, device=feat.device, dtype=feat.dtype) * DT
            t_cpa = frac[:, None] * max(length - 1, 1) * DT
            x = speed[:, None] * (t[None, :] - t_cpa)
            y = cpa[:, None].expand_as(x)
            return torch.stack([x, y], dim=-1), aux
        if self.mode in ("polar", "range_polar"):
            direction = raw[:, :, :2]
            direction = direction / direction.norm(dim=-1, keepdim=True).clamp(min=1e-4)
            radius = F.softplus(raw[:, :, 2:3]) + 0.5
            enc = _upsample(direction * radius, length)
            radius_up = enc.norm(dim=-1, keepdim=True).clamp(min=1e-4)
            xy = enc / radius_up * _upsample(radius, length)
            if gt_xy is not None:
                weight = 2.0 if self.mode == "range_polar" else 0.5
                aux = weight * _masked_huber(xy.norm(dim=-1, keepdim=True), gt_xy.norm(dim=-1, keepdim=True), lengths)
            return xy, aux
        if self.mode in ("integrate", "integrate_gt0"):
            vel = _upsample(raw, length)
            if self.mode == "integrate_gt0":
                if gt_xy is None:
                    raise RuntimeError("integrate_gt0 needs the first true position")
                origin = gt_xy[:, 0, :]
            else:
                origin = self.origin(hidden[:, 0, :])
            xy = _integrate(vel, origin, DT)
            if gt_xy is not None:
                gt_v = (gt_xy[:, 1:, :] - gt_xy[:, :-1, :]) / DT
                aux = 0.15 * _masked_huber(vel[:, :-1, :], gt_v, (lengths - 1).clamp(min=1))
            return xy, aux
        if self.mode == "delta":
            incr = _upsample(raw, length)
            origin = self.origin(hidden[:, 0, :])
            zeros = torch.zeros(incr.shape[0], 1, 2, device=incr.device, dtype=incr.dtype)
            xy = origin[:, None, :] + torch.cat([zeros, torch.cumsum(incr[:, :-1, :], dim=1)], dim=1)
            return xy, aux
        xy = _upsample(raw, length)
        if self.mode == "residual":
            base = _upsample(self.mean_path[None, :, :], length).expand(xy.shape[0], -1, -1)
            xy = base + xy
        if self.mode == "multitask" and gt_xy is not None:
            pred_v = (xy[:, 1:, :] - xy[:, :-1, :]) / DT
            gt_v = (gt_xy[:, 1:, :] - gt_xy[:, :-1, :]) / DT
            aux = 0.5 * _masked_huber(xy.norm(dim=-1, keepdim=True), gt_xy.norm(dim=-1, keepdim=True), lengths)
            aux = aux + 0.15 * _masked_huber(pred_v, gt_v, (lengths - 1).clamp(min=1))
        return xy, aux


def _batch(
    rows: list[dict],
    rng: np.random.Generator | None,
    crop: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    feats, xys, lengths = [], [], []
    for row in rows:
        feat, xy = row["feat"], row["xy"]
        if rng is not None:
            feat = augment_features(feat, rng)
        feat = standardize_mag(feat)
        if rng is not None and feat.shape[1] > crop and rng.random() < 0.25:
            feat, xy = crop_pair(feat, xy, rng, crop)
        feats.append(torch.from_numpy(np.ascontiguousarray(feat)))
        xys.append(torch.from_numpy(np.ascontiguousarray(xy)))
        lengths.append(feat.shape[-1])
    length = max(lengths)
    feat_batch = torch.zeros(len(rows), feats[0].shape[0], length)
    xy_batch = torch.zeros(len(rows), length, 2)
    for i, (feat, xy) in enumerate(zip(feats, xys)):
        feat_batch[i, :, : feat.shape[-1]] = feat
        xy_batch[i, : xy.shape[0]] = xy
    return feat_batch, xy_batch, torch.tensor(lengths, dtype=torch.long)


def _predict_rows(model: PositionModel, rows: list[dict], device: torch.device) -> list[np.ndarray]:
    model.eval()
    preds = []
    with torch.no_grad():
        for row in rows:
            feat, gt, lengths = _batch([row], None, CROP_FRAMES)
            pred, _ = model(feat.to(device), gt.to(device), lengths.to(device))
            n = row["xy"].shape[0]
            preds.append(pred[0, :n].cpu().numpy())
    return preds


def _evaluate(model: PositionModel, rows: list[dict], device: torch.device) -> dict[str, float]:
    scores = [score_path(pred, row["xy"]) for pred, row in zip(_predict_rows(model, rows, device), rows)]
    return aggregate(scores)


def _kinematic_report(preds: list[np.ndarray], rows: list[dict]) -> dict[str, float | dict]:
    raw_violation = [violation_fraction(pred, DT) for pred in preds]
    projected = [project_path(pred, DT) for pred in preds]
    projected_violation = [violation_fraction(path, DT) for path in projected]
    gt_violation = [violation_fraction(row["xy"], DT) for row in rows]
    return {
        "prediction_yaw_violation_fraction_mean": float(np.mean(raw_violation)),
        "projected_yaw_violation_fraction_mean": float(np.mean(projected_violation)),
        "ground_truth_yaw_violation_fraction_mean": float(np.mean(gt_violation)),
        "projected_scores": aggregate([score_path(path, row["xy"]) for path, row in zip(projected, rows)]),
    }


def _mean_path(train: list[dict], grid: int = 200) -> np.ndarray:
    stacked = []
    t_out = np.linspace(0.0, 1.0, grid)
    for row in train:
        xy = row["xy"]
        t_in = np.linspace(0.0, 1.0, len(xy))
        stacked.append(np.stack([np.interp(t_out, t_in, xy[:, 0]), np.interp(t_out, t_in, xy[:, 1])], axis=1))
    return np.mean(stacked, axis=0).astype(np.float32)


def _mean_trajectory_baseline(train: list[dict], test: list[dict]) -> tuple[dict[str, float], np.ndarray]:
    mean = _mean_path(train)
    scores = []
    src = np.linspace(0.0, 1.0, len(mean))
    for row in test:
        n = len(row["xy"])
        dst = np.linspace(0.0, 1.0, n)
        pred = np.stack([np.interp(dst, src, mean[:, 0]), np.interp(dst, src, mean[:, 1])], axis=1)
        scores.append(score_path(pred, row["xy"]))
    return aggregate(scores), mean


def _train_one(
    mode: str,
    train: list[dict],
    val: list[dict],
    device: torch.device,
    mean_path: np.ndarray,
    *,
    epochs: int,
    patience: int,
    batch_size: int,
    lr: float,
    seed: int,
    resume: Path | None = None,
    kinematic_weight: float = 0.0,
) -> tuple[PositionModel, dict]:
    torch.manual_seed(seed)
    model = PositionModel(mode, mean_path if mode == "residual" else None).to(device)
    if resume is not None:
        state = torch.load(resume, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"])
        warm = _evaluate(model, val, device)
        print(
            f"  {mode} resume val orbit={warm['orbit_rms_m_mean']:.2f}m "
            f"range={warm['range_mae_m_mean']:.2f}m",
            flush=True,
        )
        best_rms = warm["orbit_rms_m_mean"]
        if kinematic_weight > 0.0:
            warm_preds = _predict_rows(model, val, device)
            warm_violation = float(np.mean([violation_fraction(pred, DT) for pred in warm_preds]))
            best_rms = warm["orbit_rms_m_mean"] + 8.0 * warm_violation
            print(f"  {mode} resume yaw_violation={warm_violation:.3f}", flush=True)
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    else:
        best_rms = float("inf")
        best_state = None
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    stale = 0
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        rng = np.random.default_rng(seed + epoch)
        order = list(train)
        rng.shuffle(order)
        total, seen = 0.0, 0
        for start in range(0, len(order), batch_size):
            chunk = order[start : start + batch_size]
            if len(chunk) < 2:
                continue
            feat, gt, lengths = _batch(chunk, rng, CROP_FRAMES)
            feat, gt, lengths = feat.to(device), gt.to(device), lengths.to(device)
            pred, aux = model(feat, gt, lengths)
            loss = _masked_huber(pred, gt, lengths) + aux
            if kinematic_weight > 0.0:
                loss = loss + kinematic_weight * kinematic_penalty(pred, DT, lengths)
            if not torch.isfinite(loss):
                continue
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            total += float(loss.item()) * len(chunk)
            seen += len(chunk)
        val_preds = _predict_rows(model, val, device)
        val_scores = aggregate([score_path(pred, row["xy"]) for pred, row in zip(val_preds, val)])
        if kinematic_weight > 0.0:
            violation = float(np.mean([violation_fraction(pred, DT) for pred in val_preds]))
            select = val_scores["orbit_rms_m_mean"] + 8.0 * violation
        else:
            violation = None
            select = val_scores["orbit_rms_m_mean"]
        row = {
            "epoch": epoch,
            "train_loss": total / max(seen, 1),
            "selection_score": select,
            "yaw_violation_fraction": violation,
            **val_scores,
        }
        history.append(row)
        extra = f" yaw_violation={violation:.3f}" if violation is not None else ""
        print(
            f"  {mode} epoch {epoch}/{epochs} loss={row['train_loss']:.3f} "
            f"orbit={val_scores['orbit_rms_m_mean']:.2f}m "
            f"range={val_scores['range_mae_m_mean']:.2f}m "
            f"cpa={val_scores['cpa_abs_m_mean']:.2f}m{extra}",
            flush=True,
        )
        if select + 0.02 < best_rms:
            best_rms = val_scores["orbit_rms_m_mean"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                print(f"  {mode} early stop at epoch {epoch}", flush=True)
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, {"best_val_orbit_rms_m": best_rms, "history": history}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("/Users/seetharamkkv/Desktop/Seetharam/CMU/projects/doppler/datasets/audio_clips"),
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--modes", nargs="+", default=list(MODES))
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument(
        "--kinematic-weight",
        type=float,
        default=0.0,
        help="Weight on the feasible yaw-rate penalty. 0 leaves the original loss unchanged.",
    )
    args = parser.parse_args()

    out_dir = ROOT / "position_ctc" / "outputs" / "ablate"
    out_dir.mkdir(parents=True, exist_ok=True)
    clips = discover_clips(args.data)
    splits = split_by_vehicle(clips, seed=args.seed)
    print(
        f"clips={len(clips)} train={len(splits['train'])} val={len(splits['val'])} test={len(splits['test'])}",
        flush=True,
    )
    print("loading audio and canonical paths", flush=True)
    rows = {name: _prepare(clips, idxs) for name, idxs in splits.items()}
    device = _device()
    print(f"device {device}", flush=True)
    baseline, mean_path = _mean_trajectory_baseline(rows["train"], rows["val"])
    test_baseline, _ = _mean_trajectory_baseline(rows["train"], rows["test"])
    print(
        f"mean-path baseline val orbit={baseline['orbit_rms_m_mean']:.2f}m "
        f"range={baseline['range_mae_m_mean']:.2f}m",
        flush=True,
    )
    print(
        f"mean-path baseline test orbit={test_baseline['orbit_rms_m_mean']:.2f}m "
        f"range={test_baseline['range_mae_m_mean']:.2f}m "
        f"cpa={test_baseline['cpa_abs_m_mean']:.2f}m",
        flush=True,
    )
    results = {"baseline_mean_path_val": baseline, "modes": {}}
    for mode in args.modes:
        print(f"ablation {mode}", flush=True)
        model, train_info = _train_one(
            mode,
            rows["train"],
            rows["val"],
            device,
            mean_path,
            epochs=args.epochs,
            patience=args.patience,
            batch_size=args.batch_size,
            lr=args.lr,
            seed=args.seed,
            resume=args.resume,
            kinematic_weight=args.kinematic_weight,
        )
        preds = _predict_rows(model, rows["test"], device)
        test_scores = aggregate([score_path(pred, row["xy"]) for pred, row in zip(preds, rows["test"])])
        val_scores = _evaluate(model, rows["val"], device)
        kin = _kinematic_report(preds, rows["test"]) if args.kinematic_weight > 0.0 else None
        if args.kinematic_weight > 0.0:
            ckpt_name = f"{mode}_kin.pt"
            report_name = "kinematic.json"
        elif args.resume is not None:
            ckpt_name = f"{mode}_ft.pt"
            report_name = "finetune.json"
        else:
            ckpt_name = f"{mode}.pt"
            report_name = "comparison.json"
        torch.save(
            {
                "model": model.state_dict(),
                "mode": mode,
                "kinematic_weight": args.kinematic_weight,
            },
            out_dir / ckpt_name,
        )
        results["modes"][mode] = {
            "train": train_info,
            "val": val_scores,
            "test": test_scores,
            "kinematic": kin,
        }
        results["baseline_mean_path_test"] = test_baseline
        print(
            f"  {mode} TEST orbit={test_scores['orbit_rms_m_mean']:.2f}m "
            f"median={test_scores['orbit_rms_m_median']:.2f}m "
            f"range={test_scores['range_mae_m_mean']:.2f}m "
            f"cpa={test_scores['cpa_abs_m_mean']:.2f}m",
            flush=True,
        )
        if kin is not None:
            projected = kin["projected_scores"]
            print(
                f"  {mode} yaw violation pred={kin['prediction_yaw_violation_fraction_mean']:.3f} "
                f"projected={kin['projected_yaw_violation_fraction_mean']:.3f} "
                f"ground_truth={kin['ground_truth_yaw_violation_fraction_mean']:.3f}",
                flush=True,
            )
            print(
                f"  {mode} PROJECTED orbit={projected['orbit_rms_m_mean']:.2f}m "
                f"range={projected['range_mae_m_mean']:.2f}m "
                f"cpa={projected['cpa_abs_m_mean']:.2f}m",
                flush=True,
            )
        (out_dir / report_name).write_text(json.dumps(results, indent=2), encoding="utf-8")
    ranking = sorted(
        results["modes"].items(),
        key=lambda item: (
            item[1]["test"]["orbit_rms_m_mean"],
            item[1]["test"]["range_mae_m_mean"],
        ),
    )
    print("ranking by test orbit RMS then range MAE", flush=True)
    for name, payload in ranking:
        test = payload["test"]
        print(
            f"  {name}: orbit {test['orbit_rms_m_mean']:.2f}m, "
            f"range {test['range_mae_m_mean']:.2f}m, cpa {test['cpa_abs_m_mean']:.2f}m",
            flush=True,
        )


if __name__ == "__main__":
    main()
