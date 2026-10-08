"""Separate a constant +14, 0 velocity into data, labels, probabilities, or Viterbi.

Then try to memorize a few turning training clips with framewise cross-entropy
only. This is a debugging run, not the training configuration.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "position_ctc" / "src"))

from position_ctc.bins import CENTERS_MPS, velocity_bin_index  # noqa: E402
from position_ctc.data import discover_clips, load_arrays, split_by_vehicle, standardize_mag  # noqa: E402
from position_ctc.features import DATA_HOP, DATA_SR  # noqa: E402
from position_ctc.motion_model import MotionNet  # noqa: E402
from position_ctc.transitions import DEFAULT_JUMP_MPS, a_max_for_jump, transition_matrix  # noqa: E402
from position_ctc.viterbi import viterbi  # noqa: E402

DT = DATA_HOP / float(DATA_SR)
OUT = ROOT / "position_ctc" / "outputs" / "motion" / "diagnose"
DATA = Path("/Users/seetharamkkv/Desktop/Seetharam/CMU/projects/doppler/datasets/audio_clips")


def _bin_hist(velocity: np.ndarray) -> np.ndarray:
    counts = np.bincount(velocity_bin_index(velocity), minlength=CENTERS_MPS.size).astype(np.float64)
    return counts / max(counts.sum(), 1.0)


def _probs(model: MotionNet, feat: np.ndarray, device: torch.device, origin: np.ndarray):
    model.eval()
    frames = int(feat.shape[-1])
    batch = torch.from_numpy(np.ascontiguousarray(feat)).unsqueeze(0).to(device)
    lengths = torch.tensor([frames], dtype=torch.long, device=device)
    start = torch.tensor(origin, dtype=torch.float32, device=device).view(1, 2)
    with torch.no_grad():
        out = model(batch, lengths, start)
        hidden = model.conv(batch).transpose(1, 2)
    px = out["vel_x"][0, :frames].softmax(-1).cpu().numpy()
    py = out["vel_y"][0, :frames].softmax(-1).cpu().numpy()
    return px, py, hidden[0].detach().cpu().numpy(), {k: tuple(v.shape) for k, v in out.items()}


def _heatmap(prob: np.ndarray, truth: np.ndarray, title: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.2, 3.6))
    show = prob.T
    ax.imshow(show, origin="lower", aspect="auto", cmap="magma", vmin=0.0, vmax=max(0.2, float(show.max())))
    time = np.arange(prob.shape[0])
    ax.plot(time, velocity_bin_index(truth), color="cyan", lw=1.0, label="label bin")
    ax.plot(time, np.argmax(prob, axis=1), color="white", lw=0.8, ls="--", label="argmax")
    ax.set_ylabel("bin  (−30 … +30 m/s)")
    ax.set_xlabel("frame")
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=8, frameon=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    clips = discover_clips(DATA)
    splits = split_by_vehicle(clips, seed=0)
    train_idx = splits["train"]
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    rows = []
    vx_all = []
    vy_all = []
    for i in train_idx:
        _, state = load_arrays(clips[i])
        raw = np.load(clips[i].sample_dir / "metadata" / "canonical_state_frames.npy")
        n = min(raw.shape[0], state.shape[0])
        # The loader trims stationary pads, so the training target is a slice, not a copy of one speed.
        match = np.where(np.isclose(raw[:, 0], state[0, 0]) & np.isclose(raw[:, 2], state[0, 2]))[0]
        aligned = int(match[0]) if match.size else -1
        same = False
        if aligned >= 0 and aligned + state.shape[0] <= raw.shape[0]:
            same = bool(np.allclose(raw[aligned : aligned + state.shape[0]], state, atol=1e-5))
        rows.append(
            {
                "id": clips[i].clip_id,
                "vehicle": clips[i].vehicle,
                "index": i,
                "T": int(state.shape[0]),
                "vx_std": float(state[:, 1].std()),
                "vy_std": float(state[:, 3].std()),
                "vx_span": float(state[:, 1].max() - state[:, 1].min()),
                "vy_span": float(state[:, 3].max() - state[:, 3].min()),
                "label_is_slice": same,
                "unique_vx": int(np.unique(np.round(state[:, 1], 3)).size),
                "unique_vy": int(np.unique(np.round(state[:, 3], 3)).size),
            }
        )
        vx_all.append(state[:, 1])
        vy_all.append(state[:, 3])
    vx_all = np.concatenate(vx_all)
    vy_all = np.concatenate(vy_all)
    fx, fy = _bin_hist(vx_all), _bin_hist(vy_all)
    order_x = np.argsort(fx)[::-1][:5]
    order_y = np.argsort(fy)[::-1][:5]
    print("train clips", len(rows), "frames", int(vx_all.size))
    print(
        "clips with vy span > 4 m/s",
        sum(r["vy_span"] > 4 for r in rows),
        "vx span > 4",
        sum(r["vx_span"] > 4 for r in rows),
    )
    print("median within-clip std vx", float(np.median([r["vx_std"] for r in rows])), "vy", float(np.median([r["vy_std"] for r in rows])))
    print("label slice matches npy", sum(r["label_is_slice"] for r in rows), "/", len(rows))
    print("clips with a single rounded vx", sum(r["unique_vx"] == 1 for r in rows), "single vy", sum(r["unique_vy"] == 1 for r in rows))
    print("top vx bins", [(float(CENTERS_MPS[k]), round(float(fx[k]), 4)) for k in order_x])
    print("top vy bins", [(float(CENTERS_MPS[k]), round(float(fy[k]), 4)) for k in order_y])

    turning = sorted(rows, key=lambda r: r["vy_span"], reverse=True)
    turning = [r for r in turning if r["vx_span"] > 2.0][:4]
    print("memorization clips", [(r["id"], round(r["vx_span"], 2), round(r["vy_span"], 2)) for r in turning])

    fig, axes = plt.subplots(4, 1, figsize=(8.5, 8.0), sharex=False)
    for ax, spec in zip(axes, turning):
        _, state = load_arrays(clips[spec["index"]])
        t = np.arange(state.shape[0]) * DT
        ax.plot(t, state[:, 1], color="black", lw=1.1, label="vx")
        ax.plot(t, state[:, 3], color="tab:orange", lw=1.1, label="vy")
        ax.set_ylabel("m/s")
        ax.set_title(f"{spec['id']}  {spec['vehicle']}  vx span {spec['vx_span']:.1f}  vy span {spec['vy_span']:.1f}")
        ax.grid(True, alpha=0.3)
    axes[0].legend(loc="upper right", fontsize=8)
    axes[-1].set_xlabel("t (s)")
    fig.tight_layout()
    fig.savefig(OUT / "train_velocity_examples.png", dpi=140)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.4), sharey=True)
    for ax, freq, name in ((axes[0], fx, "vx"), (axes[1], fy, "vy")):
        ax.bar(CENTERS_MPS, freq, width=1.6, color="0.25")
        ax.set_xlabel(f"{name} bin center (m/s)")
        ax.set_title(name)
    axes[0].set_ylabel("fraction of training frames")
    fig.tight_layout()
    fig.savefig(OUT / "bin_frequencies.png", dpi=140)
    plt.close(fig)

    psi = transition_matrix(DT, a_max_for_jump(DEFAULT_JUMP_MPS, DT), lam=0.25)
    held = next(c for c in clips if c.clip_id == "sample_0000172")
    targets = [("heldout", held)] + [("train", clips[spec["index"]]) for spec in turning[:1]]
    for tag, ckpt_name in (("epoch38", "motion_epoch38.pt"), ("epoch27", "motion.pt")):
        ckpt_path = ROOT / "position_ctc" / "outputs" / "motion" / ckpt_name
        if not ckpt_path.is_file():
            print("missing", ckpt_path)
            continue
        model = MotionNet().to(device)
        saved = torch.load(ckpt_path, map_location=device, weights_only=False)
        incompat = model.load_state_dict(saved["model"], strict=False)
        print(tag, "epoch", saved.get("epoch"), "missing", list(incompat.missing_keys), "unexpected", list(incompat.unexpected_keys))
        for role, clip in targets:
            feat, state = load_arrays(clip)
            feat = standardize_mag(feat)
            n = min(feat.shape[1], state.shape[0])
            px, py, hidden, shapes = _probs(model, feat[:, :n], device, state[0, [0, 2]])
            _heatmap(px, state[:n, 1], f"{tag} {clip.clip_id} vx  before Viterbi", OUT / f"heatmap_{tag}_{role}_vx.png")
            _heatmap(py, state[:n, 3], f"{tag} {clip.clip_id} vy  before Viterbi", OUT / f"heatmap_{tag}_{role}_vy.png")
            ax_x = np.argmax(px, axis=1)
            ax_y = np.argmax(py, axis=1)
            vit_x = viterbi(np.log(np.clip(px, 1e-12, 1.0)), psi)
            vit_y = viterbi(np.log(np.clip(py, 1e-12, 1.0)), psi)
            gt_x = velocity_bin_index(state[:n, 1])
            gt_y = velocity_bin_index(state[:n, 3])
            mass_x = float(px[:, ax_x[0]].mean()) if np.unique(ax_x).size == 1 else float(px[np.arange(n), ax_x].mean())
            print(
                f"{tag} {role} {clip.clip_id} shapes {shapes['vel_x']} enc {hidden.shape} "
                f"enc temporal std {hidden.std(axis=0).mean():.4f} "
                f"argmax=viterbi x {float(np.mean(ax_x == vit_x)):.3f} y {float(np.mean(ax_y == vit_y)):.3f} "
                f"argmax acc x {float(np.mean(ax_x == gt_x)):.3f} y {float(np.mean(ax_y == gt_y)):.3f} "
                f"unique argmax x {np.unique(CENTERS_MPS[ax_x]).round(1).tolist()} "
                f"y {np.unique(CENTERS_MPS[ax_y]).round(1).tolist()} "
                f"mean maxprob x {px.max(1).mean():.3f} y {py.max(1).mean():.3f} "
                f"mode-mass x {mass_x:.3f}"
            )

    # Memorize four turning clips with framewise CE only.
    model = MotionNet().to(device)
    model.drop.p = 0.0
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.0)
    pack = []
    for spec in turning:
        feat, state = load_arrays(clips[spec["index"]])
        feat = standardize_mag(feat)
        n = min(feat.shape[1], state.shape[0])
        pack.append(
            (
                spec,
                torch.from_numpy(np.ascontiguousarray(feat[:, :n])).unsqueeze(0).to(device),
                torch.from_numpy(np.ascontiguousarray(state[:n, 1])),
                torch.from_numpy(np.ascontiguousarray(state[:n, 3])),
                torch.tensor([[float(state[0, 0]), float(state[0, 2])]], dtype=torch.float32, device=device),
            )
        )
    length = max(item[1].shape[-1] for item in pack)
    feat_b = torch.zeros(len(pack), pack[0][1].shape[1], length, device=device)
    vx_b = torch.zeros(len(pack), length, device=device)
    vy_b = torch.zeros(len(pack), length, device=device)
    origin_b = torch.zeros(len(pack), 2, device=device)
    lengths = torch.zeros(len(pack), dtype=torch.long, device=device)
    for i, (_, feat, vx, vy, origin) in enumerate(pack):
        n = feat.shape[-1]
        feat_b[i, :, :n] = feat[0]
        vx_b[i, :n] = vx.to(device)
        vy_b[i, :n] = vy.to(device)
        origin_b[i] = origin[0]
        lengths[i] = n
    tx = torch.from_numpy(velocity_bin_index(vx_b.cpu().numpy())).to(device)
    ty = torch.from_numpy(velocity_bin_index(vy_b.cpu().numpy())).to(device)
    print("overfit device", device, "clips", len(pack), "T", int(length), "dropout", model.drop.p, "wd 0 losses CE only")
    for epoch in range(1, 81):
        model.train()
        opt.zero_grad(set_to_none=True)
        out = model(feat_b, lengths, origin_b)
        valid = torch.arange(length, device=device)[None, :] < lengths[:, None]
        ce_x = F.cross_entropy(out["vel_x"].reshape(-1, out["vel_x"].shape[-1]), tx.reshape(-1), reduction="none")
        ce_y = F.cross_entropy(out["vel_y"].reshape(-1, out["vel_y"].shape[-1]), ty.reshape(-1), reduction="none")
        ce = ((ce_x + ce_y).view(len(pack), length) * valid).sum() / valid.sum().clamp(min=1) / 2
        ce.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        if epoch in (1, 5, 10, 20, 40, 80) or epoch % 20 == 0:
            with torch.no_grad():
                pred_x = out["vel_x"].argmax(-1)
                pred_y = out["vel_y"].argmax(-1)
                acc_x = float(((pred_x == tx) & valid).sum() / valid.sum())
                acc_y = float(((pred_y == ty) & valid).sum() / valid.sum())
            print(f"overfit epoch {epoch:02d}  ce {float(ce):.4f}  argmax acc x {acc_x:.3f} y {acc_y:.3f}")

    model.eval()
    fig, axes = plt.subplots(len(pack), 2, figsize=(10.0, 2.3 * len(pack)), sharex=False)
    for i, (spec, feat, vx, vy, origin) in enumerate(pack):
        n = int(feat.shape[-1])
        with torch.no_grad():
            out = model(feat, torch.tensor([n], device=device), origin)
            px = out["vel_x"][0, :n].softmax(-1).cpu().numpy()
            py = out["vel_y"][0, :n].softmax(-1).cpu().numpy()
        t = np.arange(n) * DT
        axes[i, 0].plot(t, vx.cpu().numpy(), color="black", lw=1.2, label="label")
        axes[i, 0].plot(t, CENTERS_MPS[px.argmax(1)], color="tab:blue", lw=1.0, label="argmax")
        axes[i, 1].plot(t, vy.cpu().numpy(), color="black", lw=1.2, label="label")
        axes[i, 1].plot(t, CENTERS_MPS[py.argmax(1)], color="tab:blue", lw=1.0, label="argmax")
        axes[i, 0].set_ylabel(spec["id"])
        axes[i, 0].grid(True, alpha=0.3)
        axes[i, 1].grid(True, alpha=0.3)
        if i == 0:
            _heatmap(px, vx.cpu().numpy(), f"overfit {spec['id']} vx", OUT / "overfit_heatmap_vx.png")
            _heatmap(py, vy.cpu().numpy(), f"overfit {spec['id']} vy", OUT / "overfit_heatmap_vy.png")
    axes[0, 0].set_title("vx before Viterbi")
    axes[0, 1].set_title("vy before Viterbi")
    axes[0, 0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "overfit_argmax.png", dpi=140)
    plt.close(fig)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
