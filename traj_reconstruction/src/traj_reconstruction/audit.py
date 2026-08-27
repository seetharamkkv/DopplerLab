"""Diversity / integrity audit for simulated Tier-1 batches."""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from traj_reconstruction.dataset import DatasetError, load_phase1_sample
from traj_reconstruction.kinematics import polyline_arclength
from traj_reconstruction.orbit import xy_from_state


def audit_batch(batch_dir: Path | str, *, max_load: int | None = None) -> dict[str, Any]:
    """Check coverage and Phase 1 integrity; return a JSON-serializable report."""
    batch_dir = Path(batch_dir)
    manifest = batch_dir / "dataset.csv"
    if not manifest.is_file():
        raise DatasetError(f"missing dataset.csv under {batch_dir}")

    with manifest.open(newline="") as f:
        rows = list(csv.DictReader(f))

    families = Counter(r.get("path_family", "") for r in rows)
    speeds = np.array([float(r["speed_mps"]) for r in rows], dtype=np.float64)
    cpas = np.array([float(r["cpa_distance_m"]) for r in rows], dtype=np.float64)

    issues: list[str] = []
    checked = 0
    for r in rows[: max_load or len(rows)]:
        sample_dir = batch_dir / "audio_clips" / r["sample_id"]
        try:
            sample = load_phase1_sample(sample_dir)
        except Exception as exc:  # noqa: BLE001 — collect for report
            issues.append(f"{r['sample_id']}: load failed: {exc}")
            continue
        checked += 1
        if sample.canonical_state_frames is None:
            issues.append(f"{r['sample_id']}: missing canonical_state_frames")
        if sample.stft_db.shape[1] != sample.n_frames:
            issues.append(
                f"{r['sample_id']}: STFT T={sample.stft_db.shape[1]} "
                f"!= state T={sample.n_frames}"
            )
        if sample.path_polyline is not None and sample.path_polyline.shape[0] >= 2:
            # State should track polyline: sample a few positions vs nearest poly point.
            xy = xy_from_state(sample.state_frames)
            mid = xy[len(xy) // 2]
            dmin = float(
                np.min(np.linalg.norm(sample.path_polyline - mid[None, :], axis=1))
            )
            # Loose check — mid frame may be near path within tens of meters.
            if dmin > 50.0:
                issues.append(f"{r['sample_id']}: mid-state far from polyline ({dmin:.1f} m)")
            length = float(polyline_arclength(sample.path_polyline)[-1])
            if length < 1.0:
                issues.append(f"{r['sample_id']}: polyline too short ({length:.3f} m)")

    report = {
        "batch_dir": str(batch_dir.resolve()),
        "n_samples": len(rows),
        "family_counts": dict(families),
        "speed_mps": {
            "min": float(speeds.min()) if len(speeds) else None,
            "max": float(speeds.max()) if len(speeds) else None,
            "mean": float(speeds.mean()) if len(speeds) else None,
        },
        "cpa_distance_m": {
            "min": float(cpas.min()) if len(cpas) else None,
            "max": float(cpas.max()) if len(cpas) else None,
            "mean": float(cpas.mean()) if len(cpas) else None,
        },
        "checked_samples": checked,
        "n_issues": len(issues),
        "issues": issues[:50],
        "ok": len(issues) == 0 and len(rows) > 0,
    }
    out = batch_dir / "diversity_audit.json"
    out.write_text(json.dumps(report, indent=2))
    return report
