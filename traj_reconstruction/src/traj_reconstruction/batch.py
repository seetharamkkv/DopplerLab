"""Tier-1 freehand batch builder (simulated Phase 1 packages)."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from traj_reconstruction.contract import TIER1
from traj_reconstruction.export_phase1 import export_phase1_sample
from traj_reconstruction.path_families import generate_path


MANIFEST_FIELDS = (
    "sample_id",
    "path_type",
    "path_family",
    "tier",
    "speed_mps",
    "cpa_time_sec",
    "cpa_distance_m",
    "n_frames",
    "wav",
    "seed",
    "notes",
)


@dataclass(frozen=True)
class BatchPlan:
    sample_id: str
    family: str
    speed_mps: float
    cpa_distance_m: float
    seed: int


def plan_tier1_batch(
    *,
    n_per_family: dict[str, int] | None = None,
    speed_range: tuple[float, float] = (5.0, 40.0),
    cpa_range: tuple[float, float] = (5.0, 80.0),
    seed: int = 42,
    families: Sequence[str] | None = None,
) -> list[BatchPlan]:
    """Deterministic stratified plan over path families / speed / CPA distance."""
    fams = tuple(families) if families is not None else (
        PATH_FAMILY_STRAIGHT,
        PATH_FAMILY_ARC,
        "s_curve",
        "u_turn",
        "multi_cpa",
    )
    counts = n_per_family or {f: 20 for f in fams}
    rng = np.random.default_rng(int(seed))
    plans: list[BatchPlan] = []
    idx = 0
    for family in fams:
        n = int(counts.get(family, 0))
        for _ in range(n):
            speed = float(rng.uniform(*speed_range))
            cpa = float(rng.uniform(*cpa_range))
            sample_seed = int(rng.integers(0, 2**31 - 1))
            plans.append(
                BatchPlan(
                    sample_id=f"sample_{idx:07d}",
                    family=family,
                    speed_mps=speed,
                    cpa_distance_m=cpa,
                    seed=sample_seed,
                )
            )
            idx += 1
    return plans


def build_tier1_batch(
    output_dir: Path | str,
    *,
    n_per_family: dict[str, int] | None = None,
    speed_range: tuple[float, float] = (5.0, 40.0),
    cpa_range: tuple[float, float] = (5.0, 80.0),
    seed: int = 42,
    families: Sequence[str] | None = None,
    resume: bool = True,
    f0_hz: float = 500.0,
) -> Path:
    """Generate a simulated Tier-1 batch under ``output_dir``."""
    output_dir = Path(output_dir)
    clips_dir = output_dir / "audio_clips"
    clips_dir.mkdir(parents=True, exist_ok=True)

    plans = plan_tier1_batch(
        n_per_family=n_per_family,
        speed_range=speed_range,
        cpa_range=cpa_range,
        seed=seed,
        families=families,
    )
    rows: list[dict] = []
    for plan in plans:
        sample_dir = clips_dir / plan.sample_id
        schema_path = sample_dir / "metadata" / "phase1_schema.json"
        stft_path = sample_dir / "spectrograms" / "stft.npy"
        if resume and schema_path.is_file() and stft_path.is_file():
            # Reload row from schema + derived files when possible.
            schema = json.loads(schema_path.read_text())
            cpa_d = float(np.load(sample_dir / "metadata" / "cpa_distance_m.npy")[0])
            cpa_t = float(np.load(sample_dir / "metadata" / "cpa_time.npy")[0])
            state = np.load(sample_dir / "metadata" / "state_frames.npy")
            wavs = sorted(sample_dir.glob("*.wav"))
            rows.append(
                {
                    "sample_id": plan.sample_id,
                    "path_type": schema.get("path_type", ""),
                    "path_family": schema.get("path_family", plan.family),
                    "tier": schema.get("tier", TIER1),
                    "speed_mps": float(schema.get("kinematics", {}).get("speed_mps", plan.speed_mps)),
                    "cpa_time_sec": cpa_t,
                    "cpa_distance_m": cpa_d,
                    "n_frames": int(state.shape[0]),
                    "wav": wavs[0].name if wavs else "",
                    "seed": int(schema.get("seed", plan.seed)),
                    "notes": schema.get("kinematics", {}).get("notes", ""),
                }
            )
            continue

        rng = np.random.default_rng(plan.seed)
        spec = generate_path(
            plan.family,
            speed_mps=plan.speed_mps,
            cpa_distance_m=plan.cpa_distance_m,
            rng=rng,
        )
        row = export_phase1_sample(
            sample_dir,
            spec,
            sample_id=plan.sample_id,
            seed=plan.seed,
            tier=TIER1,
            f0_hz=f0_hz,
        )
        rows.append(row)

    manifest = output_dir / "dataset.csv"
    with manifest.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(MANIFEST_FIELDS))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in MANIFEST_FIELDS})

    meta = {
        "batch_name": output_dir.name,
        "tier": TIER1,
        "data_scope": "simulated_dopplersim_only",
        "seed": int(seed),
        "n_samples": len(rows),
        "families": sorted({r["path_family"] for r in rows}),
        "speed_range_mps": list(speed_range),
        "cpa_range_m": list(cpa_range),
        "acoustics": "pure_tone_retarded_time",
        "note": "Generated by traj_reconstruction (sim only). Compatible with Phase 1 layout.",
    }
    (output_dir / "batch_meta.json").write_text(json.dumps(meta, indent=2))
    (output_dir / "sampler_state.json").write_text(
        json.dumps({"seed": seed, "n_planned": len(plans)}, indent=2)
    )
    return output_dir
