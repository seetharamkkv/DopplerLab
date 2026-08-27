# Trajectory reconstruction (simulated)

Recover an **observer-centered trajectory orbit** from monaural vehicle pass-by audio: the path the vehicle traced, as a **full rotational family about the microphone** (rotate / mirror freely; absolute world heading is not identified).

> **Simulated data only (current scope)**  
> This track trains and evaluates **only on synthetic Phase 1 exports** (Tier-1 freehand batches from this package, and/or DopplerSim `2D_3D` path2d/path3d packages). Real roadside recordings (IDMT, VS13, field mics) are **out of scope until a later phase**. Do not point this package at real audio expecting calibrated orbits.

**End product (across phases):** `audio → model → rotatable trajectory image about the observer`, with no metadata at inference.

Phase checklists: [`../ref_docs/checklists/`](../ref_docs/checklists/). Design plan: [`../ref_docs/trajectory_orbit_plan.md`](../ref_docs/trajectory_orbit_plan.md).

---

## What this folder is now (Phase 0 + Phase 1)

| Module | Role |
|--------|------|
| `contract` | Audio-only inference rules; forbidden GT paths |
| `orbit` | Rotation ± reflection alignment + shape RMS |
| `dataset` / `Phase1Batch` | Read Phase 1 sample folders / batches (sim only) |
| `path_families` | Programmatic straight / arc / S / U-turn / multi-CPA polylines |
| `synthesize` + `export_phase1` | Tier-1 pure-tone + retarded-time → Phase 1 package |
| `batch` / `splits` / `audit` | Build, split, and audit Tier-1 freehand batches |
| `frontend` | Audio-only ridge tracker + amplitude envelope (**no metadata**) |
| `parametric` | Phase 3a straight/arc physics fit → orbit polyline |
| `flexible` | Phase 3b freeform physics fit + STFT→path OrbitMLP |

Not yet: tiered robustness (4), rotatable UI (5).

---

## Install

```bash
cd traj_reconstruction
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
python -m pip install -e .
pytest
```

---

## Build a simulated Tier-1 freehand batch

```bash
source .venv/bin/activate
# Smoke (fast)
python scripts/build_tier1_batch.py --out data/tier1_smoke \
  --n-straight 4 --n-arc 4 --n-s-curve 2 --n-u-turn 1 --n-multi-cpa 1

# Larger Tier-1 set (scale up as needed toward 500+ per family)
python scripts/build_tier1_batch.py --out data/tier1_v1 \
  --n-straight 100 --n-arc 100 --n-s-curve 50 --n-u-turn 30 --n-multi-cpa 20
```

Each run writes:

```
data/<batch>/
  dataset.csv
  batch_meta.json
  splits.json / split_{train,val,test}.txt
  diversity_audit.json
  audio_clips/sample_XXXXXXX/
    *.wav
    spectrograms/stft.npy          # inference-allowed (A)
    metadata/
      state_frames.npy             # train/eval GT only
      canonical_state_frames.npy   # orbit-gauge GT
      path_polyline.npy
      phase1_schema.json
      …
```

Acoustics for these batches: **pure tone + 1/r + retarded time** (`tier1`). Compatible Phase 1 layout with DopplerSim freehand exports; you can also load DopplerSim `renders/<id>/phase1/` packages with the same reader.

```python
from traj_reconstruction import Phase1Batch, extract_ridges, orbit_align, xy_from_state

batch = Phase1Batch.from_dir("data/tier1_smoke")
sample = batch.load(0)
# Audio-only front end — do not pass metadata into extract_ridges
feats = extract_ridges(stft_db=sample.stft_db)  # or wav_path=sample.wav_path
print(feats.quality_mean, feats.f_obs_hz[:5])
```

Ridge overlay (sim sample):

```bash
python scripts/extract_ridges.py --phase1 data/tier1_smoke/audio_clips/sample_0000000 \
  --out outputs/ridge_overlay.png

# Parametric orbit fit (straight/arc baseline)
python scripts/fit_orbit.py --phase1 data/tier1_smoke/audio_clips/sample_0000000 \
  --out outputs/fit_orbit.json --plot outputs/fit_orbit.png

# Flexible freehand fit (Fourier-lateral physics)
python scripts/infer_orbit.py --phase1 data/tier1_smoke/audio_clips/sample_0000002 \
  --mode flexible --out outputs/infer_flexible.json

# Train / run OrbitMLP (sim batches only)
python scripts/train_orbit_mlp.py --batch data/tier1_smoke --epochs 30 \
  --checkpoint checkpoints/orbit_mlp.npz
python scripts/infer_orbit.py --phase1 data/tier1_smoke/audio_clips/sample_0000000 \
  --mode mlp --checkpoint checkpoints/orbit_mlp.npz

# Phase 4 tiered validation dossier (sim only)
python scripts/run_tiered_validation.py --out outputs/tiered_validation

# Phase 5 rotatable orbit product (open the HTML)
python scripts/demo_orbit_product.py \\
  --phase1 data/tier1_smoke/audio_clips/sample_0000002 \\
  --out outputs/orbit_product
# → outputs/orbit_product/orbit_product.html

# Phase 6 leakage audit + WAV-only smoke + release pack
python scripts/audit_leakage.py --batch data/tier1_smoke --out outputs/leakage --wav-smoke
python scripts/package_release.py --out outputs/release \\
  --batch data/tier1_smoke --product-dir outputs/orbit_product
```

Compliance note: [`docs/NO_LEAKAGE.md`](docs/NO_LEAKAGE.md).

Orbit metric demo (no batch required):

```bash
python scripts/demo_orbit_metric.py
python scripts/demo_orbit_metric.py --phase1 data/tier1_smoke/audio_clips/sample_0000000
```

---

## Inference contract (non-negotiable)

**Allowed at inference:** WAV and/or `spectrograms/stft.npy`.

**Forbidden at inference:** any `metadata/*`, labels, simulation JSON, vehicle/site IDs, path polyline, CPA/speed sidecars.

Scoring uses **orbit alignment** (Procrustes rotation ± reflection about the mic), never raw world-frame XY error as the headline metric.

---

## Layout

```
traj_reconstruction/
├── README.md                 ← you are here (SIMULATED ONLY)
├── pyproject.toml
├── src/traj_reconstruction/
│   ├── contract.py
│   ├── orbit.py
│   ├── dataset.py
│   ├── path_families.py
│   ├── kinematics.py
│   ├── synthesize.py
│   ├── export_phase1.py
│   ├── batch.py
│   ├── splits.py
│   └── audit.py
├── scripts/
│   ├── demo_orbit_metric.py
│   └── build_tier1_batch.py
├── data/                     # local batches (gitignored)
└── tests/
```

---

## Status

| Phase | Status |
|-------|--------|
| 0 Contract + orbit metrics | Done |
| 1 Freehand dataset factory | Done (Tier-1 tone batches in-package) |
| 2 Signal front-end | Done (audio-only ridge + envelope) |
| 3a Parametric fit | Done (straight/arc physics baseline) |
| 3b Flexible orbit model | Done (freeform fit + OrbitMLP) |
| 4 Tiered sim validation | Done (Tier 1–5 scorecard CLI) |
| 5 Rotatable image product | Done (HTML orbit viewer + predict API) |
| 6 Leakage + real transfer | Done (audit + release pack; real data deferred) |
