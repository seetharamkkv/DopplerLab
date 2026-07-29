# Leakage-aware pass-by data loading

Reusable discover → split → mel-stats → verify pipeline for DopplerLab models
(SE-ResNet, physics speed estimators, length estimation, and anything else that
reads `<Vehicle>/<Vehicle>_<speed>.wav` trees).

**Dataset note:** the folder layout and labeling convention were based on the
public **VS13** roadside pass-by release. Point `--real_root` at your local
copy (in this repo that is often `speed_estimation/vs13`, or symlink / copy it
to `speed_estimation/passby`).

**Design goal:** train never sees held-out *scenes*, and normalization stats are
never fit on val/test. Soft near-duplicate and unknown-vehicle concerns are
opt-in (spread / LOVO), not the default.

---

## Method (what “leakage-aware” means)

### Scene key

Every clip is identified by a **scene key**:

```text
condition_key = vehicle|integer_speed_kmh
```

Example: `Mazda3/Mazda3_72.wav` → `Mazda3|72`.

If a scene is held out for testing, **no** training clip (real or synthetic twin
with the same key) may use that scene. That blocks the common failure mode where
a model trains on one recording of a car at 72 km/h and is evaluated on another
file of the *same* car at the *same* speed (or a synth twin of that scene).

### Outer split → inner fit / val → stats → eval

```text
data root
   │
   ▼
 discover clips (uids, speeds, optional CPA)
   │
   ▼
 outer split  ──►  Train  /  Test     (protocol-dependent)
   │
   ▼
 carve early-stopping Val from Train scenes only
   │                    (never from Test)
   ▼
 fit mel mean/std on Fit paths only
   │                    (never Val, never Test)
   ▼
 assert_no_leakage(...)  ──► fail loud if anything overlaps
   │
   ▼
 train on Fit, early-stop on Val, report metrics on Test only
```

### Protocols (pick the claim you need)

| Protocol | Held out | Same car in train+test? | Fair claim |
|----------|----------|-------------------------|------------|
| **`scene`** (default) | `vehicle\|speed` scenes | Yes, different speeds | Held-out *scenes* |
| **`scene` + `--min-speed-gap 2`** | scenes + no ±1 km/h twins | Yes, speeds spread apart | Stricter scene holdout |
| **`lovo`** | Entire vehicle | **No** | Unknown car |
| **`speed_stratified`** | Random clips (speed bins) | Yes | Mixed-fleet clip holdout |

**Default does not run LOVO or spreading.** Use flags or the dedicated scripts
when you need those guarantees. Do **not** quote a scene-split MAE as LOVO.

### How leakage is prevented

| Risk | Mitigation |
|------|------------|
| Same clip in train and test | UID sets must be disjoint; verifier fails otherwise |
| Same scene (car@speed) in train and test | Scene-key blocking on real **and** synth partitions |
| Synth twin of a test scene in training | Forced into `synth_test`; banned from `synth_train` |
| Early-stopping on outer test | Inner val is carved only from **train** scene keys |
| Mel / feature stats peeking at test | Stats fit on **fit_uids only**; including val/test UIDs fails the verifier |
| Full-set K-fold ensemble called “held-out” | This package never builds that; eval lists must be holdout-only |
| Calling scene results “unknown vehicle” | Soft warning when cars overlap; use LOVO for that claim |
| ±1 km/h near-twins (same car) | Soft warning by default; hard ban with `--min-speed-gap 2` |

### What the verifier hard-fails on

1. Train ∩ test clip UIDs  
2. Train ∩ test **scene keys** (including synth)  
3. Fit / val intersecting outer test  
4. Mel-stats UID list intersecting val or test  
5. LOVO fold where the held-out vehicle appears in train  
6. Eval UIDs whose scene key is in train  
7. If `meta.min_speed_gap > 1`, any same-vehicle train/test pair closer than that gap  

---

## Layout

```text
data_loading/
├── README.md                 ← this file
├── pyproject.toml            ← installable package `passby_data`
├── src/passby_data/          ← library
│   ├── catalog.py            # discover / uid / condition_key
│   ├── splits.py             # scene / LOVO / speed-stratified / spread
│   ├── leakage.py            # assert_no_leakage / leakage_report
│   ├── mel_stats.py          # train-only mel mean/std
│   ├── loader.py             # fit/val/test bundles for trainers
│   └── cli.py                # CLI entry points
├── scripts/
│   ├── make_split.py                 # default scene (+ optional flags)
│   ├── make_lovo_split.py            # opt-in unknown-vehicle folds
│   ├── make_spread_scene_split.py    # opt-in +/-1 twin ban
│   ├── verify_no_leakage.py
│   └── compute_mel_stats.py
├── splits/                   # generated JSON artifacts
└── tests/                    # pytest leakage tests
```

---

## Install

```powershell
cd D:\Antigravity\DopplerLab\data_loading
D:\Antigravity\venv\Scripts\python.exe -m pip install -e .
```

Or without install: put `data_loading/src` on `PYTHONPATH`.

**Data:** pass-by wavs under `<root>/<Vehicle>/*.wav`. Default root is
`speed_estimation/passby`. If your tree still lives at the original checkout
path, pass it explicitly:

```powershell
--real_root D:\Antigravity\DopplerLab\speed_estimation\vs13
```

---

## How to run

Set `REAL` to your data root once:

```powershell
$REAL = "D:\Antigravity\DopplerLab\speed_estimation\vs13"
cd D:\Antigravity\DopplerLab\data_loading
```

### 1. Default scene split (recommended starting point)

Cars may appear in both sets at different speeds; ±1 km/h twins are allowed
(warned, not failed).

```powershell
D:\Antigravity\venv\Scripts\python.exe scripts\make_split.py `
  --protocol scene `
  --real_root $REAL `
  --seed 42 `
  --out splits\scene_split_s42.json
```

**Expected (~400 clips, seed 42, no synth):** roughly

```text
Wrote splits\scene_split_s42.json
protocol=scene min_speed_gap=1
real train/test=340/60  fit/val≈289/51  synth train/test=0/0
WARN: N test scenes have a +/-1 km/h twin ...
WARN: Scene protocol: 13 vehicles appear in both train and test ...
```

Then verify:

```powershell
D:\Antigravity\venv\Scripts\python.exe scripts\verify_no_leakage.py `
  --split splits\scene_split_s42.json
```

**Expected:**

```text
LEAKAGE CHECK OK
  protocol=scene min_speed_gap=1 train=340 test=60 test_scenes=60
```

### 2. Opt-in: ban ±1 km/h near-twins (spread)

```powershell
D:\Antigravity\venv\Scripts\python.exe scripts\make_spread_scene_split.py `
  --real_root $REAL `
  --out splits\scene_split_s42_gap2.json
```

Same via main script: `make_split.py --protocol scene --min-speed-gap 2`.

**Expected (seed 42):** test set shrinks (e.g. 60 → ~51) as conflicting test
clips are demoted into train; no +/-1 twin warning; still warns that cars
appear in both sets (still not LOVO).

### 3. Opt-in: leave-one-vehicle-out (unknown car)

```powershell
D:\Antigravity\venv\Scripts\python.exe scripts\make_lovo_split.py `
  --real_root $REAL `
  --out splits\lovo
```

**Expected:** one fold JSON per vehicle under `splits/lovo/` plus `index.json`.
Each fold holds out one vehicle entirely from train.

### 4. Mel stats on fit only

```powershell
D:\Antigravity\venv\Scripts\python.exe scripts\compute_mel_stats.py `
  --split splits\scene_split_s42.json `
  --real_root $REAL `
  --out splits\mel_stats_fit_s42.json
```

**Expected:** JSON with per-mel `mean` / `std` arrays; built only from `fit_uids`.
If you pass val/test into stats, `assert_no_leakage` raises.

### 5. Use from Python

```python
from passby_data import (
    build_scene_split,
    build_lovo_folds,
    load_for_training,
    assert_no_leakage,
)

split = build_scene_split(r"D:\Antigravity\DopplerLab\speed_estimation\vs13", seed=42)
assert_no_leakage(split, stats_uids=split.fit_uids, eval_uids=split.test_uids)

bundle = load_for_training(split, mode="real", real_root=r"D:\Antigravity\DopplerLab\speed_estimation\vs13")
X_fit, y_fit = bundle.fit.paths, bundle.fit.speeds_kmh
```

Optional synth root for paired real/synth scene splits:

```python
split = build_scene_split(real_root, synth_root=r"path\to\synth", seed=42)
bundle = load_for_training(split, real_root=..., synth_root=..., mode="mixed")
```

### 6. Tests

```powershell
D:\Antigravity\venv\Scripts\python.exe -m pytest data_loading\tests -q
```

**Expected:** all tests pass when the data root is present (7 tests covering
discover, scene, spread, LOVO, stats leak catch). Tests look for
`speed_estimation/passby` by default; symlink or set the tree there, or edit
the test root if your copy lives elsewhere.

---

## CLI cheat sheet

| Command | Purpose |
|---------|---------|
| `scripts/make_split.py` | Default scene (or `--protocol lovo` / `speed_stratified`) |
| `scripts/make_split.py --min-speed-gap 2` | Scene + no ±1 twins |
| `scripts/make_spread_scene_split.py` | Same as gap=2 (dedicated entry) |
| `scripts/make_lovo_split.py` | Unknown-vehicle folds |
| `scripts/verify_no_leakage.py --split …` | Fail loud on leakage |
| `scripts/compute_mel_stats.py --split … --out …` | Fit-only mel stats |

Also: `python -m passby_data make-split|make-lovo|make-spread|verify …`

---

## Honest limits

- **Scene ≠ LOVO.** Same car at different speeds can sit in train and test.
- **Default allows ±1 km/h twins** of the same car across the split; use
  `--min-speed-gap 2` if that matters for your paper claim.
- **Labels** come from the filename speed integer.
- This package does **not** train models; it only supplies clean splits,
  stats, and loaders. Wire `bundle.fit` / `bundle.test` into your trainer.

---

## Relation to prior SE-ResNet work

Ported and tightened from the scene-level protocol used in the SE-ResNet
downstream package (`downstream_split_lib.py`, `train_seresnet_scene.py`).
Improvements here:

- Mel stats default to **fit only** (not fit+inner-val).
- Explicit opt-in LOVO and speed-gap spreading.
- Single verifier API for train / synth / DiT / eval lists.

Do **not** revive full-set K-fold ensemble eval for paper “held-out” numbers.
