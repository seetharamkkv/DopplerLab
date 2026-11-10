# Position from pass-by audio

Recover an observer-relative vehicle trajectory from one omnidirectional microphone. Labels are timed `x, vx, y, vy` on the STFT frame clock. Position is an integral of velocity, not a class. The microphone is the origin.

This track is separate from [`../ctc_exp`](../ctc_exp). That package classifies one clip-level speed bin. The 91.9% figure there is that classifier, not a CTC decode.

## Datasets

Audio is not in this repository. Point the scripts at a local clip directory.

| Corpus | Clips | Environment variable | What it is |
|--------|------:|----------------------|------------|
| Free-path 2D | 1300 (13 vehicles × 100) | `DOPPLER_AUDIO_CLIPS` | Constant-speed polylines. Vehicle holdout seed 0: 900 / 200 / 200. Used by the path ablations and MotionNet runs. |
| Path2D 10 speed bins | 130 | `DOPPLER_PATH2D_CLIPS` | Later DopplerSim export. Used by the oracle-range decoder and the continuous velocity tracker. |
| `vs13_path2d_7500_varied` | 7500 | local `datasets/` (not in git) | Constant-speed free paths, 20–90 km/h. Ordered-velocity 7500 runs. Split seed 42: train 5250 / val 1125 / test 1125. |
| `vs13_dopplersim_2d_5000` | 5000 | local `datasets/` (not in git) | Accelerating free paths, five families of 1000. Preview only is tracked. |

Each clip is a `sample_*/` folder with a WAV plus `metadata/canonical_state_frames.npy` (`x, vx, y, vy`) and usually `metadata/frame_times.npy`. Canonical labels already place closest approach on the positive y-axis. Stationary pads (`hypot(vx, vy) ≤ 0.5`) are dropped unless a script says otherwise.

```bash
export DOPPLER_AUDIO_CLIPS=/path/to/audio_clips
export DOPPLER_PATH2D_CLIPS=/path/to/vs13_path2d_10_speed_bins/audio_clips
```

`--data` overrides the environment variable. Do not hardcode a home directory.

## Install and test

```bash
cd position_ctc
python -m venv .venv
.venv/bin/python -m pip install -e ".[dev,exp]"
.venv/bin/python -m pytest
```

From the DopplerLab root:

```bash
.venv/bin/python -m pip install -e "position_ctc[dev,exp]"
.venv/bin/python -m pytest position_ctc/tests
```

## What was tried

Three families of experiments live here. Weights (`.pt`) are not in git; retrain with the commands below. Metrics, configs, and plots are under `outputs/`.

### 1. Audio-only path heads (no velocity CTC)

Nine heads on the 1300-clip holdout, plus a fine-tune of the winner. Record: [`outputs/ablate/ablations.json`](outputs/ablate/ablations.json). Write-up: [`history.md`](history.md), [`ABLATIONS.md`](ABLATIONS.md).

On 200 held-out clips the selected model is **range + direction, fine-tuned**:

| Model | Orbit | Range MAE | Closest point |
|-------|------:|----------:|--------------:|
| Mean path (no audio) | 7.58 m | 4.63 m | 6.08 m |
| **Range + direction, fine-tuned** | **2.39 m** | **0.85 m** | **0.56 m** |
| Range + direction | 2.38 m | 0.88 m | 0.69 m |
| Residual on the mean path | 2.53 m | 1.21 m | 0.97 m |
| Cartesian path | 2.88 m | 1.47 m | 1.29 m |
| Integrated velocity | 6.12 m | 4.14 m | 1.08 m |

Orbit RMS is after the best rotation and reflection about the microphone. Range does not uniquely determine a free 2D path: knowing `R(t)` and `Ṙ(t)` still leaves `θ(t)` free.

```bash
position_ctc/.venv/bin/python position_ctc/scripts/ablate_position.py \
    --data "$DOPPLER_AUDIO_CLIPS" --epochs 30 --patience 6 --batch-size 8
position_ctc/.venv/bin/python position_ctc/scripts/ablate_position.py \
    --modes range_polar --resume position_ctc/outputs/ablate/range_polar.pt \
    --epochs 20 --patience 8 --lr 0.0002 --batch-size 8
```

### 2. Geometric ceiling from exact range (no audio)

Exact ground-truth `R(t)` into a family of geometric decoders. Record: [`outputs/oracle_range/metrics.json`](outputs/oracle_range/metrics.json), [`outputs/oracle_range/report.md`](outputs/oracle_range/report.md). 130 clips from `$DOPPLER_PATH2D_CLIPS`. 128 have a heading jump of at least 12°.

| Decoder | Motion assumption | Orbit | Range MAE |
|---------|-------------------|------:|----------:|
| `range_rdot_speed_true_sign` | Known speed, true sign of θ̇ | 0.05 m | 0.00 m |
| `global_sign_flip` | Known speed, global sign flip | 0.05 m | 0.00 m |
| `range_rdot_no_speed` | Unknown speed, true sign of θ̇ | 0.07 m | 0.00 m |
| `straight_family` | Constant-speed straight pass | 3.06 m | 1.31 m |
| `range_rdot_speed_constant_sign` | Known speed, one constant sign | 6.69 m | 0.00 m |
| `corner_branch` | Sign switch after first corner or CPA | 9.68 m | 0.00 m |

A global sign flip is a gauge and is removed by orbit alignment. A sign switch at a corner is a different shape with the same range profile.

```bash
position_ctc/.venv/bin/python position_ctc/scripts/oracle_range.py \
    --data "$DOPPLER_PATH2D_CLIPS"
```

### 3. Framewise velocity bins, CTC, and Viterbi

CNN / BiLSTM (or CNN–GRU) emits per-frame `vx`, `vy` bins (2 m/s from −30 to +30). Viterbi times the path. Trapezoidal integration starts at the labeled first position.

`scripts/train_motion.py` is the MotionNet trainer. `scripts/trajectory_framewise.py` is the 86 Hz STFT trainer (`44100/512`). Bin classifiers collapse to the majority (`vx ≈ +14`, `vy ≈ 0`) because most frames are a straight pass.

**MotionNet polar fine-tune** (200 held-out clips, epoch 27), [`outputs/motion/metrics.json`](outputs/motion/metrics.json):

| Metric | Value |
|--------|------:|
| vx MAE | 5.62 m/s |
| vy MAE | 2.39 m/s |
| Integrated position RMSE | 16.76 m |
| Viterbi integrated RMSE | 7.77 m |

The earlier epoch-38 Viterbi run: vx MAE 0.98 m/s, vy MAE 1.48 m/s, integrated RMSE 6.74 m. Constant velocity still beats a mistimed bend on average.

**trajectory86_fixed** test split (10 clips, epoch 43), [`outputs/trajectory86_fixed/test_cases/TEST_CASES.md`](outputs/trajectory86_fixed/test_cases/TEST_CASES.md):

| | Trajectory RMSE | vy MAE |
|--|----------------:|-------:|
| Mean over test clips | 23.38 m | 6.39 m/s |
| Best clip | 2.62 m | 0.29 m/s |
| Median clip | 16.87 m | 2.81 m/s |
| Worst clip | 64.12 m | 10.11 m/s |

```bash
position_ctc/.venv/bin/python position_ctc/scripts/train_motion.py \
    --data "$DOPPLER_AUDIO_CLIPS" \
    --out position_ctc/outputs/motion
position_ctc/.venv/bin/python position_ctc/scripts/trajectory_framewise.py \
    --root "$DOPPLER_PATH2D_CLIPS" \
    --out position_ctc/outputs/trajectory86_fixed
```

### 4. Continuous velocity regression (no CTC, no bins)

CNN–GRU emits continuous `vx(t), vy(t)`. Position is

```text
p_k = p0 + sum_{i=1..k} (v_{i-1} + v_i) / 2 * Δt_i
```

with the true first position as `p0`. Loss: velocity Huber (x and y separate) + integrated-position Huber + multi-duration displacement. Corners are oversampled. There is no smooth-turn penalty.

Record: [`outputs/velocity_track/report.md`](outputs/velocity_track/report.md), [`outputs/velocity_track/gt_integrate_audit.json`](outputs/velocity_track/gt_integrate_audit.json), [`outputs/velocity_track/overfit.json`](outputs/velocity_track/overfit.json).

Ground-truth velocity on the actual frame clock reconstructs labeled `x, y` to **0.041 m** RMSE (trapezoid) and **0.080 m** (Euler) on 130 clips. The integrator matches the labels, including corners.

Four-clip overfit (200 epochs, no early stop; clips `0000087`, `0000075`, `0000009`, `0000124`):

| | Epoch 1 | Epoch 200 |
|--|--------:|----------:|
| Path RMSE | 80.3 m | 11.0 m |
| vx MAE | 12.6 m/s | 3.8 m/s |
| vy MAE | 9.3 m/s | 7.6 m/s |

The joint objective did not reconstruct the bends (`fitted=False`), so the vehicle-holdout run was not started. The same encoder on one clip with a `vy`-only Huber reached **0.55 m/s** MAE in 120 epochs.

```bash
position_ctc/.venv/bin/python position_ctc/scripts/train_velocity.py \
    --data "$DOPPLER_PATH2D_CLIPS"
```

The full split starts only if overfit path RMSE < 2 m, vx MAE < 1 m/s, and vy MAE < 1.5 m/s.

### 5. Ordered velocity bins on the 7500-clip Path2D set

CNN–GRU and E-Branchformer each emit ordered `vx`, `vy` bins (1 m/s from −60 to +60). Position is the trapezoid integral from the labeled first position. Both runs use the same seed-42 split of `vs13_path2d_7500_varied`: 5250 / 1125 / 1125. Weights, STFT caches, and the audio stay local. Metrics and the figures below are the GitHub copy.

**1000-clip CNN–GRU** (`outputs/ordered_velocity_full`, test n = 150, epoch 150, Viterbi penalty 3.0). Write-up: [`outputs/ordered_velocity_full/RESULTS.md`](outputs/ordered_velocity_full/RESULTS.md).

| ADE | FDE | Path RMSE | vx MAE | vy MAE |
|----:|----:|----------:|-------:|-------:|
| 8.08 m | 14.25 m | 9.14 m | 0.95 m/s | 1.94 m/s |

Median ADE 6.02 m, median FDE 10.16 m. Gallery: `outputs/ordered_velocity_full/inference/gallery_paths.png`.

**7500-clip CNN–GRU** (`outputs/ordered_velocity_7500`, test n = 1125, epoch 75, penalty 0). Record: [`outputs/ordered_velocity_7500/inference/test_metrics.json`](outputs/ordered_velocity_7500/inference/test_metrics.json).

| ADE | Median ADE | FDE | Median FDE | Path RMSE | vx MAE | vy MAE |
|----:|-----------:|----:|-----------:|----------:|-------:|-------:|
| 6.03 m | 4.74 m | 10.33 m | 7.83 m | 6.81 m | 0.73 m/s | 1.54 m/s |

Penalty 3.0 is almost the same (ADE 6.06 m). Representative clips, best to worst:

| Role | Clip | Path RMSE | ADE | Endpoint |
|------|------|----------:|----:|---------:|
| Best | `sample_0001126` | 0.39 m | 0.35 m | 0.28 m |
| Median | `sample_0006007` | 5.46 m | 3.87 m | 14.81 m |
| Turning | `sample_0006479` | 2.02 m | 1.79 m | 1.74 m |
| Worst | `sample_0006589` | 43.23 m | 40.02 m | 65.23 m |

Plots: `outputs/ordered_velocity_7500/inference/` (`gallery_paths.png`, `detail_best_path.png`, `detail_turning_path.png`, and the per-clip pairs named in `figures_manifest.json`).

**7500-clip E-Branchformer** (`outputs/ordered_velocity_7500_ebranchformer`, 6 layers, 4 heads, kernel 15, hidden 128, epoch 95). Same split. Record: [`outputs/ordered_velocity_7500_ebranchformer/test_metrics.json`](outputs/ordered_velocity_7500_ebranchformer/test_metrics.json). Viterbi penalty 0, max jump 8, matches these numbers. Oracle integration of the labels is 0.005 m ADE, so the integrator is not the error.

| ADE | FDE | Orbit ADE | Orbit RMS | Range MAE | CPA error | vx MAE | vy MAE | Turn ADE |
|----:|----:|----------:|----------:|----------:|----------:|-------:|-------:|---------:|
| 6.21 m | 10.52 m | 5.72 m | 6.11 m | 3.38 m | 5.46 m | 0.69 m/s | 1.65 m/s | 8.49 m |

Best validation ADE was 6.12 m. Twenty test drawings are in `outputs/ordered_velocity_7500_ebranchformer/inference_test_viterbi/` (for example `sample_0000002.png`). Per-clip scores: `inference_test_viterbi/per_clip.csv`.

vx stays near 0.7 m/s MAE on both 7500 models. vy stays near 1.6 m/s, and the worst paths are tens of metres. The E-Branchformer does not beat the CNN–GRU on ADE (6.21 m versus 6.03 m).

### 6. Accelerating free-path set (generated, not trained)

`datasets/vs13_dopplersim_2d_5000` is a 5000-clip DopplerSim run (seed 20261110, recipe `waypoint-walk-2d-v3`). Each clip is 10 s at 44.1 kHz. Speed stays in 3–36 m/s and acceleration magnitude stays at or below 5 m/s². Families are balanced: 1000 cruise, 1000 accelerate, 1000 decelerate, 1000 brake, 1000 mixed. The axis-balance check passed on the first attempt (worst normalized gap 0.0879, limit 0.16). Three WAVs were regenerated and matched.

The audio is about 86 GB and is not in git. The tracked preview is:

- [`outputs/dopplersim_2d_5000/summary.json`](outputs/dopplersim_2d_5000/summary.json)
- [`outputs/dopplersim_2d_5000/trajectory_preview.png`](outputs/dopplersim_2d_5000/trajectory_preview.png)
- [`outputs/dopplersim_2d_5000/axis_balance.png`](outputs/dopplersim_2d_5000/axis_balance.png)

## Identifiability

Write the path as `p(t) = R(t) [cos θ(t), sin θ(t)]`. Audio on one omni mic is informative about range and range rate. That does not uniquely determine `θ(t)`. Orbit alignment removes a global rotation and reflection; it does not remove time-varying heading. Persistent signed velocity bias accumulates in the integral; plot cumulative signed error per component to see bias versus mistimed corners.

## Layout

| Path | Role |
|------|------|
| `src/position_ctc/` | Features, labels, encoder, losses, trapezoid integrator, oracle-range decoder |
| `scripts/ablate_position.py` | Path-head ablations |
| `scripts/oracle_range.py` | Geometric ceiling from GT `R(t)` |
| `scripts/train_motion.py` | MotionNet bins + CTC + Viterbi |
| `scripts/trajectory_framewise.py` | 86 Hz framewise trainer |
| `scripts/train_velocity.py` | Continuous `vx, vy` + trapezoid |
| `outputs/` | Metrics, reports, and plots from the runs above |

Not in git: WAV files, `.pt` weights, STFT caches, and `records.json` files that stored machine-local absolute paths.
