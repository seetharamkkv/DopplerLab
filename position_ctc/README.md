# Position from pass-by audio

Recover an observer-relative vehicle trajectory from one omnidirectional microphone. Labels are timed `x, vx, y, vy` on the STFT frame clock. Position is an integral of velocity, not a class. The microphone is the origin.

This track is separate from [`../ctc_exp`](../ctc_exp). That package classifies one clip-level speed bin. The 91.9% figure there is that classifier, not a CTC decode.

## Datasets

Audio is not in this repository. Point the scripts at a local clip directory.

| Corpus | Clips | Environment variable | What it is |
|--------|------:|----------------------|------------|
| Free-path 2D | 1300 (13 vehicles × 100) | `DOPPLER_AUDIO_CLIPS` | Constant-speed polylines. Vehicle holdout seed 0: 900 / 200 / 200. Used by the path ablations and MotionNet runs. |
| Path2D 10 speed bins | 130 | `DOPPLER_PATH2D_CLIPS` | Later DopplerSim export. Used by the oracle-range decoder and the continuous velocity tracker. |

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
