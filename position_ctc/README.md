# Position CTC

CTC model for an observer-relative vehicle trajectory. The target is a short string of quantized horizontal and vertical velocity, plus an acceleration pair when that velocity bin changes. Position is the integral of the decoded velocity, not a class.

Design plan: [`../ref_docs/position_detection_plan.md`](../ref_docs/position_detection_plan.md).

The replication record for these CTC position experiments is [`outputs/ablate/ablations.json`](outputs/ablate/ablations.json). That file is enough to rebuild the dataset contract, features, heads, loss, split, and reported metrics. `train_time_seconds` is null because per-model duration was not logged.

This track is separate from [`../ctc_exp`](../ctc_exp). That package classifies one clip-level speed bin. The 91.9% figure there is that classifier, not a CTC decode, and its record is [`../ctc_exp/outputs/replications.json`](../ctc_exp/outputs/replications.json).

## What is implemented

Phase 0 is the 100 Hz clock, the velocity and acceleration alphabet, the CTC string, and the integrator. The framewise model below is the trajectory network.

| Module | Role |
|--------|------|
| `clock` | 22 050 Hz, hop 220 (100.2 Hz), FFT 2048. Shared by features and labels |
| `alphabet` | `VX` / `VY` at 2 m/s from −30 to +30. `AX` / `AY` at 0.5 m/s² from −10 to +10. Widths are the plan's proposal |
| `tokens` | `state_frames` `[x, vx, y, vy]` → initial velocity pair, then acceleration pairs at bin changes |
| `integrate` | Hold each velocity bin and sum it into `x(t), y(t)` |
| `audio` | Magnitude plus sin/cos phase on that clock. DopplerSim `stft.npy` is magnitude-only at 43 Hz and is not the input |

A straight constant-speed pass is `VX+20 VY+0` and nothing else. A turn adds acceleration tokens. Strings longer than 40 tokens mean the alphabet is wrong.

## Install and test

```bash
cd position_ctc
python -m venv .venv
.venv/bin/python -m pip install -e ".[dev]"
.venv/bin/python -m pytest
```

From the DopplerLab root, the existing `.venv` can be used instead:

```bash
.venv/bin/python -m pip install -e position_ctc
.venv/bin/python -m pytest position_ctc/tests
```

## Labels

Read DopplerSim Phase 1 `metadata/state_frames.npy`. Columns are `x_m, vx_mps, y_m, vy_mps`, microphone at the origin. Resample onto `frame_times` from the WAV at hop 220 before calling `motion_tokens`. Do not use the CSV `speed_mps` column as the sequence target.

Score paths in the same gauge as `canonical_state_frames.npy`: rotation and reflection about the microphone are free.

## Framewise velocity model

`scripts/train_motion.py` trains one network. A CNN and a bidirectional LSTM read the spectrogram. Each frame then has six heads: signed `vx` and `vy` classifiers on the 2 m/s alphabet, a CTC head per axis, and continuous position and acceleration. CTC is trained on the collapsed bin string, so it learns the order of velocity states. The classifiers keep the timestamps. Viterbi, not CTC, chooses the timed path.

The velocity loss is cross-entropy plus the expected squared distance from the continuous label, so a far bin costs more than a neighbor. Position and acceleration use Huber losses. Two extra terms tie the heads together: trapezoidal expected velocity against the position step, and the acceleration head against the change in expected velocity. Acceleration labels are ignored where they exceed 5 m/s². Velocity frames whose step exceeds the jump limit, or whose velocity disagrees with `Δp/Δt` by more than 1 m/s, are left out of the classifier and CTC targets. Those frames are the polyline corners.

At inference a change of size `d` scores `-λ (d / 2 m/s)²`, and a change past 6 m/s (three bins on this 86 Hz grid) is forbidden. A literal passenger-car acceleration would move velocity by much less than one bin per frame and would freeze the path; four-bin jumps in these clips are the corner spikes. The decoded velocity is integrated with the trapezoid rule from the true first position. That integral is what the `x` and `y` panels show. The velocity panels also draw the framewise argmax, and the position panels also draw the direct head.

```bash
position_ctc/.venv/bin/python position_ctc/scripts/train_motion.py \
    --data /path/to/audio_clips \
    --out position_ctc/outputs/motion
```

The run writes `trajectory.png` (`vx`, `vy`, `x`, `y`) and `metrics.json`.

## Later phases

Phase 1 training is `scripts/train.py`. It holds out entire vehicles and retokenizes random time crops so a few hundred clips can still teach velocity change. Oracle `Δx, Δy` inputs and the closed-loop Phase 2 integrator are not in this script yet.

The position ablations can penalize infeasible yaw with `--kinematic-weight`, and `project_path` hard-clips a predicted trajectory onto the same cap. The cap is the tighter of a 2.70 m / 35° bicycle-model steering lock and Polack et al.'s `0.5 μ g` lateral acceleration (`position_ctc.kinematics`). On the held-out set the fine-tuned range-and-direction path violates that cap on 68% of tenth-second chords; the ground-truth polylines violate it on 12%. Projecting the prediction removes the violations and raises test orbit error from 2.39 m to 4.86 m. A short fine-tune with the penalty did not reduce the violation, so the enforced guard is the projection. Autoregressive displacement feedback, a phase-front-end change, 3D, and language conditioning are not part of this constraint.

```bash
position_ctc/.venv/bin/python position_ctc/scripts/train.py \
    --data /path/to/audio_clips
```
