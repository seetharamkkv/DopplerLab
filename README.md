# DopplerLab

Modular experiment code for recovering motion-related information from vehicle pass-by audio: direction of travel, length, speed, and related quantities.

This branch is an archive of the CTC / path-ablation / continuous-velocity work. Audio, model weights, and machine-local paths are not in git. Point scripts at a local clip directory with environment variables or `--data`.

For the Doppler audio simulator used in some tracks, see [DopplerSim 2.0](https://github.com/rohitharumugams/dopplersim_2.0).

## What this repository is

DopplerLab holds model packages, training pipelines, and evaluation outputs for pass-by audio tasks. Work spans two settings:

- Simulated pass-bys from DopplerSim, where geometry and kinematics are known exactly.
- Real roadside recordings (IDMT-Traffic, VS13) where labels cover a subset of tasks and site/microphone generalization matters.

Each subfolder is a self-contained package or notebook track with its own README.

## Problem context

Most audio ML targets static events (class labels, onsets). Dynamic motion understanding asks whether a microphone recording of a moving source can recover kinematics and geometry: speed, closest approach, trajectory shape, lateral direction, and related structure.

| Theme | Question from audio |
|--------|---------------------|
| Kinematics | How fast is the source? Is it accelerating? |
| Geometry | How close does it pass? What is the path shape? |
| Direction and timing | Which way relative to the sensor? When is CPA? |
| Temporal structure | Can approach / CPA / recede phases be segmented? |
| Multi-source scenes | Can multiple movers be separated or counted? |
| Identity under motion | Can vehicle type be recognized while moving? |

Simulated data supports the full breadth of these tasks. Real datasets implement overlapping subsets and test transfer to deployment-like conditions.

## Active tracks

| Track | Focus | Status |
|-------|--------|--------|
| [traj_reconstruction](traj_reconstruction/) | Sim-only: observer-centered trajectory orbit from DopplerSim Phase 1 / freehand | Active (Phase 0) |
| [IDMT_experiments](IDMT_experiments/) | Real traffic audio: direction, vehicle type, splits, baselines | Active |
| [length_estimation](length_estimation/) | VS13: vehicle length from a single mic | Active |
| [speed_estimation](speed_estimation/) | VS13 physics Ridge speed; simulated multi-task notebooks | Active |
| [engine_acoustics](engine_acoustics/) | Four-stroke engine RPM / order tracking | Active |
| [data_loading](data_loading/) | Shared VS13 / pass-by loaders | Active |
| [ctc_exp](ctc_exp/) | Clip-level speed bins and kinematic token retrieval | Active |
| [position_ctc](position_ctc/) | Time-resolved canonical position from pass-by audio | Active |

Local planning notes, manuscript drafts, and figures live under `ref_docs/` (not tracked).

## Data (not in git)

Copy [`dataset_paths.example.json`](dataset_paths.example.json) to `dataset_paths.json`, or export:

| Variable | Used by | Contents |
|----------|---------|----------|
| `DOPPLER_AUDIO_CLIPS` | `position_ctc` path ablations and MotionNet | 1300 free-path clips (`sample_*/` + `canonical_state_frames.npy`) |
| `DOPPLER_PATH2D_CLIPS` | oracle-range decoder, continuous velocity tracker, 86 Hz trainer | 130 Path2D clips |
| `DOPPLER_VS13` | `ctc_exp` | VS13 pass-bys |
| `DOPPLER_SIM_TEST` | `ctc_exp` | DopplerSim test set |
| `DOPPLER_SIM_UNCONSTRAINED` | `ctc_exp` tracker | unconstrained free-path set |
| `DOPPLER_BATCH_OUTPUTS` | `speed_estimation` notebook | DopplerSim batch export |

Do not commit `dataset_paths.json`, `.env`, or any absolute home-directory path.

## Replication records

The replication record for the CTC position experiments is [`position_ctc/outputs/ablate/ablations.json`](position_ctc/outputs/ablate/ablations.json). The readable write-up of every ablation is [`position_ctc/ABLATIONS.md`](position_ctc/ABLATIONS.md). `train_time_seconds` is null where a run did not record duration.

Headline numbers from this archive:

| Track | Result |
|-------|--------|
| Path head, range + direction (fine-tuned) | 2.39 m orbit, 0.85 m range MAE, 0.56 m closest-point (200 held-out clips) |
| Oracle range, true sign of θ̇ | 0.05 m orbit; a corner sign-switch with the same `R(t)` is 9.68 m |
| MotionNet Viterbi, epoch 38 | vx MAE 0.98 m/s, vy MAE 1.48 m/s, integrated RMSE 6.74 m |
| trajectory86_fixed test | mean RMSE 23.38 m, mean vy MAE 6.39 m/s |
| GT velocity trapezoid audit | 0.041 m RMSE vs labeled `x, y` (130 clips) |
| Continuous velocity 4-clip overfit | path 11.0 m, vx 3.8 m/s, vy 7.6 m/s after 200 epochs; holdout not started |
| Clip-level speed classifier (`ctc_exp` unconstrained_velcond) | 91.9% speed-bin accuracy; this is not CTC |

The other tracks keep the same kind of file:

| Track | Record |
|-------|--------|
| Clip-level CTC and the speed-classifier tracker | [`ctc_exp/outputs/replications.json`](ctc_exp/outputs/replications.json) |
| IDMT direction, weather, vehicle, physics, and fusion | [`IDMT_experiments/outputs/replications.json`](IDMT_experiments/outputs/replications.json) |
| Simulated trajectory OrbitMLP | [`traj_reconstruction/outputs/replications.json`](traj_reconstruction/outputs/replications.json) (val orbit mean 8.38 m) |
| VS13 vehicle length | [`length_estimation/replications.json`](length_estimation/replications.json) (LOVO MAE 0.235 m, matches the mean baseline) |
| Simulated DopplerNet speed notebook | [`speed_estimation/replications.json`](speed_estimation/replications.json) |
| Engine-acoustics synthesizer | [`engine_acoustics/replications.json`](engine_acoustics/replications.json) |
| Continuous velocity tracker | [`position_ctc/outputs/velocity_track/report.md`](position_ctc/outputs/velocity_track/report.md) |
| Oracle range decoder | [`position_ctc/outputs/oracle_range/report.md`](position_ctc/outputs/oracle_range/report.md) |

`data_loading/` builds splits and does not train a model, so it has no replication record. The 91.9% speed-bin figure is in the `ctc_exp` record. It is the clip-level speed classifier, not the CTC decode and not the position ablation.

## Repository layout

```
DopplerLab/
├── traj_reconstruction/  # sim-only trajectory orbit (Phase 0+)
├── IDMT_experiments/
├── length_estimation/
├── speed_estimation/
├── engine_acoustics/
├── data_loading/
├── ctc_exp/              # clip-level CTC retrieval
├── position_ctc/         # canonical position from audio
├── notebooks/
└── ref_docs/             # local write-ups and manuscript (not tracked)
```

Typically not in git: raw audio, `.pt` weights, `ref_docs/`, and PDFs. See each package README and `.gitignore`.

## Authors

Seetharam Killivalavan and Rohith Arumugam Suresh  
School of Computer Science, Carnegie Mellon University

## Acknowledgments

Carnegie Mellon University, the Language Technologies Institute, Bradley Warren, and Professor Bhiksha Raj for research guidance and support.
