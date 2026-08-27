# DopplerLab

Modular experiment code for recovering motion-related information from vehicle pass-by audio: direction of travel, length, speed, and related quantities.

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

Local planning notes, manuscript drafts, and figures live under `ref_docs/` (gitignored).

## Repository layout

```
DopplerLab/
├── traj_reconstruction/  # sim-only trajectory orbit (Phase 0+)
├── IDMT_experiments/
├── length_estimation/
├── speed_estimation/
├── engine_acoustics/
├── data_loading/
├── notebooks/
└── ref_docs/             # local write-ups and manuscript (not tracked)
```

Typically not in git: raw audio, `.pt` weights, `ref_docs/`, and PDFs. See each package README and `.gitignore`.

## Authors

Seetharam Killivalavan and Rohith Arumugam Suresh  
School of Computer Science, Carnegie Mellon University

## Acknowledgments

Carnegie Mellon University, the Language Technologies Institute, Bradley Warren, and Professor Bhiksha Raj for research guidance and support.
