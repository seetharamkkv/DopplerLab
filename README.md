# DopplerLab: Modular ML System

A modular, reproducible machine learning system for multi-task Doppler audio analysis. Predicts vehicle trajectory path, source speed, and source distance from simulated Doppler-shifted audio clips using CNN and Transformer-based architectures.

> **Looking for the dataset simulator?**  
> Audio clips are generated using **DopplerNet**, a Flask-based Doppler audio simulator with physically accurate wave modelling, multi-path trajectory support, and a full web UI.  
> → [github.com/rohitharumugams/dopplersim_2.0](https://github.com/rohitharumugams/dopplersim_2.0)

---

## What this repository is

**DopplerLab** is a collection of model packages and experiment code for recovering **motion-related information** from pass-by audio. The goal is not only to detect that something moved, but to infer **how** it moved: speed, path shape, closest approach, direction of travel, vehicle identity, and related quantities.

This repo holds the models, training pipelines, evaluation outputs, and baselines for each task. Work spans two complementary settings:

- **Simulated pass-bys:** synthetic clips from [DopplerSim 2.0](https://github.com/rohitharumugams/dopplersim_2.0) where geometry and kinematics are known exactly, so models can be tested on controlled physics.
- **Real roadside recordings:** public traffic and pass-by datasets where labels exist for a subset of tasks (direction, vehicle type, length, etc.) and generalization (site, weather, microphone) matters.

Each subfolder is a self-contained package or notebook track with its own README, install steps, and (where applicable) tracked metrics under `outputs/`.

---

## Problem context

Most audio machine learning still targets **static** events, such as classifying a sound type or detecting an onset. **Dynamic audio motion understanding** asks a harder question: given a microphone recording of a moving source, can we recover physically meaningful **kinematics and geometry**?

A structured benchmark suite motivates this work. At a high level it covers:

| Theme | What we ask from audio |
|--------|-------------------------|
| **Kinematics** | How fast is the source moving? Is it accelerating or decelerating? |
| **Geometry** | How close does it pass? What is the overall trajectory shape? |
| **Direction and timing** | Which way is it traveling relative to the sensor? When will a key event (e.g. closest approach) occur? |
| **Temporal structure** | Can motion phases (approach, nearest point, recede) be segmented over time? |
| **Multi-source scenes** | Can multiple moving sources be separated or counted? |
| **Identity under motion** | Can vehicle or source type be recognized while the source is moving? |
| **Structured and causal reasoning** | Do models respect interventions (e.g. flipped geometry)? Can motion be inferred on manifolds or inverted from observations? |

Simulated data supports the full breadth of these tasks. Real-world datasets implement **overlapping subsets** (for example lateral pass-by direction and vehicle type on stereo roadside clips, or vehicle length from single-mic pass-bys) and are used to test whether ideas learned in simulation transfer to deployment-like conditions.

---

## What we are working on now

| Track | Focus | Status |
|-------|--------|--------|
| [**IDMT_experiments**](IDMT_experiments/) | Real traffic audio: pass-by direction, vehicle type, generalization splits, classical and neural baselines | Active: code + eval outputs in repo |
| [**length_estimation**](length_estimation/) | Real pass-bys (VS13): estimate vehicle length from a single-microphone recording | Active package |
| [**speed_estimation**](speed_estimation/) | Simulated multi-task benchmarks (path, speed, distance) on DopplerSim batch exports | Notebook-first training track |
| [**engine_acoustics**](engine_acoustics/) | Four-stroke engine source model: RPM / cylinder acoustics synthesis + order tracking | Active package |

Archived or scratch runs may appear under `backup/`. Local planning notes and PDFs live under `ref_docs/` (not tracked in git).

---

## Subprojects

Each implemented track has its own README with setup, splits, and usage.

| Package | README |
|---------|--------|
| [**IDMT_experiments**](IDMT_experiments/) | [IDMT_experiments/README.md](IDMT_experiments/README.md) |
| [**length_estimation**](length_estimation/) | [length_estimation/README.md](length_estimation/README.md) |
| [**speed_estimation**](speed_estimation/) | [speed_estimation/README.md](speed_estimation/README.md) |
| [**engine_acoustics**](engine_acoustics/) | [engine_acoustics/README.md](engine_acoustics/README.md) |

---

## Repository layout

```
DopplerLab/
├── IDMT_experiments/     # Real-world traffic experiments + committed eval outputs
├── length_estimation/    # VS13 vehicle length
├── speed_estimation/     # Simulated speed / path / distance (notebooks)
├── engine_acoustics/     # Engine RPM / cylinder physics synthesis + order analysis
├── backup/               # Archived experiment artifacts
└── ref_docs/             # Local benchmarks and write-ups (gitignored)
```

**Typically not in git:** raw audio datasets, trained `.pt` weight files, `ref_docs/`, and PDFs. See each package README and root `.gitignore` for details.

---

## Authors

**Seetharam Killivalavan & Rohith Arumugam Suresh**  
School of Computer Science, Carnegie Mellon University

---

## Acknowledgments

Carnegie Mellon University, the Language Technologies Institute, Bradley Warren, and Professor Bhiksha Raj for research guidance and support.
