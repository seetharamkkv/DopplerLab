# CTC experiments (kinematics retrieval)

Phases 0–2: token labels, a small CNN-GRU CTC model, string retrieval, and a clip-level speed classifier used as a tracker head.

The position-from-audio record is separate: [`../position_ctc/outputs/ablate/ablations.json`](../position_ctc/outputs/ablate/ablations.json). Runs in this folder are recorded in [`outputs/replications.json`](outputs/replications.json). `train_time_seconds` is null where duration was not logged.

## Data

Audio is not in this repository. Copy [`../dataset_paths.example.json`](../dataset_paths.example.json) to `dataset_paths.json` at the DopplerLab root, or set:

```bash
export DOPPLER_VS13=/path/to/vs13
export DOPPLER_SIM_TEST=/path/to/dopplersim_test_dataset
export DOPPLER_SIM_UNCONSTRAINED=/path/to/dopplersim_unconstrained
```

VS13 and DopplerSim `test_dataset` are trained separately.

## Install

```bash
# from DopplerLab repo root
.venv/bin/python -m pip install -e ctc_exp
.venv/bin/python ctc_exp/scripts/run_phases.py --dataset both --epochs 200
.venv/bin/python ctc_exp/scripts/run_track.py --epochs 200
```

Outputs: `ctc_exp/outputs/vs13/`, `ctc_exp/outputs/sim/`, and unconstrained tracking in `ctc_exp/outputs/unconstrained/`.

## Recorded metrics

From [`outputs/replications.json`](outputs/replications.json):

| Run | What the number is | Test n | Result |
|-----|--------------------|-------:|--------|
| sim CTC | token error rate | 148 | 0.177 |
| sim CTC | decoded speed-bin accuracy | 148 | 0.054 (majority 0.115) |
| sim retrieval @10 | speed recall | 148 | 0.926 |
| vs13 CTC | see `outputs/vs13/summary.json` | 58 | in the JSON |
| unconstrained tracker | clip-level speed-bin accuracy | 148 | 0.926 |
| unconstrained_velcond | clip-level speed-bin accuracy | 148 | **0.919** |
| unconstrained_velcond | orbit RMS | 148 | 38.4 m |
| straight_pos | clip-level speed-bin accuracy | 148 | 0.115 (all predictions V20) |

The 0.919 figure is the argmax of a clip-level speed classifier, not a CTC decode. CTC logits are unpacked as `_ctc` in `run_track.py` and are not in that objective.

See [`PLAN.md`](PLAN.md).
