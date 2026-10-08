# CTC experiments (kinematics retrieval)

Phases 0–2: token labels, a small CNN-GRU CTC model, string retrieval.

The replication record for the CTC position experiments is [`../position_ctc/outputs/ablate/ablations.json`](../position_ctc/outputs/ablate/ablations.json). The runs in this folder — token CTC on DopplerSim and VS13, plus the tracker whose speed number is a clip-level classifier — are recorded in [`outputs/replications.json`](outputs/replications.json). `train_time_seconds` is null in both files where duration was not logged.

Dataset roots live in [`../ref_docs/dataset_paths.json`](../ref_docs/dataset_paths.json) (VS13 and DopplerSim `test_dataset` are trained **separately**).

```bash
# from DopplerLab repo root
.venv/bin/python -m pip install -e ctc_exp
.venv/bin/python ctc_exp/scripts/run_phases.py --dataset both --epochs 8
```

Outputs: `ctc_exp/outputs/vs13/`, `ctc_exp/outputs/sim/`, and unconstrained tracking in `ctc_exp/outputs/unconstrained/`.

```bash
.venv/bin/python ctc_exp/scripts/run_track.py --epochs 200
```

See [`PLAN.md`](PLAN.md).
