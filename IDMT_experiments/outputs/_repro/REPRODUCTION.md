# CNN baseline reproduction — verification report

**Date:** 2026-07-02
**Purpose:** Re-run evaluation on the original CNN direction checkpoints and verify the
outputs match the previously reported metrics. No retraining, no weight restoration — the
original `best.pt` files were already on disk.

## Checkpoints used (unchanged, pre-existing)

| Run | Path | Size | SHA-256 (first 16) | best val acc |
|-----|------|-----:|--------------------|-------------:|
| `mel_3class` (mean) | `checkpoints/cnn/direction/mel_3class/best.pt` | 1,736,197 B | `75d3a9989bcfb0c1` | 0.9511 |
| `mel_3class_left` | `checkpoints/cnn/direction/mel_3class_left/best.pt` | 1,736,197 B | `38da3ae0c8d7f008` | 0.9319 |
| `mel_3class_right` | `checkpoints/cnn/direction/mel_3class_right/best.pt` | 1,736,197 B | `805522afa42e7cd9` | 0.9581 |

Each `best.pt` is a complete, self-contained model artifact (PyTorch `torch.save`): it
bundles `state_dict`, `config`, `norm_stats`, and training metadata. All three deserialize
cleanly via `load_checkpoint()` and ran inference end-to-end.

Re-run command (per model), writing to a separate dir to preserve originals:

```
python -m idmt_experiments.cnn.eval_cli --run-name <run> --split test --output-dir outputs/_repro
```

Repro artifacts: `outputs/_repro/cnn/direction/<run>/{eval_metrics.json,eval_summary.txt,eval_predictions.csv}`.

## Result: exact reproduction

3-class test metrics (EUSIPCO, n=2758) — **repro == original, bit-for-bit** on every shared field:

| Metric | mean | left | right |
|--------|-----:|-----:|------:|
| Accuracy | 0.90754 | 0.90102 | 0.87708 |
| Balanced acc (3-class) | 0.87627 | 0.86176 | 0.82366 |
| Macro F1 | 0.87455 | 0.86282 | 0.81994 |
| L2R recall | 0.86551 | 0.71203 | 0.51582 |
| R2L recall | 0.76471 | 0.87395 | 0.95658 |
| no_vehicle recall | 0.99858 | 0.99929 | 0.99858 |
| Channel-swap flip consistency | n/a* | 0.06771 (91/1344) | 0.11078 (149/1345) |

\*mean run uses downmixed mono; channel-swap diagnostic not applicable.

### Vehicle-only direction (derived from the reproduced confusion matrices)

Vehicle clips = 1346 (632 L2R + 714 R2L). Predicting `no_vehicle` on a vehicle clip counts wrong.

| Metric | mean | left | right | Reported table |
|--------|-----:|-----:|------:|----------------|
| 3-Class Acc. | 90.8 | 90.1 | 87.7 | 90.8 / 90.1 / 87.7 ✓ |
| Vehicle Acc. | 81.2 | 79.8 | 75.0 | 81.2 / 79.8 / 75.0 ✓ |
| Balanced Acc. (L2R/R2L) | 81.5 | 79.3 | 73.6 | 81.5 / 79.3 / 73.6 ✓ |
| Macro F1 (vehicle 2-class) | 81.3 | 79.5 | 73.2 | 81.3 / 79.5 / 73.2 ✓ |

Every value in the reported table is reproduced exactly.

## Discrepancies

**No numeric discrepancies.** One format-only difference:

- The original `mel_3class` (mean) `eval_metrics.json` was written by an **earlier version of
  the eval script** and stored a smaller field set (no `balanced_accuracy`,
  `per_class_precision`, `per_class_recall`). The reproduced file is a **superset**: it adds
  those fields (balanced acc 0.87627, precision/recall per class). All fields present in the
  original are byte-identical in the repro. The `left` and `right` original JSONs already had
  the full field set and are byte-identical to the repro.

This is expected: the additional metrics were added to `cnn/eval.py` after the mean run's
original report was generated. Re-running now simply produces the fuller, current report
format while leaving the underlying predictions/metrics identical.

## Model persistence

The trained models are already persisted as `best.pt` (PyTorch checkpoint = pickled payload).
This is the appropriate artifact for a torch model; a separate `.pkl` is unnecessary and would
duplicate the same pickled tensors. Physics models remain saved as `model.joblib`. The
baseline `best.pt` files were **not** modified, re-saved, or overwritten by this reproduction.
