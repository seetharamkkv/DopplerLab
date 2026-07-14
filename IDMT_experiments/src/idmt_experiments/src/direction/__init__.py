"""Deprecated import path — use idmt_experiments.cnn instead.

REPRODUCIBILITY BASELINE (CNN direction: mel_3class, mel_3class_left, mel_3class_right)
---------------------------------------------------------------------------------
Re-exports the CNN baseline package; behaviour is defined in ``idmt_experiments.cnn``.
"""

from idmt_experiments.cnn import dataset, eval, inference, metrics, model, train

__all__ = ["dataset", "eval", "inference", "metrics", "model", "train"]
