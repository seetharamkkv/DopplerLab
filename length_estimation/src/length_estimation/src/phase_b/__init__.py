"""Phase B: CNN length regression on VS13 pass-by spectrograms."""

from length_estimation.src.phase_b.eval import run_eval
from length_estimation.src.phase_b.inference import predict_clip, predict_wav
from length_estimation.src.phase_b.train import train_lovo, train_split

__all__ = ["train_split", "train_lovo", "run_eval", "predict_wav", "predict_clip"]
