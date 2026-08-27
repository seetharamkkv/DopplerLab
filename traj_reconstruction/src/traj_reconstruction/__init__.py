"""Observer-centered trajectory orbit recovery from **simulated** DopplerSim audio.

This package currently supports:
  Phase 0 — contract, orbit metrics, Phase 1 I/O
  Phase 1 — Tier-1 freehand batch generation (pure-tone sim)

Real-world recordings are out of scope for now.
"""

from __future__ import annotations

from traj_reconstruction.audit import audit_batch
from traj_reconstruction.batch import build_tier1_batch, plan_tier1_batch
from traj_reconstruction.contract import (
    DATA_SCOPE,
    INFERENCE_ALLOWED_RELPATHS,
    INFERENCE_FORBIDDEN_RELPATHS,
    PATH_FAMILIES,
    PATH_TYPES,
    TRAINING_TARGET,
)
from traj_reconstruction.dataset import (
    DatasetError,
    InferenceBundle,
    Phase1Batch,
    Phase1Sample,
    assert_inference_safe,
    iter_batch_samples,
    load_phase1_sample,
    to_inference_bundle,
)
from traj_reconstruction.flexible import (
    OrbitMLP,
    fit_flexible_from_audio,
    fit_flexible_orbit,
    infer_orbit_mlp,
    train_orbit_mlp,
)
from traj_reconstruction.frontend import RidgeFeatures, extract_ridges, plot_ridge_overlay
from traj_reconstruction.orbit import (
    OrbitAlignResult,
    canonical_xy,
    orbit_align,
    orbit_family,
    xy_from_state,
)
from traj_reconstruction.parametric import (
    FitResult,
    fit_orbit_from_audio,
    fit_parametric_orbit,
    plot_fit_overlay,
)
from traj_reconstruction.leakage import run_leakage_audit, wav_only_smoke
from traj_reconstruction.product import (
    OrbitProduct,
    export_orbit_product,
    predict_orbit,
    render_orbit_png,
    write_orbit_viewer_html,
)
from traj_reconstruction.splits import write_splits
from traj_reconstruction.validate import run_all_tiers

__all__ = [
    "DATA_SCOPE",
    "INFERENCE_ALLOWED_RELPATHS",
    "INFERENCE_FORBIDDEN_RELPATHS",
    "PATH_FAMILIES",
    "PATH_TYPES",
    "TRAINING_TARGET",
    "DatasetError",
    "FitResult",
    "InferenceBundle",
    "OrbitAlignResult",
    "OrbitMLP",
    "OrbitProduct",
    "Phase1Batch",
    "Phase1Sample",
    "RidgeFeatures",
    "assert_inference_safe",
    "audit_batch",
    "build_tier1_batch",
    "canonical_xy",
    "export_orbit_product",
    "extract_ridges",
    "fit_flexible_from_audio",
    "fit_flexible_orbit",
    "fit_orbit_from_audio",
    "fit_parametric_orbit",
    "infer_orbit_mlp",
    "iter_batch_samples",
    "load_phase1_sample",
    "orbit_align",
    "orbit_family",
    "plan_tier1_batch",
    "plot_fit_overlay",
    "plot_ridge_overlay",
    "predict_orbit",
    "render_orbit_png",
    "run_all_tiers",
    "run_leakage_audit",
    "to_inference_bundle",
    "train_orbit_mlp",
    "wav_only_smoke",
    "write_orbit_viewer_html",
    "write_splits",
    "xy_from_state",
]

__version__ = "0.8.0"
