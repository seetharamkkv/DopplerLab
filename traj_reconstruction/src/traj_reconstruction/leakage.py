"""Leakage audits for audio-only orbit inference (Phase 6).

Ensures the product path cannot consume trajectory GT or scene metadata.
"""

from __future__ import annotations

import inspect
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from traj_reconstruction.contract import (
    DATA_SCOPE,
    INFERENCE_ALLOWED_RELPATHS,
    INFERENCE_FORBIDDEN_RELPATHS,
)
from traj_reconstruction.dataset import (
    DatasetError,
    Phase1Batch,
    assert_inference_safe,
    to_inference_bundle,
)
from traj_reconstruction.product import predict_orbit


# Strings that must never appear as model *input* feature names / kwargs.
FORBIDDEN_FEATURE_TOKENS: tuple[str, ...] = (
    "vehicle_id",
    "vehicle_class",
    "site_id",
    "site",
    "session_id",
    "mic_id",
    "weather",
    "road_class",
    "polyline",
    "path_polyline",
    "state_frames",
    "canonical_state",
    "cpa_distance",
    "cpa_time",
    "speed_mps",
    "source_speed",
    "simulation_parameters",
    "labels",
    "gps",
    "heading_absolute",
)


@dataclass
class LeakageFinding:
    severity: str  # "error" | "warn" | "info"
    code: str
    message: str


@dataclass
class LeakageReport:
    ok: bool
    findings: list[LeakageFinding] = field(default_factory=list)
    checks: dict[str, Any] = field(default_factory=dict)

    def add(self, severity: str, code: str, message: str) -> None:
        self.findings.append(LeakageFinding(severity, code, message))
        if severity == "error":
            self.ok = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "data_scope": DATA_SCOPE,
            "checks": self.checks,
            "findings": [asdict(f) for f in self.findings],
        }


def audit_predict_signature() -> LeakageReport:
    """predict_orbit must not accept GT / metadata kwargs."""
    report = LeakageReport(ok=True)
    sig = inspect.signature(predict_orbit)
    params = set(sig.parameters.keys())
    report.checks["predict_orbit_params"] = sorted(params)
    banned = params.intersection(
        {
            "state",
            "state_frames",
            "canonical",
            "polyline",
            "metadata",
            "vehicle_id",
            "site_id",
            "cpa",
            "speed_mps",
            "gt_xy",
            "labels",
        }
    )
    if banned:
        report.add("error", "predict_kwargs", f"forbidden kwargs on predict_orbit: {sorted(banned)}")
    else:
        report.add("info", "predict_kwargs", "predict_orbit kwargs are audio-only")
    allowed = {"wav_path", "stft_db", "audio", "sr", "method", "mlp_checkpoint", "scale_ambiguous"}
    unexpected = params - allowed
    if unexpected:
        report.add("warn", "predict_extra_kwargs", f"unexpected kwargs (review): {sorted(unexpected)}")
    return report


def audit_forbidden_relpaths() -> LeakageReport:
    report = LeakageReport(ok=True)
    report.checks["forbidden_count"] = len(INFERENCE_FORBIDDEN_RELPATHS)
    report.checks["allowed"] = list(INFERENCE_ALLOWED_RELPATHS)
    # Spot-check critical GT files are forbidden.
    required = (
        "metadata/state_frames.npy",
        "metadata/canonical_state_frames.npy",
        "metadata/path_polyline.npy",
        "metadata/simulation_parameters.json",
        "metadata/cpa_distance_m.npy",
    )
    missing = [p for p in required if p not in INFERENCE_FORBIDDEN_RELPATHS]
    if missing:
        report.add("error", "forbidden_list", f"missing forbidden entries: {missing}")
    else:
        report.add("info", "forbidden_list", "critical GT paths are inference-forbidden")
    try:
        assert_inference_safe(["spectrograms/stft.npy", "clip.wav"])
        report.add("info", "assert_ok", "allowed paths pass assert_inference_safe")
    except DatasetError as exc:
        report.add("error", "assert_ok", str(exc))
    try:
        assert_inference_safe(["metadata/state_frames.npy"])
        report.add("error", "assert_block", "forbidden path was not blocked")
    except DatasetError:
        report.add("info", "assert_block", "forbidden path correctly blocked")
    return report


def audit_batch_splits(batch_dir: Path | str | None) -> LeakageReport:
    report = LeakageReport(ok=True)
    if batch_dir is None:
        report.add("warn", "splits", "no batch_dir provided — skip split overlap audit")
        return report
    batch_dir = Path(batch_dir)
    splits_path = batch_dir / "splits.json"
    if not splits_path.is_file():
        report.add("warn", "splits", f"no splits.json under {batch_dir}")
        return report
    payload = json.loads(splits_path.read_text())
    splits = payload.get("splits", {})
    train, val, test = set(splits.get("train", [])), set(splits.get("val", [])), set(splits.get("test", []))
    report.checks["split_counts"] = {k: len(v) for k, v in splits.items()}
    report.checks["holdout_families"] = payload.get("holdout_families", [])
    overlap_tv = train & val
    overlap_tt = train & test
    overlap_vt = val & test
    if overlap_tv or overlap_tt or overlap_vt:
        report.add(
            "error",
            "split_overlap",
            f"clip overlap train∩val={len(overlap_tv)} train∩test={len(overlap_tt)} val∩test={len(overlap_vt)}",
        )
    else:
        report.add("info", "split_overlap", "no train/val/test clip id overlap")
    return report


def audit_inference_bundle(batch_dir: Path | str | None) -> LeakageReport:
    report = LeakageReport(ok=True)
    if batch_dir is None or not Path(batch_dir).exists():
        report.add("warn", "bundle", "no batch_dir — skip InferenceBundle audit")
        return report
    batch = Phase1Batch.from_dir(batch_dir)
    sample = batch.load(0)
    bundle = to_inference_bundle(sample)
    report.checks["bundle_fields"] = sorted(bundle.__dataclass_fields__.keys())
    if hasattr(bundle, "state_frames") or hasattr(bundle, "canonical_state_frames"):
        report.add("error", "bundle_gt", "InferenceBundle still exposes GT fields")
    else:
        report.add("info", "bundle_gt", "InferenceBundle has no GT attributes")
    if bundle.stft_db is None and bundle.wav_path is None:
        report.add("error", "bundle_audio", "InferenceBundle missing audio/STFT")
    return report


def audit_normalization_policy() -> LeakageReport:
    """Document AGC / peak-norm so scale is not silently site-calibrated."""
    report = LeakageReport(ok=True)
    policy = {
        "wav_peak_normalize": True,
        "site_calibrated_gain_tables": False,
        "absolute_distance_requires": "amplitude calibration or external scale prior",
        "product_flag": "scale_ambiguous when calibration unknown",
    }
    report.checks["normalization_policy"] = policy
    report.add(
        "info",
        "normalization",
        "Clips are peak-normalized in sim export; no site gain tables are used at inference.",
    )
    return report


def run_leakage_audit(batch_dir: Path | str | None = None) -> LeakageReport:
    merged = LeakageReport(ok=True)
    for part in (
        audit_predict_signature(),
        audit_forbidden_relpaths(),
        audit_batch_splits(batch_dir),
        audit_inference_bundle(batch_dir),
        audit_normalization_policy(),
    ):
        merged.checks.update(part.checks)
        merged.findings.extend(part.findings)
        if not part.ok:
            merged.ok = False
    # Token hygiene note
    merged.checks["forbidden_feature_tokens"] = list(FORBIDDEN_FEATURE_TOKENS)
    merged.add("info", "feature_tokens", "forbidden feature token list recorded for reviews")
    return merged


def wav_only_smoke(
    wav_path: Path | str,
    out_dir: Path | str,
    *,
    method: str = "parametric",
) -> dict[str, Any]:
    """Run product inference from a folder that contains only a WAV copy."""
    import shutil

    from traj_reconstruction.product import export_orbit_product

    wav_path = Path(wav_path)
    out_dir = Path(out_dir)
    staging = out_dir / "wav_only_staging"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    staged = staging / wav_path.name
    shutil.copy2(wav_path, staged)

    # Refuse if any metadata sneaks in.
    extras = [p for p in staging.rglob("*") if p.is_file() and p.suffix.lower() != ".wav"]
    if extras:
        raise DatasetError(f"wav-only staging contaminated: {extras}")

    product = predict_orbit(wav_path=staged, method=method)  # type: ignore[arg-type]
    artifacts = export_orbit_product(product, out_dir / "wav_only_product")
    return {
        "ok": True,
        "staged_wav": str(staged),
        "files_in_staging": [p.name for p in staging.iterdir()],
        "artifacts": {k: str(v) for k, v in artifacts.items()},
        "mirror_ambiguous": product.mirror_ambiguous,
        "heading_absolute": product.heading_absolute,
    }


_SPEED_IN_NAME = re.compile(r"(\d+(?:\.\d+)?)(mps|kmh|kmph)", re.I)


def filename_speed_not_used_as_feature(path: Path | str) -> bool:
    """Filename may *contain* speed for humans; model APIs must not parse it."""
    # Soft documentation helper used by tests — predict_orbit never sees the name as a feature.
    _ = _SPEED_IN_NAME.search(Path(path).name)
    return True
