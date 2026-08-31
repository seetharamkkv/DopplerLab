"""Side-by-side 2D whiteboard generate + OrbitCNN predict (live best checkpoint)."""

from __future__ import annotations

import json
import os
import sys
import threading
import time
import traceback
import uuid
from pathlib import Path
from typing import Any

import numpy as np

from traj_reconstruction.dataset import load_phase1_sample, to_inference_bundle
from traj_reconstruction.flexible import _sample_target, fit_flexible_from_audio
from traj_reconstruction.kinematics import polyline_arclength
from traj_reconstruction.orbit import orbit_align, xy_from_state
from traj_reconstruction.orbit_cnn import infer_learned_orbit, load_orbit_model
from traj_reconstruction.parametric import _pyplot
from traj_reconstruction.paths import (
    DEFAULT_ORBIT_CNN_BEST,
    DEFAULT_ORBIT_MLP_BEST,
    DEFAULT_ORBIT_SEQ_BEST,
    default_learned_checkpoint,
    REPO_ROOT,
)

KMH_PER_MPS = 3.6
JOBS_DIR = Path(__file__).resolve().parents[2] / "outputs" / "draw_eval_jobs"
WEB_DIR = Path(__file__).resolve().parents[2] / "web"

INFERENCE_MODEL_SPECS: tuple[dict[str, Any], ...] = (
    {
        "id": "cnn",
        "label": "2D CNN — complex STFT",
        "kind": "learned",
        "checkpoint": DEFAULT_ORBIT_CNN_BEST,
    },
    {
        "id": "seq",
        "label": "1D CNN — ridge f_obs + A_env",
        "kind": "learned",
        "checkpoint": DEFAULT_ORBIT_SEQ_BEST,
    },
    {
        "id": "mlp",
        "label": "MLP — log-magnitude STFT",
        "kind": "learned",
        "checkpoint": DEFAULT_ORBIT_MLP_BEST,
    },
    {
        "id": "flexible",
        "label": "Physics freeform fit",
        "kind": "physics",
        "checkpoint": None,
    },
)

_jobs: dict[str, dict[str, Any]] = {}
_lock = threading.Lock()


def dopplersim_root() -> Path:
    override = os.environ.get("DOPPLERSIM_ROOT", "").strip()
    if override:
        return Path(override).expanduser().resolve()
    sibling = REPO_ROOT.parent / "DopplerSim"
    return sibling.resolve()


def default_inputs_dir() -> Path:
    return dopplersim_root() / "static" / "inputs"


def list_source_clips() -> list[dict[str, Any]]:
    root = default_inputs_dir()
    if not root.is_dir():
        return []
    rows = []
    for wav in sorted(root.glob("*.wav")):
        meta = source_clip_meta(wav)
        rows.append({"name": wav.name, **meta})
    return rows


def source_clip_meta(wav: Path) -> dict[str, Any]:
    wav = Path(wav)
    sidecar = wav.with_suffix(".txt")
    v1_kmph = None
    t_cpa1 = None
    if sidecar.is_file():
        parts = sidecar.read_text(encoding="utf-8").strip().replace(",", " ").split()
        if len(parts) >= 1:
            v1_kmph = float(parts[0])
        if len(parts) >= 2:
            t_cpa1 = float(parts[1])
    return {
        "path": str(wav),
        "v1_kmph": v1_kmph if v1_kmph is not None else 50.0,
        "t_cpa1_s": t_cpa1 if t_cpa1 is not None else 2.0,
        "h1_m": 0.5,
    }


def checkpoint_banner(checkpoint: Path | None = None) -> dict[str, Any]:
    ckpt = Path(checkpoint or default_learned_checkpoint())
    status_path = ckpt.with_name(ckpt.stem + ".status.json")
    payload: dict[str, Any] = {
        "checkpoint": str(ckpt),
        "exists": ckpt.is_file(),
        "training": False,
    }
    if status_path.is_file():
        try:
            st = json.loads(status_path.read_text())
        except json.JSONDecodeError:
            st = {}
        payload["training"] = bool(st.get("running"))
        payload["epoch_completed"] = st.get("epoch_completed")
        payload["epochs_target"] = st.get("epochs_target")
        payload["best_val_orbit_rms"] = st.get("best_val_orbit_rms")
        payload["best_epoch"] = st.get("best_epoch")
        payload["note"] = st.get("note")
        payload["arch"] = st.get("arch")
    return payload


def list_inference_models() -> list[dict[str, Any]]:
    default_ckpt = default_learned_checkpoint()
    rows: list[dict[str, Any]] = []
    default_id = "flexible"
    for spec in INFERENCE_MODEL_SPECS:
        ckpt: Path | None = spec["checkpoint"]
        available = spec["kind"] == "physics" or (ckpt is not None and ckpt.is_file())
        row: dict[str, Any] = {
            "id": spec["id"],
            "label": spec["label"],
            "kind": spec["kind"],
            "available": available,
            "checkpoint": str(ckpt) if ckpt is not None else None,
        }
        if ckpt is not None:
            banner = checkpoint_banner(ckpt)
            row["exists"] = banner["exists"]
            row["training"] = banner["training"]
            row["best_val_orbit_rms"] = banner.get("best_val_orbit_rms")
            row["epoch_completed"] = banner.get("epoch_completed")
            row["epochs_target"] = banner.get("epochs_target")
            row["arch"] = banner.get("arch")
            if ckpt.is_file() and ckpt.resolve() == default_ckpt.resolve():
                default_id = spec["id"]
        else:
            row["exists"] = True
            row["training"] = False
        rows.append(row)
    if default_id == "flexible":
        for row in rows:
            if row["available"] and row["kind"] == "learned":
                default_id = row["id"]
                break
    for row in rows:
        row["default"] = row["id"] == default_id
    return rows


def resolve_inference_model(model_id: str | None) -> dict[str, Any]:
    rows = list_inference_models()
    by_id = {row["id"]: row for row in rows}
    chosen = (model_id or "").strip() or next(row["id"] for row in rows if row.get("default"))
    if chosen not in by_id:
        raise ValueError(f"Unknown model {chosen!r}. Choose one of: {sorted(by_id)}")
    row = by_id[chosen]
    if not row["available"]:
        raise FileNotFoundError(f"Checkpoint for {row['label']} is not available: {row.get('checkpoint')}")
    return row


def _job_dir(job_id: str) -> Path:
    return JOBS_DIR / job_id


def _set_job(job_id: str, **fields: Any) -> None:
    with _lock:
        job = _jobs.setdefault(job_id, {"id": job_id, "log": []})
        log_line = fields.pop("log_line", None)
        job.update(fields)
        job["updated_at"] = time.time()
        if log_line:
            job.setdefault("log", []).append({"t": time.time(), "msg": log_line})
        _jobs[job_id] = job
        snap = dict(job)
    out = _job_dir(job_id)
    out.mkdir(parents=True, exist_ok=True)
    (out / "status.json").write_text(json.dumps(snap, default=str, indent=2))


def get_job(job_id: str) -> dict[str, Any] | None:
    with _lock:
        job = _jobs.get(job_id)
        if job is not None:
            return dict(job)
    path = _job_dir(job_id) / "status.json"
    if path.is_file():
        return json.loads(path.read_text())
    return None


def _score(pred: np.ndarray, gt: np.ndarray, *, duration_s: float) -> dict[str, Any]:
    n = min(len(pred), len(gt))
    orb = orbit_align(pred[:n], gt[:n])
    gt_cpa = float(np.min(np.sqrt(gt[:n, 0] ** 2 + gt[:n, 1] ** 2)))
    pred_cpa = float(np.min(np.sqrt(orb.aligned_pred[:, 0] ** 2 + orb.aligned_pred[:, 1] ** 2)))
    gt_len = float(polyline_arclength(gt[:n])[-1])
    pred_len = float(polyline_arclength(pred[:n])[-1])
    dur = max(float(duration_s), 1e-6)
    gt_v = gt_len / dur
    pred_v = pred_len / dur
    return {
        "orbit_rms_m": float(orb.rms),
        "orbit_len_norm": float(orb.length_normalized_rms),
        "cpa_gt_m": gt_cpa,
        "cpa_pred_m": pred_cpa,
        "cpa_abs_err_m": abs(pred_cpa - gt_cpa),
        "cpa_rel_err": abs(pred_cpa - gt_cpa) / max(gt_cpa, 1e-6),
        "speed_gt_mps": gt_v,
        "speed_pred_mps": pred_v,
        "speed_rel_err": abs(pred_v - gt_v) / max(gt_v, 1e-6),
        "path_length_gt_m": gt_len,
        "path_length_pred_m": pred_len,
        "orbit_reflected": bool(orb.reflected),
        "n_frames": int(n),
    }


def _write_overlay_png(
    path: Path,
    *,
    gt_xy: np.ndarray,
    pred_xy: np.ndarray,
    aligned: np.ndarray,
    title: str = "Predicted orbit vs drawn path",
) -> None:
    plt = _pyplot()
    fig, ax = plt.subplots(figsize=(6.2, 5.4))
    ax.plot(gt_xy[:, 0], gt_xy[:, 1], "--", color="#64748b", lw=2, label="drawn / GT")
    ax.plot(pred_xy[:, 0], pred_xy[:, 1], color="#94a3b8", lw=1.4, alpha=0.55, label="pred (raw)")
    ax.plot(aligned[:, 0], aligned[:, 1], color="#3dbb9a", lw=2.4, label="pred (orbit-aligned)")
    ax.scatter([0.0], [0.0], c="#e0a454", marker="x", s=80, zorder=5, label="mic")
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)


def _ensure_dopplersim() -> Path:
    root = dopplersim_root()
    if not (root / "doppler_sim").is_dir():
        raise RuntimeError(
            f"DopplerSim not found at {root}. Set DOPPLERSIM_ROOT to the DopplerSim repo."
        )
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root


def run_job(job_id: str, spec: dict[str, Any]) -> None:
    out = _job_dir(job_id)
    out.mkdir(parents=True, exist_ok=True)
    try:
        _set_job(
            job_id,
            stage="load_source",
            percent=8,
            message="Starting DopplerSim synthesis (first run can take several seconds)…",
            log_line="Importing DopplerSim path2d pipeline",
            running=True,
            error=None,
        )
        _ensure_dopplersim()
        import librosa
        import soundfile as sf
        from doppler_sim.application import (
            OUTPUT_SR,
            RenderParams,
            estimate_source_signature,
            synthesize_psd_noise,
        )
        from doppler_sim.batch.phase1_state import (
            PATH_TYPE_FREE_2D,
            export_phase1_package,
            mic_centric_state_2d,
        )
        from doppler_sim.path2d.synthesis import synthesize_path_audio

        _set_job(
            job_id,
            stage="load_source",
            percent=14,
            message="Loading source recording…",
            log_line="Loading source WAV",
        )
        source_path = Path(spec["source_path"])
        if not source_path.is_file():
            raise FileNotFoundError(f"source WAV not found: {source_path}")

        xy = np.asarray(spec["path_xy"], dtype=np.float64)
        if xy.ndim != 2 or xy.shape[0] < 2 or xy.shape[1] != 2:
            raise ValueError("Draw a path with at least two points.")

        v1 = float(spec["v1_mps"])
        v2 = float(spec["v2_mps"])
        h1 = float(spec["h1_m"])
        t_cpa1 = float(spec["t_cpa1_s"])
        mic = (float(spec["mic_x"]), float(spec["mic_y"]))
        vehicle_length = float(spec["vehicle_length_m"])
        num_emitters = int(spec["num_emitters"])

        audio, sr = librosa.load(str(source_path), sr=None, mono=True)
        if audio.size == 0:
            raise ValueError("Source WAV is empty.")

        _set_job(
            job_id,
            stage="invert",
            percent=22,
            message="Inverting Doppler / 1/R to recover the source spectrum…",
            log_line=f"Inverting source ({source_path.name}, {len(audio) / sr:.1f}s @ {sr} Hz)",
        )
        params = RenderParams(
            v1=v1,
            h1=h1,
            t_cpa1=t_cpa1,
            vehicle_length=vehicle_length,
            num_emitters=num_emitters,
            v2=v2,
            h2=10.0,
            t_cpa2=2.5,
            t_out=6.0,
        )
        freqs, _psd_obs, psd_inverted, _stft, _times = estimate_source_signature(
            np.asarray(audio, dtype=float), int(sr), params
        )

        _set_job(
            job_id,
            stage="synthesize",
            percent=48,
            message="Synthesizing pass-by audio along the drawn path (retarded time)…",
            log_line=f"Rendering path audio at {v2 * KMH_PER_MPS:.1f} km/h, {len(xy)} vertices",
        )
        result = synthesize_path_audio(
            xy,
            speed_mps=v2,
            sr=OUTPUT_SR,
            freqs=freqs,
            psd=psd_inverted,
            synthesize_psd_noise=synthesize_psd_noise,
            mic_xy=mic,
            vehicle_length=vehicle_length,
            num_emitters=num_emitters,
        )
        generated = np.asarray(result["audio"], dtype=np.float64)
        traj = result["trajectory"]
        wav_path = out / "generated.wav"
        sf.write(wav_path, generated, OUTPUT_SR, subtype="PCM_16")

        _set_job(
            job_id,
            stage="phase1",
            percent=68,
            message="Writing Phase 1 package (STFT + canonical path)…",
            log_line=f"Wrote {wav_path.name} ({len(generated) / OUTPUT_SR:.2f}s)",
            wav_ready=True,
        )
        timed_state = mic_centric_state_2d(
            traj["x"], traj["vx"], traj["y"], traj["vy"], mic_xy=mic
        )
        phase1_root = out / "phase1"
        export_phase1_package(
            phase1_root,
            speed_mps=v2,
            cpa_distance_m=float(result["cpa_distance_m"]),
            cpa_time_sec=float(result["cpa_time_sec"]),
            t_out_s=float(traj["duration_s"][0]),
            audio=generated,
            wav_sr=OUTPUT_SR,
            source="path2d_draw_eval",
            timed_t=traj["t"],
            timed_state=timed_state,
            path_type=PATH_TYPE_FREE_2D,
            polyline=xy,
            mic_world=np.array(mic, dtype=np.float64),
        )

        model_info = resolve_inference_model(spec.get("model"))
        _set_job(
            job_id,
            stage="predict",
            percent=82,
            message=f"Loading {model_info['label']}…",
            log_line=f"Inference model: {model_info['id']}",
        )
        banner = dict(model_info)
        sample = load_phase1_sample(phase1_root)
        bundle = to_inference_bundle(sample)
        if model_info["kind"] == "physics":
            _set_job(
                job_id,
                stage="predict",
                percent=90,
                message="Running physics freeform fit (audio only)…",
                log_line="Fitting flexible orbit from WAV/STFT",
            )
            if bundle.wav_path is not None:
                fit = fit_flexible_from_audio(wav_path=bundle.wav_path)
            else:
                fit = fit_flexible_from_audio(stft_db=bundle.stft_db)
            pred = fit.xy_pred
        else:
            ckpt = Path(model_info["checkpoint"])
            if not ckpt.is_file():
                raise FileNotFoundError(f"{model_info['label']} checkpoint missing: {ckpt}")
            model = load_orbit_model(ckpt)
            banner = checkpoint_banner(ckpt)
            banner["id"] = model_info["id"]
            banner["label"] = model_info["label"]
            _set_job(
                job_id,
                stage="predict",
                percent=90,
                message=f"Running {model_info['label']} on generated audio…",
                log_line="Predicting observer-centered orbit (audio/STFT only)",
            )
            pred = infer_learned_orbit(model, wav_path=bundle.wav_path, stft_db=bundle.stft_db)
        gt = _sample_target(sample)
        duration = float(sample.frame_times[-1] - sample.frame_times[0]) if len(sample.frame_times) else 1.0
        n = min(len(pred), len(gt))
        orb = orbit_align(pred[:n], gt[:n])
        metrics = _score(pred, gt, duration_s=duration)

        overlay = out / "overlay.png"
        _write_overlay_png(
            overlay,
            gt_xy=gt[:n],
            pred_xy=pred[:n],
            aligned=orb.aligned_pred,
            title=f"{model_info['label']} vs drawn path",
        )

        payload = {
            "gt_xy": gt[:n].tolist(),
            "pred_xy": pred[:n].tolist(),
            "aligned_xy": orb.aligned_pred.tolist(),
            "drawn_xy": xy.tolist(),
            "metrics": metrics,
            "model": model_info,
            "checkpoint": banner,
            "wav_name": wav_path.name,
            "duration_s": float(len(generated) / OUTPUT_SR),
            "cpa_distance_m": float(result["cpa_distance_m"]),
            "cpa_time_sec": float(result["cpa_time_sec"]),
            "source_name": source_path.name,
        }
        (out / "result.json").write_text(json.dumps(payload, indent=2))

        _set_job(
            job_id,
            stage="done",
            percent=100,
            message="Done. Predicted orbit is on the right.",
            log_line=f"Orbit RMS {metrics['orbit_rms_m']:.2f} m",
            running=False,
            result=payload,
        )
    except Exception as exc:
        _set_job(
            job_id,
            stage="error",
            percent=100,
            message=str(exc),
            log_line=f"Failed: {exc}",
            running=False,
            error=str(exc),
            traceback=traceback.format_exc(),
        )


def start_job(spec: dict[str, Any]) -> str:
    job_id = uuid.uuid4().hex[:16]
    _set_job(
        job_id,
        stage="queued",
        percent=2,
        message="Queued generate + predict…",
        log_line="Job queued",
        running=True,
        error=None,
    )
    thread = threading.Thread(target=run_job, args=(job_id, spec), daemon=True)
    thread.start()
    return job_id


def create_app():
    from flask import Flask, jsonify, request, send_file

    app = Flask(
        __name__,
        template_folder=str(WEB_DIR),
        static_folder=str(WEB_DIR / "static") if (WEB_DIR / "static").is_dir() else None,
    )
    app.config["MAX_CONTENT_LENGTH"] = 80 * 1024 * 1024

    @app.get("/")
    def index():
        html = WEB_DIR / "draw_eval.html"
        return send_file(html)

    @app.get("/api/bootstrap")
    def bootstrap():
        clips = list_source_clips()
        default_name = "RenaultCaptur_47.wav"
        default = next((c for c in clips if c["name"] == default_name), clips[0] if clips else None)
        models = list_inference_models()
        return jsonify(
            {
                "checkpoint": checkpoint_banner(),
                "models": models,
                "default_model": next((m["id"] for m in models if m.get("default")), "cnn"),
                "sources": clips,
                "default_source": default,
                "dopplersim_root": str(dopplersim_root()),
            }
        )

    @app.post("/api/jobs")
    def create_job():
        path_raw = request.form.get("path_json", "").strip()
        if not path_raw:
            return jsonify({"error": "Draw a path first."}), 400
        try:
            points = json.loads(path_raw)
            xy = [[float(p["x"]), float(p["y"])] for p in points]
        except Exception as exc:
            return jsonify({"error": f"Invalid path: {exc}"}), 400

        speed_unit = request.form.get("speed_unit", "kmph")
        v1 = float(request.form.get("v1", "47"))
        v2 = float(request.form.get("v2", "50"))
        if speed_unit == "kmph":
            v1_mps = v1 / KMH_PER_MPS
            v2_mps = v2 / KMH_PER_MPS
        else:
            v1_mps = v1
            v2_mps = v2

        try:
            model_info = resolve_inference_model(request.form.get("model"))
        except (ValueError, FileNotFoundError) as exc:
            return jsonify({"error": str(exc)}), 400

        uploads = JOBS_DIR / "_uploads"
        uploads.mkdir(parents=True, exist_ok=True)
        upload = request.files.get("audio_file")
        if upload and upload.filename:
            dest = uploads / f"{uuid.uuid4().hex}_{Path(upload.filename).name}"
            upload.save(dest)
            source_path = dest
        else:
            name = request.form.get("source_name", "RenaultCaptur_47.wav")
            source_path = default_inputs_dir() / name
            if not source_path.is_file():
                return jsonify({"error": f"Source clip not found: {name}"}), 400

        spec = {
            "path_xy": xy,
            "source_path": str(source_path),
            "v1_mps": v1_mps,
            "v2_mps": v2_mps,
            "h1_m": float(request.form.get("h1", "0.5")),
            "t_cpa1_s": float(request.form.get("t_cpa1", "5.28")),
            "mic_x": float(request.form.get("mic_x", "0")),
            "mic_y": float(request.form.get("mic_y", "0")),
            "vehicle_length_m": float(request.form.get("vehicle_length", "4.5")),
            "num_emitters": int(request.form.get("num_emitters", "1")),
            "model": model_info["id"],
            "checkpoint": model_info.get("checkpoint"),
        }
        job_id = start_job(spec)
        return jsonify({"id": job_id})

    @app.get("/api/jobs/<job_id>")
    def job_status(job_id: str):
        job = get_job(job_id)
        if job is None:
            return jsonify({"error": "unknown job"}), 404
        public = {k: v for k, v in job.items() if k != "traceback"}
        return jsonify(public)

    @app.get("/api/jobs/<job_id>/audio.wav")
    def job_audio(job_id: str):
        path = _job_dir(job_id) / "generated.wav"
        if not path.is_file():
            return jsonify({"error": "audio not ready"}), 404
        return send_file(path, mimetype="audio/wav", download_name="generated.wav")

    @app.get("/api/jobs/<job_id>/overlay.png")
    def job_overlay(job_id: str):
        path = _job_dir(job_id) / "overlay.png"
        if not path.is_file():
            return jsonify({"error": "overlay not ready"}), 404
        return send_file(path, mimetype="image/png")

    @app.get("/api/checkpoint")
    def api_checkpoint():
        model_id = request.args.get("model")
        if model_id:
            try:
                info = resolve_inference_model(model_id)
            except (ValueError, FileNotFoundError) as exc:
                return jsonify({"error": str(exc)}), 400
            if info.get("checkpoint"):
                banner = checkpoint_banner(Path(info["checkpoint"]))
                banner["id"] = info["id"]
                banner["label"] = info["label"]
                return jsonify(banner)
            return jsonify(info)
        return jsonify(checkpoint_banner())

    return app
