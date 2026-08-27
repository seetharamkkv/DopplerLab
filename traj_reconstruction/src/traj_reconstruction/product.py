"""User-facing rotational-family orbit product (Phase 5).

From monaural audio only, recover a trajectory about the observer and expose
the **full rotational (± mirror) family** — never a fake absolute heading.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np

from traj_reconstruction.flexible import (
    OrbitMLP,
    fit_flexible_from_audio,
    infer_orbit_mlp,
)
from traj_reconstruction.frontend import extract_ridges
from traj_reconstruction.orbit import orbit_family, xy_from_state
from traj_reconstruction.parametric import fit_orbit_from_audio


MethodName = Literal["flexible", "parametric", "mlp"]


@dataclass
class OrbitProduct:
    """Recovered observer-centered trajectory orbit (product payload)."""

    xy: np.ndarray  # (T, 2) meters, mic at origin
    frame_times: np.ndarray  # (T,)
    r: np.ndarray  # (T,)
    theta_rel: np.ndarray  # (T,) radians
    mirror_ambiguous: bool = True
    scale_ambiguous: bool = False
    heading_absolute: bool = False
    confidence: float = 0.0  # [0, 1] residual-based
    residual_rms_f: float | None = None
    method: str = "flexible"
    disclaimer: str = (
        "Absolute world heading is not determined. "
        "All rotations about the observer (± mirror) are equally valid."
    )
    meta: dict[str, Any] = field(default_factory=dict)

    def rotated(self, angle_rad: float, *, mirror: bool = False) -> np.ndarray:
        """Return XY for one member of the rotational family."""
        xy = np.asarray(self.xy, dtype=np.float64)
        if mirror:
            xy = xy.copy()
            xy[:, 0] *= -1.0
        c, s = math.cos(angle_rad), math.sin(angle_rad)
        return np.column_stack(
            [c * xy[:, 0] - s * xy[:, 1], s * xy[:, 0] + c * xy[:, 1]]
        )

    def family(self, n_rotations: int = 36, *, include_mirror: bool = True) -> list[np.ndarray]:
        return orbit_family(self.xy, n_rotations=n_rotations, include_mirror=include_mirror)

    def to_dict(self) -> dict[str, Any]:
        return {
            "xy": np.asarray(self.xy, dtype=float).tolist(),
            "frame_times": np.asarray(self.frame_times, dtype=float).tolist(),
            "r": np.asarray(self.r, dtype=float).tolist(),
            "theta_rel": np.asarray(self.theta_rel, dtype=float).tolist(),
            "mirror_ambiguous": self.mirror_ambiguous,
            "scale_ambiguous": self.scale_ambiguous,
            "heading_absolute": self.heading_absolute,
            "confidence": float(self.confidence),
            "residual_rms_f": self.residual_rms_f,
            "method": self.method,
            "disclaimer": self.disclaimer,
            "meta": self.meta,
            "data_scope": "simulated_demo_ok_real_deferred",
        }

    def save_json(self, path: Path | str) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2))
        return path


def _xy_to_polar(xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    xy = np.asarray(xy, dtype=np.float64)
    r = np.sqrt(xy[:, 0] ** 2 + xy[:, 1] ** 2)
    theta = np.arctan2(xy[:, 1], xy[:, 0])
    return r, theta


def _confidence_from_residual(residual_rms_f: float | None) -> float:
    if residual_rms_f is None or not np.isfinite(residual_rms_f):
        return 0.5
    # Map ~0–80 Hz residual into ~1–0 confidence.
    return float(np.clip(1.0 - float(residual_rms_f) / 80.0, 0.05, 0.99))


def predict_orbit(
    *,
    wav_path: Path | str | None = None,
    stft_db: np.ndarray | None = None,
    audio: np.ndarray | None = None,
    sr: int | None = None,
    method: MethodName = "flexible",
    mlp_checkpoint: Path | str | None = None,
    scale_ambiguous: bool = False,
) -> OrbitProduct:
    """Audio-only prediction → OrbitProduct (no metadata)."""
    residual = None
    times = None
    meta: dict[str, Any] = {}

    if method == "mlp":
        if mlp_checkpoint is None:
            raise ValueError("mlp method requires mlp_checkpoint")
        model = OrbitMLP.load(mlp_checkpoint)
        if stft_db is not None:
            xy = infer_orbit_mlp(model, stft_db=stft_db)
            times = np.arange(len(xy), dtype=np.float64)
        elif wav_path is not None:
            feats = extract_ridges(wav_path=wav_path, n_harmonics=1)
            xy = infer_orbit_mlp(model, stft_db=feats.stft_db)
            times = feats.frame_times
            residual = None
        else:
            raise ValueError("provide wav_path or stft_db for mlp")
        meta["checkpoint"] = str(mlp_checkpoint)
    elif method == "parametric":
        fit = fit_orbit_from_audio(
            wav_path=wav_path,
            stft_db=stft_db,
            audio=audio,
            sr=sr,
            family="straight",
            use_amplitude=True,
        )
        xy = fit.xy_pred
        times = fit.frame_times
        residual = fit.residual_rms_f
        meta["params"] = fit.params
    else:
        fit = fit_flexible_from_audio(
            wav_path=wav_path,
            stft_db=stft_db,
            audio=audio,
            sr=sr,
            n_modes=3,
        )
        xy = fit.xy_pred
        times = fit.frame_times
        residual = fit.residual_rms_f
        meta["params"] = {k: v for k, v in fit.params.items() if not str(k).startswith("c")}

    r, theta = _xy_to_polar(xy)
    return OrbitProduct(
        xy=np.asarray(xy, dtype=np.float64),
        frame_times=np.asarray(times, dtype=np.float64),
        r=r,
        theta_rel=theta,
        mirror_ambiguous=True,
        scale_ambiguous=bool(scale_ambiguous),
        heading_absolute=False,
        confidence=_confidence_from_residual(residual),
        residual_rms_f=float(residual) if residual is not None else None,
        method=method,
        meta=meta,
    )


def render_orbit_png(
    product: OrbitProduct,
    out_path: Path | str,
    *,
    angle_rad: float = 0.0,
    mirror: bool = False,
    title: str | None = None,
) -> Path:
    """Static figure for papers (one family member)."""
    import matplotlib.pyplot as plt

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    xy = product.rotated(angle_rad, mirror=mirror)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(xy[:, 0], xy[:, 1], lw=2, label="recovered path")
    ax.scatter([0.0], [0.0], c="red", marker="x", s=80, label="observer")
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(title or "Trajectory orbit about the observer")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.25)
    fig.text(
        0.5,
        0.01,
        product.disclaimer,
        ha="center",
        fontsize=8,
        style="italic",
        wrap=True,
    )
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return out_path


def write_orbit_viewer_html(
    product: OrbitProduct,
    out_path: Path | str,
    *,
    title: str = "DopplerLab — Trajectory Orbit",
) -> Path:
    """Standalone HTML viewer with rotation slider + mirror toggle."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(product.to_dict())
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>{title}</title>
  <style>
    :root {{
      --bg: #0f1419;
      --panel: #1a2332;
      --ink: #e7eef7;
      --muted: #9db0c7;
      --accent: #3dbb9a;
      --warn: #e0a454;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0; font-family: "IBM Plex Sans", "Segoe UI", sans-serif;
      background: radial-gradient(1200px 600px at 20% 0%, #1b2a3d, var(--bg));
      color: var(--ink); min-height: 100vh;
    }}
    main {{ max-width: 960px; margin: 0 auto; padding: 1.5rem; }}
    h1 {{ font-size: 1.35rem; font-weight: 600; margin: 0 0 0.35rem; }}
    .sub {{ color: var(--muted); font-size: 0.95rem; margin-bottom: 1rem; }}
    .disclaimer {{
      background: #243247; border-left: 3px solid var(--warn);
      padding: 0.75rem 1rem; border-radius: 6px; color: #f0e2c8;
      font-size: 0.9rem; margin-bottom: 1rem;
    }}
    .flags {{ display: flex; flex-wrap: wrap; gap: 0.5rem; margin-bottom: 1rem; }}
    .flag {{
      font-size: 0.75rem; letter-spacing: 0.04em; text-transform: uppercase;
      padding: 0.3rem 0.55rem; border-radius: 999px; background: #2a3b52; color: var(--muted);
    }}
    .flag.on {{ background: #2d4a3f; color: var(--accent); }}
    .panel {{
      background: var(--panel); border-radius: 12px; padding: 1rem;
      box-shadow: 0 8px 30px rgba(0,0,0,0.25);
    }}
    canvas {{ width: 100%; height: auto; background: #0c1118; border-radius: 8px; display: block; }}
    .controls {{ display: grid; gap: 0.85rem; margin-top: 1rem; }}
    label {{ display: flex; flex-direction: column; gap: 0.35rem; font-size: 0.85rem; color: var(--muted); }}
    input[type=range] {{ width: 100%; }}
    .row {{ display: flex; align-items: center; gap: 0.75rem; }}
    button {{
      background: var(--accent); color: #06241c; border: 0; border-radius: 8px;
      padding: 0.55rem 0.9rem; font-weight: 600; cursor: pointer;
    }}
    .meta {{ margin-top: 0.75rem; font-size: 0.8rem; color: var(--muted); }}
  </style>
</head>
<body>
<main>
  <h1>{title}</h1>
  <p class="sub">Monaural audio → observer-centered trajectory <strong>orbit</strong> (simulated demo path OK).</p>
  <div class="disclaimer" id="disclaimer"></div>
  <div class="flags">
    <span class="flag" id="flag-mirror">mirror ambiguous</span>
    <span class="flag" id="flag-scale">scale ambiguous</span>
    <span class="flag" id="flag-heading">no absolute heading</span>
    <span class="flag" id="flag-conf">confidence</span>
  </div>
  <div class="panel">
    <canvas id="orbit" width="900" height="700"></canvas>
    <div class="controls">
      <label>Rotation about observer
        <input id="rot" type="range" min="0" max="360" value="0"/>
        <span id="rot-val">0°</span>
      </label>
      <div class="row">
        <label class="row" style="flex-direction:row; align-items:center;">
          <input id="mirror" type="checkbox"/> Mirror family
        </label>
        <button type="button" id="spin">Animate family</button>
      </div>
      <label>Time scrub (vehicle marker)
        <input id="scrub" type="range" min="0" max="100" value="0"/>
      </label>
    </div>
    <div class="meta" id="meta"></div>
  </div>
</main>
<script type="application/json" id="payload">{payload}</script>
<script>
const data = JSON.parse(document.getElementById('payload').textContent);
document.getElementById('disclaimer').textContent = data.disclaimer;
const fm = document.getElementById('flag-mirror');
const fs = document.getElementById('flag-scale');
const fh = document.getElementById('flag-heading');
const fc = document.getElementById('flag-conf');
if (data.mirror_ambiguous) fm.classList.add('on');
if (data.scale_ambiguous) fs.classList.add('on');
if (!data.heading_absolute) fh.classList.add('on');
fc.textContent = 'confidence ' + (100 * (data.confidence || 0)).toFixed(0) + '%';
fc.classList.add('on');
document.getElementById('meta').textContent =
  'method=' + data.method +
  (data.residual_rms_f != null ? (' · f residual RMS=' + data.residual_rms_f.toFixed(2) + ' Hz') : '');

const canvas = document.getElementById('orbit');
const ctx = canvas.getContext('2d');
const rot = document.getElementById('rot');
const rotVal = document.getElementById('rot-val');
const mirror = document.getElementById('mirror');
const scrub = document.getElementById('scrub');

function transform(xy, angDeg, mir) {{
  const a = angDeg * Math.PI / 180;
  const c = Math.cos(a), s = Math.sin(a);
  return xy.map(p => {{
    let x = mir ? -p[0] : p[0];
    let y = p[1];
    return [c*x - s*y, s*x + c*y];
  }});
}}

function draw() {{
  const ang = +rot.value;
  rotVal.textContent = ang + '°';
  const pts = transform(data.xy, ang, mirror.checked);
  const pad = 40;
  let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
  for (const p of pts) {{
    minX = Math.min(minX, p[0]); maxX = Math.max(maxX, p[0]);
    minY = Math.min(minY, p[1]); maxY = Math.max(maxY, p[1]);
  }}
  minX = Math.min(minX, -1); maxX = Math.max(maxX, 1);
  minY = Math.min(minY, -1); maxY = Math.max(maxY, 1);
  const span = Math.max(maxX - minX, maxY - minY) * 1.15;
  const cx = (minX + maxX) / 2, cy = (minY + maxY) / 2;
  const scale = (Math.min(canvas.width, canvas.height) - 2*pad) / span;
  const toPx = (x, y) => [
    canvas.width/2 + (x - cx) * scale,
    canvas.height/2 - (y - cy) * scale
  ];

  ctx.clearRect(0, 0, canvas.width, canvas.height);
  // grid
  ctx.strokeStyle = '#1e2a3a'; ctx.lineWidth = 1;
  for (let g = -5; g <= 5; g++) {{
    const [x0, yA] = toPx(cx + g * span/10, cy - span/2);
    const [x1, yB] = toPx(cx + g * span/10, cy + span/2);
    ctx.beginPath(); ctx.moveTo(x0, yA); ctx.lineTo(x1, yB); ctx.stroke();
    const [xA, y0] = toPx(cx - span/2, cy + g * span/10);
    const [xB, y1] = toPx(cx + span/2, cy + g * span/10);
    ctx.beginPath(); ctx.moveTo(xA, y0); ctx.lineTo(xB, y1); ctx.stroke();
  }}
  // path
  ctx.strokeStyle = '#3dbb9a'; ctx.lineWidth = 3; ctx.beginPath();
  pts.forEach((p, i) => {{
    const [X, Y] = toPx(p[0], p[1]);
    if (i === 0) ctx.moveTo(X, Y); else ctx.lineTo(X, Y);
  }});
  ctx.stroke();
  // observer
  const [ox, oy] = toPx(0, 0);
  ctx.fillStyle = '#e0a454';
  ctx.beginPath(); ctx.arc(ox, oy, 6, 0, Math.PI*2); ctx.fill();
  ctx.fillStyle = '#e7eef7'; ctx.font = '14px sans-serif';
  ctx.fillText('observer', ox + 10, oy - 10);
  // vehicle scrub
  const i = Math.min(pts.length - 1, Math.floor((+scrub.value / 100) * (pts.length - 1)));
  const [vx, vy] = toPx(pts[i][0], pts[i][1]);
  ctx.fillStyle = '#7ec8ff';
  ctx.beginPath(); ctx.arc(vx, vy, 7, 0, Math.PI*2); ctx.fill();
}}

rot.addEventListener('input', draw);
mirror.addEventListener('change', draw);
scrub.addEventListener('input', draw);
let anim = null;
document.getElementById('spin').addEventListener('click', () => {{
  if (anim) {{ cancelAnimationFrame(anim); anim = null; return; }}
  const tick = () => {{
    rot.value = (+rot.value + 1) % 361;
    draw();
    anim = requestAnimationFrame(tick);
  }};
  anim = requestAnimationFrame(tick);
}});
draw();
</script>
</body>
</html>
"""
    out_path.write_text(html)
    return out_path


def export_orbit_product(
    product: OrbitProduct,
    out_dir: Path | str,
    *,
    stem: str = "orbit_product",
) -> dict[str, Path]:
    """Write JSON + static PNG + interactive HTML."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "json": product.save_json(out_dir / f"{stem}.json"),
        "png": render_orbit_png(product, out_dir / f"{stem}.png"),
        "html": write_orbit_viewer_html(product, out_dir / f"{stem}.html"),
    }
    return paths
