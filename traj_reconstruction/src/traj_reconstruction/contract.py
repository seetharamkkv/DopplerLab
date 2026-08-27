"""Audio-only Phase 1 contract for simulated trajectory orbit recovery.

Learning problem (DopplerSim Phase 1):

    A(1:T) → orbit[s(1:T)]

``A`` is the STFT magnitude map. ``s`` is mic-centric centerline state.
The identifiable object is the path **up to rotation (± reflection) about
the observer** — a rotational family, not a unique world-frame heading.

**Scope:** simulated DopplerSim exports only (straight + freehand path2d/path3d).

At inference the model may use WAV and/or ``spectrograms/stft.npy``.
It must not be given any file under ``INFERENCE_FORBIDDEN_RELPATHS``.
"""

from __future__ import annotations

# Spectrogram defaults matching DopplerSim Spectrogram Explorer / Phase 1.
SPEC_SR_HZ = 22050
STFT_N_FFT = 2048
STFT_HOP_LENGTH = 512
STFT_WINDOW = "hann"

STATE_COLUMNS_2D: tuple[str, ...] = ("x_m", "vx_mps", "y_m", "vy_mps")
STATE_COLUMNS_3D: tuple[str, ...] = (
    "x_m",
    "vx_mps",
    "y_m",
    "vy_mps",
    "z_m",
    "vz_mps",
)

PATH_TYPE_STRAIGHT = "straight"
PATH_TYPE_FREE_2D = "free_path_2d"
PATH_TYPE_FREE_3D = "free_path_3d"
PATH_TYPES: tuple[str, ...] = (
    PATH_TYPE_STRAIGHT,
    PATH_TYPE_FREE_2D,
    PATH_TYPE_FREE_3D,
)

# Programmatic freehand families for Tier-1 batch generation.
PATH_FAMILY_STRAIGHT = "straight"
PATH_FAMILY_ARC = "arc"
PATH_FAMILY_S_CURVE = "s_curve"
PATH_FAMILY_U_TURN = "u_turn"
PATH_FAMILY_MULTI_CPA = "multi_cpa"
PATH_FAMILIES: tuple[str, ...] = (
    PATH_FAMILY_STRAIGHT,
    PATH_FAMILY_ARC,
    PATH_FAMILY_S_CURVE,
    PATH_FAMILY_U_TURN,
    PATH_FAMILY_MULTI_CPA,
)

TIER1 = "tier1"

# Preferred supervised target for orbit models (gauge-fixed in sim export).
TRAINING_TARGET = "canonical_state_frames"  # alternative: "polar_state"

ACOUSTIC_PRIMARY_RELPATH = "spectrograms/stft.npy"
STATE_FRAMES_RELPATH = "metadata/state_frames.npy"
CANONICAL_STATE_FRAMES_RELPATH = "metadata/canonical_state_frames.npy"
FRAME_TIMES_RELPATH = "metadata/frame_times.npy"
SCHEMA_RELPATH = "metadata/phase1_schema.json"
PATH_POLYLINE_RELPATH = "metadata/path_polyline.npy"
POLAR_STATE_RELPATH = "metadata/polar_state.npy"

INFERENCE_ALLOWED_RELPATHS: tuple[str, ...] = (
    ACOUSTIC_PRIMARY_RELPATH,
    # WAV basename varies; loaders treat any *.wav in the sample root as allowed.
)

INFERENCE_FORBIDDEN_RELPATHS: tuple[str, ...] = (
    "metadata/state.npy",
    STATE_FRAMES_RELPATH,
    "metadata/state_times.npy",
    FRAME_TIMES_RELPATH,
    CANONICAL_STATE_FRAMES_RELPATH,
    POLAR_STATE_RELPATH,
    PATH_POLYLINE_RELPATH,
    "metadata/labels.npy",
    "metadata/simulation_parameters.json",
    SCHEMA_RELPATH,
    "metadata/cpa_time.npy",
    "metadata/cpa_distance_m.npy",
    "metadata/direction.npy",
    "metadata/speed_series_mps.npy",
    "metadata/range_m.npy",
    "metadata/radial_velocity_mps.npy",
    "metadata/acceleration_xy_mps2.npy",
    "metadata/kinematics.npy",
    "metadata/heading_rad.npy",
    "metadata/bearing_rad.npy",
    "velocity.npy",
    "cpa.npy",
)

DATA_SCOPE = "simulated_dopplersim_only"
