#!/usr/bin/env bash
# VM one-shot: install package → run complex-STFT queue → sync artifacts to Drive.
#
# Prereqs on the VM:
#   1. git clone / pull DopplerLab
#   2. IDMT_Traffic available (audio/ + annotation/) — set IDMT_DATA_DIR if not default
#   3. Drive destination reachable — mount Shared Drive OR set IDMT_DRIVE_ROOT
#
# Example:
#   export IDMT_DRIVE_ROOT="/mnt/gdrive/Shareddrives/Spectral Transformers - Doppler/DopplerLab/cpx"
#   export IDMT_DATA_DIR="/data/IDMT_Traffic"
#   export IDMT_DEVICE=cuda          # or cpu
#   bash IDMT_experiments/run_vm.sh

set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

PYTHON="${IDMT_PYTHON:-python3}"
DRIVE_ROOT="${IDMT_DRIVE_ROOT:-}"
DEVICE="${IDMT_DEVICE:-cuda}"
SKIP_LAST="${IDMT_SYNC_SKIP_LAST_PT:-1}"

echo "============================================================"
echo "DopplerLab IDMT VM run"
echo "  ROOT       : $ROOT"
echo "  PYTHON     : $PYTHON"
echo "  DEVICE     : $DEVICE"
echo "  DRIVE_ROOT : ${DRIVE_ROOT:-"(not set — will only train locally)"}"
echo "============================================================"

# Install package from src/
"$PYTHON" -m pip install -q -e "$ROOT/src"

# Prefer writing checkpoints straight onto Drive when mounted
if [[ -n "$DRIVE_ROOT" ]]; then
  mkdir -p "$DRIVE_ROOT/checkpoints" "$DRIVE_ROOT/outputs"
  export IDMT_CHECKPOINT_DIR="$DRIVE_ROOT/checkpoints"
  export IDMT_OUTPUT_DIR="$DRIVE_ROOT/outputs"
  echo "Writing checkpoints → $IDMT_CHECKPOINT_DIR"
  echo "Writing outputs     → $IDMT_OUTPUT_DIR"
fi

export IDMT_DEVICE="$DEVICE"
export PYTHONUNBUFFERED=1

# Queue uses sys.executable when Windows venv path is missing
"$PYTHON" "$ROOT/run_complex_stft_queue.py"

# If training wrote under local dirs (no drive mount), copy out now
if [[ -n "$DRIVE_ROOT" ]]; then
  SYNC_ARGS=(--drive-root "$DRIVE_ROOT")
  if [[ "$SKIP_LAST" == "1" ]]; then
    SYNC_ARGS+=(--skip-last-pt)
  fi
  # Only needed when local dirs were used; harmless if already on Drive
  "$PYTHON" "$ROOT/sync_artifacts_to_drive.py" "${SYNC_ARGS[@]}" \
    --checkpoints "${IDMT_CHECKPOINT_DIR:-$ROOT/checkpoints}" \
    --outputs "${IDMT_OUTPUT_DIR:-$ROOT/outputs}"
fi

echo "VM run finished."
