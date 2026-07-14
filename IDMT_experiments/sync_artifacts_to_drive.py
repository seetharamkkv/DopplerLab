#!/usr/bin/env python3
"""Copy checkpoints + outputs to a Drive folder (or any remote path).

Heavy ``*.pt`` weights are gitignored — use this after training on a VM so results
land on Google Drive (mounted path, rclone remote, or shared network folder).

Examples
--------
# Drive for desktop / gdrivefuse / rclone mount on the VM:
python sync_artifacts_to_drive.py \\
  --drive-root "/mnt/gdrive/Shareddrives/Spectral Transformers - Doppler/DopplerLab/cpx"

# Only sync finished transfer runs (skip huge last.pt to save space):
python sync_artifacts_to_drive.py --drive-root "$DRIVE" --skip-last-pt

# Env form (used by run_vm.sh):
export IDMT_DRIVE_ROOT=...
python sync_artifacts_to_drive.py
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DEFAULT_DRIVE = os.environ.get(
    "IDMT_DRIVE_ROOT",
    "/content/drive/Shareddrives/Spectral Transformers - Doppler/DopplerLab/cpx",
)


def _copy_tree(src: Path, dst: Path, *, skip_last_pt: bool) -> tuple[int, int]:
    """Copy files under src → dst. Returns (n_files, n_bytes)."""
    n_files = 0
    n_bytes = 0
    if not src.is_dir():
        print(f"  skip (missing): {src}")
        return 0, 0
    for path in src.rglob("*"):
        if not path.is_file():
            continue
        if skip_last_pt and path.name == "last.pt":
            continue
        rel = path.relative_to(src)
        out = dst / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, out)
        n_files += 1
        n_bytes += path.stat().st_size
    return n_files, n_bytes


def main() -> None:
    p = argparse.ArgumentParser(description="Sync IDMT checkpoints/outputs to Drive")
    p.add_argument(
        "--drive-root",
        type=Path,
        default=Path(DEFAULT_DRIVE),
        help="Destination root (Shared Drive cpx folder, or any path)",
    )
    p.add_argument(
        "--checkpoints",
        type=Path,
        default=ROOT / "checkpoints",
        help="Local checkpoints dir",
    )
    p.add_argument(
        "--outputs",
        type=Path,
        default=ROOT / "outputs",
        help="Local outputs dir",
    )
    p.add_argument(
        "--skip-last-pt",
        action="store_true",
        help="Omit last.pt (keeps best.pt + JSON metrics; much smaller)",
    )
    p.add_argument(
        "--only",
        choices=["checkpoints", "outputs", "all"],
        default="all",
    )
    args = p.parse_args()

    drive = args.drive_root.expanduser().resolve()
    drive.mkdir(parents=True, exist_ok=True)
    print(f"Drive root: {drive}")
    print(f"skip last.pt: {args.skip_last_pt}")

    total_f = total_b = 0
    jobs: list[tuple[str, Path, Path]] = []
    if args.only in ("checkpoints", "all"):
        jobs.append(("checkpoints", args.checkpoints, drive / "checkpoints"))
    if args.only in ("outputs", "all"):
        jobs.append(("outputs", args.outputs, drive / "outputs"))

    for label, src, dst in jobs:
        print(f"\n→ {label}: {src}  →  {dst}")
        n_f, n_b = _copy_tree(src, dst, skip_last_pt=args.skip_last_pt)
        print(f"  copied {n_f} files ({n_b / 1e6:.1f} MB)")
        total_f += n_f
        total_b += n_b

    print(f"\nDone: {total_f} files, {total_b / 1e6:.1f} MB → {drive}")
    print("On your laptop/Colab: open the same Shared Drive folder (or sync via Drive for desktop).")


if __name__ == "__main__":
    main()
    sys.exit(0)
