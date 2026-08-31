#!/usr/bin/env python3
"""Side-by-side 2D whiteboard generate + orbit predict.

Uses DopplerSim path2d synthesis (sibling repo) and a selectable inference
model (2D CNN, ridge 1D CNN, MLP, or physics freeform).

  .venv/bin/python traj_reconstruction/scripts/serve_draw_eval.py

Open http://127.0.0.1:5055
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

os.environ.setdefault("MPLCONFIGDIR", str(Path("/tmp") / "matplotlib-dopplerlab"))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=5055)
    p.add_argument("--dopplersim", type=Path, default=None)
    args = p.parse_args()
    if args.dopplersim is not None:
        os.environ["DOPPLERSIM_ROOT"] = str(args.dopplersim.resolve())

    from traj_reconstruction.draw_eval import create_app, dopplersim_root

    print(f"DopplerSim: {dopplersim_root()}")
    print(f"Open http://{args.host}:{args.port}")
    app = create_app()
    app.run(host=args.host, port=args.port, debug=False, use_reloader=False, threaded=True)


if __name__ == "__main__":
    main()
