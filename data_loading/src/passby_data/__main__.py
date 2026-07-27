from .cli import make_lovo_main, make_split_main, make_spread_scene_main, verify_main
import sys

if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    rest = sys.argv[2:]
    if cmd in ("make-split", "make_split"):
        raise SystemExit(make_split_main(rest))
    if cmd in ("make-lovo", "lovo"):
        raise SystemExit(make_lovo_main(rest))
    if cmd in ("make-spread", "spread"):
        raise SystemExit(make_spread_scene_main(rest))
    if cmd in ("verify", "verify-leakage"):
        raise SystemExit(verify_main(rest))
    print(
        "Usage:\n"
        "  python -m passby_data make-split [--min-speed-gap 2] ...\n"
        "  python -m passby_data make-lovo ...\n"
        "  python -m passby_data make-spread [--min-speed-gap 2] ...\n"
        "  python -m passby_data verify --split path/to/split.json",
        file=sys.stderr,
    )
    raise SystemExit(2)
