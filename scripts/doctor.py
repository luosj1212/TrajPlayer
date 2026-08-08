from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from trajplayer.diagnostics import diagnostics_json, probe_opengl
from trajplayer.startup import prefer_high_performance_gpu


def main() -> None:
    parser = argparse.ArgumentParser(description="Print privacy-safe TrajPlayer diagnostics")
    parser.add_argument("--output", type=Path, help="Also write the diagnostics JSON to this file")
    args = parser.parse_args()

    prefer_high_performance_gpu()
    report = diagnostics_json(opengl=probe_opengl())
    print(report)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
