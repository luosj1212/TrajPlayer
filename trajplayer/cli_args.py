from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CliArgs:
    paths: list[Path]
    startup_smoke: bool = False
    smoke_exit_ms: int | None = None
    benchmark_output: Path | None = None
    benchmark_root: Path | None = None
    benchmark_atoms: int = 100_000
    benchmark_frames: int = 64
    benchmark_render_frames: int = 120
    benchmark_finish_gpu: bool = False
    benchmark_bonds: bool = False
    benchmark_mode: str = "ball_stick"


def parse_cli_args(argv: list[str]) -> CliArgs:
    paths: list[Path] = []
    startup_smoke = False
    smoke_exit_ms: int | None = None
    benchmark_output: Path | None = None
    benchmark_root: Path | None = None
    benchmark_atoms = 100_000
    benchmark_frames = 64
    benchmark_render_frames = 120
    benchmark_finish_gpu = False
    benchmark_bonds = False
    benchmark_mode = "ball_stick"

    for arg in argv:
        if arg == "--startup-smoke":
            startup_smoke = True
        elif arg.startswith("--smoke-exit-ms="):
            smoke_exit_ms = max(0, int(arg.split("=", 1)[1]))
        elif arg.startswith("--benchmark-output="):
            benchmark_output = Path(arg.split("=", 1)[1])
        elif arg.startswith("--benchmark-root="):
            benchmark_root = Path(arg.split("=", 1)[1])
        elif arg.startswith("--benchmark-atoms="):
            benchmark_atoms = int(arg.split("=", 1)[1])
        elif arg.startswith("--benchmark-frames="):
            benchmark_frames = int(arg.split("=", 1)[1])
        elif arg.startswith("--benchmark-render-frames="):
            benchmark_render_frames = int(arg.split("=", 1)[1])
        elif arg == "--benchmark-finish-gpu":
            benchmark_finish_gpu = True
        elif arg == "--benchmark-bonds":
            benchmark_bonds = True
        elif arg.startswith("--benchmark-mode="):
            benchmark_mode = arg.split("=", 1)[1].strip().lower()
        else:
            paths.append(Path(arg).expanduser())

    return CliArgs(
        paths=paths,
        startup_smoke=startup_smoke,
        smoke_exit_ms=smoke_exit_ms,
        benchmark_output=benchmark_output,
        benchmark_root=benchmark_root,
        benchmark_atoms=benchmark_atoms,
        benchmark_frames=benchmark_frames,
        benchmark_render_frames=benchmark_render_frames,
        benchmark_finish_gpu=benchmark_finish_gpu,
        benchmark_bonds=benchmark_bonds,
        benchmark_mode=benchmark_mode,
    )
