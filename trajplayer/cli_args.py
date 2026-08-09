from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CliArgs:
    paths: list[Path]
    startup_smoke: bool = False
    native_smoke: bool = False
    smoke_exit_ms: int | None = None
    gui_smoke: bool = False
    gui_smoke_output: Path | None = None
    gui_smoke_timeout_ms: int = 15_000
    reader_smoke: Path | None = None
    reader_smoke_output: Path | None = None
    doctor_output: Path | None = None
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
    native_smoke = False
    smoke_exit_ms: int | None = None
    gui_smoke = False
    gui_smoke_output: Path | None = None
    gui_smoke_timeout_ms = 15_000
    reader_smoke: Path | None = None
    reader_smoke_output: Path | None = None
    doctor_output: Path | None = None
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
        elif arg == "--native-smoke":
            native_smoke = True
        elif arg == "--gui-smoke":
            gui_smoke = True
        elif arg.startswith("--gui-smoke-output="):
            gui_smoke_output = Path(arg.split("=", 1)[1])
        elif arg.startswith("--gui-smoke-timeout-ms="):
            gui_smoke_timeout_ms = max(1_000, int(arg.split("=", 1)[1]))
        elif arg.startswith("--reader-smoke="):
            reader_smoke = Path(arg.split("=", 1)[1])
        elif arg.startswith("--reader-smoke-output="):
            reader_smoke_output = Path(arg.split("=", 1)[1])
        elif arg.startswith("--doctor-output="):
            doctor_output = Path(arg.split("=", 1)[1])
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
        native_smoke=native_smoke,
        smoke_exit_ms=smoke_exit_ms,
        gui_smoke=gui_smoke,
        gui_smoke_output=gui_smoke_output,
        gui_smoke_timeout_ms=gui_smoke_timeout_ms,
        reader_smoke=reader_smoke,
        reader_smoke_output=reader_smoke_output,
        doctor_output=doctor_output,
        benchmark_output=benchmark_output,
        benchmark_root=benchmark_root,
        benchmark_atoms=benchmark_atoms,
        benchmark_frames=benchmark_frames,
        benchmark_render_frames=benchmark_render_frames,
        benchmark_finish_gpu=benchmark_finish_gpu,
        benchmark_bonds=benchmark_bonds,
        benchmark_mode=benchmark_mode,
    )
