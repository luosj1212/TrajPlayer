from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .random_access_cache import open_direct_random_access_store
from .trajectory_source import resolve_trajectory_source


SMOKE_CASES = (
    ("trajectory.traj",),
    ("trajectory.extxyz",),
    ("structure.gro",),
    ("structure.pdb",),
    ("structure.cif",),
    ("structure.gro", "trajectory.xtc"),
    ("structure.gro", "trajectory.trr"),
)


def run_reader_smoke(root: Path) -> dict[str, object]:
    fixture_root = root.resolve()
    results: list[dict[str, object]] = []
    for relative_paths in SMOKE_CASES:
        paths = tuple(fixture_root / name for name in relative_paths)
        missing = [path.name for path in paths if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"Reader smoke fixture is missing: {', '.join(missing)}")
        source = resolve_trajectory_source(paths)
        store = open_direct_random_access_store(source)
        try:
            first_positions, first_cell = store.read_frame_arrays(0)
            last_positions, _last_cell = store.read_frame_arrays(store.frame_count - 1)
            expected_shape = (store.atom_count, 3)
            if first_positions.shape != expected_shape or last_positions.shape != expected_shape:
                raise ValueError(f"Unexpected frame shape for {source.display_name}")
            if not np.all(np.isfinite(first_positions)) or not np.all(np.isfinite(last_positions)):
                raise ValueError(f"Non-finite coordinates in {source.display_name}")
            results.append(
                {
                    "source": source.display_name,
                    "frames": store.frame_count,
                    "atoms": store.atom_count,
                    "has_cell": first_cell is not None,
                }
            )
        finally:
            store.close()
    return {"passed": True, "cases": results}


def write_reader_smoke_report(report: dict[str, object], output_path: Path) -> None:
    path = output_path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
