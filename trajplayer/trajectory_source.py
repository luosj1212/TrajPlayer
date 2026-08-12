from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


GROMACS_TRAJECTORY_SUFFIXES = frozenset({".xtc", ".trr"})
GROMACS_TOPOLOGY_SUFFIX = ".gro"
SUPPORTED_TRAJECTORY_SUFFIXES = frozenset(
    {".traj", ".xyz", ".extxyz", ".pdb", ".cif", ".gro", ".xtc", ".trr"}
)


class TrajectorySelectionError(ValueError):
    pass


@dataclass(frozen=True)
class TrajectorySource:
    trajectory_path: Path
    topology_path: Path | None = None

    @property
    def is_gromacs_trajectory(self) -> bool:
        return self.trajectory_path.suffix.lower() in GROMACS_TRAJECTORY_SUFFIXES

    @property
    def paths(self) -> tuple[Path, ...]:
        if self.topology_path is None:
            return (self.trajectory_path,)
        return (self.topology_path, self.trajectory_path)

    @property
    def display_name(self) -> str:
        if self.topology_path is None:
            return self.trajectory_path.name
        return f"{self.topology_path.name} + {self.trajectory_path.name}"

    @property
    def tooltip(self) -> str:
        return "\n".join(str(path) for path in self.paths)


def resolve_trajectory_source(paths: Iterable[Path]) -> TrajectorySource:
    selected = _unique_paths(paths)
    if not selected:
        raise TrajectorySelectionError("No trajectory file was selected")
    if len(selected) > 2:
        raise TrajectorySelectionError("Select one trajectory, or one GRO topology with one XTC/TRR trajectory")

    unsupported = [path.name for path in selected if path.suffix.lower() not in SUPPORTED_TRAJECTORY_SUFFIXES]
    if unsupported:
        raise TrajectorySelectionError(f"Unsupported trajectory file: {', '.join(unsupported)}")

    if len(selected) == 1:
        path = selected[0]
        if path.suffix.lower() not in GROMACS_TRAJECTORY_SUFFIXES:
            return TrajectorySource(path)
        topology = _matching_gro(path)
        if topology is None:
            raise TrajectorySelectionError(
                f"{path.name} needs a GRO topology. Select or drop the .gro and {path.suffix.lower()} files together."
            )
        return TrajectorySource(path, topology)

    topologies = [path for path in selected if path.suffix.lower() == GROMACS_TOPOLOGY_SUFFIX]
    trajectories = [path for path in selected if path.suffix.lower() in GROMACS_TRAJECTORY_SUFFIXES]
    if len(topologies) != 1 or len(trajectories) != 1:
        raise TrajectorySelectionError("Select exactly one GRO topology and one XTC or TRR trajectory")
    return TrajectorySource(trajectories[0], topologies[0])


def paths_are_supported_for_drop(paths: Iterable[Path]) -> bool:
    selected = _unique_paths(paths)
    return bool(selected) and len(selected) <= 2 and all(
        path.suffix.lower() in SUPPORTED_TRAJECTORY_SUFFIXES for path in selected
    )


def drop_requires_gro(paths: Iterable[Path]) -> bool:
    selected = _unique_paths(paths)
    suffixes = {path.suffix.lower() for path in selected}
    return bool(suffixes & GROMACS_TRAJECTORY_SUFFIXES) and (
        GROMACS_TOPOLOGY_SUFFIX not in suffixes
    )


def _unique_paths(paths: Iterable[Path]) -> list[Path]:
    result: list[Path] = []
    seen: set[str] = set()
    for value in paths:
        path = Path(value).expanduser().resolve()
        key = str(path).casefold()
        if key not in seen:
            result.append(path)
            seen.add(key)
    return result


def _matching_gro(trajectory_path: Path) -> Path | None:
    exact = trajectory_path.with_suffix(GROMACS_TOPOLOGY_SUFFIX)
    if exact.exists():
        return exact.resolve()
    try:
        for candidate in trajectory_path.parent.iterdir():
            if (
                candidate.is_file()
                and candidate.suffix.lower() == GROMACS_TOPOLOGY_SUFFIX
                and candidate.stem.casefold() == trajectory_path.stem.casefold()
            ):
                return candidate.resolve()
    except OSError:
        return None
    return None
