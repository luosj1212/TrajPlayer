from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.io.trajectory import Trajectory as AseTrajectory
from chemfiles import Frame, Trajectory, UnitCell


def create_smoke_fixtures(output: Path) -> Path:
    root = output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    base_positions = np.asarray(
        [[1.0, 2.0, 3.0], [1.9, 2.0, 3.0], [0.7, 2.8, 3.0]],
        dtype=np.float32,
    )
    cell = np.diag([20.0, 21.0, 22.0]).astype(np.float32)

    with AseTrajectory(str(root / "trajectory.traj"), "w") as trajectory:
        for frame_index in range(2):
            trajectory.write(
                Atoms(
                    "OH2",
                    positions=base_positions + np.float32(frame_index * 0.1),
                    cell=cell,
                    pbc=True,
                )
            )

    extxyz_frames = []
    for frame_index in range(2):
        positions = base_positions + np.float32(frame_index * 0.1)
        rows = "\n".join(
            f"{symbol} {position[0]:.6f} {position[1]:.6f} {position[2]:.6f}"
            for symbol, position in zip(("O", "H", "H"), positions)
        )
        extxyz_frames.append(
            "3\n"
            'Lattice="20 0 0 0 21 0 0 0 22" Properties=species:S:1:pos:R:3 pbc="T T T"\n'
            f"{rows}\n"
        )
    (root / "trajectory.extxyz").write_text("".join(extxyz_frames), encoding="utf-8")

    gro_lines = ["smoke fixture\n", "    3\n"]
    for atom_index, (symbol, position) in enumerate(
        zip(("O", "H1", "H2"), base_positions),
        start=1,
    ):
        scaled = position / np.float32(10.0)
        gro_lines.append(
            f"{1:5d}{'WAT':<5s}{symbol:>5s}{atom_index:5d}"
            f"{scaled[0]:8.3f}{scaled[1]:8.3f}{scaled[2]:8.3f}\n"
        )
    gro_lines.append("   2.00000   2.10000   2.20000\n")
    (root / "structure.gro").write_text("".join(gro_lines), encoding="utf-8")

    for suffix in ("xtc", "trr"):
        writer = Trajectory(str(root / f"trajectory.{suffix}"), mode="w")
        try:
            for frame_index in range(2):
                frame = Frame()
                frame.resize(3)
                frame.positions[:] = base_positions + np.float32(frame_index * 0.1)
                frame.cell = UnitCell([20.0, 21.0, 22.0])
                writer.write(frame)
        finally:
            writer.close()

    (root / "structure.pdb").write_text(
        "CRYST1   20.000   21.000   22.000  90.00  90.00  90.00 P 1           1\n"
        "ATOM      1  O   HOH A   1       1.000   2.000   3.000  1.00 20.00           O  \n"
        "ATOM      2  H1  HOH A   1       1.900   2.000   3.000  1.00 20.00           H  \n"
        "ATOM      3  H2  HOH A   1       0.700   2.800   3.000  1.00 20.00           H  \n"
        "END\n",
        encoding="utf-8",
    )
    (root / "structure.cif").write_text(
        "data_smoke\n"
        "_cell_length_a 20\n"
        "_cell_length_b 21\n"
        "_cell_length_c 22\n"
        "_cell_angle_alpha 90\n"
        "_cell_angle_beta 90\n"
        "_cell_angle_gamma 90\n"
        "loop_\n"
        "_atom_site_label\n"
        "_atom_site_type_symbol\n"
        "_atom_site_Cartn_x\n"
        "_atom_site_Cartn_y\n"
        "_atom_site_Cartn_z\n"
        "O1 O 1.0 2.0 3.0\n"
        "H1 H 1.9 2.0 3.0\n"
        "H2 H 0.7 2.8 3.0\n",
        encoding="utf-8",
    )
    return root


def main() -> None:
    parser = argparse.ArgumentParser(description="Create portable reader smoke-test fixtures")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(create_smoke_fixtures(args.output))


if __name__ == "__main__":
    main()
