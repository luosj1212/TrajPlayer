import builtins
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from ase import Atoms
from ase.io.trajectory import Trajectory

from trajplayer.ase_traj_reader import AseUlmTrajectoryReader
from trajplayer.gromacs_reader import ChemfilesGromacsReader
from trajplayer.random_access_cache import open_direct_random_access_store
from trajplayer.structure_reader import read_cif, read_gro, read_pdb
from trajplayer.trajectory_source import TrajectorySource


class LightweightReaderTests(unittest.TestCase):
    def test_ase_reader_reuses_the_frame_decoded_during_initialization(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "first-frame.traj"
            with Trajectory(str(path), "w") as trajectory:
                trajectory.write(Atoms("H", positions=[[1.0, 2.0, 3.0]]))

            reader = AseUlmTrajectoryReader(path)
            try:
                with patch.object(reader, "_read_item", wraps=reader._read_item) as decode:
                    first, _cell = reader.read_frame(0)
                    self.assertEqual(decode.call_count, 0)
                    np.testing.assert_allclose(first, [[1.0, 2.0, 3.0]])
                    reader.read_frame(0)
                    self.assertEqual(decode.call_count, 1)
            finally:
                reader.close()

    def test_ase_traj_reader_does_not_import_ase_io_or_scipy(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "two-frames.traj"
            with Trajectory(str(path), "w") as trajectory:
                trajectory.write(
                    Atoms("HC", positions=[[0, 0, 0], [1, 0, 0]], cell=[5, 6, 7], pbc=True)
                )
                trajectory.write(
                    Atoms("HC", positions=[[0.5, 0, 0], [1.5, 0, 0]], cell=[5, 6, 7], pbc=True)
                )

            original_import = builtins.__import__

            def guarded_import(name, *args, **kwargs):
                if name == "ase.io" or name.startswith("ase.io.") or name == "scipy" or name.startswith("scipy."):
                    raise AssertionError(f"unexpected heavyweight import: {name}")
                return original_import(name, *args, **kwargs)

            with patch("builtins.__import__", side_effect=guarded_import):
                store = open_direct_random_access_store(TrajectorySource(path))
                try:
                    self.assertEqual(store.frame_count, 2)
                    np.testing.assert_allclose(store.frame(1), [[0.5, 0, 0], [1.5, 0, 0]])
                    np.testing.assert_allclose(store.cell(1), np.diag([5, 6, 7]))
                finally:
                    store.close()

    def test_extxyz_supports_reordered_properties_and_atomic_numbers(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "reordered.extxyz"
            path.write_text(
                "2\n"
                'Properties=pos:R:3:Z:I:1:charge:R:1 Lattice="4 0 0 1 5 0 0 0 6" pbc="T T T"\n'
                "0.0 0.1 0.2 8 -0.2\n"
                "1.0 1.1 1.2 1 0.2\n",
                encoding="utf-8",
            )
            store = open_direct_random_access_store(TrajectorySource(path))
            try:
                np.testing.assert_array_equal(store.atom_numbers, [8, 1])
                np.testing.assert_allclose(store.frame(0), [[0, 0.1, 0.2], [1, 1.1, 1.2]])
                np.testing.assert_allclose(store.cell(0), [[4, 0, 0], [1, 5, 0], [0, 0, 6]])
            finally:
                store.close()

    def test_gro_reader_preserves_coordinates_elements_and_triclinic_cell(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "water.gro"
            atom_lines = [
                f"{1:5d}{'WAT':<5s}{'O':>5s}{1:5d}{0.100:8.3f}{0.200:8.3f}{0.300:8.3f}\n",
                f"{1:5d}{'WAT':<5s}{'H1':>5s}{2:5d}{0.190:8.3f}{0.200:8.3f}{0.300:8.3f}\n",
            ]
            path.write_text(
                "water\n    2\n"
                + "".join(atom_lines)
                + " 2.0 2.1 2.2 0.1 0.2 0.3 0.4 0.5 0.6\n",
                encoding="utf-8",
            )
            frame = read_gro(path)
            np.testing.assert_array_equal(frame.atom_numbers, [8, 1])
            np.testing.assert_allclose(frame.positions, [[1, 2, 3], [1.9, 2, 3]])
            np.testing.assert_allclose(
                frame.cell,
                [[20, 1, 2], [3, 21, 4], [5, 6, 22]],
            )

    def test_pdb_reader_uses_first_model_and_cryst1_cell(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.pdb"
            path.write_text(
                "CRYST1   10.000   11.000   12.000  90.00  90.00  90.00 P 1           1\n"
                "MODEL        1\n"
                "ATOM      1  CA  ALA A   1       1.000   2.000   3.000  1.00 20.00           C  \n"
                "HETATM    2  O   HOH A   2       4.000   5.000   6.000  1.00 20.00           O  \n"
                "ENDMDL\n"
                "MODEL        2\n"
                "ATOM      1  CA  ALA A   1       9.000   9.000   9.000  1.00 20.00           C  \n"
                "ENDMDL\n",
                encoding="utf-8",
            )
            frame = read_pdb(path)
            np.testing.assert_array_equal(frame.atom_numbers, [6, 8])
            np.testing.assert_allclose(frame.positions, [[1, 2, 3], [4, 5, 6]])
            np.testing.assert_allclose(frame.cell, np.diag([10, 11, 12]), atol=1.0e-5)

    def test_cif_reader_converts_fractional_coordinates(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "structure.cif"
            path.write_text(
                "data_test\n"
                "_cell_length_a 10\n"
                "_cell_length_b 12\n"
                "_cell_length_c 14\n"
                "_cell_angle_alpha 90\n"
                "_cell_angle_beta 90\n"
                "_cell_angle_gamma 90\n"
                "loop_\n"
                "_atom_site_label\n"
                "_atom_site_type_symbol\n"
                "_atom_site_fract_x\n"
                "_atom_site_fract_y\n"
                "_atom_site_fract_z\n"
                "C1 C 0.1 0.2 0.3\n"
                "O1 O 0.5 0.5 0.5\n",
                encoding="utf-8",
            )
            frame = read_cif(path)
            np.testing.assert_array_equal(frame.atom_numbers, [6, 8])
            np.testing.assert_allclose(frame.positions, [[1, 2.4, 4.2], [5, 6, 7]], atol=1.0e-5)


if __name__ == "__main__":
    unittest.main()
