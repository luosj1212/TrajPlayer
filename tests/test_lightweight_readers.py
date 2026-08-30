import builtins
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from ase import Atoms
from ase.io.trajectory import Trajectory

from trajplayer.ase_traj_reader import AseUlmTrajectoryReader, MAX_ULM_METADATA_BYTES
from trajplayer.gromacs_reader import ChemfilesGromacsReader
from trajplayer.random_access_cache import open_direct_random_access_store
from trajplayer.reader_common import normalize_symbol
from trajplayer.structure_reader import read_cif, read_gro, read_pdb, read_structure
from trajplayer.trajectory_source import TrajectorySource
from trajplayer.xyz_reader import read_xyz_frame


def _first_ulm_item(payload: bytearray) -> tuple[np.ndarray, int, dict[str, object]]:
    header = np.frombuffer(payload[24:48], dtype="<i8", count=3)
    offsets_position = int(header[2])
    frame_offset = int(
        np.frombuffer(payload[offsets_position : offsets_position + 8], dtype="<i8", count=1)[0]
    )
    metadata_size = int(
        np.frombuffer(payload[frame_offset : frame_offset + 8], dtype="<i8", count=1)[0]
    )
    metadata_start = frame_offset + 8
    metadata = json.loads(
        bytes(payload[metadata_start : metadata_start + metadata_size]).decode("utf-8")
    )
    return header, frame_offset, metadata


class LightweightReaderTests(unittest.TestCase):
    def test_xyz_rejects_impossible_atom_count_before_allocation(self) -> None:
        source = io.BytesIO(b"2000000000\n\n")
        with patch(
            "trajplayer.xyz_reader.np.empty",
            side_effect=AssertionError("allocation must not be reached"),
        ):
            with self.assertRaisesRegex(ValueError, "cannot fit"):
                read_xyz_frame(source, 2_000_000_000, 0)

    def test_gro_rejects_impossible_atom_count_before_allocation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "oversized.gro"
            path.write_text("oversized\n2000000000\n", encoding="utf-8")
            with patch(
                "trajplayer.structure_reader.np.empty",
                side_effect=AssertionError("allocation must not be reached"),
            ):
                with self.assertRaisesRegex(ValueError, "cannot fit"):
                    read_gro(path)

    def test_xyz_rejects_nonfinite_coordinates_and_out_of_range_numbers(self) -> None:
        with self.assertRaisesRegex(ValueError, "Invalid XYZ data at frame 0, atom 0"):
            read_xyz_frame(io.BytesIO(b"1\n\nH nan 0 0\n"), 1, 0)
        with self.assertRaisesRegex(ValueError, "Invalid XYZ data at frame 0, atom 0"):
            read_xyz_frame(
                io.BytesIO(b"1\nProperties=Z:I:1:pos:R:3\n70000 0 0 0\n"),
                1,
                0,
            )

    def test_structure_fallbacks_reject_nonfinite_coordinates(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            gro = Path(tmp) / "bad.gro"
            gro.write_text(
                "bad\n1\n"
                + f"{1:5d}{'LIG':<5s}{'C':>5s}{1:5d}{float('nan'):8.3f}{0.0:8.3f}{0.0:8.3f}\n"
                + "1 1 1\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "GRO coordinates are invalid"):
                read_gro(gro)

            pdb = Path(tmp) / "bad.pdb"
            pdb.write_text(
                "ATOM      1  C   LIG A   1         nan   0.000   0.000  1.00 20.00           C  \n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "PDB coordinates are invalid"):
                read_pdb(pdb)

            cif = Path(tmp) / "bad.cif"
            cif.write_text(
                "data_bad\nloop_\n_atom_site_type_symbol\n"
                "_atom_site_cartn_x\n_atom_site_cartn_y\n_atom_site_cartn_z\n"
                "C nan 0 0\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "Non-finite CIF numeric value"):
                read_cif(cif)

    def test_gro_box_requires_three_or_nine_finite_values(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad-box.gro"
            atom_line = f"{1:5d}{'LIG':<5s}{'C':>5s}{1:5d}{0.0:8.3f}{0.0:8.3f}{0.0:8.3f}\n"
            path.write_text(
                "bad box\n1\n" + atom_line + "1 1 1 0\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "exactly 3 or 9"):
                read_gro(path)

    def test_ase_reader_rejects_nonfinite_positions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "nonfinite.traj"
            with Trajectory(str(path), "w") as trajectory:
                trajectory.write(Atoms("H", positions=[[np.nan, 0.0, 0.0]]))
            with self.assertRaisesRegex(ValueError, "contains NaN or infinity"):
                AseUlmTrajectoryReader(path)

    def test_ase_reader_caps_metadata_before_json_allocation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "metadata-limit.traj"
            with Trajectory(str(path), "w") as trajectory:
                trajectory.write(Atoms("H", positions=[[0.0, 0.0, 0.0]]))
            payload = bytearray(path.read_bytes())
            _header, frame_offset, _metadata = _first_ulm_item(payload)
            payload[frame_offset : frame_offset + 8] = np.asarray(
                MAX_ULM_METADATA_BYTES + 1,
                dtype="<i8",
            ).tobytes()
            path.write_bytes(payload)

            with self.assertRaisesRegex(ValueError, "safety limit"):
                AseUlmTrajectoryReader(path)

    def test_ase_reader_rejects_atomic_numbers_before_uint16_narrowing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad-number.traj"
            with Trajectory(str(path), "w") as trajectory:
                trajectory.write(Atoms("H", positions=[[0.0, 0.0, 0.0]]))
            payload = bytearray(path.read_bytes())
            _header, _frame_offset, metadata = _first_ulm_item(payload)
            _shape, dtype_value, number_offset = metadata["numbers."]["ndarray"]
            number_dtype = np.dtype(str(dtype_value)).newbyteorder("<")
            encoded = np.asarray([70000], dtype=number_dtype).tobytes()
            start = int(number_offset)
            payload[start : start + len(encoded)] = encoded
            path.write_bytes(payload)

            with self.assertRaisesRegex(ValueError, "atomic numbers from 0 to 118"):
                AseUlmTrajectoryReader(path)

    def test_atom_label_suffixes_do_not_form_spurious_two_letter_elements(self) -> None:
        self.assertEqual(normalize_symbol("C00l"), "C")
        self.assertEqual(normalize_symbol("C_Cl"), "C")
        self.assertEqual(normalize_symbol("Cl1"), "Cl")
        self.assertEqual(normalize_symbol("00Cl"), "Cl")

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

    def test_extxyz_atom_labels_are_consistent_across_streamed_frames(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "labels.extxyz"
            path.write_text(
                "2\n"
                "Properties=species:S:1:pos:R:3\n"
                "C00l 0.0 0.0 0.0\n"
                "Cl1 1.0 0.0 0.0\n"
                "2\n"
                "Properties=species:S:1:pos:R:3\n"
                "C00l 0.1 0.0 0.0\n"
                "Cl1 1.1 0.0 0.0\n",
                encoding="utf-8",
            )
            store = open_direct_random_access_store(TrajectorySource(path))
            try:
                count = store.frame_count
                while not store.frame_count_is_final:
                    count, _complete = store.wait_for_index_update(count, timeout_s=1.0)
                np.testing.assert_array_equal(store.atom_numbers, [6, 17])
                positions = np.empty((2, 3), dtype=np.float32)
                store.read_frame_into(1, positions, None)
                np.testing.assert_allclose(positions, [[0.1, 0.0, 0.0], [1.1, 0.0, 0.0]])
            finally:
                store.close()

    def test_gro_atom_labels_do_not_join_letters_across_digits(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "labels.gro"
            atom_lines = [
                f"{1:5d}{'LIG':<5s}{'C00l':>5s}{1:5d}{0.100:8.3f}{0.0:8.3f}{0.0:8.3f}\n",
                f"{2:5d}{'CL':<5s}{'CL':>5s}{2:5d}{0.200:8.3f}{0.0:8.3f}{0.0:8.3f}\n",
            ]
            path.write_text(
                "labels\n    2\n" + "".join(atom_lines) + " 2.0 2.0 2.0\n",
                encoding="utf-8",
            )
            frame = read_structure(path)
            np.testing.assert_array_equal(frame.atom_numbers, [6, 17])

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

    def test_pdb_fallback_labels_respect_numeric_position_prefixes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "labels.pdb"
            path.write_text(
                "ATOM      1 1HG1 LIG A   1       1.000   2.000   3.000  1.00 20.00              \n"
                "ATOM      2 C00l LIG A   1       2.000   2.000   3.000  1.00 20.00              \n"
                "END\n",
                encoding="utf-8",
            )
            frame = read_structure(path)
            np.testing.assert_array_equal(frame.atom_numbers, [1, 6])

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
