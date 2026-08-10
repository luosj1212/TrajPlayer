import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from ase import Atoms
from ase.io import write
from chemfiles import Frame, Trajectory, UnitCell

from trajplayer.ase_cache import build_cache_from_source, open_valid_cache
from trajplayer.gromacs_cache import inspect_gromacs_source
from trajplayer.gromacs_reader import ChemfilesGromacsReader
from trajplayer.random_access_cache import open_random_access_session, write_reader_frame
from trajplayer.trajectory_source import resolve_trajectory_source


class GromacsCacheTests(unittest.TestCase):
    def test_gromacs_reader_reuses_the_frame_decoded_during_initialization(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            topology = root / "water.gro"
            trajectory = root / "water.xtc"
            self._write_gromacs_pair(topology, trajectory)

            reader = ChemfilesGromacsReader(trajectory, expected_atom_count=3)
            try:
                with patch.object(
                    reader,
                    "_decode_frame",
                    wraps=reader._decode_frame,
                ) as decode:
                    reader.read_frame(0)
                    self.assertEqual(decode.call_count, 0)
                    reader.read_frame(0)
                    self.assertEqual(decode.call_count, 1)
            finally:
                reader.close()

    def test_xtc_and_trr_stream_into_float32_store(self) -> None:
        for suffix in (".xtc", ".trr"):
            with self.subTest(suffix=suffix), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                topology = root / "water.gro"
                trajectory = root / f"water{suffix}"
                expected = self._write_gromacs_pair(topology, trajectory)
                source = resolve_trajectory_source((trajectory, topology))

                summary = inspect_gromacs_source(source)
                self.assertEqual(summary.frame_count, 4)
                self.assertEqual(summary.atom_count, 3)
                np.testing.assert_array_equal(summary.atom_numbers, np.array([8, 1, 1]))

                progress: list[tuple[int, int]] = []
                with build_cache_from_source(
                    source,
                    progress_callback=lambda done, total: progress.append((done, total)),
                ) as store:
                    self.assertEqual(store.positions.dtype, np.float32)
                    self.assertEqual(store.positions.shape, (4, 3, 3))
                    self.assertTrue(store.has_cells)
                    np.testing.assert_allclose(store.frame(3), expected[3], atol=0.02)
                    np.testing.assert_allclose(store.cell(0), np.diag([20.0, 21.0, 22.0]), atol=1.0e-4)
                    self.assertTrue(store.is_valid_for_sources(source.paths))
                self.assertEqual(progress[-1], (4, 4))

                with open_valid_cache(source) as reopened:
                    self.assertEqual(reopened.frame_count, 4)

                topology.write_text(topology.read_text(encoding="utf-8") + "\n", encoding="utf-8")
                with self.assertRaises(FileNotFoundError):
                    open_valid_cache(source)

    def test_xtc_random_access_can_publish_a_distant_frame_first(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            topology = root / "water.gro"
            trajectory = root / "water.xtc"
            expected = self._write_gromacs_pair(topology, trajectory)
            source = resolve_trajectory_source((trajectory, topology))

            reader, store = open_random_access_session(source)
            try:
                write_reader_frame(reader, store, 3)
                self.assertTrue(store.is_frame_available(3))
                self.assertFalse(store.is_frame_available(0))
                np.testing.assert_allclose(store.frame(3), expected[3], atol=0.02)
            finally:
                reader.close()
                store.close()

    @staticmethod
    def _write_gromacs_pair(topology: Path, trajectory: Path) -> np.ndarray:
        initial = np.array(
            [[1.0, 2.0, 3.0], [1.9, 2.0, 3.0], [0.7, 2.8, 3.0]],
            dtype=np.float32,
        )
        write(
            topology,
            Atoms("OH2", positions=initial, cell=[20.0, 21.0, 22.0], pbc=True),
            format="gromacs",
        )
        frames = np.empty((4, 3, 3), dtype=np.float32)
        writer = Trajectory(str(trajectory), mode="w")
        try:
            for frame_index in range(4):
                frames[frame_index] = initial + np.float32(frame_index * 0.25)
                frame = Frame()
                frame.resize(3)
                frame.positions[:] = frames[frame_index]
                frame.cell = UnitCell([20.0, 21.0, 22.0])
                writer.write(frame)
        finally:
            writer.close()
        return frames


if __name__ == "__main__":
    unittest.main()
