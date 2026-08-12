import tempfile
import unittest
from pathlib import Path

from trajplayer.trajectory_source import (
    TrajectorySelectionError,
    drop_requires_gro,
    paths_are_supported_for_drop,
    resolve_trajectory_source,
)


class TrajectorySourceTests(unittest.TestCase):
    def test_gro_and_xtc_can_be_selected_in_either_order(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            topology = root / "system.gro"
            trajectory = root / "run.xtc"
            topology.touch()
            trajectory.touch()

            source = resolve_trajectory_source((trajectory, topology))

            self.assertEqual(source.topology_path, topology.resolve())
            self.assertEqual(source.trajectory_path, trajectory.resolve())
            self.assertEqual(source.display_name, "system.gro + run.xtc")

    def test_xtc_alone_uses_same_named_gro(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            topology = root / "production.gro"
            trajectory = root / "production.xtc"
            topology.touch()
            trajectory.touch()

            source = resolve_trajectory_source((trajectory,))

            self.assertEqual(source.topology_path, topology.resolve())

    def test_trr_without_topology_explains_required_pair(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            trajectory = Path(tmp) / "run.trr"
            trajectory.touch()

            with self.assertRaisesRegex(TrajectorySelectionError, "GRO topology"):
                resolve_trajectory_source((trajectory,))

    def test_drop_accepts_one_regular_file_or_gromacs_pair(self) -> None:
        self.assertTrue(paths_are_supported_for_drop((Path("frame.gro"),)))
        self.assertTrue(paths_are_supported_for_drop((Path("frame.gro"), Path("run.xtc"))))
        self.assertFalse(paths_are_supported_for_drop((Path("run.dcd"),)))
        self.assertFalse(
            paths_are_supported_for_drop((Path("a.gro"), Path("b.xtc"), Path("c.trr")))
        )

    def test_drop_only_requests_gro_for_gromacs_trajectory_without_topology(self) -> None:
        self.assertTrue(drop_requires_gro((Path("run.xtc"),)))
        self.assertFalse(drop_requires_gro((Path("run.xtc"), Path("run.gro"))))
        self.assertFalse(drop_requires_gro((Path("a.xyz"), Path("b.xyz"))))


if __name__ == "__main__":
    unittest.main()
