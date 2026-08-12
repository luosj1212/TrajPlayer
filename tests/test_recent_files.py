from __future__ import annotations

from PySide6.QtCore import QSettings

from trajplayer.trajectory_source import TrajectorySource
from trajplayer.ui.recent_files import RecentFiles


def test_recent_files_preserve_gromacs_pair_and_remove_missing(tmp_path) -> None:
    settings = QSettings(str(tmp_path / "settings.ini"), QSettings.Format.IniFormat)
    recent = RecentFiles(settings, limit=10)
    gro = tmp_path / "md.gro"
    xtc = tmp_path / "md.xtc"
    gro.write_text("topology", encoding="ascii")
    xtc.write_bytes(b"trajectory")
    source = TrajectorySource(xtc, gro)
    recent.record(source)
    recent.record(source)

    sources = recent.sources()
    assert len(sources) == 1
    assert sources[0].paths == (gro.resolve(), xtc.resolve())

    xtc.unlink()
    assert recent.sources() == ()
