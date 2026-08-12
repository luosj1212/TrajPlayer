from __future__ import annotations

import json
import os
from pathlib import Path

from PySide6.QtCore import QSettings

from trajplayer.trajectory_source import (
    TrajectorySelectionError,
    TrajectorySource,
    resolve_trajectory_source,
)


class RecentFiles:
    SETTINGS_KEY = "recent/trajectory_sources"

    def __init__(self, settings: QSettings, *, limit: int = 10) -> None:
        self.settings = settings
        self.limit = max(1, int(limit))

    def record(self, source: TrajectorySource) -> None:
        paths = [str(path.resolve()) for path in source.paths]
        source_key = _entry_key(paths)
        entries = [entry for entry in self._read() if _entry_key(entry) != source_key]
        self._write([paths, *entries][: self.limit])

    def sources(self) -> tuple[TrajectorySource, ...]:
        valid_entries: list[list[str]] = []
        sources: list[TrajectorySource] = []
        seen: set[tuple[str, ...]] = set()
        for entry in self._read():
            paths = [Path(value) for value in entry]
            if not paths or any(not path.exists() for path in paths):
                continue
            try:
                source = resolve_trajectory_source(paths)
            except TrajectorySelectionError:
                continue
            normalized = [str(path) for path in source.paths]
            key = _entry_key(normalized)
            if key in seen:
                continue
            seen.add(key)
            valid_entries.append(normalized)
            sources.append(source)
        valid_entries = valid_entries[: self.limit]
        if valid_entries != self._read():
            self._write(valid_entries)
        return tuple(sources[: self.limit])

    def clear(self) -> None:
        self.settings.remove(self.SETTINGS_KEY)

    def _read(self) -> list[list[str]]:
        raw = str(self.settings.value(self.SETTINGS_KEY, "[]") or "[]")
        try:
            values = json.loads(raw)
        except (TypeError, ValueError):
            return []
        if not isinstance(values, list):
            return []
        return [
            [str(path) for path in entry]
            for entry in values
            if isinstance(entry, list) and all(isinstance(path, str) for path in entry)
        ]

    def _write(self, entries: list[list[str]]) -> None:
        self.settings.setValue(self.SETTINGS_KEY, json.dumps(entries, ensure_ascii=False))


def _entry_key(entry: list[str]) -> tuple[str, ...]:
    return tuple(os.path.normcase(str(Path(path).expanduser().resolve())) for path in entry)
