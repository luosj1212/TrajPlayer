from __future__ import annotations

import json
import mmap
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator

import numpy as np

from .binary_store import SourceIdentity


FRAME_OFFSETS_FILE = "frame_offsets.u64"
FRAME_INDEX_METADATA_FILE = "frame_index.json"
INDEX_CHUNK_BYTES = 16 * 1024 * 1024
INDEX_CHECKPOINT_BYTES = 64 * 1024 * 1024
INDEX_CHECKPOINT_INTERVAL_S = 0.25
INDEX_BACKGROUND_THROTTLE_S = 0.001
INDEX_FOREGROUND_RESUME_GRACE_S = 0.15
INDEX_METADATA_VERSION = 2
UINT64_BYTES = np.dtype(np.uint64).itemsize


@dataclass(frozen=True, slots=True)
class IndexIoStats:
    foreground_reads: int
    pause_count: int
    paused_seconds: float
    foreground_active: int


class IndexIoCoordinator:
    """Give trajectory reads priority over the optional background index scan."""

    def __init__(
        self,
        *,
        resume_grace_s: float = INDEX_FOREGROUND_RESUME_GRACE_S,
        background_throttle_s: float = INDEX_BACKGROUND_THROTTLE_S,
    ) -> None:
        self._condition = threading.Condition()
        self._resume_grace_s = max(0.0, float(resume_grace_s))
        self._background_throttle_s = max(0.0, float(background_throttle_s))
        self._foreground_active = 0
        self._foreground_reads = 0
        self._resume_not_before_s = 0.0
        self._pause_count = 0
        self._paused_seconds = 0.0

    @contextmanager
    def foreground(self) -> Iterator[None]:
        with self._condition:
            self._foreground_active += 1
            self._foreground_reads += 1
            self._condition.notify_all()
        try:
            yield
        finally:
            with self._condition:
                self._foreground_active = max(0, self._foreground_active - 1)
                self._resume_not_before_s = max(
                    self._resume_not_before_s,
                    time.monotonic() + self._resume_grace_s,
                )
                self._condition.notify_all()

    def wait_for_background_turn(self, stop_event: threading.Event) -> bool:
        pause_started_s: float | None = None
        with self._condition:
            while True:
                if stop_event.is_set():
                    self._finish_pause(pause_started_s)
                    return False
                now_s = time.monotonic()
                wait_s = max(0.0, self._resume_not_before_s - now_s)
                blocked = self._foreground_active > 0 or wait_s > 0.0
                if not blocked:
                    self._finish_pause(pause_started_s)
                    break
                if pause_started_s is None:
                    pause_started_s = now_s
                    self._pause_count += 1
                self._condition.wait(timeout=max(0.001, min(0.05, wait_s or 0.05)))

        if self._background_throttle_s > 0.0:
            return not stop_event.wait(self._background_throttle_s)
        return not stop_event.is_set()

    def wake(self) -> None:
        with self._condition:
            self._condition.notify_all()

    def stats_snapshot(self) -> IndexIoStats:
        with self._condition:
            return IndexIoStats(
                foreground_reads=self._foreground_reads,
                pause_count=self._pause_count,
                paused_seconds=self._paused_seconds,
                foreground_active=self._foreground_active,
            )

    def _finish_pause(self, pause_started_s: float | None) -> None:
        if pause_started_s is not None:
            self._paused_seconds += max(0.0, time.monotonic() - pause_started_s)


class ProgressiveXyzIndex:
    """Disk-backed XYZ frame offsets published while a background scan runs."""

    def __init__(
        self,
        source: Path,
        cache_root: Path,
        *,
        atom_count: int,
        progress_callback: Callable[[int, bool], None] | None = None,
        io_coordinator: IndexIoCoordinator | None = None,
    ) -> None:
        self.source = source.resolve()
        self.cache_root = cache_root.resolve()
        self.cache_root.mkdir(parents=True, exist_ok=True)
        self.offsets_path = self.cache_root / FRAME_OFFSETS_FILE
        self.metadata_path = self.cache_root / FRAME_INDEX_METADATA_FILE
        self.atom_count = int(atom_count)
        self.io_coordinator = io_coordinator or IndexIoCoordinator()
        self._progress_callback = progress_callback
        self._condition = threading.Condition()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._error: BaseException | None = None
        self._closed = False
        source_stat = self.source.stat()
        self._source_identity = SourceIdentity.from_path(
            self.source,
            source_stat.st_mtime_ns,
            source_stat.st_size,
        )

        metadata = self._valid_metadata()
        if metadata is None:
            self.offsets_path.write_bytes(np.uint64(0).tobytes())
            self._known_frame_count = 1
            self._complete = False
            self._scan_offset = 0
            self._scan_line_count = 0
            self._write_metadata(
                complete=False,
                frame_count=1,
                scan_offset=0,
                line_count=0,
            )
        else:
            self._known_frame_count = int(metadata["frame_count"])
            self._complete = bool(metadata["complete"])
            self._scan_offset = int(metadata["scan_offset"])
            self._scan_line_count = int(metadata["line_count"])
            expected_size = self._known_frame_count * UINT64_BYTES
            if self.offsets_path.stat().st_size > expected_size:
                with self.offsets_path.open("r+b") as offsets:
                    offsets.truncate(expected_size)
        self._offset_handle = self.offsets_path.open("rb", buffering=0)

    @property
    def known_frame_count(self) -> int:
        with self._condition:
            return self._known_frame_count

    @property
    def complete(self) -> bool:
        with self._condition:
            return self._complete

    @property
    def scan_offset(self) -> int:
        with self._condition:
            return self._scan_offset

    @property
    def error(self) -> BaseException | None:
        with self._condition:
            return self._error

    @property
    def io_stats(self) -> IndexIoStats:
        return self.io_coordinator.stats_snapshot()

    def start(self) -> None:
        with self._condition:
            if self._complete or self._closed:
                return
            if self._thread is not None and self._thread.is_alive():
                return
            self._thread = threading.Thread(
                target=self._scan,
                name="ProgressiveXyzIndex",
                daemon=True,
            )
            self._thread.start()

    def offset(self, frame_index: int) -> int:
        index = int(frame_index)
        with self._condition:
            self._raise_if_failed()
            if index < 0 or index >= self._known_frame_count:
                raise IndexError(index)
            self._offset_handle.seek(index * UINT64_BYTES)
            raw = self._offset_handle.read(UINT64_BYTES)
        if len(raw) != UINT64_BYTES:
            raise RuntimeError("XYZ frame index ended unexpectedly")
        return int(np.frombuffer(raw, dtype=np.uint64, count=1)[0])

    def wait_for_update(self, previous_count: int, *, timeout_s: float) -> tuple[int, bool]:
        deadline = time.monotonic() + max(0.0, float(timeout_s))
        with self._condition:
            while (
                not self._closed
                and not self._complete
                and self._error is None
                and self._known_frame_count <= int(previous_count)
            ):
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    break
                self._condition.wait(timeout=remaining)
            self._raise_if_failed()
            return self._known_frame_count, self._complete

    def wait_until_complete(self, *, stop_event: threading.Event | None = None) -> int:
        with self._condition:
            while not self._complete and self._error is None and not self._closed:
                if stop_event is not None and stop_event.is_set():
                    raise RuntimeError("XYZ indexing was cancelled")
                self._condition.wait(timeout=0.05)
            self._raise_if_failed()
            return self._known_frame_count

    def close(self) -> None:
        with self._condition:
            if self._closed:
                return
            self._closed = True
            self._stop_event.set()
            self._condition.notify_all()
            thread = self._thread
        self.io_coordinator.wake()
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=2.0)
        self._offset_handle.close()

    def _scan(self) -> None:
        try:
            self._scan_offsets()
        except Exception as exc:
            with self._condition:
                self._error = exc
                self._condition.notify_all()

    def _scan_offsets(self) -> None:
        file_size = self._source_identity.size
        if file_size <= 0:
            raise ValueError("No frames found in trajectory")
        lines_per_frame = self.atom_count + 2
        frame_count = self._known_frame_count
        scan_offset = self._scan_offset
        line_count = self._scan_line_count
        checkpoint_offset = scan_offset
        last_checkpoint_s = time.monotonic()

        with self.source.open("rb") as source_handle, self.offsets_path.open(
            "ab", buffering=1024 * 1024
        ) as output:
            with mmap.mmap(source_handle.fileno(), length=0, access=mmap.ACCESS_READ) as mapped:
                while scan_offset < file_size:
                    if not self.io_coordinator.wait_for_background_turn(self._stop_event):
                        self._checkpoint(
                            output,
                            frame_count=frame_count,
                            scan_offset=scan_offset,
                            line_count=line_count,
                            complete=False,
                        )
                        return
                    chunk_size = min(INDEX_CHUNK_BYTES, file_size - scan_offset)
                    chunk = np.frombuffer(
                        mapped,
                        dtype=np.uint8,
                        count=chunk_size,
                        offset=scan_offset,
                    )
                    newlines = np.flatnonzero(chunk == 10)
                    first_boundary = lines_per_frame - 1 - (line_count % lines_per_frame)
                    if first_boundary < newlines.size:
                        boundaries = newlines[first_boundary::lines_per_frame]
                        offsets = boundaries.astype(np.uint64, copy=True)
                        offsets += np.uint64(scan_offset + 1)
                        offsets = offsets[offsets < file_size]
                        if offsets.size:
                            offsets.tofile(output)
                            frame_count += int(offsets.size)
                    line_count += int(newlines.size)
                    scan_offset += chunk_size
                    del newlines
                    del chunk

                    now_s = time.monotonic()
                    if (
                        scan_offset - checkpoint_offset >= INDEX_CHECKPOINT_BYTES
                        or now_s - last_checkpoint_s >= INDEX_CHECKPOINT_INTERVAL_S
                    ):
                        self._checkpoint(
                            output,
                            frame_count=frame_count,
                            scan_offset=scan_offset,
                            line_count=line_count,
                            complete=False,
                        )
                        checkpoint_offset = scan_offset
                        last_checkpoint_s = now_s

        if line_count % lines_per_frame != 0:
            raise ValueError("XYZ trajectory ended in an incomplete frame")
        expected_size = frame_count * UINT64_BYTES
        if self.offsets_path.stat().st_size != expected_size:
            raise RuntimeError("XYZ frame index size is inconsistent")
        self._checkpoint(
            None,
            frame_count=frame_count,
            scan_offset=file_size,
            line_count=line_count,
            complete=True,
        )

    def _checkpoint(
        self,
        output,
        *,
        frame_count: int,
        scan_offset: int,
        line_count: int,
        complete: bool,
    ) -> None:
        if output is not None:
            output.flush()
        self._write_metadata(
            complete=complete,
            frame_count=frame_count,
            scan_offset=scan_offset,
            line_count=line_count,
        )
        with self._condition:
            self._known_frame_count = int(frame_count)
            self._scan_offset = int(scan_offset)
            self._scan_line_count = int(line_count)
            self._complete = bool(complete)
            self._condition.notify_all()
        self._notify_progress(frame_count, complete)

    def _notify_progress(self, frame_count: int, complete: bool) -> None:
        callback = self._progress_callback
        if callback is not None:
            callback(int(frame_count), bool(complete))

    def _valid_metadata(self) -> dict[str, object] | None:
        if not self.metadata_path.exists() or not self.offsets_path.exists():
            return None
        try:
            metadata = json.loads(self.metadata_path.read_text(encoding="utf-8"))
            stored = SourceIdentity.from_json(metadata["source"])
            frame_count = int(metadata["frame_count"])
            scan_offset = int(metadata["scan_offset"])
            line_count = int(metadata["line_count"])
            complete = bool(metadata.get("complete"))
            offsets_size = self.offsets_path.stat().st_size
            if (
                int(metadata.get("version", -1)) != INDEX_METADATA_VERSION
                or stored != self._source_identity
                or int(metadata.get("atom_count", -1)) != self.atom_count
                or frame_count <= 0
                or offsets_size < frame_count * UINT64_BYTES
                or scan_offset < 0
                or scan_offset > self._source_identity.size
                or line_count < 0
                or (complete and scan_offset != self._source_identity.size)
            ):
                return None
            return metadata
        except Exception:
            return None

    def _write_metadata(
        self,
        *,
        complete: bool,
        frame_count: int,
        scan_offset: int,
        line_count: int,
    ) -> None:
        current_stat = self.source.stat()
        current_identity = SourceIdentity.from_path(
            self.source,
            current_stat.st_mtime_ns,
            current_stat.st_size,
        )
        if current_identity != self._source_identity:
            raise RuntimeError("XYZ trajectory changed while its index was being built")
        temporary = self.metadata_path.with_suffix(f"{self.metadata_path.suffix}.tmp")
        temporary.write_text(
            json.dumps(
                {
                    "version": INDEX_METADATA_VERSION,
                    "source": self._source_identity.to_json(),
                    "frame_count": int(frame_count),
                    "atom_count": self.atom_count,
                    "complete": bool(complete),
                    "scan_offset": int(scan_offset),
                    "line_count": int(line_count),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        temporary.replace(self.metadata_path)

    def _raise_if_failed(self) -> None:
        if self._error is not None:
            raise RuntimeError(f"XYZ indexing failed: {self._error}") from self._error
