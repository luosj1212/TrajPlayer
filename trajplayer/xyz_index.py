from __future__ import annotations

import json
import mmap
import threading
import time
from pathlib import Path
from typing import Callable

import numpy as np

from .binary_store import SourceIdentity


FRAME_OFFSETS_FILE = "frame_offsets.u64"
FRAME_INDEX_METADATA_FILE = "frame_index.json"
INDEX_CHUNK_BYTES = 16 * 1024 * 1024


class ProgressiveXyzIndex:
    """Disk-backed XYZ frame offsets published while a background scan runs."""

    def __init__(
        self,
        source: Path,
        cache_root: Path,
        *,
        atom_count: int,
        progress_callback: Callable[[int, bool], None] | None = None,
    ) -> None:
        self.source = source.resolve()
        self.cache_root = cache_root.resolve()
        self.cache_root.mkdir(parents=True, exist_ok=True)
        self.offsets_path = self.cache_root / FRAME_OFFSETS_FILE
        self.metadata_path = self.cache_root / FRAME_INDEX_METADATA_FILE
        self.atom_count = int(atom_count)
        self._progress_callback = progress_callback
        self._condition = threading.Condition()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._error: BaseException | None = None
        self._closed = False

        metadata = self._valid_complete_metadata()
        if metadata is not None:
            self._known_frame_count = int(metadata["frame_count"])
            self._complete = True
        else:
            self.offsets_path.write_bytes(np.uint64(0).tobytes())
            self._known_frame_count = 1
            self._complete = False
            self._write_metadata(complete=False, frame_count=1)
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
    def error(self) -> BaseException | None:
        with self._condition:
            return self._error

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
            self._offset_handle.seek(index * np.dtype(np.uint64).itemsize)
            raw = self._offset_handle.read(np.dtype(np.uint64).itemsize)
        if len(raw) != np.dtype(np.uint64).itemsize:
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
        file_size = self.source.stat().st_size
        if file_size <= 0:
            raise ValueError("No frames found in trajectory")
        lines_per_frame = self.atom_count + 2
        frame_count = 1
        last_callback_s = 0.0

        with self.source.open("rb") as source_handle, self.offsets_path.open("ab") as output:
            with mmap.mmap(source_handle.fileno(), length=0, access=mmap.ACCESS_READ) as mapped:
                line_count = 0
                for chunk_start in range(0, file_size, INDEX_CHUNK_BYTES):
                    if self._stop_event.is_set():
                        return
                    chunk_size = min(INDEX_CHUNK_BYTES, file_size - chunk_start)
                    chunk = np.frombuffer(
                        mapped,
                        dtype=np.uint8,
                        count=chunk_size,
                        offset=chunk_start,
                    )
                    newlines = np.flatnonzero(chunk == 10)
                    first_boundary = lines_per_frame - 1 - (line_count % lines_per_frame)
                    added = 0
                    if first_boundary < newlines.size:
                        boundaries = newlines[first_boundary::lines_per_frame]
                        offsets = boundaries.astype(np.uint64, copy=True)
                        offsets += np.uint64(chunk_start + 1)
                        offsets = offsets[offsets < file_size]
                        if offsets.size:
                            offsets.tofile(output)
                            output.flush()
                            added = int(offsets.size)
                            frame_count += added
                    line_count += int(newlines.size)
                    del newlines
                    del chunk

                    if added:
                        with self._condition:
                            self._known_frame_count = frame_count
                            self._condition.notify_all()
                        now_s = time.monotonic()
                        if now_s - last_callback_s >= 0.05:
                            last_callback_s = now_s
                            self._notify_progress(frame_count, False)

        expected_size = frame_count * np.dtype(np.uint64).itemsize
        if self.offsets_path.stat().st_size != expected_size:
            raise RuntimeError("XYZ frame index size is inconsistent")
        self._write_metadata(complete=True, frame_count=frame_count)
        with self._condition:
            self._known_frame_count = frame_count
            self._complete = True
            self._condition.notify_all()
        self._notify_progress(frame_count, True)

    def _notify_progress(self, frame_count: int, complete: bool) -> None:
        callback = self._progress_callback
        if callback is not None:
            callback(int(frame_count), bool(complete))

    def _valid_complete_metadata(self) -> dict[str, object] | None:
        if not self.metadata_path.exists() or not self.offsets_path.exists():
            return None
        try:
            metadata = json.loads(self.metadata_path.read_text(encoding="utf-8"))
            stat = self.source.stat()
            expected = SourceIdentity.from_path(self.source, stat.st_mtime_ns, stat.st_size)
            stored = SourceIdentity.from_json(metadata["source"])
            frame_count = int(metadata["frame_count"])
            return metadata if (
                bool(metadata.get("complete"))
                and stored == expected
                and int(metadata.get("atom_count", -1)) == self.atom_count
                and frame_count > 0
                and self.offsets_path.stat().st_size
                == frame_count * np.dtype(np.uint64).itemsize
            ) else None
        except Exception:
            return None

    def _write_metadata(self, *, complete: bool, frame_count: int) -> None:
        stat = self.source.stat()
        identity = SourceIdentity.from_path(self.source, stat.st_mtime_ns, stat.st_size)
        temporary = self.metadata_path.with_suffix(f"{self.metadata_path.suffix}.tmp")
        temporary.write_text(
            json.dumps(
                {
                    "source": identity.to_json(),
                    "frame_count": int(frame_count),
                    "atom_count": self.atom_count,
                    "complete": bool(complete),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        temporary.replace(self.metadata_path)

    def _raise_if_failed(self) -> None:
        if self._error is not None:
            raise RuntimeError(f"XYZ indexing failed: {self._error}") from self._error
