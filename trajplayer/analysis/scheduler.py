from __future__ import annotations

import threading

import numpy as np
from PySide6.QtCore import QObject, QThread, Signal

from trajplayer.frame_store import FrameStore
from trajplayer.interaction.models import AnalysisRequest

from .runner import AnalysisCancelled, run_analysis


class _AnalysisThread(QThread):
    progress = Signal(int, int)
    resultReady = Signal(object)
    failed = Signal(str)
    cancelled = Signal()

    def __init__(
        self,
        store: FrameStore,
        request: AnalysisRequest,
        atom_numbers: np.ndarray,
        playback_event: threading.Event,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self.store = store
        self.request = request
        self.atom_numbers = np.ascontiguousarray(atom_numbers, dtype=np.uint16)
        self.playback_event = playback_event
        self.cancel_event = threading.Event()

    def cancel(self) -> None:
        self.cancel_event.set()

    def run(self) -> None:  # type: ignore[override]
        try:
            result = run_analysis(
                self.store,
                self.request,
                self.atom_numbers,
                self.cancel_event,
                self.playback_event,
                self.progress.emit,
            )
        except AnalysisCancelled:
            self.cancelled.emit()
        except Exception as exc:
            self.failed.emit(f"{type(exc).__name__}: {exc}")
        else:
            if self.cancel_event.is_set():
                self.cancelled.emit()
            else:
                self.resultReady.emit(result)


class AnalysisScheduler(QObject):
    started = Signal(object)
    progress = Signal(int, int)
    resultReady = Signal(object)
    failed = Signal(str)
    cancelled = Signal()
    idle = Signal(object)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._thread: _AnalysisThread | None = None
        self._playback_event = threading.Event()

    @property
    def busy(self) -> bool:
        return self._thread is not None and self._thread.isRunning()

    @property
    def active_store(self) -> FrameStore | None:
        return None if self._thread is None else self._thread.store

    def set_playback_active(self, active: bool) -> None:
        if active:
            self._playback_event.set()
        else:
            self._playback_event.clear()

    def submit(
        self,
        store: FrameStore,
        request: AnalysisRequest,
        atom_numbers: np.ndarray,
    ) -> None:
        if self._thread is not None:
            raise RuntimeError("Another analysis is already scanning the trajectory")
        thread = _AnalysisThread(store, request, atom_numbers, self._playback_event, self)
        thread.progress.connect(self.progress)
        thread.resultReady.connect(self.resultReady)
        thread.failed.connect(self.failed)
        thread.cancelled.connect(self.cancelled)
        thread.finished.connect(lambda worker=thread: self._on_finished(worker))
        self._thread = thread
        self.started.emit(request)
        thread.start()

    def cancel(self) -> None:
        if self._thread is not None:
            self._thread.cancel()

    def shutdown(self, timeout_ms: int = 5000) -> bool:
        thread = self._thread
        if thread is None:
            return True
        thread.cancel()
        return bool(thread.wait(max(0, int(timeout_ms))))

    def _on_finished(self, thread: _AnalysisThread) -> None:
        store = thread.store
        if self._thread is thread:
            self._thread = None
        thread.deleteLater()
        self.idle.emit(store)
