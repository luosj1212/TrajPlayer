from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Iterable

PROCESS_STARTED_S = time.perf_counter()

from trajplayer.startup import error_log_path, initialize_runtime, report_numpy_import_error


initialize_runtime()

try:
    import numpy as np
except Exception as exc:
    report_numpy_import_error(exc)
    raise SystemExit(1) from None

from PySide6.QtCore import QEvent, QTimer, Qt, Signal
from PySide6.QtGui import QSurfaceFormat
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QMessageBox,
)

from trajplayer.benchmark_stats import BenchmarkDiagnostics
from trajplayer.cli_args import CliArgs, parse_cli_args
from trajplayer.commands import WindowCommands
from trajplayer.gl_view import MoleculeGLWidget, default_surface_format
from trajplayer.frame_store import FrameStore
from trajplayer.playback import PlaybackEngine
from trajplayer.present_scheduler import PresentScheduler
from trajplayer.process_memory import ProcessMemorySnapshot, process_memory_snapshot
from trajplayer.scrubbing import SliderScrubState
from trajplayer.selection import (
    ChainSelectionError,
    format_chain_selection,
    parse_chain_selection,
)
from trajplayer.streaming import FrameStreamer
from trajplayer.topology import BondSource, BondTopology, empty_topology
from trajplayer.ui import MainWindowView
from trajplayer.trajectory_source import (
    TrajectorySelectionError,
    TrajectorySource,
    paths_are_supported_for_drop,
    resolve_trajectory_source,
)
if TYPE_CHECKING:
    from trajplayer.binary_store import BinaryTrajectoryStore
    from trajplayer.gui_smoke import GuiSmokeController
    from trajplayer.workers import BondInferenceThread, TrajectoryOpenThread

class TrajPlayerApplication(QApplication):
    """Capture Finder file-open events before or after the main window exists."""

    def __init__(self, argv: list[str]) -> None:
        super().__init__(argv)
        self._file_open_handler: Callable[[Path], None] | None = None
        self._pending_file_open_paths: list[Path] = []

    def set_file_open_handler(
        self,
        handler: Callable[[Path], None],
        *,
        replay_pending: bool = True,
    ) -> None:
        self._file_open_handler = handler
        pending = tuple(self._pending_file_open_paths) if replay_pending else ()
        self._pending_file_open_paths.clear()
        for path in pending:
            handler(path)

    def event(self, event) -> bool:  # type: ignore[override]
        if event.type() == QEvent.Type.FileOpen:
            file_name = event.file()
            if not file_name and not event.url().isEmpty():
                file_name = event.url().toLocalFile()
            if file_name:
                path = Path(file_name)
                if self._file_open_handler is None:
                    self._pending_file_open_paths.append(path)
                else:
                    self._file_open_handler(path)
                return True
        return super().event(event)


class TrajPlayerWindow(MainWindowView):
    stream_frame_ready = Signal(int)
    stream_failed = Signal(object, str)
    error_reported = Signal(str)

    TARGET_FPS = 60.0
    PREFETCH_RADIUS = 200
    SCRUB_PREVIEW_TIMER_MS = 4
    SCRUB_PREVIEW_FPS = 60.0

    @property
    def current_frame(self) -> int:
        return self.present_scheduler.target_frame

    @current_frame.setter
    def current_frame(self, frame_index: int) -> None:
        self.present_scheduler.set_target_frame(frame_index)

    @property
    def displayed_frame(self) -> int:
        return self.present_scheduler.displayed_frame

    @displayed_frame.setter
    def displayed_frame(self, frame_index: int) -> None:
        if int(frame_index) < 0:
            self.present_scheduler.invalidate_display()
        else:
            self.present_scheduler.set_displayed_frame(frame_index)

    def __init__(self) -> None:
        super().__init__()
        self.store: FrameStore | None = None
        self.streamer: FrameStreamer | None = None
        self.open_thread: TrajectoryOpenThread | None = None
        self._pending_open_source: TrajectorySource | None = None
        self._retired_open_stores: dict[
            TrajectoryOpenThread, FrameStore | None
        ] = {}
        self._retired_streamers: dict[FrameStreamer, FrameStore] = {}
        self.cache_build_in_progress = False
        self.bond_thread: BondInferenceThread | None = None
        self.bond_inference_pending = False
        self._retired_bond_threads: set[BondInferenceThread] = set()
        self.playback: PlaybackEngine | None = None
        self.present_scheduler = PresentScheduler()
        self.last_stream_seek_frame = 0
        self.trajectory_generation = 0
        self.reset_view_on_next_frame = False
        self.internal_slider_change = False
        self.suppress_slider_value: int | None = None
        self.component_ids = np.empty((0,), dtype=np.int32)
        self.component_sizes = np.empty((0,), dtype=np.int32)
        self.bond_topology = empty_topology()
        self.filter_mode = "all"
        self.filter_values = {"atom": 1}
        self.selected_chains = (1,)
        self._external_open_paths: list[Path] = []
        self.slider_scrub = SliderScrubState(preview_interval_s=1.0 / self.SCRUB_PREVIEW_FPS)
        self.benchmark_output: Path | None = None
        self.benchmark_target_frames = 0
        self.benchmark_started_s = 0.0
        self.benchmark_base_metrics: dict[str, object] = {}
        self.benchmark_diagnostics: BenchmarkDiagnostics | None = None
        self.benchmark_finish_gpu = False
        self.benchmark_warmup_started_s = 0.0
        self.benchmark_memory_idle: ProcessMemorySnapshot | None = None
        self.automation_mode = False
        self.gui_smoke_controller: GuiSmokeController | None = None
        self.startup_metrics: dict[str, float] = {
            "process_to_qapplication_ms": 0.0,
            "process_to_window_visible_ms": 0.0,
            "process_to_first_gl_frame_ms": 0.0,
        }
        self._open_started_s: float | None = None
        self._open_first_frame_ms = 0.0

        self.gl_view = MoleculeGLWidget()
        self.gl_view.frameSwapped.connect(self.on_frame_swapped)
        self.stream_frame_ready.connect(self.on_stream_frame_ready)
        self.stream_failed.connect(self.on_stream_failed)

        self.retired_streamer_timer = QTimer(self)
        self.retired_streamer_timer.setInterval(50)
        self.retired_streamer_timer.timeout.connect(self.reap_retired_streamers)

        self.setup_ui(self.gl_view)
        self.commands = WindowCommands(
            self,
            open_file=self.open_file,
            toggle_playback=self.toggle_playback,
            step_previous=self.step_prev,
            step_next=self.step_next,
            jump_first=self.jump_first,
            jump_last=self.jump_last,
            reset_camera=self.reset_view,
        )
        self.commands.bind_buttons(self)

        self.render_timer = QTimer(self)
        self.render_timer.setTimerType(Qt.TimerType.PreciseTimer)
        self.render_timer.setSingleShot(True)
        self.render_timer.timeout.connect(self.on_render_tick)
        self.benchmark_poll_timer = QTimer(self)
        self.benchmark_poll_timer.setInterval(100)
        self.benchmark_poll_timer.timeout.connect(self.check_benchmark_finished)
        self.benchmark_warmup_timer = QTimer(self)
        self.benchmark_warmup_timer.setInterval(10)
        self.benchmark_warmup_timer.timeout.connect(self.check_benchmark_warmup)
        self.scrub_preview_timer = QTimer(self)
        self.scrub_preview_timer.setTimerType(Qt.TimerType.PreciseTimer)
        self.scrub_preview_timer.setInterval(self.SCRUB_PREVIEW_TIMER_MS)
        self.scrub_preview_timer.timeout.connect(self.on_scrub_preview_tick)
        self.visibility_filter_timer = QTimer(self)
        self.visibility_filter_timer.setSingleShot(True)
        self.visibility_filter_timer.setInterval(16)
        self.visibility_filter_timer.timeout.connect(self.apply_visibility_filter)
        self.external_open_timer = QTimer(self)
        self.external_open_timer.setSingleShot(True)
        self.external_open_timer.setInterval(120)
        self.external_open_timer.timeout.connect(self.open_queued_external_paths)

    def showEvent(self, event) -> None:  # type: ignore[override]
        super().showEvent(event)
        if self.startup_metrics["process_to_window_visible_ms"] <= 0.0:
            self.startup_metrics["process_to_window_visible_ms"] = (
                time.perf_counter() - PROCESS_STARTED_S
            ) * 1000.0

    def dragEnterEvent(self, event) -> None:  # type: ignore[override]
        paths = [Path(url.toLocalFile()) for url in event.mimeData().urls() if url.isLocalFile()]
        if paths_are_supported_for_drop(paths):
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event) -> None:  # type: ignore[override]
        paths = [Path(url.toLocalFile()) for url in event.mimeData().urls() if url.isLocalFile()]
        if paths:
            self.load_trajectory_paths(paths)

    def queue_external_open_path(self, path: Path) -> None:
        candidate = Path(path)
        if candidate not in self._external_open_paths:
            self._external_open_paths.append(candidate)
        self.external_open_timer.start()

    def open_queued_external_paths(self) -> None:
        paths = tuple(self._external_open_paths)
        self._external_open_paths.clear()
        if paths:
            self.load_trajectory_paths(paths)

    def open_file(self) -> None:
        file_paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Open trajectory (select GRO + XTC/TRR together)",
            "",
            "Trajectory files (*.traj *.xyz *.extxyz *.pdb *.cif *.gro *.xtc *.trr);;"
            "Gromacs files (*.gro *.xtc *.trr);;All files (*.*)",
        )
        if file_paths:
            self.load_trajectory_paths(Path(path) for path in file_paths)

    def load_trajectory(self, path: Path) -> None:
        self.load_trajectory_paths((path,))

    def load_trajectory_paths(self, paths: Iterable[Path]) -> None:
        try:
            source = resolve_trajectory_source(paths)
        except TrajectorySelectionError as exc:
            self.show_error(str(exc))
            return
        missing = [path for path in source.paths if not path.exists()]
        if missing:
            self.show_error(f"File not found: {missing[0]}")
            return
        self._open_started_s = time.perf_counter()
        self._open_first_frame_ms = 0.0
        self.close_current_trajectory()
        if self._retired_open_stores:
            self._pending_open_source = source
            self.set_loading_state(source)
            self.info_label.setText("Waiting for the previous cache operation to stop")
            self.status_bar.showMessage("Queued trajectory open; background I/O is stopping")
            return
        self._pending_open_source = None
        self._start_trajectory_open(source)

    def _start_trajectory_open(self, source: TrajectorySource) -> None:
        from trajplayer.workers import TrajectoryOpenThread

        self.set_loading_state(source)
        thread = TrajectoryOpenThread(source)
        thread.progress.connect(
            lambda done, total, worker=thread: self.on_open_progress(worker, done, total)
        )
        thread.stage_changed.connect(
            lambda message, worker=thread: self.on_open_stage_changed(worker, message)
        )
        thread.index_progress.connect(
            lambda count, complete, worker=thread: self.on_index_progress(
                worker,
                count,
                complete,
            )
        )
        thread.preview_ready.connect(
            lambda opened_source, store, worker=thread: self.on_trajectory_preview(
                worker,
                opened_source,
                store,
            )
        )
        thread.loaded.connect(
            lambda opened_source, store, from_cache, worker=thread: self.on_trajectory_loaded(
                worker,
                opened_source,
                store,
                from_cache,
            )
        )
        thread.failed.connect(
            lambda message, worker=thread: self.on_trajectory_failed(worker, message)
        )
        thread.finished.connect(
            lambda worker=thread: self.on_open_thread_finished(worker)
        )
        self.open_thread = thread
        thread.start()

    def set_loading_state(self, source: TrajectorySource) -> None:
        self.open_button.setEnabled(False)
        self.file_label.setText(f"Opening {source.display_name}")
        self.file_label.setToolTip(source.tooltip)
        self.info_label.setText("Opening trajectory metadata and first frame")
        self.status_bar.showMessage("Opening trajectory without blocking the UI")
        self.progress_bar.setRange(0, 0)
        self.progress_bar.show()
        self.set_controls_enabled(False)
        self.update_frame_label()

    def on_open_stage_changed(self, thread: TrajectoryOpenThread, message: str) -> None:
        if thread is not self.open_thread:
            return
        self.info_label.setText(message)
        self.status_bar.showMessage(message)

    def on_open_progress(self, thread: TrajectoryOpenThread, done: int, total: int) -> None:
        if thread is not self.open_thread:
            return
        random_access = self.store is not None and self.store.supports_random_access
        if not random_access:
            self.progress_bar.setRange(0, max(total, 1))
            self.progress_bar.setValue(done)
        if self.streamer is not None:
            self.streamer.notify_store_updated()
        if self.store is not None and self.cache_build_in_progress:
            self.update_available_controls()
        if not random_access:
            self.status_bar.showMessage(
                f"Converting to contiguous float32 cache: {done}/{total} frames"
            )

    def on_index_progress(
        self,
        thread: TrajectoryOpenThread,
        frame_count: int,
        complete: bool,
    ) -> None:
        if thread is not self.open_thread or self.store is None:
            return
        if self.streamer is not None:
            self.streamer.notify_store_updated()
        self.update_available_controls()
        self.update_trajectory_info()
        self.update_frame_label()
        if complete:
            self.status_bar.showMessage(
                f"Indexed {frame_count} frames; direct trajectory reading is ready"
            )
        else:
            self.status_bar.showMessage(
                f"First frame ready; background index has found {frame_count} frames"
            )

    def on_trajectory_preview(
        self,
        thread: TrajectoryOpenThread,
        source: TrajectorySource,
        store: FrameStore,
    ) -> None:
        if thread is not self.open_thread:
            return
        if self.store is not None:
            return
        self.cache_build_in_progress = True
        self.activate_trajectory(source, store)
        self.progress_bar.hide()
        self.open_button.setEnabled(True)
        self.update_available_controls()
        self.status_bar.showMessage(
            "First frame ready; nearby frames will be decoded on demand"
        )

    def on_trajectory_loaded(
        self,
        thread: TrajectoryOpenThread,
        source: TrajectorySource,
        store: FrameStore,
        from_cache: bool,
    ) -> None:
        if thread is not self.open_thread:
            return
        self.open_thread = None
        self.finish_trajectory_load(source, store, from_cache=from_cache)

    def finish_trajectory_load(
        self,
        source: TrajectorySource,
        store: FrameStore,
        *,
        from_cache: bool,
    ) -> None:
        if self.store is not store:
            self.activate_trajectory(source, store)
        self.cache_build_in_progress = False
        if self.streamer is not None:
            self.streamer.notify_store_updated()
        self.internal_slider_change = True
        self.frame_slider.setMaximum(store.frame_count - 1)
        self.internal_slider_change = False
        self.frame_slider.setEnabled(store.frame_count > 1)
        self.progress_bar.hide()
        if bool(store.metadata.get("direct_reader")):
            source_label = "direct reader"
        else:
            source_label = "cache" if from_cache else "new binary cache"
        self.update_trajectory_info()
        self.status_bar.showMessage(f"Loaded {source.display_name} from {source_label}")
        self.open_button.setEnabled(True)
        self.set_controls_enabled(store.frame_count > 1)
        self.commands.set_reset_enabled(True)
        self.update_frame_label()

    def on_open_thread_finished(self, thread: TrajectoryOpenThread) -> None:
        retired_store = self._retired_open_stores.pop(thread, None)
        if (
            retired_store is not None
            and retired_store not in self._retired_streamers.values()
        ):
            retired_store.close()
        if self.open_thread is thread and thread.preview_store is None:
            self.open_thread = None
        thread.deleteLater()
        if (
            self.open_thread is None
            and not self._retired_open_stores
            and self._pending_open_source is not None
        ):
            source = self._pending_open_source
            self._pending_open_source = None
            self._start_trajectory_open(source)

    def activate_trajectory(
        self,
        source: TrajectorySource,
        store: FrameStore,
    ) -> None:
        self.present_scheduler.begin_generation(target_frame=0)
        self.store = store

        def report_stream_error(error: BaseException) -> None:
            self.stream_failed.emit(
                streamer,
                f"Frame streaming failed: {type(error).__name__}: {error}",
            )

        streamer = FrameStreamer(
            store,
            prefetch_radius=self.PREFETCH_RADIUS,
            frame_ready_callback=self.stream_frame_ready.emit,
            error_callback=report_stream_error,
        )
        self.streamer = streamer
        streamer.start()
        self.last_stream_seek_frame = 0
        self.reset_view_on_next_frame = True
        self.component_ids = np.empty((0,), dtype=np.int32)
        self.component_sizes = np.empty((0,), dtype=np.int32)
        self.bond_topology = empty_topology()
        self.gl_view.set_atoms(store.atom_numbers)
        self.gl_view.set_render_mode(str(self.render_mode_combo.currentData()))
        self.gl_view.set_atom_size_scale(self.atom_size_slider.value() / 100.0)
        self.gl_view.set_bond_size_scale(self.bond_size_slider.value() / 100.0)
        self.set_representation_controls_enabled(True)
        self.gl_view.set_box_enabled(self.box_check.isChecked())
        self.box_check.setEnabled(store.has_cells)
        self.infer_bonds_check.setEnabled(not bool(store.metadata.get("synthetic")))
        self.configure_filter_controls(store.atom_count)
        if self.infer_bonds_check.isChecked():
            self.start_bond_inference(store)
        else:
            self.gl_view.set_bonds(np.empty((0, 2), dtype=np.int32))
        self.request_stream_frame(0)

        self.internal_slider_change = True
        self.frame_slider.setMaximum(max(0, store.navigable_frame_count - 1))
        self.frame_slider.setValue(0)
        self.internal_slider_change = False
        self.frame_slider.setEnabled(store.navigable_frame_count > 1)

        self.file_label.setText(source.display_name)
        self.file_label.setToolTip(source.tooltip)
        self.update_trajectory_info()
        self.set_controls_enabled(store.navigable_frame_count > 1)
        self.commands.set_reset_enabled(True)
        self.update_frame_label()

    def update_available_controls(self) -> None:
        if self.store is None:
            return
        available = self.store.navigable_frame_count
        self.internal_slider_change = True
        self.frame_slider.setMaximum(max(0, available - 1))
        self.internal_slider_change = False
        enabled = available > 1
        self.frame_slider.setEnabled(enabled)
        self.set_controls_enabled(enabled)

    def on_box_toggled(self, checked: bool) -> None:
        self.gl_view.set_box_enabled(checked)
        self.displayed_frame = -1

    def on_infer_bonds_toggled(self, checked: bool) -> None:
        if self.store is None or self.store.metadata.get("synthetic"):
            return
        if checked:
            self.start_bond_inference(self.store)
            return

        self.stop_bond_inference(wait_ms=0)
        self.trajectory_generation += 1
        self.bond_topology = empty_topology()
        self.component_ids = self.bond_topology.component_ids
        self.component_sizes = self.bond_topology.component_sizes
        self.gl_view.set_bonds(self.bond_topology.bonds)
        self.filter_mode_buttons["chain"].setEnabled(False)
        if self.filter_mode == "chain":
            self.on_filter_mode_changed("all")
        self.update_trajectory_info()
        self.status_bar.showMessage("Bond inference disabled")

    def on_render_mode_changed(self) -> None:
        mode = str(self.render_mode_combo.currentData())
        self.gl_view.set_render_mode(mode)
        self.set_representation_controls_enabled(self.store is not None)
        self.update_trajectory_info()
        if self.store is not None:
            self.status_bar.showMessage(f"Representation: {self.render_mode_combo.currentText()}")

    def on_atom_size_changed(self, value: int) -> None:
        self.gl_view.set_atom_size_scale(int(value) / 100.0)
        self.atom_size_value_label.setText(f"{int(value)}%")

    def on_bond_size_changed(self, value: int) -> None:
        self.gl_view.set_bond_size_scale(int(value) / 100.0)
        self.bond_size_value_label.setText(f"{int(value)}%")

    def set_representation_controls_enabled(self, enabled: bool) -> None:
        mode = str(self.render_mode_combo.currentData())
        self.render_mode_combo.setEnabled(enabled)
        atom_enabled = enabled and mode != "bond"
        bond_enabled = enabled and mode != "ball"
        self.atom_size_label.setEnabled(atom_enabled)
        self.atom_size_slider.setEnabled(atom_enabled)
        self.atom_size_value_label.setEnabled(atom_enabled)
        self.bond_size_label.setEnabled(bond_enabled)
        self.bond_size_slider.setEnabled(bond_enabled)
        self.bond_size_value_label.setEnabled(bond_enabled)

    def update_trajectory_info(self) -> None:
        if self.store is None:
            return
        bond_count = self.gl_view.bond_count
        bond_text = f", Bonds: {self.bond_topology.description} ({bond_count})"
        cache_text = ""
        if self.streamer is not None:
            cache_mib = self.streamer.memory_bytes / (1024.0 * 1024.0)
            budget_mode = (
                "auto"
                if self.streamer.memory_budget.mode.startswith("auto")
                else "fixed"
            )
            cache_text = (
                f", {self.streamer.capacity}-frame/{cache_mib:.0f} MiB "
                f"{budget_mode} cache"
            )
        if bool(self.store.metadata.get("direct_reader")):
            disk_cache_text = (
                "direct source"
                if self.store.frame_count_is_final
                else f"direct source, indexing ({self.store.frame_count}+ frames found)"
            )
        else:
            disk_cache_text = (
                f"{self.store.available_frame_count}/{self.store.frame_count} frames cached on demand"
                if self.store.supports_random_access and not self.store.is_complete
                else f"{self.store.available_frame_count}/{self.store.frame_count} frames on disk"
            )
        self.info_label.setText(
            f"{self.store.frame_count} frames, {self.store.atom_count} atoms{bond_text}, "
            f"directional prefetch{cache_text}, "
            f"{disk_cache_text}, "
            f"{self.render_mode_combo.currentText()} GPU instancing"
        )

    def configure_filter_controls(self, atom_count: int) -> None:
        enabled = atom_count > 0
        largest_index = max(1, int(atom_count))
        label_width = max(
            76,
            self.filter_value_label.fontMetrics().horizontalAdvance(f"Atom {largest_index}") + 6,
        )
        self.filter_value_label.setFixedWidth(label_width)
        self.filter_mode = "all"
        self.filter_values = {"atom": 1}
        self.selected_chains = (1,)
        self.filter_mode_buttons["all"].setChecked(True)
        self.filter_mode_buttons["all"].setEnabled(enabled)
        self.filter_mode_buttons["atom"].setEnabled(enabled)
        self.filter_mode_buttons["chain"].setEnabled(enabled and self.component_sizes.size > 0)
        self.filter_value_slider.blockSignals(True)
        self.filter_value_slider.setRange(1, max(1, int(atom_count)))
        self.filter_value_slider.setValue(1)
        self.filter_value_slider.setEnabled(False)
        self.filter_value_slider.blockSignals(False)
        self.filter_value_slider.setToolTip("All atoms are visible")
        self.filter_value_label.setText("All atoms")
        self.filter_value_label.setEnabled(False)
        self.chain_selection_edit.blockSignals(True)
        self.chain_selection_edit.setText("1")
        self.chain_selection_edit.blockSignals(False)
        self._set_chain_selection_invalid(False)
        self._set_filter_value_control_mode("all")

    def on_filter_mode_changed(self, mode: str) -> None:
        if self.store is None:
            return
        mode = str(mode)
        if mode == "chain" and self.component_sizes.size == 0:
            mode = "all"
            self.status_bar.showMessage("Chain groups are still being prepared")

        self.filter_mode = mode if mode in {"all", "chain", "atom"} else "atom"
        self.filter_mode_buttons[self.filter_mode].setChecked(True)
        self.filter_value_slider.blockSignals(True)
        if self.filter_mode == "all":
            self.filter_value_slider.setEnabled(False)
            self.filter_value_label.setText("All atoms")
            self.filter_value_label.setEnabled(False)
            self.filter_value_slider.setToolTip("All atoms are visible")
        elif self.filter_mode == "atom":
            maximum = self.store.atom_count
            self.filter_value_slider.setRange(1, maximum)
            self.filter_value_slider.setPageStep(max(1, maximum // 100))
            self.filter_value_slider.setValue(min(self.filter_values["atom"], maximum))
            self.filter_value_slider.setEnabled(True)
            self.filter_value_label.setEnabled(True)
        self.filter_value_slider.blockSignals(False)
        self._set_filter_value_control_mode(self.filter_mode)
        self.update_filter_value_label()
        self.schedule_visibility_filter()
        if self.filter_mode == "chain":
            self.chain_selection_edit.setFocus()
            self.chain_selection_edit.selectAll()

    def on_filter_value_changed(self, value: int) -> None:
        if self.filter_mode == "atom":
            self.filter_values["atom"] = int(value)
        self.update_filter_value_label()
        self.schedule_visibility_filter()

    def on_chain_selection_changed(self, _text: str) -> None:
        if self.filter_mode == "chain":
            self.schedule_visibility_filter()

    def normalize_chain_selection(self) -> None:
        if self.component_sizes.size == 0:
            return
        try:
            chains = parse_chain_selection(
                self.chain_selection_edit.text(),
                int(self.component_sizes.shape[0]),
            )
        except ChainSelectionError:
            return
        normalized = format_chain_selection(chains)
        if normalized != self.chain_selection_edit.text():
            self.chain_selection_edit.setText(normalized)

    def _set_filter_value_control_mode(self, mode: str) -> None:
        chain_mode = mode == "chain"
        self.chain_selection_edit.setVisible(chain_mode)
        self.chain_selection_edit.setEnabled(chain_mode)
        self.filter_value_slider.setVisible(not chain_mode)
        self.filter_value_label.setVisible(not chain_mode)

    def _set_chain_selection_invalid(self, invalid: bool) -> None:
        if bool(self.chain_selection_edit.property("invalid")) == bool(invalid):
            return
        self.chain_selection_edit.setProperty("invalid", bool(invalid))
        style = self.chain_selection_edit.style()
        style.unpolish(self.chain_selection_edit)
        style.polish(self.chain_selection_edit)
        self.chain_selection_edit.update()

    def update_filter_value_label(self) -> None:
        if self.filter_mode == "all":
            self.filter_value_label.setText("All atoms")
            return
        value = self.filter_value_slider.value()
        self.filter_value_label.setText(f"Atom {value}")
        self.filter_value_slider.setToolTip(
            f"Atom {value} of {self.filter_value_slider.maximum()}"
        )

    def schedule_visibility_filter(self, _value: int | None = None) -> None:
        self.visibility_filter_timer.start()

    def apply_visibility_filter(self) -> None:
        if self.store is None:
            return
        mode = self.filter_mode
        visible_atoms: np.ndarray | None
        if mode == "all":
            visible_atoms = None
            message = f"Showing all {self.store.atom_count} atoms"
            unwrap_group_ids = None
            self._set_chain_selection_invalid(False)
        elif mode == "chain":
            try:
                chains = parse_chain_selection(
                    self.chain_selection_edit.text(),
                    int(self.component_sizes.shape[0]),
                )
            except ChainSelectionError as exc:
                self._set_chain_selection_invalid(True)
                self.status_bar.showMessage(f"Invalid chain selection: {exc}")
                return
            self._set_chain_selection_invalid(False)
            self.selected_chains = chains
            selected_components = np.zeros(
                self.component_sizes.shape[0],
                dtype=np.bool_,
            )
            selected_components[np.asarray(chains, dtype=np.int32) - 1] = True
            visible_atoms = np.flatnonzero(
                selected_components[self.component_ids]
            ).astype(
                np.int32,
                copy=False,
            )
            selection_text = format_chain_selection(chains)
            noun = "chain" if len(chains) == 1 else "chains"
            message = f"Showing {noun} {selection_text}: {len(visible_atoms)} atoms"
            unwrap_group_ids = self.component_ids
        else:
            atom_index = self.filter_value_slider.value() - 1
            visible_atoms = np.array([atom_index], dtype=np.int32)
            message = f"Showing atom {atom_index + 1}"
            unwrap_group_ids = None
            self._set_chain_selection_invalid(False)
        self.gl_view.set_visible_atoms(
            visible_atoms,
            fit_view=self.displayed_frame >= 0,
            unwrap_periodic=mode == "chain",
            unwrap_group_ids=unwrap_group_ids,
        )
        self.status_bar.showMessage(message)

    def start_bond_inference(self, store: FrameStore) -> None:
        self.stop_bond_inference(wait_ms=0)
        self.trajectory_generation += 1
        self.gl_view.set_bonds(np.empty((0, 2), dtype=np.int32))
        if store.metadata.get("synthetic"):
            self.bond_topology = empty_topology(BondSource.GENERATED)
            self.update_trajectory_info()
            self.status_bar.showMessage("Synthetic benchmark loaded without bond inference")
            return
        if not self.infer_bonds_check.isChecked():
            self.bond_topology = empty_topology()
            self.update_trajectory_info()
            return

        self.bond_topology = empty_topology(BondSource.INFERENCE_PENDING)
        self.component_ids = self.bond_topology.component_ids
        self.component_sizes = self.bond_topology.component_sizes
        self.filter_mode_buttons["chain"].setEnabled(False)
        self.update_trajectory_info()
        self.bond_inference_pending = True
        self.try_start_bond_inference()

    def try_start_bond_inference(self) -> None:
        from trajplayer.workers import BondInferenceThread

        if (
            not self.bond_inference_pending
            or self.bond_thread is not None
            or self.store is None
            or self.streamer is None
        ):
            return
        lease = self.streamer.acquire_frame(0)
        if lease is None:
            self.request_stream_frame(0)
            return
        self.bond_inference_pending = False
        atom_numbers = np.array(
            self.store.atom_numbers,
            dtype=np.uint16,
            copy=True,
            order="C",
        )
        try:
            thread = BondInferenceThread(
                self.trajectory_generation,
                lease.positions,
                atom_numbers,
                lease.cell,
                release_callback=lease.release,
            )
        except Exception:
            lease.release()
            self.bond_inference_pending = True
            raise
        thread.ready.connect(self.on_bonds_ready)
        thread.failed.connect(self.on_bonds_failed)
        thread.finished.connect(lambda thread=thread: self.on_bond_thread_finished(thread))
        self.bond_thread = thread
        self.status_bar.showMessage("Inferring bonds in the background")
        thread.start()

    def stop_bond_inference(self, *, wait_ms: int) -> None:
        self.bond_inference_pending = False
        thread = self.bond_thread
        self.bond_thread = None
        if thread is not None and thread.isRunning():
            thread.requestInterruption()
            if wait_ms > 0:
                thread.wait(wait_ms)
            if thread.isRunning():
                self._retired_bond_threads.add(thread)

    def on_bonds_ready(
        self,
        generation: int,
        topology: BondTopology,
        elapsed_ms: float,
    ) -> None:
        if generation != self.trajectory_generation or self.store is None:
            return
        self.bond_topology = topology
        self.component_ids = topology.component_ids
        self.component_sizes = topology.component_sizes
        self.filter_mode_buttons["chain"].setEnabled(
            topology.chain_selection_available
        )
        self.gl_view.set_bonds(topology.bonds)
        self.update_trajectory_info()
        self.status_bar.showMessage(
            f"Bonds ready: {len(topology.bonds)} inferred from frame 1, "
            f"{len(self.component_sizes)} components in {elapsed_ms:.0f} ms"
        )

    def on_bonds_failed(self, generation: int, message: str) -> None:
        if generation != self.trajectory_generation:
            return
        self.bond_topology = empty_topology(BondSource.INFERENCE_FAILED)
        self.update_trajectory_info()
        self.status_bar.showMessage(message)

    def on_bond_thread_finished(self, thread: object) -> None:
        if self.bond_thread is thread:
            self.bond_thread = None
        self._retired_bond_threads.discard(thread)
        thread.deleteLater()

    def start_benchmark(
        self,
        *,
        label: Path,
        store: BinaryTrajectoryStore,
        output_path: Path,
        render_frames: int,
        finish_gpu: bool,
        base_metrics: dict[str, object],
    ) -> None:
        self.benchmark_output = output_path
        self._open_started_s = time.perf_counter()
        self._open_first_frame_ms = 0.0
        self.benchmark_target_frames = max(1, int(render_frames))
        self.benchmark_base_metrics = dict(base_metrics)
        self.benchmark_finish_gpu = bool(finish_gpu)
        self.benchmark_warmup_started_s = time.perf_counter()
        self.benchmark_started_s = self.benchmark_warmup_started_s
        self.finish_trajectory_load(TrajectorySource(label), store, from_cache=True)
        benchmark_mode = str(self.benchmark_base_metrics.get("benchmark_mode", "ball_stick"))
        self.gl_view.set_render_mode(benchmark_mode)
        if self.benchmark_base_metrics.get("benchmark_bonds") and store.atom_count > 1:
            atom_indices = np.arange(store.atom_count, dtype=np.int32)
            bonds = np.column_stack((atom_indices[:-1], atom_indices[1:]))
            self.gl_view.set_bonds(np.ascontiguousarray(bonds, dtype=np.int32))
            self.benchmark_base_metrics["bond_count"] = int(bonds.shape[0])
            self.update_trajectory_info()
        self.benchmark_warmup_timer.start()
        QTimer.singleShot(120_000, self.finish_benchmark_timeout)

    def check_benchmark_warmup(self) -> None:
        if self.benchmark_output is None:
            return
        if self.streamer is None or self.store is None:
            return
        if not self.streamer.is_window_ready(self.current_frame):
            return
        self.benchmark_warmup_timer.stop()
        self.benchmark_base_metrics["prefetch_warmup_s"] = (
            time.perf_counter() - self.benchmark_warmup_started_s
        )
        self.begin_benchmark_playback()

    def begin_benchmark_playback(self) -> None:
        if self.store is None:
            self.finish_benchmark(timed_out=True)
            return
        self.benchmark_diagnostics = BenchmarkDiagnostics()
        self.benchmark_memory_idle = process_memory_snapshot()
        self.gl_view.enable_benchmark_stats(finish_gpu=self.benchmark_finish_gpu)
        self.benchmark_started_s = time.perf_counter()
        self.playback = PlaybackEngine(
            total_frames=self.store.frame_count,
            fps=self.TARGET_FPS,
            loop=True,
        )
        self.playback.start(frame_index=0, now_s=time.perf_counter())
        self.set_play_button_state(True)
        self.schedule_next_render_tick()
        self.benchmark_poll_timer.start()

    def check_benchmark_finished(self) -> None:
        if self.benchmark_output is None:
            return
        stats = self.gl_view.render_stats
        if stats is not None and stats.summary()["frames"] >= self.benchmark_target_frames:
            self.finish_benchmark(timed_out=False)

    def finish_benchmark_timeout(self) -> None:
        if self.benchmark_output is not None:
            self.finish_benchmark(timed_out=True)

    def finish_benchmark(self, *, timed_out: bool) -> None:
        if self.benchmark_output is None:
            return
        self.benchmark_poll_timer.stop()
        self.benchmark_warmup_timer.stop()
        stats = self.gl_view.render_stats
        render_summary = stats.summary() if stats is not None else {}
        streamer_stats = (
            self.streamer.stats_snapshot() if self.streamer is not None else None
        )
        memory_idle = self.benchmark_memory_idle or process_memory_snapshot()
        memory_playback = process_memory_snapshot()
        elapsed_s = max(0.0, time.perf_counter() - self.benchmark_started_s)
        pipeline_summary = (
            self.benchmark_diagnostics.summary()
            if self.benchmark_diagnostics is not None
            else {}
        )
        streamer_io = (
            {}
            if streamer_stats is None
            else {
                "loads": streamer_stats.loads,
                "cache_hits": streamer_stats.cache_hits,
                "cache_misses": streamer_stats.cache_misses,
                "cache_hit_rate": streamer_stats.cache_hit_rate,
                "load_latency_ms": streamer_stats.load_latency_ms,
                "frame_read_ms_p50": streamer_stats.read_latency_ms_p50,
                "frame_read_ms_p95": streamer_stats.read_latency_ms_p95,
                "frame_read_ms_p99": streamer_stats.read_latency_ms_p99,
                "decoded_megabytes": streamer_stats.decoded_megabytes,
                "decode_mb_s": streamer_stats.decode_megabytes_per_second,
                "decode_megabytes_per_second": streamer_stats.decode_megabytes_per_second,
                "effective_prefetch_frames": streamer_stats.effective_prefetch_frames,
                "lease_acquisitions": streamer_stats.lease_acquisitions,
                "lease_releases": streamer_stats.lease_releases,
                "active_leases": streamer_stats.active_leases,
                "peak_active_leases": streamer_stats.peak_active_leases,
                "stale_lease_releases": streamer_stats.stale_lease_releases,
                "allocated_capacity": streamer_stats.allocated_capacity,
                "max_capacity": streamer_stats.max_capacity,
                "slab_count": streamer_stats.slab_count,
                "allocated_cache_bytes": streamer_stats.allocated_cache_bytes,
                "memory_target_bytes": streamer_stats.memory_target_bytes,
                "memory_target_reason": streamer_stats.memory_target_reason,
            }
        )
        frame_cache_bytes = self.streamer.memory_bytes if self.streamer is not None else 0
        result = {
            **self.benchmark_base_metrics,
            "timed_out": timed_out,
            "benchmark_elapsed_s": elapsed_s,
            "measured_fps": render_summary.get("cadence_fps", 0.0),
            "startup": dict(self.startup_metrics),
            "open": {
                "metadata_ms": float(
                    self.benchmark_base_metrics.get("cache_open_s", 0.0)
                )
                * 1000.0,
                "first_frame_ms": self._open_first_frame_ms,
                "index_complete_ms": float(
                    self.benchmark_base_metrics.get("index_complete_ms", 0.0)
                ),
            },
            "render": render_summary,
            "pipeline": pipeline_summary,
            "io": streamer_io,
            "memory": {
                "rss_idle_mib": memory_idle.rss_mib,
                "rss_playback_mib": memory_playback.rss_mib,
                "rss_peak_mib": max(
                    memory_idle.peak_rss_mib,
                    memory_playback.peak_rss_mib,
                ),
                "frame_cache_mib": frame_cache_bytes / (1024.0 * 1024.0),
            },
            "copies": {
                "renderer_full_frame_copy_bytes": int(
                    render_summary.get("renderer_full_frame_copy_bytes", 0)
                ),
                "renderer_full_frame_copy_bytes_per_frame": float(
                    render_summary.get("renderer_full_frame_copy_bytes_per_frame", 0.0)
                ),
            },
            "streamer_memory_bytes": frame_cache_bytes,
            "streamer_budget_bytes": self.streamer.max_memory_bytes if self.streamer is not None else 0,
            "streamer_budget_mode": self.streamer.memory_budget.mode if self.streamer is not None else "none",
            "streamer_io": streamer_io,
            "conservative_depth": self.gl_view.conservative_depth_enabled,
            "opengl": self.gl_view.gl_diagnostics,
        }
        self.benchmark_output.parent.mkdir(parents=True, exist_ok=True)
        self.benchmark_output.write_text(json.dumps(result, indent=2), encoding="utf-8")
        output_path = self.benchmark_output
        self.benchmark_output = None
        self.benchmark_diagnostics = None
        print(f"[benchmark] wrote {output_path}", flush=True)
        self.close()
        QApplication.instance().quit()

    def on_trajectory_failed(self, thread: TrajectoryOpenThread, message: str) -> None:
        if thread is not self.open_thread:
            return
        self.open_thread = None
        self.close_current_trajectory()
        self.progress_bar.hide()
        self.open_button.setEnabled(True)
        self.set_controls_enabled(False)
        self.show_error(message)

    def close_current_trajectory(self) -> None:
        thread = self.open_thread
        deferred_store: FrameStore | None = None
        if thread is not None and thread.isRunning():
            thread.cancel()
            if self.store is not None and self.store is thread.preview_store:
                deferred_store = self.store
            self._retired_open_stores[thread] = deferred_store
        self.open_thread = None

        self.stop_playback()
        self.present_scheduler.begin_generation(target_frame=0)
        self.gl_view.release_frame_reference()
        self.cache_build_in_progress = False
        self.stop_bond_inference(wait_ms=0)
        self.trajectory_generation += 1
        streamer = self.streamer
        store = self.store
        if streamer is not None:
            if not streamer.stop(timeout_s=0.0) and store is not None:
                self._retired_streamers[streamer] = store
                self.retired_streamer_timer.start()
            self.streamer = None
        if self.store is not None:
            if (
                self.store is not deferred_store
                and self.store not in self._retired_streamers.values()
            ):
                self.store.close()
            self.store = None
        self.gl_view.set_cell(None)
        self.box_check.setEnabled(False)
        self.infer_bonds_check.setEnabled(False)
        self.set_representation_controls_enabled(False)
        self.component_ids = np.empty((0,), dtype=np.int32)
        self.component_sizes = np.empty((0,), dtype=np.int32)
        self.bond_topology = empty_topology()
        self.filter_mode = "all"
        self.filter_values = {"atom": 1}
        self.selected_chains = (1,)
        self.filter_mode_buttons["all"].setChecked(True)
        for button in self.filter_mode_buttons.values():
            button.setEnabled(False)
        self.filter_value_slider.blockSignals(True)
        self.filter_value_slider.setRange(1, 1)
        self.filter_value_slider.setValue(1)
        self.filter_value_slider.setEnabled(False)
        self.filter_value_slider.blockSignals(False)
        self.filter_value_label.setText("All atoms")
        self.filter_value_label.setFixedWidth(76)
        self.filter_value_label.setEnabled(False)
        self.chain_selection_edit.blockSignals(True)
        self.chain_selection_edit.setText("1")
        self.chain_selection_edit.blockSignals(False)
        self._set_chain_selection_invalid(False)
        self._set_filter_value_control_mode("all")
        self.current_frame = 0
        self.displayed_frame = -1
        self.scrub_preview_timer.stop()
        self.visibility_filter_timer.stop()
        self.slider_scrub.release(0)
        self.frame_slider.setMaximum(0)
        self.frame_slider.setValue(0)
        self.set_controls_enabled(False)
        self.commands.set_reset_enabled(False)

    def reap_retired_streamers(self) -> None:
        for streamer, store in tuple(self._retired_streamers.items()):
            if streamer.is_alive:
                continue
            streamer.stop(timeout_s=0.0)
            self._retired_streamers.pop(streamer, None)
            if store not in self._retired_open_stores.values():
                store.close()
        if not self._retired_streamers:
            self.retired_streamer_timer.stop()

    def set_controls_enabled(self, enabled: bool) -> None:
        self.commands.set_transport_enabled(enabled)
        self.playback_speed_label.setEnabled(enabled)
        self.playback_speed_slider.setEnabled(enabled)
        self.playback_speed_value_label.setEnabled(enabled)

    def on_render_tick(self) -> None:
        if self.store is None or self.streamer is None:
            return
        if self.present_scheduler.has_pending_frame:
            return

        tick_time = time.perf_counter()
        if self.benchmark_diagnostics is not None:
            self.benchmark_diagnostics.record_tick(timestamp_s=tick_time)

        if (
            self.playback is not None
            and self.playback.running
            and self.displayed_frame == self.current_frame
        ):
            decision = self.playback.schedule(tick_time)
            if decision is not None:
                if self.benchmark_diagnostics is not None:
                    self.benchmark_diagnostics.record_decision(
                        timestamp_s=tick_time,
                        dropped_frames=decision.dropped_frames,
                    )
                self.current_frame = decision.frame_index
                self.request_stream_frame(self.current_frame, direction=1)
                self.set_slider_value(self.current_frame)
                if decision.stop_playback:
                    self.set_play_button_state(False)
                    self.playback = None
            elif self.benchmark_diagnostics is not None:
                self.benchmark_diagnostics.record_no_decision()
        elif self.playback is not None and self.playback.running:
            if self.benchmark_diagnostics is not None:
                self.benchmark_diagnostics.record_no_decision()
        elif not self.slider_scrub.active:
            self.request_stream_frame(self.current_frame)

        if self.displayed_frame == self.current_frame:
            self.schedule_next_render_tick()
            return
        lease = self.streamer.acquire_frame(self.current_frame)
        if lease is None:
            if self.benchmark_diagnostics is not None:
                self.benchmark_diagnostics.record_no_frame()
            self.schedule_next_render_tick()
            return

        frame_index = self.current_frame
        present_token = self.present_scheduler.submit(frame_index, now_s=tick_time)
        if present_token is None:
            lease.release()
            return
        try:
            self.gl_view.set_frame(
                lease.positions,
                reset_view=self.reset_view_on_next_frame,
                cell=lease.cell,
                release_callback=lease.release,
            )
        except Exception:
            self.present_scheduler.clear_pending()
            lease.release()
            raise
        if self.benchmark_diagnostics is not None:
            self.benchmark_diagnostics.record_upload(timestamp_s=time.perf_counter())
        self.reset_view_on_next_frame = False
        self.schedule_next_render_tick()

    def on_frame_swapped(self) -> None:
        now_s = time.perf_counter()
        if self.startup_metrics["process_to_first_gl_frame_ms"] <= 0.0:
            self.startup_metrics["process_to_first_gl_frame_ms"] = (
                now_s - PROCESS_STARTED_S
            ) * 1000.0
        acknowledgement = self.present_scheduler.acknowledge_swap(
            now_s=now_s
        )
        if acknowledgement is None or not acknowledgement.accepted:
            return
        frame_index = acknowledgement.token.frame_index
        if self.benchmark_diagnostics is not None:
            self.benchmark_diagnostics.record_present_latency(
                acknowledgement.latency_ms
            )
        if self._open_started_s is not None and self._open_first_frame_ms <= 0.0:
            self._open_first_frame_ms = max(
                0.0,
                (now_s - self._open_started_s) * 1000.0,
            )
            self._open_started_s = None
        self.update_frame_label()
        if self.current_frame != self.displayed_frame:
            self.render_timer.start(0)
            return
        self.schedule_next_render_tick()

    def on_stream_frame_ready(self, frame_index: int) -> None:
        if int(frame_index) == 0:
            self.try_start_bond_inference()
        if (
            self.store is None
            or self.streamer is None
            or int(frame_index) != self.current_frame
            or self.displayed_frame == self.current_frame
        ):
            return
        self.on_render_tick()

    def on_stream_failed(self, streamer: object, message: str) -> None:
        if streamer is not self.streamer:
            return
        self.set_controls_enabled(False)
        self.show_error(message)

    def request_stream_frame(
        self,
        frame_index: int,
        *,
        direction: int | None = None,
        interactive: bool = False,
    ) -> None:
        if self.streamer is None:
            return
        target = int(frame_index)
        if direction is None:
            delta = target - self.last_stream_seek_frame
            direction = 1 if delta > 0 else -1 if delta < 0 else 0
        self.last_stream_seek_frame = target
        self.streamer.seek(target, direction=direction, interactive=interactive)
        if (
            self.streamer.capacity == 1
            and not self.streamer.has_frame(target)
            and not self.present_scheduler.has_pending_frame
        ):
            self.gl_view.release_frame_reference()
        if (
            self.streamer.has_frame(target)
            and not self.present_scheduler.has_pending_frame
        ):
            self.render_timer.start(0)
    def toggle_playback(self) -> None:
        if self.store is None:
            return
        if self.playback is not None and self.playback.running:
            self.stop_playback()
            return
        available_frames = self.store.navigable_frame_count
        if available_frames <= 1:
            return
        self.playback = PlaybackEngine(
            total_frames=available_frames,
            fps=float(self.playback_speed_slider.value()),
            loop=self.loop_check.isChecked(),
        )
        self.playback.start(frame_index=self.current_frame, now_s=time.perf_counter())
        self.set_play_button_state(True)
        self.schedule_next_render_tick()

    def stop_playback(self) -> None:
        if self.playback is not None:
            self.playback.stop()
        self.playback = None
        self.set_play_button_state(False)
        if self.benchmark_output is None:
            self.render_timer.stop()

    def schedule_next_render_tick(self) -> None:
        frame_available = (
            self.streamer is not None and self.streamer.has_frame(self.current_frame)
        )
        delay_ms = self.present_scheduler.next_timer_delay_ms(
            playback=self.playback,
            frame_available=frame_available,
            now_s=time.perf_counter(),
        )
        if delay_ms is None:
            self.render_timer.stop()
            return
        self.render_timer.start(delay_ms)

    def on_playback_speed_changed(self, value: int) -> None:
        self.playback_speed_value_label.setText(f"{int(value)} FPS")
        if (
            self.benchmark_output is not None
            or self.store is None
            or self.playback is None
            or not self.playback.running
        ):
            return
        self.playback = PlaybackEngine(
            total_frames=self.store.navigable_frame_count,
            fps=float(value),
            loop=self.loop_check.isChecked(),
        )
        self.playback.start(frame_index=self.current_frame, now_s=time.perf_counter())
        self.schedule_next_render_tick()

    def set_play_button_state(self, playing: bool) -> None:
        commands = getattr(self, "commands", None)
        if commands is not None:
            commands.set_playing(playing)
        if playing:
            self.play_button.setAccessibleName("Pause trajectory")
        else:
            self.play_button.setAccessibleName("Play trajectory")

    def on_frame_slider_changed(self, value: int) -> None:
        if self.internal_slider_change or self.store is None:
            return
        if self.suppress_slider_value == int(value):
            self.suppress_slider_value = None
            return
        if not self.slider_scrub.should_commit_value_change():
            return
        self.commit_frame_seek(int(value))

    def on_frame_slider_pressed(self) -> None:
        if self.store is None:
            return
        self.stop_playback()
        frame_index = int(self.frame_slider.sliderPosition())
        self.slider_scrub.begin(frame_index)
        self.current_frame = frame_index
        self.update_frame_label()
        self.scrub_preview_timer.start()
        self.submit_scrub_preview(time.perf_counter(), force=True)

    def on_frame_slider_moved(self, value: int) -> None:
        if self.store is None:
            return
        frame_index = self.slider_scrub.move(int(value))
        self.current_frame = frame_index
        self.update_frame_label()
        self.submit_scrub_preview(time.perf_counter())

    def on_frame_slider_released(self) -> None:
        if self.store is None:
            return
        self.scrub_preview_timer.stop()
        frame_index = self.slider_scrub.release(int(self.frame_slider.sliderPosition()))
        self.suppress_slider_value = frame_index
        self.commit_frame_seek(frame_index)

    def on_scrub_preview_tick(self) -> None:
        self.submit_scrub_preview(time.perf_counter())

    def submit_scrub_preview(self, now_s: float, *, force: bool = False) -> None:
        if self.store is None or self.streamer is None:
            return
        if not self.slider_scrub.active:
            return
        if not force and not self.slider_scrub.preview_due(now_s):
            return
        frame_index = self.slider_scrub.mark_preview(now_s)
        if frame_index is None:
            return
        self.current_frame = max(0, min(int(frame_index), self.store.frame_count - 1))
        self.displayed_frame = -1
        self.request_stream_frame(self.current_frame, interactive=True)
        self.on_render_tick()

    def commit_frame_seek(self, frame_index: int) -> None:
        if self.store is None:
            return
        self.stop_playback()
        last_available = max(0, self.store.navigable_frame_count - 1)
        self.current_frame = max(0, min(int(frame_index), last_available))
        self.displayed_frame = -1
        self.request_stream_frame(self.current_frame)
        self.update_frame_label()

    def set_slider_value(self, value: int) -> None:
        self.internal_slider_change = True
        self.frame_slider.setValue(value)
        self.internal_slider_change = False

    def step_prev(self) -> None:
        if self.store is None:
            return
        self.stop_playback()
        self.current_frame = max(0, self.current_frame - 1)
        self.displayed_frame = -1
        self.set_slider_value(self.current_frame)
        self.request_stream_frame(self.current_frame, direction=-1)

    def step_next(self) -> None:
        if self.store is None:
            return
        self.stop_playback()
        last_available = max(0, self.store.navigable_frame_count - 1)
        self.current_frame = min(last_available, self.current_frame + 1)
        self.displayed_frame = -1
        self.set_slider_value(self.current_frame)
        self.request_stream_frame(self.current_frame, direction=1)

    def jump_first(self) -> None:
        if self.store is None:
            return
        self.stop_playback()
        self.current_frame = 0
        self.displayed_frame = -1
        self.set_slider_value(0)
        self.request_stream_frame(0, direction=-1)

    def jump_last(self) -> None:
        if self.store is None:
            return
        self.stop_playback()
        self.current_frame = max(0, self.store.navigable_frame_count - 1)
        self.displayed_frame = -1
        self.set_slider_value(self.current_frame)
        self.request_stream_frame(self.current_frame, direction=1)

    def reset_view(self) -> None:
        self.gl_view.reset_view()

    def update_frame_label(self) -> None:
        total = self.store.frame_count if self.store is not None else 0
        current = self.current_frame + 1 if total else 0
        suffix = "+" if self.store is not None and not self.store.frame_count_is_final else ""
        self.frame_label.setText(f"Frame {current} / {total}{suffix}")

    def show_error(self, message: str) -> None:
        self.stop_playback()
        print(f"[error] {message}", flush=True)
        if self.automation_mode:
            self.status_bar.showMessage("Error")
            self.error_reported.emit(message)
            return
        QMessageBox.critical(self, "TrajPlayer", message)
        self.status_bar.showMessage("Error")

    def closeEvent(self, event) -> None:  # type: ignore[override]
        self._pending_open_source = None
        self.close_current_trajectory()
        self.retired_streamer_timer.stop()
        shutdown_deadline = time.monotonic() + 5.0
        retired = tuple(self._retired_open_stores)
        for thread in retired:
            thread.cancel()
        for thread in retired:
            remaining_ms = max(0, int((shutdown_deadline - time.monotonic()) * 1000.0))
            if not thread.wait(remaining_ms):
                print("[shutdown] trajectory worker did not stop before deadline", flush=True)
        for streamer in tuple(self._retired_streamers):
            remaining_s = max(0.0, shutdown_deadline - time.monotonic())
            if not streamer.stop(timeout_s=remaining_s):
                print("[shutdown] frame streamer did not stop before deadline", flush=True)
        for store in set(self._retired_open_stores.values()) | set(
            self._retired_streamers.values()
        ):
            if store is not None and not any(
                worker.isRunning()
                for worker, worker_store in self._retired_open_stores.items()
                if worker_store is store
            ) and not any(
                worker.is_alive
                for worker, worker_store in self._retired_streamers.items()
                if worker_store is store
            ):
                store.close()
        self.stop_bond_inference(wait_ms=0)
        for thread in tuple(self._retired_bond_threads):
            thread.requestInterruption()
        for thread in tuple(self._retired_bond_threads):
            remaining_ms = max(0, int((shutdown_deadline - time.monotonic()) * 1000.0))
            if not thread.wait(remaining_ms):
                print("[shutdown] bond worker did not stop before deadline", flush=True)
        self.gl_view.cleanup()
        super().closeEvent(event)


def main() -> None:
    cli_args = parse_cli_args(sys.argv[1:])
    if cli_args.startup_smoke:
        return
    if cli_args.native_smoke:
        from trajplayer.trajcore import NATIVE_AVAILABLE, NATIVE_IMPORT_ERROR

        if not NATIVE_AVAILABLE:
            print(
                "[native-smoke] trajplayer._trajcore is unavailable: "
                f"{NATIVE_IMPORT_ERROR!r}",
                flush=True,
            )
            raise SystemExit(2)
        print("[native-smoke] trajplayer._trajcore is available", flush=True)
        return
    if cli_args.reader_smoke is not None:
        from trajplayer.reader_smoke import run_reader_smoke, write_reader_smoke_report

        try:
            report = run_reader_smoke(cli_args.reader_smoke)
            if cli_args.reader_smoke_output is not None:
                write_reader_smoke_report(report, cli_args.reader_smoke_output)
            print(f"[reader-smoke] {json.dumps(report, sort_keys=True)}", flush=True)
        except Exception as exc:
            print(f"[reader-smoke] failed: {exc}", flush=True)
            raise SystemExit(2) from exc
        return
    if cli_args.doctor_output is not None:
        from trajplayer.diagnostics import diagnostics_json, probe_opengl

        report = diagnostics_json(
            opengl=probe_opengl(),
            log_path=error_log_path(),
        )
        try:
            cli_args.doctor_output.parent.mkdir(parents=True, exist_ok=True)
            cli_args.doctor_output.write_text(report + "\n", encoding="utf-8")
        except OSError as exc:
            print(f"[doctor] failed to write diagnostics: {exc}", flush=True)
            raise SystemExit(2) from exc
        print(f"[doctor] wrote {cli_args.doctor_output}", flush=True)
        return
    if cli_args.gui_smoke and not cli_args.paths:
        print("[gui-smoke] a trajectory path is required", flush=True)
        raise SystemExit(2)
    benchmark_store: BinaryTrajectoryStore | None = None
    benchmark_label: Path | None = None
    benchmark_metrics: dict[str, object] = {}
    if cli_args.benchmark_output is not None:
        benchmark_store, benchmark_label, benchmark_metrics = prepare_benchmark_store(cli_args)

    QSurfaceFormat.setDefaultFormat(default_surface_format())
    app = TrajPlayerApplication(sys.argv)
    qapplication_created_s = time.perf_counter()
    window = TrajPlayerWindow()
    window.startup_metrics["process_to_qapplication_ms"] = (
        qapplication_created_s - PROCESS_STARTED_S
    ) * 1000.0
    window.show()
    app.set_file_open_handler(
        window.queue_external_open_path,
        replay_pending=not bool(cli_args.paths),
    )
    if cli_args.gui_smoke:
        from trajplayer.gui_smoke import GuiSmokeController

        window.gui_smoke_controller = GuiSmokeController(
            window,
            timeout_ms=cli_args.gui_smoke_timeout_ms,
            output_path=cli_args.gui_smoke_output,
        )
        window.gui_smoke_controller.start()
    if benchmark_store is not None and benchmark_label is not None and cli_args.benchmark_output is not None:
        QTimer.singleShot(
            100,
            lambda: window.start_benchmark(
                label=benchmark_label,
                store=benchmark_store,
                output_path=cli_args.benchmark_output,
                render_frames=cli_args.benchmark_render_frames,
                finish_gpu=cli_args.benchmark_finish_gpu,
                base_metrics=benchmark_metrics,
            ),
        )
    elif cli_args.paths:
        paths = tuple(cli_args.paths[:2])
        QTimer.singleShot(100, lambda: window.load_trajectory_paths(paths))
    if cli_args.smoke_exit_ms is not None:
        QTimer.singleShot(cli_args.smoke_exit_ms, lambda: (window.close(), app.quit()))
    sys.exit(app.exec())


def prepare_benchmark_store(args: CliArgs) -> tuple[BinaryTrajectoryStore, Path, dict[str, object]]:
    from trajplayer.benchmark_store import create_synthetic_store
    from trajplayer.binary_store import BinaryTrajectoryStore

    assert args.benchmark_output is not None
    root = args.benchmark_root or args.benchmark_output.with_suffix(".tpdata")
    root = root.resolve()
    metrics: dict[str, object] = {
        "benchmark_root": str(root),
        "atom_count": args.benchmark_atoms,
        "frame_count": args.benchmark_frames,
        "trajectory_bytes": args.benchmark_atoms * args.benchmark_frames * 3 * 4,
        "target_fps": TrajPlayerWindow.TARGET_FPS,
        "prefetch_radius": TrajPlayerWindow.PREFETCH_RADIUS,
        "finish_gpu": args.benchmark_finish_gpu,
        "benchmark_bonds": args.benchmark_bonds,
        "benchmark_mode": args.benchmark_mode,
    }

    create_s = 0.0
    if not benchmark_store_matches(root, args.benchmark_frames, args.benchmark_atoms):
        start = time.perf_counter()
        with create_synthetic_store(
            root,
            frame_count=args.benchmark_frames,
            atom_count=args.benchmark_atoms,
            chunk_frames=max(1, min(8, args.benchmark_frames)),
        ):
            pass
        create_s = time.perf_counter() - start
    start = time.perf_counter()
    store = BinaryTrajectoryStore.open(root)
    open_s = time.perf_counter() - start
    metrics["cache_create_s"] = create_s
    metrics["cache_open_s"] = open_s
    return store, root, metrics


def benchmark_store_matches(root: Path, frame_count: int, atom_count: int) -> bool:
    from trajplayer.binary_store import BinaryTrajectoryStore

    try:
        with BinaryTrajectoryStore.open(root, mode="r") as store:
            return (
                bool(store.metadata.get("synthetic"))
                and store.frame_count == frame_count
                and store.atom_count == atom_count
                and store.positions.dtype.name == "float32"
            )
    except Exception:
        return False


if __name__ == "__main__":
    main()
