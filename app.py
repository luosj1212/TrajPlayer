from __future__ import annotations

import json
import math
import sys
import threading
import time
import traceback
from pathlib import Path
from typing import Iterable

from trajplayer.startup import initialize_runtime, report_numpy_import_error


initialize_runtime()

try:
    import numpy as np
except Exception as exc:
    report_numpy_import_error(exc)
    raise SystemExit(1) from None

from PySide6.QtCore import QSize, QThread, QTimer, Qt, Signal
from PySide6.QtGui import QKeySequence, QShortcut, QSurfaceFormat
from PySide6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QProgressBar,
    QSizePolicy,
    QSlider,
    QStatusBar,
    QStyle,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from trajplayer.ase_cache import ConversionCancelled, build_cache_from_source, open_valid_cache
from trajplayer.benchmark_stats import BenchmarkDiagnostics
from trajplayer.benchmark_store import create_synthetic_store
from trajplayer.binary_store import BinaryTrajectoryStore
from trajplayer.bonds import connected_components, infer_bonds
from trajplayer.cli_args import CliArgs, parse_cli_args
from trajplayer.gl_view import MoleculeGLWidget, default_surface_format
from trajplayer.playback import PlaybackEngine
from trajplayer.random_access_cache import (
    open_random_access_session,
    supports_random_access_source,
    write_reader_frame,
)
from trajplayer.scrubbing import SliderScrubState
from trajplayer.streaming import FrameStreamer
from trajplayer.trajectory_source import (
    TrajectorySelectionError,
    TrajectorySource,
    paths_are_supported_for_drop,
    resolve_trajectory_source,
)


APP_STYLESHEET = """
QMainWindow, QWidget#centralWidget {
    background: #f4f6f8;
    color: #20242a;
    font-size: 10pt;
}
QFrame#topBar {
    background: #ffffff;
    border-bottom: 1px solid #dfe3e8;
}
QFrame#transportBar {
    background: #ffffff;
    border-top: 1px solid #dfe3e8;
}
QLabel#fileLabel {
    color: #20242a;
    font-size: 10pt;
    font-weight: 600;
}
QLabel#infoLabel, QLabel#fpsLabel {
    color: #68717d;
    font-size: 9pt;
}
QLabel#frameLabel {
    color: #303740;
    font-family: "Consolas";
    font-size: 9pt;
}
QLabel#controlLabel {
    color: #59636f;
    font-size: 9pt;
}
QLabel#sizeValueLabel {
    color: #303740;
    font-family: "Consolas";
    font-size: 9pt;
}
QPushButton, QToolButton {
    min-height: 30px;
    border: 1px solid #cdd3da;
    border-radius: 4px;
    background: #ffffff;
    color: #20242a;
    padding: 0 10px;
}
QPushButton:hover, QToolButton:hover {
    border-color: #8aa8c7;
    background: #eef4fa;
}
QPushButton:pressed, QToolButton:pressed {
    background: #dfeaf5;
}
QPushButton:disabled, QToolButton:disabled {
    color: #9ca3ab;
    border-color: #e3e6ea;
    background: #f7f8f9;
}
QPushButton#openButton {
    background: #1769aa;
    border-color: #1769aa;
    color: #ffffff;
    font-weight: 600;
}
QPushButton#openButton:hover {
    background: #125b94;
    border-color: #125b94;
}
QToolButton#playButton {
    min-width: 38px;
    min-height: 38px;
    background: #1769aa;
    border-color: #1769aa;
}
QToolButton#playButton:hover {
    background: #125b94;
    border-color: #125b94;
}
QCheckBox {
    color: #38414b;
    spacing: 6px;
}
QCheckBox::indicator {
    width: 15px;
    height: 15px;
}
QComboBox {
    min-height: 30px;
    border: 1px solid #cdd3da;
    border-radius: 4px;
    background: #ffffff;
    color: #20242a;
    padding: 0 7px;
}
QComboBox:hover {
    border-color: #8aa8c7;
}
QComboBox:disabled {
    color: #9ca3ab;
    border-color: #e3e6ea;
    background: #f7f8f9;
}
QFrame#filterModeSegment {
    min-height: 30px;
    max-height: 30px;
    border: 1px solid #cdd3da;
    border-radius: 4px;
    background: #edf1f5;
}
QToolButton#filterModeButton {
    min-height: 28px;
    max-height: 28px;
    border: 0;
    border-radius: 3px;
    background: transparent;
    color: #4f5965;
    padding: 0 7px;
}
QToolButton#filterModeButton:hover {
    background: #e2e9f0;
    color: #20242a;
}
QToolButton#filterModeButton:checked {
    background: #1769aa;
    color: #ffffff;
    font-weight: 600;
}
QToolButton#filterModeButton:disabled {
    background: transparent;
    color: #a5acb4;
}
QLabel#filterValueLabel {
    color: #303740;
    font-family: "Consolas";
    font-size: 9pt;
}
QLabel#filterValueLabel:disabled {
    color: #9ca3ab;
}
QSlider::groove:horizontal {
    height: 4px;
    border-radius: 2px;
    background: #d7dce2;
}
QSlider::sub-page:horizontal {
    border-radius: 2px;
    background: #2b6fae;
}
QSlider::handle:horizontal {
    width: 14px;
    height: 14px;
    margin: -5px 0;
    border: 2px solid #2b6fae;
    border-radius: 7px;
    background: #ffffff;
}
QSlider:disabled::handle:horizontal {
    border-color: #aeb5bd;
}
QSlider#sizeSlider::groove:horizontal {
    height: 3px;
}
QSlider#sizeSlider::handle:horizontal {
    width: 12px;
    height: 12px;
    margin: -5px 0;
    border-width: 1px;
    border-radius: 6px;
}
QProgressBar {
    min-height: 3px;
    max-height: 3px;
    border: 0;
    background: #e6e9ed;
}
QProgressBar::chunk {
    background: #2f855a;
}
QStatusBar {
    background: #f7f8fa;
    color: #626b76;
    border-top: 1px solid #dfe3e8;
}
"""


class TrajectoryOpenThread(QThread):
    preview_ready = Signal(object, object)
    loaded = Signal(object, object, bool)
    failed = Signal(str)
    progress = Signal(int, int)
    stage_changed = Signal(str)
    cache_frame_ready = Signal(int)

    def __init__(self, source: TrajectorySource) -> None:
        super().__init__()
        self.source = source
        self._cancel_event = threading.Event()
        self._preview_store: BinaryTrajectoryStore | None = None
        self._request_condition = threading.Condition()
        self._requested_indices: tuple[int, ...] = ()
        self._request_serial = 0

    @property
    def preview_store(self) -> BinaryTrajectoryStore | None:
        return self._preview_store

    def cancel(self) -> None:
        self._cancel_event.set()
        with self._request_condition:
            self._request_condition.notify_all()

    def request_frames(self, frame_indices: Iterable[int]) -> None:
        requested = tuple(dict.fromkeys(max(0, int(index)) for index in frame_indices))
        with self._request_condition:
            if requested == self._requested_indices:
                return
            self._requested_indices = requested
            self._request_serial += 1
            self._request_condition.notify_all()

    def run(self) -> None:
        store: BinaryTrajectoryStore | None = None
        reader = None
        last_progress_s = 0.0

        def emit_progress(done: int, total: int) -> None:
            nonlocal last_progress_s
            now_s = time.monotonic()
            if done == 1 or done == total or now_s - last_progress_s >= 0.05:
                last_progress_s = now_s
                self.progress.emit(done, total)

        def emit_preview(preview_store: BinaryTrajectoryStore) -> None:
            self._preview_store = preview_store
            self.preview_ready.emit(self.source, preview_store)

        try:
            try:
                store = open_valid_cache(self.source)
                from_cache = True
            except Exception:
                from_cache = False
                if supports_random_access_source(self.source):
                    reader, store = open_random_access_session(
                        self.source,
                        status_callback=self.stage_changed.emit,
                    )
                    if not store.is_frame_available(0):
                        write_reader_frame(reader, store, 0)
                    emit_preview(store)
                    emit_progress(store.available_frame_count, store.frame_count)
                    self._fill_random_access_cache(reader, store, emit_progress)
                    if not self._cancel_event.is_set():
                        store.mark_complete()
                else:
                    store = build_cache_from_source(
                        self.source,
                        progress_callback=emit_progress,
                        preview_callback=emit_preview,
                        cancel_event=self._cancel_event,
                    )
            if self._cancel_event.is_set():
                if store is not None and self._preview_store is None:
                    store.close()
                return
            self.loaded.emit(self.source, store, from_cache)
        except ConversionCancelled:
            if store is not None and self._preview_store is None:
                store.close()
        except Exception as exc:
            failed_store = store if store is not None else self._preview_store
            if failed_store is not None and self._preview_store is None:
                failed_store.close()
            traceback.print_exc()
            self.failed.emit(f"Failed to open trajectory:\n{exc}")
        finally:
            if reader is not None:
                reader.close()

    def _fill_random_access_cache(
        self,
        reader,
        store: BinaryTrajectoryStore,
        progress_callback,
    ) -> None:
        background_index = store.available_prefix_count
        while (
            not self._cancel_event.is_set()
            and store.available_frame_count < store.frame_count
        ):
            with self._request_condition:
                request_serial = self._request_serial
                requested = self._requested_indices

            frame_index = next(
                (index for index in requested if index < store.frame_count and not store.is_frame_available(index)),
                None,
            )
            requested_frame = frame_index is not None
            if frame_index is None:
                while background_index < store.frame_count and store.is_frame_available(background_index):
                    background_index += 1
                if background_index >= store.frame_count:
                    break
                frame_index = background_index

            write_reader_frame(reader, store, frame_index)
            if frame_index == background_index:
                background_index += 1

            if requested_frame:
                with self._request_condition:
                    is_current_target = (
                        request_serial == self._request_serial
                        and bool(self._requested_indices)
                        and frame_index == self._requested_indices[0]
                    )
                if is_current_target:
                    self.cache_frame_ready.emit(frame_index)
            progress_callback(store.available_frame_count, store.frame_count)


class BondInferenceThread(QThread):
    ready = Signal(int, object, object, object, float)
    failed = Signal(int, str)

    def __init__(
        self,
        generation: int,
        positions: np.ndarray,
        atom_numbers: np.ndarray,
        cell: np.ndarray | None,
    ) -> None:
        super().__init__()
        self.generation = int(generation)
        self.positions = np.ascontiguousarray(positions, dtype=np.float32)
        self.atom_numbers = np.ascontiguousarray(atom_numbers, dtype=np.uint16)
        self.cell = None if cell is None else np.ascontiguousarray(cell, dtype=np.float32)

    def run(self) -> None:
        start = time.perf_counter()
        try:
            bonds = infer_bonds(
                self.positions,
                self.atom_numbers,
                cell=self.cell,
                cancelled=lambda: self.isInterruptionRequested(),
            )
            if self.isInterruptionRequested():
                return
            component_ids, component_sizes = connected_components(len(self.atom_numbers), bonds)
            if self.isInterruptionRequested():
                return
            self.ready.emit(
                self.generation,
                bonds,
                component_ids,
                component_sizes,
                (time.perf_counter() - start) * 1000.0,
            )
        except Exception as exc:
            traceback.print_exc()
            self.failed.emit(self.generation, f"Failed to infer bonds: {exc}")


class TrajPlayerWindow(QMainWindow):
    stream_frame_ready = Signal(int)

    TARGET_FPS = 60.0
    PREFETCH_RADIUS = 200
    IDLE_RENDER_TIMER_MS = 16
    SCRUB_PREVIEW_TIMER_MS = 4
    SCRUB_PREVIEW_FPS = 60.0

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("TrajPlayer GPU")
        self.resize(1280, 820)
        self.setAcceptDrops(True)
        self.setStyleSheet(APP_STYLESHEET)

        self.store: BinaryTrajectoryStore | None = None
        self.streamer: FrameStreamer | None = None
        self.open_thread: TrajectoryOpenThread | None = None
        self._pending_open_source: TrajectorySource | None = None
        self._retired_open_stores: dict[
            TrajectoryOpenThread, BinaryTrajectoryStore | None
        ] = {}
        self.cache_build_in_progress = False
        self.bond_thread: BondInferenceThread | None = None
        self._retired_bond_threads: set[BondInferenceThread] = set()
        self.playback: PlaybackEngine | None = None
        self.current_frame = 0
        self.displayed_frame = -1
        self.last_stream_seek_frame = 0
        self.trajectory_generation = 0
        self.reset_view_on_next_frame = False
        self.internal_slider_change = False
        self.suppress_slider_value: int | None = None
        self.component_ids = np.empty((0,), dtype=np.int32)
        self.component_sizes = np.empty((0,), dtype=np.int32)
        self.filter_mode = "all"
        self.filter_values = {"chain": 1, "atom": 1}
        self.slider_scrub = SliderScrubState(preview_interval_s=1.0 / self.SCRUB_PREVIEW_FPS)
        self.benchmark_output: Path | None = None
        self.benchmark_target_frames = 0
        self.benchmark_started_s = 0.0
        self.benchmark_base_metrics: dict[str, object] = {}
        self.benchmark_diagnostics: BenchmarkDiagnostics | None = None
        self.benchmark_finish_gpu = False
        self.benchmark_warmup_started_s = 0.0

        self.gl_view = MoleculeGLWidget()
        self.stream_frame_ready.connect(self.on_stream_frame_ready)

        self.open_button = QPushButton("Open")
        self.open_button.setObjectName("openButton")
        self.open_button.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_DialogOpenButton))
        self.open_button.setIconSize(QSize(17, 17))
        self.open_button.setToolTip("Open trajectory (Ctrl+O)")
        self.open_button.clicked.connect(self.open_file)

        self.prev_button = QToolButton()
        self.prev_button.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaSkipBackward))
        self.prev_button.setIconSize(QSize(18, 18))
        self.prev_button.setToolTip("Previous frame (Left)")
        self.prev_button.setAccessibleName("Previous frame")
        self.prev_button.clicked.connect(self.step_prev)
        self.prev_button.setEnabled(False)

        self._play_icon = self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPlay)
        self._pause_icon = self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPause)
        self.play_button = QToolButton()
        self.play_button.setObjectName("playButton")
        self.play_button.setIconSize(QSize(20, 20))
        self.play_button.setAccessibleName("Play trajectory")
        self.play_button.clicked.connect(self.toggle_playback)
        self.play_button.setEnabled(False)
        self.set_play_button_state(False)

        self.next_button = QToolButton()
        self.next_button.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaSkipForward))
        self.next_button.setIconSize(QSize(18, 18))
        self.next_button.setToolTip("Next frame (Right)")
        self.next_button.setAccessibleName("Next frame")
        self.next_button.clicked.connect(self.step_next)
        self.next_button.setEnabled(False)

        self.reset_view_button = QToolButton()
        self.reset_view_button.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_BrowserReload))
        self.reset_view_button.setIconSize(QSize(17, 17))
        self.reset_view_button.setToolTip("Reset view (R)")
        self.reset_view_button.setAccessibleName("Reset view")
        self.reset_view_button.clicked.connect(self.reset_view)
        self.reset_view_button.setEnabled(False)

        self.loop_check = QCheckBox("Loop")
        self.loop_check.setChecked(True)

        self.box_check = QCheckBox("Box")
        self.box_check.setChecked(True)
        self.box_check.setEnabled(False)
        self.box_check.toggled.connect(self.on_box_toggled)

        self.playback_speed_label = QLabel("Speed")
        self.playback_speed_label.setObjectName("controlLabel")
        self.playback_speed_label.setEnabled(False)
        self.playback_speed_slider = QSlider(Qt.Orientation.Horizontal)
        self.playback_speed_slider.setObjectName("sizeSlider")
        self.playback_speed_slider.setRange(1, int(self.TARGET_FPS))
        self.playback_speed_slider.setSingleStep(1)
        self.playback_speed_slider.setPageStep(5)
        self.playback_speed_slider.setValue(int(self.TARGET_FPS))
        self.playback_speed_slider.setFixedWidth(96)
        self.playback_speed_slider.setToolTip("Playback speed")
        self.playback_speed_slider.setEnabled(False)
        self.playback_speed_slider.valueChanged.connect(self.on_playback_speed_changed)
        self.playback_speed_value_label = QLabel(f"{int(self.TARGET_FPS)} FPS")
        self.playback_speed_value_label.setObjectName("sizeValueLabel")
        self.playback_speed_value_label.setAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        )
        self.playback_speed_value_label.setFixedWidth(48)
        self.playback_speed_value_label.setEnabled(False)

        self.render_mode_combo = QComboBox()
        self.render_mode_combo.addItem("Ball-stick", "ball_stick")
        self.render_mode_combo.addItem("Ball", "ball")
        self.render_mode_combo.addItem("Bond", "bond")
        self.render_mode_combo.setFixedWidth(112)
        self.render_mode_combo.setToolTip("Molecular representation")
        self.render_mode_combo.setEnabled(False)
        self.render_mode_combo.currentIndexChanged.connect(self.on_render_mode_changed)

        self.atom_size_label = QLabel("Atom")
        self.atom_size_label.setObjectName("controlLabel")
        self.atom_size_label.setEnabled(False)
        self.atom_size_slider = QSlider(Qt.Orientation.Horizontal)
        self.atom_size_slider.setObjectName("sizeSlider")
        self.atom_size_slider.setRange(10, 250)
        self.atom_size_slider.setSingleStep(5)
        self.atom_size_slider.setPageStep(25)
        self.atom_size_slider.setValue(100)
        self.atom_size_slider.setFixedWidth(96)
        self.atom_size_slider.setToolTip(
            "Atom radius: 100% is Chimera ball scale in Ball-stick, physical VDW radius in Ball"
        )
        self.atom_size_slider.setEnabled(False)
        self.atom_size_slider.valueChanged.connect(self.on_atom_size_changed)
        self.atom_size_value_label = QLabel("100%")
        self.atom_size_value_label.setObjectName("sizeValueLabel")
        self.atom_size_value_label.setAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        )
        self.atom_size_value_label.setFixedWidth(40)
        self.atom_size_value_label.setEnabled(False)

        self.bond_size_label = QLabel("Bond")
        self.bond_size_label.setObjectName("controlLabel")
        self.bond_size_label.setEnabled(False)
        self.bond_size_slider = QSlider(Qt.Orientation.Horizontal)
        self.bond_size_slider.setObjectName("sizeSlider")
        self.bond_size_slider.setRange(10, 300)
        self.bond_size_slider.setSingleStep(5)
        self.bond_size_slider.setPageStep(25)
        self.bond_size_slider.setValue(100)
        self.bond_size_slider.setFixedWidth(96)
        self.bond_size_slider.setToolTip("Bond radius: 100% is 0.20 angstrom")
        self.bond_size_slider.setEnabled(False)
        self.bond_size_slider.valueChanged.connect(self.on_bond_size_changed)
        self.bond_size_value_label = QLabel("100%")
        self.bond_size_value_label.setObjectName("sizeValueLabel")
        self.bond_size_value_label.setAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        )
        self.bond_size_value_label.setFixedWidth(40)
        self.bond_size_value_label.setEnabled(False)

        self.filter_mode_segment = QFrame()
        self.filter_mode_segment.setObjectName("filterModeSegment")
        filter_mode_layout = QHBoxLayout(self.filter_mode_segment)
        filter_mode_layout.setContentsMargins(1, 1, 1, 1)
        filter_mode_layout.setSpacing(0)
        self.filter_mode_group = QButtonGroup(self)
        self.filter_mode_group.setExclusive(True)
        self.filter_mode_buttons: dict[str, QToolButton] = {}
        for label, mode, width in (("All", "all", 38), ("Chain", "chain", 52), ("Atom", "atom", 46)):
            button = QToolButton()
            button.setObjectName("filterModeButton")
            button.setText(label)
            button.setCheckable(True)
            button.setFixedWidth(width)
            button.setEnabled(False)
            button.setToolTip(f"Show {label.lower() if mode != 'all' else 'all atoms'}")
            button.setAccessibleName(f"Show {label.lower()}")
            button.clicked.connect(
                lambda _checked=False, selected_mode=mode: self.on_filter_mode_changed(selected_mode)
            )
            self.filter_mode_group.addButton(button)
            self.filter_mode_buttons[mode] = button
            filter_mode_layout.addWidget(button)
        self.filter_mode_buttons["all"].setChecked(True)

        self.filter_value_slider = QSlider(Qt.Orientation.Horizontal)
        self.filter_value_slider.setObjectName("filterValueSlider")
        self.filter_value_slider.setRange(1, 1)
        self.filter_value_slider.setSingleStep(1)
        self.filter_value_slider.setPageStep(1)
        self.filter_value_slider.setValue(1)
        self.filter_value_slider.setFixedWidth(92)
        self.filter_value_slider.setToolTip("All atoms are visible")
        self.filter_value_slider.setEnabled(False)
        self.filter_value_slider.valueChanged.connect(self.on_filter_value_changed)
        self.filter_value_label = QLabel("All atoms")
        self.filter_value_label.setObjectName("filterValueLabel")
        self.filter_value_label.setAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        )
        self.filter_value_label.setFixedWidth(76)
        self.filter_value_label.setEnabled(False)

        self.frame_label = QLabel("Frame 0 / 0")
        self.frame_label.setObjectName("frameLabel")
        self.frame_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self.frame_label.setMinimumWidth(150)
        self.file_label = QLabel("Drop a trajectory here or click Open")
        self.file_label.setObjectName("fileLabel")
        self.file_label.setMinimumWidth(0)
        self.file_label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        self.info_label = QLabel("GPU instancing idle")
        self.info_label.setObjectName("infoLabel")

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 1)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(False)
        self.progress_bar.hide()

        self.frame_slider = QSlider(Qt.Orientation.Horizontal)
        self.frame_slider.setMinimum(0)
        self.frame_slider.setMaximum(0)
        self.frame_slider.setEnabled(False)
        self.frame_slider.setTracking(False)
        self.frame_slider.sliderPressed.connect(self.on_frame_slider_pressed)
        self.frame_slider.sliderMoved.connect(self.on_frame_slider_moved)
        self.frame_slider.sliderReleased.connect(self.on_frame_slider_released)
        self.frame_slider.valueChanged.connect(self.on_frame_slider_changed)

        top_bar = QFrame()
        top_bar.setObjectName("topBar")
        top_layout = QHBoxLayout(top_bar)
        top_layout.setContentsMargins(12, 8, 12, 8)
        top_layout.setSpacing(12)
        top_layout.addWidget(self.open_button)
        file_layout = QVBoxLayout()
        file_layout.setContentsMargins(0, 0, 0, 0)
        file_layout.setSpacing(1)
        file_layout.addWidget(self.file_label)
        file_layout.addWidget(self.info_label)
        top_layout.addLayout(file_layout, stretch=1)

        transport_bar = QFrame()
        transport_bar.setObjectName("transportBar")
        transport_layout = QVBoxLayout(transport_bar)
        transport_layout.setContentsMargins(12, 9, 12, 10)
        transport_layout.setSpacing(8)
        transport_layout.addWidget(self.frame_slider)

        controls = QHBoxLayout()
        controls.setSpacing(6)
        controls.addWidget(self.prev_button)
        controls.addWidget(self.play_button)
        controls.addWidget(self.next_button)
        controls.addSpacing(6)
        controls.addWidget(self.reset_view_button)
        controls.addSpacing(10)
        controls.addWidget(self.loop_check)
        controls.addWidget(self.box_check)
        controls.addStretch(1)
        controls.addWidget(self.filter_mode_segment)
        controls.addSpacing(4)
        controls.addWidget(self.filter_value_slider)
        controls.addWidget(self.filter_value_label)
        transport_layout.addLayout(controls)

        display_controls = QHBoxLayout()
        display_controls.setSpacing(6)
        display_controls.addWidget(self.render_mode_combo)
        display_controls.addSpacing(8)
        display_controls.addWidget(self.atom_size_label)
        display_controls.addWidget(self.atom_size_slider)
        display_controls.addWidget(self.atom_size_value_label)
        display_controls.addSpacing(8)
        display_controls.addWidget(self.bond_size_label)
        display_controls.addWidget(self.bond_size_slider)
        display_controls.addWidget(self.bond_size_value_label)
        display_controls.addStretch(1)
        display_controls.addWidget(self.playback_speed_label)
        display_controls.addWidget(self.playback_speed_slider)
        display_controls.addWidget(self.playback_speed_value_label)
        display_controls.addSpacing(8)
        display_controls.addWidget(self.frame_label)
        transport_layout.addLayout(display_controls)

        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(top_bar)
        layout.addWidget(self.progress_bar)
        layout.addWidget(self.gl_view, stretch=1)
        layout.addWidget(transport_bar)

        central = QWidget()
        central.setObjectName("centralWidget")
        central.setLayout(layout)
        self.setCentralWidget(central)

        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready")

        self.render_timer = QTimer(self)
        self.render_timer.setTimerType(Qt.TimerType.PreciseTimer)
        self.render_timer.setInterval(self.IDLE_RENDER_TIMER_MS)
        self.render_timer.timeout.connect(self.on_render_tick)
        self.render_timer.start()
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

        self._shortcuts: list[QShortcut] = []
        for key, slot in (
            (QKeySequence(Qt.Key.Key_Space), self.toggle_playback),
            (QKeySequence(Qt.Key.Key_Left), self.step_prev),
            (QKeySequence(Qt.Key.Key_Right), self.step_next),
            (QKeySequence(Qt.Key.Key_Home), self.jump_first),
            (QKeySequence(Qt.Key.Key_End), self.jump_last),
            (QKeySequence("R"), self.reset_view),
            (QKeySequence("O"), self.open_file),
        ):
            shortcut = QShortcut(key, self)
            shortcut.activated.connect(slot)
            self._shortcuts.append(shortcut)

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
        self.set_loading_state(source)
        thread = TrajectoryOpenThread(source)
        thread.progress.connect(
            lambda done, total, worker=thread: self.on_open_progress(worker, done, total)
        )
        thread.stage_changed.connect(
            lambda message, worker=thread: self.on_open_stage_changed(worker, message)
        )
        thread.cache_frame_ready.connect(
            lambda frame_index, worker=thread: self.on_cache_frame_ready(worker, frame_index)
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
        self.info_label.setText("Preparing binary float32 trajectory cache")
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
        self.progress_bar.setRange(0, max(total, 1))
        self.progress_bar.setValue(done)
        if self.streamer is not None:
            self.streamer.notify_store_updated()
        if self.store is not None and self.cache_build_in_progress:
            self.update_available_controls()
        operation = "Caching requested and nearby frames" if (
            self.store is not None and self.store.supports_random_access
        ) else "Converting to contiguous float32 cache"
        self.status_bar.showMessage(f"{operation}: {done}/{total} frames")

    def on_cache_frame_ready(
        self,
        thread: TrajectoryOpenThread,
        frame_index: int,
    ) -> None:
        if thread is not self.open_thread or self.streamer is None:
            return
        self.streamer.notify_store_updated()
        if int(frame_index) == self.current_frame:
            self.request_stream_frame(self.current_frame, interactive=self.slider_scrub.active)

    def on_trajectory_preview(
        self,
        thread: TrajectoryOpenThread,
        source: TrajectorySource,
        store: BinaryTrajectoryStore,
    ) -> None:
        if thread is not self.open_thread:
            return
        if self.store is not None:
            return
        self.cache_build_in_progress = True
        self.activate_trajectory(source, store)
        self.progress_bar.show()
        self.open_button.setEnabled(True)
        self.update_available_controls()
        self.status_bar.showMessage(
            f"First frame ready; on-demand cache {store.available_frame_count}/{store.frame_count} frames"
        )

    def on_trajectory_loaded(
        self,
        thread: TrajectoryOpenThread,
        source: TrajectorySource,
        store: BinaryTrajectoryStore,
        from_cache: bool,
    ) -> None:
        if thread is not self.open_thread:
            return
        self.open_thread = None
        self.finish_trajectory_load(source, store, from_cache=from_cache)

    def finish_trajectory_load(
        self,
        source: TrajectorySource,
        store: BinaryTrajectoryStore,
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
        source_label = "cache" if from_cache else "new binary cache"
        self.update_trajectory_info()
        self.status_bar.showMessage(f"Loaded {source.display_name} from {source_label}")
        self.open_button.setEnabled(True)
        self.set_controls_enabled(store.frame_count > 1)
        self.reset_view_button.setEnabled(True)
        self.update_frame_label()

    def on_open_thread_finished(self, thread: TrajectoryOpenThread) -> None:
        retired_store = self._retired_open_stores.pop(thread, None)
        if retired_store is not None:
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
        store: BinaryTrajectoryStore,
    ) -> None:
        self.store = store
        self.streamer = FrameStreamer(
            store,
            prefetch_radius=self.PREFETCH_RADIUS,
            frame_ready_callback=self.stream_frame_ready.emit,
        )
        self.streamer.start()
        self.current_frame = 0
        self.displayed_frame = -1
        self.last_stream_seek_frame = 0
        self.reset_view_on_next_frame = True
        self.component_ids = np.empty((0,), dtype=np.int32)
        self.component_sizes = np.empty((0,), dtype=np.int32)
        self.gl_view.set_atoms(store.atom_numbers)
        self.gl_view.set_render_mode(str(self.render_mode_combo.currentData()))
        self.gl_view.set_atom_size_scale(self.atom_size_slider.value() / 100.0)
        self.gl_view.set_bond_size_scale(self.bond_size_slider.value() / 100.0)
        self.set_representation_controls_enabled(True)
        self.gl_view.set_box_enabled(self.box_check.isChecked())
        self.box_check.setEnabled(store.has_cells)
        self.configure_filter_controls(store.atom_count)
        self.start_bond_inference(store)
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
        self.reset_view_button.setEnabled(True)
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
        bond_text = f", {self.gl_view.bond_count} bonds" if self.gl_view.bond_count else ""
        cache_text = ""
        if self.streamer is not None:
            cache_mib = self.streamer.memory_bytes / (1024.0 * 1024.0)
            cache_text = f", {self.streamer.capacity}-frame/{cache_mib:.0f} MiB cache"
        self.info_label.setText(
            f"{self.store.frame_count} frames, {self.store.atom_count} atoms{bond_text}, "
            f"directional prefetch{cache_text}, "
            f"{self.store.available_frame_count}/{self.store.frame_count} frames on disk, "
            f"{self.render_mode_combo.currentText()} GPU instancing"
        )

    def configure_filter_controls(self, atom_count: int) -> None:
        enabled = atom_count > 0
        largest_index = max(1, int(atom_count))
        label_width = max(
            76,
            self.filter_value_label.fontMetrics().horizontalAdvance(f"Atom {largest_index}") + 6,
            self.filter_value_label.fontMetrics().horizontalAdvance(f"Chain {largest_index}") + 6,
        )
        self.filter_value_label.setFixedWidth(label_width)
        self.filter_mode = "all"
        self.filter_values = {"chain": 1, "atom": 1}
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

    def on_filter_mode_changed(self, mode: str) -> None:
        if self.store is None:
            return
        mode = str(mode)
        self.filter_value_slider.blockSignals(True)
        if mode == "all":
            self.filter_mode = "all"
            self.filter_mode_buttons["all"].setChecked(True)
            self.filter_value_slider.setEnabled(False)
            self.filter_value_label.setText("All atoms")
            self.filter_value_label.setEnabled(False)
            self.filter_value_slider.setToolTip("All atoms are visible")
        elif mode == "chain":
            if self.component_sizes.size == 0:
                self.filter_mode = "all"
                self.filter_mode_buttons["all"].setChecked(True)
                self.filter_value_slider.setEnabled(False)
                self.filter_value_label.setText("All atoms")
                self.filter_value_label.setEnabled(False)
                self.status_bar.showMessage("Chain groups are still being prepared")
                self.filter_value_slider.blockSignals(False)
                return
            self.filter_mode = "chain"
            maximum = int(self.component_sizes.shape[0])
            self.filter_value_slider.setRange(1, maximum)
            self.filter_value_slider.setPageStep(1)
            self.filter_value_slider.setValue(min(self.filter_values["chain"], maximum))
            self.filter_value_slider.setEnabled(True)
            self.filter_value_label.setEnabled(True)
        else:
            self.filter_mode = "atom"
            maximum = self.store.atom_count
            self.filter_value_slider.setRange(1, maximum)
            self.filter_value_slider.setPageStep(max(1, maximum // 100))
            self.filter_value_slider.setValue(min(self.filter_values["atom"], maximum))
            self.filter_value_slider.setEnabled(True)
            self.filter_value_label.setEnabled(True)
        self.filter_value_slider.blockSignals(False)
        self.update_filter_value_label()
        self.schedule_visibility_filter()

    def on_filter_value_changed(self, value: int) -> None:
        if self.filter_mode in self.filter_values:
            self.filter_values[self.filter_mode] = int(value)
        self.update_filter_value_label()
        self.schedule_visibility_filter()

    def update_filter_value_label(self) -> None:
        if self.filter_mode == "all":
            self.filter_value_label.setText("All atoms")
            return
        value = self.filter_value_slider.value()
        title = "Chain" if self.filter_mode == "chain" else "Atom"
        self.filter_value_label.setText(f"{title} {value}")
        self.filter_value_slider.setToolTip(
            f"{title} {value} of {self.filter_value_slider.maximum()}"
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
        elif mode == "chain":
            component_index = self.filter_value_slider.value() - 1
            if component_index < 0 or component_index >= self.component_sizes.shape[0]:
                return
            visible_atoms = np.flatnonzero(self.component_ids == component_index).astype(
                np.int32,
                copy=False,
            )
            message = f"Showing chain {component_index + 1}: {len(visible_atoms)} atoms"
        else:
            atom_index = self.filter_value_slider.value() - 1
            visible_atoms = np.array([atom_index], dtype=np.int32)
            message = f"Showing atom {atom_index + 1}"
        self.gl_view.set_visible_atoms(
            visible_atoms,
            fit_view=self.displayed_frame >= 0,
            unwrap_periodic=mode == "chain",
        )
        self.status_bar.showMessage(message)

    def start_bond_inference(self, store: BinaryTrajectoryStore) -> None:
        self.stop_bond_inference(wait_ms=0)
        self.trajectory_generation += 1
        self.gl_view.set_bonds(np.empty((0, 2), dtype=np.int32))
        if store.metadata.get("synthetic"):
            self.status_bar.showMessage("Synthetic benchmark loaded without bond inference")
            return

        first_frame = np.ascontiguousarray(store.frame(0), dtype=np.float32)
        atom_numbers = np.ascontiguousarray(store.atom_numbers, dtype=np.uint16)
        cell = store.cell(0)
        thread = BondInferenceThread(
            self.trajectory_generation,
            first_frame,
            atom_numbers,
            cell,
        )
        thread.ready.connect(self.on_bonds_ready)
        thread.failed.connect(self.on_bonds_failed)
        thread.finished.connect(lambda thread=thread: self.on_bond_thread_finished(thread))
        self.bond_thread = thread
        self.status_bar.showMessage("Inferring bonds in the background")
        thread.start()

    def stop_bond_inference(self, *, wait_ms: int) -> None:
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
        bonds: np.ndarray,
        component_ids: np.ndarray,
        component_sizes: np.ndarray,
        elapsed_ms: float,
    ) -> None:
        if generation != self.trajectory_generation or self.store is None:
            return
        self.component_ids = np.ascontiguousarray(component_ids, dtype=np.int32)
        self.component_sizes = np.ascontiguousarray(component_sizes, dtype=np.int32)
        self.filter_mode_buttons["chain"].setEnabled(self.component_sizes.size > 0)
        self.gl_view.set_bonds(bonds)
        self.update_trajectory_info()
        self.status_bar.showMessage(
            f"Bonds ready: {len(bonds)} bonds, {len(self.component_sizes)} chains in {elapsed_ms:.0f} ms"
        )

    def on_bonds_failed(self, generation: int, message: str) -> None:
        if generation != self.trajectory_generation:
            return
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
        self.gl_view.enable_benchmark_stats(finish_gpu=self.benchmark_finish_gpu)
        self.benchmark_started_s = time.perf_counter()
        self.playback = PlaybackEngine(
            total_frames=self.store.frame_count,
            fps=self.TARGET_FPS,
            loop=True,
        )
        self.playback.start(frame_index=0, now_s=time.perf_counter())
        self.set_play_button_state(True)
        self.gl_view.set_immediate_paint(True)
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
        elapsed_s = max(0.0, time.perf_counter() - self.benchmark_started_s)
        frames = int(render_summary.get("frames", 0))
        result = {
            **self.benchmark_base_metrics,
            "timed_out": timed_out,
            "benchmark_elapsed_s": elapsed_s,
            "measured_fps": render_summary.get("cadence_fps", 0.0),
            "render": render_summary,
            "pipeline": self.benchmark_diagnostics.summary()
            if self.benchmark_diagnostics is not None
            else {},
            "streamer_memory_bytes": self.streamer.memory_bytes if self.streamer is not None else 0,
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
        deferred_store: BinaryTrajectoryStore | None = None
        if thread is not None and thread.isRunning():
            thread.cancel()
            if self.store is not None and self.store is thread.preview_store:
                deferred_store = self.store
            self._retired_open_stores[thread] = deferred_store
        self.open_thread = None

        self.stop_playback()
        self.cache_build_in_progress = False
        self.stop_bond_inference(wait_ms=0)
        self.trajectory_generation += 1
        if self.streamer is not None:
            self.streamer.stop()
            self.streamer = None
        if self.store is not None:
            if self.store is not deferred_store:
                self.store.close()
            self.store = None
        self.gl_view.set_cell(None)
        self.box_check.setEnabled(False)
        self.set_representation_controls_enabled(False)
        self.component_ids = np.empty((0,), dtype=np.int32)
        self.component_sizes = np.empty((0,), dtype=np.int32)
        self.filter_mode = "all"
        self.filter_values = {"chain": 1, "atom": 1}
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
        self.current_frame = 0
        self.displayed_frame = -1
        self.scrub_preview_timer.stop()
        self.visibility_filter_timer.stop()
        self.slider_scrub.release(0)
        self.frame_slider.setMaximum(0)
        self.frame_slider.setValue(0)

    def set_controls_enabled(self, enabled: bool) -> None:
        self.prev_button.setEnabled(enabled)
        self.play_button.setEnabled(enabled)
        self.next_button.setEnabled(enabled)
        self.playback_speed_label.setEnabled(enabled)
        self.playback_speed_slider.setEnabled(enabled)
        self.playback_speed_value_label.setEnabled(enabled)

    def on_render_tick(self) -> None:
        if self.store is None or self.streamer is None:
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
            if self.benchmark_diagnostics is not None:
                self.benchmark_diagnostics.record_duplicate_frame()
            self.schedule_next_render_tick()
            return
        frame = self.streamer.get_frame(self.current_frame)
        if frame is None:
            if self.benchmark_diagnostics is not None:
                self.benchmark_diagnostics.record_no_frame()
            self.schedule_next_render_tick()
            return

        cell = self.streamer.get_cell(self.current_frame)
        self.gl_view.set_frame(frame, reset_view=self.reset_view_on_next_frame, cell=cell)
        if self.benchmark_diagnostics is not None:
            self.benchmark_diagnostics.record_upload(timestamp_s=time.perf_counter())
        self.reset_view_on_next_frame = False
        self.displayed_frame = self.current_frame
        self.update_frame_label()
        self.schedule_next_render_tick()

    def on_stream_frame_ready(self, frame_index: int) -> None:
        if (
            self.store is None
            or self.streamer is None
            or int(frame_index) != self.current_frame
            or self.displayed_frame == self.current_frame
        ):
            return
        self.on_render_tick()

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
        thread = self.open_thread
        if (
            self.store is not None
            and not self.store.is_frame_available(target)
            and thread is not None
            and thread.isRunning()
        ):
            thread.request_frames(
                self.streamer.target_indices(
                    target,
                    direction=direction,
                    interactive=interactive,
                )
            )

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
        self.gl_view.set_immediate_paint(True)
        self.schedule_next_render_tick()

    def stop_playback(self) -> None:
        if self.playback is not None:
            self.playback.stop()
        self.playback = None
        self.set_play_button_state(False)
        if self.benchmark_output is None:
            self.gl_view.set_immediate_paint(False)
            self.render_timer.setInterval(self.IDLE_RENDER_TIMER_MS)

    def schedule_next_render_tick(self) -> None:
        if self.playback is None or not self.playback.running:
            self.render_timer.setInterval(self.IDLE_RENDER_TIMER_MS)
            return
        if self.displayed_frame != self.current_frame:
            self.render_timer.setInterval(self.IDLE_RENDER_TIMER_MS)
            return
        delay_s = self.playback.next_frame_delay_s(time.perf_counter())
        if delay_s is None:
            self.render_timer.setInterval(self.IDLE_RENDER_TIMER_MS)
            return
        self.render_timer.setInterval(max(1, int(math.floor(delay_s * 1000.0))))

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
        if playing:
            self.play_button.setIcon(self._pause_icon)
            self.play_button.setToolTip("Pause (Space)")
            self.play_button.setAccessibleName("Pause trajectory")
        else:
            self.play_button.setIcon(self._play_icon)
            self.play_button.setToolTip("Play (Space)")
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
        self.frame_label.setText(f"Frame {current} / {total}")

    def show_error(self, message: str) -> None:
        self.stop_playback()
        QMessageBox.critical(self, "TrajPlayer", message)
        self.status_bar.showMessage("Error")

    def closeEvent(self, event) -> None:  # type: ignore[override]
        self._pending_open_source = None
        self.close_current_trajectory()
        retired = tuple(self._retired_open_stores)
        for thread in retired:
            thread.cancel()
        for thread in retired:
            thread.wait()
        for store in self._retired_open_stores.values():
            if store is not None:
                store.close()
        self._retired_open_stores.clear()
        self.stop_bond_inference(wait_ms=0)
        for thread in tuple(self._retired_bond_threads):
            thread.requestInterruption()
        for thread in tuple(self._retired_bond_threads):
            thread.wait()
        self._retired_bond_threads.clear()
        self.gl_view.cleanup()
        super().closeEvent(event)


def main() -> None:
    cli_args = parse_cli_args(sys.argv[1:])
    if cli_args.startup_smoke:
        return
    benchmark_store: BinaryTrajectoryStore | None = None
    benchmark_label: Path | None = None
    benchmark_metrics: dict[str, object] = {}
    if cli_args.benchmark_output is not None:
        benchmark_store, benchmark_label, benchmark_metrics = prepare_benchmark_store(cli_args)

    QSurfaceFormat.setDefaultFormat(default_surface_format())
    app = QApplication(sys.argv)
    window = TrajPlayerWindow()
    window.show()
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
