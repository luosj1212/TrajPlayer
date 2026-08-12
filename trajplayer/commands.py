from __future__ import annotations

from collections.abc import Callable

from PySide6.QtCore import Qt
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import QMainWindow, QStyle


class WindowCommands:
    """Single QAction registry shared by buttons and keyboard shortcuts."""

    def __init__(
        self,
        window: QMainWindow,
        *,
        open_file: Callable[[], None],
        toggle_playback: Callable[[], None],
        step_previous: Callable[[], None],
        step_next: Callable[[], None],
        step_back_ten: Callable[[], None],
        step_forward_ten: Callable[[], None],
        show_recent_files: Callable[[], None],
        jump_first: Callable[[], None],
        jump_last: Callable[[], None],
        reset_camera: Callable[[], None],
        clear_selection: Callable[[], None],
        focus_selection: Callable[[], None],
        create_measurement: Callable[[], None],
        add_marker: Callable[[], None],
        export_frame: Callable[[], None],
        export_analysis: Callable[[], None],
        toggle_analysis_panel: Callable[[], None],
        delete_context_item: Callable[[], None],
    ) -> None:
        style = window.style()
        self._translate = lambda key: key
        self._playing = False
        self.open_file = self._action(
            window,
            text="Open",
            tooltip="Open trajectory",
            icon=style.standardIcon(QStyle.StandardPixmap.SP_DialogOpenButton),
            shortcut=QKeySequence.StandardKey.Open,
            callback=open_file,
        )
        self.recent_files = self._action(
            window,
            text="Recent files",
            tooltip="Open recent files (Ctrl+Shift+O)",
            shortcut=QKeySequence("Ctrl+Shift+O"),
            callback=show_recent_files,
        )
        self.previous_frame = self._action(
            window,
            text="Previous frame",
            tooltip="Previous frame (Left)",
            icon=style.standardIcon(QStyle.StandardPixmap.SP_MediaSkipBackward),
            shortcut=QKeySequence(Qt.Key.Key_Left),
            callback=step_previous,
            enabled=False,
        )
        self.play = self._action(
            window,
            text="Play",
            tooltip="Play (Space)",
            icon=style.standardIcon(QStyle.StandardPixmap.SP_MediaPlay),
            shortcut=QKeySequence(Qt.Key.Key_Space),
            callback=toggle_playback,
            enabled=False,
        )
        self.next_frame = self._action(
            window,
            text="Next frame",
            tooltip="Next frame (Right)",
            icon=style.standardIcon(QStyle.StandardPixmap.SP_MediaSkipForward),
            shortcut=QKeySequence(Qt.Key.Key_Right),
            callback=step_next,
            enabled=False,
        )
        self.back_ten_frames = self._action(
            window,
            text="Back 10 frames",
            tooltip="Back 10 frames (Shift+Left)",
            shortcut=QKeySequence("Shift+Left"),
            callback=step_back_ten,
            enabled=False,
        )
        self.forward_ten_frames = self._action(
            window,
            text="Forward 10 frames",
            tooltip="Forward 10 frames (Shift+Right)",
            shortcut=QKeySequence("Shift+Right"),
            callback=step_forward_ten,
            enabled=False,
        )
        self.first_frame = self._action(
            window,
            text="First frame",
            tooltip="First frame (Home)",
            shortcut=QKeySequence(Qt.Key.Key_Home),
            callback=jump_first,
            enabled=False,
        )
        self.last_frame = self._action(
            window,
            text="Last frame",
            tooltip="Last frame (End)",
            shortcut=QKeySequence(Qt.Key.Key_End),
            callback=jump_last,
            enabled=False,
        )
        self.reset_camera = self._action(
            window,
            text="Reset view",
            tooltip="Reset view (R)",
            icon=style.standardIcon(QStyle.StandardPixmap.SP_BrowserReload),
            shortcut=QKeySequence("R"),
            callback=reset_camera,
            enabled=False,
        )
        selection_host = getattr(window, "gl_view", window)
        self.clear_selection = self._action(
            selection_host,
            text="Clear selection",
            tooltip="Clear selection (Esc)",
            shortcut=QKeySequence(Qt.Key.Key_Escape),
            callback=clear_selection,
            enabled=False,
            shortcut_context=Qt.ShortcutContext.WidgetWithChildrenShortcut,
        )
        self.focus_selection = self._action(
            selection_host,
            text="Focus selection",
            tooltip="Focus selection (F)",
            shortcut=QKeySequence("F"),
            callback=focus_selection,
            enabled=False,
            shortcut_context=Qt.ShortcutContext.WidgetWithChildrenShortcut,
        )
        self.create_measurement = self._action(
            selection_host,
            text="Pin measurement",
            tooltip="Pin measurement (M)",
            shortcut=QKeySequence("M"),
            callback=create_measurement,
            enabled=False,
            shortcut_context=Qt.ShortcutContext.WidgetWithChildrenShortcut,
        )
        self.add_marker = self._action(
            window,
            text="Add marker",
            tooltip="Add marker (Ctrl+M)",
            shortcut=QKeySequence("Ctrl+M"),
            callback=add_marker,
            enabled=False,
        )
        self.export_frame = self._action(
            window,
            text="Export frame",
            tooltip="Export frame (Ctrl+E)",
            shortcut=QKeySequence("Ctrl+E"),
            callback=export_frame,
            enabled=False,
        )
        self.export_analysis = self._action(
            window,
            text="Export analysis CSV",
            tooltip="Export analysis CSV (Ctrl+Shift+E)",
            shortcut=QKeySequence("Ctrl+Shift+E"),
            callback=export_analysis,
            enabled=False,
        )
        self.toggle_analysis_panel = self._action(
            window,
            text="Toggle analysis",
            tooltip="Show or hide analysis (Ctrl+L)",
            shortcut=QKeySequence("Ctrl+L"),
            callback=toggle_analysis_panel,
        )
        self.delete_context_item = self._action(
            selection_host,
            text="Delete current item",
            tooltip="Delete current measurement or marker (Delete)",
            shortcut=QKeySequence(Qt.Key.Key_Delete),
            callback=delete_context_item,
            shortcut_context=Qt.ShortcutContext.WidgetWithChildrenShortcut,
        )

    def bind_buttons(self, window) -> None:
        window.open_button.clicked.connect(self.open_file.trigger)
        window.prev_button.setDefaultAction(self.previous_frame)
        window.play_button.setDefaultAction(self.play)
        window.next_button.setDefaultAction(self.next_frame)
        window.reset_view_button.setDefaultAction(self.reset_camera)

    def set_transport_enabled(self, enabled: bool) -> None:
        for action in (
            self.previous_frame,
            self.play,
            self.next_frame,
            self.back_ten_frames,
            self.forward_ten_frames,
            self.first_frame,
            self.last_frame,
        ):
            action.setEnabled(bool(enabled))

    def set_reset_enabled(self, enabled: bool) -> None:
        self.reset_camera.setEnabled(bool(enabled))

    def set_selection_enabled(self, enabled: bool) -> None:
        self.clear_selection.setEnabled(bool(enabled))
        self.focus_selection.setEnabled(bool(enabled))

    def set_measurement_enabled(self, enabled: bool) -> None:
        self.create_measurement.setEnabled(bool(enabled))

    def set_timeline_enabled(self, enabled: bool) -> None:
        self.add_marker.setEnabled(bool(enabled))

    def set_export_enabled(self, *, frame: bool, analysis: bool) -> None:
        self.export_frame.setEnabled(bool(frame))
        self.export_analysis.setEnabled(bool(analysis))

    def set_playing(self, playing: bool) -> None:
        self._playing = bool(playing)
        if playing:
            self.play.setText(self._translate("pause"))
            self.play.setToolTip(self._translate("pause_tooltip"))
            self.play.setIcon(
                self.play.parent().style().standardIcon(
                    QStyle.StandardPixmap.SP_MediaPause
                )
            )
        else:
            self.play.setText(self._translate("play"))
            self.play.setToolTip(self._translate("play_tooltip"))
            self.play.setIcon(
                self.play.parent().style().standardIcon(
                    QStyle.StandardPixmap.SP_MediaPlay
                )
            )

    def retranslate(self, translate) -> None:
        self._translate = translate
        for action, text_key, tooltip_key in (
            (self.open_file, "open", "open_tooltip"),
            (self.recent_files, "recent", "recent_shortcut_tooltip"),
            (self.previous_frame, "previous", "previous_tooltip"),
            (self.next_frame, "next", "next_tooltip"),
            (self.first_frame, "first_frame", "first_frame"),
            (self.last_frame, "last_frame", "last_frame"),
            (self.reset_camera, "reset", "reset_tooltip"),
            (self.clear_selection, "clear_selection", "clear_selection_tooltip"),
            (self.focus_selection, "focus_selection", "focus_selection_tooltip"),
            (self.create_measurement, "pin_measurement", "pin_measurement_tooltip"),
            (self.add_marker, "add_marker", "add_marker_tooltip"),
            (self.export_frame, "export_frame", "export_frame_tooltip"),
            (self.export_analysis, "export_csv", "export_csv"),
            (self.back_ten_frames, "back_ten", "back_ten_tooltip"),
            (self.forward_ten_frames, "forward_ten", "forward_ten_tooltip"),
            (self.toggle_analysis_panel, "toggle_analysis", "toggle_analysis_tooltip"),
            (self.delete_context_item, "delete_current", "delete_current_tooltip"),
        ):
            action.setText(translate(text_key))
            action.setToolTip(translate(tooltip_key))
            action.setStatusTip(translate(tooltip_key))
        self.set_playing(self._playing)

    @staticmethod
    def _action(
        window: QMainWindow,
        *,
        text: str,
        tooltip: str,
        callback: Callable[[], None],
        icon=None,
        shortcut: QKeySequence | QKeySequence.StandardKey | None = None,
        enabled: bool = True,
        shortcut_context: Qt.ShortcutContext = Qt.ShortcutContext.WindowShortcut,
    ) -> QAction:
        action = QAction(icon, text, window) if icon is not None else QAction(text, window)
        action.setToolTip(tooltip)
        action.setStatusTip(tooltip)
        action.setShortcutContext(shortcut_context)
        if shortcut is not None:
            if isinstance(shortcut, QKeySequence.StandardKey):
                action.setShortcuts(shortcut)
            else:
                action.setShortcut(shortcut)
        action.setEnabled(enabled)
        action.triggered.connect(callback)
        window.addAction(action)
        return action
