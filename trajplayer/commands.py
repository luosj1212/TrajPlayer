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
        jump_first: Callable[[], None],
        jump_last: Callable[[], None],
        reset_camera: Callable[[], None],
    ) -> None:
        style = window.style()
        self.open_file = self._action(
            window,
            text="Open",
            tooltip="Open trajectory",
            icon=style.standardIcon(QStyle.StandardPixmap.SP_DialogOpenButton),
            shortcut=QKeySequence.StandardKey.Open,
            callback=open_file,
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
            self.first_frame,
            self.last_frame,
        ):
            action.setEnabled(bool(enabled))

    def set_reset_enabled(self, enabled: bool) -> None:
        self.reset_camera.setEnabled(bool(enabled))

    def set_playing(self, playing: bool) -> None:
        if playing:
            self.play.setText("Pause")
            self.play.setToolTip("Pause (Space)")
            self.play.setIcon(
                self.play.parent().style().standardIcon(
                    QStyle.StandardPixmap.SP_MediaPause
                )
            )
        else:
            self.play.setText("Play")
            self.play.setToolTip("Play (Space)")
            self.play.setIcon(
                self.play.parent().style().standardIcon(
                    QStyle.StandardPixmap.SP_MediaPlay
                )
            )

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
    ) -> QAction:
        action = QAction(icon, text, window) if icon is not None else QAction(text, window)
        action.setToolTip(tooltip)
        action.setStatusTip(tooltip)
        action.setShortcutContext(Qt.ShortcutContext.WindowShortcut)
        if shortcut is not None:
            if isinstance(shortcut, QKeySequence.StandardKey):
                action.setShortcuts(shortcut)
            else:
                action.setShortcut(shortcut)
        action.setEnabled(enabled)
        action.triggered.connect(callback)
        window.addAction(action)
        return action
