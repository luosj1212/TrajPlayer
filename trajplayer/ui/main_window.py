from __future__ import annotations

from PySide6.QtCore import QSize, Qt
from PySide6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
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

from .theme import APP_STYLESHEET


class MainWindowView(QMainWindow):
    """Qt-only view construction; behavior comes from the controller subclass."""

    def setup_ui(self, viewport: QWidget) -> None:
        self.setWindowTitle("TrajPlayer GPU")
        self.resize(1280, 820)
        self.setAcceptDrops(True)
        self.setStyleSheet(APP_STYLESHEET)
        self.gl_view = viewport

        self.open_button = QPushButton("Open")
        self.open_button.setObjectName("openButton")
        self.open_button.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_DialogOpenButton))
        self.open_button.setIconSize(QSize(17, 17))
        self.open_button.setToolTip("Open trajectory (Ctrl+O)")

        self.prev_button = QToolButton()
        self.prev_button.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaSkipBackward))
        self.prev_button.setIconSize(QSize(18, 18))
        self.prev_button.setToolTip("Previous frame (Left)")
        self.prev_button.setAccessibleName("Previous frame")
        self.prev_button.setEnabled(False)

        self._play_icon = self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPlay)
        self._pause_icon = self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPause)
        self.play_button = QToolButton()
        self.play_button.setObjectName("playButton")
        self.play_button.setIconSize(QSize(20, 20))
        self.play_button.setAccessibleName("Play trajectory")
        self.play_button.setEnabled(False)
        self.set_play_button_state(False)

        self.next_button = QToolButton()
        self.next_button.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaSkipForward))
        self.next_button.setIconSize(QSize(18, 18))
        self.next_button.setToolTip("Next frame (Right)")
        self.next_button.setAccessibleName("Next frame")
        self.next_button.setEnabled(False)

        self.reset_view_button = QToolButton()
        self.reset_view_button.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_BrowserReload))
        self.reset_view_button.setIconSize(QSize(17, 17))
        self.reset_view_button.setToolTip("Reset view (R)")
        self.reset_view_button.setAccessibleName("Reset view")
        self.reset_view_button.setEnabled(False)

        self.loop_check = QCheckBox("Loop")
        self.loop_check.setChecked(True)

        self.box_check = QCheckBox("Box")
        self.box_check.setChecked(True)
        self.box_check.setEnabled(False)
        self.box_check.toggled.connect(self.on_box_toggled)

        self.infer_bonds_check = QCheckBox("Infer bonds")
        self.infer_bonds_check.setChecked(True)
        self.infer_bonds_check.setEnabled(False)
        self.infer_bonds_check.setToolTip(
            "Infer a static bond topology from frame 1 when no file topology is available"
        )
        self.infer_bonds_check.setAccessibleName("Infer bonds from frame 1")
        self.infer_bonds_check.toggled.connect(self.on_infer_bonds_toggled)

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

        self.chain_selection_edit = QLineEdit("1")
        self.chain_selection_edit.setObjectName("chainSelectionEdit")
        self.chain_selection_edit.setPlaceholderText("1,3-5")
        self.chain_selection_edit.setFixedWidth(124)
        self.chain_selection_edit.setMaxLength(512)
        self.chain_selection_edit.setToolTip(
            "Enter chain numbers separated by commas; use a dash for ranges"
        )
        self.chain_selection_edit.setAccessibleName("Visible chain numbers")
        self.chain_selection_edit.setProperty("invalid", False)
        self.chain_selection_edit.setEnabled(False)
        self.chain_selection_edit.hide()
        self.chain_selection_edit.textChanged.connect(
            self.on_chain_selection_changed
        )
        self.chain_selection_edit.editingFinished.connect(
            self.normalize_chain_selection
        )

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
        controls.addWidget(self.infer_bonds_check)
        controls.addStretch(1)
        controls.addWidget(self.filter_mode_segment)
        controls.addSpacing(4)
        controls.addWidget(self.filter_value_slider)
        controls.addWidget(self.filter_value_label)
        controls.addWidget(self.chain_selection_edit)
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
