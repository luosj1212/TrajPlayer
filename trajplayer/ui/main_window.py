from __future__ import annotations

from PySide6.QtCore import QLocale, QSettings, QSize, Qt
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
    QScrollArea,
    QSizePolicy,
    QSlider,
    QSplitter,
    QStatusBar,
    QStyle,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from .theme import APP_STYLESHEET


UI_TEXT = {
    "window_title": {"en": "TrajPlayer GPU", "zh": "TrajPlayer GPU"},
    "open": {"en": "Open", "zh": "打开"},
    "open_tooltip": {"en": "Open trajectory (Ctrl+O)", "zh": "打开轨迹 (Ctrl+O)"},
    "previous": {"en": "Previous frame", "zh": "上一帧"},
    "previous_tooltip": {"en": "Previous frame (Left)", "zh": "上一帧 (左方向键)"},
    "next": {"en": "Next frame", "zh": "下一帧"},
    "next_tooltip": {"en": "Next frame (Right)", "zh": "下一帧 (右方向键)"},
    "play": {"en": "Play", "zh": "播放"},
    "play_tooltip": {"en": "Play (Space)", "zh": "播放 (空格)"},
    "play_accessible": {"en": "Play trajectory", "zh": "播放轨迹"},
    "pause": {"en": "Pause", "zh": "暂停"},
    "pause_tooltip": {"en": "Pause (Space)", "zh": "暂停 (空格)"},
    "pause_accessible": {"en": "Pause trajectory", "zh": "暂停轨迹"},
    "reset": {"en": "Reset view", "zh": "重置视角"},
    "reset_tooltip": {"en": "Reset view (R)", "zh": "重置视角 (R)"},
    "loop": {"en": "Loop", "zh": "循环"},
    "box": {"en": "Periodic box", "zh": "周期盒"},
    "infer_bonds": {"en": "Infer bonds", "zh": "推断化学键"},
    "infer_bonds_tooltip": {
        "en": "Infer a static bond topology from frame 1 when the file has no topology",
        "zh": "文件不含拓扑时，根据第 1 帧推断静态化学键",
    },
    "display": {"en": "Display", "zh": "显示"},
    "selection": {"en": "Selection", "zh": "选择"},
    "playback": {"en": "Playback", "zh": "播放"},
    "advanced": {"en": "Advanced", "zh": "高级"},
    "interface": {"en": "Interface", "zh": "界面"},
    "language": {"en": "Language", "zh": "语言"},
    "inspector": {"en": "Show controls", "zh": "显示控制面板"},
    "representation": {"en": "Representation", "zh": "显示模式"},
    "ball_stick": {"en": "Ball-stick", "zh": "球棍"},
    "ball": {"en": "Ball", "zh": "原子球"},
    "bond": {"en": "Bond", "zh": "化学键"},
    "atom_size": {"en": "Atom size", "zh": "原子大小"},
    "bond_size": {"en": "Bond size", "zh": "化学键大小"},
    "atom_size_tooltip": {
        "en": "100% uses Chimera-like ball scale in Ball-stick and physical VDW radius in Ball",
        "zh": "100% 时，球棍模式使用接近 Chimera 的比例，原子球模式使用范德华半径",
    },
    "bond_size_tooltip": {
        "en": "100% uses a 0.20 angstrom bond radius",
        "zh": "100% 对应 0.20 埃的化学键半径",
    },
    "speed": {"en": "Speed", "zh": "播放速度"},
    "speed_tooltip": {"en": "Playback speed", "zh": "调节播放速度"},
    "all": {"en": "All", "zh": "全部"},
    "chain": {"en": "Chain", "zh": "链"},
    "atom": {"en": "Atom", "zh": "原子"},
    "show_all": {"en": "Show all atoms", "zh": "显示全部原子"},
    "show_chain": {"en": "Show selected chains", "zh": "显示指定链"},
    "show_atom": {"en": "Show one atom", "zh": "显示单个原子"},
    "all_atoms": {"en": "All atoms", "zh": "全部原子"},
    "all_atoms_tooltip": {"en": "All atoms are visible", "zh": "当前显示全部原子"},
    "chain_tooltip": {
        "en": "Enter chain numbers separated by commas; use a dash for ranges",
        "zh": "输入链编号，逗号分隔，连续范围可使用短横线",
    },
    "empty_file": {
        "en": "Drop a trajectory here or click Open",
        "zh": "将轨迹拖到这里，或点击“打开”",
    },
    "idle": {"en": "GPU instancing idle", "zh": "GPU 实例化渲染待机"},
    "frame": {"en": "Frame {current} / {total}{suffix}", "zh": "帧 {current} / {total}{suffix}"},
    "ready": {"en": "Ready", "zh": "就绪"},
    "first_frame": {"en": "First frame", "zh": "第一帧"},
    "last_frame": {"en": "Last frame", "zh": "最后一帧"},
    "open_dialog": {
        "en": "Open trajectory (select GRO + XTC/TRR together)",
        "zh": "打开轨迹（可同时选择 GRO 与 XTC/TRR）",
    },
    "trajectory_files": {"en": "Trajectory files", "zh": "轨迹文件"},
    "gromacs_files": {"en": "Gromacs files", "zh": "Gromacs 文件"},
    "all_files": {"en": "All files", "zh": "所有文件"},
    "waiting_previous": {
        "en": "Waiting for the previous cache operation to stop",
        "zh": "正在等待上一次缓存任务停止",
    },
    "queued_open": {
        "en": "Trajectory queued; background I/O is stopping",
        "zh": "轨迹已加入队列，正在停止后台 I/O",
    },
    "opening": {"en": "Opening {name}", "zh": "正在打开 {name}"},
    "opening_metadata": {
        "en": "Opening trajectory metadata and first frame",
        "zh": "正在读取轨迹元数据和第一帧",
    },
    "opening_ui": {
        "en": "Opening trajectory without blocking the UI",
        "zh": "正在后台打开轨迹，界面可继续操作",
    },
    "converting": {
        "en": "Converting to contiguous float32 cache: {done}/{total} frames",
        "zh": "正在转换为连续 float32 缓存：{done}/{total} 帧",
    },
    "indexed": {
        "en": "Indexed {count} frames; direct reading is ready",
        "zh": "已索引 {count} 帧，可直接读取轨迹",
    },
    "indexing": {
        "en": "First frame ready; background index found {count} frames",
        "zh": "第一帧已就绪；后台索引已发现 {count} 帧",
    },
    "nearby_ready": {
        "en": "First frame ready; nearby frames will be decoded on demand",
        "zh": "第一帧已就绪；附近帧将按需解码",
    },
    "loaded": {"en": "Loaded {name} from {source}", "zh": "已从{source}加载 {name}"},
    "direct_reader": {"en": "direct reader", "zh": "直接读取器"},
    "cache": {"en": "cache", "zh": "缓存"},
    "new_cache": {"en": "new binary cache", "zh": "新建二进制缓存"},
    "bond_disabled": {"en": "Bond inference disabled", "zh": "已关闭化学键推断"},
    "representation_status": {"en": "Representation: {mode}", "zh": "显示模式：{mode}"},
    "chain_preparing": {
        "en": "Chain groups are still being prepared",
        "zh": "链分组仍在后台准备中",
    },
    "atom_value": {"en": "Atom {value}", "zh": "原子 {value}"},
    "atom_of": {"en": "Atom {value} of {maximum}", "zh": "第 {value} / {maximum} 个原子"},
    "invalid_chain": {"en": "Invalid chain selection: {error}", "zh": "链选择无效：{error}"},
    "showing_all": {"en": "Showing all {count} atoms", "zh": "正在显示全部 {count} 个原子"},
    "showing_chain": {
        "en": "Showing chain {selection}: {count} atoms",
        "zh": "正在显示链 {selection}：{count} 个原子",
    },
    "showing_chains": {
        "en": "Showing chains {selection}: {count} atoms",
        "zh": "正在显示链 {selection}：{count} 个原子",
    },
    "showing_atom": {"en": "Showing atom {value}", "zh": "正在显示原子 {value}"},
    "synthetic_ready": {
        "en": "Synthetic benchmark loaded without bond inference",
        "zh": "合成基准已加载，未执行化学键推断",
    },
    "inferring_bonds": {
        "en": "Inferring bonds in the background",
        "zh": "正在后台推断化学键",
    },
    "bonds_ready": {
        "en": "Bonds ready: {bonds} from frame 1, {components} components in {elapsed:.0f} ms",
        "zh": "化学键已就绪：第 1 帧推断出 {bonds} 条键，{components} 个组分，用时 {elapsed:.0f} ms",
    },
    "error": {"en": "Error", "zh": "错误"},
    "direct_source": {"en": "direct source", "zh": "直接读取源文件"},
    "direct_indexing": {
        "en": "direct source, indexing ({count}+ frames found)",
        "zh": "直接读取源文件，索引中（已发现 {count}+ 帧）",
    },
    "frames_cached": {
        "en": "{available}/{total} frames cached on demand",
        "zh": "已按需缓存 {available}/{total} 帧",
    },
    "frames_on_disk": {
        "en": "{available}/{total} frames on disk",
        "zh": "磁盘中有 {available}/{total} 帧",
    },
    "trajectory_info": {
        "en": "{frames} frames, {atoms} atoms, {bonds} bonds, {cache}, {disk}, {mode} GPU instancing",
        "zh": "{frames} 帧，{atoms} 个原子，{bonds} 条键，{cache}，{disk}，{mode} GPU 实例化",
    },
    "prefetch_cache": {
        "en": "directional prefetch, {capacity}-frame/{mib:.0f} MiB {mode} cache",
        "zh": "方向感知预取，{capacity} 帧/{mib:.0f} MiB {mode}缓存",
    },
    "prefetch_pending": {"en": "directional prefetch", "zh": "方向感知预取"},
    "auto": {"en": "auto", "zh": "自动"},
    "fixed": {"en": "fixed", "zh": "固定"},
}


class MainWindowView(QMainWindow):
    """Qt-only view construction; behavior comes from the controller subclass."""

    RESPONSIVE_INSPECTOR_WIDTH = 920

    def setup_ui(self, viewport: QWidget) -> None:
        self._settings = QSettings("TrajPlayer", "TrajPlayer")
        stored_language = str(self._settings.value("ui/language", "") or "")
        system_is_chinese = QLocale.system().language() == QLocale.Language.Chinese
        self.ui_language = (
            stored_language if stored_language in {"en", "zh"} else "zh" if system_is_chinese else "en"
        )
        self._inspector_preferred_visible = True
        self._responsive_inspector_hidden = False

        self.setWindowTitle(self._t("window_title"))
        self.resize(1280, 820)
        self.setMinimumSize(720, 520)
        self.setAcceptDrops(True)
        self.setStyleSheet(APP_STYLESHEET)
        self.gl_view = viewport

        self.open_button = QPushButton()
        self.open_button.setObjectName("openButton")
        self.open_button.setIcon(
            self.style().standardIcon(QStyle.StandardPixmap.SP_DialogOpenButton)
        )
        self.open_button.setIconSize(QSize(17, 17))

        self.inspector_toggle_button = QToolButton()
        self.inspector_toggle_button.setObjectName("inspectorToggleButton")
        self.inspector_toggle_button.setCheckable(True)
        self.inspector_toggle_button.setChecked(True)
        self.inspector_toggle_button.setIcon(
            self.style().standardIcon(QStyle.StandardPixmap.SP_FileDialogDetailedView)
        )
        self.inspector_toggle_button.setIconSize(QSize(18, 18))
        self.inspector_toggle_button.toggled.connect(self.on_inspector_toggled)

        self.prev_button = self._transport_button(
            QStyle.StandardPixmap.SP_MediaSkipBackward,
            18,
        )
        self._play_icon = self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPlay)
        self._pause_icon = self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPause)
        self.play_button = self._transport_button(
            QStyle.StandardPixmap.SP_MediaPlay,
            20,
        )
        self.play_button.setObjectName("playButton")
        self.next_button = self._transport_button(
            QStyle.StandardPixmap.SP_MediaSkipForward,
            18,
        )
        self.reset_view_button = self._transport_button(
            QStyle.StandardPixmap.SP_BrowserReload,
            17,
        )
        for button in (
            self.prev_button,
            self.play_button,
            self.next_button,
            self.reset_view_button,
        ):
            button.setEnabled(False)

        self.loop_check = QCheckBox()
        self.loop_check.setChecked(True)

        self.box_check = QCheckBox()
        self.box_check.setChecked(True)
        self.box_check.setEnabled(False)
        self.box_check.toggled.connect(self.on_box_toggled)

        self.infer_bonds_check = QCheckBox()
        self.infer_bonds_check.setChecked(True)
        self.infer_bonds_check.setEnabled(False)
        self.infer_bonds_check.toggled.connect(self.on_infer_bonds_toggled)

        self.render_mode_label = self._control_label()
        self.render_mode_combo = QComboBox()
        self.render_mode_combo.addItem("", "ball_stick")
        self.render_mode_combo.addItem("", "ball")
        self.render_mode_combo.addItem("", "bond")
        self.render_mode_combo.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Fixed,
        )
        self.render_mode_combo.setEnabled(False)
        self.render_mode_combo.currentIndexChanged.connect(self.on_render_mode_changed)

        self.atom_size_label = self._control_label()
        self.atom_size_slider = self._size_slider(10, 250, 100, 5, 25)
        self.atom_size_slider.valueChanged.connect(self.on_atom_size_changed)
        self.atom_size_value_label = self._value_label("100%")
        self.atom_size_label.setEnabled(False)
        self.atom_size_value_label.setEnabled(False)

        self.bond_size_label = self._control_label()
        self.bond_size_slider = self._size_slider(10, 300, 100, 5, 25)
        self.bond_size_slider.valueChanged.connect(self.on_bond_size_changed)
        self.bond_size_value_label = self._value_label("100%")
        self.bond_size_label.setEnabled(False)
        self.bond_size_value_label.setEnabled(False)

        self.playback_speed_label = self._control_label()
        self.playback_speed_slider = self._size_slider(
            1,
            int(self.TARGET_FPS),
            int(self.TARGET_FPS),
            1,
            5,
        )
        self.playback_speed_slider.valueChanged.connect(self.on_playback_speed_changed)
        self.playback_speed_value_label = self._value_label(
            f"{int(self.TARGET_FPS)} FPS"
        )
        self.transport_speed_label = self._value_label(
            f"{int(self.TARGET_FPS)} FPS"
        )
        self.playback_speed_label.setEnabled(False)
        self.playback_speed_value_label.setEnabled(False)
        self.transport_speed_label.setEnabled(False)

        self.filter_mode_segment = QFrame()
        self.filter_mode_segment.setObjectName("filterModeSegment")
        filter_mode_layout = QHBoxLayout(self.filter_mode_segment)
        filter_mode_layout.setContentsMargins(1, 1, 1, 1)
        filter_mode_layout.setSpacing(0)
        self.filter_mode_group = QButtonGroup(self)
        self.filter_mode_group.setExclusive(True)
        self.filter_mode_buttons: dict[str, QToolButton] = {}
        for mode in ("all", "chain", "atom"):
            button = QToolButton()
            button.setObjectName("filterModeButton")
            button.setCheckable(True)
            button.setEnabled(False)
            button.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            button.clicked.connect(
                lambda _checked=False, selected_mode=mode: self.on_filter_mode_changed(
                    selected_mode
                )
            )
            self.filter_mode_group.addButton(button)
            self.filter_mode_buttons[mode] = button
            filter_mode_layout.addWidget(button, stretch=1)
        self.filter_mode_buttons["all"].setChecked(True)

        self.filter_value_slider = QSlider(Qt.Orientation.Horizontal)
        self.filter_value_slider.setObjectName("filterValueSlider")
        self.filter_value_slider.setRange(1, 1)
        self.filter_value_slider.setSingleStep(1)
        self.filter_value_slider.setPageStep(1)
        self.filter_value_slider.setValue(1)
        self.filter_value_slider.setEnabled(False)
        self.filter_value_slider.valueChanged.connect(self.on_filter_value_changed)
        self.filter_value_label = self._value_label("")
        self.filter_value_label.setObjectName("filterValueLabel")

        self.chain_selection_edit = QLineEdit("1")
        self.chain_selection_edit.setObjectName("chainSelectionEdit")
        self.chain_selection_edit.setPlaceholderText("1,3-5")
        self.chain_selection_edit.setMaxLength(512)
        self.chain_selection_edit.setProperty("invalid", False)
        self.chain_selection_edit.setEnabled(False)
        self.chain_selection_edit.hide()
        self.chain_selection_edit.textChanged.connect(self.on_chain_selection_changed)
        self.chain_selection_edit.editingFinished.connect(self.normalize_chain_selection)

        self.frame_label = self._value_label("")
        self.frame_label.setObjectName("frameLabel")
        self.frame_label.setMinimumWidth(145)
        self.file_label = QLabel()
        self.file_label.setObjectName("fileLabel")
        self.file_label.setMinimumWidth(0)
        self.file_label.setSizePolicy(
            QSizePolicy.Policy.Ignored,
            QSizePolicy.Policy.Preferred,
        )
        self.info_label = QLabel()
        self.info_label.setObjectName("infoLabel")
        self.info_label.setMinimumWidth(0)
        self.info_label.setSizePolicy(
            QSizePolicy.Policy.Ignored,
            QSizePolicy.Policy.Preferred,
        )

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 1)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(False)
        self.progress_bar.hide()

        self.frame_slider = QSlider(Qt.Orientation.Horizontal)
        self.frame_slider.setRange(0, 0)
        self.frame_slider.setEnabled(False)
        self.frame_slider.setTracking(False)
        self.frame_slider.sliderPressed.connect(self.on_frame_slider_pressed)
        self.frame_slider.sliderMoved.connect(self.on_frame_slider_moved)
        self.frame_slider.sliderReleased.connect(self.on_frame_slider_released)
        self.frame_slider.valueChanged.connect(self.on_frame_slider_changed)

        self.language_label = self._control_label()
        self.language_combo = QComboBox()
        self.language_combo.addItem("中文", "zh")
        self.language_combo.addItem("English", "en")
        self.language_combo.setCurrentIndex(
            max(0, self.language_combo.findData(self.ui_language))
        )
        self.language_combo.currentIndexChanged.connect(self.on_language_changed)

        self.advanced_toggle = QToolButton()
        self.advanced_toggle.setObjectName("advancedToggle")
        self.advanced_toggle.setCheckable(True)
        self.advanced_toggle.setChecked(False)
        self.advanced_toggle.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.advanced_toggle.setArrowType(Qt.ArrowType.RightArrow)
        self.advanced_toggle.toggled.connect(self.on_advanced_toggled)
        self.advanced_content = QFrame()
        self.advanced_content.setObjectName("advancedContent")
        advanced_layout = QVBoxLayout(self.advanced_content)
        advanced_layout.setContentsMargins(0, 2, 0, 0)
        advanced_layout.setSpacing(8)
        advanced_layout.addWidget(self.infer_bonds_check)
        self.advanced_content.hide()

        top_bar = self._build_top_bar()
        self.inspector_panel = self._build_inspector()
        self.inspector_scroll = QScrollArea()
        self.inspector_scroll.setObjectName("inspectorScroll")
        self.inspector_scroll.setWidgetResizable(True)
        self.inspector_scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.inspector_scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self.inspector_scroll.setWidget(self.inspector_panel)
        self.inspector_scroll.setMinimumWidth(244)
        self.inspector_scroll.setMaximumWidth(320)

        self.content_splitter = QSplitter(Qt.Orientation.Horizontal)
        self.content_splitter.setObjectName("contentSplitter")
        self.content_splitter.setChildrenCollapsible(False)
        self.content_splitter.addWidget(self.gl_view)
        self.content_splitter.addWidget(self.inspector_scroll)
        self.content_splitter.setStretchFactor(0, 1)
        self.content_splitter.setStretchFactor(1, 0)
        self.content_splitter.setSizes([1000, 280])

        transport_bar = self._build_transport_bar()
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(top_bar)
        layout.addWidget(self.progress_bar)
        layout.addWidget(self.content_splitter, stretch=1)
        layout.addWidget(transport_bar)

        central = QWidget()
        central.setObjectName("centralWidget")
        central.setLayout(layout)
        self.setCentralWidget(central)

        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.retranslate_ui()

    def _build_top_bar(self) -> QFrame:
        top_bar = QFrame()
        top_bar.setObjectName("topBar")
        top_layout = QHBoxLayout(top_bar)
        top_layout.setContentsMargins(12, 8, 10, 8)
        top_layout.setSpacing(12)
        top_layout.addWidget(self.open_button)
        file_layout = QVBoxLayout()
        file_layout.setContentsMargins(0, 0, 0, 0)
        file_layout.setSpacing(1)
        file_layout.addWidget(self.file_label)
        file_layout.addWidget(self.info_label)
        top_layout.addLayout(file_layout, stretch=1)
        top_layout.addWidget(self.inspector_toggle_button)
        return top_bar

    def _build_inspector(self) -> QFrame:
        panel = QFrame()
        panel.setObjectName("inspectorPanel")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(16, 14, 16, 16)
        layout.setSpacing(8)

        self.display_section_label = self._section_label()
        layout.addWidget(self.display_section_label)
        layout.addWidget(self.render_mode_label)
        layout.addWidget(self.render_mode_combo)
        layout.addLayout(
            self._slider_row(
                self.atom_size_label,
                self.atom_size_slider,
                self.atom_size_value_label,
            )
        )
        layout.addLayout(
            self._slider_row(
                self.bond_size_label,
                self.bond_size_slider,
                self.bond_size_value_label,
            )
        )
        layout.addWidget(self.box_check)
        layout.addSpacing(10)

        self.selection_section_label = self._section_label()
        layout.addWidget(self.selection_section_label)
        layout.addWidget(self.filter_mode_segment)
        filter_value_layout = QHBoxLayout()
        filter_value_layout.setContentsMargins(0, 0, 0, 0)
        filter_value_layout.setSpacing(8)
        filter_value_layout.addWidget(self.filter_value_slider, stretch=1)
        filter_value_layout.addWidget(self.filter_value_label)
        filter_value_layout.addWidget(self.chain_selection_edit, stretch=1)
        layout.addLayout(filter_value_layout)
        layout.addSpacing(10)

        self.playback_section_label = self._section_label()
        layout.addWidget(self.playback_section_label)
        layout.addLayout(
            self._slider_row(
                self.playback_speed_label,
                self.playback_speed_slider,
                self.playback_speed_value_label,
            )
        )
        layout.addSpacing(8)
        layout.addWidget(self.advanced_toggle)
        layout.addWidget(self.advanced_content)
        layout.addSpacing(10)

        self.interface_section_label = self._section_label()
        layout.addWidget(self.interface_section_label)
        language_layout = QHBoxLayout()
        language_layout.setContentsMargins(0, 0, 0, 0)
        language_layout.setSpacing(8)
        language_layout.addWidget(self.language_label)
        language_layout.addWidget(self.language_combo, stretch=1)
        layout.addLayout(language_layout)
        layout.addStretch(1)
        return panel

    def _build_transport_bar(self) -> QFrame:
        transport_bar = QFrame()
        transport_bar.setObjectName("transportBar")
        layout = QVBoxLayout(transport_bar)
        layout.setContentsMargins(12, 8, 12, 9)
        layout.setSpacing(7)
        layout.addWidget(self.frame_slider)

        controls = QHBoxLayout()
        controls.setContentsMargins(0, 0, 0, 0)
        controls.setSpacing(6)
        controls.addWidget(self.reset_view_button)
        controls.addWidget(self.loop_check)
        controls.addStretch(1)
        controls.addWidget(self.prev_button)
        controls.addWidget(self.play_button)
        controls.addWidget(self.next_button)
        controls.addStretch(1)
        controls.addWidget(self.transport_speed_label)
        controls.addSpacing(8)
        controls.addWidget(self.frame_label)
        layout.addLayout(controls)
        return transport_bar

    def retranslate_ui(self) -> None:
        self.setWindowTitle(self._t("window_title"))
        self.open_button.setText(self._t("open"))
        self.open_button.setToolTip(self._t("open_tooltip"))
        self.inspector_toggle_button.setToolTip(self._t("inspector"))
        self.prev_button.setToolTip(self._t("previous_tooltip"))
        self.prev_button.setAccessibleName(self._t("previous"))
        self.next_button.setToolTip(self._t("next_tooltip"))
        self.next_button.setAccessibleName(self._t("next"))
        self.reset_view_button.setToolTip(self._t("reset_tooltip"))
        self.reset_view_button.setAccessibleName(self._t("reset"))
        self.loop_check.setText(self._t("loop"))
        self.box_check.setText(self._t("box"))
        self.infer_bonds_check.setText(self._t("infer_bonds"))
        self.infer_bonds_check.setToolTip(self._t("infer_bonds_tooltip"))
        self.infer_bonds_check.setAccessibleName(self._t("infer_bonds"))

        self.display_section_label.setText(self._t("display"))
        self.selection_section_label.setText(self._t("selection"))
        self.playback_section_label.setText(self._t("playback"))
        self.interface_section_label.setText(self._t("interface"))
        self.advanced_toggle.setText(self._t("advanced"))
        self.render_mode_label.setText(self._t("representation"))
        for key, data in (("ball_stick", "ball_stick"), ("ball", "ball"), ("bond", "bond")):
            index = self.render_mode_combo.findData(data)
            if index >= 0:
                self.render_mode_combo.setItemText(index, self._t(key))
        self.render_mode_combo.setToolTip(self._t("representation"))
        self.atom_size_label.setText(self._t("atom_size"))
        self.atom_size_slider.setToolTip(self._t("atom_size_tooltip"))
        self.bond_size_label.setText(self._t("bond_size"))
        self.bond_size_slider.setToolTip(self._t("bond_size_tooltip"))
        self.playback_speed_label.setText(self._t("speed"))
        self.playback_speed_slider.setToolTip(self._t("speed_tooltip"))
        self.language_label.setText(self._t("language"))

        for mode in ("all", "chain", "atom"):
            button = self.filter_mode_buttons[mode]
            button.setText(self._t(mode))
            button.setToolTip(self._t(f"show_{mode}"))
            button.setAccessibleName(self._t(f"show_{mode}"))
        self.chain_selection_edit.setToolTip(self._t("chain_tooltip"))
        self.chain_selection_edit.setAccessibleName(self._t("show_chain"))

        commands = getattr(self, "commands", None)
        if commands is not None:
            commands.retranslate(self._t)
        else:
            self.play_button.setAccessibleName(self._t("play_accessible"))

        if (
            getattr(self, "store", None) is None
            and getattr(self, "open_thread", None) is None
        ):
            self.file_label.setText(self._t("empty_file"))
            self.info_label.setText(self._t("idle"))
            self.status_bar.showMessage(self._t("ready"))
        self.update_frame_label() if hasattr(self, "update_frame_label") else None
        if hasattr(self, "update_filter_value_label"):
            self.update_filter_value_label()

    def on_language_changed(self) -> None:
        selected = str(self.language_combo.currentData())
        if selected not in {"en", "zh"} or selected == self.ui_language:
            return
        self.ui_language = selected
        self._settings.setValue("ui/language", selected)
        self.retranslate_ui()
        if getattr(self, "store", None) is not None and hasattr(self, "update_trajectory_info"):
            self.update_trajectory_info()

    def on_advanced_toggled(self, expanded: bool) -> None:
        self.advanced_content.setVisible(bool(expanded))
        self.advanced_toggle.setArrowType(
            Qt.ArrowType.DownArrow if expanded else Qt.ArrowType.RightArrow
        )

    def on_inspector_toggled(self, visible: bool) -> None:
        self._inspector_preferred_visible = bool(visible)
        self.inspector_scroll.setVisible(bool(visible))

    def resizeEvent(self, event) -> None:  # type: ignore[override]
        super().resizeEvent(event)
        if not hasattr(self, "inspector_scroll"):
            return
        compact = self.width() < self.RESPONSIVE_INSPECTOR_WIDTH
        if compact == self._responsive_inspector_hidden:
            return
        self._responsive_inspector_hidden = compact
        visible = self._inspector_preferred_visible and not compact
        self.inspector_toggle_button.blockSignals(True)
        self.inspector_toggle_button.setChecked(visible)
        self.inspector_toggle_button.blockSignals(False)
        self.inspector_scroll.setVisible(visible)

    def _t(self, key: str, **values) -> str:
        translations = UI_TEXT.get(key)
        if translations is None:
            return key.format(**values)
        template = translations.get(self.ui_language, translations["en"])
        return template.format(**values)

    def _transport_button(self, pixmap: QStyle.StandardPixmap, icon_size: int) -> QToolButton:
        button = QToolButton()
        button.setIcon(self.style().standardIcon(pixmap))
        button.setIconSize(QSize(icon_size, icon_size))
        return button

    @staticmethod
    def _control_label() -> QLabel:
        label = QLabel()
        label.setObjectName("controlLabel")
        return label

    @staticmethod
    def _section_label() -> QLabel:
        label = QLabel()
        label.setObjectName("sectionLabel")
        return label

    @staticmethod
    def _value_label(text: str) -> QLabel:
        label = QLabel(text)
        label.setObjectName("sizeValueLabel")
        label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        label.setMinimumWidth(48)
        return label

    @staticmethod
    def _size_slider(
        minimum: int,
        maximum: int,
        value: int,
        step: int,
        page_step: int,
    ) -> QSlider:
        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setObjectName("sizeSlider")
        slider.setRange(minimum, maximum)
        slider.setSingleStep(step)
        slider.setPageStep(page_step)
        slider.setValue(value)
        slider.setEnabled(False)
        slider.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        return slider

    @staticmethod
    def _slider_row(label: QLabel, slider: QSlider, value: QLabel) -> QHBoxLayout:
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(8)
        label.setMinimumWidth(70)
        row.addWidget(label)
        row.addWidget(slider, stretch=1)
        row.addWidget(value)
        return row
