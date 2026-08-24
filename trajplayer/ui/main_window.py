from __future__ import annotations

from PySide6.QtCore import QCoreApplication, QEvent, QLocale, QSettings, QSize, Qt, QTranslator
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QButtonGroup,
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QLayout,
    QMainWindow,
    QMenu,
    QPushButton,
    QProgressBar,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QSplitter,
    QStackedLayout,
    QStatusBar,
    QStyle,
    QTabWidget,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from .theme import APP_STYLESHEET, build_stylesheet, resolve_theme
from .analysis_plot import AnalysisPlotWidget
from .viewport_overlay import ViewportOverlay
from trajplayer.timeline import TimelineWidget
from trajplayer.i18n import translation_file


UI_TEXT = {
    "window_title": {"en": "TrajPlayer GPU", "zh": "TrajPlayer GPU"},
    "open": {"en": "Open", "zh": "打开"},
    "open_tooltip": {"en": "Open trajectory (Ctrl+O)", "zh": "打开轨迹 (Ctrl+O)"},
    "recent": {"en": "Recent", "zh": "最近"},
    "recent_tooltip": {"en": "Open a recent trajectory", "zh": "打开最近使用的轨迹"},
    "recent_shortcut_tooltip": {"en": "Open recent files (Ctrl+Shift+O)", "zh": "打开最近文件 (Ctrl+Shift+O)"},
    "no_recent": {"en": "No recent trajectories", "zh": "暂无最近轨迹"},
    "drop_open": {"en": "Release to open {name}", "zh": "松开以打开 {name}"},
    "drop_need_gro": {"en": "This trajectory needs a GRO topology", "zh": "此轨迹需要 GRO 拓扑文件"},
    "drop_unsupported": {"en": "This file type is not supported", "zh": "不支持此文件类型"},
    "previous": {"en": "Previous frame", "zh": "上一帧"},
    "previous_tooltip": {"en": "Previous frame (Left)", "zh": "上一帧 (左方向键)"},
    "next": {"en": "Next frame", "zh": "下一帧"},
    "next_tooltip": {"en": "Next frame (Right)", "zh": "下一帧 (右方向键)"},
    "back_ten": {"en": "Back 10 frames", "zh": "后退 10 帧"},
    "back_ten_tooltip": {"en": "Back 10 frames (Shift+Left)", "zh": "后退 10 帧 (Shift+左方向键)"},
    "forward_ten": {"en": "Forward 10 frames", "zh": "前进 10 帧"},
    "forward_ten_tooltip": {"en": "Forward 10 frames (Shift+Right)", "zh": "前进 10 帧 (Shift+右方向键)"},
    "toggle_analysis": {"en": "Toggle analysis", "zh": "切换分析面板"},
    "toggle_analysis_tooltip": {"en": "Show or hide analysis (Ctrl+L)", "zh": "显示或隐藏分析面板 (Ctrl+L)"},
    "delete_current": {"en": "Delete current item", "zh": "删除当前项目"},
    "delete_current_tooltip": {"en": "Delete current measurement or marker (Delete)", "zh": "删除当前测量或标记 (Delete)"},
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
    "visibility": {"en": "Visibility", "zh": "可见范围"},
    "selection": {"en": "Selection", "zh": "选择"},
    "no_selection": {"en": "Click an atom to select it", "zh": "点击原子进行选择"},
    "selected_count": {"en": "{count} atoms selected", "zh": "已选择 {count} 个原子"},
    "selected_atoms": {"en": "Atoms {atoms}", "zh": "原子 {atoms}"},
    "clear_selection": {"en": "Clear", "zh": "清除"},
    "clear_selection_tooltip": {"en": "Clear selection (Esc)", "zh": "清除选择 (Esc)"},
    "focus_selection": {"en": "Focus", "zh": "聚焦"},
    "focus_selection_tooltip": {"en": "Focus selected atoms (F)", "zh": "聚焦已选原子 (F)"},
    "measurement": {"en": "Measurement", "zh": "测量"},
    "measurement_hint": {
        "en": "Select 2, 3, or 4 atoms in order",
        "zh": "依次选择 2、3 或 4 个原子",
    },
    "measurement_invalid": {
        "en": "Select exactly 2, 3, or 4 distinct atoms",
        "zh": "请选择恰好 2、3 或 4 个不同原子",
    },
    "measurement_pbc": {"en": "Periodic minimum image", "zh": "周期最小镜像"},
    "pin_measurement": {"en": "Pin", "zh": "固定"},
    "pin_measurement_tooltip": {
        "en": "Pin the current distance, angle, or dihedral (M)",
        "zh": "固定当前距离、角度或二面角 (M)",
    },
    "remove_measurement": {"en": "Remove", "zh": "删除"},
    "analyze_measurement": {"en": "Over time", "zh": "随时间分析"},
    "distance_name": {"en": "Distance", "zh": "距离"},
    "angle_name": {"en": "Angle", "zh": "角度"},
    "dihedral_name": {"en": "Dihedral", "zh": "二面角"},
    "timeline": {"en": "Timeline", "zh": "时间轴"},
    "add_marker": {"en": "Add marker", "zh": "添加标记"},
    "add_marker_tooltip": {
        "en": "Mark the current frame (Ctrl+M); double-click the timeline to mark",
        "zh": "标记当前帧 (Ctrl+M)；双击时间轴也可添加",
    },
    "remove_marker": {"en": "Remove", "zh": "删除"},
    "playback_range": {"en": "Playback range", "zh": "播放区间"},
    "range_start": {"en": "Start", "zh": "起点"},
    "range_end": {"en": "End", "zh": "终点"},
    "analysis": {"en": "Analysis", "zh": "分析"},
    "view_tab": {"en": "View", "zh": "查看"},
    "inspect_tab": {"en": "Inspect", "zh": "检查"},
    "analysis_scope": {"en": "Scope", "zh": "范围"},
    "scope_all": {"en": "Entire system", "zh": "整个体系"},
    "scope_selection": {"en": "Current selection", "zh": "当前选择"},
    "analysis_scope_tooltip": {
        "en": "Analyze the entire system or the current atom selection",
        "zh": "分析整个体系，或仅分析当前选择的原子",
    },
    "density_system_scope": {
        "en": "Density is always calculated for the entire system",
        "zh": "密度始终按整个体系计算",
    },
    "analysis_stride": {"en": "Stride", "zh": "步长"},
    "timestep": {"en": "Timestep", "zh": "帧间隔"},
    "timestep_unknown": {"en": "Unknown", "zh": "未知"},
    "analysis_pbc": {"en": "PBC / no-jump", "zh": "周期边界 / 连续坐标"},
    "analysis_pbc_make_whole": {"en": "PBC / make whole", "zh": "周期边界 / 保持整体"},
    "analysis_fit": {"en": "Align / fit", "zh": "对齐拟合"},
    "analysis_mass": {"en": "Mass weighted", "zh": "质量加权"},
    "analysis_mass_density": {"en": "Mass density", "zh": "质量密度"},
    "analysis_dimensions": {"en": "Dimensions", "zh": "方向"},
    "analysis_max_lag": {"en": "Max lag", "zh": "最大延迟"},
    "analysis_all_lags": {"en": "All", "zh": "全部"},
    "analysis_remove_drift": {"en": "Remove selection COM drift", "zh": "去除所选原子的质心漂移"},
    "analysis_axis": {"en": "Profile axis", "zh": "剖面方向"},
    "analysis_bins": {"en": "Bins", "zh": "分箱数"},
    "reference_frame": {"en": "Reference", "zh": "参考帧"},
    "run_analysis": {"en": "Run", "zh": "运行"},
    "cancel_analysis": {"en": "Cancel", "zh": "取消"},
    "export_csv": {"en": "Export CSV", "zh": "导出 CSV"},
    "export_plot": {"en": "Save plot", "zh": "保存图表"},
    "export_plot_tooltip": {"en": "Save the analysis plot as PNG", "zh": "将分析图表保存为 PNG"},
    "reset_plot": {"en": "Reset plot zoom", "zh": "重置图表缩放"},
    "log_x": {"en": "Log X", "zh": "X 对数"},
    "log_y": {"en": "Log Y", "zh": "Y 对数"},
    "export_frame": {"en": "Export frame", "zh": "导出当前帧"},
    "export_frame_tooltip": {"en": "Export current coordinates (Ctrl+E)", "zh": "导出当前坐标 (Ctrl+E)"},
    "export_screenshot": {"en": "Save screenshot", "zh": "保存截图"},
    "export_screenshot_tooltip": {"en": "Save the current viewport as PNG", "zh": "将当前视口保存为 PNG"},
    "export_vector": {"en": "Export vector SVG", "zh": "导出矢量 SVG"},
    "export_vector_tooltip": {
        "en": "Export only the current molecular viewport as editable vector SVG",
        "zh": "仅将当前分子视口导出为可编辑的矢量 SVG",
    },
    "close_analysis": {"en": "Close analysis", "zh": "关闭分析"},
    "analysis_idle": {"en": "Choose an analysis", "zh": "选择一项分析"},
    "analysis_no_result": {"en": "No analysis result", "zh": "暂无分析结果"},
    "density_name": {"en": "Density over time", "zh": "密度随时间"},
    "density_profile_name": {"en": "Density profile over time", "zh": "密度剖面随时间"},
    "msd_name": {"en": "MSD from origin", "zh": "相对起点 MSD"},
    "msd_windowed_name": {"en": "Time-averaged MSD", "zh": "时间平均 MSD"},
    "rmsd_name": {"en": "RMSD", "zh": "均方根偏差 RMSD"},
    "rmsf_name": {"en": "RMSF", "zh": "均方根涨落 RMSF"},
    "com_name": {"en": "Center of mass", "zh": "质心 COM"},
    "rg_name": {"en": "Radius of gyration", "zh": "回转半径 Rg"},
    "msd_warning": {
        "en": "Periodic trajectories require continuous no-jump coordinates for a physically meaningful MSD.",
        "zh": "周期轨迹必须使用连续（no-jump）坐标，MSD 才具有正确物理意义。",
    },
    "analysis_old_selection": {"en": "Result based on old selection", "zh": "此结果基于旧选择"},
    "analysis_points": {"en": "{count} points", "zh": "{count} 个数据点"},
    "analysis_running": {"en": "Analysis is running in the background", "zh": "正在后台分析"},
    "analysis_busy": {"en": "Another analysis is already running", "zh": "已有一项分析正在运行"},
    "analysis_no_cell": {"en": "This analysis requires periodic cell information", "zh": "此分析需要周期晶胞信息"},
    "analysis_no_mass": {"en": "One or more selected elements have no reliable atomic mass", "zh": "一个或多个所选元素没有可靠的原子质量"},
    "analysis_temp_storage": {
        "en": "Time-averaged MSD needs more temporary disk space. Reduce the selection, range, or stride.",
        "zh": "时间平均 MSD 需要更多临时磁盘空间，请减小原子范围、帧区间或增大步长。",
    },
    "export_busy": {"en": "Another export is already running", "zh": "已有一项导出正在进行"},
    "exported": {"en": "Exported {path}", "zh": "已导出 {path}"},
    "playback": {"en": "Playback", "zh": "播放"},
    "advanced": {"en": "Advanced", "zh": "高级"},
    "interface": {"en": "Interface", "zh": "界面"},
    "language": {"en": "Language", "zh": "语言"},
    "theme": {"en": "Theme", "zh": "主题"},
    "theme_system": {"en": "Follow system", "zh": "跟随系统"},
    "theme_light": {"en": "Light", "zh": "浅色"},
    "theme_dark": {"en": "Dark", "zh": "深色"},
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
        self._qt_translator = QTranslator(self)
        self._install_translator(self.ui_language)
        stored_theme = str(self._settings.value("ui/theme", "light") or "light")
        self.ui_theme = stored_theme if stored_theme in {"system", "light", "dark"} else "system"
        self._applying_theme = False
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

        self.recent_button = QToolButton()
        self.recent_button.setObjectName("recentButton")
        self.recent_button.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        self.recent_button.setIcon(
            self.style().standardIcon(QStyle.StandardPixmap.SP_FileDialogListView)
        )
        self.recent_menu = QMenu(self.recent_button)
        self.recent_button.setMenu(self.recent_menu)

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

        self.selection_summary_label = QLabel()
        self.selection_summary_label.setObjectName("selectionSummaryLabel")
        self.selection_summary_label.setWordWrap(True)
        self.clear_selection_button = QPushButton()
        self.clear_selection_button.setEnabled(False)
        self.clear_selection_button.clicked.connect(self.clear_selection)
        self.focus_selection_button = QPushButton()
        self.focus_selection_button.setEnabled(False)
        self.focus_selection_button.clicked.connect(self.focus_selection)

        self.measurement_draft_label = QLabel()
        self.measurement_draft_label.setObjectName("measurementDraftLabel")
        self.measurement_draft_label.setWordWrap(True)
        self.measurement_pbc_check = QCheckBox()
        self.measurement_pbc_check.setChecked(True)
        self.measurement_pbc_check.toggled.connect(self.on_measurement_pbc_toggled)
        self.pin_measurement_button = QPushButton()
        self.pin_measurement_button.setEnabled(False)
        self.pin_measurement_button.clicked.connect(self.create_measurement_from_selection)
        self.measurement_combo = QComboBox()
        self.measurement_combo.setEnabled(False)
        self.measurement_combo.hide()
        self.measurement_combo.currentIndexChanged.connect(self.on_measurement_selected)
        self.remove_measurement_button = QPushButton()
        self.remove_measurement_button.setEnabled(False)
        self.remove_measurement_button.hide()
        self.remove_measurement_button.clicked.connect(self.remove_selected_measurement)
        self.analyze_measurement_button = QPushButton()
        self.analyze_measurement_button.setEnabled(False)
        self.analyze_measurement_button.clicked.connect(self.analyze_selected_measurement)

        self.add_marker_button = QPushButton()
        self.add_marker_button.setEnabled(False)
        self.add_marker_button.clicked.connect(self.add_timeline_marker)
        self.marker_combo = QComboBox()
        self.marker_combo.setEnabled(False)
        self.marker_combo.hide()
        self.marker_combo.currentIndexChanged.connect(self.on_marker_selected)
        self.remove_marker_button = QPushButton()
        self.remove_marker_button.setEnabled(False)
        self.remove_marker_button.hide()
        self.remove_marker_button.clicked.connect(self.remove_selected_marker)
        self.timeline_range_check = QCheckBox()
        self.timeline_range_check.setEnabled(False)
        self.timeline_range_check.toggled.connect(self.on_timeline_range_changed)
        self.range_start_label = self._control_label()
        self.range_end_label = self._control_label()
        self.range_start_spin = QSpinBox()
        self.range_end_spin = QSpinBox()
        for spin in (self.range_start_spin, self.range_end_spin):
            spin.setRange(1, 1)
            spin.setEnabled(False)
            spin.valueChanged.connect(self.on_timeline_range_changed)
            spin.hide()
        self.range_start_label.hide()
        self.range_end_label.hide()

        self.analysis_kind_combo = QComboBox()
        for kind in ("density", "density_profile", "msd", "msd_windowed", "rmsd", "rmsf", "com", "rg"):
            self.analysis_kind_combo.addItem("", kind)
        self.analysis_kind_combo.setEnabled(False)
        self.analysis_kind_combo.currentIndexChanged.connect(self.on_analysis_kind_changed)
        self.analysis_scope_label = self._control_label()
        self.analysis_scope_combo = QComboBox()
        self.analysis_scope_combo.addItem("", "all")
        self.analysis_scope_combo.addItem("", "selection")
        self.analysis_scope_combo.setEnabled(False)
        self.analysis_stride_label = self._control_label()
        self.analysis_stride_spin = QSpinBox()
        self.analysis_stride_spin.setRange(1, 1_000_000)
        self.analysis_stride_spin.setValue(1)
        self.analysis_stride_spin.setEnabled(False)
        self.timestep_label = self._control_label()
        self.timestep_spin = QDoubleSpinBox()
        self.timestep_spin.setRange(0.0, 1.0e12)
        self.timestep_spin.setDecimals(6)
        self.timestep_spin.setValue(0.0)
        self.timestep_spin.setEnabled(False)
        self.time_unit_combo = QComboBox()
        for unit in ("fs", "ps", "ns"):
            self.time_unit_combo.addItem(unit, unit)
        self.time_unit_combo.setCurrentIndex(1)
        self.time_unit_combo.setEnabled(False)
        self.analysis_pbc_check = QCheckBox()
        self.analysis_pbc_check.setChecked(True)
        self.analysis_pbc_check.setEnabled(False)
        self.analysis_fit_check = QCheckBox()
        self.analysis_fit_check.setChecked(True)
        self.analysis_fit_check.setEnabled(False)
        self.analysis_mass_check = QCheckBox()
        self.analysis_mass_check.setChecked(True)
        self.analysis_mass_check.setEnabled(False)
        self.analysis_dimensions_label = self._control_label()
        self.analysis_dimensions_combo = QComboBox()
        for dimensions in ("xyz", "xy", "x", "y", "z"):
            self.analysis_dimensions_combo.addItem(dimensions.upper(), dimensions)
        self.analysis_max_lag_label = self._control_label()
        self.analysis_max_lag_spin = QSpinBox()
        self.analysis_max_lag_spin.setRange(0, 1_000_000_000)
        self.analysis_max_lag_spin.setValue(0)
        self.analysis_remove_drift_check = QCheckBox()
        self.analysis_remove_drift_check.setChecked(False)
        self.analysis_axis_label = self._control_label()
        self.analysis_axis_combo = QComboBox()
        for axis in ("x", "y", "z"):
            self.analysis_axis_combo.addItem(axis.upper(), axis)
        self.analysis_axis_combo.setCurrentIndex(2)
        self.analysis_bins_label = self._control_label()
        self.analysis_bins_spin = QSpinBox()
        self.analysis_bins_spin.setRange(4, 4096)
        self.analysis_bins_spin.setValue(100)
        self.reference_frame_label = self._control_label()
        self.reference_frame_spin = QSpinBox()
        self.reference_frame_spin.setRange(1, 1)
        self.run_analysis_button = QPushButton()
        self.run_analysis_button.setEnabled(False)
        self.run_analysis_button.clicked.connect(self.run_selected_analysis)
        self.cancel_analysis_button = QPushButton()
        self.cancel_analysis_button.setEnabled(False)
        self.cancel_analysis_button.clicked.connect(self.cancel_analysis)
        self.analysis_warning_label = QLabel()
        self.analysis_warning_label.setObjectName("analysisWarningLabel")
        self.analysis_warning_label.setWordWrap(True)
        self.analysis_warning_label.hide()

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

        self.frame_slider = TimelineWidget(self.timeline_model)
        self.frame_slider.setRange(0, 0)
        self.frame_slider.setEnabled(False)
        self.frame_slider.setTracking(False)
        self.frame_slider.sliderPressed.connect(self.on_frame_slider_pressed)
        self.frame_slider.sliderMoved.connect(self.on_frame_slider_moved)
        self.frame_slider.sliderReleased.connect(self.on_frame_slider_released)
        self.frame_slider.valueChanged.connect(self.on_frame_slider_changed)
        self.frame_slider.markerRequested.connect(self.add_timeline_marker)
        self.frame_slider.markerActivated.connect(self.seek_from_timeline)

        self.language_label = self._control_label()
        self.language_combo = QComboBox()
        self.language_combo.addItem("中文", "zh")
        self.language_combo.addItem("English", "en")
        self.language_combo.setCurrentIndex(
            max(0, self.language_combo.findData(self.ui_language))
        )
        self.language_combo.currentIndexChanged.connect(self.on_language_changed)

        self.theme_label = self._control_label()
        self.theme_combo = QComboBox()
        self.theme_combo.addItem("", "system")
        self.theme_combo.addItem("", "light")
        self.theme_combo.addItem("", "dark")
        self.theme_combo.setCurrentIndex(max(0, self.theme_combo.findData(self.ui_theme)))
        self.theme_combo.currentIndexChanged.connect(self.on_theme_changed)

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
        self.export_frame_button = QPushButton()
        self.export_frame_button.setEnabled(False)
        self.export_frame_button.clicked.connect(self.export_current_frame)
        self.export_screenshot_button = QPushButton()
        self.export_screenshot_button.setEnabled(False)
        self.export_screenshot_button.clicked.connect(self.export_viewport_screenshot)
        self.export_vector_button = QPushButton()
        self.export_vector_button.setEnabled(False)
        self.export_vector_button.clicked.connect(self.export_viewport_vector)
        advanced_layout.addWidget(self.export_frame_button)
        advanced_layout.addWidget(self.export_screenshot_button)
        advanced_layout.addWidget(self.export_vector_button)
        self.advanced_content.hide()

        top_bar = self._build_top_bar()
        self.inspector_panel = self._build_inspector()
        self.inspector_scroll = self.inspector_panel
        self.inspector_scroll.setMinimumWidth(280)
        self.inspector_scroll.setMaximumWidth(360)

        self.content_splitter = QSplitter(Qt.Orientation.Horizontal)
        self.content_splitter.setObjectName("contentSplitter")
        self.content_splitter.setChildrenCollapsible(False)
        self.viewport_container = QWidget()
        self.viewport_container.setObjectName("viewportContainer")
        viewport_stack = QStackedLayout(self.viewport_container)
        viewport_stack.setContentsMargins(0, 0, 0, 0)
        viewport_stack.setStackingMode(QStackedLayout.StackingMode.StackAll)
        viewport_stack.addWidget(self.gl_view)
        self.viewport_overlay = ViewportOverlay(self.gl_view, self.viewport_container)
        viewport_stack.addWidget(self.viewport_overlay)
        self.gl_view.viewChanged.connect(self.viewport_overlay.update)
        self.drop_feedback_label = QLabel()
        self.drop_feedback_label.setObjectName("dropFeedbackLabel")
        self.drop_feedback_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.drop_feedback_label.setWordWrap(True)
        self.drop_feedback_label.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.drop_feedback_label.hide()
        viewport_stack.addWidget(self.drop_feedback_label)
        self.content_splitter.addWidget(self.viewport_container)
        self.content_splitter.addWidget(self.inspector_scroll)
        self.content_splitter.setStretchFactor(0, 1)
        self.content_splitter.setStretchFactor(1, 0)
        self.content_splitter.setSizes([1000, 320])

        self.analysis_panel = self._build_analysis_panel()
        self.analysis_panel.hide()

        transport_bar = self._build_transport_bar()
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(top_bar)
        layout.addWidget(self.progress_bar)
        layout.addWidget(self.content_splitter, stretch=1)
        layout.addWidget(self.analysis_panel)
        layout.addWidget(transport_bar)

        central = QWidget()
        central.setObjectName("centralWidget")
        central.setLayout(layout)
        self.setCentralWidget(central)

        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.retranslate_ui()
        self.apply_theme()

    def _build_top_bar(self) -> QFrame:
        top_bar = QFrame()
        top_bar.setObjectName("topBar")
        top_layout = QHBoxLayout(top_bar)
        top_layout.setContentsMargins(12, 8, 10, 8)
        top_layout.setSpacing(12)
        top_layout.addWidget(self.open_button)
        top_layout.addWidget(self.recent_button)
        file_layout = QVBoxLayout()
        file_layout.setContentsMargins(0, 0, 0, 0)
        file_layout.setSpacing(1)
        file_layout.addWidget(self.file_label)
        file_layout.addWidget(self.info_label)
        top_layout.addLayout(file_layout, stretch=1)
        top_layout.addWidget(self.inspector_toggle_button)
        return top_bar

    def _inspector_page(self):
        panel = QFrame()
        panel.setObjectName("inspectorPanel")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(16, 14, 16, 16)
        layout.setSpacing(8)
        layout.setSizeConstraint(QLayout.SizeConstraint.SetMinimumSize)

        scroll = QScrollArea()
        scroll.setObjectName("inspectorScroll")
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setWidget(panel)
        return scroll, layout

    def _build_inspector(self) -> QTabWidget:
        tabs = QTabWidget()
        tabs.setObjectName("inspectorTabs")
        tabs.setDocumentMode(True)
        tabs.tabBar().setExpanding(True)
        tabs.tabBar().setUsesScrollButtons(False)

        view_page, view_layout = self._inspector_page()
        inspect_page, inspect_layout = self._inspector_page()
        analysis_page, analysis_layout = self._inspector_page()

        self.display_section_label = self._section_label()
        view_layout.addWidget(self.display_section_label)
        view_layout.addWidget(self.render_mode_label)
        view_layout.addWidget(self.render_mode_combo)
        view_layout.addLayout(
            self._slider_row(
                self.atom_size_label,
                self.atom_size_slider,
                self.atom_size_value_label,
            )
        )
        view_layout.addLayout(
            self._slider_row(
                self.bond_size_label,
                self.bond_size_slider,
                self.bond_size_value_label,
            )
        )
        view_layout.addWidget(self.box_check)
        view_layout.addSpacing(10)

        self.visibility_section_label = self._section_label()
        view_layout.addWidget(self.visibility_section_label)
        view_layout.addWidget(self.filter_mode_segment)
        filter_value_layout = QHBoxLayout()
        filter_value_layout.setContentsMargins(0, 0, 0, 0)
        filter_value_layout.setSpacing(8)
        filter_value_layout.addWidget(self.filter_value_slider, stretch=1)
        filter_value_layout.addWidget(self.filter_value_label)
        filter_value_layout.addWidget(self.chain_selection_edit, stretch=1)
        view_layout.addLayout(filter_value_layout)
        view_layout.addSpacing(10)

        self.playback_section_label = self._section_label()
        view_layout.addWidget(self.playback_section_label)
        view_layout.addLayout(
            self._slider_row(
                self.playback_speed_label,
                self.playback_speed_slider,
                self.playback_speed_value_label,
            )
        )
        view_layout.addSpacing(8)
        view_layout.addWidget(self.advanced_toggle)
        view_layout.addWidget(self.advanced_content)
        view_layout.addSpacing(10)

        self.interface_section_label = self._section_label()
        view_layout.addWidget(self.interface_section_label)
        language_layout = QHBoxLayout()
        language_layout.setContentsMargins(0, 0, 0, 0)
        language_layout.setSpacing(8)
        language_layout.addWidget(self.language_label)
        language_layout.addWidget(self.language_combo, stretch=1)
        view_layout.addLayout(language_layout)
        theme_layout = QHBoxLayout()
        theme_layout.setContentsMargins(0, 0, 0, 0)
        theme_layout.setSpacing(8)
        theme_layout.addWidget(self.theme_label)
        theme_layout.addWidget(self.theme_combo, stretch=1)
        view_layout.addLayout(theme_layout)
        view_layout.addStretch(1)

        self.selection_section_label = self._section_label()
        inspect_layout.addWidget(self.selection_section_label)
        inspect_layout.addWidget(self.selection_summary_label)
        selection_actions = QHBoxLayout()
        selection_actions.setContentsMargins(0, 0, 0, 0)
        selection_actions.setSpacing(8)
        selection_actions.addWidget(self.clear_selection_button)
        selection_actions.addWidget(self.focus_selection_button)
        inspect_layout.addLayout(selection_actions)
        inspect_layout.addSpacing(10)

        self.measurement_section_label = self._section_label()
        inspect_layout.addWidget(self.measurement_section_label)
        inspect_layout.addWidget(self.measurement_draft_label)
        inspect_layout.addWidget(self.measurement_pbc_check)
        measurement_actions = QHBoxLayout()
        measurement_actions.setContentsMargins(0, 0, 0, 0)
        measurement_actions.setSpacing(8)
        measurement_actions.addWidget(self.pin_measurement_button)
        measurement_actions.addWidget(self.analyze_measurement_button)
        inspect_layout.addLayout(measurement_actions)
        inspect_layout.addWidget(self.measurement_combo)
        inspect_layout.addWidget(self.remove_measurement_button)
        inspect_layout.addSpacing(10)

        self.timeline_section_label = self._section_label()
        inspect_layout.addWidget(self.timeline_section_label)
        marker_actions = QHBoxLayout()
        marker_actions.setContentsMargins(0, 0, 0, 0)
        marker_actions.setSpacing(8)
        marker_actions.addWidget(self.add_marker_button)
        marker_actions.addWidget(self.remove_marker_button)
        inspect_layout.addLayout(marker_actions)
        inspect_layout.addWidget(self.marker_combo)
        inspect_layout.addWidget(self.timeline_range_check)
        range_layout = QHBoxLayout()
        range_layout.setContentsMargins(0, 0, 0, 0)
        range_layout.setSpacing(6)
        range_layout.addWidget(self.range_start_label)
        range_layout.addWidget(self.range_start_spin)
        range_layout.addWidget(self.range_end_label)
        range_layout.addWidget(self.range_end_spin)
        inspect_layout.addLayout(range_layout)
        inspect_layout.addStretch(1)

        self.analysis_section_label = self._section_label()
        self.analysis_section_label.hide()
        analysis_layout.addWidget(self.analysis_kind_combo)
        analysis_scope_layout = QHBoxLayout()
        analysis_scope_layout.setContentsMargins(0, 0, 0, 0)
        analysis_scope_layout.setSpacing(8)
        analysis_scope_layout.addWidget(self.analysis_scope_label)
        analysis_scope_layout.addWidget(self.analysis_scope_combo, stretch=1)
        analysis_layout.addLayout(analysis_scope_layout)
        analysis_stride_layout = QHBoxLayout()
        analysis_stride_layout.setContentsMargins(0, 0, 0, 0)
        analysis_stride_layout.setSpacing(8)
        analysis_stride_layout.addWidget(self.analysis_stride_label)
        analysis_stride_layout.addWidget(self.analysis_stride_spin, stretch=1)
        analysis_layout.addLayout(analysis_stride_layout)
        timestep_layout = QHBoxLayout()
        timestep_layout.setContentsMargins(0, 0, 0, 0)
        timestep_layout.setSpacing(8)
        timestep_layout.addWidget(self.timestep_label)
        timestep_layout.addWidget(self.timestep_spin, stretch=1)
        timestep_layout.addWidget(self.time_unit_combo)
        analysis_layout.addLayout(timestep_layout)
        analysis_layout.addWidget(self.analysis_pbc_check)
        analysis_layout.addWidget(self.analysis_fit_check)
        analysis_layout.addWidget(self.analysis_mass_check)
        direction_layout = QHBoxLayout()
        direction_layout.setContentsMargins(0, 0, 0, 0)
        direction_layout.setSpacing(6)
        direction_layout.addWidget(self.analysis_dimensions_label)
        direction_layout.addWidget(self.analysis_dimensions_combo)
        direction_layout.addWidget(self.analysis_axis_label)
        direction_layout.addWidget(self.analysis_axis_combo)
        analysis_layout.addLayout(direction_layout)
        msd_layout = QHBoxLayout()
        msd_layout.setContentsMargins(0, 0, 0, 0)
        msd_layout.setSpacing(8)
        msd_layout.addWidget(self.analysis_max_lag_label)
        msd_layout.addWidget(self.analysis_max_lag_spin, stretch=1)
        analysis_layout.addLayout(msd_layout)
        analysis_layout.addWidget(self.analysis_remove_drift_check)
        numeric_layout = QHBoxLayout()
        numeric_layout.setContentsMargins(0, 0, 0, 0)
        numeric_layout.setSpacing(6)
        numeric_layout.addWidget(self.analysis_bins_label)
        numeric_layout.addWidget(self.analysis_bins_spin)
        numeric_layout.addWidget(self.reference_frame_label)
        numeric_layout.addWidget(self.reference_frame_spin)
        analysis_layout.addLayout(numeric_layout)
        analysis_layout.addWidget(self.analysis_warning_label)
        analysis_actions = QHBoxLayout()
        analysis_actions.setContentsMargins(0, 0, 0, 0)
        analysis_actions.setSpacing(8)
        analysis_actions.addWidget(self.run_analysis_button)
        analysis_actions.addWidget(self.cancel_analysis_button)
        analysis_layout.addLayout(analysis_actions)
        analysis_layout.addStretch(1)

        self.inspector_view_tab = tabs.addTab(view_page, "")
        self.inspector_inspect_tab = tabs.addTab(inspect_page, "")
        self.inspector_analysis_tab = tabs.addTab(analysis_page, "")
        self.inspector_tabs = tabs
        return tabs

    def _build_analysis_panel(self) -> QFrame:
        panel = QFrame()
        panel.setObjectName("analysisPanel")
        panel.setMinimumHeight(190)
        panel.setMaximumHeight(285)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(12, 7, 12, 8)
        layout.setSpacing(5)
        header = QHBoxLayout()
        header.setContentsMargins(0, 0, 0, 0)
        self.analysis_result_label = QLabel()
        self.analysis_result_label.setObjectName("analysisResultLabel")
        self.analysis_status_label = QLabel()
        self.analysis_status_label.setObjectName("infoLabel")
        self.analysis_progress = QProgressBar()
        self.analysis_progress.setRange(0, 1)
        self.analysis_progress.setValue(0)
        self.analysis_progress.setMaximumWidth(130)
        self.analysis_progress.hide()
        self.export_analysis_button = QPushButton()
        self.export_analysis_button.setEnabled(False)
        self.export_analysis_button.clicked.connect(self.export_analysis_csv)
        self.analysis_log_x_check = QCheckBox()
        self.analysis_log_x_check.toggled.connect(self.on_analysis_log_axes_changed)
        self.analysis_log_y_check = QCheckBox()
        self.analysis_log_y_check.toggled.connect(self.on_analysis_log_axes_changed)
        self.reset_analysis_plot_button = QToolButton()
        self.reset_analysis_plot_button.setIcon(
            self.style().standardIcon(QStyle.StandardPixmap.SP_BrowserReload)
        )
        self.reset_analysis_plot_button.clicked.connect(self.analysis_plot_reset)
        self.export_analysis_plot_button = QToolButton()
        self.export_analysis_plot_button.setIcon(
            self.style().standardIcon(QStyle.StandardPixmap.SP_DialogSaveButton)
        )
        self.export_analysis_plot_button.setEnabled(False)
        self.export_analysis_plot_button.clicked.connect(self.export_analysis_plot_png)
        self.close_analysis_button = QToolButton()
        self.close_analysis_button.setIcon(
            self.style().standardIcon(QStyle.StandardPixmap.SP_TitleBarCloseButton)
        )
        self.close_analysis_button.clicked.connect(panel.hide)
        header.addWidget(self.analysis_result_label)
        header.addWidget(self.analysis_status_label, stretch=1)
        header.addWidget(self.analysis_progress)
        header.addWidget(self.reset_analysis_plot_button)
        header.addWidget(self.export_analysis_plot_button)
        header.addWidget(self.export_analysis_button)
        header.addWidget(self.close_analysis_button)
        layout.addLayout(header)
        plot_options = QHBoxLayout()
        plot_options.setContentsMargins(0, 0, 0, 0)
        plot_options.setSpacing(12)
        plot_options.addWidget(self.analysis_log_x_check)
        plot_options.addWidget(self.analysis_log_y_check)
        plot_options.addStretch(1)
        layout.addLayout(plot_options)
        self.analysis_plot = AnalysisPlotWidget(panel)
        self.analysis_plot.frameRequested.connect(self.seek_from_analysis)
        layout.addWidget(self.analysis_plot, stretch=1)
        return panel

    def on_analysis_log_axes_changed(self) -> None:
        self.analysis_plot.set_log_axes(
            x=self.analysis_log_x_check.isChecked(),
            y=self.analysis_log_y_check.isChecked(),
        )

    def analysis_plot_reset(self) -> None:
        self.analysis_plot.reset_zoom()

    def toggle_analysis_panel(self) -> None:
        self.analysis_panel.setVisible(not self.analysis_panel.isVisible())

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
        self.recent_button.setText(self._t("recent"))
        self.recent_button.setToolTip(self._t("recent_tooltip"))
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
        self.export_frame_button.setText(self._t("export_frame"))
        self.export_frame_button.setToolTip(self._t("export_frame_tooltip"))
        self.export_screenshot_button.setText(self._t("export_screenshot"))
        self.export_screenshot_button.setToolTip(self._t("export_screenshot_tooltip"))
        self.export_vector_button.setText(self._t("export_vector"))
        self.export_vector_button.setToolTip(self._t("export_vector_tooltip"))

        self.display_section_label.setText(self._t("display"))
        self.visibility_section_label.setText(self._t("visibility"))
        self.selection_section_label.setText(self._t("selection"))
        self.measurement_section_label.setText(self._t("measurement"))
        self.timeline_section_label.setText(self._t("timeline"))
        self.analysis_section_label.setText(self._t("analysis"))
        self.playback_section_label.setText(self._t("playback"))
        self.interface_section_label.setText(self._t("interface"))
        self.inspector_tabs.setTabText(self.inspector_view_tab, self._t("view_tab"))
        self.inspector_tabs.setTabText(
            self.inspector_inspect_tab,
            self._t("inspect_tab"),
        )
        self.inspector_tabs.setTabText(
            self.inspector_analysis_tab,
            self._t("analysis"),
        )
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
        self.theme_label.setText(self._t("theme"))
        for theme in ("system", "light", "dark"):
            index = self.theme_combo.findData(theme)
            self.theme_combo.setItemText(index, self._t(f"theme_{theme}"))

        for mode in ("all", "chain", "atom"):
            button = self.filter_mode_buttons[mode]
            button.setText(self._t(mode))
            button.setToolTip(self._t(f"show_{mode}"))
            button.setAccessibleName(self._t(f"show_{mode}"))
        self.chain_selection_edit.setToolTip(self._t("chain_tooltip"))
        self.chain_selection_edit.setAccessibleName(self._t("show_chain"))
        self.clear_selection_button.setText(self._t("clear_selection"))
        self.clear_selection_button.setToolTip(self._t("clear_selection_tooltip"))
        self.focus_selection_button.setText(self._t("focus_selection"))
        self.focus_selection_button.setToolTip(self._t("focus_selection_tooltip"))
        self.measurement_pbc_check.setText(self._t("measurement_pbc"))
        self.pin_measurement_button.setText(self._t("pin_measurement"))
        self.pin_measurement_button.setToolTip(self._t("pin_measurement_tooltip"))
        self.remove_measurement_button.setText(self._t("remove_measurement"))
        self.analyze_measurement_button.setText(self._t("analyze_measurement"))
        self.add_marker_button.setText(self._t("add_marker"))
        self.add_marker_button.setToolTip(self._t("add_marker_tooltip"))
        self.remove_marker_button.setText(self._t("remove_marker"))
        self.timeline_range_check.setText(self._t("playback_range"))
        self.range_start_label.setText(self._t("range_start"))
        self.range_end_label.setText(self._t("range_end"))
        self.analysis_scope_label.setText(self._t("analysis_scope"))
        self.analysis_stride_label.setText(self._t("analysis_stride"))
        self.timestep_label.setText(self._t("timestep"))
        self.timestep_spin.setSpecialValueText(self._t("timestep_unknown"))
        analysis_kind = str(self.analysis_kind_combo.currentData() or "density")
        self.analysis_pbc_check.setText(
            self._t(
                "analysis_pbc"
                if analysis_kind in {"msd", "msd_windowed"}
                else "analysis_pbc_make_whole"
            )
        )
        self.analysis_fit_check.setText(self._t("analysis_fit"))
        self.analysis_mass_check.setText(
            self._t(
                "analysis_mass_density"
                if analysis_kind in {"density", "density_profile"}
                else "analysis_mass"
            )
        )
        self.analysis_dimensions_label.setText(self._t("analysis_dimensions"))
        self.analysis_max_lag_label.setText(self._t("analysis_max_lag"))
        self.analysis_max_lag_spin.setSpecialValueText(self._t("analysis_all_lags"))
        self.analysis_remove_drift_check.setText(self._t("analysis_remove_drift"))
        self.analysis_axis_label.setText(self._t("analysis_axis"))
        self.analysis_bins_label.setText(self._t("analysis_bins"))
        self.reference_frame_label.setText(self._t("reference_frame"))
        for kind in ("density", "density_profile", "msd", "msd_windowed", "rmsd", "rmsf", "com", "rg"):
            index = self.analysis_kind_combo.findData(kind)
            self.analysis_kind_combo.setItemText(index, self._t(f"{kind}_name"))
        self.analysis_scope_combo.setItemText(self.analysis_scope_combo.findData("all"), self._t("scope_all"))
        self.analysis_scope_combo.setItemText(self.analysis_scope_combo.findData("selection"), self._t("scope_selection"))
        self.analysis_scope_combo.setToolTip(
            self._t(
                "density_system_scope"
                if analysis_kind in {"density", "density_profile"}
                else "analysis_scope_tooltip"
            )
        )
        self.run_analysis_button.setText(self._t("run_analysis"))
        self.cancel_analysis_button.setText(self._t("cancel_analysis"))
        self.export_analysis_button.setText(self._t("export_csv"))
        self.analysis_log_x_check.setText(self._t("log_x"))
        self.analysis_log_y_check.setText(self._t("log_y"))
        self.reset_analysis_plot_button.setToolTip(self._t("reset_plot"))
        self.reset_analysis_plot_button.setAccessibleName(self._t("reset_plot"))
        self.export_analysis_plot_button.setToolTip(self._t("export_plot_tooltip"))
        self.export_analysis_plot_button.setAccessibleName(self._t("export_plot"))
        self.close_analysis_button.setToolTip(self._t("close_analysis"))
        self.analysis_plot.set_empty_text(self._t("analysis_no_result"))
        if not self.analysis_result_label.text():
            self.analysis_result_label.setText(self._t("analysis"))
            self.analysis_status_label.setText(self._t("analysis_idle"))
        if hasattr(self, "update_selection_ui"):
            self.update_selection_ui()
        if hasattr(self, "update_measurement_ui"):
            self.update_measurement_ui()

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
        if hasattr(self, "update_recent_menu"):
            self.update_recent_menu()

    def on_language_changed(self) -> None:
        selected = str(self.language_combo.currentData())
        if selected not in {"en", "zh"} or selected == self.ui_language:
            return
        self.ui_language = selected
        self._settings.setValue("ui/language", selected)
        self._install_translator(selected)
        self.retranslate_ui()
        if getattr(self, "store", None) is not None and hasattr(self, "update_trajectory_info"):
            self.update_trajectory_info()

    def on_theme_changed(self) -> None:
        selected = str(self.theme_combo.currentData())
        if selected not in {"system", "light", "dark"}:
            return
        self.ui_theme = selected
        self._settings.setValue("ui/theme", selected)
        self.apply_theme()

    def apply_theme(self) -> None:
        if self._applying_theme:
            return
        self._applying_theme = True
        try:
            app = QApplication.instance()
            palette = resolve_theme(
                self.ui_theme,
                None if app is None else app.palette(),
            )
            self.setStyleSheet(build_stylesheet(palette))
            self.gl_view.set_background_rgb(palette.viewport_bg)
            selection = QColor(palette.selection)
            self.gl_view.set_selection_color(
                (selection.redF(), selection.greenF(), selection.blueF())
            )
            self.viewport_overlay.set_colors(
                line=palette.selection,
                draft=palette.accent,
                text=palette.text,
                background=palette.panel_bg,
            )
            self.analysis_plot.set_colors(
                background=palette.panel_bg,
                text=palette.text,
                grid=palette.plot_grid,
                cursor=palette.selection,
                series=(
                    palette.accent,
                    palette.selection,
                    palette.success,
                    "#b58be0" if palette.theme_id == "dark" else "#7b4ab5",
                    "#d79a45" if palette.theme_id == "dark" else "#c47a15",
                ),
            )
            self.frame_slider.set_colors(
                accent=palette.accent,
                marker=palette.selection,
                playback_range=palette.success,
                cursor=palette.selection,
            )
        finally:
            self._applying_theme = False

    def changeEvent(self, event) -> None:  # type: ignore[override]
        super().changeEvent(event)
        if (
            event.type() == QEvent.Type.PaletteChange
            and getattr(self, "ui_theme", "light") == "system"
            and hasattr(self, "theme_combo")
        ):
            self.apply_theme()

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
        translated = QCoreApplication.translate("TrajPlayer", key)
        if translated != key:
            return translated.format(**values)
        translations = UI_TEXT.get(key)
        if translations is None:
            return key.format(**values)
        template = translations.get(self.ui_language, translations["en"])
        return template.format(**values)

    def _install_translator(self, language: str) -> None:
        app = QApplication.instance()
        if app is None:
            return
        app.removeTranslator(self._qt_translator)
        if self._qt_translator.load(str(translation_file(language))):
            app.installTranslator(self._qt_translator)

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
