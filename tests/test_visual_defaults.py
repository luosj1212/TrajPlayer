import unittest
from pathlib import Path


APP_SOURCE = Path("app.py").read_text(encoding="utf-8")
UI_SOURCE = Path("trajplayer/ui/main_window.py").read_text(encoding="utf-8")
WORKER_SOURCE = Path("trajplayer/workers.py").read_text(encoding="utf-8")
COMMAND_SOURCE = Path("trajplayer/commands.py").read_text(encoding="utf-8")
PRESENT_SCHEDULER_SOURCE = Path("trajplayer/present_scheduler.py").read_text(
    encoding="utf-8"
)
WINDOW_SOURCE = APP_SOURCE + UI_SOURCE


class VisualDefaultTests(unittest.TestCase):
    def test_window_accepts_gromacs_topology_and_trajectory_together(self) -> None:
        source = WINDOW_SOURCE
        self.assertIn("getOpenFileNames", source)
        self.assertIn("*.gro *.xtc *.trr", source)
        self.assertIn("load_trajectory_paths", source)

    def test_macos_finder_open_events_are_coalesced_for_gromacs_pairs(self) -> None:
        source = APP_SOURCE

        self.assertIn("class TrajPlayerApplication(QApplication):", source)
        self.assertIn("QEvent.Type.FileOpen", source)
        self.assertIn("self._pending_file_open_paths.append(path)", source)
        self.assertIn("self.external_open_timer.setInterval(120)", source)
        self.assertIn("def open_queued_external_paths", source)
        self.assertIn("self.load_trajectory_paths(paths)", source)

    def test_window_enables_ball_and_stick_for_real_trajectories_by_default(self) -> None:
        source = APP_SOURCE

        loaded_start = source.index("def on_trajectory_loaded")
        loaded_stop = source.index("def start_benchmark")
        loaded_source = source[loaded_start:loaded_stop]

        self.assertIn("class BondInferenceThread(QThread):", WORKER_SOURCE)
        self.assertIn("self.start_bond_inference(store)", loaded_source)
        self.assertIn("self.gl_view.set_bonds(topology.bonds)", source)
        self.assertIn('store.metadata.get("synthetic")', source)
        self.assertIn("self.gl_view.set_render_mode", loaded_source)

    def test_bond_inference_source_is_visible_and_can_be_disabled(self) -> None:
        source = WINDOW_SOURCE

        self.assertIn("self.infer_bonds_check = QCheckBox()", source)
        self.assertIn('"infer_bonds": {"en": "Infer bonds"', source)
        self.assertIn("layout.addWidget(self.advanced_toggle)", source)
        self.assertIn("layout.addWidget(self.advanced_content)", source)
        self.assertIn("self.infer_bonds_check.toggled.connect", source)
        self.assertIn("def on_infer_bonds_toggled", source)
        self.assertIn("self.bond_topology = topology", source)
        self.assertIn('"bonds_ready",', source)

    def test_open_shortcut_uses_the_platform_standard_binding(self) -> None:
        source = COMMAND_SOURCE

        self.assertIn("QKeySequence.StandardKey.Open", source)
        self.assertNotIn("QShortcut", APP_SOURCE)

    def test_render_timer_sleeps_while_the_viewer_is_idle(self) -> None:
        source = APP_SOURCE

        self.assertIn("self.render_timer.setSingleShot(True)", source)
        self.assertIn("self.present_scheduler.next_timer_delay_ms(", source)
        self.assertIn("self.render_timer.start(delay_ms)", source)
        self.assertNotIn("self.render_timer.start()", source)

    def test_switching_trajectories_does_not_wait_for_stream_io_on_the_ui_thread(self) -> None:
        source = APP_SOURCE

        self.assertIn("streamer.stop(timeout_s=0.0)", source)
        self.assertIn("self._retired_streamers[streamer] = store", source)

    def test_window_exposes_gpu_ball_stick_ball_and_bond_size_controls(self) -> None:
        source = WINDOW_SOURCE

        self.assertIn('self.render_mode_combo.addItem("", "ball_stick")', source)
        self.assertIn('self.render_mode_combo.addItem("", "ball")', source)
        self.assertIn('self.render_mode_combo.addItem("", "bond")', source)
        self.assertIn('"ball_stick": {"en": "Ball-stick"', source)
        self.assertIn("self.atom_size_label = self._control_label()", source)
        self.assertIn("self.bond_size_label = self._control_label()", source)
        self.assertIn("self._size_slider(10, 250, 100, 5, 25)", source)
        self.assertIn("self._size_slider(10, 300, 100, 5, 25)", source)
        self.assertIn('self.atom_size_value_label = self._value_label("100%")', source)
        self.assertNotIn('self.atom_size_spin.setPrefix("Atom ")', source)
        self.assertNotIn('self.bond_size_spin.setPrefix("Bond ")', source)
        self.assertIn("self.gl_view.set_atom_size_scale", source)
        self.assertIn("self.gl_view.set_bond_size_scale", source)

    def test_current_molecular_view_can_be_exported_as_true_vector_svg(self) -> None:
        source = WINDOW_SOURCE

        self.assertIn("self.export_vector_button = QPushButton()", source)
        self.assertIn("self.export_viewport_vector", source)
        self.assertIn("self.gl_view.vector_scene_snapshot()", source)
        self.assertIn('self._start_export("molecule_svg"', source)

    def test_window_exposes_box_toggle_when_cells_are_available(self) -> None:
        source = WINDOW_SOURCE

        self.assertIn("self.box_check = QCheckBox()", source)
        self.assertIn('"box": {"en": "Periodic box"', source)
        self.assertIn("self.box_check.toggled.connect", source)
        self.assertIn("self.gl_view.set_box_enabled", source)
        self.assertIn("cell=lease.cell", source)

    def test_playback_waits_for_frame_swap_and_never_uses_synchronous_repaint(self) -> None:
        source = APP_SOURCE
        renderer = Path("trajplayer/gl_view.py").read_text(encoding="utf-8")

        self.assertIn("self.gl_view.frameSwapped.connect(self.on_frame_swapped)", source)
        self.assertIn(
            "self.gl_view.renderTicketPainted.connect(self.on_render_ticket_painted)",
            source,
        )
        self.assertIn("self.present_scheduler.submit(", source)
        self.assertIn("self.present_scheduler.mark_painted(ticket)", source)
        self.assertIn("self.present_scheduler.acknowledge_swap(", source)
        self.assertNotIn("set_immediate_paint", source)
        self.assertNotIn("self.repaint()", renderer)

    def test_random_access_sources_bypass_the_decoded_sidecar(self) -> None:
        source = WORKER_SOURCE
        reader = Path("trajplayer/random_access_cache.py").read_text(encoding="utf-8")

        self.assertIn("open_direct_random_access_store", source)
        self.assertIn('"direct_reader": True', reader)
        self.assertIn('"persistent_decoded_cache": False', reader)
        self.assertNotIn("def _fill_random_access_cache(", source)

    def test_transport_ui_uses_compact_icon_controls_and_60fps_scrubbing(self) -> None:
        source = WINDOW_SOURCE

        self.assertIn("self.prev_button = self._transport_button(", source)
        self.assertIn("self.play_button = self._transport_button(", source)
        self.assertIn("self.next_button = self._transport_button(", source)
        self.assertIn("SP_MediaPlay", source)
        self.assertIn("SP_MediaPause", source)
        self.assertIn("SCRUB_PREVIEW_FPS = 60.0", source)
        self.assertIn("self.schedule_next_render_tick()", source)
        self.assertIn("playback.next_frame_delay_s", PRESENT_SCHEDULER_SOURCE)
        self.assertIn('transport_bar.setObjectName("transportBar")', source)

    def test_playback_speed_is_adjustable_without_frame_skipping(self) -> None:
        app_source = WINDOW_SOURCE
        engine_source = Path("trajplayer/playback.py").read_text(encoding="utf-8")

        self.assertIn("self.playback_speed_slider = self._size_slider(", app_source)
        self.assertIn("int(self.TARGET_FPS),\n            int(self.TARGET_FPS),", app_source)
        self.assertIn("def on_playback_speed_changed", app_source)
        self.assertIn("fps=float(self.playback_speed_slider.value())", app_source)
        self.assertIn("self.streamer.set_playback_fps", app_source)
        self.assertNotIn(
            "playback_fps=60.0",
            Path("trajplayer/streaming.py").read_text(encoding="utf-8"),
        )
        self.assertIn("and self.displayed_frame == self.current_frame", app_source)
        self.assertIn("dropped_frames=0", engine_source)

    def test_window_can_isolate_a_chain_or_atom_without_per_frame_filtering(self) -> None:
        source = WINDOW_SOURCE

        self.assertIn("self.filter_mode_group = QButtonGroup(self)", source)
        self.assertIn('for mode in ("all", "chain", "atom"):', source)
        self.assertIn('button.setText(self._t(mode))', source)
        self.assertIn("self.filter_value_slider = QSlider", source)
        self.assertIn('self.filter_value_label = self._value_label("")', source)
        self.assertIn('"all_atoms": {"en": "All atoms"', source)
        self.assertIn('self.chain_selection_edit = QLineEdit("1")', source)
        self.assertIn('self.chain_selection_edit.setPlaceholderText("1,3-5")', source)
        self.assertIn("parse_chain_selection", source)
        self.assertIn("unwrap_group_ids = self.component_ids", source)
        self.assertIn("unwrap_group_ids=unwrap_group_ids", source)
        self.assertNotIn("self.filter_mode_combo", source)
        self.assertNotIn("self.filter_value_spin", source)
        self.assertNotIn("self.filter_value_spin = QSpinBox", source)
        self.assertIn("def apply_visibility_filter(self) -> None:", source)
        self.assertIn("self.gl_view.set_visible_atoms", source)
        self.assertIn("connected_components", WORKER_SOURCE)

    def test_window_uses_responsive_inspector_and_runtime_i18n(self) -> None:
        source = UI_SOURCE

        self.assertIn("self.content_splitter = QSplitter", source)
        self.assertIn('tabs.setObjectName("inspectorTabs")', source)
        self.assertIn('scroll.setObjectName("inspectorScroll")', source)
        self.assertIn("self.inspector_view_tab = tabs.addTab", source)
        self.assertIn("self.inspector_analysis_tab = tabs.addTab", source)
        self.assertIn("RESPONSIVE_INSPECTOR_WIDTH = 920", source)
        self.assertIn("def resizeEvent", source)
        self.assertIn("self.language_combo.addItem(\"中文\", \"zh\")", source)
        self.assertIn("QSettings(\"TrajPlayer\", \"TrajPlayer\")", source)
        self.assertIn("def retranslate_ui", source)
        self.assertNotIn("setFixedWidth", source)


if __name__ == "__main__":
    unittest.main()
