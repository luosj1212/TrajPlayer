import unittest
from pathlib import Path


class VisualDefaultTests(unittest.TestCase):
    def test_window_accepts_gromacs_topology_and_trajectory_together(self) -> None:
        source = Path("app.py").read_text(encoding="utf-8")
        self.assertIn("getOpenFileNames", source)
        self.assertIn("*.gro *.xtc *.trr", source)
        self.assertIn("load_trajectory_paths", source)

    def test_window_enables_ball_and_stick_for_real_trajectories_by_default(self) -> None:
        source = Path("app.py").read_text(encoding="utf-8")

        loaded_start = source.index("def on_trajectory_loaded")
        loaded_stop = source.index("def start_benchmark")
        loaded_source = source[loaded_start:loaded_stop]

        self.assertIn("class BondInferenceThread(QThread):", source)
        self.assertIn("self.start_bond_inference(store)", loaded_source)
        self.assertIn("self.gl_view.set_bonds(bonds)", source)
        self.assertIn('store.metadata.get("synthetic")', source)
        self.assertIn("self.gl_view.set_render_mode", loaded_source)

    def test_window_exposes_gpu_ball_stick_ball_and_bond_size_controls(self) -> None:
        source = Path("app.py").read_text(encoding="utf-8")

        self.assertIn('self.render_mode_combo.addItem("Ball-stick", "ball_stick")', source)
        self.assertIn('self.render_mode_combo.addItem("Ball", "ball")', source)
        self.assertIn('self.render_mode_combo.addItem("Bond", "bond")', source)
        self.assertIn('self.atom_size_label = QLabel("Atom")', source)
        self.assertIn('self.bond_size_label = QLabel("Bond")', source)
        self.assertIn("self.atom_size_slider.setValue(100)", source)
        self.assertIn("self.bond_size_slider.setValue(100)", source)
        self.assertIn('self.atom_size_value_label = QLabel("100%")', source)
        self.assertNotIn('self.atom_size_spin.setPrefix("Atom ")', source)
        self.assertNotIn('self.bond_size_spin.setPrefix("Bond ")', source)
        self.assertIn("self.gl_view.set_atom_size_scale", source)
        self.assertIn("self.gl_view.set_bond_size_scale", source)

    def test_window_exposes_box_toggle_when_cells_are_available(self) -> None:
        source = Path("app.py").read_text(encoding="utf-8")

        self.assertIn('QCheckBox("Box")', source)
        self.assertIn("self.box_check.toggled.connect", source)
        self.assertIn("self.gl_view.set_box_enabled", source)
        self.assertIn("self.streamer.get_cell(self.current_frame)", source)

    def test_transport_ui_uses_compact_icon_controls_and_60fps_scrubbing(self) -> None:
        source = Path("app.py").read_text(encoding="utf-8")

        self.assertIn("self.prev_button = QToolButton()", source)
        self.assertIn("self.play_button = QToolButton()", source)
        self.assertIn("self.next_button = QToolButton()", source)
        self.assertIn("SP_MediaPlay", source)
        self.assertIn("SP_MediaPause", source)
        self.assertIn("SCRUB_PREVIEW_FPS = 60.0", source)
        self.assertIn("self.schedule_next_render_tick()", source)
        self.assertIn("self.playback.next_frame_delay_s", source)
        self.assertIn('transport_bar.setObjectName("transportBar")', source)

    def test_playback_speed_is_adjustable_without_frame_skipping(self) -> None:
        app_source = Path("app.py").read_text(encoding="utf-8")
        engine_source = Path("trajplayer/playback.py").read_text(encoding="utf-8")

        self.assertIn("self.playback_speed_slider.setRange(1, int(self.TARGET_FPS))", app_source)
        self.assertIn("self.playback_speed_slider.setValue(int(self.TARGET_FPS))", app_source)
        self.assertIn("def on_playback_speed_changed", app_source)
        self.assertIn("fps=float(self.playback_speed_slider.value())", app_source)
        self.assertIn("and self.displayed_frame == self.current_frame", app_source)
        self.assertIn("dropped_frames=0", engine_source)

    def test_window_can_isolate_a_chain_or_atom_without_per_frame_filtering(self) -> None:
        source = Path("app.py").read_text(encoding="utf-8")

        self.assertIn("self.filter_mode_group = QButtonGroup(self)", source)
        self.assertIn('(("All", "all", 38), ("Chain", "chain", 52), ("Atom", "atom", 46))', source)
        self.assertIn("self.filter_value_slider = QSlider", source)
        self.assertIn('self.filter_value_label = QLabel("All atoms")', source)
        self.assertIn('horizontalAdvance(f"Chain {largest_index}") + 6', source)
        self.assertNotIn("self.filter_mode_combo", source)
        self.assertNotIn("self.filter_value_spin", source)
        self.assertNotIn("QSpinBox", source)
        self.assertIn("def apply_visibility_filter(self) -> None:", source)
        self.assertIn("self.gl_view.set_visible_atoms", source)
        self.assertIn("connected_components", source)


if __name__ == "__main__":
    unittest.main()
