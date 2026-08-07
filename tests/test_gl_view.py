import unittest

import numpy as np

from trajplayer import gl_view


class GlViewDefaultsTests(unittest.TestCase):
    def test_hidpi_viewport_uses_physical_framebuffer_pixels(self) -> None:
        self.assertEqual(gl_view.framebuffer_pixel_size(982, 551, 1.5), (1473, 827))
        self.assertEqual(gl_view.framebuffer_pixel_size(982, 551, 1.0), (982, 551))
        self.assertIn("_set_physical_viewport", gl_view.MoleculeGLWidget.paintGL.__code__.co_names)

    def test_shader_preserves_physical_projected_atom_radius(self) -> None:
        self.assertIn("gl_VertexID", gl_view.VERTEX_SHADER)
        self.assertIn("layout(location = 1) in float a_atom_index", gl_view.VERTEX_SHADER)
        self.assertIn("uniform samplerBuffer u_positions", gl_view.VERTEX_SHADER)
        self.assertIn("uniform float u_atom_size_scale", gl_view.VERTEX_SHADER)
        self.assertIn("uniform int u_unwrap_anchor_index", gl_view.VERTEX_SHADER)
        self.assertIn("display_position", gl_view.VERTEX_SHADER)
        self.assertIn("minimum_image_delta", gl_view.VERTEX_SHADER)
        self.assertIn("texelFetch(u_positions", gl_view.VERTEX_SHADER)
        self.assertNotIn("layout(location = 0) in vec2 a_corner", gl_view.VERTEX_SHADER)
        self.assertIn("MIN_RADIUS_NDC_X", gl_view.VERTEX_SHADER)
        self.assertIn("u_proj[1][1] /", gl_view.VERTEX_SHADER)
        self.assertIn("gl_Position.xy += ndc_offset * gl_Position.w", gl_view.VERTEX_SHADER)
        self.assertIn("edge_shadow", gl_view.FRAGMENT_SHADER)
        self.assertIn("rim_shadow", gl_view.FRAGMENT_SHADER)
        self.assertIn("atom_outline", gl_view.FRAGMENT_SHADER)
        self.assertIn("gl_FragDepth", gl_view.FRAGMENT_SHADER)
        self.assertIn("surface_clip", gl_view.FRAGMENT_SHADER)
        self.assertIn("float z14 = z8 * z4 * z2", gl_view.FRAGMENT_SHADER)
        self.assertIn("layout(depth_less) out float gl_FragDepth", gl_view.CONSERVATIVE_FRAGMENT_SHADER)
        self.assertLessEqual(gl_view.MIN_ATOM_RADIUS_NDC_X, 1.0e-5)
        self.assertEqual(gl_view.BALL_STICK_ATOM_RADIUS_SCALE, 0.25)
        self.assertEqual(gl_view.BALL_ATOM_RADIUS_SCALE, 1.0)
        self.assertEqual(gl_view.atom_radius_scale_for_mode("ball_stick", 1.0), 0.25)
        self.assertEqual(gl_view.atom_radius_scale_for_mode("ball", 1.0), 1.0)

    def test_visibility_filter_compacts_static_atom_and_bond_instances(self) -> None:
        self.assertIn("_visible_atom_count", gl_view.MoleculeGLWidget.paintGL.__code__.co_names)
        self.assertIn("_rebuild_visible_bonds", gl_view.MoleculeGLWidget.set_visible_atoms.__code__.co_names)
        self.assertNotIn("set_visible_atoms", gl_view.MoleculeGLWidget._upload_frame_buffers.__code__.co_names)

    def test_gl_resources_are_destroyed_with_a_current_context(self) -> None:
        self.assertIn("cleanup", gl_view.MoleculeGLWidget.initializeGL.__code__.co_names)
        self.assertIn("destroy", gl_view.MoleculeGLWidget.cleanup.__code__.co_names)
        self.assertIn("makeCurrent", gl_view.MoleculeGLWidget.cleanup.__code__.co_names)

    def test_default_camera_fit_keeps_molecule_large_enough_to_read(self) -> None:
        self.assertGreaterEqual(gl_view.CAMERA_DISTANCE_SCALE, 2.55)
        self.assertLessEqual(gl_view.CAMERA_DISTANCE_SCALE, 2.85)

    def test_default_background_is_white(self) -> None:
        background = gl_view.DEFAULT_BACKGROUND

        self.assertGreaterEqual(min(background[:3]), 0.98)
        self.assertEqual(background[3], 1.0)

    def test_cell_box_vertices_create_twelve_edges(self) -> None:
        cell = np.diag([2.0, 3.0, 4.0]).astype(np.float32)

        vertices = gl_view.cell_box_vertices(cell)

        self.assertEqual(vertices.shape, (24, 3))
        np.testing.assert_array_equal(vertices[0], np.array([0.0, 0.0, 0.0], dtype=np.float32))
        np.testing.assert_array_equal(vertices[1], np.array([2.0, 0.0, 0.0], dtype=np.float32))
        np.testing.assert_array_equal(vertices[-1], np.array([2.0, 3.0, 4.0], dtype=np.float32))
        self.assertTrue(vertices.flags.c_contiguous)

    def test_minimum_image_displacements_remove_periodic_box_spans(self) -> None:
        cell = np.diag([10.0, 12.0, 14.0]).astype(np.float32)
        displacements = np.array(
            [
                [9.1, 0.0, 0.0],
                [-9.1, 0.0, 0.0],
                [0.0, 11.2, -13.3],
            ],
            dtype=np.float32,
        )

        wrapped = gl_view.minimum_image_displacements(displacements, cell)

        np.testing.assert_allclose(
            wrapped,
            [[-0.9, 0.0, 0.0], [0.9, 0.0, 0.0], [0.0, -0.8, 0.7]],
            atol=1.0e-5,
        )
        self.assertTrue(wrapped.flags.c_contiguous)

    def test_chain_positions_are_unwrapped_around_stable_anchor(self) -> None:
        positions = np.array(
            [
                [0.2, 5.0, 5.0],
                [9.3, 5.0, 5.0],
                [0.8, 5.0, 5.0],
            ],
            dtype=np.float32,
        )
        indices = np.array([0, 1, 2], dtype=np.int32)
        cell = np.diag([10.0, 10.0, 10.0]).astype(np.float32)

        unwrapped = gl_view.unwrap_positions_around_anchor(
            positions,
            indices,
            0,
            np.array([4.0, 4.0, 4.0], dtype=np.float32),
            cell,
        )

        np.testing.assert_allclose(
            unwrapped,
            [[4.0, 4.0, 4.0], [3.1, 4.0, 4.0], [4.6, 4.0, 4.0]],
            atol=1.0e-5,
        )

    def test_periodic_anchor_is_selected_near_the_wrapped_chain_center(self) -> None:
        positions = np.array(
            [
                [9.4, 5.0, 5.0],
                [9.9, 5.0, 5.0],
                [0.2, 5.0, 5.0],
                [0.9, 5.0, 5.0],
            ],
            dtype=np.float32,
        )
        cell = np.diag([10.0, 10.0, 10.0]).astype(np.float32)

        anchor = gl_view.periodic_anchor_index(
            positions,
            np.arange(4, dtype=np.int32),
            cell,
        )

        self.assertEqual(anchor, 2)

    def test_chain_anchor_stays_stable_when_raw_coordinates_wrap(self) -> None:
        positions = np.array(
            [
                [9.8, 5.0, 5.0],
                [8.9, 5.0, 5.0],
                [0.4, 5.0, 5.0],
            ],
            dtype=np.float32,
        )
        cell = np.diag([10.0, 10.0, 10.0]).astype(np.float32)

        unwrapped = gl_view.unwrap_positions_around_anchor(
            positions,
            np.arange(3, dtype=np.int32),
            0,
            np.array([0.2, 5.0, 5.0], dtype=np.float32),
            cell,
        )

        np.testing.assert_allclose(
            unwrapped,
            [[0.2, 5.0, 5.0], [-0.7, 5.0, 5.0], [0.8, 5.0, 5.0]],
            atol=1.0e-5,
        )

    def test_bond_shader_uses_instanced_two_tone_sticks(self) -> None:
        self.assertIn("layout(location = 4) in vec4 a_bond_data", gl_view.BOND_VERTEX_SHADER)
        self.assertIn("layout(location = 5) in vec3 a_bond_color_a", gl_view.BOND_VERTEX_SHADER)
        self.assertIn("layout(location = 6) in vec3 a_bond_color_b", gl_view.BOND_VERTEX_SHADER)
        self.assertIn("uniform samplerBuffer u_positions", gl_view.BOND_VERTEX_SHADER)
        self.assertIn("uniform float u_bond_size_scale", gl_view.BOND_VERTEX_SHADER)
        self.assertIn("uniform float u_endpoint_radius_scale", gl_view.BOND_VERTEX_SHADER)
        self.assertIn("uniform int u_has_periodic_cell", gl_view.BOND_VERTEX_SHADER)
        self.assertIn("uniform int u_unwrap_anchor_index", gl_view.BOND_VERTEX_SHADER)
        self.assertIn("minimum_image_delta", gl_view.BOND_VERTEX_SHADER)
        self.assertIn("display_position", gl_view.BOND_VERTEX_SHADER)
        self.assertIn("fractional -= round(fractional)", gl_view.BOND_VERTEX_SHADER)
        self.assertIn("texelFetch(u_positions", gl_view.BOND_VERTEX_SHADER)
        self.assertIn("MAX_ENDPOINT_LENGTH_FRACTION", gl_view.BOND_VERTEX_SHADER)
        self.assertIn("BOND_RADIUS_SCALE", gl_view.BOND_VERTEX_SHADER)
        self.assertIn("a_bond_color_a", gl_view.BOND_VERTEX_SHADER)
        self.assertIn("a_bond_color_b", gl_view.BOND_VERTEX_SHADER)
        self.assertNotIn("BOND_COLOR", gl_view.BOND_VERTEX_SHADER)
        self.assertNotIn("_update_bond_segment_arrays", gl_view.MoleculeGLWidget._upload_frame_buffers.__code__.co_names)
        self.assertIn("glDrawArraysInstanced", gl_view.MoleculeGLWidget.paintGL.__code__.co_names)
        self.assertIn("_set_periodic_uniforms", gl_view.MoleculeGLWidget.paintGL.__code__.co_names)
        self.assertEqual(gl_view.BOND_RADIUS_SCALE, 0.20)
        self.assertLessEqual(gl_view.MIN_BOND_RADIUS_NDC_X, 1.0e-5)
        self.assertIn("vec3(0.11)", gl_view.BOND_FRAGMENT_SHADER)
        self.assertIn("v_along < 0.5", gl_view.BOND_FRAGMENT_SHADER)

    def test_frame_upload_uses_three_stream_buffers_inside_paint(self) -> None:
        self.assertEqual(gl_view.POSITION_BUFFER_COUNT, 3)
        self.assertIn("_upload_frame_buffers", gl_view.MoleculeGLWidget.paintGL.__code__.co_names)
        self.assertNotIn("makeCurrent", gl_view.MoleculeGLWidget.set_frame.__code__.co_names)
        self.assertNotIn("doneCurrent", gl_view.MoleculeGLWidget.set_frame.__code__.co_names)

    def test_render_modes_switch_draw_paths_without_per_frame_geometry_rebuilds(self) -> None:
        self.assertEqual(
            gl_view.RENDER_MODES,
            frozenset(("ball_stick", "ball", "bond")),
        )
        self.assertIn("_show_atoms", gl_view.MoleculeGLWidget.paintGL.__code__.co_names)
        self.assertIn("_show_bonds", gl_view.MoleculeGLWidget.paintGL.__code__.co_names)
        self.assertIn("glUniform1f", gl_view.MoleculeGLWidget.paintGL.__code__.co_names)
        self.assertNotIn("set_render_mode", gl_view.MoleculeGLWidget._upload_frame_buffers.__code__.co_names)
        self.assertNotIn("set_atom_size_scale", gl_view.MoleculeGLWidget._upload_frame_buffers.__code__.co_names)

    def test_box_shader_uses_gpu_line_vertices(self) -> None:
        self.assertIn("layout(location = 7) in vec3 a_box_position", gl_view.BOX_VERTEX_SHADER)
        self.assertIn("glDrawArrays", gl_view.MoleculeGLWidget.paintGL.__code__.co_names)
        self.assertIn("set_cell", gl_view.MoleculeGLWidget.set_frame.__code__.co_names)

    def test_bond_segment_colors_stay_crisp_element_colors(self) -> None:
        red_and_white = np.array(
            [
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 1.0],
            ],
            dtype=np.float32,
        )
        pairs = np.array([[0, 1]], dtype=np.int32)

        segment_colors = gl_view.bond_segment_colors_for_pairs(red_and_white, pairs)

        self.assertEqual(segment_colors.shape, (2, 3))
        self.assertGreaterEqual(gl_view.BOND_ELEMENT_COLOR_MIX, 0.92)
        self.assertLessEqual(gl_view.BOND_ELEMENT_COLOR_MIX, 0.98)
        self.assertGreater(segment_colors[0, 0], 0.94)
        self.assertLess(segment_colors[0, 1], 0.08)
        self.assertLess(segment_colors[0, 2], 0.08)
        self.assertGreater(segment_colors[1, 0], 0.90)
        self.assertTrue(segment_colors.flags.c_contiguous)

    def test_bond_segments_are_clipped_away_from_atom_centers(self) -> None:
        positions = np.array(
            [
                [0.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        )
        radii = np.array([0.5, 0.5], dtype=np.float32)
        pairs = np.array([[0, 1]], dtype=np.int32)

        starts, ends = gl_view.bond_segment_endpoints_for_frame(positions, pairs, radii)

        self.assertEqual(starts.shape, (2, 3))
        self.assertEqual(ends.shape, (2, 3))
        self.assertAlmostEqual(float(starts[0, 0]), 0.115, places=5)
        self.assertAlmostEqual(float(starts[1, 0]), 1.885, places=5)
        np.testing.assert_allclose(ends[0], [1.0, 0.0, 0.0], rtol=0.0, atol=1.0e-6)
        np.testing.assert_allclose(ends[1], [1.0, 0.0, 0.0], rtol=0.0, atol=1.0e-6)
        self.assertTrue(starts.flags.c_contiguous)
        self.assertTrue(ends.flags.c_contiguous)


if __name__ == "__main__":
    unittest.main()
