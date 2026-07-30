import unittest

import numpy as np

from trajplayer.element_style import atom_render_arrays, atom_style_arrays


class ElementStyleTests(unittest.TestCase):
    def test_atom_style_arrays_are_static_contiguous_float32_buffers(self) -> None:
        radii, colors = atom_style_arrays(np.array([1, 6, 8], dtype=np.uint16))

        self.assertEqual(radii.dtype, np.float32)
        self.assertEqual(colors.dtype, np.float32)
        self.assertEqual(radii.shape, (3,))
        self.assertEqual(colors.shape, (3, 3))
        self.assertTrue(radii.flags["C_CONTIGUOUS"])
        self.assertTrue(colors.flags["C_CONTIGUOUS"])
        self.assertGreater(float(radii[1]), float(radii[0]))
        self.assertTrue(np.all((colors >= 0.0) & (colors <= 1.0)))

    def test_render_arrays_keep_physical_covalent_and_vdw_radii_in_angstrom(self) -> None:
        covalent, vdw, colors = atom_render_arrays(np.array([1, 6, 8], dtype=np.uint16))

        self.assertEqual(covalent.dtype, np.float32)
        self.assertEqual(vdw.dtype, np.float32)
        self.assertTrue(covalent.flags["C_CONTIGUOUS"])
        self.assertTrue(vdw.flags["C_CONTIGUOUS"])
        self.assertTrue(colors.flags["C_CONTIGUOUS"])
        np.testing.assert_allclose(covalent, [0.31, 0.76, 0.66], rtol=0.0, atol=0.02)
        np.testing.assert_allclose(vdw, [1.20, 1.70, 1.52], rtol=0.0, atol=0.02)
        self.assertTrue(np.all(vdw > covalent))


if __name__ == "__main__":
    unittest.main()
