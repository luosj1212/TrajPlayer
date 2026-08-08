import unittest

import numpy as np

from trajplayer.topology import BondSource, BondTopology, empty_topology


class BondTopologyTests(unittest.TestCase):
    def test_static_inference_is_explicitly_labeled_with_its_source_frame(self) -> None:
        topology = BondTopology(
            bonds=np.array([[0, 1]], dtype=np.int64),
            component_ids=np.array([0, 0], dtype=np.int64),
            component_sizes=np.array([2], dtype=np.int64),
            source=BondSource.INFERRED_STATIC,
            source_frame=0,
        )

        self.assertEqual(topology.description, "inferred from frame 1")
        self.assertTrue(topology.chain_selection_available)
        self.assertEqual(topology.bonds.dtype, np.int32)
        self.assertTrue(topology.bonds.flags.c_contiguous)

    def test_disabled_topology_has_no_selectable_components(self) -> None:
        topology = empty_topology()

        self.assertEqual(topology.source, BondSource.DISABLED)
        self.assertEqual(topology.description, "disabled")
        self.assertFalse(topology.chain_selection_available)
        self.assertEqual(topology.bonds.shape, (0, 2))


if __name__ == "__main__":
    unittest.main()
