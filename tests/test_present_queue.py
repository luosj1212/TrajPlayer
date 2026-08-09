import unittest

from trajplayer.present_queue import FramePresentQueue


class FramePresentQueueTests(unittest.TestCase):
    def test_queue_allows_only_one_frame_until_swap_acknowledgement(self) -> None:
        queue = FramePresentQueue()

        self.assertTrue(queue.begin(7))
        self.assertFalse(queue.begin(8))
        self.assertTrue(queue.has_pending_frame)
        self.assertEqual(queue.pending_frame, 7)
        self.assertEqual(queue.acknowledge(), 7)
        self.assertFalse(queue.has_pending_frame)
        self.assertTrue(queue.begin(8))

    def test_clear_discards_pending_submission(self) -> None:
        queue = FramePresentQueue()
        queue.begin(4)

        self.assertEqual(queue.clear(), 4)
        self.assertIsNone(queue.acknowledge())


if __name__ == "__main__":
    unittest.main()
