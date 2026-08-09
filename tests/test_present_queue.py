import unittest

from trajplayer.playback import PlaybackEngine
from trajplayer.present_scheduler import PresentScheduler


class PresentSchedulerTests(unittest.TestCase):
    def test_scheduler_allows_only_one_frame_until_swap_acknowledgement(self) -> None:
        scheduler = PresentScheduler()

        token = scheduler.submit(7, now_s=10.0)
        self.assertIsNotNone(token)
        self.assertIsNone(scheduler.submit(8, now_s=10.1))
        self.assertTrue(scheduler.has_pending_frame)
        self.assertEqual(scheduler.pending_frame, 7)

        acknowledgement = scheduler.acknowledge_swap(now_s=10.012)
        self.assertIsNotNone(acknowledgement)
        assert acknowledgement is not None
        self.assertTrue(acknowledgement.accepted)
        self.assertEqual(acknowledgement.token, token)
        self.assertAlmostEqual(acknowledgement.latency_ms, 12.0)
        self.assertEqual(scheduler.displayed_frame, 7)
        self.assertFalse(scheduler.has_pending_frame)
        self.assertIsNotNone(scheduler.submit(8, now_s=10.2))

    def test_generation_change_discards_pending_submission(self) -> None:
        scheduler = PresentScheduler()
        token = scheduler.submit(4, now_s=1.0)

        generation = scheduler.begin_generation(target_frame=0)

        self.assertIsNotNone(token)
        self.assertEqual(generation, 1)
        self.assertIsNone(scheduler.acknowledge_swap(now_s=2.0))
        self.assertEqual(scheduler.displayed_frame, -1)

    def test_timer_policy_preserves_no_skip_playback_deadline(self) -> None:
        scheduler = PresentScheduler()
        scheduler.set_target_frame(0)
        scheduler.set_displayed_frame(0)
        playback = PlaybackEngine(total_frames=10, fps=60.0, loop=True)
        playback.start(frame_index=0, now_s=20.0)

        self.assertEqual(
            scheduler.next_timer_delay_ms(
                playback=playback,
                frame_available=True,
                now_s=20.0,
            ),
            17,
        )
        scheduler.set_target_frame(1)
        self.assertEqual(
            scheduler.next_timer_delay_ms(
                playback=playback,
                frame_available=True,
                now_s=20.0,
            ),
            0,
        )
        self.assertIsNone(
            scheduler.next_timer_delay_ms(
                playback=playback,
                frame_available=False,
                now_s=20.0,
            )
        )


if __name__ == "__main__":
    unittest.main()
