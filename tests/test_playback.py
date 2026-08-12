import unittest

from trajplayer.playback import PlaybackEngine


class PlaybackEngineTests(unittest.TestCase):
    def test_fixed_60fps_schedule_never_skips_late_frames(self) -> None:
        engine = PlaybackEngine(total_frames=100, fps=60.0, loop=True)
        engine.start(frame_index=0, now_s=10.0)

        first = engine.schedule(10.0168)
        self.assertIsNotNone(first)
        self.assertEqual(first.frame_index, 1)
        self.assertEqual(first.dropped_frames, 0)

        late = engine.schedule(10.0501)
        self.assertIsNotNone(late)
        self.assertEqual(late.frame_index, 2)
        self.assertEqual(late.dropped_frames, 0)

        same_tick = engine.schedule(10.051)
        self.assertIsNone(same_tick)

    def test_looping_playback_wraps_one_frame_at_a_time(self) -> None:
        engine = PlaybackEngine(total_frames=3, fps=30.0, loop=True)
        engine.start(frame_index=2, now_s=1.0)

        decision = engine.schedule(1.034)

        self.assertIsNotNone(decision)
        self.assertEqual(decision.frame_index, 0)
        self.assertEqual(decision.dropped_frames, 0)

    def test_non_looping_playback_stops_on_last_frame(self) -> None:
        engine = PlaybackEngine(total_frames=5, fps=60.0, loop=False)
        engine.start(frame_index=3, now_s=1.0)

        decision = engine.schedule(1.100)

        self.assertIsNotNone(decision)
        self.assertEqual(decision.frame_index, 4)
        self.assertTrue(decision.stop_playback)

    def test_next_frame_delay_tracks_the_wall_clock_deadline(self) -> None:
        engine = PlaybackEngine(total_frames=100, fps=60.0, loop=True)
        engine.start(frame_index=0, now_s=10.0)

        self.assertAlmostEqual(engine.next_frame_delay_s(10.0), 1.0 / 60.0, places=6)
        self.assertAlmostEqual(engine.next_frame_delay_s(10.016), (1.0 / 60.0) - 0.016, places=6)

        engine.schedule(10.017)

        self.assertAlmostEqual(engine.next_frame_delay_s(10.017), (2.0 / 60.0) - 0.017, places=6)

    def test_playback_range_loops_without_skipping(self) -> None:
        engine = PlaybackEngine(
            total_frames=20,
            fps=60.0,
            loop=True,
            range_start=4,
            range_end=7,
        )
        engine.start(frame_index=7, now_s=1.0)
        decision = engine.schedule(1.1)
        self.assertEqual(decision.frame_index, 4)
        self.assertEqual(decision.dropped_frames, 0)


if __name__ == "__main__":
    unittest.main()
