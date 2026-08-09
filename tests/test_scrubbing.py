import unittest
from pathlib import Path

from trajplayer.scrubbing import SliderScrubState


class SliderScrubStateTests(unittest.TestCase):
    def test_drag_moves_are_coalesced_for_realtime_preview(self) -> None:
        scrub = SliderScrubState(preview_interval_s=1 / 30)

        self.assertTrue(scrub.should_commit_value_change())
        scrub.begin(10)
        scrub.move(20)
        scrub.move(30)

        self.assertFalse(scrub.should_commit_value_change())
        self.assertEqual(scrub.pending_frame, 30)
        self.assertTrue(scrub.preview_due(0.0))
        self.assertEqual(scrub.mark_preview(0.0), 30)

        scrub.move(40)
        self.assertFalse(scrub.preview_due(0.01))
        self.assertTrue(scrub.preview_due(0.04))
        self.assertEqual(scrub.mark_preview(0.04), 40)

        self.assertEqual(scrub.release(42), 42)
        self.assertTrue(scrub.should_commit_value_change())
        self.assertIsNone(scrub.pending_frame)

    def test_window_wires_slider_drag_to_realtime_preview(self) -> None:
        source = Path("app.py").read_text(encoding="utf-8") + Path(
            "trajplayer/ui/main_window.py"
        ).read_text(encoding="utf-8")

        self.assertIn("self.frame_slider.setTracking(False)", source)
        self.assertIn("self.frame_slider.sliderMoved.connect(self.on_frame_slider_moved)", source)
        self.assertIn("self.frame_slider.sliderReleased.connect(self.on_frame_slider_released)", source)
        self.assertIn("self.scrub_preview_timer.timeout.connect(self.on_scrub_preview_tick)", source)
        self.assertIn("def on_scrub_preview_tick(self) -> None:", source)
        self.assertNotIn("if self.slider_scrub.active:\n            return", source)
        self.assertIn("self.commit_frame_seek(frame_index)", source)


if __name__ == "__main__":
    unittest.main()
